"""Recovery must advance even when validation loss gets worse."""

import importlib.util
import signal
from pathlib import Path

import pytest
import pytorch_lightning as L
import torch
from torch.utils.data import DataLoader, TensorDataset

spec = importlib.util.spec_from_file_location(
    "run_comparison", Path(__file__).resolve().parents[1] / "experiments/run_comparison.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class WorseningModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def training_step(self, batch, batch_idx):
        return self.weight.square()

    def validation_step(self, batch, batch_idx):
        self.log("val/loss", float(self.current_epoch + 1))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def test_recovery_advances_without_validation_improvement(tmp_path):
    best, recovery = module.comparison_checkpoints(tmp_path)
    trainer = L.Trainer(
        accelerator="cpu",
        max_epochs=3,
        callbacks=[best, recovery],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    loader = DataLoader(TensorDataset(torch.ones(2, 1)), batch_size=1)
    trainer.fit(WorseningModel(), loader, loader)
    selected = torch.load(best.best_model_path, weights_only=False, map_location="cpu")
    latest = torch.load(recovery.last_model_path, weights_only=False, map_location="cpu")
    assert selected["epoch"] == 0
    assert latest["epoch"] == 2
    assert latest["global_step"] == 6
    # Restoring the recovery checkpoint must preserve best-model selection state.
    new_best, new_recovery = module.comparison_checkpoints(tmp_path)
    resumed = L.Trainer(
        accelerator="cpu",
        max_epochs=4,
        callbacks=[new_best, new_recovery],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    resumed.fit(
        WorseningModel(), loader, loader, ckpt_path=recovery.last_model_path, weights_only=False
    )
    assert new_best.best_model_path == best.best_model_path
    latest = torch.load(new_recovery.last_model_path, weights_only=False, map_location="cpu")
    assert latest["epoch"] == 3
    assert latest["global_step"] == 8


def test_keyboard_interrupt_preserves_completed_optimizer_updates(tmp_path):
    class InterruptedModel(WorseningModel):
        def training_step(self, batch, batch_idx):
            if batch_idx == 1:
                raise KeyboardInterrupt
            return super().training_step(batch, batch_idx)

    best, recovery = module.comparison_checkpoints(tmp_path)
    trainer = L.Trainer(
        accelerator="cpu",
        max_epochs=2,
        callbacks=[best, recovery],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    loader = DataLoader(TensorDataset(torch.ones(2, 1)), batch_size=1)
    handler = signal.getsignal(signal.SIGINT)
    try:
        with pytest.raises(SystemExit):
            trainer.fit(InterruptedModel(), loader, loader)
    finally:
        # Lightning ignores further SIGINTs during its graceful shutdown.
        signal.signal(signal.SIGINT, handler)
    saved = torch.load(recovery.last_model_path, weights_only=False, map_location="cpu")
    assert saved["global_step"] == 1
    assert saved["state_dict"]["weight"].item() == pytest.approx(0.98)
    assert not best.best_model_path
