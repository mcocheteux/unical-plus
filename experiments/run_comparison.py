"""Resumable local-GPU experiments using the selected checkout's normal model/data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

import pytorch_lightning as L
import torch
from fast_c2l import configure_loading
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

import unical


class ThroughputTimer(L.Callback):
    """Measure steady-state end-to-end batches after loader/CUDA warmup."""

    def __init__(self, warmup=8):
        self.warmup = warmup
        self.batches = 0
        self.start = None
        self.elapsed = None
        self.losses = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.batches += 1
        self.losses.append(outputs["loss"].detach())
        if self.batches == self.warmup:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self.start = time.perf_counter()

    def on_train_end(self, trainer, pl_module):
        torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self.start
        assert torch.isfinite(torch.stack(self.losses)).all(), "Nonfinite benchmark losses"


def comparison_checkpoints(output: Path) -> tuple[ModelCheckpoint, ModelCheckpoint]:
    """Keep validation selection separate from unconditional recovery saves."""
    best = ModelCheckpoint(
        dirpath=str(output / "checkpoints"),
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=False,
        filename="unical-{epoch:03d}",
        auto_insert_metric_name=False,
    )
    recovery = ModelCheckpoint(
        dirpath=str(output / "checkpoints"),
        monitor=None,
        save_top_k=1,
        save_last=True,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        save_on_exception=True,
        filename="resume-{epoch:03d}",
        auto_insert_metric_name=False,
        enable_version_counter=False,
    )
    return best, recovery


def main() -> None:
    """Record configuration/code, train on local CUDA, and select on validation only."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["main", "branch"], required=True)
    parser.add_argument(
        "--protocol", choices=["usual", "relative", "frames", "windows"], required=True
    )
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--raw-root", default="/home/mathieu/datasets/kitti_raw")
    parser.add_argument("--stage", type=int, default=5)
    parser.add_argument("--fusion", default="none")
    parser.add_argument("--sequence-length", type=int, default=1)
    parser.add_argument("--minimum-context-length", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--warmup-epochs", type=int)
    parser.add_argument("--target-training-frames", type=int)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--precision", choices=["32-true", "bf16-mixed", "16-mixed"], default="32-true"
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--max-loss-points", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--benchmark-steps", type=int, default=0)
    parser.add_argument(
        "--image-normalization", choices=["imagenet_rgb", "mobilevit_bgr"], default="imagenet_rgb"
    )
    parser.add_argument("--validation-period", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.image_size < 32 or args.image_size % 32:
        parser.error("Image size must be a positive multiple of 32")
    if args.max_loss_points < 0 or args.num_workers < 0:
        parser.error("Point cap and worker count must be nonnegative")
    if args.benchmark_steps and args.benchmark_steps <= 8:
        parser.error("Benchmark needs more than eight warmup steps")
    if args.protocol != "windows" and args.sequence_length != 1:
        parser.error("Sequence lengths greater than one require the windows protocol")
    if args.variant == "main" and args.fusion != "none":
        parser.error("Exact main has no temporal fusion")
    if args.warmup_epochs is not None and args.warmup_epochs < 0:
        parser.error("Warmup epochs must be nonnegative")
    if args.target_training_frames is not None and args.target_training_frames < 1:
        parser.error("Target training frames must be positive")
    assert torch.cuda.is_available(), "This runner requires the local CUDA GPU"
    root = Path(unical.__file__).resolve().parents[2]
    overrides = [
        f"data_dir={args.raw_root}",
        f"seed={args.seed}",
        f"log_dir={args.output}",
        f"data.batch_size={args.batch_size}",
        f"data.num_workers={args.num_workers}",
        f"data.pin_memory={str(args.pin_memory).lower()}",
        f"data.preprocessor.cfg.width={args.image_size}",
        f"data.preprocessor.cfg.height={args.image_size}",
        f"model.backbone.image_size={args.image_size}",
        "trainer.accelerator=gpu",
        f"trainer.precision={args.precision}",
        f"trainer.max_epochs={args.epochs}",
    ]
    if args.warmup_epochs is not None:
        overrides += [f"model.warmup_epochs={args.warmup_epochs}"]
    if args.variant == "branch":
        overrides += [f"model.temporal.fusion_type={args.fusion}"]
    if args.image_normalization != "imagenet_rgb":
        assert args.variant == "branch", "Preserve exact main preprocessing"
        overrides += [f"+data.preprocessor.cfg.image_normalization={args.image_normalization}"]
    if args.protocol != "usual":
        assert args.data_dir is not None
        overrides += [
            "data=kitti_c2l",
            f"data.data_dir={args.data_dir}",
            f"data.kitti_raw_root={args.raw_root}",
            f"data.stages=[{args.stage}]",
        ]
        if args.protocol == "relative":
            overrides += [
                "data._target_=relative_c2l.RelativeC2LDataModule",
                f"+data.temporal_batch={str(args.variant == 'branch').lower()}",
            ]
        elif args.protocol == "windows":
            if args.variant == "branch":
                overrides += [
                    "data.dataset_format=windows",
                    f"data.sequence_length={args.sequence_length}",
                    f"data.minimum_context_length={args.minimum_context_length}",
                ]
            else:
                assert args.sequence_length == 1, "Exact main consumes one anchor frame"
                overrides += [
                    "data._target_=window_c2l.MainWindowDataModule",
                    f"+data.minimum_context_length={args.minimum_context_length}",
                ]
    with initialize_config_dir(config_dir=str(root / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=overrides)
    if args.protocol == "usual":
        # The raw loader silently accepts absent drives; disallow a partial standard split.
        for split in cfg.experiment.splits.values():
            for date, drives in split:
                for drive in drives:
                    directory = Path(args.raw_root) / date / f"{date}_drive_{drive:04d}_sync"
                    if not (directory / ".unical_complete.json").exists():
                        raise FileNotFoundError(f"Incomplete standard KITTI drive: {directory}")
    L.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    data = instantiate(cfg.data)
    if args.max_loss_points or args.num_workers or args.pin_memory or args.benchmark_steps:
        configure_loading(data, args.max_loss_points, args.benchmark_steps * args.batch_size)
    data.setup()
    model = instantiate(cfg.model)
    assert hasattr(model, "temporal") == (args.variant == "branch"), "Wrong code checkout imported"
    counts = {s: len(getattr(data, f"{s}_ds")) for s in ["train", "val", "test"]}
    assert min(counts.values()) > 0
    if args.target_training_frames is not None:
        args.epochs = math.ceil(
            args.target_training_frames / (counts["train"] * args.sequence_length)
        )
        cfg.trainer.max_epochs = args.epochs
    print("Experiment samples:", counts, flush=True)
    print(
        "Training schedule:",
        {"epochs": args.epochs, "warmup_epochs": model.hparams.warmup_epochs},
        flush=True,
    )
    if args.dry_run:
        print("Configuration validated; no training launched.", flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    run_file = args.output / "run.json"
    if run_file.exists():
        if not args.resume:
            raise FileExistsError(f"Experiment already exists; use --resume: {run_file}")
        previous_manifest = json.loads(run_file.read_text())
        previous = previous_manifest["arguments"]
        for key in [
            "variant",
            "protocol",
            "stage",
            "fusion",
            "sequence_length",
            "minimum_context_length",
            "seed",
            "batch_size",
            "precision",
            "image_normalization",
            "epochs",
            "warmup_epochs",
            "target_training_frames",
            "data_dir",
            "raw_root",
            "image_size",
            "max_loss_points",
            "num_workers",
            "pin_memory",
        ]:
            default = {
                "image_normalization": "imagenet_rgb",
                "image_size": 512,
                "max_loss_points": 0,
                "num_workers": 0,
                "pin_memory": False,
            }.get(key)
            value = getattr(args, key)
            if isinstance(value, Path):
                value = str(value)
            if previous.get(key, default) != value:
                raise ValueError(f"Resume configuration changed: {key}")
    manifest = {
        "experiment_code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, text=True
        ).strip(),
        "experiment_source_hashes": {
            name: hashlib.sha256((Path(__file__).resolve().parent / name).read_bytes()).hexdigest()
            for name in ["run_comparison.py", "relative_c2l.py", "window_c2l.py", "fast_c2l.py"]
        },
        "code_root": str(root),
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "code_changes": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True
        ),
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "counts": counts,
        "configuration": OmegaConf.to_container(cfg, resolve=True),
        "pretrained_revision": getattr(model.backbone.model.config, "_commit_hash", None),
        "dataset_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in args.data_dir.glob("*.parquet")
        }
        if args.data_dir
        else {},
    }
    if args.resume:
        manifest["resume_history"] = previous_manifest.get("resume_history", []) + [
            {
                k: previous_manifest[k]
                for k in ["experiment_code_commit", "experiment_source_hashes", "arguments"]
            }
        ]
    run_file.write_text(json.dumps(manifest, indent=2) + "\n")
    if not args.resume:
        torch.save(model.state_dict(), args.output / "initial_state.pt")
    if args.benchmark_steps:
        timer = ThroughputTimer()
        trainer = L.Trainer(
            **OmegaConf.to_container(cfg.trainer, resolve=True),
            max_steps=args.benchmark_steps,
            limit_val_batches=0,
            num_sanity_val_steps=0,
            logger=False,
            callbacks=[timer],
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        trainer.fit(model, datamodule=data)
        measured = timer.batches - timer.warmup
        manifest.update(
            status="benchmark_completed",
            benchmark={
                "measured_batches": measured,
                "seconds": timer.elapsed,
                "seconds_per_batch": timer.elapsed / measured,
                "anchors_per_second": measured * args.batch_size / timer.elapsed,
                "backbone_frames_per_second": measured
                * args.batch_size
                * args.sequence_length
                / timer.elapsed,
                "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
                "finite_losses": True,
            },
        )
        run_file.write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest["benchmark"], indent=2), flush=True)
        return
    checkpoint, recovery = comparison_checkpoints(args.output)
    trainer = L.Trainer(
        **OmegaConf.to_container(cfg.trainer, resolve=True),
        check_val_every_n_epoch=min(args.validation_period, args.epochs),
        logger=CSVLogger(str(args.output), name="metrics"),
        callbacks=[checkpoint, recovery, LearningRateMonitor(logging_interval="epoch")],
        enable_progress_bar=False,
    )
    # These are checkpoints produced locally by this experiment, including resume.
    trainer.fit(model, datamodule=data, ckpt_path=args.resume, weights_only=False)
    optimizer_steps = trainer.global_step
    completed_epochs = trainer.current_epoch
    results = trainer.test(
        model, datamodule=data, ckpt_path=checkpoint.best_model_path, weights_only=False
    )
    manifest.update(
        {
            "status": "trained_and_tested",
            "best_checkpoint": checkpoint.best_model_path,
            "last_checkpoint": recovery.last_model_path,
            "test_metrics": results,
            "optimizer_steps": optimizer_steps,
            "completed_epochs": completed_epochs,
            "backbone_training_frames": counts["train"] * completed_epochs * args.sequence_length,
        }
    )
    run_file.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
