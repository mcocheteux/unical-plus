"""Paired evaluation with explicit protocols, geodesic angles and per-sample errors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from scipy.spatial.transform import Rotation

import unical
from unical.utils.transform import rotation_6d_to_matrix


def main() -> None:
    """Evaluate a saved main/branch model against a named common test protocol."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--raw-root", default="/home/mathieu/datasets/kitti_raw")
    parser.add_argument("--protocol", choices=["frames", "relative", "windows"], required=True)
    parser.add_argument("--variant", choices=["main", "branch"], required=True)
    parser.add_argument("--fusion", default="none")
    parser.add_argument("--sequence-length", type=int, default=1)
    parser.add_argument("--minimum-context-length", type=int, default=3)
    parser.add_argument(
        "--image-normalization", choices=["imagenet_rgb", "mobilevit_bgr"], default="imagenet_rgb"
    )
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--stage", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    training_seed = None
    run_file = args.checkpoint.parent.parent / "run.json"
    if run_file.exists():
        recorded = json.loads(run_file.read_text())["arguments"]
        training_seed = recorded["seed"]
        expected = recorded.get("image_normalization", "imagenet_rgb")
        if expected != args.image_normalization:
            raise ValueError(f"Checkpoint was trained with image normalization {expected}")
    root = Path(unical.__file__).resolve().parents[2]
    assert torch.cuda.is_available(), "Only local CUDA evaluation is supported"
    torch.set_float32_matmul_precision("high")
    overrides = [
        "data=kitti_c2l",
        f"data.data_dir={args.data_dir}",
        f"data.kitti_raw_root={args.raw_root}",
        "data.batch_size=2",
        "data.num_workers=0",
        f"data.stages=[{args.stage}]",
    ]
    if args.variant == "branch":
        overrides += [f"model.temporal.fusion_type={args.fusion}"]
    if args.image_normalization != "imagenet_rgb":
        assert args.variant == "branch", "Preserve exact main preprocessing"
        overrides += [f"+data.preprocessor.cfg.image_normalization={args.image_normalization}"]
    if args.protocol == "windows":
        assert args.variant == "branch", "Use branch T=1 as the temporal implementation control"
        overrides += [
            "data.dataset_format=windows",
            f"data.sequence_length={args.sequence_length}",
            f"data.minimum_context_length={args.minimum_context_length}",
        ]
    elif args.protocol == "relative":
        overrides += [
            "data._target_=relative_c2l.RelativeC2LDataModule",
            f"+data.temporal_batch={str(args.variant == 'branch').lower()}",
        ]
    with initialize_config_dir(config_dir=str(root / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=overrides)
    dm = instantiate(cfg.data)
    dm.setup("test")
    model = instantiate(cfg.model).cuda().eval()
    assert hasattr(model, "temporal") == (args.variant == "branch"), "Wrong model checkout imported"
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"], strict=True)
    records = []
    loader = dm.val_dataloader() if args.split == "val" else dm.test_dataloader()
    with torch.no_grad():
        for batch in loader:
            t, r6 = model(batch._replace(img=batch.img.cuda(), lidar_map=batch.lidar_map.cuda()))
            t = t.float().cpu().numpy()
            R = rotation_6d_to_matrix(r6.float()).cpu().numpy()
            target_t, target_R = [x.numpy() for x in batch.target_reg]
            rel = Rotation.from_matrix(R).inv() * Rotation.from_matrix(target_R)
            angles = np.degrees(rel.magnitude())
            initial_angles = np.degrees(Rotation.from_matrix(target_R).magnitude())
            metadata = batch.metadata[-1] if args.variant == "branch" else batch.metadata
            for i, m in enumerate(metadata):
                group = m.get("window_id") or f"{m['sequence']}_block{int(m['frame_index']) // 30}"
                records.append(
                    {
                        "id": f"{m['img_name']}_anchor{m['frame_index']}",
                        "cluster": group,
                        "target_translation_m": target_t[i].tolist(),
                        "target_rotation_matrix": target_R[i].tolist(),
                        "sequence": m["sequence"],
                        "translation_mae_cm": float(np.mean(np.abs(t[i] - target_t[i]) * 100)),
                        "rotation_degrees": float(angles[i]),
                        "initial_translation_mae_cm": float(np.mean(np.abs(target_t[i]) * 100)),
                        "initial_rotation_degrees": float(initial_angles[i]),
                    }
                )
    keys = [
        "translation_mae_cm",
        "rotation_degrees",
        "initial_translation_mae_cm",
        "initial_rotation_degrees",
    ]
    values = np.array([[r[k] for k in keys] for r in records])
    assert np.isfinite(values).all() and len(records) > 0
    summary = {
        "checkpoint": str(args.checkpoint),
        "code_root": str(root),
        "checkpoint_epoch": state["epoch"],
        "checkpoint_steps": state["global_step"],
        "training_seed": training_seed,
        "protocol": args.protocol,
        "variant": args.variant,
        "image_normalization": args.image_normalization,
        "stage": args.stage,
        "samples": len(records),
        "clusters": len(set(r["cluster"] for r in records)),
        "metrics": dict(zip(keys, values.mean(0).tolist())),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
