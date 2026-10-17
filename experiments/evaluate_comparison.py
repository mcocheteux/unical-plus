"""Paired evaluation with explicit protocols, geodesic angles and per-sample errors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from fast_c2l import configure_loading
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from scipy.spatial.transform import Rotation

import unical
from unical.utils.transform import rotation_6d_to_matrix


def guard_paper_alpha_training(manifest: dict) -> None:
    """Reject drive-28 train/validation exposure before claiming an unseen alpha test."""
    recorded = manifest.get("arguments", {})
    if recorded.get("protocol") == "usual":
        splits = manifest["configuration"]["experiment"]["splits"]
        for split in ["train", "val"]:
            if any(
                date == "2011_09_30" and 28 in map(int, drives) for date, drives in splits[split]
            ):
                raise ValueError("Paper-alpha test drive 28 appeared in training/validation")
        return
    if recorded.get("protocol") not in ["frames", "relative", "windows"]:
        raise ValueError("Paper-alpha evaluation requires recorded training provenance")
    expected_hash = manifest.get("dataset_hashes", {}).get("train.parquet")
    if not recorded.get("data_dir") or not expected_hash:
        raise ValueError("Paper-alpha evaluation requires recorded training metadata and its hash")
    path = Path(recorded["data_dir"]) / "train.parquet"
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
        raise ValueError("Recorded training metadata changed")
    rows = pq.read_table(path, columns=["raw_date", "raw_drive"]).to_pylist()
    if any(row["raw_date"] == "2011_09_30" and int(row["raw_drive"]) == 28 for row in rows):
        raise ValueError("Paper-alpha drive 28 is C2L sequence 08; exclude it before training")


def main() -> None:
    """Evaluate a saved main/branch model against a named common test protocol."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--raw-root", default="/home/mathieu/datasets/kitti_raw")
    parser.add_argument(
        "--protocol", choices=["frames", "relative", "windows", "paper-alpha"], required=True
    )
    parser.add_argument("--variant", choices=["main", "branch"], required=True)
    parser.add_argument("--fusion", default="none")
    parser.add_argument("--sequence-length", type=int, default=1)
    parser.add_argument("--minimum-context-length", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--image-normalization", choices=["imagenet_rgb", "mobilevit_bgr"], default="imagenet_rgb"
    )
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--stage", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    training_seed = None
    training_manifest = {}
    run_file = args.checkpoint.parent.parent / "run.json"
    if run_file.exists():
        training_manifest = json.loads(run_file.read_text())
        recorded = training_manifest["arguments"]
        training_seed = recorded["seed"]
        expected = recorded.get("image_normalization", "imagenet_rgb")
        if expected != args.image_normalization:
            raise ValueError(f"Checkpoint was trained with image normalization {expected}")
        if recorded.get("image_size", 512) != args.image_size:
            raise ValueError("Evaluation image size must match training")
    if args.protocol == "paper-alpha":
        if args.split != "test" or args.sequence_length != 1:
            parser.error("Paper-alpha currently evaluates single-frame models on its test split")
        if training_manifest.get("arguments", {}).get("sequence_length", 1) != 1:
            parser.error("Use the matched window protocol for models trained with temporal inputs")
        guard_paper_alpha_training(training_manifest)
    root = Path(unical.__file__).resolve().parents[2]
    if not args.dry_run:
        assert torch.cuda.is_available(), "Only local CUDA evaluation is supported"
    torch.set_float32_matmul_precision("high")
    overrides = [
        "data.batch_size=2",
        f"data.num_workers={args.num_workers}",
        f"data.preprocessor.cfg.width={args.image_size}",
        f"data.preprocessor.cfg.height={args.image_size}",
        f"model.backbone.image_size={args.image_size}",
    ]
    if args.protocol == "paper-alpha":
        overrides += ["data=kitti", "experiment=default", f"data_dir={args.raw_root}"]
    else:
        overrides += [
            "data=kitti_c2l",
            f"data.data_dir={args.data_dir}",
            f"data.kitti_raw_root={args.raw_root}",
            f"data.stages=[{args.stage}]",
        ]
    if args.variant == "branch":
        overrides += [f"model.temporal.fusion_type={args.fusion}"]
    if args.image_normalization != "imagenet_rgb":
        assert args.variant == "branch", "Preserve exact main preprocessing"
        overrides += [f"+data.preprocessor.cfg.image_normalization={args.image_normalization}"]
    if args.protocol == "windows":
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
    elif args.protocol == "relative":
        overrides += [
            "data._target_=relative_c2l.RelativeC2LDataModule",
            f"+data.temporal_batch={str(args.variant == 'branch').lower()}",
        ]
    with initialize_config_dir(config_dir=str(root / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=overrides)
    dm = instantiate(cfg.data)
    if args.num_workers:
        configure_loading(dm, recorded.get("max_loss_points", 0) if run_file.exists() else 0)
    dm.setup("test")
    if args.dry_run:
        dataset = dm.val_ds if args.split == "val" else dm.test_ds
        sample = dataset[0]
        print(
            json.dumps(
                {
                    "protocol": args.protocol,
                    "split": args.split,
                    "samples": len(dataset),
                    "image_shape": list(sample["img"].shape),
                },
                indent=2,
            )
        )
        return
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
            # The UniCal paper regresses roll/pitch/yaw and reports rotation MAE.
            # Keep this per-axis metric distinct from geodesic SO(3) error.
            predicted_euler = Rotation.from_matrix(R).as_euler("xyz", degrees=True)
            target_euler = Rotation.from_matrix(target_R).as_euler("xyz", degrees=True)
            euler_error = np.abs((predicted_euler - target_euler + 180) % 360 - 180).mean(1)
            initial_euler_error = np.abs((target_euler + 180) % 360 - 180).mean(1)
            metadata = batch.metadata[-1] if args.variant == "branch" else batch.metadata
            for i, m in enumerate(metadata):
                if args.protocol == "paper-alpha":
                    m = m | {
                        "sequence": "raw_2011_09_30_0028",
                        "frame_index": int(Path(m["img_name"]).stem),
                    }
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
                        "rotation_euler_mae_degrees": float(euler_error[i]),
                        "initial_translation_mae_cm": float(np.mean(np.abs(target_t[i]) * 100)),
                        "initial_rotation_degrees": float(initial_angles[i]),
                        "initial_rotation_euler_mae_degrees": float(initial_euler_error[i]),
                    }
                )
    keys = [
        "translation_mae_cm",
        "rotation_degrees",
        "initial_translation_mae_cm",
        "initial_rotation_degrees",
        "rotation_euler_mae_degrees",
        "initial_rotation_euler_mae_degrees",
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
        "image_size": args.image_size,
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
