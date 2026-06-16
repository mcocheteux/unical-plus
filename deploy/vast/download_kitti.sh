#!/usr/bin/env bash
#
# Download the KITTI raw drives required by UniCal's Hydra experiment configs.
#
#   configs/experiment/default.yaml -> KITTI_MODE=default (full RegNet/LCCNet split)
#   configs/experiment/debug.yaml   -> KITTI_MODE=debug   (tiny: drives 1 and 5)
#
# The KITTI raw "sync" zips already unpack into <date>/<date>_drive_NNNN_sync/...
# and the calib zips into <date>/calib_*.txt, which is exactly the on-disk layout
# unical.data.dataset.KittiDataset expects. So unzip everything into DATA_DIR and
# pass `data_dir=$DATA_DIR` to train.py / evaluate.py.
#
# Idempotent: existing drives / calibration are skipped.
#
# Usage:
#   DATA_DIR=/workspace/kitti_raw KITTI_MODE=default bash download_kitti.sh
#
set -euo pipefail

DATA_DIR="${DATA_DIR:-/workspace/kitti_raw}"
KITTI_MODE="${KITTI_MODE:-default}"
BASE_URL="https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"

DATE_A="2011_09_26"
DATE_B="2011_09_30"

if [ "$KITTI_MODE" = "debug" ]; then
  DRIVES_A=(1 5)
  DRIVES_B=()
  CALIB_DATES=("$DATE_A")
else
  # default.yaml split: 2011_09_26 train + val, 2011_09_30 test
  DRIVES_A=(1 2 9 11 13 14 15 17 18 19 20 22 23 27 28 29 32 35 36 39 46 48 \
            51 52 56 57 59 60 61 64 79 84 86 87 91 93 95 96 101 104 106 113 117 \
            5 70)
  DRIVES_B=(28)
  CALIB_DATES=("$DATE_A" "$DATE_B")
fi

command -v wget >/dev/null 2>&1 || { echo "ERROR: wget is required" >&2; exit 1; }
command -v unzip >/dev/null 2>&1 || { echo "ERROR: unzip is required" >&2; exit 1; }

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

download_calib() {
  local date="$1"
  if [ -f "$date/calib_cam_to_cam.txt" ] && [ -f "$date/calib_velo_to_cam.txt" ]; then
    echo "[kitti] calib $date already present, skipping"
    return
  fi
  echo "[kitti] downloading calib $date"
  wget -q --show-progress -O "${date}_calib.zip" "$BASE_URL/${date}_calib.zip"
  unzip -o -q "${date}_calib.zip"
  rm -f "${date}_calib.zip"
}

download_drive() {
  local date="$1" drive="$2"
  local d4 name
  d4="$(printf '%04d' "$drive")"
  name="${date}_drive_${d4}"
  if [ -d "$date/${name}_sync" ]; then
    echo "[kitti] $name already present, skipping"
    return
  fi
  echo "[kitti] downloading $name"
  wget -q --show-progress -O "${name}_sync.zip" "$BASE_URL/${name}/${name}_sync.zip"
  unzip -o -q "${name}_sync.zip"
  rm -f "${name}_sync.zip"
}

echo "[kitti] target=$DATA_DIR mode=$KITTI_MODE"
for date in "${CALIB_DATES[@]}"; do download_calib "$date"; done
for d in "${DRIVES_A[@]}"; do download_drive "$DATE_A" "$d"; done
for d in "${DRIVES_B[@]:-}"; do
  [ -n "$d" ] && download_drive "$DATE_B" "$d"
done

echo "[kitti] done. Dataset root: $DATA_DIR"
echo "[kitti] sanity:"; ls -1 "$DATA_DIR"
