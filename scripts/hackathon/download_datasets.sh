#!/usr/bin/env bash
# Download the robomimic v1.5 PH datasets at a pinned revision, verify sha256,
# build the symlink tree expected by the configs and the absolute-action files.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
TASKS=${TASKS:-"can square"}
REV=$(python3 -c "import json;print(json.load(open('configs/hackathon/dataset_manifest.json'))['revision'])")
for t in $TASKS; do
  for f in low_dim_v15.hdf5 demo_v15.hdf5; do
    rel="v1.5/$t/ph/$f"; dst="data/robomimic/hf/$rel"
    mkdir -p "$(dirname "$dst")"
    [[ -f "$dst" ]] || curl -sSL --fail -o "$dst" "https://huggingface.co/datasets/robomimic/robomimic_datasets/resolve/$REV/$rel"
    want=$(python3 -c "import json;print(json.load(open('configs/hackathon/dataset_manifest.json'))['files']['$rel']['sha256'])")
    got=$(sha256sum "$dst" | cut -d' ' -f1)
    [[ "$want" == "$got" ]] || { echo "sha256 mismatch for $dst"; exit 1; }
    echo "ok $rel"
  done
  mkdir -p "data/robomimic/datasets/$t/ph"
  ln -sfn "../../../hf/v1.5/$t/ph/low_dim_v15.hdf5" "data/robomimic/datasets/$t/ph/low_dim.hdf5"
  ln -sfn "../../../hf/v1.5/$t/ph/demo_v15.hdf5" "data/robomimic/datasets/$t/ph/demo.hdf5"
done
source scripts/hackathon/env.sh
for t in $TASKS; do
  out="data/robomimic/datasets/$t/ph/low_dim_abs.hdf5"
  [[ -f "$out" ]] || python diffusion_policy/scripts/robomimic_dataset_conversion.py \
      --input "data/robomimic/datasets/$t/ph/low_dim.hdf5" --output "$out" --num_workers "${NUM_WORKERS:-8}"
  echo "abs $out"
done
