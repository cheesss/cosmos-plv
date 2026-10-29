# PLV Dataset Download and Preparation Plan

Last updated: 2026-06-23 KST

This document tells a second training agent where to get the robomimic data, which files are needed for the PLV experiments, where to place them, and how to prepare absolute-action low-dimensional datasets.

## Official sources

Use the robomimic v1.5 datasets from the official Hugging Face dataset repository:

```text
https://huggingface.co/datasets/robomimic/robomimic_datasets/tree/main/v1.5
```

The robomimic documentation describes the standard dataset pipeline as:

```text
download dataset -> postprocess dataset -> train agent
```

Relevant docs:

```text
https://robomimic.github.io/docs/datasets/overview.html
https://robomimic.github.io/docs/datasets/robomimic_v0.1.html
https://github.com/ARISE-Initiative/robomimic/releases
```

This repo uses the Hugging Face v1.5 files directly because the training configs expect robosuite v1.5-compatible data filenames such as `low_dim_v15.hdf5`.

## Local data layout expected by this repo

Raw downloaded files should live under:

```text
data/robomimic/hf/v1.5/<task>/<dataset_type>/<file>
```

The training configs read from:

```text
data/robomimic/datasets/<task>/<dataset_type>/low_dim.hdf5
data/robomimic/datasets/<task>/<dataset_type>/low_dim_abs.hdf5
```

The normal `low_dim.hdf5` files are symlinks to the Hugging Face mirror. The `low_dim_abs.hdf5` files are generated locally by `diffusion_policy/scripts/robomimic_dataset_conversion.py`.

Do not commit any `data/` files to GitHub.

## Fast path: copy data from Machine A

If Machine A is reachable, this is preferred because it preserves the exact files, symlinks, generated `low_dim_abs.hdf5` files, and train/valid masks.

Run from Machine A:

```bash
rsync -av --info=progress2 \
  <machine-A path> \
  USER@MACHINE_B:/PATH/TO/plv-robot-imitation-learning/data/
```

After copying, on Machine B:

```bash
cd /PATH/TO/plv-robot-imitation-learning
find data/robomimic/datasets -maxdepth 3 -name 'low_dim*.hdf5' -o -name 'demo.hdf5' | sort
bash diffusion_policy/scripts/prepare_robomimic_extended_lowdim_abs.sh
```

The preparation script is idempotent and should skip files that already exist.

## Direct download path: Hugging Face

Use this path only if copying from Machine A is not possible.

Install the Hugging Face download helper:

```bash
pip install -U huggingface_hub hf_xet
```

Download the full low-dimensional file set used by the current PLV plan:

```bash
cd /PATH/TO/plv-robot-imitation-learning

python - <<'PY'
from huggingface_hub import hf_hub_download

REPO_ID = "robomimic/robomimic_datasets"
LOCAL_DIR = "data/robomimic/hf"

FILES = [
    # PH base tasks.
    "v1.5/lift/ph/low_dim_v15.hdf5",
    "v1.5/lift/ph/demo_v15.hdf5",
    "v1.5/can/ph/low_dim_v15.hdf5",
    "v1.5/can/ph/demo_v15.hdf5",
    "v1.5/square/ph/low_dim_v15.hdf5",
    "v1.5/square/ph/demo_v15.hdf5",
    "v1.5/transport/ph/low_dim_v15.hdf5",
    "v1.5/transport/ph/demo_v15.hdf5",
    "v1.5/tool_hang/ph/low_dim_v15.hdf5",
    "v1.5/tool_hang/ph/demo_v15.hdf5",

    # MH tasks used by the hard/MH queue.
    "v1.5/can/mh/low_dim_v15.hdf5",
    "v1.5/can/mh/demo_v15.hdf5",
    "v1.5/square/mh/low_dim_v15.hdf5",
    "v1.5/square/mh/demo_v15.hdf5",
    "v1.5/transport/mh/low_dim_v15.hdf5",
    "v1.5/transport/mh/demo_v15.hdf5",

    # Extended Paired/MG tasks for Machine B.
    "v1.5/can/paired/low_dim_v15.hdf5",
    "v1.5/can/paired/demo_v15.hdf5",
    "v1.5/can/mg/low_dim_sparse_v15.hdf5",
    "v1.5/can/mg/low_dim_dense_v15.hdf5",
    "v1.5/can/mg/demo_v15.hdf5",
    "v1.5/lift/mg/low_dim_sparse_v15.hdf5",
    "v1.5/lift/mg/low_dim_dense_v15.hdf5",
    "v1.5/lift/mg/demo_v15.hdf5",
]

for filename in FILES:
    print(f"download {filename}")
    hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=filename,
        local_dir=LOCAL_DIR,
    )
PY
```

If Hugging Face Xet causes an environment-specific issue, retry after:

```bash
export HF_HUB_DISABLE_XET=1
```

## Build the config-compatible symlink tree

After direct download, create the symlinks expected by the repo:

```bash
cd /PATH/TO/plv-robot-imitation-learning

python - <<'PY'
from pathlib import Path

root = Path("data/robomimic")
hf_root = Path("../../../hf/v1.5")

links = [
    ("lift/ph/low_dim_v15.hdf5", "lift/ph/low_dim.hdf5"),
    ("lift/ph/demo_v15.hdf5", "lift/ph/demo.hdf5"),
    ("can/ph/low_dim_v15.hdf5", "can/ph/low_dim.hdf5"),
    ("can/ph/demo_v15.hdf5", "can/ph/demo.hdf5"),
    ("square/ph/low_dim_v15.hdf5", "square/ph/low_dim.hdf5"),
    ("square/ph/demo_v15.hdf5", "square/ph/demo.hdf5"),
    ("transport/ph/low_dim_v15.hdf5", "transport/ph/low_dim.hdf5"),
    ("transport/ph/demo_v15.hdf5", "transport/ph/demo.hdf5"),
    ("tool_hang/ph/low_dim_v15.hdf5", "tool_hang/ph/low_dim.hdf5"),
    ("tool_hang/ph/demo_v15.hdf5", "tool_hang/ph/demo.hdf5"),

    ("can/mh/low_dim_v15.hdf5", "can/mh/low_dim.hdf5"),
    ("can/mh/demo_v15.hdf5", "can/mh/demo.hdf5"),
    ("square/mh/low_dim_v15.hdf5", "square/mh/low_dim.hdf5"),
    ("square/mh/demo_v15.hdf5", "square/mh/demo.hdf5"),
    ("transport/mh/low_dim_v15.hdf5", "transport/mh/low_dim.hdf5"),
    ("transport/mh/demo_v15.hdf5", "transport/mh/demo.hdf5"),

    ("can/paired/low_dim_v15.hdf5", "can/paired/low_dim.hdf5"),
    ("can/paired/demo_v15.hdf5", "can/paired/demo.hdf5"),
    ("can/mg/low_dim_sparse_v15.hdf5", "can/mg/low_dim.hdf5"),
    ("can/mg/low_dim_sparse_v15.hdf5", "can/mg/low_dim_sparse.hdf5"),
    ("can/mg/low_dim_dense_v15.hdf5", "can/mg/low_dim_dense.hdf5"),
    ("can/mg/demo_v15.hdf5", "can/mg/demo.hdf5"),
    ("lift/mg/low_dim_sparse_v15.hdf5", "lift/mg/low_dim.hdf5"),
    ("lift/mg/low_dim_sparse_v15.hdf5", "lift/mg/low_dim_sparse.hdf5"),
    ("lift/mg/low_dim_dense_v15.hdf5", "lift/mg/low_dim_dense.hdf5"),
    ("lift/mg/demo_v15.hdf5", "lift/mg/demo.hdf5"),
]

for src_rel, dst_rel in links:
    dst = root / "datasets" / dst_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        continue
    dst.symlink_to(hf_root / src_rel)
    print(f"{dst} -> {hf_root / src_rel}")
PY
```

Then run the extended data script, which also creates MG train/valid masks:

```bash
bash diffusion_policy/scripts/prepare_robomimic_extended_lowdim_abs.sh
```

## Generate absolute-action datasets required by DP

Diffusion Policy runs use `*_lowdim_abs` configs, so they require `low_dim_abs.hdf5`.

For Machine B priority 1, generate `can/ph` and `square/ph` absolute-action files:

```bash
cd /PATH/TO/plv-robot-imitation-learning

for item in can/ph square/ph; do
  task="${item%/*}"
  dtype="${item#*/}"
  input="data/robomimic/datasets/${task}/${dtype}/low_dim.hdf5"
  output="data/robomimic/datasets/${task}/${dtype}/low_dim_abs.hdf5"
  if [[ -f "$output" ]]; then
    echo "skip existing $output"
  else
    python diffusion_policy/scripts/robomimic_dataset_conversion.py \
      --input "$input" \
      --output "$output" \
      --num_workers 2
  fi
done
```

For hard/MH and extended experiments, run:

```bash
bash diffusion_policy/scripts/prepare_robomimic_hard_lowdim_abs.sh
bash diffusion_policy/scripts/prepare_robomimic_extended_lowdim_abs.sh
```

Expected generated files include:

```text
data/robomimic/datasets/can/ph/low_dim_abs.hdf5
data/robomimic/datasets/square/ph/low_dim_abs.hdf5
data/robomimic/datasets/can/paired/low_dim_abs.hdf5
data/robomimic/datasets/can/mg/low_dim_abs.hdf5
data/robomimic/datasets/lift/mg/low_dim_abs.hdf5
data/robomimic/datasets/can/mh/low_dim_abs.hdf5
data/robomimic/datasets/square/mh/low_dim_abs.hdf5
data/robomimic/datasets/transport/mh/low_dim_abs.hdf5
data/robomimic/datasets/transport/ph/low_dim_abs.hdf5
data/robomimic/datasets/tool_hang/ph/low_dim_abs.hdf5
```

## Minimal Machine B data checklist

For the currently assigned Machine B workload, the minimum required data is:

| Workload | Required raw datasets | Generated files |
| --- | --- | --- |
| DP PH remaining | `can/ph/low_dim_v15.hdf5`, `square/ph/low_dim_v15.hdf5` | `can/ph/low_dim_abs.hdf5`, `square/ph/low_dim_abs.hdf5` |
| BC-RNN Paired | `can/paired/low_dim_v15.hdf5` | none |
| DP Paired | `can/paired/low_dim_v15.hdf5` | `can/paired/low_dim_abs.hdf5` |
| BC-RNN MG | `can/mg/low_dim_sparse_v15.hdf5`, `lift/mg/low_dim_sparse_v15.hdf5` | deterministic train/valid masks |
| DP MG | `can/mg/low_dim_sparse_v15.hdf5`, `lift/mg/low_dim_sparse_v15.hdf5` | `can/mg/low_dim_abs.hdf5`, `lift/mg/low_dim_abs.hdf5` |

The `demo_v15.hdf5` files are not always read by low-dimensional training, but keep them in the mirror so the dataset tree matches the original local setup and symlink scripts do not create broken links.

## Verify before training

Run this before starting Machine B training:

```bash
python - <<'PY'
from pathlib import Path

required = [
    "data/robomimic/datasets/can/ph/low_dim_abs.hdf5",
    "data/robomimic/datasets/square/ph/low_dim_abs.hdf5",
    "data/robomimic/datasets/can/paired/low_dim.hdf5",
    "data/robomimic/datasets/can/paired/low_dim_abs.hdf5",
    "data/robomimic/datasets/can/mg/low_dim.hdf5",
    "data/robomimic/datasets/can/mg/low_dim_abs.hdf5",
    "data/robomimic/datasets/lift/mg/low_dim.hdf5",
    "data/robomimic/datasets/lift/mg/low_dim_abs.hdf5",
]

missing = [p for p in required if not Path(p).exists()]
if missing:
    print("MISSING:")
    for p in missing:
        print("  " + p)
    raise SystemExit(1)
print("Machine B PLV data is ready.")
PY
```

Also verify that MG masks exist:

```bash
python - <<'PY'
import h5py

for path in [
    "data/robomimic/datasets/can/mg/low_dim.hdf5",
    "data/robomimic/datasets/lift/mg/low_dim.hdf5",
]:
    with h5py.File(path, "r") as f:
        print(path, sorted(f["mask"].keys()))
PY
```

After this checklist passes, follow `docs/plv_multi_machine_training_plan.md` and start the Machine B training queue.
