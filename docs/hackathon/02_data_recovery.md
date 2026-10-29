# 02 — Dataset recovery

## Source

* Repository: `robomimic/robomimic_datasets` (Hugging Face dataset), path `v1.5/`
  (robosuite 1.5-compatible files), revision
  `74fa018461f479cd9fd15b924a16103012096203`.
* Files, sizes and sha256: `configs/hackathon/dataset_manifest.json`.
* Licensing: robomimic datasets are distributed by the robomimic authors; they
  are not redistributed in this repository (HDF5 files are git-ignored).

Reproduce (about 1 minute plus ~20 s conversion):

```bash
bash scripts/hackathon/download_datasets.sh      # TASKS="can square" by default
```

This downloads with the pinned revision, checks sha256, builds the symlink tree
the configs expect (`data/robomimic/datasets/<task>/ph/low_dim.hdf5 -> ../../../hf/v1.5/...`),
and regenerates `low_dim_abs.hdf5` with the ORIGINAL converter
`diffusion_policy/scripts/robomimic_dataset_conversion.py` (replays every demo in
robosuite and replaces delta OSC actions by absolute end-effector targets).

## Validation (tests/hackathon/test_dataset.py, run in Environment A)

| Check | Can PH | Square PH |
|---|---|---|
| HDF5 opens, `env_args.env_name` | PickPlaceCan | NutAssemblySquare |
| Demonstrations | 200 | 200 |
| Total transitions | 23 207 | 30 154 |
| Masks | `train` 180 / `valid` 20, disjoint (shipped in v1.5 files) | same |
| Obs keys used | object(14), robot0_eef_pos(3), robot0_eef_quat(4), robot0_gripper_qpos(2) = 23 | same (object 14) |
| Raw actions | 7-D delta OSC in [-1, 1] | same |
| Absolute actions (`low_dim_abs`) | 7-D: pos(3) m, axis-angle(3), gripper(1); position targets track the next achieved EEF position (mean abs error < 5 cm asserted) | same |
| Policy action | 10-D: pos(3) + rotation_6d(6) + gripper(1) via `RotationTransformer('axis_angle','rotation_6d')`, normalized by the dataset normalizer | same |
| Controller | robosuite 1.5 `BASIC` composite, right arm `OSC_POSE`, `control_delta` set to False by `set_absolute_controller_config` | same |

Training uses the paper split (`task.dataset.val_ratio=0.0
+task.dataset.train_filter_key=train +task.dataset.val_filter_key=valid`), as in
`diffusion_policy/scripts/run_robomimic_paper_dp_lowdim.sh`.

Lift PH was also downloaded (verified hashes) but not used.

## Not recovered

MH / MG / Paired datasets and Transport / Tool Hang tasks were not downloaded
(out of scope for the reduced hackathon pool). The historical per-checkpoint
results for those tasks remain frozen in the paper artifacts only.
