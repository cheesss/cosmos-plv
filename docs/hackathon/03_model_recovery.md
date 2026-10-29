# 03 — Policy checkpoint recovery

## Original checkpoints

The paper's 1,365-checkpoint retrospective pool cannot be recovered.
`docs/plv_arxiv/STORAGE_RETENTION.md` records that every policy payload was
deleted on 2026-09-28. Searches of the repository, git history (no LFS, no tags,
no releases) and this machine found no robomimic Can/Square low-dim Diffusion
Policy weights. The only weights on the machine belong to unrelated real-robot
projects. See `00_repository_audit.md`.

## Reduced retrained pool

The checkpoints were retrained with the **unchanged** original script
`diffusion_policy/scripts/run_robomimic_paper_dp_lowdim.sh`, called through
`scripts/hackathon/train_checkpoint_pool.sh`.

| | value |
|---|---|
| Policy | Diffusion Policy, ConditionalUnet1D (`train_diffusion_unet_lowdim_workspace`), 65.9 M parameters, EMA weights |
| Execution contract (read from each checkpoint) | horizon 16, n_obs_steps 2, n_action_steps 8, absolute 10-D actions (pos + rot6d + gripper), 100 DDPM inference steps |
| Data | robomimic v1.5 PH, `low_dim_abs.hdf5`, paper split (`train` 180 / `valid` 20 demos) |
| Optimisation | AdamW lr 1e-4, cosine schedule, 500 warm-up steps, batch 128, up to 200 steps per epoch (both tasks have ≤ 200 batches per epoch) |
| Success label per checkpoint | the repository's own runner at save time: 50 test rollouts, env seeds 100000–100049, max 400 steps; per-episode outcomes in `logs.json.txt` |
| Logging | W&B offline; `logs.json.txt` per run |
| Hardware | 1x RTX 3090, all three runs concurrently, `nice -n 10`; each run used about 2.3 GB of GPU memory. GPU utilisation was not recorded per run. |

| run | task | seed | checkpoints | epochs | success min / max / last | best epoch | wall clock |
|---|---|---|---|---|---|---|---|
| `paper_dp_can_abs_ph_seed1_e200_s200_r5_inf100` | Can | 1 | 41 (every 5 epochs) | 0–200 | 0.00 / 1.00 / 0.98 | 85 | 3.88 h |
| `paper_dp_can_abs_ph_seed2_e200_s200_r10_inf100` | Can | 2 | 21 (every 10) | 0–200 | 0.00 / 1.00 / 1.00 | 90 | 2.62 h |
| `paper_dp_square_abs_ph_seed1_e300_s200_r10_inf100` | Square | 1 | 31 (every 10) | 0–300 | 0.00 / 0.90 / 0.88 | 120 | 4.01 h |

That is **93 checkpoints** with 4,650 training-time simulator rollouts (each 1.05 GB,
95 GB in total, git-ignored). Success ranges from 0 to 1.0 within every run, so the
pool can support ranking statistics. The three runs differ in character:

* **Can**: success rises quickly and saturates near 0.9–1.0 after about epoch 60.
  Checkpoints in this regime are separated mostly by measurement noise.
* **Square**: success climbs to about 0.7 by epoch 30, then plateaus and
  fluctuates between 0.76 and 0.90 from epoch 90 to 300.

Validation loss reaches its minimum early and then rises in all three runs
(0.017 → about 0.06) while success keeps rising or holds steady. This is the
paper's motivating observation, reproduced on the new pool.

## Discovery and validation manifest

`python scripts/hackathon/make_pool_manifest.py` writes
`results/hackathon/checkpoint_manifest.csv` and `.json`. They contain one row per
checkpoint: run, seed, epoch, logged validation loss and train loss, training-time
success, sha256, size, save time and hours since run start. The script also lists
validation problems: a filename that does not parse, missing rollouts, wrong
chunking, or `n_test` ≠ 50. Checkpoints are discovered by
`cosmos_plv.checkpoint_adapter.discover_checkpoints`; `latest.ckpt` is ignored.

## Differences from the paper pool (limitations)

* 3 runs instead of 45. Only seeds 1 and 2 (Can) and seed 1 (Square). Only PH data.
* 200–300 epochs instead of 1,000. Our runs stop before the long plateau and
  overfitting phase, so "later epoch is better" is a strong free baseline.
  Experiments 1 and 3 report it.
* New hardware and software (torch 2.1, robosuite 1.5.1, mujoco 3.4). The
  retrained checkpoints are not the paper's binaries. Historical numbers are
  quoted as history only.
