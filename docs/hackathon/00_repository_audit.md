# 00 — Repository audit (Cosmos-PLV, Nebius x NVIDIA Global AI Hackathon, Physical AI track)

Audience: judges and engineers new to this repo. Scope: what the repo at the audited commit contains,
what is missing, what runs today, and what must not be published. Read-only audits ran on 2026-10-29
between 02:36 and 02:45 KST. The environment checks in §5 and the pool status in §6 are more recent;
§6 is a snapshot taken at 03:49 KST.

Asset classes: **AVAILABLE_AND_VERIFIED** (present and checked by checksum, parse, import or execution),
**PRESENT_BUT_UNTESTED**, **MISSING_AND_RECONSTRUCTABLE** (absent, with a known recipe or source),
**MISSING_WITHOUT_KNOWN_SOURCE**. Anything not checked is marked **UNVERIFIED**. **INFERRED** means it was
deduced from reading code, without running it.

## 1. Audited commit and repository facts

| Item | Finding |
|---|---|
| Audited commit | `main` @ `bbdcece` (2026-10-02 15:25 +0900), "Tighten PLV paper claims and correct generated copy" |
| Hackathon branch | `hackathon/cosmos-plv-2026`, created from `bbdcece`. At 03:49 KST it had no commits of its own. The hackathon additions are uncommitted working-tree files: `cosmos_plv/`, `configs/`, `scripts/hackathon/`, `tests/hackathon/`, `results/`, `docs/hackathon/`, `.env.example`, `requirements-cosmos.txt`, and a 9-line `.gitignore` addition |
| History | 38 commits by 3 author identities. The first commit is dated 2026-01-14. Only 2 files were ever deleted (`4f6bd97`, two `act_hand52` configs) |
| Tracked files | 1,338. Of these, 403 are force-added under the otherwise gitignored `data/robomimic/plv_assets/`: 265 CSV, 74 JSON and 64 MD frozen analysis outputs |
| Size | `.git` is 27 MB (1 pack, 26.00 MiB, 1,929 objects). The largest blobs are a kitchen texture (4.46 MB), six versions of the arXiv tarball (about 1.3–1.4 MB each) and Franka meshes. No weights or datasets are in history. The working tree was 335 MB at audit time, mostly HDF5 downloads. It is now about 31 GB because of the gitignored training outputs in `data/outputs/hackathon_pool/` (§6) |
| Git LFS / tags / releases | No LFS objects. No local or remote tags. The remote has a single head, `refs/heads/main`. GitHub Releases were not queried (UNVERIFIED) |
| CI | None. There is no `.github/`, `.gitlab-ci.yml`, `Makefile`, `Dockerfile`, `tox.ini` or pre-commit config |
| Remote visibility | The repo docs describe the GitHub repo as **private** |
| Upstream | A fork of Diffusion Policy (real-stanford) plus robomimic tooling. `media/*.png` are byte-identical to the upstream images |

## 2. Current vs superseded documents

**Current paper:** *"Cheap Metrics First, Replanning Diagnostics Second: Offline Checkpoint Screening for Robot
Imitation Learning"*, by Hyun Joon Cho (Sungkyunkwan University).

**README.md is an older research-workspace description**, last changed 2026-07-14. Its framing ("PLV should rank checkpoints better than validation loss"), its Machine A/B training split, its status text and its document list all predate the paper's final claims. Do not use it as a description of the results.

| Document | Status |
|---|---|
| `docs/plv_arxiv/main.tex`, `appendix.tex`, `supplementary.tex`, `generated_reframed/*.tex`, `figures/*.pdf`, `main.pdf` | **CURRENT** (HEAD, 2026-10-02). The 10-02 editorial pass left the experimental inputs and numeric tables unchanged |
| `docs/plv_arxiv/README.md` | Current build and claim-boundary guide. It references several files that are not in this clone (the `analysis/` package, `make_main_results.py`, the claim sheet). Some of these have copies inside the tarball |
| `docs/plv_arxiv/plv_arxiv_submission_20260928.tar.gz` | Current frozen ancillary bundle: `anc/results/` plus 6 scripts in `anc/code/`. Its own README calls it an "analysis inventory, not a complete runnable training repository" |
| `docs/plv_arxiv/STORAGE_RETENTION.md` (2026-09-28) | Current fact record. It lists what was deleted and what was retained |
| `docs/plv_paper_current.pdf` | **Stale alias** (INFERRED): its sha256 and abstract differ from `docs/plv_arxiv/main.pdf` |
| `docs/plv_paper_draft.md` (2026-07-01, "Beyond Validation Loss: ...") | **Superseded** preliminary draft |
| `docs/plv_chunk_aware_study.md`, `docs/plv_chunk_aware_results_and_paper_update.md` (07-12) | Historical, but still the method reference for the Level-3 metrics (Panel A of the execution table) |
| `docs/plv_research_claims_and_writing_notes.md` | Working notes, partly superseded. Its "main metric = PLV-stability" choice was dropped. Its dynamics protocol section is still accurate |
| `docs/plv_multi_machine_training_plan.md`, `current_machine_assignment.md`, `machine_b_status_2026-06-23.md` | **Obsolete** June 2026 operations notes |
| `docs/plv_active_*` (V5–V44) | Post-July extension track, not part of the main claims. Final status of V42–V44 is UNVERIFIED. `plv_active_stage1_after_dp_execution_withdrawal.md` is explicitly withdrawn |

Headline result, for context (`generated_reframed/table_policy_split.tex`): on the 24 Diffusion Policy runs, PLV-1 smoothness reaches ρ 0.556 against 0.462 for Action L1, with Regret@30 of 0.006 against 0.017. BC-RNN runs show the reverse order. Pooled Regret@30 is best for open-loop smoothness (0.019). The paper says outright that learned dynamics is **not** validated as an automatic reranker.

## 3. Asset inventory

| Asset | Location | Class at audit | Current state / note |
|---|---|---|---|
| robomimic v1.5 PH lift/can/square `low_dim` + `demo` HDF5 | `data/robomimic/hf/v1.5/*/ph/`, symlinked from `data/robomimic/datasets/<task>/ph/` | AVAILABLE_AND_VERIFIED | Downloaded 2026-10-29. Each has 200 demos with masks `train`=180 / `valid`=20. Integrity was not hash-checked against Hugging Face (no reference hash; UNVERIFIED) |
| `low_dim_abs.hdf5`, can/square PH (needed by `*_lowdim_abs` configs) | `data/robomimic/datasets/{can,square}/ph/` | MISSING_AND_RECONSTRUCTABLE | **Reconstructed** with `robomimic_dataset_conversion.py` (§5). Masks are carried over |
| DP low-dim training path (`train.py`, workspace, policy, dataset, env runner) | `diffusion_policy/...` | AVAILABLE_AND_VERIFIED | Imports pass. The README smoke test passes (§5) |
| Paper DP run recipe | `diffusion_policy/scripts/run_robomimic_paper_dp_lowdim.sh` | AVAILABLE_AND_VERIFIED | Defaults: 1001 epochs × 200 steps, rollout/checkpoint every 50, `N_TEST=50`, `N_ENVS=28`, 100 DDPM steps, batch 128, `topk.k=50`, `LOGGING_MODE=online` |
| arXiv tarball incl. paper master checkpoint manifest | `docs/plv_arxiv/plv_arxiv_submission_20260928.tar.gz` → `anc/results/paper_master_manifest.{csv,json}` | AVAILABLE_AND_VERIFIED | The tarball sha256 and `anc/SHA256SUMS` both match. The manifest has 2,501 rows: 1,365 primary and 1,136 external. By policy family: bc_rnn 1,218, act 727, diffusion_policy 556. There are 45 primary runs. Checkpoint paths in it are historical identifiers only |
| DP PH can/square **seed-0** per-checkpoint success (21 + 21, 50 rollouts each) | the same manifest; `docs/plv_paper_figures/run_success_summary.csv` | AVAILABLE_AND_VERIFIED | Frozen reference only. Can: best 1.00 at epoch 300, final 0.96. Square: best 0.96 at epoch 750, final 0.88 |
| DP PH seeds 1/2 final scores | `docs/machine_b_status_2026-06-23.md` | AVAILABLE_AND_VERIFIED (text only) | There are no per-checkpoint rows for seeds 1/2 in the manifest |
| Selected-checkpoint controls, execution controls, same-start results | tarball `anc/results/` | AVAILABLE_AND_VERIFIED | Frozen outputs only (e.g. `selected_checkpoints.csv` has 47 DP rows) |
| Frozen PLV analysis outputs; paper figure data | `data/robomimic/plv_assets/` (403 files); `docs/plv_paper_figures/` (29 CSV) | AVAILABLE_AND_VERIFIED | The CSVs parse. Per-checkpoint PLV metrics are present only for some cohorts (e.g. DP can/paired seed 0) |
| Manuscript sources and PDFs | `docs/plv_arxiv/`, `docs/*.pdf` | AVAILABLE_AND_VERIFIED | All 10 `.sha256` files pass |
| PLV metric scripts (Action L1, PLV-1, open-loop, chunk-aware) | `diffusion_policy/scripts/` | PRESENT_BUT_UNTESTED | **Now executed** on two test checkpoints (§5) |
| Other scripts (298 tracked files in `diffusion_policy/scripts/`) | same | PRESENT_BUT_UNTESTED | Not run. About 85 queue/run shell scripts hard-code Machine A paths |
| Unit tests | `tests/` (49 tracked files) | PRESENT_BUT_UNTESTED | Not run. `tests/test_robomimic_lowdim_runner.py` uses an upstream author path |
| Site rollout videos | `site/assets/videos/` | PRESENT_BUT_UNTESTED | They play (128×128, 10 fps, 20 s). Which checkpoint produced them is undocumented (UNVERIFIED) |
| RB10 real-robot scripts and configs; `.beads/` tracker | repo root `rb10_*.py`, `son_utils/`; `.beads/` | PRESENT_BUT_UNTESTED | Unrelated to the sim track. The `.beads` issue files are empty |
| **DP PH can/square policy checkpoints** | historical Machine A outputs | MISSING_AND_RECONSTRUCTABLE | Deleted 2026-09-28 (STORAGE_RETENTION). They can be retrained, but **not bit-identically**. A reduced pool is being retrained now (§6) |
| Same-start (E/M/S) evaluator, paper version | tarball `anc/code/plv_chunk_aware_sim_fidelity.py` (881 lines) | MISSING_AND_RECONSTRUCTABLE | **Only in the tarball.** The repo copy is an older 314-line version. The archived evaluator reads `expert_obs_histories`, which the repo's start sampler does not produce, so it will not run unmodified (INFERRED) |
| Source CSVs behind the manifest (`plv_final_merged_checkpoint_analysis.csv`, `active_plv/v147…v168`) | Machine A (retained per STORAGE_RETENTION) | MISSING_AND_RECONSTRUCTABLE | Machine A reachability is UNVERIFIED. The outcome content is largely mirrored in the manifest |
| Other robomimic datasets (MH, MG, paired, transport, tool_hang) | Hugging Face `robomimic/robomimic_datasets` v1.5 | MISSING_AND_RECONSTRUCTABLE | Not needed for the hackathon pool |
| W&B run history (project `plv_robomimic_validation`) | cloud | MISSING_AND_RECONSTRUCTABLE | Access UNVERIFIED. Not needed |
| Exact paper checkpoint binaries (all 2,501 rows), evaluator/latent-dynamics/hazard payloads, the model-inclusive `plv_remaining_validation_reproducibility.tar.zst` | — | MISSING_WITHOUT_KNOWN_SOURCE | Deleted 2026-09-28. No copy exists under `<local path>{outputs,diffusion_policy,rmdit-policy,data}`. The 115 weight files under `<local path>` are all real-robot `son_pick_and_place_*` runs |
| Four-seed execution-control generator (`action_l1_seed_median`, `model_state_replan_common_seed_*`, …) | — | MISSING_WITHOUT_KNOWN_SOURCE | Not in the repo and not in the tarball. Only its outputs (`anc/results/execution_control*.csv`) exist |
| Task-balanced hierarchical bootstrap (`analysis/` package, `make_main_results.py`), `plv_merge_same_start_calibration.py` | — | MISSING_WITHOUT_KNOWN_SOURCE | Only the appendix text describes them |
| Machine B outputs (DP PH seeds 1/2), raw per-episode logs and videos for primary checkpoints, the paper-run `.hydra/` configs | — | MISSING_WITHOUT_KNOWN_SOURCE | The hydra configs can be re-derived from the run script and base config, but the originals are gone. Whether the Machine B outputs survive is UNVERIFIED |
| Analysis code commit `815d205…` (from tarball `run_config.json`) | — | MISSING_WITHOUT_KNOWN_SOURCE | Not present in this clone's history |

## 4. Key scripts and what they compute

Conventions shared by every metric:
- Each metric is a **cost** (lower is better). ρ is Spearman computed on the negated cost.
- The label is `test/mean_score` from `<run_dir>/logs.json.txt`: success over 50 training-time rollouts per checkpoint, env seeds 100000–100049.
- EMA weights are loaded.
- DP convention: H=16, To=2, stride m=8, offset s=1, so the executable slice is `action_pred[:, 1:9]`. Actions are absolute and 10-D (pos 3 + rot6d 6 + gripper 1).
- Evaluation inputs are the **first** 64 (Action L1, open-loop) or 32 (PLV starts) windows of the validation sampler. Under the `valid` mask these all come from early windows of a single demo, `demo_0`.
- Every metric script `chdir`s to the repo root, so pass absolute `--roots`/`--output`.

| Level | Metric (output column) | Script and location | Paper flags | CLI-default differences |
|---|---|---|---|---|
| 1 | Action L1 (`val_action_l1`): mean \|π(x)−a\| over 64 windows × 8 steps × dims, in raw action units, from one **unseeded** diffusion sample | `plv_lowdim_checkpoint_metrics.py` `evaluate_action_metrics` (:457-479). Also emitted by the open-loop and chunk-aware scripts | `--max-eval-samples 64` | Defaults match. Note that DDPM sampling is never seeded in this script, so values depend on process RNG history |
| 2 | Open-loop smoothness (`open_loop_smoothness`): mean squared second difference of the first executable action along expert windows (teacher-forced) | `plv_open_loop_smoothness_baseline.py` (:33-80) | `--max-eval-samples 64 --horizon 10 --skip-latest` | `--roots`/`--output` are required. `--policy-seed` defaults to None; pass it (e.g. 0) for reproducibility |
| 2 | PLV-1 smoothness (`smoothness` raw, `plv_smoothness_score_norm` robust z). Same curvature statistic, but each next state comes from f_ψ with replanning every primitive step ("one-step receding-replan proxy"). PLV-1 U+S = `plv_stability_score_norm` | `plv_lowdim_checkpoint_metrics.py` `rollout_plv_lite` (:1104-1207). f_ψ = `build_mlp_dynamics_ensemble` (:760-832): 5 × MLP(256), train-mask transitions, rebuilt in-process, never saved | `--rollout-starts 32 --plv-horizon 10 --dyn-type mlp --dyn-models 5 --dyn-max-transitions 30000 --dyn-epochs 8 --dyn-hidden-dim 256 --skip-latest` | **The defaults differ:** `--plv-horizon 5`, `--dyn-type linear`, `--dyn-max-transitions 20000`, and repo-relative `--roots`/`--output`. Which column the paper's "PLV-1 smoothness" label uses is UNVERIFIED (likely `plv_smoothness_score_norm` at H=10) |
| 3 | Boundary jump (`boundary_jump_l1_norm`, `chunk_boundary_score_norm`): first new action minus last executed old action, normalized by train-split action std | `plv_chunk_aware_checkpoint_metrics.py` (:207-213) | main pass `--action-horizons 8 10 16 --sampling-seeds 0` (main table H=10). Four-seed: `--sampling-seeds 0 1 2 3`, then the per-checkpoint median | Defaults are `--action-horizons 10 --sampling-seeds 0`, and the dyn defaults already match (mlp, 30000). **`--action-horizons 8` yields NaN** because there is no replan boundary |
| 3 | Replan overlap (`replan_overlap_l1_norm`, `chunk_replan_score_norm`): retained old suffix `[9:16]` vs aligned new prefix `[1:8]`, L=7 | same script (:193-206). Optional four-seed recursive-vs-direct variant: `plv_v2_replan_checkpoint_metrics.py` | as above | as above. Whether the paper's Level-3 headline numbers come from the chunk-aware script or the v2 script is UNVERIFIED |

Analysis tooling:
- `analyze_plv_checkpoint_metrics.py` computes oriented Spearman, Pearson and Regret@k for integer k. Use k = ⌈0.3·N⌉ for Regret@30. It ignores the Level-3 columns.
- `make_plv_paper_tables.py` prints only Regret@1 and Regret@3.
- `plv_chunk_aware_analysis.py` **crashes on a Can/Square-only pool** with `KeyError: 'metric'`, because its hard-task (transport/tool_hang) prescreening runs on an empty frame.
- `plv_extract_rollout_episodes.py` and `plv_validate_checkpoint_artifacts.py` are structural guards. They do not compute metrics.
- The paper's within-run, task-balanced aggregation with a hierarchical bootstrap has no code in this repo (§3).

## 5. Environment (current state)

- **Conda env `robodiff`** (`<conda>/envs/robodiff`): Python 3.10.19, torch 2.1.2+cu121, robosuite 1.5.1, robomimic 0.3.0, mujoco 3.4.0, free-mujoco-py 2.1.6, diffusers 0.11.1, hydra-core 1.3.2, wandb 0.21.3, h5py 3.14.0. The editable `diffusion_policy` install points at this checkout.
- **gym 0.26.2 comes from `~/.local/lib/python3.10/site-packages`**, not from the env. `conda_environment.yaml` pins `gym=0.21.0`. Training-time rollouts in the current pool still return non-trivial success (e.g. a can checkpoint saved as `test_mean_score=0.960`), so the env runner works in practice. The code paths beyond that were not audited.
- **The one required fix:** robomimic 0.3.0 imports `mujoco_py`. Its compiled extension links the system `libLLVM-12`, which needs libffi 7, but the env's `lib/libffi.so.7` is a symlink to `libffi.so.8.1.2`. The fix is `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7`, set in `scripts/hackathon/env.sh`. That script also sets `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl` and `WANDB_MODE=offline`.
- **Verified with the fix in place:**
  - `robomimic_dataset_conversion.py` converted can PH and square PH to `low_dim_abs.hdf5` in about 10 s each.
  - The README "Minimal Smoke Test" exited 0 in 77 s.
  - `plv_lowdim_checkpoint_metrics.py`, `plv_open_loop_smoothness_baseline.py` and `plv_chunk_aware_checkpoint_metrics.py` all ran successfully on two test checkpoints.
- **Differs from the original paper stack**: Python 3.9.18, PyTorch 1.12.1, robomimic 0.5.0, mujoco 3.3.7, per the research notes. Numerical equivalence with the paper's environment is UNVERIFIED. This is moot for the binaries, which are retrained anyway.
- **Operational notes:**
  - Each DP low-dim checkpoint is about 1.05 GB (1,054,396,210 bytes) because it includes optimizer state.
  - Disk had 313 GB free at 03:49 KST.
  - The repo's queue scripts default to a Machine A interpreter path. Call the Python scripts directly, or set `PYTHON`.
  - The default `checkpoint.topk.k=5` in the base config would outcome-filter the pool. The paper wrapper sets k=50.

## 6. Recovery opportunities and what was done

1. **Data:** the PH lift/can/square v1.5 HDF5 files were downloaded, and can/square were converted to `low_dim_abs.hdf5` (§5).
2. **A fresh, reduced checkpoint pool** is being trained with the **unmodified original** `run_robomimic_paper_dp_lowdim.sh`. The wrapper `scripts/hackathon/train_checkpoint_pool.sh` only sets environment variables:
   - kept from the paper recipe: `MAX_TRAIN_STEPS=200`, `N_TEST=50`, `NUM_INFERENCE_STEPS=100`, `BATCH_SIZE=128`;
   - changed for the hackathon: `N_TRAIN=0`, `N_ENVS=25`, `LOGGING_MODE=offline`, `nice -n 10`.

   Each saved checkpoint receives the same 50 training-time rollouts on seeds 100000–100049.

   | Run (`data/outputs/hackathon_pool/`) | Epochs | Save/rollout every | Planned checkpoints | Saved at 03:49 KST |
   |---|---|---|---|---|
   | `paper_dp_can_abs_ph_seed1_e200_s200_r5_inf100` | 0–200 | 5 | 41 | 10 (to epoch 45), running |
   | `paper_dp_can_abs_ph_seed2_e200_s200_r10_inf100` | 0–200 | 10 | 21 | 9 (to epoch 80), running |
   | `paper_dp_square_abs_ph_seed1_e300_s200_r10_inf100` | 0–300 | 10 | 31 | 8 (to epoch 70), running |

   Total runtime is UNVERIFIED until the runs finish.
3. **Historical seed-0 curves are a frozen reference only.** The new pool uses different seeds (1/2), a shorter schedule (200–300 epochs instead of 1000, so the cosine LR decays faster) and a different software stack. Paper numbers therefore cannot be re-derived checkpoint-for-checkpoint, and the new and old values may only be compared as distributions.
4. **Not recovered:** the four-seed control generator and the task-balanced bootstrap. The hackathon analysis needs its own small aggregation script (within-run oriented Spearman and Regret@30, ties broken by checkpoint path), documented in `06_experiment_protocol.md`.

## 7. Publication and sensitivity findings (categories and counts only)

No credential patterns were found in HEAD or in any commit of history. The scan covered API keys, passwords, HF/W&B tokens, `sk-`, `ghp_`, `github_pat_`, AWS keys, private-key headers and Slack tokens; its only hits were false positives on variable names. The remaining concerns are privacy and exposure:

| Category | Where | Count |
|---|---|---|
| Machine A home paths (`<machine-A path>`) | queue/run shell scripts (~85), docs (5), audit JSONs (3), cleanup tool, `plv_assets` files (5) | 99 tracked files, plus about 20 `.meta.json` files and `run_config.json` inside the arXiv tarball |
| This host's home paths (`<local path>`) | `son_*` configs, `son_utils/`, one dataset script, `.beads/export-state` | 11 tracked files |
| Host name | tarball `anc/results/run_config.json` | 1 |
| W&B run URLs (personal/university entity) | `data/robomimic/plv_assets/plv_status_report.md` | 2 occurrences in 1 tracked file. In addition, 64 tracked files mention `wandb`, mostly configs, and the default logging mode is online |
| Private LAN IP of the lab robot | `rb10_*.py` (including the stray `rb10_eval_vr_collect copy.py`) | 6 files. Two more upstream real-robot scripts carry upstream's private-range default IP |
| Emails | author institutional emails in `docs/plv_arxiv/main.tex` and `site/index.html`; upstream contact emails in the Franka XMLs; git commit metadata | 2 tracked files with author emails. Git history has 3 author names / 2 distinct emails |
| Private research and coordination notes | `docs/plv_active_*` preregistration and result notes; `data/robomimic/plv_assets/plv_status_report*.md`; other research-direction, machine-coordination and paper-process docs (e.g. `plv_research_claims_and_writing_notes.md`, `machine_b_status_2026-06-23.md`, `REVIEW_REVISION_20260930.md`) | 38 + 28 + 9 tracked files |

- **License:** the root `LICENSE` is MIT, "Copyright (c) 2023 Columbia Artificial Intelligence and Robotics Lab", byte-identical to upstream Diffusion Policy. The tarball's `anc/code/LICENSE` is the same text. **There is no separate license statement for the PLV-authored code.**
- **Third-party notices present:**
  - Apache-2.0 for `adept_models` (DSuite) and `third_party/franka`.
  - MIT for minGPT (Karpathy).
  - Per-file headers in `block_pushing/**` (Apache-2.0, 11 files), `adept_envs/**` (Apache-2.0, about 17 files), `pymunk_override.py` (MIT-style), `imagecodecs_numcodecs.py` (BSD-3) and `bet/libraries/loss_fn.py` (MIT).
  - Dependencies: robosuite and robomimic are MIT, mujoco is Apache-2.0.
  - The robomimic dataset license was not checked (UNVERIFIED).
- **Implication:** any public hackathon release needs a scrubbed export with fresh history, not this tree or its history.

## 8. Known limitations of this audit (UNVERIFIED items)

- Dataset integrity was not checked against Hugging Face hashes. GitHub Releases were not queried, and the project-site URL was not fetched.
- Machine A reachability is unknown, so it is not confirmed that its retained CSVs and cleanup manifests exist. Whether the Machine B outputs (DP PH seeds 1/2) survive anywhere is also unknown.
- It is unknown which CSV column the paper's "PLV-1 smoothness" label uses, and whether the Level-3 headline numbers come from `plv_chunk_aware_checkpoint_metrics.py` or from `plv_v2_replan_checkpoint_metrics.py`.
- The exact computation of the four-seed controls and of the task-balanced bootstrap is unknown beyond the appendix equations.
- `docs/plv_paper_current.pdf` being stale is INFERRED from its hash and abstract, not confirmed with the authors.
- The repo unit tests were not run. Only the three metric scripts were run, and only on two test checkpoints, not on a full pool.
- How well the current stack (torch 2.1.2, robomimic 0.3.0, gym 0.26.2) matches the paper stack numerically is unknown. Full-pool training time and the final checkpoint counts are pending (§6).
- The provenance of the site rollout videos is undocumented.
- The audit, the training jobs and other agents shared one checkout, so file timestamps after 02:37 KST reflect concurrent activity.
