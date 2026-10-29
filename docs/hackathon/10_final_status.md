# 10 — Final status

Status values:
* **COMPLETE** means verifiable evidence exists in this repository.
* **PARTIAL** means it is done in part; the gap is stated.
* **BLOCKED** means a technical reason or a missing authorization prevents it.
* **OWNER** means only the repository owner can do the step.

## Competition requirements

| requirement | status | evidence |
|---|---|---|
| Genuinely working application | COMPLETE | `dashboard/app.py`; recorded live session `media/hackathon/demo.mp4`; `scripts/hackathon/smoke_test.sh` |
| Runtime use of Nebius Token Factory | COMPLETE | 7,003 logged calls, 0 failures, served model `nvidia/Nemotron-3.5-Super-VL` (`results/hackathon/nebius_calls.jsonl`, `explore/nebius_calls_explore.jsonl`, `docs/hackathon/05_nebius_execution.md`) |
| Uses at least one NVIDIA open model | COMPLETE | `nvidia/Cosmos3-Edge` @344d602b running locally (2,970 eval calls); `nvidia/Nemotron-3.5-Super-VL` and `nvidia/Nemotron-3_5-Lightning` on Nebius |
| Real functional integration (not branding) | COMPLETE | `cosmos_plv/cosmos_backend.py`, `nebius_backend.py`, `visual_diagnostics.py`; Experiment 2 on 330 real rollouts |
| Public repository with an OSS license | COMPLETE | https://github.com/cheesss/cosmos-plv (public, MIT). It is a sanitized export made by `scripts/hackathon/export_public_repo.sh`, which validates the export and scans it for sensitive strings. Created 2026-10-29 with the owner's approval. |
| Setup / run / test instructions | COMPLETE | `README_HACKATHON.md` (exported as `README.md`) |
| English project description | COMPLETE | `docs/hackathon/09_devpost_submission.md` |
| Public/unlisted YouTube video ≤ 3 min | PARTIAL / OWNER | `media/hackathon/demo.mp4` (2:23, 1600x992, English captions). The upload needs the owner's YouTube account. |
| ≥ 1 min of key modules operating | COMPLETE | About 84 s of the 143 s video is real dashboard and rollout footage: live metric script, live Cosmos3-Edge, live Nebius call, the judge-audit panel, real simulator videos |
| Feedback on Nebius / NVIDIA tools | COMPLETE | `09_devpost_submission.md` → "Feedback" |
| New functionality after 2026-08-26 explained | COMPLETE | `08_original_vs_new_work.md` |
| Devpost submission | OWNER | text ready to paste in `09_devpost_submission.md` |

## Engineering deliverables

| item | status | notes |
|---|---|---|
| Repository audit (00) | COMPLETE | |
| Resource audit (01) | COMPLETE | |
| Data recovery (02) | COMPLETE | pinned revision and sha256; download script |
| Model recovery (03) | COMPLETE | 93 retrained checkpoints. The paper checkpoints are unrecoverable (deleted 2026-09-28). |
| Cosmos integration (04) | COMPLETE (Mode A) / PARTIAL (Mode B, exploratory) | Forward dynamics run through the trained Franka action slot: a coarse directional signal, not a faithful rollout (07 §C3). The first attempt (E4) went through an untrained slot and is retracted. |
| Nebius execution (05) | COMPLETE | Token Factory only. AI Cloud GPU deployment of Cosmos was not authorized (owner chose to hold). |
| Experiment protocol (06) | COMPLETE | Frozen before eval diagnostics; amendments are dated. |
| Experiment results (07) | COMPLETE | Includes negative results. |
| Original vs new (08) | COMPLETE | |
| Devpost text (09) | COMPLETE | Repository URL to be filled in. |
| Exploratory studies (11) | COMPLETE | Post-hoc waves 1–5 plus the low-data wave 4, each verified. Also e4/e4b (Cosmos world model), and four pre-registered audits with independent verification: r1 (replanning), c1 (coverage shift), s1 and s2 (self-training). |
| `results/hackathon/baseline_metrics.csv` | COMPLETE | 93 checkpoints |
| `results/hackathon/cosmos_diagnostics.csv` | COMPLETE | 660 rows; raw outputs in `results/hackathon/raw_model_outputs/` |
| `results/hackathon/experiment_manifest.json` / `execution_summary.json` | COMPLETE | `scripts/hackathon/finalize_results.py` |
| `media/hackathon/architecture.png` / `checkpoint_comparison.png` / `cosmos_example.png` / `robot_execution.png` / `cost_latency.png` | COMPLETE | generated from result files by `make_figures.py` |
| `media/hackathon/demo.mp4` | COMPLETE | |
| Unit tests | COMPLETE | `tests/hackathon/`: 38 pass in each environment; environment-specific tests (h5py / omegaconf / openai) skip where the package is absent |
| Live integration test on real infrastructure | COMPLETE | smoke test and the dashboard session both ran real Cosmos and Nebius calls |

## Known limitations (also in the docs)

* Simulation only (robomimic Can and Square PH). There are 3 short training runs, so the
  free epoch baseline is strong and no PLV metric beats it here.
* Visual diagnosis still needs rollouts. Failure-type classification by the VLMs is
  below a majority baseline.
* The prompts and exploratory analyses were chosen on small development sets; the
  exploratory analyses are post-hoc.
* The shared GPU caused OOM incidents (logged). The timings are upper bounds.
