# Reproducibility report (2026-10-30, about 00:30 KST)

Machine: Ubuntu 20.04, 1x RTX 3090 (24 GB), Environments A, B, C and TTS as in the README.

## PASS

| check | result | how |
|---|---|---|
| Unit tests, Environment A | 59 passed, 1 skipped | `source scripts/hackathon/env.sh && python -m pytest -q tests/hackathon` |
| Unit tests, Environment B | 59 passed, 2 skipped | `source scripts/hackathon/envb.sh && python -m pytest -q tests/hackathon` |
| End-to-end smoke test | **SMOKE PASS**, every stage | `bash scripts/hackathon/smoke_test.sh`, which runs:<br>• unit tests A and B<br>• a 1-step Diffusion Policy training run (25 s)<br>• one recorded robosuite rollout (21 s)<br>• one local Cosmos3-Edge call (15 s)<br>• one real Nebius call (11 s; request `chatcmpl-bf8ab9cab75f85bc`, parse ok)<br>• the model-output validity check |
| Dashboard startup, live mode | no exceptions; all three live backends detected | `streamlit.testing.v1.AppTest` on `dashboard/app.py` |
| Dashboard, recorded replay mode, fresh public export | no exceptions; replay banner shown; all 4 live buttons disabled; recorded Cosmos and Nemotron answers and judge-trust warning rendered | `COSMOS_PLV_REPLAY=1`, run in a fresh export with no `data/` directory |
| Real simulator playback | the recorded rollout videos play in the dashboard; 5 sample checkpoints ship in `media/hackathon/episodes/` | dashboard and video frames |
| Real Cosmos response | live call in the recorded session (model id `nvidia/Cosmos3-Edge@344d602b`) and in the smoke test | `submission/screenshots/04_cosmos3_edge_live.png` |
| Real Nebius response | live call in the recorded session (request id on screen) and in the smoke test | `submission/screenshots/05_nemotron_nebius_live.png`; `results/hackathon/nebius_calls.jsonl` |
| Claim ledger | 48 claims; 0 mismatches against the result files on recompute | `python scripts/hackathon/build_claim_ledger.py` → `submission/claim_ledger.csv` |
| Figures regenerate from result files | yes | `python scripts/hackathon/make_submission_figures.py` (submission figures); `python scripts/hackathon/make_figures.py` (media figures) |
| Final video | 165.2 s, 1920x1080, 30 fps, H.264, AAC, English subtitles | `submission/video_validation.md` |
| Public export validation | compiles, parses, JSON loads; sensitive-pattern scan clean; 0 files with Hangul | `scripts/hackathon/export_public_repo.sh` |

## SKIPPED (environment or time)

* **Retraining the 93-checkpoint pool.** This takes hours of GPU. The training script and its config are in the repo; the 1-step training smoke test passes.
* **Re-running every experiment end to end.** These are 2,000 to 5,000 rollouts per study. Each study's analysis script re-runs on the stored results.
* **A real robot.** This is a simulation-only project.

## NOT REPRODUCIBLE FROM PUBLIC ARTIFACTS ALONE

* **Live rollouts and live cheap metrics in the dashboard.** These need the checkpoint pool, which is not redistributed (several GB). Recorded replay mode covers the judge-facing flow.
* **The full rollout-video archive (330 eval and 620 c1 episodes) and the s1 / s2 branch data.** Only a few sample episodes ship. The per-episode results are in `results/hackathon/` (JSON, CSV, JSONL).
* **Pre-registration timing evidence (commit timestamps).** It lives in the private development history. The public repository was exported as fresh commits. The protocols state the timestamps and commit ids.
* **Live Nebius calls without a key.** A user needs their own `NEBIUS_API_KEY`. The call evidence (request ids, tokens, latency) is in the logs.
