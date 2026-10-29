# 06 — Experiment protocol (frozen before evaluation-set diagnostics were run)

Written: 2026-10-29 03:50 KST, before any evaluation-set (seed 200000+) episode
was shown to a vision-language model. Changes after this point are listed in the
"Amendments" section with a reason; nothing above it is edited afterwards.

## Data splits

| Split | Checkpoints | Env seeds | Used for |
|---|---|---|---|
| Training-time rollouts | every saved checkpoint | 100000–100049 (50 per ckpt) | target success `S_i` (original repo runner, unchanged) |
| Prompt development (dev) | can seed2 epoch 10, can seed2 epoch 30, square seed1 epoch 20 | 300000–300011, 300100–300111, 300200–300211 | choosing the prompt variant only |
| Evaluation (eval) | can seed1 and seed2: epochs 0, 20, …, 200; square seed1: epochs 0, 30, …, 300 (33 ckpts) | 200000–200009 (10 per ckpt) | Experiments 2 and 3 |

Dev checkpoints are not in the eval checkpoint list and dev seeds are disjoint
from eval seeds. Diffusion sampling is seeded per batch (`sampling_seed=0`).

## Ground truth

Verified labels come only from privileged simulator state
(`cosmos_plv/failure_annotator.py`): success = robosuite `_check_success()` at any
step; failure types `no_grasp`, `dropped`, `not_placed`, `other` from finger-pad
contacts and object height. VLM outputs never define labels.

## Prompt selection rule (decided before eval)

For each backend (local Cosmos3-Edge, Nebius Nemotron-3.5-Super-VL) the variant
with the highest **balanced success accuracy** on the 36 dev episodes is frozen
(ties: lower mean latency). For failure-type reporting the variant with the
highest failure-type accuracy on dev failures is frozen. All variants tried on
dev are reported, including the poor ones.

## Experiment 1 — baseline reproduction (all checkpoints)

Original scripts with paper flags (`scripts/hackathon/compute_baseline_metrics.sh`):
Action L1, PLV-1 smoothness (MLP ensemble f_psi, 5 x 256, H=10, 32 starts),
open-loop smoothness (H=10, policy seed 0), execution-matched replanning
(boundary jump, replan overlap; horizons 10 and 16; sampling seeds 0–3, median).
Oriented Spearman vs `S_i` and Regret@30 **within each run**, then the mean over
runs; 95% bootstrap CI over checkpoints within a run.

## Experiment 2 — visual failure assessment (eval split)

For each eval episode: frozen local Cosmos variant, frozen Nebius variant.
Report: inference count / failures / parse failures, latency, success accuracy
(Wilson 95% CI), confusion counts, failure-type accuracy on failed episodes,
per task.

## Experiment 3 — hybrid checkpoint screening (eval checkpoints)

Per checkpoint, scores computed WITHOUT the target `S_i`:
1. cheap offline metrics (Experiment 1);
2. `K=10` instrumented simulator success on eval seeds (an extra-rollout reference
   that uses privileged success detection);
3. `K=10` VLM-judged success rate (same rollouts, no privileged signal);
4. hybrid: keep the top 30% by a cheap metric, then pick by VLM-judged success;
5. hybrid rank-average of a cheap metric and VLM-judged success (no fitted weights).
No scoring rule is fitted on any outcome. Metrics: within-run Spearman vs `S_i`,
Regret@30, and Regret@1 of the final pick; compute cost per checkpoint.
With 11 checkpoints per run, results are descriptive; no superiority claim is
made unless the bootstrap CI excludes the cheap-only baseline.

## Experiment 4 — action-conditioned prediction (optional)

Originally blocked unless a faithful action adapter exists (see `cosmos_plv/action_adapter.py`). Later run as an exploratory study; see the 2026-10-29 afternoon amendment below and `07_experiment_results.md` §C3.

## Amendments and decisions (append-only)

* 2026-10-29 04:00 KST — clarification made after looking at DEV results only
  (no eval episode had been shown to any model): "balanced success accuracy on
  the 36 dev episodes" is computed over all 36 episodes with an unparsable output
  counted as a wrong prediction (`balanced_acc_all` in
  `scripts/hackathon/summarize_diagnostics.py`). Reason: otherwise a variant could
  look better by failing to answer.
* 2026-10-29 04:00 KST — 23 dev rows from the local backend failed with CUDA
  out-of-memory (GPU shared with training). They are infrastructure failures, not
  model outputs; they were moved to `dev_infra_failures.jsonl` and those episodes
  are re-run when the GPU is free.
* 2026-10-29 04:00 KST — DECISION (rule above), Nebius backend
  `nvidia/Nemotron-3.5-Super-VL`: `v2_evidence_first` (video, no reasoning),
  dev balanced_acc_all 0.871 vs `v6_perception_last` 0.743 (6 invalid JSON).
* 2026-10-29 04:10 KST — screening step for the LOCAL backend (decided before any
  local eval run): variants v1–v4 were only run on the first 12 dev episodes
  (dev_can_s2_e0010) and are screened out there (balanced_acc_all 0.50, 0.06,
  0.39, 0.56). The local decision is made among v5_endstate, v6_perception_last
  and v7_perception_track on all 36 dev episodes; v5 is completed when the GPU
  is free (it uses ~5.5 GB).
* 2026-10-29 04:30 KST — DECISION (rule above), local backend `nvidia/Cosmos3-Edge`,
  on all 36 dev episodes: success detection -> `v6_perception_last`
  (balanced_acc_all 0.824, tied with `v7_perception_track` 0.824 whose success
  decision uses the same last-frame question; tie broken by latency 1.8 s vs 17.0 s;
  `v5_endstate` 0.429). Failure-type reporting -> `v7_perception_track` (the only
  remaining local variant that outputs failure types; dev accuracy on failures 0.238).
  Both are run on the eval split. No eval-split diagnostic had been inspected.
* 2026-10-29 05:20 KST — GROUND-TRUTH FIX (found by the adversarial verifier of
  exploration x2, after eval diagnostics had been run): `failure_annotator` used the
  object height at step 0 as reference, but square nuts spawn ~6 cm above the table
  and settle within 10 steps, so settled nuts were labelled `other` ("fell") and
  lifts were under-detected. The reference is now the settled height (min over the
  first 10 steps) and the settle window is excluded from lift detection. All
  recorded episodes were re-annotated from their saved state traces
  (`scripts/hackathon/refresh_truth.py`; originals kept as `summary.orig.json` /
  `*.orig.jsonl`). Changes: 15 square `other` -> `no_grasp`, 2 square `no_grasp` ->
  `dropped`. Success labels (robosuite's own check) are unchanged, so no
  success-based number or prompt decision changes; failure-type accuracies are
  reported with the corrected labels.
* 2026-10-29 05:55 KST — ROBUSTNESS CHECK added after a verifier finding (x4):
  successful rollouts stop 16 steps after success while failures run to the 400-step
  timeout, so clip length alone separates success from failure (220/220 eval).
  The frozen Nebius prompt states the clip duration. The frozen v2 results stay as
  recorded; in addition `v2_evidence_first_pad201` re-runs the same prompt with every
  clip padded (final frame held) to the timeout length (201 source frames -> 101
  sampled frames) and the prompt stating the fixed duration (20.1 s). Both are
  reported. Local v6/v7 prompts contain no length or time information (each frame
  is judged alone), but the last frame of a successful clip is still taken shortly
  after success; this is reported as a limitation.
* 2026-10-29 (afternoon, KST): **Experiment 4 is run as an exploratory study, not under
  this protocol.**
  * E4 (385 generations through the `libero` action slot) is retracted, because that
    slot is untrained in Cosmos3-Edge@344d602b.
  * E4b (stages S0–S3) re-tests the trained Franka slot.
  * Criteria: S0–S2 were fixed before running. The S3 criteria were written about 20 s
    after launch and before any output was read.
  * Results and caveats: `07_experiment_results.md` §C3. E4b is not used for any
    ranking claim.
* 2026-10-29 (afternoon, KST): **study r1 (counterfactual replanning)** is pre-registered
  separately in `docs/hackathon/explore/r1_counterfactual_replan_protocol.md`, with its
  own dated amendments.
