# 07 — Experiment results

Results are grouped by kind and the groups are not mixed:

* (A) historical paper results, quoted, not re-run;
* (B) Experiment 1, a reproduced baseline on a new checkpoint pool;
* (C) Experiments 2–3, new hackathon experiments that follow the protocol in
  `06_experiment_protocol.md`;
* (D) exploratory post-hoc studies, summarised in `11_explorations.md`;
* (E) failed or blocked items.

Every number below is read from a file in `results/hackathon/`.

## A. Historical (paper, frozen 2026-08-25; not reproduced here)

These come from the primary Diffusion Policy cohort of the paper's 1,365-checkpoint
study (45 runs, 50 rollouts per checkpoint):
* PLV-1 smoothness: Spearman 0.556, Regret@30 0.006.
* Action L1: Spearman 0.462, Regret@30 0.017.
* Pooled open-loop smoothness: Regret@30 0.019.
* A larger dynamics model lowered prediction MSE but did not improve
  execution-matched ranking.

## B. Experiment 1 — baseline reproduction on the retrained pool

* Pool: 93 checkpoints in 3 runs (see `03_model_recovery.md`).
* Target: 50-rollout training-time success.
* Metrics: computed by the original scripts with the paper flags.
* Statistics: oriented Spearman within each run (95% bootstrap CI over
  checkpoints), then the mean over runs.
* Data: `results/hackathon/baseline_ranking.csv`, `baseline_metrics.csv`.

| metric | can s1 (41) | can s2 (21) | square s1 (31) | mean ρ | mean Regret@30 |
|---|---|---|---|---|---|
| Training epoch (free baseline) | 0.78 [0.55, 0.91] | 0.88 [0.65, 0.98] | 0.79 [0.52, 0.93] | **0.815** | 0.007 |
| Replan overlap (model feedback, L3) | 0.78 [0.55, 0.91] | 0.86 [0.58, 0.97] | 0.76 [0.50, 0.92] | 0.801 | 0.007 |
| Open-loop smoothness (L2) | 0.76 [0.54, 0.87] | 0.85 [0.60, 0.96] | 0.75 [0.49, 0.89] | 0.788 | 0.007 |
| Boundary jump (L3) | 0.77 [0.52, 0.91] | 0.85 [0.56, 0.97] | 0.73 [0.42, 0.90] | 0.782 | 0.007 |
| Action L1 (L1) | 0.75 [0.52, 0.88] | 0.83 [0.57, 0.95] | 0.73 [0.42, 0.91] | 0.772 | 0.007 |
| PLV-1 smoothness (L2) | 0.73 [0.50, 0.87] | 0.76 [0.45, 0.90] | 0.79 [0.52, 0.93] | 0.759 | 0.007 |
| Validation loss | −0.46 [−0.69, −0.14] | −0.48 [−0.83, 0.05] | −0.51 [−0.80, −0.12] | −0.481 | 0.047 |

Late half of each run (epoch ≥ run median, 48 checkpoints; subset chosen
without looking at outcomes): epoch 0.33, Action L1 0.29, open-loop 0.26,
replan overlap 0.24, PLV-1 0.20, boundary jump 0.12, validation loss −0.33. Most
per-run CIs include 0.

What reproduces:
* All PLV-family metrics rank checkpoints well (ρ 0.76–0.80).
* Each keeps a near-best checkpoint in its top 30% (Regret@30 0.007).
* Validation loss is **anti**-correlated with closed-loop success (it rises while
  success rises), which is the paper's motivating observation.

What does not:
* No PLV metric beats the free "later epoch is better" baseline in this pool.
* Replanning disagreement in particular falls almost monotonically with training
  (ρ with epoch 0.96–1.00), so here it mostly measures training progress.
* Our runs are short (200–300 epochs) and nearly monotone. The paper's 1,000-epoch
  runs had the plateau and overfitting regime where epoch order fails.
* H=10 and H=16 replanning give identical values. This is structural, not a bug:
  with an execution stride of 8, both horizons contain exactly one replan boundary.

## C1. Experiment 2 — can an NVIDIA VLM tell success from failure in a real rollout video?

* Episodes: 330 eval episodes from 33 checkpoints × 10 seeds (200000–200009);
  263 succeeded and 67 failed.
* Ground truth: privileged simulator state.
* Prompt variants: frozen on a disjoint dev split (36 episodes, 3 other checkpoints).
* Data: `results/hackathon/experiment2_backend_comparison*.json` and
  `cosmos_diagnostics.csv`.

| judge | accuracy [95% CI] | balanced acc. | false pos. (of 67 failures) | false neg. (of 263 successes) | Can acc. | Square acc. | latency / episode |
|---|---|---|---|---|---|---|---|
| **Cosmos3-Edge** (local, last-frame perception v6) | 0.918 [0.884, 0.943] | 0.832 | 21 | 6 | 0.964 | 0.827 | 2.1 s |
| **Nemotron-3.5-Super-VL** (Nebius, video v2) | 0.964 [0.938, 0.979] | 0.927 | 9 | 3 | 0.977 | 0.936 | 3.1 s |
| Nemotron, length-neutral clips (robustness) | 0.961 [0.934, 0.977] | 0.909 | 12 | 1 | 0.973 | 0.936 | 3.6 s |

* **Agreement:** the two judges agree on 93.6% of episodes; when they agree they
  are right 97.1% of the time. When they disagree, Nemotron is right in 18 of 21.
* **Failure types are not solved:**
  * Accuracy on failed episodes is 0.31 for Nemotron, 0.36 with length-neutral
    clips, and 0.21 for the Cosmos 8-keyframe track.
  * All of these are below a constant "no_grasp" guess.
  * Nemotron calls most missed grasps and drops "not_placed".
* **Error analysis** (`experiment2_error_analysis.json`):
  * Cans dropped into the right bin's **wrong compartment** were called successes
    by both judges.
  * Square clips that end 0.8 s after success sometimes show a nut still sliding.
* **Clip length:** successful clips stop early, so clip length alone separates the
  outcomes. Re-running Nemotron with every clip padded to the same length changed
  4 of 250 judgements (first 250 episodes), so it does not rely on this cue.
* **Dev → eval:** both judges scored higher on eval than on dev (dev: local 0.806,
  Nemotron 0.861). Eval has more successes.

Scope:
* Both judges were used **zero-shot**, with no fine-tuning.
* Tested on one simulator, one robot (Franka Panda, common in robot datasets) and two tasks.
* Transfer to other embodiments or to real video was not tested.

Prompt development (dev only; full table in `11_explorations.md`):
* Giving Cosmos3-Edge the task goal plus the video gave "success" for 12/12 dev
  episodes (3 true). Its reasoning read like a robot narrating its plan.
* Asking only "where is the object?" fixed most of this.

## C2. Experiment 3 — hybrid screening

* Checkpoints: 33 eval checkpoints (11 per run).
* Target: 50-rollout success. No score is fitted on outcomes.
* Data: `results/hackathon/hybrid_ranking.csv`, `hybrid_screening.csv`.

| score | mean within-run ρ | Regret@30 | regret of the single pick (mean of 3 runs) |
|---|---|---|---|
| Training epoch (free) | 0.854 | 0.007 | 0.007 |
| Replan overlap, real sim feedback (needs 10 rollouts) | 0.854 | 0.007 | 0.007 |
| Replan overlap, model feedback | 0.845 | 0.007 | 0.013 |
| Open-loop smoothness | 0.829 | 0.007 | 0.020 |
| PLV-1 smoothness | 0.773 | 0.007 | 0.007 |
| Action L1 | 0.731 | 0.007 | 0.013 |
| Rank-average(Action L1, Nemotron-judged) | 0.760 | 0.007 | 0.013 |
| Nemotron-judged success, 10 rollouts | 0.743 | 0.013 | 0.060 |
| Cosmos-judged success, 10 rollouts | 0.651 | 0.020 | 0.080 |
| Instrumented sim success, 10 rollouts (privileged) | 0.607 | 0.020 | 0.087 |
| Two-stage: top-30% Action L1 → VLM-judged | — | 0.007 | 0.040 |

**Result (negative):**
* In this pool, scores built from 10 extra rollouts rank checkpoints *worse* than
  the free cheap scores. This holds whether the success signal is privileged or
  VLM-judged. Among near-equal checkpoints, 10 rollouts are too noisy.
* The two-stage rule (cheap screen first, then VLM-judged rollouts) does not beat
  the cheap metric alone.
* The VLM-judged K=10 scores are not worse than the instrumented K=10 score. So a
  VLM can stand in for the simulator's success detector without losing ranking
  quality. That is the property needed where no detector exists, such as on real
  robots.

## C3. Experiment 4 — Cosmos3-Edge action-conditioned forward dynamics (optional, exploratory)

Data: `results/hackathon/explore/e4_cosmos_fd.json`, `e5_domain_weights.json`,
`e5_probe.json`, `e5_probe_s3_units.csv`. Code: `scripts/hackathon/explore/e4_*.py`, `e5_*.py`.
Independent verification ran four lenses: statistics, measurement, alternative
explanations, and reproduction.

**E4 is retracted as a test of action conditioning.**
* E4 sent policy actions through Cosmos3-Edge's `libero` action slot (385 generations,
  7 conditions). No condition came closer to the real future than any other.
* A CPU read of the checkpoint (`e5_domain_weights.py`) shows that this slot is
  **untrained**: zero biases, and weights inside the initialization bound.
* Only slots 1, 2, 3, 6, 7, 8, 12, 13, 15 and 20 are trained. These match the model
  card's embodiment list.
* diffusers refuses `libero`, and E4 bypassed that guard. So E4 tested a random
  projection.
* The LIBERO adapter itself matches the Cosmos framework's transform to 3e-8.

**E4b re-tests the trained Franka slot (domain 8, RoboMIND encoding).**

| stage | what | result |
|---|---|---|
| S0 positive control | model-card UMI example | reproduces the card's output (30.8 dB vs 26–27 dB between seeds); an action change moves pixels 12.6–16.7 vs 5.9 for a seed change (one scene) |
| S1 axis sweep | ±x/y/z commands on 4 robomimic frames | each axis gives a sign-dependent gripper motion of 14–16 px in a consistent image direction (mean cos 0.87–0.94 over 12 probes, 3 trackers; next-best proper rotation 0.43–0.53). A reflection ties, so the 3-D frame is not identified. Unnormalised DROID encoding ≈ 0 px; untrained slots ≈ 2 px |
| S3 recorded rollouts | actions from 22 recorded policy-rollout clips (11 episodes × starts 16/48; 5 initial-state families) | pre-registered test passes: on 15 moving clips, the drift-cancelled motion (main − time-reversed) points along the true gripper motion, cos 0.78 / 0.79 / 0.85, against 0.10–0.19 for mismatched clips |

What weakens S3:
* **Family-level evidence is limited.** The mean of the 5 family means is 0.67–0.84,
  with t(4) lower bounds 0.15 / 0.41 / 0.63. The exact sign-flip p is 0.031–0.062,
  and 1/32 is the smallest p possible with 5 families.
* **Can carries the effect** (cos 0.91–0.95; end-point error 16–18 px vs 44 px with
  zero actions).
* **Square is weak** (cos 0.47–0.63), and its end-point error is *worse* than with zero
  actions (45–47 vs 39–40 px).
* **The motion is too small.** Generated motion is about 0.4x the true motion and
  partly ignores the action sign.
* **The action effect is close to seed noise.** At matched starts it is about 2x the
  seed floor (17–20 vs 9–10 px).
* **No pixel gain.** Full-frame PSNR is no better than repeating the first frame
  (19.9 vs 20.4 dB; +0.7 dB at start 16, −1.8 dB at start 48).
* **Timing of the criteria.** The S3 criteria were written about 20 s after launch,
  before any output was read.

**Meaning:** zero-shot, Cosmos3-Edge carries a coarse directional action signal for this
robot. It is not a faithful rollout, and it cannot yet replace rollouts as a PLV stage.

## D. Exploratory follow-ups (post-hoc; all but the deterministic cascade x0 adversarially verified)

Summary in `11_explorations.md`:
* **Adaptive allocation:** cuts rollouts about 2–3x versus uniform allocation.
* **Early abort:** cuts robot time by 16–20%.
* **Local-first cascade:** 16% fewer Nebius calls (84% of calls) at accuracy 0.952 [0.923, 0.970]
  vs 0.964 for Nemotron alone; false negatives rise from 3 to 8 (`explore/x0_cascade.json`, n=330).
* **Camera choice:** matters more than image quality.
* **Grounded LLM reports:** not reliable (5/16 contained a false statement).
* **Noise ceiling:** on the plateau, two halves of 25 rollouts do not agree on a
  ranking (split-half ρ 0.02). Plateau ranking cannot be evaluated with 50 rollouts.
* **PLV priors:** they never beat the epoch prior (0/336 comparisons).

## E. Failed or blocked

* **E4 (libero action slot):** invalid as a test, because the slot is untrained
  (section C3). The trained-slot re-test (E4b) is exploratory and not used for ranking.
* **Cosmos3-Edge reasoning mode:** often produced no parsable JSON (dev: 8/12 for
  v2 with reasoning; exploration x5: 9/25).
* **Visual plan critic (x6):** no ranking value.
* **Prompt variants for failure types (x4, x5):** no significant improvement.
* **Infrastructure:**
  * 23 dev rows lost to GPU out-of-memory; re-run and recorded separately.
  * One rollout recorder leaked EGL memory and was replaced by one process per
    checkpoint.
  * One square-label bug was fixed (settled height, `06_experiment_protocol.md`).

## Seeds and counts

| item | value |
|---|---|
| Training seeds | can 1, can 2, square 1 |
| Training-time rollout seeds | 100000–100049 (50 per checkpoint, 4,650 total) |
| Eval rollout seeds | 200000–200009 (10 per eval checkpoint, 330 total) |
| Dev seeds | 300000–300011, 300100–300111, 300200–300211 (36) |
| Diffusion sampling | recorder: `0 + 7919 × batch_index`; metric scripts: policy seed 0, sampling seeds 0–3 |
| f_psi | MLP ensemble 5 × 256, 8 epochs, ≤ 30k transitions, seed 0 |
| VLM calls on eval | Cosmos3-Edge 330 (v6) + 2,640 (v7, 8 per episode); Nemotron 660 (v2 + padded) |
