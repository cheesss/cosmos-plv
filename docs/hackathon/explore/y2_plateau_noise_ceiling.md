# y2_plateau_noise_ceiling: can anything rank plateau checkpoints, and where is the noise ceiling? (EXPLORATORY, post-hoc)

> **Label:** exploratory and post-hoc. Nothing here was pre-registered in `06_experiment_protocol.md`.
> The "plateau" cut (can epoch >= 60, square epoch >= 90) was **chosen by looking at the learning
> curves**, so it is post hoc. The other late subset (epoch >= run median) is outcome-free.
> No score is fitted: the held-out half (seeds 100025-100049) is used only for evaluation, the
> "more rollouts" reference uses only the other half (100000-100024), and the rank-average is
> unweighted. Every number below comes from `results/hackathon/explore/y2_plateau_noise_ceiling.json`
> (main) and `results/hackathon/explore/y2_plateau_noise_ceiling_trend.json` (trend decomposition).
> CPU only. No GPU work and no Nebius calls.

**One-line summary:** On the full pool (93), epoch (0.815 on the Exp-1 basis) and the top PLV metrics (0.78–0.80) already touch the noise ceiling of the 50-rollout target (about 0.84, from split-half). So there is no room for any metric to beat epoch significantly. In the plateau, even two independent 25-rollout halves do not reproduce each other's ranking (within-run Spearman +0.02 [−0.16, +0.21]). The only real signal left is a gentle upward trend with epoch (2–4 %p per 100 epochs), and epoch itself captures it at almost the ceiling (+0.35 against the held-out 25, trend-model ceiling +0.34). Once epoch is controlled for, the PLV metrics, K=10 instrumented success and K=10 VLM-judged success all add no information (partial Spearman ≈ 0, rank-average Δ ≤ 0). No per-checkpoint deviation outside the epoch trend is detected (F_dev p = 0.46 / 0.21 / 0.99), and the partial Spearman ceiling such deviations could provide is about 0.14 on average. In the plateau, Regret@30 is about 0–2 %p for every signal, random choice included, so within-plateau selection barely affects the outcome in this pool.

## 1. Question

Near the best checkpoints, where checkpoints sit on a plateau, is **any** signal able to rank
checkpoints? What ceiling does the measurement noise of the success target itself impose?
Experiment 1 found that every metric degrades from Spearman ~0.8 on the full pool to 0.2–0.33 on the
late half. This study asks whether that drop reflects weak metrics or a target with nothing left
to rank.

## 2. Data

* 93 checkpoints in 3 runs (can_seed1: 41, can_seed2: 21, square_seed1: 31). Each has 50
  training-time rollouts `test/sim_max_reward_<seed>`, seeds 100000-100049, re-read from each
  run's `logs.json.txt`. The per-checkpoint mean matches `baseline_metrics.csv` `success` for all 93.
* **Target B** = mean of seeds 100025-100049 (25 episodes). **Half A** = seeds 100000-100024, the
  "more rollouts" reference (K=25, plus random 10-subsets of A for K=10).
* Cheap metrics from `results/hackathon/baseline_metrics.csv` (oriented so higher is better).
  `boundary_jump_h10 == boundary_jump_h16`, `replan_overlap_h10 == replan_overlap_h16` and
  `val_loss_logged == val_loss` are bit-identical in that file, so only the h16 / val_loss copies
  are used. `train_action_mse` comes from the logs, with 20 missing values linearly interpolated
  over epoch (outcome-free).
* Eval checkpoints only (33 = can every 20 epochs, square every 30): K=10 instrumented success
  (`hackathon_rollouts_eval/<ckpt>/summary.json`, seeds 200000-200009, disjoint from the target).
  Also K=10 VLM-judged success from Cosmos3-Edge local v6, local v7, Nemotron Nebius v2 and v2_pad201
  (all 330 rows scored and parsed). Also the K=10 executed-rollout PLV metrics `sim_replan_overlap`,
  `sim_boundary_jump` and `executed_smoothness` from `results/hackathon/hybrid_screening.csv`
  (Experiment 3, sha256 prefix f87ba3bd83d86f9c at analysis time).

Subsets (checkpoints per run, can_s1 / can_s2 / square):

| subset | rule | n | eval-ckpt n |
|---|---|---|---|
| all | all checkpoints | 41/21/31 = 93 | 11/11/11 = 33 |
| late_median | epoch >= run median (100/100/150), outcome-free | 21/11/16 = 48 | 6/6/6 = 18 |
| plateau_posthoc | can >= 60, square >= 90, **post hoc from the learning curves** | 29/15/22 = 66 | 8/8/8 = 24 |

## 3. Method

**Statistics.** Each statistic is computed within a run and then averaged over runs without
weights, as in the paper. Spearman uses average ranks. A constant predictor gets Spearman 0 (it
carries no ranking information) and the regret of a random pick. Regret@30 = max_B − max B over the
top ceil(0.3 n) checkpoints by score. **Ties are broken uniformly at random (exact expectation)**, not
by checkpoint path as in `cosmos_plv/ranking.py`, because many plateau signals are tied (for
example 25/25 successes). Path tie-breaking would systematically favour early epochs.

**(1) Noise ceiling.**
* Split-half reliability: within-run Spearman(S_A, S_B) on the fixed split and averaged over 400
  random 25/25 seed splits, with the 2.5–97.5 % range across splits.
* Spearman-Brown reliability for 50 rollouts: rel50 = 2r/(1+r).
* Implied ceiling for any predictor of the true success rate (index of reliability): sqrt(r) when
  the target is a 25-rollout half, sqrt(rel50) when it is the 50-rollout S. Negative r gives a
  ceiling of 0.
* Two-way (checkpoint × shared seed) ANOVA reliability ICC(C,k). Seeds are shared across
  checkpoints, so seed difficulty is a common effect, not between-checkpoint noise. The naive
  independent-binomial estimate is kept in the JSON for reference only.
* Seed-stratified permutation tests (2000 permutations; checkpoints permuted within each seed
  column): between-checkpoint heterogeneity (variance of S50), and the epoch trend (Spearman of
  epoch with S50, one-sided).
* Trend decomposition (supplementary script): the checkpoint factor is split into a linear epoch
  trend (1 df) and checkpoint-specific deviations (n−2 df), both tested against the interaction
  mean square. The deviation reliability for a 25-rollout half gives the **ceiling of any partial
  Spearman beyond epoch**, sqrt(rel_dev25). A Gaussian simulation gives the expected Spearman of a
  perfect trend follower (≈ epoch) against a fresh 25-rollout half.

**(2) Signals vs held-out B.** Spearman and Regret@30 for every signal listed in §2. For context,
outcome-free signals are also scored against S50.

**(3) Incremental value beyond epoch.**
* Partial Spearman of signal and B given epoch, within run: rank-transform, then
  (r_xy − r_xz r_yz)/sqrt((1−r_xz²)(1−r_yz²)).
* Δ = Spearman/Regret of rank-average(epoch, signal) minus that of epoch alone.

**Uncertainty (4 kinds, all in the JSON):**
* **block**: circular moving-block bootstrap over epoch-ordered checkpoints within each run,
  block length ceil(n^(1/3)) with a minimum of 2, 600 replicates. The target is fixed.
* **hier**: the same block bootstrap combined with resampling of the 25 B seeds, the 25 A seeds
  and the 10 eval seeds (jointly across runs). This counts target noise twice (once in the sample,
  once in the resample), so it is conservative and pulls level statistics toward 0. For example,
  epoch on `all` has point 0.83 and hier CI [0.48, 0.86]. Use it for Δs, and use the random-split
  mean for levels.
* **splits**: 400 random 25/25 re-splits with checkpoints fixed. This is the target-noise interval,
  and its mean is the fairest point estimate, because the fixed split turns out to be "lucky" for
  epoch (see §5.2).
* **run-level**: exact bootstrap over the 3 runs (27 resamples). With 3 runs this mostly shows the
  spread across runs.

## 4. Commands and exit codes

```
# smoke tests (scratch outputs only), exit 0
PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/y2_plateau_noise_ceiling.py \
  --boot 6 --splits 6 --perm 20 --asub 3 --out-json data/outputs/hackathon_explore/y2_plateau_noise_ceiling/smoke2.json \
  --out-csv data/outputs/hackathon_explore/y2_plateau_noise_ceiling/smoke2.csv                      # exit 0
# main analysis (218 s, 6 worker processes), exit 0
PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/y2_plateau_noise_ceiling.py \
  --boot 600 --splits 400 --perm 2000 --asub 100 --workers 6 \
  --out-json data/outputs/hackathon_explore/y2_plateau_noise_ceiling/final.json \
  --out-csv data/outputs/hackathon_explore/y2_plateau_noise_ceiling/final_table.csv                  # exit 0
#   -> copied verbatim to results/hackathon/explore/y2_plateau_noise_ceiling.json / _table.csv
#      (the default --out-json/--out-csv of the script are exactly those paths)
# trend decomposition, exit 0
PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/y2_plateau_noise_ceiling_trend.py
# markdown tables from the JSON, exit 0
PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/y2_plateau_noise_ceiling_tables.py
```

An earlier full run of the same script used a non-circular block bootstrap and a single process.
It was superseded: non-circular blocks under-sample the epoch-0 and last checkpoints. That run was
left to finish rather than killed, and its log is kept as
`data/outputs/hackathon_explore/y2_plateau_noise_ceiling/run_superseded_noncircular.log`. The
canonical JSON is the circular-block run.

## 5. Results

### 5.1 Noise ceiling of the success target

Mean over runs. Per-run values are in the JSON and in the per-run table produced by the tables
script.

| subset | n | split-half rho, fixed split | split-half rho, random splits mean [2.5, 97.5] | SB rel50 | ceiling vs 25-half | ceiling vs S50 | ANOVA rel25 / rel50 |
|---|---|---|---|---|---|---|---|
| all | 93 | +0.61 | **+0.59** [+0.51, +0.68] | +0.72 | **0.76** | **0.84** | 0.94 / 0.97 |
| late_median | 48 | −0.18 | **−0.17** [−0.38, +0.05] | −0.66 | **0.16** | **0.21** | −0.23 / −0.83 |
| plateau_posthoc | 66 | −0.02 | **+0.02** [−0.16, +0.21] | −0.14 | **0.22** | **0.26** | +0.02 / −0.15 |
| all (eval ckpts) | 33 | +0.70 | +0.67 [+0.52, +0.81] | +0.80 | 0.81 | 0.89 | 0.96 / 0.98 |
| late_median (eval) | 18 | −0.01 | −0.17 [−0.58, +0.32] | −0.76 | 0.19 | 0.23 | −0.26 / −1.12 |
| plateau_posthoc (eval) | 24 | +0.04 | +0.07 [−0.27, +0.43] | −0.07 | 0.24 | 0.28 | 0.00 / −0.20 |

Per run, plateau_posthoc:

| run | n | S50 range | split-half rho, random splits | ANOVA F p (heterogeneity) | perm. heterogeneity p | epoch trend rho(S50), perm. p |
|---|---|---|---|---|---|---|
| can_seed1 | 29 | 0.88–1.00 | −0.07 [−0.33, +0.22] | 0.290 | 0.344 | +0.36, p = 0.029 |
| can_seed2 | 15 | 0.90–1.00 | +0.45 [+0.05, +0.77] | 0.008 | 0.018 | +0.66, p = 0.004 |
| square_seed1 | 22 | 0.76–0.90 | −0.30 [−0.57, +0.01] | 0.978 | 0.984 | +0.42, p = 0.029 |

Trend decomposition (two-way ANOVA; the linear trend is meaningful only for the late and plateau
subsets):

| subset / run | trend slope pp/100 ep [block CI] | F_trend p | F_dev (p) | trend SD | noise SD (25-half, shared seeds) | ceiling of partial rho beyond epoch | expected rho of perfect trend follower vs 25-half |
|---|---|---|---|---|---|---|---|
| plateau can_seed1 | +2.1 [−0.2, +3.7] | 0.033 | 1.01 (0.46) | 0.9 pp | 3.1 pp | 0.05 | +0.27 |
| plateau can_seed2 | +4.4 [+0.5, +7.7] | 0.0003 | 1.29 (0.21) | 1.9 pp | 2.9 pp | 0.36 | +0.56 |
| plateau square_seed1 | +2.3 [−1.3, +4.8] | 0.18 | 0.41 (0.99) | 1.4 pp | 7.0 pp | 0.00 | +0.20 |
| **plateau mean** | | | | | | **0.14** | **+0.34** |
| late can_seed1 | −0.3 [−3.3, +1.1] | 0.80 | 0.50 (0.97) | 0.1 pp | 2.4 pp | 0.00 | +0.04 |
| late can_seed2 | +2.0 [−2.8, +7.1] | 0.15 | 1.19 (0.30) | 0.6 pp | 2.0 pp | 0.30 | +0.28 |
| late square_seed1 | +3.7 [+1.1, +6.5] | 0.16 | 0.27 (0.997) | 1.7 pp | 6.7 pp | 0.00 | +0.25 |
| **late mean** | | | | | | **0.10** | **+0.19** |

### 5.2 Signals vs held-out half B (all checkpoints of the subset)

"splits" is the mean (and 2.5–97.5 %) over 400 random 25/25 re-splits; "fixed" uses the protocol
split. Regret@30 is in success-rate units (0.01 = 1 pp).

| subset | signal | Spearman fixed | block CI | splits mean [range] | Regret@30 fixed | Regret@30 splits mean |
|---|---|---|---|---|---|---|
| plateau | epoch | +0.50 | [+0.23, +0.66] | **+0.35** [+0.05, +0.61] | 0.000 | 0.007 |
| plateau | val_action_l1 | +0.51 | [+0.29, +0.65] | +0.30 [+0.02, +0.54] | 0.000 | 0.007 |
| plateau | open_loop_smoothness | +0.49 | [+0.22, +0.62] | +0.33 [+0.02, +0.61] | 0.000 | 0.007 |
| plateau | replan_overlap_h16 | +0.48 | [+0.21, +0.63] | +0.32 [+0.03, +0.60] | 0.000 | 0.009 |
| plateau | val_loss | −0.51 | | −0.36 | 0.013 | |
| plateau | other half, K=25 rollouts | −0.02 | [−0.27, +0.14] | **+0.02** [−0.18, +0.22] | 0.013 | 0.018 |
| plateau | random | 0 | | 0 | 0.002 | 0.010 |
| late_median | epoch | +0.37 | [+0.06, +0.56] | **+0.22** [−0.11, +0.56] | 0.000 | 0.005 |
| late_median | val_action_l1 | +0.43 | [+0.20, +0.66] | +0.19 [−0.10, +0.52] | 0.000 | 0.006 |
| late_median | open_loop_smoothness | +0.33 | [+0.01, +0.51] | +0.17 [−0.13, +0.50] | 0.000 | 0.008 |
| late_median | replan_overlap_h16 | +0.32 | [−0.04, +0.52] | +0.15 [−0.17, +0.50] | 0.000 | 0.006 |
| late_median | other half, K=25 rollouts | −0.18 | [−0.64, +0.03] | **−0.18** [−0.41, +0.03] | 0.009 | 0.021 |
| late_median | random | 0 | | 0 | 0.002 | 0.010 |
| all | epoch | +0.83 | [+0.61, +0.90] | +0.74 [+0.61, +0.85] | 0.000 | 0.006 |
| all | replan_overlap_h16 | +0.82 | [+0.61, +0.89] | +0.73 [+0.61, +0.84] | 0.000 | 0.006 |
| all | other half, K=25 rollouts | +0.61 | [+0.25, +0.78] | +0.59 [+0.50, +0.67] | 0.000 | 0.012 |

The full per-signal tables (12 cheap metrics, hier and run-level CIs, Spearman vs S50) are in
`results/hackathon/explore/y2_plateau_noise_ceiling_table.csv` and are printed by the tables script.
On the plateau, every epoch-like metric (train_loss, train_action_mse, val_action_l1, the PLV
smoothness and replan metrics, dynamics_uncertainty) lands at +0.28 to +0.35 on the split mean.
None is above epoch. val_loss is negative (it rises with epoch through overfitting).
support_drift and teacher_forced_latent_mse are about 0.

### 5.3 Eval checkpoints only: K=10 instrumented and VLM-judged rollouts

plateau_posthoc (24 checkpoints, 8 per run):

| signal | Spearman fixed | block CI | splits mean [range] | partial rho given epoch [block CI] | Δ rank-avg(epoch, ·) − epoch [hier CI] |
|---|---|---|---|---|---|
| epoch | +0.57 | [+0.12, +0.79] | **+0.45** [0.00, +0.78] | n/a | n/a |
| open_loop_smoothness | +0.55 | [+0.12, +0.76] | +0.44 | −0.21 [−0.66, −0.04] | −0.01 [−0.09, +0.04] |
| replan_overlap_h16 | +0.50 | [−0.04, +0.74] | +0.44 | −0.23 [−0.79, +0.14] | −0.03 [−0.18, +0.05] |
| sim_replan_overlap_K10 (Exp-3) | +0.57 | | +0.45 | +0.04 [0.00, +0.41] | 0.00 [0.00, +0.02] |
| executed_smoothness_K10 (Exp-3) | +0.64 | [+0.20, +0.86] | **+0.12** [−0.21, +0.41] | +0.45 [−0.11, +0.92] | +0.13 [−0.23, +0.48] |
| eval_sim_success_K10 (instrumented) | +0.12 | [−0.39, +0.55] | +0.19 [−0.27, +0.55] | −0.24 [−0.75, +0.30] | −0.14 [−0.45, +0.23] |
| VLM Cosmos local v6, K=10 | +0.07 | [−0.24, +0.30] | −0.01 [−0.22, +0.20] | −0.05 [−0.33, +0.13] | −0.06 [−0.47, +0.23] |
| VLM Cosmos local v7, K=10 | +0.07 | | −0.01 | −0.05 | −0.06 |
| VLM Nemotron Nebius v2, K=10 | −0.01 | [−0.41, +0.32] | +0.20 [−0.06, +0.41] | −0.28 [−0.56, +0.03] | −0.05 [−0.46, +0.15] |
| VLM Nemotron v2_pad201, K=10 | −0.02 | | +0.25 | −0.27 | −0.08 |
| other half, K=25 rollouts | +0.04 | [−0.42, +0.40] | +0.07 [−0.26, +0.39] | −0.16 [−0.56, +0.32] | −0.21 [−0.70, +0.33] |
| other half, K=10 rollouts | −0.04 | | +0.06 | −0.22 | −0.16 |

late_median (18 eval checkpoints, 6 per run) shows the same picture. Epoch is +0.47 fixed and +0.31
on the split mean. K=10 instrumented success is −0.19 fixed and −0.11 split mean. The VLMs give
v6 −0.09 / −0.04, Nemotron v2 −0.24 / −0.13 and pad201 −0.09 / −0.05. The other half (K=25) gives
−0.01 / −0.17. All Regret@30 values are ≤ 0.013 fixed and ≤ 0.023 on the split mean, and random
picking gives 0.006 / 0.014.

On all 33 eval checkpoints, the VLM-judged K=10 success follows instrumented K=10 success closely:
v6 +0.70, Nemotron +0.74, instrumented +0.59, other-half K=10 +0.65, all on the fixed split. This
matches the earlier finding that success detection works. All of these stay below epoch (+0.85).

### 5.4 Incremental value beyond epoch (all checkpoints of the subset)

| subset | signal | partial rho given epoch | block CI | run-level CI | Δ rank-avg Spearman | hier CI | P(Δ > 0) |
|---|---|---|---|---|---|---|---|
| plateau | val_action_l1 | +0.21 | [−0.04, +0.50] | [+0.15, +0.27] | +0.01 | [−0.12, +0.15] | 0.60 |
| plateau | open_loop_smoothness | −0.07 | [−0.29, +0.26] | [−0.18, +0.04] | −0.01 | [−0.06, +0.04] | 0.38 |
| plateau | replan_overlap_h16 | +0.00 | [−0.28, +0.25] | [−0.06, +0.07] | −0.02 | [−0.08, +0.03] | 0.27 |
| plateau | boundary_jump_h16 | +0.02 | [−0.13, +0.15] | [−0.08, +0.09] | −0.02 | [−0.10, +0.02] | 0.18 |
| plateau | plv1_smoothness | −0.14 | [−0.43, +0.25] | [−0.22, −0.05] | −0.05 | [−0.14, +0.04] | 0.23 |
| plateau | teacher_forced_latent_mse | +0.13 | [−0.19, +0.43] | [−0.06, +0.33] | −0.15 | [−0.41, +0.17] | 0.17 |
| plateau | val_loss | −0.05 | [−0.35, +0.23] | [−0.18, +0.08] | −0.48 | [−0.64, +0.02] | 0.03 |
| plateau | other half K=25 | −0.14 | [−0.38, +0.05] | [−0.35, +0.08] | −0.23 | [−0.37, +0.10] | 0.11 |
| late_median | val_action_l1 | +0.25 | [+0.03, +0.61] | [+0.14, +0.35] | +0.07 | [−0.11, +0.23] | 0.78 |
| late_median | replan_overlap_h16 | −0.02 | [−0.44, +0.25] | [−0.10, +0.06] | −0.04 | [−0.23, +0.09] | 0.26 |
| late_median | other half K=25 | −0.24 | [−0.63, +0.02] | [−0.52, +0.10] | −0.31 | [−0.58, +0.16] | 0.13 |
| all | val_action_l1 | +0.27 | [+0.05, +0.50] | [+0.21, +0.33] | +0.01 | [−0.05, +0.06] | 0.64 |
| all | teacher_forced_latent_mse | +0.35 | [+0.05, +0.55] | [+0.15, +0.55] | −0.11 | [−0.22, +0.04] | 0.09 |
| all | replan_overlap_h16 | +0.05 | [−0.23, +0.27] | [−0.01, +0.12] | −0.01 | [−0.03, +0.02] | 0.27 |
| all | other half K=25 | +0.14 | [−0.11, +0.38] | [−0.09, +0.33] | −0.10 | [−0.19, +0.02] | 0.07 |

Every Δ Regret@30 (rank-average minus epoch) lies within ±0.04 and is 0.000 at the point estimate.

## 6. Interpretation

1. **On the full pool, epoch is already at the noise ceiling.** The 50-rollout target has a
   split-half rank reliability of only 0.59 (random-split mean). This happens because half of each
   run's checkpoints sit on a noisy plateau, and Spearman weights them equally. That implies a
   ceiling of about 0.84 for any predictor scored against S50, and 0.76 against a 25-rollout half.
   Experiment 1's epoch (0.815) and the best PLV metrics (0.77–0.80) are within a few hundredths of
   it. The original question therefore has no headroom: no metric can be shown to beat epoch on
   this pool, whatever it measures. The Pearson-scale ANOVA reliability is high (0.97), because the
   0 → 1 rise dominates the variance. It does not describe the rank-based evaluation.
2. **On the plateau, the success target has nothing left to rank except a weak common trend.** Two
   independent 25-rollout halves agree at +0.02 [−0.16, +0.21] (plateau) and −0.17 [−0.38, +0.05]
   (late half). Collecting 25 more rollouts per checkpoint therefore does **not** reproduce the
   plateau ranking. Checkpoint-specific deviations from a linear epoch trend are not detectable in
   any run (F_dev p = 0.46, 0.21, 0.99). In square they are smaller than chance (F p ≈ 0.98; the
   residual SD of S50 is 3.2 pp against a binomial SD of 5.2 pp). What remains is a small upward
   trend of 2–4 pp per 100 epochs. The permutation p-values for it are 0.03 / 0.004 / 0.03 on the
   plateau, but only can_seed2 is clearly non-zero by block CI.
3. **Epoch captures that trend at about the ceiling, and nothing adds to it.** Against random
   25-rollout halves epoch scores +0.35 (plateau) and +0.22 (late). A perfect trend follower is
   expected to score +0.34 and +0.19 under the trend model. The split-half ceiling (0.22 / 0.16) is
   below epoch because a split-half Spearman of ~0 is itself very imprecise with 15–29 binary-noisy
   checkpoints. The trend-model ceiling is the more informative one. Its ceiling for any metric's
   partial Spearman beyond epoch averages 0.14 on the plateau and 0.10 on the late half, and it is 0
   in 2 of 3 runs. Measured partial Spearmans are all consistent with 0. Rank-averaging any metric
   with epoch never improves Spearman beyond noise (Δ between −0.05 and +0.07 for the epoch-like
   metrics; P(Δ > 0) ≤ 0.78) and never changes Regret@30. val_action_l1 has the largest positive
   partial (+0.21 plateau, +0.25 late, block CIs touching 0). It is the only lead worth re-testing
   on an independent pool, and with 12+ metrics × 3 subsets tested, one such value is expected by
   chance.
4. **More rollouts and VLM judges do not help on the plateau either.** K=10 instrumented success
   (−0.11 to +0.19 split mean), K=10 VLM-judged success (Cosmos v6 −0.04 / −0.01, Nemotron −0.13 /
   +0.20) and even K=25 more training rollouts (−0.18 / +0.02) are all around 0. The VLMs are not
   the weak link: they track instrumented success over the full range. Rollout outcomes of any kind
   simply cannot separate checkpoints whose true success differs by about 1–2 pp. A 25-rollout
   half has 3–7 pp noise SD. The single eye-catching K=10 signal, executed_smoothness_K10 (+0.64
   fixed), falls to +0.12 on the split mean and +0.19 vs S50. It is a lucky-split artifact.
5. **Selection on the plateau barely matters in this pool.** Regret@30 against held-out halves is
   0–0.02 for every signal, random picking included (random 0.010 split mean, epoch 0.007, other
   half 0.018). Any plateau checkpoint is within about 1–2 pp of the best at 25-rollout resolution.
   PLV's practical value here is to reject undertrained checkpoints (the 0 → 0.9 rise). It is not
   about picking among plateau checkpoints.

**Bottom line:** in this pool, neither PLV metrics nor VLM-judged rollouts add information beyond
the training epoch. That holds on the full pool, where epoch is at the ceiling, and on the plateau,
where the target itself is mostly noise. All signals sit at or below the noise ceiling. The gap is
not a failing of the metrics. The pool does not contain enough real between-checkpoint variation
to reward them.

## 7. Limitations

* Post hoc and exploratory. The plateau cut was chosen from the learning curves. Many metrics and
  subsets were examined without multiplicity correction. Only 3 runs over 2 tasks, so run-level
  CIs are crude (27 resamples).
* The ceilings use classical test theory: index of reliability, ANOVA ICC, and a Gaussian trend
  model applied to Spearman on binary-outcome means with many ties. They are approximations. The
  split-half estimate is itself very noisy on 15–29 checkpoints (the per-run range spans ±0.3). A
  linear trend is a crude model of the plateau; a saturating curve might absorb more.
* Seeds are shared across checkpoints, which makes the naive binomial noise model wrong. The
  two-way ANOVA handles this but assumes additivity on the probability scale.
* The hier bootstrap double-counts target noise and is conservative for level statistics.
  Partial Spearmans for signals almost collinear with epoch (sim_replan_overlap_K10,
  sim_boundary_jump_K10) are numerically unstable, which explains their degenerate [0, 1] CIs.
* K=10 eval signals exist only for 6–8 checkpoints per run in the late and plateau subsets, so
  their CIs are very wide. The eval rollouts use a different recorder (video and keyframes) than
  training-time rollouts. Exp-3's `hybrid_screening.csv` was read as of its 07:24 KST snapshot.
* In these short runs success is nearly monotone in epoch. The conclusion "nothing beats epoch" may
  not transfer to runs with overfitting, collapse or learning-rate restarts. In those, true
  differences between late checkpoints could be large, and PLV metrics would have room to help.
  This pool cannot test that.

## 8. Files

* `scripts/hackathon/explore/y2_plateau_noise_ceiling.py`: main analysis (parts 1–3)
* `scripts/hackathon/explore/y2_plateau_noise_ceiling_trend.py`: trend and deviation decomposition
* `scripts/hackathon/explore/y2_plateau_noise_ceiling_tables.py`: markdown tables from the JSON
* `results/hackathon/explore/y2_plateau_noise_ceiling.json`, `..._table.csv`, `..._trend.json`
* scratch: `data/outputs/hackathon_explore/y2_plateau_noise_ceiling/` (logs, smoke runs, `final_tables.md`)
