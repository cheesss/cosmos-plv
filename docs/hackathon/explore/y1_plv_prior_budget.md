# y1_plv_prior_budget — does a cheap PLV prior cut the rollout budget?

**EXPLORATORY / post-hoc.** This is a simulation on logged training-time rollouts, run after
Experiments 1–3. It is not a pre-registered test. The PLV metrics tested were chosen after
Experiment 1, which had already seen all 50 training-time outcomes. That matters little here
because the result is negative.

## Question

"Cheap metrics first" claims that screening with an outcome-free PLV metric (keep the top 30 %
of checkpoints) lowers the number of rollouts needed to find a near-best checkpoint. We compare
it against three alternatives:

* (a) no prior: all checkpoints;
* (b) a random top-30 % subset;
* (c) the free "latest epoch" top-30 % prior.

Two further questions:

* Does any PLV prior beat **both** (b) and (c)?
* Does the answer change when training history is cut short, where the epoch prior should be
  weakest?

## Answer (short)

* **A prior helps a lot against no prior.** Any reasonable prior (epoch or a PLV metric) gets
  within 0.01 of the full-pool reference at the smallest budget tested, 0.25N. The exception
  is trunc_2, where it takes 0.25–1.5N (up to 4N for `val_action_l1`). Without a prior, TTTS needs
  2–8N and uniform allocation 6–20N with a perfect judge. With the noisy judges, no-prior
  allocation often does not get there within 25N (threshold: excess regret ≤ 0.01). The paired
  budget ratio "no prior / epoch prior" has a seed-bootstrap median of 2–8 for TTTS and 6–20
  for uniform allocation with a perfect judge. Across all judges and levels the range is 2–24
  (TTTS) and 6–48 (uniform). These ratios are lower bounds, because the prior arms already hit
  the grid floor.
* **No PLV prior beats the free epoch prior.** All 12 PLV priors were tested (6 single
  metrics + 6 two-metric rank averages) × 7 judge models × 4 strategies = 336 cells per
  history level. In **0 of 336 cells** at any of the 4 levels does a PLV prior beat both the
  random and the epoch prior with a paired 95 % CI. This holds for the 0.5N–8N average and for
  each of 0.5/1/2/4/8N, under both the seed bootstrap and random re-splits. Paired differences
  against epoch are ≈ 0.000 ± 0.006 (full history). In the full history the top-30 % sets of
  the PLV metrics are 85–100 % the same checkpoints as the latest-epoch top-30 %.
* **The truncated-history stress test does not rescue PLV.** At the most truncated level
  (can ≤ 60, square ≤ 90) `val_action_l1` has the best point estimate: average regret
  0.035 vs 0.051 for epoch, Δ −0.017 [−0.068, +0.040]. This is an artifact of the original
  25/25 split: on random re-splits `val_action_l1` beats epoch in only 8 % of splits.
  `open_loop_smoothness` and `val_loss_logged` are worse than epoch there.
* **PLV vs random.** PLV and epoch priors beat the random 30 % subset at low budgets. The
  effect is clear under noisy judges and truncated histories: e.g. Cosmos-bb at 1N, trunc_1,
  Δ −0.11 [−0.20, −0.02]. With a perfect judge on the full history the CI touches 0. Random
  top-30 % is barely better than no prior, because the success plateau is wide: 21/41 can_seed1
  checkpoints tie at held-out success 1.0.
* `val_loss_logged` is harmful as a prior. Its average regret is 0.105 vs 0.023 for epoch on the
  full history; its top-30 % is epochs 20–110, because DP val loss rises late in training. This
  agrees with x1.
* **Zero rollouts beat the reference.** The free latest-epoch pick has held-out regret 0.000
  [0.000, 0.053] on the full history. That is below the 25N full-pool uniform reference
  (0.033). After the epoch prior, extra rollouts mostly buy noise on these runs.

## Method

* **Data.** 93 checkpoints: can_seed1 41 (every 5 epochs, 0–200), can_seed2 21 (every 10),
  square_seed1 31 (every 10, 0–300). Each has 50 training-time rollouts
  (`test/sim_max_reward_<seed>`, seeds 100000–100049), read by x1's `load_runs`, which was
  imported unmodified.
  * The `success` column of `baseline_metrics.csv` matches the 50-episode log mean exactly
    (max abs diff 0.0). It is used only for this alignment check.
* **Split, no leakage.**
  * Seeds 100000–100024 form the rollout pool, the only outcomes a strategy may observe. They
    are drawn in a random order shared by all checkpoints.
  * Seeds 100025–100049 are held out. They are used only to score the pick:
    regret = max_c H(c) − H(pick), with H = held-out success.
  * Priors read only outcome-free columns: PLV metrics, `val_loss_logged`, epoch.
* **Priors.** Each prior keeps the top ⌈0.3 N⌉ checkpoints; ties go to the later epoch.
  * Single: `val_action_l1`, `open_loop_smoothness`, `plv1_smoothness`, `replan_overlap_h16`,
    `boundary_jump_h16`, `val_loss_logged` (lower is better); `epoch` (later is better);
    `random` (a fresh random 30 % per repetition).
  * Two-metric rank averages: epoch+{replan_overlap_h16, open_loop_smoothness, val_action_l1,
    boundary_jump_h16, plv1_smoothness}, and val_action_l1+replan_overlap_h16.
  * `none` = all N checkpoints.
* **Strategies.** Applied inside the subset, using x1's code unmodified: uniform, successive
  halving, Thompson, top-two Thompson (TTTS).
  * Budget B = mult × N, where N counts **all** checkpoints of the problem;
    mult ∈ {0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 20, 25}.
  * Per-arm cap = 25 pool episodes, so a top-30 % arm set saturates at about 7.5N.
  * Recommendation = highest Beta(1,1) posterior mean.
* **Judges.** The error models are re-estimated from the FINAL eval JSONLs: 330 episodes,
  seeds 200000+, disjoint from the training-time seeds.
  * `perfect`.
  * `*_iid`: per-episode flips; task-specific FPR and FNR ~ Beta(1+k, 1+n−k), redrawn each
    repetition (the x1 model).
  * `*_bb`: a beta-binomial model, i.e. checkpoint-specific FPR_c ~ Beta(μκ, (1−μ)κ), for
    within-checkpoint error correlation. μ ~ the task posterior; κ is drawn from the profile
    likelihood of per-checkpoint FP counts on a log grid from 0.25 to 1e4.
  * `*_bbq`: like `bb`, but FPR rises with checkpoint quality:
    logit μ_c = a_task + b (s_c − 0.5). It is fitted on eval truth. In the simulation s_c is
    the checkpoint's pool success, a world-model choice the strategies never see. (a, b) are
    Laplace draws; κ uses profile weights.
  * FNR is task-specific iid for all noisy judges.
* **Uncertainty.**
  * **Seed bootstrap:** resample the 25 pool seeds and the 25 held-out seeds with replacement,
    using the same resample for every checkpoint and run (a block bootstrap over seeds).
  * 200 replicates × 10 repetitions, plus the original seeds × 400 repetitions for the point
    estimates.
  * Percentile CIs over replicates. They include Monte-Carlo noise, so they are conservative.
    They are also shifted upward relative to the point estimate, because resampling with
    replacement duplicates seeds.
  * Paired differences use common random numbers (order, judge noise and judge rates are
    shared across priors) and are taken within each replicate.
  * Sensitivity: 200 random 25/25 **re-splits** of the 50 seeds, with no duplicated episodes
    (`*_splits.json`).
  * Runs are pooled by an unweighted mean over the 3 runs. Checkpoints within a run are never
    treated as independent units.
* **Budget-to-target.** The smallest grid mult where (mean-over-runs regret − reference) ≤ thr,
  for thr ∈ {0, 0.01, 0.02}. The reference is the perfect-judge, no-prior uniform pick at 25N
  (all 25 pool episodes per checkpoint). "> 25" means not reached.
* **Truncated history.** Keep only checkpoints with epoch ≤ E, then recompute priors, subsets
  and regret within that set:
  * trunc_1: can ≤ 60, square ≤ 90 (N = 13/7/10)
  * trunc_2: can ≤ 100, square ≤ 150 (N = 21/11/16)
  * trunc_3: can ≤ 150, square ≤ 240 (N = 31/16/25)

## Commands (all exit 0)

```
# smoke tests (scratch outputs only)
PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/y1_plv_prior_budget.py \
  --r0 40 --n-boot 10 --r-in 4 --workers 8 --out data/outputs/hackathon_explore/y1_plv_prior_budget/smoke.json ...   # exit 0
# main (seed bootstrap), 244 s on 10 CPU workers
PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/y1_plv_prior_budget.py --workers 10   # exit 0
# sensitivity: random 25/25 re-splits, 149 s
PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/y1_plv_prior_budget.py --workers 12 \
  --resample split --out results/hackathon/explore/y1_plv_prior_budget_splits.json --fig ""   # exit 0
# tables
PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/y1_plv_prior_budget_tables.py [json]   # exit 0
PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/y1_plv_prior_budget_tables.py --headline [json]   # exit 0
```

The first full run, made before multi-threshold outputs were added, produced identical numbers:
same seeds, exit 0. It was overwritten. Logs are in
`data/outputs/hackathon_explore/y1_plv_prior_budget/`. Full tables, about 2,000 lines per JSON,
are in `tables_main.md`, `tables_splits.md`, `headline_main.md` and `headline_splits.md` in the
same folder.

## Judge error models (final eval JSONLs)

| judge | task | FP / true failures | FN / true successes | bb κ MLE (LR vs binomial) | P(κ≤10), bb | bbq slope b (SE) | bbq FPR at s=0.5 / 0.9 |
|---|---|---|---|---|---|---|---|
| Nemotron v2 | can | 5/38 | 0/182 | 1.6 (4.66) | 0.72 | 4.77 (1.81) | 0.11 / 0.46 |
| Nemotron v2 | square | 4/29 | 3/81 | (shared) | | | 0.09 / 0.40 |
| Cosmos3-Edge v6 | can | 8/38 | 0/182 | 3.5 (3.29) | 0.47 | 3.35 (0.97) | 0.24 / 0.55 |
| Cosmos3-Edge v6 | square | 13/29 | 6/81 | (shared) | | | 0.48 / 0.78 |

**False positives cluster within checkpoints** (LR 3.3–4.7 against no clustering). The
clustering is largely explained by checkpoint quality: once the slope b is in the model, κ
moves to the binomial limit. The failures of good checkpoints are near-misses, and the judges
call them successes. For example, 1/1 FP at checkpoints with 9/10 true successes, and 2/3 FP at one with 7/10.

## Full history (N = 41/21/31), TTTS

Mean held-out regret over runs, with a 95 % seed-bootstrap CI. After the slash: the average over
0.5N–8N.

| judge | none | random | epoch | replan_overlap_h16 | val_action_l1 | epoch+replan_overlap_h16 | val_loss_logged |
|---|---|---|---|---|---|---|---|
| perfect @1N | 0.046 [0.036, 0.107] / 0.042 | 0.041 [0.035, 0.095] / 0.040 | 0.020 [0.015, 0.064] / 0.023 | 0.019 [0.016, 0.067] / 0.023 | 0.020 [0.012, 0.065] / 0.023 | 0.018 [0.013, 0.069] / 0.023 | 0.103 [0.072, 0.177] / 0.105 |
| nemotron_bb @1N | 0.051 [0.036, 0.119] / 0.049 | 0.047 [0.035, 0.113] / 0.047 | 0.017 [0.015, 0.065] / 0.019 | 0.018 [0.017, 0.067] / 0.019 | 0.019 [0.016, 0.065] / 0.020 | 0.017 [0.015, 0.064] / 0.019 | 0.108 [0.080, 0.180] / 0.110 |
| cosmos_bb @1N | 0.052 [0.040, 0.124] / 0.051 | 0.051 [0.039, 0.111] / 0.048 | 0.015 [0.016, 0.060] / 0.017 | 0.015 [0.015, 0.063] / 0.017 | 0.017 [0.016, 0.063] / 0.017 | 0.016 [0.013, 0.064] / 0.017 | 0.114 [0.075, 0.184] / 0.108 |
| cosmos_bbq @1N | 0.049 [0.041, 0.105] / 0.044 | 0.047 [0.037, 0.101] / 0.044 | 0.016 [0.013, 0.063] / 0.016 | 0.015 [0.015, 0.063] / 0.016 | 0.017 [0.015, 0.059] / 0.018 | 0.016 [0.013, 0.063] / 0.016 | 0.112 [0.081, 0.183] / 0.104 |

Paired TTTS differences at 1N (prior − control; negative means the prior is better):

| judge | prior | vs none | vs random | vs epoch |
|---|---|---|---|---|
| perfect | epoch | −0.026 [−0.069, 0.004] | −0.021 [−0.056, 0.004] | — |
| perfect | replan_overlap_h16 | −0.027 [−0.069, 0.003] | −0.022 [−0.057, 0.007] | −0.001 [−0.019, 0.017] |
| perfect | epoch+replan_overlap_h16 | −0.028 [−0.068, −0.003] | −0.023 [−0.057, 0.007] | −0.001 [−0.017, 0.017] |
| nemotron_bb | replan_overlap_h16 | −0.032 [−0.079, −0.009] | −0.028 [−0.071, 0.001] | +0.002 [−0.016, 0.015] |
| cosmos_bb | replan_overlap_h16 | −0.037 [−0.092, −0.003] | −0.036 [−0.072, −0.003] | +0.000 [−0.019, 0.015] |
| cosmos_bb | val_action_l1 | −0.036 [−0.087, 0.000] | −0.035 [−0.072, −0.003] | +0.001 [−0.020, 0.017] |

We also took the post-hoc best PLV prior per judge (lowest point average 0.5N–8N), which gives
the PLV side its most favourable selection. Even then, Δ vs epoch is between −0.0003 and +0.0003,
with CIs of about ±0.006–0.010. For example, perfect, epoch+replan_overlap_h16: −0.0002
[−0.0062, 0.0062].

## Budget to reach excess regret ≤ thr vs the perfect no-prior uniform 25N reference

The reference regret is 0.033 for the full history, 0.060 for trunc_1, 0.020 for trunc_2 and
0.035 for trunc_3. Each cell gives the point estimate (original seeds), with the seed-bootstrap
median in brackets, as multiples of N. "Prior" means every prior except random, val_loss_logged,
and open_loop/plv1 at trunc_1, which reach the target later.

| level | strategy | judge | none thr 0.01 / 0.02 | random thr 0.01 / 0.02 | epoch & PLV priors thr 0.01 / 0.02 | paired ratio none/epoch at 0.01, median [p10, p90] |
|---|---|---|---|---|---|---|
| full | TTTS | perfect | 2 (0.5) / 0.5 | 1 / 0.25 | 0.25 / 0.25 | 2 [1, 24] |
| full | TTTS | nemotron_bb | 12 (1) / 1 | >25 / 1 | 0.25 / 0.25 | 4 [1, 35] |
| full | TTTS | cosmos_bb | >25 (1) / 1 | 6 / 1 | 0.25 / 0.25 | 4 [1, 200*] |
| full | uniform | perfect | 6 (1.5) / 2 | 2 / 0.5 | 0.25 / 0.25 | 6 [1, 48] |
| full | uniform | nemotron_bb | 25 (3.5) / 4 | 3 / 1 | 0.25 / 0.25 | 12 [1, 200*] |
| full | uniform | cosmos_bb | >25 (4) / 8 | >25 / 2 | 0.25 / 0.25 | 12 [1, 200*] |
| trunc_1 | TTTS | perfect | 4 (3) / 3 | 8 / 3 | 0.25 / 0.25 | 6 [2, 32] |
| trunc_1 | TTTS | nemotron_bb | >25 (6) / >25 | >25 / >25 | 0.25 / 0.25 | 12 [1.3, 200*] |
| trunc_1 | uniform | cosmos_bb | >25 (16) / 20 | >25 / >25 | 0.25 / 0.25 | 33 [4, 200*] |
| trunc_2 | TTTS | perfect | 8 (3) / 6 | >25 / >25 | 0.25–1.5 / 0.25 | 8 [1, 32] |
| trunc_2 | uniform | nemotron_bb | >25 (22.5) / >25 | >25 / >25 | 0.25–3 / 0.25 | 48 [5.8, 200*] |
| trunc_3 | TTTS | perfect | 6 (1) / 2 | >25 / 1 | 0.25 / 0.25 | 4 [1, 24] |
| trunc_3 | uniform | cosmos_bb | >25 (8) / 12 | >25 / 4 | 0.25 / 0.25 | 32 [1, 200*] |

\* "Not reached within 25N" is coded as 50N, so a ratio of 200 means 50/0.25.

**All priors saturate at the grid floor (0.25N).** That is fewer than one rollout per subset
checkpoint, so the budget metric cannot separate PLV from epoch. The zero-rollout pick of the
epoch prior is already below the reference: regret 0.000 [0.000, 0.053] against the reference
0.033. The paired crossing comparison PLV vs epoch has median ratio 1 [1, 1]. In only 1–7 % of
bootstrap replicates does either one cross first.

## Truncated-history stress test (TTTS, average regret over 0.5N–8N)

| level | judge | none | random | epoch | val_action_l1 | replan_overlap_h16 | open_loop_smoothness | best PLV (post hoc) Δ vs epoch [95 % CI] |
|---|---|---|---|---|---|---|---|---|
| trunc_1 | perfect | 0.083 | 0.089 | 0.051 | 0.035 | 0.051 | 0.073 | val_action_l1 −0.017 [−0.068, 0.040] |
| trunc_1 | nemotron_bb | 0.115 | 0.117 | 0.048 | 0.038 | 0.048 | 0.074 | val_action_l1 −0.010 [−0.056, 0.042] |
| trunc_1 | cosmos_bb | 0.116 | 0.129 | 0.046 | 0.041 | 0.045 | 0.072 | val_action_l1 −0.005 [−0.046, 0.034] |
| trunc_2 | perfect | 0.054 | 0.055 | 0.026 | 0.025 | 0.027 | 0.025 | epoch+plv1 −0.004 [−0.018, 0.006] |
| trunc_2 | cosmos_bb | 0.081 | 0.073 | 0.027 | 0.028 | 0.027 | 0.025 | epoch+plv1 −0.005 [−0.019, 0.008] |
| trunc_3 | perfect | 0.051 | 0.050 | 0.034 | 0.035 | 0.035 | 0.040 | val_action_l1+replan −0.002 [−0.013, 0.008] |
| trunc_3 | cosmos_bb | 0.063 | 0.056 | 0.030 | 0.033 | 0.030 | 0.034 | epoch+val_action_l1 −0.002 [−0.011, 0.007] |

* **Re-split sensitivity.** Over 200 random 25/25 re-splits, `val_action_l1` beats epoch at
  trunc_1 in only 8.5 % (perfect), 8 % (Nemotron-bb) and 5 % (Cosmos-bb) of splits. The spread
  of Δ is [−0.008, +0.060]. The point advantage comes from one favourable split.
  * In square at E ≤ 90, both subsets contain the held-out best epoch (70). Which plateau
    checkpoint wins on the 25 pool seeds is decided by noise.
* At trunc_1, `open_loop_smoothness` puts epochs 20 and 30 (can_seed1) and 40–60 (square) in its
  top 30 %. It is worse than epoch by about +0.02 (CI includes 0).
* Paired TTTS at 1N, trunc_1 (perfect): epoch vs none −0.052 [−0.113, 0.001] and vs random
  −0.059 [−0.129, 0.005].
* Same comparison under the Cosmos-bb judge: −0.098 [−0.195, −0.024] vs none and −0.109
  [−0.198, −0.021] vs random. The prior's value **grows** with judge noise and with truncation,
  but it is the epoch prior's value. PLV metrics add nothing on top of it.

## Interpretation

1. The efficiency part of "cheap metrics first" holds. Screening to a top-30 % subset cuts the
   budget needed to match full-pool selection by at least 2–8x (TTTS) and 6–20x (uniform) with
   a perfect judge, and by more (up to 24x and 48x) under noisy VLM judges and truncated
   histories. It also roughly halves regret at 1N.
2. In these three runs, **the free latest-epoch prior delivers all of that gain**. The PLV
   metrics Spearman-correlate with success (0.73–0.86 in full histories) mainly because they
   correlate with epoch. Their top-30 % sets are nearly the latest 30 %, and none adds
   measurable value beyond epoch. Rank-averaging a PLV metric with epoch gives the epoch
   result. Our stated claim should therefore be "screen with the epoch prior, then allocate
   rollouts adaptively", not "PLV saves rollouts".
3. Truncated history was the regime most likely to favour PLV, because the epoch prior is
   weakest near a plateau. Here the plateau starts early, so the latest checkpoints are still
   near-best at every cut-off tested, and no PLV metric beats epoch reliably.
4. Absolute regrets below about 0.02–0.03 are not interpretable. With 25 held-out episodes,
   the full-pool reference itself has regret 0.033. Extra pool rollouts can even raise regret
   inside a plateau subset: square epoch prior 0.045 at 0.25N vs 0.080 at 25N, because the
   pool-best is not the held-out best. Noisy judges sometimes give lower regret than the
   perfect judge for the same reason: noise randomizes the pick among near-equal checkpoints.
   That is not a real benefit.

## Limitations

* Only 3 runs (2 tasks), all short and almost monotone in epoch. A run with a late collapse,
  overfitting or LR spikes would be needed to show a PLV advantage. None is in the pool. The
  run-level n = 3 makes the mean-over-runs CIs (seed bootstrap) optimistic about between-run
  variation.
* The budget metric saturates at the 0.25N grid floor for all good priors. Finer budgets, or a
  "0 rollouts" point, would only confirm that the epoch top-1 already beats the reference.
* Seed-bootstrap percentile CIs are biased upward: duplicated seeds both weaken the pool and
  add noise to H. The re-split sensitivity agrees on every qualitative conclusion. Its
  intervals are split spreads, not CIs.
* The judge models come from 330 eval episodes on 33 checkpoints, with only 67 true failures.
  The bbq slope is uncertain: SE 1.8 for Nemotron. FN is modelled iid. The world model ties
  FPR to pool success.
* Multiplicity: 336 cells per level. The negative result (0 cells beating both controls) is
  robust to this, but any single positive Δ in the tables should not be cited alone.
* PLV metrics were computed once per checkpoint with the Experiment 1 settings. Their own
  computational cost is not counted in the budget.

## Files

* Script: `scripts/hackathon/explore/y1_plv_prior_budget.py`. It imports
  `x1_adaptive_budget.py` and never modifies it.
* Tables: `scripts/hackathon/explore/y1_plv_prior_budget_tables.py`.
* Results:
  * `results/hackathon/explore/y1_plv_prior_budget.json` (seed bootstrap)
  * `results/hackathon/explore/y1_plv_prior_budget_splits.json` (random re-splits)
* Figure: `docs/hackathon/explore/y1_plv_prior_budget_regret.png`. TTTS regret vs budget for
  full/perfect, full/Nemotron-bb, full/Cosmos-bb and trunc_1/perfect.
* Scratch: `data/outputs/hackathon_explore/y1_plv_prior_budget/` (logs and full markdown
  tables).
