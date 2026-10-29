# y5_cross_seed_selection: picking ONE checkpoint across two seeds (EXPLORATORY, post-hoc)

> **Label:** exploratory and post-hoc. Nothing here was pre-registered in `06_experiment_protocol.md`.
> Every number comes from `results/hackathon/explore/y5_cross_seed_selection.json`, written by
> `scripts/hackathon/explore/y5_cross_seed_selection.py`. Tables were rendered by
> `y5_cross_seed_selection_tables.py` into `results/hackathon/explore/y5_cross_seed_selection_tables.txt`.
> CPU only. No GPU and no Nebius calls.

**One-line summary:** Even when the two Can seeds are pooled into one selection problem, PLV is no better than epoch. The pooled Spearman was 0.826 for epoch, 0.790 for Action L1 and 0.805 for rank-avg(epoch, L1), and the CIs of all differences against epoch include 0. In the pairwise test that separates the two seeds at the same epoch, only 4 pairs had a success-rate difference of 0.06 or more. PLV metric accuracy was 0.25–0.50, and the K=10 eval rollouts (both instrumented and VLM) got both of 2 pairs wrong. In this pool the differences between seeds are mostly at noise level (≤0.04), so there is almost nothing to choose. The only "significant" one, teacher-forced latent MSE (13 of 14 pairs correct), did not reproduce on the held-out half (5/9).

## 1. Question

Practitioners usually train several seeds and then have to ship one checkpoint. Within one run,
"latest epoch" is a strong baseline (Experiment 1: mean within-run Spearman 0.815). It cannot
choose between two seeds at the same epoch, though. Does a cheap PLV metric, or a small batch
of K=10 eval rollouts judged by a VLM, help when the checkpoints of both seeds are pooled into one
selection problem?

## 2. Data and selection problems

* Runs: `can_seed1` (41 ckpts, every 5 epochs, 0–200) and `can_seed2` (21 ckpts, every 10 epochs,
  0–200). This is the only task with two seeds.
* Cheap metrics: `results/hackathon/baseline_metrics.csv` (Experiment 1, frozen paper flags). All
  are costs (lower is better).
* Target: training-time success from the 50 per-episode outcomes `test/sim_max_reward_<seed>`
  (seeds 100000–100049) in each run's `logs.json.txt`. The per-episode means reproduce `success`
  exactly (asserted). Halves: **HA** = 100000–100024, **HB** = 100025–100049.
* Eval rollouts: 22 Can eval checkpoints (multiples of 20) × 10 episodes (seeds 200000–200009,
  disjoint from the target seeds). Each episode has instrumented success and four VLM judgements
  (`eval_local_v6`, `eval_local_v7`, `eval_nebius_v2`, `eval_nebius_v2_pad201`). All 220 Can rows of
  every file are scored. On Can, v6 and v7 give identical success judgements in all 220 episodes;
  v2 and pad201 differ in 1 of 220.

| pool | definition | N |
|---|---|---|
| `common10` (primary) | epochs that are multiples of 10, both seeds | 42 |
| `all` | every checkpoint (seed1 also has the x5 epochs) | 62 |
| `late_common10` | `common10` with epoch ≥ 100 | 22 |
| `eval20` | multiples of 20 (the only pool with K=10 / VLM scores) | 22 |

## 3. Method

**Scores** (higher = ranked better):
* epoch;
* each of the 12 cheap metrics, negated;
* rank-average(epoch, metric);
* lexicographic "epoch, ties → metric", the natural "latest epoch, then let PLV pick the seed" rule;
* K=10 instrumented or VLM success, alone, rank-averaged with epoch, rank-averaged with Action L1
  (the Experiment-3 hybrid), and with ties broken by Action L1;
* a random-order reference.

**Ties.** Epoch ties between seeds and the many K=10 ties at 1.0 are broken uniformly at random.
I report the exact expectation over tie-breaks, which is the limit of averaging over infinitely
many tie-breaks:
* Spearman: the midrank covariance divided by the untied rank variance.
* Regret: computed combinatorially for the tied group at the top-k cut.

A 20,000-draw Monte-Carlo check agrees: epoch Spearman 0.82593 vs exact 0.82595, top-1 regret
0.01006 vs 0.01000. The `cosmos_plv.ranking` path tie-break would always give seed1 the ties
(top-1 regret 0.020 instead of 0.010).

**Statistics.** For each pool I report:
* Spearman with S;
* Regret@q = max S − max S over the top ⌈qN⌉, for q = 0.1 and 0.3;
* top-1 regret (k = 1).

**Leakage control.** None of the outcome-free scores touch training-time outcomes, so they are
scored against S50. Three scores do use outcomes:
* K=25 training-rollout success;
* "the cheap metric with the best Spearman on the fit half";
* rank-avg(epoch, K=25).

These are fit on HA and scored only on HB (reverse fold HB→HA also reported). All scores are
reported on HB as well, so they can be compared on equal terms.

**CIs.** 2000 joint bootstrap replicates of three things:
* moving blocks of epoch bins. A block is 40 epochs, and each bin carries the checkpoints of **both**
  seeds at those epochs, which keeps the cross-seed pairing and the serial correlation of
  neighbouring checkpoints;
* training env seeds, with the same resample for every checkpoint, stratified by half;
* eval env seeds, with the same resample for every checkpoint and judge.

Differences vs epoch are paired within each replicate. A run-level bootstrap is impossible with
2 runs. The epoch-block resampling makes the *absolute* CIs very wide, because a replicate can
contain mostly saturated late epochs. The paired Δ-vs-epoch CIs are the informative ones.

**Epoch-matched pairwise test.** At each of the 21 common epochs, a metric picks the seed with
the lower cost. The pick is correct if that seed's checkpoint has the higher S.
* Primary analysis: pairs with |ΔS50| ≥ 0.06. A metric tie counts as 0.5.
* Statistics: exact Clopper-Pearson CI and binomial test vs 0.5, with ties excluded.
* Controls: always-seed1 / always-seed2 pickers, and a circular-shift null that shifts the
  metric's decision sequence across epochs.
* Robustness:
  * filter pairs on one half and score them on the other;
  * a relaxed "any difference" filter (|ΔS50| ≥ 0.02), with a Holm correction over the 12 metrics;
  * a held-out choice of the pairwise metric: choose it on one half's pairs, test it on the other's;
  * a threshold-free within-pair correlation, Spearman(Δcost, ΔS), over the 20 non-trivial pairs.

## 4. Commands and exit codes

```
PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/y5_cross_seed_selection.py --n-boot 20 \
    --out data/outputs/hackathon_explore/y5_cross_seed_selection/smoke.json        # smoke, exit 1 (All-NaN in fit-half
                                                                                   # selection on a constant resample) -> fixed, exit 0
PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/y5_cross_seed_selection.py --n-boot 2000
                                                                                   # exit 0, 541 s (log: data/outputs/hackathon_explore/
                                                                                   # y5_cross_seed_selection/run_main.log)
PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/y5_cross_seed_selection_tables.py   # exit 0
```

## 5. Results

### 5.1 What is at stake: very little

| epoch | 0 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100–200 (11 epochs) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S50 seed1 | 0.00 | 0.16 | 0.56 | 0.70 | 0.86 | 0.90 | 0.94 | 0.94 | 0.96 | 0.98 | 0.96–1.00 |
| S50 seed2 | 0.00 | 0.18 | 0.48 | 0.84 | 0.80 | 0.84 | 0.90 | 0.94 | 0.96 | 1.00 | 0.96–1.00 |
| ΔS50 (s2−s1) | 0 | +.02 | **−.08** | **+.14** | **−.06** | **−.06** | −.04 | 0 | 0 | +.02 | −.02 … +.04 |

* Only **4 of 21** epoch-matched pairs differ by ≥ 0.06 (epochs 20–50, early in training).
* From epoch 60 on, every pair differs by ≤ 0.04, which is ≤ 2 of 50 episodes. The SE of a
  difference at S ≈ 0.97 is about 0.034.
* The final checkpoints are 0.98 (seed1) and 1.00 (seed2). The largest top-1 regret possible
  among late checkpoints is 0.04.
* Split-half reliability of the target: Spearman(HA, HB) is 0.69 on `common10`, 0.66 on `eval20`
  and **−0.19 on `late_common10`**. Among late checkpoints the target ranking does not replicate
  across halves at all, so no method can be scored meaningfully there.

### 5.2 Pooled selection (target S50)

`common10` (N=42). Point estimate [95% block × episode bootstrap CI]; regrets are in success-rate units.

| score | Spearman | ΔSpearman vs epoch | Regret@10% | Regret@30% | top-1 regret | Δtop-1 vs epoch |
|---|---|---|---|---|---|---|
| random (reference) | 0 | −0.826 | 0.006 | 0.000 | 0.153 [0.017, 0.271] | +0.143 |
| **epoch (ties random)** | **0.826** [0.013, 0.943] | — | 0.000 [0, 0.010] | 0.000 | 0.010 [0, 0.060] | — |
| Action L1 | 0.790 | −0.036 [−0.300, +0.121] | 0.000 [0, 0.040] | 0.000 | 0.000 [0, 0.060] | −0.010 [−0.060, +0.050] |
| open-loop smoothness | 0.773 | −0.053 [−0.248, +0.024] | 0.020 [0, 0.060] | 0.000 | 0.020 [0, 0.080] | +0.010 [−0.020, +0.040] |
| replan overlap H=16 | 0.778 | −0.048 [−0.246, +0.033] | 0.020 [0, 0.060] | 0.000 [0, 0.020] | 0.020 [0, 0.080] | +0.010 [−0.010, +0.030] |
| PLV-1 smoothness | 0.728 | −0.098 [−0.326, +0.036] | 0.000 | 0.000 | 0.000 [0, 0.080] | −0.010 [−0.030, +0.030] |
| boundary jump H=16 | 0.770 | −0.056 [−0.289, +0.029] | 0.020 [0, 0.060] | 0.000 [0, 0.040] | 0.020 [0, 0.080] | +0.010 [−0.010, +0.030] |
| train loss | 0.819 | −0.007 [−0.047, +0.021] | 0.000 [0, 0.020] | 0.000 | 0.020 [0, 0.080] | +0.010 [−0.010, +0.030] |
| teacher-forced latent MSE | 0.696 | −0.130 [−0.641, +0.392] | 0.000 | 0.000 | 0.000 [0, 0.080] | −0.010 [−0.060, +0.060] |
| validation loss | −0.413 | −1.239 [−1.646, +0.060] | 0.100 [0, 0.140] | 0.040 [0, 0.100] | 0.160 [0.020, 0.260] | +0.150 [+0.010, +0.240] |
| rank-avg(epoch, Action L1) | 0.805 | −0.021 [−0.159, +0.089] | 0.000 [0, 0.040] | 0.000 | 0.000 [0, 0.080] | −0.010 [−0.030, +0.030] |
| rank-avg(epoch, open-loop) | 0.807 | −0.019 [−0.102, +0.030] | 0.000 [0, 0.040] | 0.000 | 0.020 [0, 0.080] | +0.010 [−0.010, +0.030] |
| rank-avg(epoch, replan H=16) | 0.798 | −0.028 [−0.144, +0.018] | 0.000 [0, 0.060] | 0.000 | 0.020 [0, 0.080] | +0.010 [−0.010, +0.030] |
| rank-avg(epoch, TF latent MSE) | 0.837 | +0.011 [−0.223, +0.314] | 0.000 | 0.000 | 0.000 [0, 0.080] | −0.010 [−0.040, +0.050] |
| epoch, ties → Action L1 | 0.823 | −0.003 [−0.037, +0.013] | 0.000 [0, 0.020] | 0.000 | 0.000 [0, 0.080] | −0.010 [−0.030, +0.030] |
| epoch, ties → open-loop | 0.822 | −0.004 [−0.042, +0.024] | 0.000 [0, 0.020] | 0.000 | 0.020 [0, 0.080] | +0.010 [−0.010, +0.030] |

* `all` (N=62) tells the same story: epoch 0.807, Action L1 0.777, rank-avg(epoch, L1) 0.801,
  "epoch, ties → L1" 0.806. Every Δ-vs-epoch CI includes 0. Replan H=16 and boundary jump have
  Δtop-1 CIs of [+0.000, +0.020] and [+0.000, +0.040]: never better than epoch, and at best tied.
* `late_common10` (N=22): epoch 0.142 [−0.47, 0.60], Action L1 0.104, open-loop −0.132,
  replan −0.167, PLV-1 −0.045, TF latent MSE 0.290. All top-1 regrets are 0.00–0.02. This is
  uninterpretable given the negative split-half reliability.

`eval20` (N=22): K=10 eval rollouts vs. the free baselines.

| score | Spearman | ΔSpearman vs epoch | Regret@10% | top-1 regret | Δtop-1 vs epoch |
|---|---|---|---|---|---|
| **epoch** | **0.891** [0.184, 0.958] | — | 0.000 | 0.010 [0, 0.040] | — |
| Action L1 | 0.795 | −0.096 [−0.436, +0.049] | 0.000 [0, 0.060] | 0.000 [0, 0.080] | −0.010 [−0.030, +0.060] |
| open-loop smoothness | 0.834 | −0.057 [−0.332, +0.093] | 0.020 [0, 0.060] | 0.020 [0, 0.080] | +0.010 [−0.010, +0.040] |
| K=10 instrumented success | 0.703 | −0.187 [−0.688, +0.029] | 0.007 [0, 0.027] | 0.020 [0.003, 0.080] | +0.010 [−0.008, +0.071] |
| K=10 Nemotron v2 (= pad201) | 0.691 | −0.199 [−0.660, +0.049] | 0.008 [0, 0.023] | 0.023 [0.005, 0.073] | +0.013 [−0.006, +0.062] |
| K=10 Cosmos3-Edge v6 (= v7) | 0.621 | −0.270 [−0.745, +0.022] | 0.009 [0, 0.026] | 0.028 [0.006, 0.085] | +0.018 [−0.004, +0.073] |
| rank-avg(epoch, K=10 instrumented) | 0.841 | −0.050 [−0.263, +0.016] | 0.000 [0, 0.020] | 0.010 [0, 0.050] | +0.000 [0, +0.020] |
| rank-avg(epoch, Nemotron v2) | 0.891 | +0.000 [−0.031, +0.027] | 0.000 | 0.010 [0, 0.040] | +0.000 [0, 0] |
| rank-avg(epoch, Cosmos v6) | 0.890 | −0.001 [−0.034, +0.011] | 0.000 | 0.010 [0, 0.040] | +0.000 [0, 0] |
| rank-avg(Action L1, Nemotron v2) (Exp-3 hybrid) | 0.852 | −0.038 [−0.319, +0.091] | 0.000 [0, 0.060] | 0.000 [0, 0.080] | −0.010 [−0.030, +0.060] |
| Nemotron v2, ties → Action L1 | 0.851 | −0.040 [−0.318, +0.094] | 0.000 [0, 0.060] | 0.000 [0, 0.080] | −0.010 [−0.030, +0.060] |

### 5.3 Held-out half (target HB; outcome-using scores fit on HA)

| score (`common10`, N=42) | Spearman vs HB | ΔSpearman vs epoch | top-1 regret |
|---|---|---|---|
| epoch | 0.861 [0.324, 0.925] | — | 0.000 |
| Action L1 | 0.846 | −0.016 [−0.167, +0.116] | 0.000 |
| rank-avg(epoch, Action L1) | 0.861 | −0.001 [−0.072, +0.087] | 0.000 |
| **K=25 real training rollouts (HA)** | 0.669 | **−0.193 [−0.655, −0.014]** | 0.007 [0, 0.056] |
| cheap metric with best HA Spearman (chose TF latent MSE) | 0.599 | −0.262 [−0.642, +0.018] | 0.000 |
| any outcome-free score chosen on HA | 0.776 | −0.085 [−0.603, +0.022] | 0.000 |
| rank-avg(epoch, K=25 HA) | 0.811 | −0.050 [−0.275, +0.020] | 0.000 [0, 0.020] |

* Reverse fold (fit HB, score HA): epoch 0.716, Action L1 0.669, K=25 0.629. The chosen metric
  was replan H=16, which scored 0.656. Top-1 regret: epoch 0.020, Action L1 0.000, K=25 0.038.
* `eval20` on HB: epoch 0.838, K=10 instrumented 0.639 (Δ −0.199 [−0.678, 0.000]),
  Nemotron 0.646 (Δ −0.192 [−0.558, +0.010]), K=25 0.639.
* The metric "chosen on the fit half" is unstable. Across bootstrap replicates the HA fold picked
  TF latent MSE 1036 times, train loss 390 and boundary jump 223; the HB fold picked replan 940,
  train loss 473 and Action L1 438 (out of 2000).

**Even 25 real rollouts per checkpoint rank the pooled two-seed set worse than "epoch".** The
pooled target is mostly a smooth function of training progress plus episode noise. Epoch captures
the smooth part with no noise.

### 5.4 Epoch-matched pairwise test (the actual cross-seed question)

Primary analysis: pairs with |ΔS50| ≥ 0.06. Only **4 pairs** (epochs 20, 30, 40, 50) qualify, and
seed1 is better in 3 of them.

| picker | correct / 4 | accuracy | 95% CP CI | binom p | picks seed2 |
|---|---|---|---|---|---|
| Action L1 | 2 | 0.50 | [0.07, 0.93] | 1.00 | 0.25 |
| open-loop smoothness | 2 | 0.50 | [0.07, 0.93] | 1.00 | 0.75 |
| replan overlap H=16 | 1 | 0.25 | [0.01, 0.81] | 0.63 | 1.00 |
| PLV-1 smoothness | 1 | 0.25 | [0.01, 0.81] | 0.63 | 0.50 |
| boundary jump H=16 | 2 | 0.50 | [0.07, 0.93] | 1.00 | 0.75 |
| validation loss | 2 | 0.50 | [0.07, 0.93] | 1.00 | 0.75 |
| train loss / support drift | 3 | 0.75 | [0.19, 0.99] | 0.63 | 0.00 / 0.50 |
| teacher-forced latent MSE | 4 | 1.00 | [0.40, 1.00] | 0.125 | 0.25 |
| epoch | ties | 0.50 | — | — | — |
| always seed1 (control) | 3 | 0.75 | | | 0 |
| K=10 instrumented (eval epochs 20, 40 only) | 0 / 2 | 0.00 | [0.00, 0.84] | 0.50 | 1.00 |
| K=10 Nemotron v2 / pad201 / Cosmos v6 / v7 | 0 / 2 (+1 tie) | 0.25 | [0.00, 0.97] | 1.00 | 1.00 |

* On both eval-epoch pairs, the K=10 eval rollouts pointed to seed2 and the 50-episode target
  pointed to seed1:
  * epoch 20: K10 0.5 vs 0.6; S50 0.56 vs 0.48;
  * epoch 40: K10 0.7 vs 0.9; S50 0.86 vs 0.80.

  K=10 cannot resolve differences of 0.06–0.08.
* Even the target disagrees with itself. Of the 4 primary pairs, half B agrees with half A in 2,
  disagrees in 1 and ties in 1.
* Split-half versions have 3 scored pairs each (filter on HA/score on HB, and the reverse). No
  picker is consistently right: Action L1 2/3 then 1/3, open-loop 2/3 then 3/3, replan 3/3 then
  1/3, PLV-1 0/3 then 1/3.
* "Any difference" (|ΔS50| ≥ 0.02): 14 pairs, seed2 better in 9.
  * Action L1 6/14 (0.43 [0.18, 0.71]), open-loop 5/14, PLV-1 6/14;
  * **replan overlap 2/14 (0.14 [0.02, 0.43], p = 0.013, i.e. worse than chance)**;
  * TF latent MSE 13/14 (p = 0.002, Holm-adjusted 0.022);
  * always seed2: 0.64.
* Threshold-free within-pair correlation, Spearman(Δcost, ΔS50) over 20 pairs:
  * Action L1 −0.01 [−0.41, 0.35], open-loop −0.24, PLV-1 −0.27, replan −0.58, TF latent +0.54
    [−0.06, 0.65], K=10 instrumented −0.56 (10 pairs).
  * Against ΔHB the signs flip for most metrics (Action L1 +0.28, open-loop +0.41, replan +0.24,
    TF latent −0.05).
  * So the within-pair signal is dominated by episode noise.

### 5.5 The one "significant" result does not hold up

Teacher-forced latent MSE gets 13/14 any-difference pairs and 4/4 primary pairs. It is not
evidence that PLV separates seeds:

1. **It amounts to one seed-level bit.** From epoch 80 on it always picks seed2. Its late plateau is
   about 0.15% lower for seed2 (≈0.00509 vs ≈0.00510). Seed2 happens to be ahead by 0–2 episodes
   in most late pairs. The circular-shift null (p = 0.00) does not remove this, because the
   late-phase seed preference is exactly the structure that gets shifted away.
2. **It does not replicate on a held-out half.** If the best pairwise metric is chosen on HA pairs,
   TF latent MSE wins (13/15). On HB pairs it then scores 5/9 = 0.56 [0.21, 0.86], below
   always-seed2 on the same pairs (0.67). Choosing on HB picks Action L1 (6/9), which then scores
   6/15 = 0.40 [0.16, 0.68] on HA.
3. **It was the best of 12 metrics.** Its within-run Spearman on can_seed1 (0.49) is among the
   weakest cheap metrics; Action L1 is 0.75.

### 5.6 Where the seed difference actually lives

* Late checkpoints (common10, epoch ≥ 100): seed1 has 12/550 failed episodes, seed2 has 6/550.
  Fisher p = 0.23; this is descriptive only and ignores clustering. By half: seed1 has 10 in HA
  and 2 in HB, seed2 has 5 in HA and 1 in HB.
* Post hoc (the worst of 50 env seeds, uncorrected): **env seed 100003 fails in 14/21 late
  seed1 checkpoints vs 2/11 for seed2** (Fisher p = 0.023). Most of seed1's late deficit is one
  initial condition that seed1 handles badly and systematically. Because 100003 sits in HA, the
  half-split looks like noise in HB.
* None of the offline PLV metrics can see this. 10 of 12 metric columns (8 of 10 distinct
  metrics, because the H=10 columns duplicate H=16), including all five PLV-style ones, have a
  *lower* (better) late mean for seed1. Replan overlap H=16 plateaus at
  0.046 for seed1 vs 0.063 for seed2 at epoch 200 (0.98 vs 1.00 success). The levels of these
  metrics are seed-specific and not calibrated to success across seeds. Within one run they track
  training progress well (within-run Spearman 0.73–0.86), which is why they look good inside a
  run and fail across runs.
* The K=10 eval seeds (200000–200009) do not contain a 100003-like start state. Late seed1 eval
  checkpoints are all 10/10 and seed2 has two 9/10, so K=10 eval would also prefer seed1.

## 6. Interpretation

* **Does PLV help pick one checkpoint across seeds? Not in this pool.** In the pooled ranking, no
  cheap metric, rank-average or "epoch then metric" rule beats epoch. All Δ CIs include 0, and the
  point estimates are −0.04 to +0.01 Spearman and ±0.01 top-1 regret (half an episode). In the
  epoch-matched test there is almost nothing to resolve: 4 informative pairs, all before epoch 60.
  The frequently used PLV metrics score 1–2 of those 4. Over the noisy late pairs, replan overlap
  is significantly worse than chance because of a seed-specific offset.
* **The eval-rollout and VLM route does not help either at K=10.** K=10 success, instrumented or
  judged, ranks worse than epoch (Δ ≈ −0.19 to −0.27). It got both resolvable pairs wrong. Adding
  it to epoch changes nothing (Δ 0.000). Nemotron and Cosmos lose roughly 0.01–0.08 Spearman
  relative to instrumented K=10. On this task the judge is not the bottleneck; K is.
* **Practical recommendation for this regime** (both seeds saturate at 0.96–1.00): any late
  checkpoint of either seed is within 0.04 of the best. The selection decision does not matter at
  the resolution of 50 rollouts. What does differ between seeds is rare, initial-condition-specific
  failure (100003). Finding it needs rollouts that cover many start states, not an offline metric
  and not 10 eval episodes.
* Negative result: **PLV metric levels should not be compared across seeds without
  calibration.** Several PLV metrics have seed-dependent plateaus that point the wrong way.

## 7. Limitations

* **Only 2 seeds, and only 1 task (Can) has 2 seeds.** The cross-seed question is effectively
  tested on one pair of runs. The seed-level comparison is n = 1, and no run-level CI is possible.
  Square has one seed and is not used.
* Can saturates (S ≥ 0.94 from epoch 60). Only 4 of 21 matched pairs are resolvable at 50
  episodes. Late-pool target reliability is negative. A task or budget where seeds differ by more
  than 0.1 at matched epochs could give a different answer.
* Epoch-matched pairs are serially correlated (same two runs). Binomial CIs treat them as
  independent and are therefore optimistic. The circular-shift null preserves autocorrelation but
  cannot separate a seed-level offset from per-checkpoint signal.
* Regret is measured against the noisy max of S50 (or HB), which is upward-biased.
* The block bootstrap over epochs mixes "which phase of training is in the pool" with checkpoint
  noise. The absolute CIs are very wide, so read the paired Δ CIs instead.
* The env seed 100003 analysis is post hoc: the worst of 50 seeds, chosen after looking, p not
  corrected.
* Exploratory, post hoc, and many comparisons: 12 metrics, 4 pools, 3 targets and several pairwise
  variants. Only the Holm-adjusted pairwise p is multiplicity-corrected.
* **Side finding (data hygiene, not fixed here):** in `results/hackathon/baseline_metrics.csv`,
  `boundary_jump_h10 == boundary_jump_h16` and `replan_overlap_h10 == replan_overlap_h16` in all
  93 rows, bit-identical. So the H=10 variants are not independent measurements, and the
  horizon flag may not have been applied in Experiment 1. `val_loss == val_loss_logged` also holds
  everywhere. This should be checked by whoever owns Experiment 1. H=10 rows are omitted from the
  tables above because they duplicate H=16.

## 8. Files

* `scripts/hackathon/explore/y5_cross_seed_selection.py`: analysis.
* `scripts/hackathon/explore/y5_cross_seed_selection_tables.py`: table rendering.
* `results/hackathon/explore/y5_cross_seed_selection.json`: all numbers, input sha256s,
  bootstrap selection frequencies, pair table and tie-handling check.
* `results/hackathon/explore/y5_cross_seed_selection_tables.txt`: full tables, every metric and
  pool and both folds.
* Scratch: `data/outputs/hackathon_explore/y5_cross_seed_selection/`, containing a smoke run and
  `run_main.log`.
