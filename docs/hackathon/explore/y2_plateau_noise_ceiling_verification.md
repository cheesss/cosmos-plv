# y2_plateau_noise_ceiling: adversarial verification (EXPLORATORY, post-hoc)

> **Label:** exploratory and post-hoc. This verifies y2, which was itself post hoc (the plateau
> cut was chosen from the learning curves). CPU only. No GPU work, no Nebius calls, no existing
> file modified. Verdict: **partially confirmed.**

**One-line summary:** y2 reproduces exactly (JSON, CSV and trend JSON identical). In the plateau, two 25-rollout halves do not reproduce each other's ranking (+0.02, checkpoint-block CI [−0.16, +0.21]), and there is no robust signal beyond epoch. This qualitative conclusion holds. However, much of the quantitative support is overstated. (1) "epoch touches the noise ceiling (≈0.84)": the CI of the ceiling estimate is [0.55, 0.93], it is biased by about +0.09 in simulation, and on the square run epoch (0.785) exceeds the ceiling (0.713). (2) "a real trend of 2–4 pp per 100 epochs": with seeds resampled it is significant only on can_seed2. y2's permutation test assumes checkpoints are exchangeable within a seed, and this assumption breaks because per-seed failures persist across many epochs. (3) "no per-checkpoint deviation (ceiling 0.14)": it was simply not detected because power is low. The 95% upper bound is 0.40 on average. (4) val_action_l1 is not "the only lead". It is luck of the fixed split, and ≈0 against S50. Conversely, teacher_forced_latent_mse is the only weak candidate that is robust to the split. (5) "plateau selection is nearly meaningless" relies on the lenient Regret@30. By top-1, random choice loses about 2.5 pp to epoch.

## 1. What was checked

The y2 claims (from its structured summary and `docs/hackathon/explore/y2_plateau_noise_ceiling.md`) were tested for
reproduction, arithmetic, leakage, independence and clustering, power, multiplicity, post-hoc
subset choice and overstatement. Two new scripts were written. Both import the y2 module and do
not modify it:

* `scripts/hackathon/explore/y2_plateau_noise_ceiling_verify.py` (V1–V11) → `results/hackathon/explore/y2_plateau_noise_ceiling_verification.json`
* `scripts/hackathon/explore/y2_plateau_noise_ceiling_verify_extra.py` (E1–E4) → `results/hackathon/explore/y2_plateau_noise_ceiling_verification_extra.json`
* scratch: `data/outputs/hackathon_explore/y2_plateau_noise_ceiling_verification/` (rerun JSON/CSV/tables, logs, JSON dump)

## 2. Commands and exit codes

```
cd /path/to/repo
V=data/outputs/hackathon_explore/y2_plateau_noise_ceiling_verification
# (1) full re-run of the main analysis into scratch (177.7 s, 6 workers)                      exit 0
PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/y2_plateau_noise_ceiling.py \
  --boot 600 --splits 400 --perm 2000 --asub 100 --workers 6 --out-json $V/rerun.json --out-csv $V/rerun_table.csv
# (2) trend script, imported with OUT redirected to scratch (its OUT is hard-coded to results/) exit 0
PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python -c "import sys; from pathlib import Path; \
  sys.path.insert(0,'scripts/hackathon/explore'); import y2_plateau_noise_ceiling_trend as t; \
  t.OUT=Path('$V/trend_rerun.json'); t.main()"
# (3) tables from the re-run JSON                                                             exit 0
PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/y2_plateau_noise_ceiling_tables.py $V/rerun.json > $V/rerun_tables.md
# (4) verification analyses (56 s)                                                            exit 0
PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/y2_plateau_noise_ceiling_verify.py
# (5) supplementary checks                                                                    exit 0
PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/y2_plateau_noise_ceiling_verify_extra.py
```

Note: y2's `rerun_command` runs `y2_plateau_noise_ceiling_trend.py` as is, which overwrites
`results/hackathon/explore/y2_plateau_noise_ceiling_trend.json`. The rewrite is deterministic and
the content is identical, but a verifier must redirect it.

## 3. Reproduction, arithmetic, leakage: confirmed

| check | result |
|---|---|
| main JSON re-run vs canonical | identical except `args`/`runtime_s`; `_table.csv` byte-identical |
| trend JSON re-run | byte-identical |
| tables re-run vs `canonical_tables.md` | byte-identical |
| canonical JSON vs scratch `final.json` (circular-block run) | byte-identical (y2's superseded-run note is correct) |
| input hashes (baseline, hybrid f87ba3bd…, 4 judge files) | match the JSON |
| loader checks | 50-rollout means match `baseline_metrics.csv` success for all 93; eval success matches summaries |
| Exp-1 numbers vs S50 | epoch 0.815, replan 0.801, open-loop 0.788, boundary 0.782, Action L1 0.772, PLV-1 0.759, val_loss −0.481; late 0.333 / 0.289 / 0.263 / 0.240 / 0.205: all reproduced |
| `exp_regret` (random tie-break, exact expectation) | matches 4000-draw Monte Carlo on 20 random tied cases, max abs diff 0.0004 |
| headline arithmetic | split-half 0.592, SB 0.723, ceilings 0.845 / 0.758, plateau +0.024, slopes, trend-follower 0.343 / 0.190, partial ceilings 0.137 / 0.099, eval-subset numbers: all match the JSON |
| leakage | no scored signal uses half B. References use A or eval seeds, and the rank-average is unweighted. One mild circularity: the "perfect trend follower" expectation is fitted on all 50 seeds, including B (see §4.4) |

## 4. Where the claims break or are overstated

### 4.1 "Epoch is at the 50-rollout noise ceiling (~0.84)": the ceiling is not a measured quantity

| evidence | numbers |
|---|---|
| per-run ceiling vs epoch (V1) | can_seed1: epoch 0.780 vs ceiling 0.868 (gap 0.09). can_seed2: 0.881 vs 0.953 (gap 0.07). **square: 0.785 > ceiling 0.713**. vs the fixed B half: can_seed1 0.853 > ceiling 0.777, square 0.784 > 0.584. The pooled "0.815 vs 0.845" averages two runs with headroom and one run where the "ceiling" is exceeded |
| checkpoint-sampling CI (V2: circular block bootstrap over checkpoints × random splits) | split-half mean 0.59 → **CI [0.29, 0.76]**, not y2's across-split bracket [0.51, 0.68]. Ceiling vs S50 0.845 → **CI [0.55, 0.93]**. Ceiling vs a 25-half 0.758 → [0.48, 0.87] |
| estimator bias (V6: monotone truth = isotonic fit, i.i.d. Bernoulli 50 rollouts, epoch is by construction the perfect ranking) | estimated ceiling vs S50 **0.877**, true achievable **0.786** (bias +0.09). vs a 25-half: 0.797 est. vs 0.732 true |
| noise model (E1) | the additive two-way model is **rejected in the under-dispersion direction**: square plateau F lower-tail p = 0.022 (rows) / 0.010 (deviations); can_seed1 late 0.024 / 0.035; square late 0.017 / 0.003 |

Interpretation. Epoch's 0.815 is about what a perfect monotone-trend predictor achieves in
simulation (0.786), so "no measurable headroom over epoch on the full pool" stays plausible.
What is not supported is a ceiling of about 0.84 that epoch sits a few hundredths below.

### 4.2 "The only real signal is a weak epoch trend of 2–4 pp / 100 epochs (perm p 0.03 / 0.004 / 0.03)": robust only in can_seed2

y2's permutation test permutes checkpoints within each seed column, so it assumes outcomes are
exchangeable across checkpoints for a fixed seed. The data violate that assumption (V3).
Failures persist over consecutive checkpoints of the same seed. Example: seed 3 fails in 17 of
the last 18 can_seed1 checkpoints (epoch 115–200), all in half A.

| plateau run | lag-1 within-seed residual autocorr (null mean) p | P(fail next \| fail) vs null | Spearman(epoch, S50) | **seed-cluster bootstrap CI** (V4) | y2 perm p |
|---|---|---|---|---|---|
| can_seed1 | 0.094 (−0.035) **p = 0.005** | 0.28 vs 0.18, p = 0.011 | +0.36 | **[−0.37, +0.67]**, 22 % ≤ 0 | 0.029 |
| can_seed2 | −0.111 (−0.072) p = 0.997 | 0 vs 0.02 | +0.66 | [+0.35, +0.78] | 0.004 |
| square_seed1 | 0.014 (−0.048) **p = 0.044** | 0.30 vs 0.26, p = 0.058 | +0.42 | **[−0.29, +0.67]**, 18 % ≤ 0 | 0.029 |

Resampling seeds keeps the persistence. With that, the trend is distinguishable from 0 only in
can_seed2. The trend script's own seed-bootstrap slope CIs agree: [−2.4, +5.4] and [−2.6, +6.5]
pp/100 ep for can_seed1 and square. Square's F_trend p is 0.18.

### 4.3 "No checkpoint-specific deviations; partial-Spearman ceiling beyond epoch averages 0.14": low-power non-detection

| plateau run | F_dev | y2 ceiling (point) | 95 % upper bound of ceiling (V5) | 80 % upper bound |
|---|---|---|---|---|
| can_seed1 | 1.01 | 0.05 | **0.51** (deviation SD ≤ 1.8 pp) | 0.37 |
| can_seed2 | 1.29 | 0.36 | **0.69** (≤ 2.8 pp) | 0.57 |
| square_seed1 | 0.41 | 0.00 | 0.00, degenerate: F = 0.41 has lower-tail p = 0.010, so the noise model is rejected; this is not evidence that deviations are absent | |
| mean | | 0.14 | **0.40** | 0.31 |

Power (V6, plateau). Checkpoint deviations with SD 2 pp would let a perfect metric reach a partial
Spearman of +0.44 beyond epoch vs a 25-half. y2's F_dev test flags them in only **42 %** of runs
(62 % at 3 pp; the baseline rate is 20 % because the isotonic truth is not linear). y2's square
reading "deviations smaller than chance (F p ≈ 0.98)" is better read as model misspecification.
The binomial comparator (5.2 pp) also ignores shared seeds; the interaction-based noise SD is
about 5.0 pp, and the data are still under-dispersed.

### 4.4 "Epoch is at the trend-model ceiling (+0.35 vs expected +0.34)": tautological

The "perfect trend follower" is a linear function of epoch, so it is epoch. Its expected
Spearman comes from a linear fit to the same 50 seeds, B included. The agreement only checks that
the linear-trend model is self-consistent. It cannot show that nothing beyond epoch exists.
That question belongs to §4.3, which has low power.

### 4.5 "val_action_l1 is the only lead (+0.21 plateau / +0.25 late)": refuted; it is a fixed-split artifact

y2 diagnosed executed_smoothness_K10 as a lucky-split artifact by comparing the fixed split with
random splits and with S50. The same test applied to val_action_l1:

| subset | partial \| epoch, fixed split | random-split 2.5–97.5 % | random-split mean (V7) | vs S50 | quadratic-epoch control vs S50 (E3) [block CI] |
|---|---|---|---|---|---|
| plateau | +0.21 | [−0.17, +0.23] | **+0.03** | +0.06 | +0.05 [−0.22, +0.40] |
| late | +0.25 | [−0.16, +0.23] (fixed split above the 97.5 %) | ~+0.03 | **+0.01** | n/a |
| all | +0.27 | [−0.07, +0.26] | +0.09 | +0.10 | +0.05 [−0.15, +0.33] |

In square, val_action_l1 has a partial of +0.29 vs B and −0.01 vs S50. The protocol split favours
it there.

The only signal whose partial beyond epoch stays positive across random splits is
**teacher_forced_latent_mse** (a PLV-1 diagnostic column). y2 does not mention it in its findings.

| subset | fixed | random-split range | vs S50 | block CI (y2) | run-level CI | quadratic-epoch control vs S50 [block CI] |
|---|---|---|---|---|---|---|
| all | +0.35 | [+0.28, +0.51] | +0.42 | [+0.05, +0.55] | [+0.15, +0.55] | +0.25 [−0.03, +0.51] |
| plateau | +0.13 | [+0.03, +0.37] | +0.24 | [−0.19, +0.43] | [−0.06, +0.33] | +0.15 [−0.18, +0.48] |

It is driven mostly by can_seed2, and its rank-average with epoch hurts (Δ −0.10 / −0.15). Given
about 99 partial tests, it is a weak candidate, not a finding. If anything is re-tested
pre-registered, it should be this, not val_action_l1.

### 4.6 "Partial Spearman given epoch is consistent with 0 for all metrics" / "neither PLV metrics nor VLM-judged rollouts add information beyond epoch in this pool"

* The claim is contradicted in the same sentence: val_action_l1 late has block CI [+0.03, +0.61].
  Of 99 partial-given-epoch values in the y2 JSON, 14 have block CIs excluding 0 (9 positive); about 5 are expected by chance.
* On the full pool (eval checkpoints), several rollout signals have positive partials across
  ≥ 97.5 % of random splits: VLM v6 +0.34 [+0.14, +0.48], Nemotron v2 [+0.08, +0.58], pad201
  [+0.14, +0.60], other-half K25 on all 93 [+0.03, +0.31], val_loss [+0.18, +0.33]. These
  **vanish under a quadratic epoch-rank control** (V11: VLM v6 −0.05, Nemotron +0.02, val_loss
  −0.07), so they are curvature artifacts of the saturating curve, not checkpoint-specific
  information. After that correction the headline is fair as "no robust evidence of incremental
  value". "Add no information" is stronger than the data.
* sim_replan_overlap_K10 / sim_boundary_jump_K10 have the same rank order as epoch on the eval
  checkpoints (identical Spearman), so their partials are numerically meaningless. Confirmed.

### 4.7 "Selection among plateau checkpoints barely matters (Regret@30 0–0.02 for all, random included)": depends on a lenient metric

Regret@30 credits a signal with the best checkpoint in its top 30 %, which is 5–9 checkpoints per
run on the plateau. For a single pick (V8, top-1 regret vs random held-out halves, 400 splits):

| subset | epoch | random | val_action_l1 | other half (argmax of 25 rollouts) |
|---|---|---|---|---|
| plateau | 0.027 | **0.052** | 0.027 | **0.063** |
| late | 0.022 | 0.040 | 0.021 | 0.058 |

These are measured against noisy-half maxima, so they inflate the true regret. Trend-implied, the
expected loss of a random plateau pick vs the last checkpoint is 1.5 / 3.1 / 2.4 pp per run, and
that trend is robust only in can_seed2. "Modest (≈1–3 pp)" is supported. "Barely matters" and
"PLV's value is rejecting undertrained checkpoints" are interpretations this study did not test.

### 4.8 Smaller points

* **VLMs vs instrumented (V10).** Direct within-run Spearman of VLM K10 vs instrumented K10 on all
  33 eval checkpoints is 0.78 (v6) / 0.83 (Nemotron), so "VLMs follow instrumented success" over
  the full range is confirmed. y2's evidence for it was indirect: each signal vs B. On the 24
  plateau eval checkpoints it falls to 0.16 / 0.47, and episode error rates (3–7 %) exceed the
  1–2 pp differences to be resolved. On the plateau the VLM is a weak link too, though not the
  binding one.
* **"ΔRegret@30 is 0.000 at every point estimate".** True for the plateau and late full subsets.
  Elsewhere it reaches +0.006 to +0.008 (val_loss on all; some eval subsets).
* **"Partial ceiling is 0 in 2 of 3 runs".** True for late. On the plateau it is 0 only for square
  (can_seed1 is 0.05).
* **Plateau-cut sensitivity (V7).** Split-half means by cut: can ≥ 40 / square ≥ 60: +0.24;
  60 / 90 (y2): +0.01; 80 / 120: −0.08; 100 / 150: −0.17; 140 / 210: +0.06. The "≈0" plateau
  result needs cuts at or after y2's. The epoch-vs-half mean drops from +0.51 to +0.17 over the
  same range.
* **Block length.** ceil(n^(1/3)) is 3–4 checkpoints, while failure persistence spans up to about
  17 checkpoints. For epoch vs B on the plateau the CI barely changes with block length 8 or 12
  ([+0.23, +0.60] vs [+0.26, +0.63]), so this is minor here.
* **The split-half ≈ 0 rests on a few seeds.** For example, late can_seed1 has split-half −0.36
  across random splits. The cause is mostly seed 3 (half A) failing persistently late while B
  failures cluster at epochs 100–110.

## 5. What survives

1. Reproduction is exact; data and arithmetic are correct; there is no B leakage into any scored
   signal; the regret tie-break is implemented correctly.
2. On the plateau and late subsets, the 25-rollout halves do not reproduce a ranking. Split-half
   is +0.02 (checkpoint-block CI [−0.16, +0.21]) and −0.17 ([−0.35, +0.10]). The other-half K25
   reference is about 0. Rollout outcomes at K = 10–25 cannot resolve 1–2 pp differences.
3. No PLV metric, Exp-3 rollout metric or VLM signal shows robust incremental value beyond epoch
   once split luck and curvature are controlled. The executed_smoothness_K10 lucky-split
   diagnosis is correct.
4. On this pool, "PLV does not beat epoch" is uninformative rather than a failure. The framing
   recommendation stands. The precise ceiling numbers should not be quoted next to every Spearman
   without their CI (0.84, block CI [0.55, 0.93]) and the bias caveat.

## 6. Corrected headline

EXPLORATORY/post-hoc, reproduced exactly. On the plateau, two disjoint 25-rollout halves do not
reproduce a checkpoint ranking (split-half +0.02, checkpoint-block CI [−0.16, +0.21]). After
controlling for split luck and curvature, no PLV, rollout or VLM signal shows robust value beyond
epoch, so plateau ranking cannot be evaluated on this pool. The quantitative support is weaker
than reported:

* The full-pool "ceiling ≈ 0.84" has block CI [0.55, 0.93], is biased by about +0.09 in
  simulation, and is exceeded by epoch in square.
* The plateau epoch trend is robust only in can_seed2 once seeds are resampled; within-seed
  failure persistence invalidates y2's permutation p-values.
* "No checkpoint-specific deviations" is a low-power non-detection: the 95 % upper bound on the
  partial-Spearman ceiling averages 0.40.
* val_action_l1 is a fixed-split artifact, about 0 vs S50. teacher_forced_latent_mse is the only
  split-robust weak candidate.

## 7. Limitations of this verification

* The V6 simulation uses an isotonic fit of S50 as truth. That fit over-fits noise, so it gives a
  slightly too-variable truth (simulated plateau split-half 0.21 vs observed 0.02). It also uses
  i.i.d. Bernoulli rollouts without seed persistence, so its bias and power figures are indicative only.
* The F-based upper bounds assume the additive ANOVA noise model, which E1 rejects in some
  run-subsets.
* The seed-cluster bootstrap is a percentile interval over 50 seeds with ties. The quadratic
  epoch-rank control is one simple curvature model. The V11 "isoA" variant is identical to the
  linear partial by construction (monotone transform, same ranks) and is not informative.
* The cause of the persistence and under-dispersion, for example deterministic per-seed rollouts
  or vectorized-env coupling, was not investigated.
* This verification adds its own multiple comparisons, on 3 runs over 2 tasks.
