# y1_plv_prior_budget adversarial verification (EXPLORATORY, post-hoc)

**Verdict: partially_confirmed.**

- **Confirmed.** The core negative result holds. On these 3 runs, no PLV prior beats the free latest-epoch prior.
  - Rerunning the original script gave bit-identical results. Max abs diff is 0.0 for both main and splits.
  - An independent implementation that does not use the y1/x1 code (log parser, priors, uniform and TTTS simulators, judge fit) gave the same values within MC error.
  - "beats both" is 0 across 4 levels × 336 cells. **Even against epoch alone, 0 cells win**.
  - `val_loss_logged` is a harmful prior. In full/trunc_2/trunc_3, all 28/28 cells are significantly worse than epoch.
  - No leakage was found. The priors use only outcome-free columns. The judge was fit on eval seeds (200000+), and regret is scored only on the held-out half.
- **Refuted or weakened.**
  1. **The fixed 25/25 split is skewed in the favorable direction.** The held-out half (100025–100049) is easier. So the absolute regret numbers (reference 0.033, 1N 0.046→0.020, epoch top-1 0.000) are about half of what a random split gives. y1 attributed the upward shift of the CIs only to "seed duplication", which is partly wrong.
  2. **The headline "at least 2–8x (TTTS, perfect judge)" is stronger than the numbers.** "At least" comes from truncation at the grid floor, not from a statistical lower bound. In full/TTTS/perfect, the epoch prior reaches the target before no-prior in only 59% of replicates. At thr 0.02 the median ratio is 1x. The CI of the regret difference under the same condition also includes 0.
  3. The denominator of "2–48x" is the 0.25N grid floor for every prior arm. The thresholds (0.01/0.02) lie inside the noise band (0.02–0.03) that y1 itself defined. "Up to 24x/48x" is the maximum over 28 cells, and the re-split gives up to 64x. The magnitude is unstable.
- The **corrected headline** is at the bottom.

## Question

This verification tried to refute the y1 report (`docs/hackathon/explore/y1_plv_prior_budget.md`), results (`results/hackathon/explore/y1_plv_prior_budget{,_splits}.json`) and scripts (`y1_plv_prior_budget.py`, `_tables.py`). The items checked were:

- leakage
- held-out handling
- clustering/independence
- multiple comparisons
- post-hoc subset selection
- denominators
- claims stronger than the numbers
- arithmetic

## Method (all CPU, no Nebius/GPU)

1. **Rerun.** Reran y1 main (seed bootstrap) and `--resample split` into scratch. Compared every row and paired value with the original JSON.
2. **Independent reimplementation.** `scripts/hackathon/explore/y1_plv_prior_budget_verify.py`. It does not import y1/x1.
   - It parses `logs.json.txt` directly.
   - The top-30% priors were built directly from the outcome-free columns of `baseline_metrics.csv`.
   - Uniform and TTTS (β=0.5, Beta(1,1), 25-episode cap per arm) were implemented from scratch and use only the perfect judge. R=400 on the original split.
   - Computed **per-run** paired Δ in two ways: seed bootstrap 200 × 20 reps and random re-split 200 × 20 reps. LORO (leave one run out) was computed as well.
   - Refit the beta-binomial judge models (bb, bbq) with continuous κ.
   - Mechanically checked the claimed numbers against the y1 JSON. Targets: the beats-both counts, the range of Δ vs epoch, the reach-budget ratios, and the number of point estimates outside their CI.
3. **Split representativeness check.** `scripts/hackathon/explore/y1_plv_prior_budget_verify_split.py`.
   - Compared the success rates of the two halves.
   - Compared the regret of the reference (perfect judge, no prior, 25N → pool-best) with its distribution over 5000 random 25/25 splits.
   - Compared the 0-rollout latest-epoch pick with pool-best, paired, over 2000 random splits.
   - Computed the Spearman between each PLV metric and epoch.

### Commands and exit codes

```
cd /path/to/repo
PY="env PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python"
S=data/outputs/hackathon_explore/y1_plv_prior_budget_verification
$PY scripts/hackathon/explore/y1_plv_prior_budget.py --workers 10 --out $S/rerun_main.json --fig ""                       # exit 0 (167 s)
$PY scripts/hackathon/explore/y1_plv_prior_budget.py --workers 10 --resample split --out $S/rerun_splits.json --fig ""    # exit 0 (184 s)
$PY scripts/hackathon/explore/y1_plv_prior_budget_tables.py $S/rerun_main.json > $S/rerun_tables_main.md                  # exit 0
$PY scripts/hackathon/explore/y1_plv_prior_budget_tables.py --headline $S/rerun_main.json > $S/rerun_headline_main.md     # exit 0
VERIFY_OUT=$S/smoke.json VERIFY_NB=4 VERIFY_R0=40 $PY scripts/hackathon/explore/y1_plv_prior_budget_verify.py             # exit 0 (smoke)
$PY scripts/hackathon/explore/y1_plv_prior_budget_verify.py          # exit 0 (139 s) -> results/hackathon/explore/y1_plv_prior_budget_verification.json
$PY scripts/hackathon/explore/y1_plv_prior_budget_verify_split.py    # exit 0        -> results/hackathon/explore/y1_plv_prior_budget_verification_split.json
```

## Results

### 1. Reproducibility and independent reimplementation: confirmed

- **Rerun.** Identical to the original JSON. Max abs diff is 0.0 for rows (mean, CI) and paired (avg point, CI), and the judge fit is also identical. This holds for both main and splits.
- **Data alignment.** The difference between CSV `success` and the 50-episode log mean is 0.0. The epochs also match.
- **Independent simulator (perfect judge, original split).** 4 levels × 6 priors × 2 strategies = 48 curves. Most agree within 1–2× the MC SE.
  - The largest difference over the whole grid is 0.013. It comes from the high-variance random/none arms.
  - Representative cells are in the table below.

| level | prior | strategy | 1N independent / y1 | avg 0.5–8N independent / y1 | 25N independent / y1 |
|---|---|---|---|---|---|
| full | none | TTTS | 0.048 / 0.046 | 0.042 / 0.042 | 0.033 / 0.033 |
| full | epoch | TTTS | 0.020 / 0.020 | 0.023 / 0.023 | 0.027 / 0.027 |
| full | replan_overlap_h16 | TTTS | 0.019 / 0.019 | 0.023 / 0.023 | 0.027 / 0.027 |
| full | val_loss_logged | TTTS | 0.103 / 0.103 | 0.105 / 0.105 | 0.106 / 0.107 |
| trunc_1 | epoch | TTTS | 0.049 / 0.048 | 0.052 / 0.051 | 0.040 / 0.040 |
| trunc_1 | val_action_l1 | TTTS | 0.040 / 0.040 | 0.035 / 0.035 | 0.014 / 0.014 |

### 2. PLV ≈ epoch (negative result): confirmed, but the breadth of the evidence is overstated

| level | cells | beats both (avg / any mult) | **beats epoch alone** | priors significantly worse than epoch | CI envelope of Δ vs epoch avg for non-vll priors |
|---|---|---|---|---|---|
| full | 336 | 0 / 0 | **0** | val_loss_logged (28/28) | [−0.012, +0.019] |
| trunc_1 | 336 | 0 / 0 | **0** | plv1_smoothness (SH 5 cells, some mult) | [−0.068, +0.090] |
| trunc_2 | 336 | 0 / 0 | **0** | val_loss_logged | [−0.047, +0.043] |
| trunc_3 | 336 | 0 / 0 | **0** | val_loss_logged | [−0.023, +0.044] |

The re-split JSON agrees: 0 cells beat epoch. As priors worse than epoch, it additionally shows open_loop, plv1 and val_action_l1 in trunc_1, and val_action_l1 in trunc_2.

- **Subset overlap "85–100%".** Confirmed. The per-run minimum is 84.6% (val_action_l1 and plv1 on can_seed1, 11/13). Pooled over runs it is 86.7–100%.
- **The "336 cells" are not 336 independent pieces of evidence.**
  - In full, for 5 of the 12 PLV priors the top-30% set equals the epoch set in all 3 runs: replan_overlap_h16, boundary_jump_h16, epoch+replan, epoch+open_loop and epoch+boundary. Apart from MC noise, the Δ of these cells has to be 0.
  - replan_overlap_h16 ranks the same as epoch on can_seed1 (Spearman 1.000). plv1_smoothness ranks the same as epoch on square (1.000).
  - replan_h16 and boundary_h16 have a within-run Spearman of 0.99.
  - In the upstream data, the `*_h10` and `*_h16` columns hold identical values in 93/93 rows. This is not a y1 error, but the Experiment 1 text that distinguishes H=10 from H=16 needs to be checked.
  - Effectively there are only about 3–4 distinct priors.
- **Correlation of the PLV metrics with epoch.** Sign-aligned Spearman is 0.86–1.00 (val_loss_logged −0.62 to −0.74). The rank-partial correlation controlling for epoch is −0.20 to +0.19 (descriptive statistics, not used for selection). This supports the reading "PLV correlates with success through epoch".
- **Per-run paired Δ (independent implementation, TTTS, perfect, avg 0.5–8N).**

| level | comparison | can_seed1 | can_seed2 | square_seed1 | 3-run mean (bootstrap) | 3-run mean (re-split, fraction with Δ<0) |
|---|---|---|---|---|---|---|
| full | replan_h16 − epoch | [−0.005, 0.005] | 0 (identical sets) | [−0.010, 0.013] | [−0.004, 0.005] | [−0.004, 0.004], 0.49 |
| full | val_action_l1 − epoch | [−0.008, 0.050] | 0 | [−0.016, 0.019] | [−0.005, 0.021] | [−0.011, 0.019], 0.49 |
| trunc_1 | val_action_l1 − epoch | [−0.053, 0.091] | [−0.011, 0.012] | [−0.224, 0.095] | [−0.074, 0.031] | [−0.009, 0.042], **0.08** |

- **The "split artifact" claim for trunc_1 val_action_l1.** Confirmed.
  - In the independent re-split as well, val_action_l1 beats epoch in 8% of splits. y1 reported 8.5%. The median of the 3-run mean is +0.016, i.e. on the side of being worse than epoch.
  - The mechanism was confirmed. In square trunc_1 the epoch subset is {70, 80, 90}. Of these, e80 is the pool best at 22/25 but the lowest on held-out at 18/25. Over 50 episodes it is 0.80 vs 0.78 for e70, effectively the same.
  - The val_action_l1 subset {60, 70, 90} leaves out e80, so it gains only on the original split.
- **Power.** In trunc_1 the PLV−epoch CI width is ±0.05–0.07. This is the same size as the epoch−none effect (about −0.03 to −0.05). So in trunc_1, "PLV adds nothing" does not show equivalence; it means that no difference was detected.
- **Unfavorable results missing from the doc.**
  - In trunc_1, plv1_smoothness is significantly worse than epoch in 5 SH cells.
  - In trunc_2, val_action_l1 needs 1–6N to reach thr 0.01 across all judges and strategies. The doc said "at most 4N".
  - Because of multiple comparisons, these should not be cited on their own either.

### 3. Prior vs no prior (budget savings): direction confirmed, magnitude and "at least" overstated

**(a) Representativeness of the fixed split.** This is the largest correction.

| item | original split (pool 100000–24 / held-out 100025–49) | random 25/25 splits |
|---|---|---|
| late-checkpoint success rate, pool half vs held-out half | can1 0.962 / 0.992, can2 0.982 / 0.996, square 0.812 / 0.865 | (the held-out half is consistently easier) |
| reference regret (perfect, no prior, 25N pool-best) | **0.033** | mean 0.065, median 0.062, 95% [0.032, 0.107]. **The original split sits at the 3.3rd percentile (low end)**. Swapping the two halves gives 0.081 |
| y1 re-split JSON, full level | For 1369/5460 row×mult, the point estimate lies **below** the re-split 2.5% lower bound. 0 lie above it | |
| 0-rollout latest-epoch pick regret | 0.000 (0 in all 3 runs) | mean 0.027 [0, 0.067]. The last epoch of square has regret 0 in only 23% of splits |
| latest-epoch − pool-best (25N), paired | −0.033 | **−0.038 [−0.088, −0.001]**. latest-epoch is better in 98.5% of splits |

- y1 explained why the seed-bootstrap CIs sit above the point estimates as "seed duplication weakens the pool and adds noise to H".
- But the same shift also appears in the re-split, which has no duplication. The original split itself is an outlier on the low-regret side.
- Some point estimates even lie outside their own bootstrap CI (e.g. cosmos_bb epoch 1N 0.015, CI [0.016, 0.060]). This is the same symptom.
- **The qualitative conclusion holds under the re-split.** "The 0-rollout epoch pick beats the 25N full-pool pick" also holds in 98.5% of splits.
- But **the absolute numbers are about 2x optimistic.** Values such as 0.046 → 0.020 and 0.000 < 0.033 should not be cited.

**(b) "At least 2–8x (TTTS, perfect judge)".**

| full / TTTS / perfect | thr 0.0 | thr 0.01 (headline) | thr 0.02 |
|---|---|---|---|
| point-estimate reach budget none / epoch | 20N / 0.25N | 2N / 0.25N | 0.5N / 0.25N |
| bootstrap ratio median [p10, p90] | 4 [1, 48] | **2 [1, 24]** | **1 [1, 8]** |
| fraction of replicates where epoch reached the target strictly first | 0.68 | **0.59** | 0.47 |
| re-split ratio median, fraction reaching first | | 2 [1, 12], 0.56 | 1 [1, 4], 0.44 |

- The 2N point estimate sits on a boundary. At 1.5N the regret of none is 0.0439 and the target is 0.0434. It narrowly missed by 0.0005, and as a result the point-estimate ratio became 8x.
- The regret difference under the same condition is not significant either.
  - epoch − none, TTTS, perfect, full: 1N is −0.026 [−0.069, 0.004] and avg is −0.019 [−0.054, 0.0095].
  - In full, only 2 of the 28 judge×strategy cells have a CI that includes 0 (perfect TTTS and perfect Thompson). These are exactly the conditions the headline cites.
  - In the independent per-run bootstrap, square is [−0.110, +0.063].
  - In LORO, leaving out can_seed1 or can_seed2 makes the CI include 0. Only leaving out square makes it significant, at [−0.035, −0.008].
  - The re-split 3-run mean is significant at [−0.040, −0.004]. In conclusion, **with the perfect judge and full history it is borderline**.
- With a noisy judge or truncated history the effect is robust.
  - trunc_1, cosmos_bb, 1N: epoch − none −0.098 [−0.195, −0.024], epoch − random −0.109 [−0.198, −0.021]. Both were reproduced.
  - Under the re-split, the epoch − none CI excludes 0 in 28/28 cells of trunc_1 and trunc_2.

**(c) On the size of the ratios.**

- Every good prior already reaches the target at 0.25N (the grid floor). So the "none/epoch ratio" is effectively "the reach budget of none ÷ 0.25".
  - If the grid had started at 0.1N, every ratio would be 2.5 times larger. Adding a 0N point would make them infinite.
  - The meaningful quantity is not the ratio but the reach budget of none (2–>25N).
- The thresholds 0.01 and 0.02 are smaller than the 0.02–0.03 that y1 called "measurement noise". 1 held-out episode is worth 0.04 per run and about 0.013 in the 3-run mean. The headline ratios are ultimately decided by whether the target is reached inside the noise band.
- "Up to 24x (TTTS)/48x (uniform)" is the maximum over 7 judges × 4 levels, i.e. 28 cells. Under the re-split, uniform goes up to 64x. This magnitude is not a stable estimate.

### 4. Judge error model: confirmed (interpretation refined)

| judge | bb κ MLE, y1 grid / independent continuous | LR vs binomial (boundary-corrected p) | bbq slope b (y1 SE) | bbq κ | quality slope LR (binomial) |
|---|---|---|---|---|---|
| Nemotron v2 | 1.6 / 1.42 | 4.66 / 4.67 (p≈0.015) | 4.77 (1.81) | → upper bound (1e4) | 12.9 |
| Cosmos3-Edge v6 | 3.5 / 4.0 | 3.29 / 3.31 (p≈0.034) | 3.35 (0.97) | → upper bound | 15.4 |

- The phrase "FPs cluster within a checkpoint" is inaccurate. What is actually there is heterogeneity between checkpoints.
  - At checkpoints with 2 or fewer failures, FPs are Nemotron 5/13 and Cosmos 9/13.
  - At checkpoints with 5 or more failures (mostly epoch 0), they are Nemotron 2/41 and Cosmos 8/41.
  - y1 also wrote "mostly explained by quality", so this agrees with its conclusion.
- That the checkpoints are nested within 3 runs is not modeled.
- The FPR at s=0.9 (0.40–0.78) is extrapolated from a few checkpoints with only 1–2 failures.

### 5. Other claims

- "`val_loss_logged` top-30% is epochs 20–110": confirmed (per run 20–95, 20–80, 20–110).
- "Δ vs random +0.065 [0.005, 0.121]": confirmed (TTTS, perfect, avg). Across all judges it is +0.060 to +0.065 and the CI excludes 0.
- "21 of 41 on can_seed1 tie at held-out 1.0": confirmed.
- "square epoch prior 0.045 (0.25N) → 0.080 (25N)": confirmed. The pool best of full square is e260 (23/25), but its held-out score is 21/25.
- "PLV vs epoch crossing median ratio 1 [1, 1], reaching first 1–7%": confirmed (full, at most 6.5%/7%).
- "post-hoc best PLV per judge Δ vs epoch ±0.0003, CI ±0.006–0.010": confirmed (full).

## Interpretation

1. **What holds.** In this pool, the PLV top-30% priors are nearly the same sets as the latest-epoch prior. They do not beat epoch in any cell. So "PLV saves rollouts" cannot be put forward as a claim distinct from epoch. `val_loss_logged` should not be used as a prior. These two conclusions are robust across the rerun, the independent reimplementation and the re-split.
2. **What weakens.** The claim that "screen first" beats no-prior is robust with noisy judges and truncated history. With the perfect judge and full history it is borderline (the seed-bootstrap CI includes 0, and it collapses when one run is left out). The headline "at least 2–8x" is a lower bound produced by grid truncation. Under that condition, 41% of replicates show no savings (ratio 1).
3. **Caution when citing numbers.** Because the held-out half of the fixed split is easier, every absolute regret came out low (the reference is in the bottom 3% of random splits). In the submission it is safe to cite only relative comparisons (paired Δ, re-split ratios). Absolute values should be shown together with the re-split mean.

## Limitations (of this verification)

- The independent simulator covers only uniform and TTTS with the perfect judge. The noisy-judge results were verified only through rerun agreement and JSON checks.
- The per-run bootstrap uses 20 reps per replicate, so MC noise is mixed into the intervals. With only 3 runs, LORO is only a sensitivity check.
- The correlation descriptive statistics (partial correlation controlling for epoch) used the full 50-episode results. They were not used for selection or evaluation.
- The `*_h10 == *_h16` identity is an upstream (Experiment 1) observation. Its cause was not investigated.
- All of this is post-hoc exploratory analysis. It is not a preregistered test.

## Corrected headline

EXPLORATORY / post-hoc. The results were confirmed by bit-level reproduction and independent reimplementation.

- On these 3 nearly monotone runs, the free latest-epoch top-30% prior is as good as any PLV prior. PLV priors beat epoch in 0 of 336 cells per level, and 84.6–100% of the PLV top-30% sets are the same as the epoch set. `val_loss_logged` is harmful.
- Screening first with a prior is clearly better than no-prior with noisy VLM judges and truncated history. With the perfect judge and full history it is borderline: the TTTS 1N Δ is −0.026 [−0.069, 0.004], epoch reaches the target first in 59% of replicates, and the median ratio at thr 0.02 is 1x.
- The "2–48x" ratios are set by the 0.25N grid floor and by thresholds that lie inside the noise band.
- The absolute regrets come from a favorable fixed split (the reference is in the bottom 3% of random splits) and are about 2x optimistic.

## Files

- Verification scripts: `scripts/hackathon/explore/y1_plv_prior_budget_verify.py`, `scripts/hackathon/explore/y1_plv_prior_budget_verify_split.py`
- Results: `results/hackathon/explore/y1_plv_prior_budget_verification.json`, `results/hackathon/explore/y1_plv_prior_budget_verification_split.json`
- Scratch (rerun JSONs, logs, tables): `data/outputs/hackathon_explore/y1_plv_prior_budget_verification/`
