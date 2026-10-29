# y5_cross_seed_selection: adversarial verification (EXPLORATORY, post-hoc)

> **Label:** exploratory and post-hoc. This note checks the y5 report
> (`docs/hackathon/explore/y5_cross_seed_selection.md`). Nothing in either document was pre-registered.
> CPU only. No GPU, no Nebius calls. No existing file was modified.
> New numbers come from `results/hackathon/explore/y5_cross_seed_selection_verification.json`,
> written by `scripts/hackathon/explore/y5_cross_seed_selection_verification.py`.

**Verdict: partially confirmed.** The main negative result holds, and holds more strongly than
reported: pooling the two Can seeds, no PLV metric beats epoch. Several supporting claims are
wrong or overstated:
* The VLM judges do not get "both pairs wrong". They tie on one of the two pairs.
* Not every Δ-vs-epoch CI includes 0. Several exclude it, all on the unfavourable side.
* Replan overlap is not "significantly worse than chance" after Holm correction.
* The reliability of the late target is about 0, not negative.
* The env-seed-100003 story does not establish a real seed difference.
* The diagnosis of the h10 == h16 duplication is wrong. The horizon flag was applied; the
  duplication is structural.

**One-line summary:** The core conclusion ("even with the two seeds pooled, PLV is no better than epoch") reproduces, and in the fixed-pool bootstrap Action L1 and open-loop are actually significantly worse than epoch. However, several detailed claims are wrong or overstated.
- The VLM ties at e20, so it is not "both of 2 pairs wrong".
- Replan's "significantly worse than chance" is p=0.14 after Holm correction.
- The late reliability is ≈0, not negative.
- The 100003 analysis does not establish a real difference between seeds. Without that seed, common10 late failures are 4 vs 4, and on the eval seeds seed1 is actually better.
- h10==h16 is not a case of the flag not being applied. Because of stride 8 there is only one replan boundary, so the two are structurally identical.

## 1. What was checked

1. **Reproducibility.** I re-ran the y5 script and table renderer to scratch outputs and diffed
   them against the published outputs.
2. **Independent re-derivation.** I wrote my own log, eval and judge parsers and recomputed:
   * the pair table, the pairwise decisions, Holm corrections and the judge agreement;
   * the pooled Spearman under both tie conventions;
   * the split-half reliability over random splits;
   * the env seed 100003 tables;
   * the h10/h16 root cause, from the raw `chunk.csv` files.
3. **Sensitivity of the bootstrap.**
   * Inclusion rates of the y5 non-circular moving-block scheme.
   * Key Δ-vs-epoch CIs recomputed with (a) circular blocks and (b) episode-only resampling on
     a fixed pool, 1000 reps each.
   * A paired CI for judge minus instrumented K=10 Spearman.
4. **Leakage and held-out handling, multiplicity, post-hoc thresholds, denominators and arithmetic.**

## 2. Commands and exit codes

```
cd /path/to/repo
PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/y5_cross_seed_selection.py --n-boot 2000 \
   --out data/outputs/hackathon_explore/y5_cross_seed_selection_verification/rerun.json            # exit 0, 462 s
PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/y5_cross_seed_selection_tables.py \
   data/outputs/hackathon_explore/y5_cross_seed_selection_verification/rerun.json \
   data/outputs/hackathon_explore/y5_cross_seed_selection_verification/rerun_tables.txt           # exit 0
PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/y5_cross_seed_selection_verification.py  # exit 0
```

* **Re-run JSON vs published JSON:** 0 differing leaves, ignoring `created_utc` and `runtime_s`.
  The `baseline_metrics.csv` sha256 is the same (`ae00dc42…`).
* **Re-rendered tables:** byte-identical to `results/hackathon/explore/y5_cross_seed_selection_tables.txt`.
* **Independent parsers:** they reproduce `success` exactly and give the same Y/E arrays as the
  y5 loader (asserted).

## 3. Leakage and held-out handling

* **No leakage found.**
  * Outcome-free scores never touch outcomes.
  * K=25, "best metric on the fit half" and rank-avg(epoch, K=25) are fit on one half and scored
    on the other, both point and per bootstrap rep. Within a rep the halves are disjoint
    episode sets.
  * The pairwise held-out metric choice selects on |ΔHA| ≥ 0.04 pairs and tests on HB pairs (and
    the reverse). The 0.04 threshold, one episode in 25, is not stated in the report's method
    section.
* **Post-hoc subset choices.**
  * The primary pair filter |ΔS50| ≥ 0.06 selects on the target magnitude. This is not leakage
    for a fixed metric, but it is a post-hoc threshold.
  * I checked thresholds 0.04, 0.06 and 0.08 below. The PLV-style result is robust to the choice.
  * The env seed 100003 analysis is post hoc, and the report says so.

## 4. Claim-by-claim

| # | y5 claim | Check | Status |
|---|---|---|---|
| 1 | Pooled `common10` Spearman: epoch 0.826, L1 0.790, rank-avg 0.805, epoch→L1 0.823 | reproduced exactly; midrank convention gives epoch 0.827, the others unchanged | confirmed |
| 2 | "Every delta-vs-epoch CI includes 0" | **false as worded.** Several CIs exclude 0, all unfavourable (see below). With a fixed-pool episode bootstrap, Action L1 (−0.036 [−0.086, −0.000]), rank-avg(epoch, L1) [−0.043, −0.001] and open-loop [−0.078, −0.015] are *significantly worse* than epoch | overstated in PLV's favour; the negative result is stronger |
| 3 | Top-1 regret differs by at most 0.01 | true for PLV, rank-avg and lex scores on `common10`. Exceptions: val loss +0.15, support drift +0.05, K=10 judges +0.013/+0.018 on `eval20` | mostly confirmed |
| 4 | Only 4/21 pairs have \|ΔS50\| ≥ 0.06, all at epochs 20–50; from e60 on, all \|Δ\| ≤ 0.04; finals 0.98 vs 1.00 | reproduced from raw logs | confirmed |
| 5 | Split-half reliability 0.69 (`common10`), −0.19 (late) → late ranking not reproducible | fixed split reproduced. Over 2000 random 25/25 splits: `common10` 0.71 [0.61, 0.80]; late **−0.07 [−0.32, +0.23]** (the fixed HA/HB split is at the 13th percentile) | conclusion confirmed; "negative" overstated, it is ≈ 0 |
| 6 | K=25 HA rollouts 0.669 vs epoch 0.861 on HB, Δ −0.193 [−0.655, −0.014] | reproduced. Circular blocks: [−0.557, −0.001]. Episode-only: [−0.308, −0.063]. **Reverse fold is not significant:** y5 [−0.486, +0.162], circular [−0.412, +0.130], episode-only [−0.225, +0.021] | partially confirmed (one fold only) |
| 7 | Best-on-fit-half cheap metric 0.599; choice unstable (TF latent / replan) | reproduced | confirmed |
| 8 | K=10 `eval20`: instrumented 0.703, Nemotron 0.691, Cosmos 0.621; epoch 0.891; "judge costs 0.01–0.08; limiting factor is K, not the judge" | numbers reproduced, but they depend on the tie convention (see below). Paired judge − instrumented CI includes 0 under both resampling schemes | numbers confirmed; "judge costs 0.01–0.08" not supported; "K is the limit" is consistent |
| 9 | v6 = v7 on all 220 Can episodes; v2 vs pad201 differ in 1 | 0 and 1 differ (the pad201 difference is seed1 e40) | confirmed |
| 10 | Primary pairs: L1 2/4, open-loop 2/4, replan 1/4, PLV-1 1/4, BJ 2/4; always-seed1 3/4 | reproduced. Thresholds: 0.04 gives L1 2/6, open-loop 3/6, replan 1/6, PLV-1 2/6, TF latent 6/6; 0.08 gives L1 1/2, TF latent 2/2 | confirmed and robust to threshold |
| 11 | Headline: "On those 4 pairs PLV metrics are right 1-2 times out of 4" | the outputs of the PLV-1 script itself score teacher-forced latent MSE **4/4** and support drift **3/4**; train loss is 3/4. The body qualifies this as "frequently used" metrics; the headline drops the qualifier | selective wording |
| 12 | "K=10 instrumented **and all VLM judges** pick seed2 on both eval-epoch pairs and are wrong both times" | **false for the VLMs.** At e20, v6/v7 give 0.7/0.7 and v2/pad201 0.6/0.6 (ties). Only at e40 (0.7/0.9) do they pick seed2 wrongly. The instrumented K=10 claim (0.5/0.6 at e20, 0.7/0.9 at e40, both wrong) is correct. The report's own table says "0/2 (+1 tie)" | refuted as worded |
| 13 | Target halves agree on 2 of 4 primary pairs | 2 agree, 1 disagrees (e50), 1 tie (e40) | confirmed |
| 14 | Any-difference (14 pairs, seed2 better in 9): replan 2/14, p = 0.013, "**significantly worse than chance**" | 2/14 and two-sided p 0.013 reproduced. **Holm: 0.142 over 12 columns, 0.116 over 10 distinct metrics**, which is the same correction the report applies to TF latent. Replan picks seed1 in all 8 late (≥ e80) any-difference pairs, so its late record equals always-seed1 (1/8): the same "one seed-level bit" artefact the report uses to dismiss TF latent | overstated / inconsistent |
| 15 | 10 of 12 columns (8 of 10 distinct) have a lower late cost for seed1; replan plateau 0.046 vs 0.063 at e200 | reproduced (0.0461 vs 0.0630) | confirmed |
| 16 | "PLV metrics have seed-dependent plateaus **that point the wrong way**" | "wrong" is relative to a late S50 gap of 12 vs 6 failures out of 550 (Fisher 0.23, ignoring clustering). That gap reverses on the disjoint eval seeds (seed1 0/60 vs seed2 2/60 failures). The plateaus being uncalibrated across seeds is supported; "wrong way" is not | partially confirmed |
| 17 | TF latent 13/14 (Holm 0.022) "amounts to one seed-level bit"; held-out choice fails (13/15 → 5/9 vs always-seed2 0.67; reverse fold L1 6/15) | numbers reproduced. Split by phase: late (≥ e80) 7/8 = exactly always-seed2's 7/8, so that part is a seed bit. **Early (≤ e60) it is 6/6** (4/4 primary), and that cannot be a seed bit because it picks s1 at e20/40/50/60 and s2 at e10/30. Not significant after correction (two-sided 6/6 p = 0.031 before correction). The held-out failure comes from e50/e60, where HA and HB disagree with each other | dismissal half-right: "inconclusive", not "explained away" |
| 18 | TF latent seed2 plateau "about 0.15% lower" | 0.0050918 vs 0.0051027 = **0.21%** lower | minor arithmetic error |
| 19 | Env seed 100003: 14/21 vs 2/11 late checkpoints, Fisher p = 0.023 (uncorrected); "the real seed difference is mostly one start state" | counts reproduced. The seed1 failures form **3 contiguous runs** (e120–125, e140–180, e190–200), so the 21 checkpoints are far from independent. Restricted to `common10`: 8/11 vs 2/11, p = 0.030. **Excluding 100003, `common10` late failures are 4 vs 4.** On the eval seeds, seed1 is better (0/60 vs 2/60). Supported: seed1 has a persistent late failure mode on one start state. Not supported: a "real" seed-level difference | partially confirmed; "real seed difference" overstated |
| 20 | h10 == h16 bit-identical; "the horizon flag was probably not applied" | identical in all 93 rows, and also row-wise in the raw `chunk.csv` (3 runs). **The flag *was* applied.** `chunk_action_horizon` is 10 vs 16, and support drift and curvature differ in 100% of rows. Execution stride is 8, so both H=10 and H=16 make exactly 2 policy calls with **one** replan boundary at step 8, and boundary jump / replan overlap are therefore identical by construction | diagnosis refuted; duplication confirmed |
| 21 | `val_loss == val_loss_logged` everywhere, "should be checked" | both columns read the same training-log `val_loss` field (`checkpoint_adapter.py:138`, `plv_lowdim_checkpoint_metrics.py:1232`) | by construction; not a defect |

Detail for row 2. These Δ-vs-epoch CIs exclude 0, all on the unfavourable side:
* support drift Spearman: [−1.02, −0.22];
* val-loss Δtop-1: [+0.01, +0.24];
* on HB: PLV-1 [−0.36, −0.02], TF latent [−0.68, −0.004] and dynamics uncertainty [−0.43, −0.002].

Detail for row 8. Under the midrank convention used by `cosmos_plv.ranking` and Experiment 1,
K=10 scores are: instrumented 0.760, Nemotron 0.837, Cosmos 0.793. The judges then rank *above*
instrumented. Paired judge − instrumented CIs:

| difference | circular blocks | episode-only |
|---|---|---|
| Nemotron − instrumented | [−0.19, +0.26] | [−0.10, +0.18] |
| Cosmos − instrumented | [−0.34, +0.25] | [−0.19, +0.15] |

## 5. Bootstrap design issue (does not change conclusions)

The y5 moving-block bootstrap is non-circular: block starts are drawn from 0 to nb − block. So
the edge epoch bins are badly under-sampled. Mean copies per replicate in the `common10` scheme
(20,000 draws):

| bin (epoch) | 0 | 10 | 20 | 30–170 | 180 | 190 | 200 |
|---|---|---|---|---|---|---|---|
| mean copies | 0.34 | 0.62 | 0.89 | ≈ 1.17 | 0.84 | 0.56 | **0.28** |

The final checkpoints are the top-1 candidates, and they appear about 4× less often than mid-run
ones. This biases the top-1 regret and Spearman CIs. Re-computed CIs (1000 reps each):

| Δ Spearman vs epoch | y5 (non-circular) | circular blocks | episode-only, fixed pool |
|---|---|---|---|
| `common10` S50, Action L1 | [−0.300, +0.121] | [−0.239, +0.095] | **[−0.086, −0.000]** |
| `common10` S50, rank-avg(epoch, L1) | [−0.159, +0.089] | [−0.157, +0.071] | **[−0.043, −0.001]** |
| `common10` S50, epoch → L1 | [−0.037, +0.013] | [−0.031, +0.009] | [−0.007, +0.001] |
| `common10` S50, open-loop | [−0.248, +0.024] | [−0.264, +0.017] | **[−0.078, −0.015]** |
| `common10` HB, K=25 (HA) | **[−0.655, −0.014]** | **[−0.557, −0.001]** | **[−0.308, −0.063]** |
| `common10` HA, K=25 (HB) | [−0.486, +0.162] | [−0.412, +0.130] | [−0.225, +0.021] |
| `eval20` S50, K=10 instrumented | [−0.688, +0.029] | [−0.687, +0.017] | **[−0.414, −0.036]** |
| `eval20` S50, K=10 Nemotron | [−0.660, +0.049] | [−0.644, +0.043] | **[−0.369, −0.015]** |
| `eval20` S50, K=10 Cosmos v6 | [−0.745, +0.022] | [−0.695, +0.005] | **[−0.404, −0.062]** |

How to read the episode-only column: it holds the set of checkpoints fixed and asks whether
the metric ranks *this* pool worse than epoch. On that question, the PLV metrics and K=10 are
significantly worse. The block columns add the uncertainty of which checkpoints enter the pool.
With only 2 runs, neither scheme can address run-to-run generalization. No scheme produces a
CI in PLV's favour.

## 6. Independence, clustering and multiplicity

* **Pairs are serially correlated.** The binomial CIs and p-values treat the 21 matched epochs
  as independent (y5 says this). The late part of both "significant" pairwise results comes from
  one constant pick per seed: TF latent always picks s2 from e80 on, and replan always picks s1.
  So the effective sample for those results is about 1.
* **Multiplicity is applied unevenly.**
  * Holm is applied to TF latent (0.022) but not to replan (0.142).
  * Roughly 12 metric columns × 4 pools × 3 targets × 4 pairwise variants × 3 thresholds were
    examined. No result in PLV's favour survives correction.
* **Env seed 100003.**
  * The seed was selected by total failure count, which is the margin of the 2×2 table.
    Under exchangeability, Fisher's conditional test is therefore not inflated by that selection.
  * The real problem is dependence between checkpoints: 3 contiguous failure runs, and 21 seed1
    checkpoints spaced 5 epochs apart against 11 for seed2.
  * Naive Bonferroni over 50 seeds gives p = 1.0.

## 7. Interpretation

* **Confirmed.** In the one available two-seed comparison (Can, saturated), there is nothing for
  PLV to resolve, and PLV does not help:
  * Only 4 matched pairs differ by ≥ 0.06.
  * PLV-style metrics get 1–2 of 4 of those pairs.
  * The pooled ranking never favours a PLV metric, rank-average or epoch→metric rule over epoch.
    On a fixed pool, L1 and open-loop are significantly worse.
  * K=10 eval, instrumented or judged, also does not help.
* **Corrections the submission should carry.**
  1. Drop "VLM judges wrong on both pairs". They are 0 right, 1 wrong, 1 tie.
  2. Replace "replan significantly worse than chance" with "replan has a constant late seed1
     preference (Holm p = 0.14)".
  3. Late target reliability is ≈ 0, not negative.
  4. Do not call the 100003 effect a "real seed difference". Seed1 has a persistent failure
     on one start state. Excluding it, the late `common10` failures are 4 vs 4, and the
     disjoint eval seeds favour seed1.
  5. h10 == h16 is structural: execution stride 8 leaves one replan boundary for both horizons.
     It is not a missing flag. The KNOWN_RISKS entry should say "the Experiment-1 H=10
     boundary/replan rows duplicate H=16 by construction". It should also say that the H=16
     boundary/replan metrics come from a single replan event per rollout start.
  6. Judge-vs-instrumented K=10 differences are not resolved, and their sign depends on the
     tie convention. Always state the convention when comparing against other docs' Spearman.
* **TF latent MSE is inconclusive, not explained away.** It is the best picker on the early,
  resolvable pairs (6/6 for e10–e60), which is not significant after correction. Its late
  "accuracy" is exactly always-seed2.

## 8. Limitations of this verification

* Same data and same 2 runs. Nothing here can speak to generalization across seeds.
* The sensitivity bootstraps use 1000 reps (y5 used 2000). The tie-convention check only
  covers point estimates.
* The circular-block and episode-only schemes are alternatives, not ground truth. None models
  run-level variance.
* Post hoc and exploratory, like the work it checks.

## 9. Files

* `scripts/hackathon/explore/y5_cross_seed_selection_verification.py`: independent checks and
  bootstrap sensitivity.
* `results/hackathon/explore/y5_cross_seed_selection_verification.json`: all verification numbers.
* Scratch: `data/outputs/hackathon_explore/y5_cross_seed_selection_verification/`, containing
  `rerun.json`, `rerun.log`, `rerun_tables.txt` and `verify.log`.
