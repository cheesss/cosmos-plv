# z1_tier_triage: how many rollouts does a BAD / OK / GOOD call need? (EXPLORATORY, post-hoc analysis)

> Post-hoc exploratory analysis. Not a pre-registered test. CPU only; no Nebius or GPU calls.
> Results: `results/hackathon/explore/z1_tier_triage.json` (main, quality-dependent judge world)
> Sensitivity: `results/hackathon/explore/z1_tier_triage_iidworld.json` (constant-FPR judge world)
> Figure: `docs/hackathon/explore/z1_tier_triage.png`
> Scripts: `scripts/hackathon/explore/z1_tier_triage.py`, `z1_tier_triage_tables.py`

## One-line conclusion

In this pool, **90% tier accuracy is expensive.** The cause is the checkpoint distribution, not the judge. 39% of checkpoints have a 50-ep success rate of 0.80–0.95, i.e. they are bunched near the 0.9 boundary.

- **perfect detector:** fixed-K plug-in needs K≈30–36. With sequential stopping it takes about 25 rollouts on average.
- **Nemotron:** 0.878 even at K=50. Sequential reaches 0.87 at about 35 rollouts.
- **Cosmos3-Edge:** does not reach 90% at K≤50. Its best is 0.88 with sequential at about 43 rollouts.
- **Looking only at checkpoints far from the boundary (74/93),** perfect needs about 14 and Nemotron about 18. Cosmos does not get there within 25.
- **The epoch/PLV prior has a small effect.** It settles BAD early and cuts mean rollouts by about 5–6%. But it does not change the K at which 90% is reached. The PLV metrics are indistinguishable from each other and from epoch (same conclusion as y1).

## Setup

### Tiers and truth

- Tier definitions: BAD is success rate < 0.5, OK is 0.5–0.9, GOOD is > 0.9. Computed with integer comparisons, so k/K = 0.9 is OK.
- **World A (main analysis, nonparametric).** Uses 93 ckpt × 50 training-time rollouts (seeds 100000–100049).
  - Each replicate draws a random 25/25 seed split. All ckpts share the same split.
  - **Truth is the tier of the 25 held-out episodes.**
  - The K rollouts are drawn without replacement from the other 25 (K = 1..25).
  - The fixed split (100000–24 / 100025–49) is reported separately.
- **World B (secondary, parametric).** Truth is the tier of the 50-ep success rate p_c. Rollouts are Bernoulli(p_c) and K goes up to 50. This setting removes the noise in the 25-ep truth label.
- **World R (validation).** Uses **real VLM judgments** of 10 real recorded rollouts (seeds 200000–9) for each of 33 eval ckpts.
  - The truth tier comes from the 50 training-time episodes. The seeds do not overlap.
  - The same rules are applied to two inputs: (i) the real judgments, (ii) judgments from a simulated judge laid over the same real truth.

### Judge error model

- Fitted on the 330-episode eval JSONL. These seeds do not overlap with A/B.
- FNR_task is given a Beta posterior.
- **quality-dependent FPR:** logit FPR = a_task + b·(s − 0.5). This is the y1 verifier's bbq model; κ sits at its upper bound, so it was set as binomial. Parameters use an N(0, 5²) MAP with a Laplace approximation.
  - Nemotron: b = 4.19 (SE 1.52). Implied FPR(can) is 0.13 at s=0.5, 0.44 at s=0.9, 0.54 at s=1.
  - Cosmos: b = 3.23 (SE 0.94). Implied FPR(square) is 0.48 at s=0.5, 0.77 at s=0.9.
- In the simulation, the world draws the parameters once per replicate. The decision-maker uses either the point fit or the parameter-marginalised likelihood. So a realistic model mismatch is built in.

### Decision rules

- **plug-in:** converts k/K to a tier. This takes the VLM judgments at face value.
- **Bayes:** uses the posterior on a 400-point grid. The likelihood is q(p) = p(1−FNR) + (1−p)·FPR(p). The call is the argmax tier.
  - Prior types:
    - `flat`: uniform in p. Only 10% of the mass goes to GOOD.
    - `tierflat`: 1/3 per tier.
    - `pop`: uses the distribution of the other runs.
    - `kde_<m>`: uses a Beta mixture from the other runs near the metric percentile.
    - `prog_<m>`: learns only P(BAD) from the metric. The remaining mass is split half and half between OK/GOOD.
  - All metric priors are **leave-one-run-out**. They are built only from the 50-ep results of the other two runs. The defaults were fixed in advance and not tuned after seeing the results.
- **sequential:** at K ≥ 1, stop when the max posterior tier probability is at least 0.9 (0.95 is also reported). If it is never reached, decide at K_max.
- **CI:** each replicate consists of a random split (A), a run-stratified checkpoint bootstrap, 1 judge-parameter draw, and 8 episode orderings per ckpt. Percentile 95% CI over 250 replicates; it includes MC noise and is therefore conservative.

## Results

### 1. Fixed-K accuracy (plug-in, 95% CI)

| judge | A: K=5 | A: K=10 | A: K=25 | A: K to reach 90% | B: K=10 | B: K=25 | B: K=50 | B: K to reach 90% (point estimate; bootstrap median [2.5, 97.5]) |
|---|---|---|---|---|---|---|---|---|
| perfect | 0.724 [0.65, 0.79] | 0.758 [0.65, 0.85] | 0.840 [0.75, 0.93] | >25 (only 14% of replicates reach it) | 0.814 [0.77, 0.85] | 0.876 [0.84, 0.91] | **0.925 [0.89, 0.95]** | **36; 30 [18, >50]** |
| Nemotron | 0.722 [0.66, 0.79] | 0.764 [0.67, 0.85] | 0.814 [0.72, 0.90] | >25 | 0.803 [0.73, 0.87] | 0.837 [0.75, 0.91] | 0.878 [0.78, 0.95] | >50 (38% reach it) |
| Cosmos3-Edge | 0.698 [0.62, 0.77] | 0.739 [0.64, 0.83] | 0.771 [0.67, 0.87] | >25 | 0.765 [0.67, 0.84] | 0.769 [0.67, 0.88] | 0.806 [0.67, 0.92] | >50 (8% reach it) |

- **World A ceiling.** The 25-ep truth is itself noisy. On the fixed split the two halves agree on the tier 0.849 of the time. So no judge can reach 90% with K≤25.
- The K needed to reach 80% is:
  - World A: perfect 14, Nemotron 17, Cosmos >25.
  - World B: perfect 9, Nemotron 10, Cosmos 29.
- The fixed split (perfect K=25: 0.850) is close to the random-split mean (0.840). In this analysis the split bias found in y1 is not large.
- **Sawtooth.** The plug-in curve dips at K=10, 20, 30, …, because at K where 0.9K is an integer a GOOD call needs k > 0.9K, i.e. 0 failures. A single boundary rule moves accuracy by several points.

### 2. Checkpoints far from the boundary only (reporting stratification)

- Stratification criterion: ckpts whose 50-ep success rate lies outside [0.45, 0.55] and [0.85, 0.95]. That is 74/93 (can 50, square 24). It was not used in the decision rules.
- Bootstrap median of the K to reach 90% in World A:
  - perfect 14 ([0.88, 1.00] at K=25)
  - Nemotron 18
  - Cosmos does not get there within 25 ([0.74, 0.93] at K=25).
- **Practical budget statement:** for checkpoints not near a boundary, about 15 rollouts with a perfect detector, or about 20 with Nemotron, give a 90% tier. Checkpoints in the 0.85–0.95 band are hard to settle even with 50 rollouts.

### 3. Sequential stopping (P(tier) ≥ 0.9)

| world | rule | accuracy | mean rollouts | BAD / OK / GOOD mean rollouts | fixed-K plug-in at the same mean budget |
|---|---|---|---|---|---|
| B | perfect, flat prior | **0.916 [0.88, 0.95]** | **25.1 [22.9, 27.3]** | 8.9 / 25.1 / 28.8 | 0.88 |
| B | perfect, prog_epoch | 0.845 [0.80, 0.90] | 18.9 | 6.8 / 25.6 / 15.5 | 0.82 |
| B | Nemotron, judge-corrected(marg.), flat | 0.873 [0.78, 0.93] | 35.0 [31.8, 39.0] | 8.5 / 33.4 / 42.7 | – |
| B | Cosmos, judge-corrected(marg.), flat | 0.881 [0.79, 0.94] | 42.9 [39.3, 46.3] | 9.4 / 44.8 / 48.8 | – |
| A | perfect, flat | 0.836 [0.72, 0.93] | 19.0 | 6.9 / 18.3 / 21.9 | 0.83 |

- Flat-prior sequential is well calibrated in World B (0.916 against the 0.9 target). It is about +4pt higher than fixed K at the same mean budget.
- BAD is done in about 7–9 rollouts. The budget goes to the OK/GOOD boundary.
- Cosmos needs about 1.7× the rollouts of perfect for the same accuracy.
- With the `tierflat` and `prog_*` priors, sequential stops sooner. In exchange, OK→GOOD errors increase and accuracy drops to around 0.84. The prior gives GOOD 1/3 of the mass and so pulls near-boundary OK checkpoints to GOOD.

### 4. Failure cases: what gets confused

- **Almost all confusion is between OK and GOOD.** World B, K=25, plug-in, row-normalised:

| call for truth OK | OK | GOOD |
|---|---|---|
| perfect | 0.78 | 0.21 |
| Nemotron | 0.67 | 0.32 |
| Cosmos | 0.54 | 0.46 |

- Recall for truth GOOD is 0.95–0.98. Missing a GOOD is rare; the problem is inflating OK to GOOD.
  - Near-plateau FPs are the cause. Square's 0.80–0.88 plateau gets pushed up to a judged rate of about 0.9.
- **Cosmos also makes many BAD→OK errors.** 0.23 in the BAD row. Square's FPR is high even at low quality (about 0.33 at s=0.3).
- Accuracy by success-rate band (A, K=10, perfect plug-in):

| 50-ep band | accuracy |
|---|---|
| [0, 0.4) | 0.92 |
| [0.6, 0.8) | 0.94 |
| [0.8, 0.9) (27 ckpt) | 0.72 |
| [0.9, 0.95) (9 ckpt) | 0.35 |
| [0.95, 1] | 0.82 |

- The worst checkpoints are square e100–e300 (0.82–0.88), can e35–e95 (0.84–0.90) and can_s2_e20 (0.48).

### 5. Does an epoch/PLV prior reduce rollouts?

- **`prog_<m>` (learns only P(BAD)):**
  - Accuracy from the K=0 prior alone is 0.56 (tierflat is 0.38).
  - It raises K=1 accuracy from 0.58 to 0.62.
  - It reduces mean sequential rollouts. For perfect, A goes from 15.9 to 15.1 and B from 20.1 to 18.9, and BAD rollouts go from 7.3 to 5.1.
  - **The K to reach 90% and the accuracy at K≥10 do not change.** The differences are within the CI.
  - epoch, replan_overlap_h16, val_action_l1 and plv1_smoothness are indistinguishable from each other (0.54–0.56 at K=0). `val_loss_logged` is worse at 0.43.
- **`kde_<m>` (transfers the other runs' whole success-rate distribution):** harmful.
  - It gives truth-OK ckpts a mean prior P(GOOD) of 0.45, and truth-GOOD ckpts a mean P(OK) of 0.44. That is, the information points the wrong way.
  - The cause is that the plateau level differs by task: can is 0.96–1.0, square is 0.80–0.88.
  - In World B at K=25 it gives 0.831 for perfect, lower than plug-in's 0.876.
  - **Lesson:** a cheap metric can transfer the information "training has progressed". But it cannot transfer whether the plateau is above 0.9. That is exactly the information behind the OK/GOOD split, the most expensive part of the tier call.

### 6. Judge correction (Bayes correction)

- At fixed K, quality-aware correction does not beat plug-in. For example:
  - Nemotron, A, K=25: plug-in 0.814, corrected (tierflat) 0.774.
  - Parameter-marginalised + flat prior catches up only at large K. For Cosmos, B, K=50 it gives 0.890, higher than plug-in's 0.806. But at K≤10 it is lower, at 0.45–0.51.
- **Constant-FPR (iid) correction is harmful for Cosmos.** 0.698 at B, K=50. It over-corrects at low quality and under-corrects at the plateau.
- **The choice of prior matters more than the correction.**
  - `flat` almost never calls GOOD at small K: 0.39–0.50 at K≤5.
  - `tierflat` calls near-boundary OK checkpoints GOOD. OK recall is 0.43–0.53 at K=10.
  - Practical recommendation: use plug-in at fixed K. Use the flat-prior posterior for the stopping rule.

### 7. Simulated-judge validation (33 eval checkpoints, leave-one-checkpoint-out)

| judge | model | judged-rate MAE | 90% PI coverage | LOO log score (sum) | FP predicted/actual, s_eval ≥ 0.85 | FP predicted/actual, s_eval < 0.5 |
|---|---|---|---|---|---|---|
| Nemotron | quality-dependent | **0.040** | 0.97 | **−22.5** | 3.8 / 4 | 1.6 / 1 |
| Nemotron | constant | 0.055 | 0.97 | −29.5 | 1.3 / 4 | 7.4 / 1 |
| Cosmos | quality-dependent | **0.049** | 1.00 | **−26.5** | 5.4 / 7 | 5.4 / 6 |
| Cosmos | constant | 0.072 | 0.91 | −34.5 | 2.6 / 7 | 14.1 / 6 |
| Nemotron pad201 | quality-dependent | 0.037 | 1.00 | −21.0 | 4.8 / 5 | 1.8 / 1 |

- **At the level of per-checkpoint judged rate:** the quality-dependent model is consistently better, by 7–10 nats in log score. The constant model under-predicts plateau FPs and over-predicts FPs for failure-heavy ckpts.
- **End-to-end error of predicting the judged rate from training-time 50-ep quality alone:** MAE 0.046. The MAE between the eval 10-ep truth itself and the 50-ep rate is 0.068, so this error is mostly seed differences.
- **It does not hold at the tier level (Figure E).**
  - K=10 accuracy of real-judgment plug-in is Nemotron 0.939 [0.85, 1.00], Cosmos 0.909 [0.82, 1.00].
  - Simulated judgments on the same truth give 0.756 and 0.764. truth_success (perfect) gives 0.758.
  - The real − sim difference is Nemotron **+0.18 [0.10, 0.28]**, Cosmos +0.15 [0.06, 0.23].
- **Cause:**
  - Four can ckpts that are GOOD on 50 ep (can_s1_e60/e80, can_s2_e100/e180) each had 1 failure in eval. The real VLMs judged all of those failures as successes. That is 4/4 real FPs, against a model prediction of about 0.44 each.
  - As a result 9/10 became 10/10. A GOOD call at K=10 needs 10/10, so the plug-in was fixed by luck.
  - For square_s1_e60 (eval truth 10/10, 50-ep 0.68), Nemotron's 2 FNs happened to land it on the correct OK.
- **Interpretation:** in the 33-checkpoint sample, the real VLMs' near-plateau FPs are more extreme than the point fit of FPR(s=0.9) (can 4/6). That direction happened to favour can GOOD.
  - On a task whose plateau is below 0.9 (square), this gain turns into errors.
  - So the simulator reproduces the judged-rate distribution well, but rates tier accuracy **lower** than the real value in this sample.
  - Do not conclude that "the VLM gets tiers right more often than a perfect detector" (33 ckpt, 10 ep, depends on the boundary rule).

### 8. Sensitivity: constant-FPR world

- Switching the world judge to iid makes Nemotron plug-in accuracy at B, K=50 0.915, and it reaches 90% at K=37. Cosmos is still short at 0.882.
- The quality-dependent world is more pessimistic. The difference comes from plateau FPs.

## Interpretation: translated into a real-robot evaluation budget

1. **The cost of a tier call is set by how close the checkpoint is to a boundary.**
   - A ckpt far from the boundary needs 15–20 rollouts with a perfect detector or Nemotron.
   - A plateau near 0.9 gives an unstable call even with 50 rollouts.
   - In practice, sequential stopping is better than fixed K. In this pool, BAD took about 7–9 rollouts and the OK/GOOD boundary took 25–49.
2. **Nemotron is close to a perfect detector.**
   - The accuracy loss is 0.02–0.04 at K=25.
   - The sequential budget for close to 90% accuracy is about 1.4× that of perfect (35 vs 25).
3. **Cosmos3-Edge is unsuitable for OK/GOOD calls.**
   - It has a strong tendency to call failures of high-quality checkpoints successes (FPR 0.6–0.8). As a result 46% of truth OK gets pushed up to GOOD (K=25).
   - It is safe to use only as 1st-pass triage that separates BAD from not-BAD.
4. **Cheap metrics (epoch/PLV) are useful only for filtering out BAD.**
   - They cut mean rollouts by about 5–6%.
   - They do not help with the OK/GOOD split, because the plateau level does not transfer between tasks.
   - Here too, PLV is indistinguishable from epoch.

## Limitations

- **The sample is small.** 3 runs, 2 tasks, 93 ckpts. All GOOD checkpoints are can; square has no GOOD. Accuracy depends heavily on the tier mix, which is GOOD 53%, OK 38%, BAD 10% (A).
- **World A's truth is a 25-ep estimate.** Truth-label noise creates a ceiling. World B is a plug-in world model that takes the 50-ep success rate as the true value, and ckpts sitting on a boundary (0.90, 0.48) are hard to get right in principle.
- **The judge error model rests on thin evidence.** It was fitted on 33 eval ckpts and 67 failures.
  - The high-quality FPR is extrapolated from 9 failures (s_eval ≥ 0.85).
  - The analysed ckpts overlap with the judge-fit ckpts. The seeds do not overlap, but they are the same checkpoints.
  - Square has only one run, so per-task parameters cannot be cross-validated by run.
  - Correlation between failure episodes (κ) was not modelled. In y1, κ sat at its upper bound.
- **World R (real judgments) has 10 rollouts per ckpt, so K ≤ 10.** Eval seeds and training seeds have different success rates (MAE 0.068).
- **The prior settings (bandwidth 0.1, 80/20 and 50/50 mixtures, prog's 50/50 OK/GOOD split) are defaults fixed in advance.** Other settings were not explored.
- The simulation CIs include MC noise. Multiple comparisons (about 46 rules × K) were not corrected.
- **Restating a known open risk.** As the y1 verifier pointed out, the `*_h10` and `*_h16` columns are identical in 93/93 rows. So replan_overlap_h16 must not be read as an H=16-specific metric.

## Rerun

```
cd /path/to/repo
PY="env PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python"
$PY scripts/hackathon/explore/z1_tier_triage.py --reps 250 --orders 8                     # 153 s, exit 0
$PY scripts/hackathon/explore/z1_tier_triage.py --reps 100 --orders 8 --world-judge iid \
    --out results/hackathon/explore/z1_tier_triage_iidworld.json --fig ""                # sensitivity, exit 0
$PY scripts/hackathon/explore/z1_tier_triage_tables.py > data/outputs/hackathon_explore/z1_tier_triage/tables_main.md
```
