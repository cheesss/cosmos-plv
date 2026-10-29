# z1_tier_triage verification (adversarial verification, EXPLORATORY / post-hoc)

> **Verdict: partially_confirmed.** The numbers reproduce. However, some of the headline wording is stronger than the numbers, and some figures are artifacts of MC noise or tie-breaking.
> CPU only; no Nebius or GPU calls. No existing files were modified.
> - Scripts: `scripts/hackathon/explore/z1_tier_triage_verify.py`, `z1_tier_triage_verify_frontier.py`
> - Results: `results/hackathon/explore/z1_tier_triage_verification.json`, `z1_tier_triage_verification_frontier.json`
> - Rerun outputs: `data/outputs/hackathon_explore/z1_tier_triage_verification/`

## 0. User question: are the Nemotron and Cosmos outputs saved?

**Yes. I checked directly.** The originals are in `data/outputs/hackathon_cosmos/`, and copies of the same files are in `results/hackathon/raw_model_outputs/`. The 7 copies are byte-identical to the originals (`cmp`). Each file is described in `results/hackathon/raw_model_outputs/README.md`.

| File | judge | Rows | Notes |
|---|---|---|---|
| `eval_nebius_v2.jsonl` | Nemotron-3.5-Super-VL (Nebius), full video | 330 (33 ckpt × 10) | raw_text 330/330, request_id 330/330 |
| `eval_nebius_v2_pad201.jsonl` | Same as above, length-matched clip | 330 | raw_text 330/330, request_id 330/330 |
| `eval_local_v6.jsonl` | Cosmos3-Edge (local), last frame | 330 | raw_text 330/330. Local run, so no request_id |
| `eval_local_v7.jsonl` | Cosmos3-Edge, 8 keyframes | 330 | per-frame `track.frames[*].raw_text` |
| `dev_prompt_selection.jsonl` | Every prompt variant tried on dev | 228 | dev 36 episodes |
| `dev_nebius_v2_pad201.jsonl` | Nemotron pad201, dev | 36 | |
| `dev_infra_failures.jsonl` | Rows lost to CUDA OOM | 23 | Not model outputs |
| `results/hackathon/nebius_calls.jsonl` | Nebius call log | 772 | |
| `results/hackathon/explore/nebius_calls_explore.jsonl` | Nebius call log from the exploration phase | 2380 | |

- Each row has `raw_text`, the parsed `cosmos_*` fields, the `truth_*` labels, latency and token counts, and the model id.
- There are also files z1 does not mention. `data/outputs/hackathon_cosmos/*.orig.jsonl` are the originals from before relabeling with the settled height (eval v6/v2/v7 have 150–152 rows).
- The videos are under the git-ignored `data/outputs/`.

## 1. Reproduction

- Reran `z1_tier_triage.py --reps 250 --orders 8`. The result has 0 numeric differences from the existing `z1_tier_triage.json`. It took 102 s and exited 0.
- The iid sensitivity run (`--reps 100 --world-judge iid`) also has 0 differences from the existing `z1_tier_triage_iidworld.json`.
- Independently cross-checked against values computed directly, without MC. World B was computed as binomial sums and World A with the hypergeometric distribution; the sequential rule was computed by dynamic programming over (K, k).

| Item | z1 (MC) | Verification (exact) |
|---|---|---|
| B perfect plug-in, K=10 / 25 / 50 | 0.814 / 0.876 / 0.925 | 0.813 / 0.876 / 0.925 |
| A perfect plug-in, K=10 / 25 | 0.758 / 0.840 | 0.757 / 0.840 |
| B sequential, flat prior, P≥0.9 | 0.916 @ 25.1 | 0.9155 @ 25.07 |
| Same rule, mean rollouts per tier (BAD / OK / GOOD) | 8.9 / 25.1 / 28.8 | 9.0 / 25.1 / 28.8 |
| B Nemotron, qmarg-flat sequential | 0.873 @ 35.0 | 0.876 @ 34.9 |
| B Cosmos, qmarg-flat sequential | 0.881 @ 42.9 | 0.883 @ 42.9 |
| A sequential, flat prior | 0.836 @ 19.0 | 0.836 @ 19.0 |
| Share of checkpoints in the 0.80–0.95 band | 39% | 36/93 = 38.7% |

These numbers are themselves correct. The problem lies in how they were interpreted and worded, as set out below.

## 2. Verdict per claim

| # | z1 claim | Verdict | Evidence |
|---|---|---|---|
| 1 | A perfect detector needs about 30–36 at fixed K (bootstrap median 30 [18, >50], reached in 96%) | **Partially confirmed; the lower end is too low** | The exact curve first exceeds 0.9 at K=36 (0.901). But it falls back below 0.9 at K=41–42, and **stays at or above 0.9 from K≥43 on**. K=30 is 0.899. The median of the noise-free ckpt bootstrap is **35 [25, 50]**, and 100% of replicates reach it within 50. z1's "30 [18, >50]" is a value in which the 'first K that crosses' was pulled earlier by the MC noise (8 orders) included in each replicate. World A's "14% of replicates reach it" is also 0.9% by exact computation |
| 2 | Nemotron is at 0.878 even at K=50 and does not reach 90% within 50 | **Depends on the judge model (borderline)** | Over 1500 parameter draws the mean is 0.882, the median 0.888 and the 5–95% range [0.82, 0.92]. 35% of draws are at or above 0.9 at K=50. Under other judge fits: point fit 0.897. Fit leaving out one can run 0.908 (reached at K=48) or 0.895. Fit with the FPR covariate replaced by the 50-ep success rate 0.907 (reached at K=40). iid world 0.915 (first crosses at K=37 but drops below again at 41–44). In short: 'at K=50 it is 0.88–0.91, and 90% is borderline' |
| 3 | Cosmos does not reach 90% at K≤50 | **Confirmed** | Falls short in every variant. point fit 0.823, run-excluded fit 0.837/0.844, covariate p50 fit 0.850, iid world 0.880. Only 4.5% of parameter draws are at or above 0.9 |
| 4 | The cost is high because of the 39% of checkpoints near the boundary, not because of the judge | **Overstated** | True for perfect and Nemotron. For Cosmos, the judge itself is a large factor. At K=50 perfect is 0.925 and Cosmos 0.806, and the rate at which truth OK is judged GOOD is 0.21 vs 0.46 |
| 5 | Looking only at the 74 far from the boundary, perfect needs about 14 and Nemotron about 18 | **Numbers confirmed, inappropriate as a practical statement** | The point minimum K for A perfect is 14. But the stratification is defined by the 50-ep **outcome**, and these 50 episodes contain both the truth and the K rollouts. The value changes a lot with the stratification width. Narrowing the excluded band to [0.48,0.52]∪[0.88,0.92] (82 checkpoints) gives 20; widening it to [0.4,0.6]∪[0.8,1.0) (28 checkpoints) gives 6. Without knowing the outcome you cannot tell which checkpoints fall in this stratum, so it cannot be used as a budget rule |
| 6 | sequential is about 4pt higher than fixed K at the same budget | **Confirmed only in World B** | In B, the paired comparison with K=25 plug-in gives **+3.9pt [2.9, 4.9]**. **In World A it is +0.6pt [−0.7, +1.9], so there is no gain.** In B the rollouts are generated from the same Bernoulli(p) as the decision model, which favors sequential |
| 7 | The epoch/PLV prior cuts sequential mean rollouts by 5–6% | **Numbers correct, but the baseline of the comparison is a weak rule** | See section 3 below. prog (on a tierflat base) and tierflat are 4–6pt below the flat-prior frontier in B. They are also on the lower side of fixed-K plug-in at the same budget, by 2–3pt (the CI just includes 0). On a flat base, the prior does not improve the frontier |
| 8 | The prior alone gives 0.56 at K=0 (tierflat 0.38), and val_loss_logged is worse at 0.43 | **Inflated by tie-breaking** | The prog prior puts exactly equal mass on OK and GOOD for 79/93 ckpt (50/50 by design). argmax calls about 40% of these GOOD because of floating-point noise. tierflat calls all of the same ties OK. With ties broken consistently, prog's gain is not +18pt but +6.5pt (tie→OK) to +10.8pt (tie→GOOD) (against 50-ep truth). val_loss_logged is tied for 93/93 and never calls BAD |
| 9 | The quality-dependent FPR model beats the constant model | **Confirmed (also under a stronger test)** | Fitting with one of the two can runs left out, the q model is still better. The log score is −10.2 vs −14.6 for Nemotron and −11.3 vs −16.9 for Cosmos. The MAE is 0.035 vs 0.044 and 0.039 vs 0.055. square has only one run, so this test cannot be done there |
| 10 | Real-VLM tier accuracy is higher than in simulation because of 'chance' FPs on the 4 can GOOD checkpoints | **Numbers confirmed; the 'chance' interpretation is not established** | See section 4 below |
| 11 | The confusion is almost entirely OK→GOOD. GOOD recall is 0.95–0.98, and Cosmos BAD→OK is 0.23 | Confirmed | All numbers match the json. The square Cosmos FPR is 0.33 at s=0.3 |
| 12 | The kde prior hurts (0.831 at B K=25, plug-in 0.876) | Confirmed | |
| 13 | The constant-FPR correction hurts Cosmos (0.698). The marginalised flat correction gives 0.890 | Confirmed | |

## 3. epoch/PLV prior: no effect when viewed as a frontier

z1 compared prog_epoch (on a tierflat base) with tierflat and reported a '5–6% saving'. In an exact paired comparison the numbers are right: B is −6.3% (mean rollouts −1.27 [−1.55, −1.02], accuracy +0.4pt) and A is −5.6%. But both rules are inferior.

- **Both rules lie below the frontier of flat-prior sequential.** At the same mean rollouts, B is −4 to −6pt and A is −3 to −4pt (thresholds 0.85–0.975).
- **Both rules are also on the lower side of fixed-K plug-in (the CI just includes 0).**
  - prog_epoch sequential gives 0.845 at 18.9 rollouts. plug-in at the same budget (K=19) gives 0.867. The paired difference is −2.2pt [−5.1, +0.6].
  - tierflat sequential gives 0.841 at 20.1. K=20 plug-in gives 0.871, and the difference is −3.0pt [−6.3, +0.1].
  - The "fixed-K plug-in at the same mean budget" column (0.82) in the z1 md table is mislabeled. The code (`fixed_K_acc_at_same_mean_budget`) interpolates **the fixed-K curve of the same Bayes rule**, not plug-in.
- **Adding only P(BAD) to a flat base (verifier variant) just moves along the same frontier.**
  - In B, mean rollouts are −18% (25.1→20.4) but accuracy is −1.7pt [−2.7, −0.8]. A is also −17% and −1.7pt.
  - The gain over the flat frontier is at most +0.7pt across the whole threshold range 0.85–0.975. That is, it has the same effect as lowering the threshold, and the prior does not improve the frontier.
  - Building the same prior from val_loss_logged gives almost the same, −13% and −1.1pt. The only information is epoch progress: 'BAD if early in training'.
- **The exception is a gain at very small budgets (about 8 rollouts).** prog (tierflat) is about +5pt above flat-seq in A and about +3pt in B. In B it does not beat fixed-K plug-in at the same budget (about 0.80 at K=9).

**Conclusion.** "The PLV/epoch prior cuts the sequential budget by 5–6%" must not be used as a practical statement. The accurate statement is this: *at budgets of K≥10, the epoch/PLV prior does not improve the accuracy-vs-rollout frontier. The PLV metric is indistinguishable from epoch (prog_epoch and prog_replan_overlap_h16 give results that agree to the fourth decimal place).* If anything, this conclusion supports z1's and y1's "PLV carries no information beyond epoch" even more strongly.

## 4. World R: the near-plateau FPs look structural rather than 'chance'

- The numbers are confirmed. At K=10 the number of checkpoints called correctly is perfect 25/33, Nemotron 31/33, Cosmos 30/33.
- perfect and Nemotron disagree on 6 checkpoints, and Nemotron is right on all 6: the 4 can GOOD, square_s1_e0030 (1 FP takes it from 4/10→5/10, i.e. it becomes OK) and square_s1_e0060 (2 FNs).
- The failure episodes of the 4 can GOOD checkpoints all have `truth_failure_type = dropped`, with grasped, lifted and dropped all True. **Both VLMs judged them as successes.** For all 4, Nemotron explained that "the can was placed in a compartment of the right bin and stays there until the end".
- **The FPs arise from properties of the episode.**
  - All 9 Nemotron FPs are failures in which a grasp and a lift happened (dropped 8, not_placed 1). Among the 38 no_grasp failures there are 0 FPs.
  - 8 of the 9 Nemotron FPs are also Cosmos FPs. If the two judges were independent the expected count would be 2.8. P(Nemotron FP | Cosmos FP) is 0.38, against a base rate of 0.13.
- **Interpretation.**
  - The rare failures of GOOD checkpoints are failures that come close to success, so in the video they look like successes. Seen this way, a structural effect that comes from the failure type fits the data better than "4/4 FPs are chance (model prediction about 0.44⁴ ≈ 0.04)".
  - The simulator flips an independent coin for each failure, with a probability that depends only on checkpoint quality. This assumption is wrong at the episode level.
  - The direction in which this structure moves tier accuracy depends on the checkpoint mix. It helps on GOOD checkpoints, and hurts on OK checkpoints at 0.84–0.90 by inflating them to GOOD (can_s2_e0060: Cosmos FP→wrong call). So "the simulator is conservative" holds only for this sample of 33.
- The label definition is consistent. Both the eval truth and the training-time outcomes use robosuite's success check (06 protocol). So this difference looks like visual ambiguity, not a label bug. However, I did not look at the videos myself.

## 5. Leakage, held-out handling, independence, multiple comparisons

- **Leakage.**
  - The prior is leave-one-run-out. I checked `other = run_idx != run_idx[c]` in the code. The metric percentiles do not use outcomes.
  - In World A the truth is drawn from the held-out 25 and the K rollouts from the other 25. The judge fit uses eval seeds (200000+), and the World R decision rule is leave-one-checkpoint-out.
  - There is no obvious leakage. Two weak leaks remain. ① The 33 ckpt used for the judge fit overlap with the analysis targets (with different seeds). ② The q model itself was chosen on the same eval data during the y1 verification.
  - The boundary stratification (section 5, claim 5) is a stratum defined by outcome.
- **CI.**
  - The bootstrap resamples checkpoints within a run. So the variance across runs or tasks (3 runs, 2 tasks) is not in the CI.
  - Adjacent epochs of the same run are correlated, so a bootstrap that treats checkpoints as independent underestimates the uncertainty.
  - z1's replicate CIs also contain MC noise. MC noise can widen intervals, but it biases statistics such as the 'first K that crosses' toward earlier values (claim 1).
- **Multiple comparisons.**
  - About 46 rules × K were compared, with no correction. The headline "sequential 0.916" is the value of the best of several rules.
  - However, the flat prior is a default fixed in advance, and it was rechecked by exact computation.
- **Boundary rule.**
  - There are 4 checkpoints with a 50-ep success rate of exactly 0.90 (can_s1_e0035, can_s1_e0050, can_s2_e0060, square_s1_e0120). Their truth is OK. Even as K grows without bound their accuracy converges to 0.5, so the theoretical ceiling for B is 0.978.
  - Changing the decision rule to GOOD: k/K ≥ 0.9 lowers B plug-in accuracy from 0.813→0.770 at K=10, from 0.915→0.888 at K=40 and from 0.925→0.902 at K=50. The claim that 'a single GOOD boundary rule changes accuracy by several points' is confirmed.
- **Minor reporting errors.**
  - The GOOD share is given as 52% in findings and 53% in the md. It is 51.7% in World A and 46% in World B (50-ep).
  - The log score difference is "7–8 nat" in findings and "7–10 nat" in the md (10 includes pad201).
  - In "tierflat OK recall 0.43–0.53", Cosmos is at 0.40.

## 6. Corrected headline

> In this pool, getting the BAD/OK/GOOD tier right 90% of the time is expensive, and how expensive depends on the boundary rule and the checkpoint distribution.
>
> - **perfect detector, fixed-K plug-in (World B, exact):** first exceeds 0.9 at K=36. Stays at or above 0.9 from K≥43 on (ckpt bootstrap median 35 [25, 50]).
> - **sequential (flat prior, P≥0.9):** 0.916 at a mean of 25.1. It is +3.9pt [2.9, 4.9] above fixed K at the same budget, but this gain appears only in World B (in World A it is +0.6 [−0.7, 1.9]).
> - **Nemotron:** K=50 plug-in accuracy is 0.88–0.91 depending on the judge model, so whether it reaches 90% is borderline. sequential gives 0.876 at 35.
> - **Cosmos3-Edge:** K=50 accuracy is at most 0.88 under every judge-model variant. For Cosmos the judge itself is also a major source of cost (46% of truth OK judged GOOD).
> - **epoch/PLV prior:** does not improve the accuracy–rollout frontier at K≥10. The 5–6% saving z1 reported is measured against a rule that is 4–6pt below the flat frontier.
> - **The statement 'checkpoints far from the boundary need 14–18':** a statement conditional on a stratum defined by outcome. It ranges from 6 to 20 depending on the width of the stratum.

## 7. Risks still open

- There are only 3 runs and 2 tasks. All GOOD checkpoints are can. Every accuracy depends heavily on the tier mix.
- The judge error model was fit on 67 failures (9 in the high-quality range). The fact that FPs depend on the failure type (section 4) is not modeled. square cannot be cross-validated by run.
- Carrying this over to a real-robot budget would need 30 or more recordings and judgments per checkpoint at near-plateau checkpoints. The Nebius calls for that require user approval.
- An issue raised in the y1 verification still stands. The `*_h10` and `*_h16` columns are identical in 93/93 rows. So replan_overlap_h16 must not be interpreted as a metric specific to H=16.

## Rerun

```
cd /path/to/repo
PY="env PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python"
# z1 reproduction (0 numeric differences from the existing results)
$PY scripts/hackathon/explore/z1_tier_triage.py --reps 250 --orders 8 \
    --out data/outputs/hackathon_explore/z1_tier_triage_verification/rerun_main.json --fig ""
$PY scripts/hackathon/explore/z1_tier_triage.py --reps 100 --orders 8 --world-judge iid \
    --out data/outputs/hackathon_explore/z1_tier_triage_verification/rerun_iid.json --fig ""
# independent exact computation (about 50 s) + frontier (about 2 min; importing it reruns verify)
$PY scripts/hackathon/explore/z1_tier_triage_verify.py
$PY scripts/hackathon/explore/z1_tier_triage_verify_frontier.py
```
