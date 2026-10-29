# x4_nemotron_modes: adversarial verification

**Status: PARTIALLY CONFIRMED. EXPLORATORY / POST-HOC.** This note checks the x4 report (`docs/hackathon/explore/x4_nemotron_modes.md`). Every new number below comes from `results/hackathon/explore/x4_nemotron_modes_verification.json`, which `scripts/hackathon/explore/x4_nemotron_modes_verify.py` produces. I changed none of x4's files.

> Summary: x4's numbers **all reproduce unchanged** when the saved raw responses are re-parsed and recomputed with independent code (0 parse/ground-truth mismatches). The main negative conclusion, "no variant significantly raises failure-type accuracy", also holds. However, the following five points need fixing.
> (1) **Clip-length leak.** Successful episodes end 16 steps after success, while failures are always 201 frames. So the zero-cost rule "success if length < 201 frames" is perfect: eval 220/220, dev 36/36. All three prompts state the length in text. The VLMs' success accuracy (0.92–0.95) and checkpoint Spearman (0.89) are below this rule (1.0). But in a probe where 20 success videos were padded to 20 s, 0/20 judgments changed. There is no evidence that the model uses the length.
> (2) **"Wrong compartment 5/5 vs 1/5" was designed after seeing the eval ground truth.** 4 of the 5 cases were already in the pre-x4 eval error analysis (experiment2_error_analysis.json), and the compartment vocabulary came from that analysis. The only cases not in that analysis are 1 eval and 2 dev cases, and the result there is 3/3 vs 0/3.
> (3) **The ground-truth labels changed during verification** (commit baa3967, settled-height fix). With the current official labels it is A_track8_r2 0.441 vs v2 0.356 (+0.085 [−0.085, 0.254]), still not significant. If x4's `score` is rerun now, the v2 rows are computed with the new labels and the variant rows with the old labels, mixed together.
> (4) **The pooled null result hides effects in opposite directions.** Dropped recall is 8/20 for both A and C, and 1/20 for v2 (p=0.039, not corrected for multiple comparisons). Conversely, C's no_grasp recall is 10/28, below v2's 20/28 (p=0.002). 30 of the 59 failures come from three epoch 0 checkpoints.
> (5) The point that the square-nut 'other' labels are an artifact is correct. But "never touched / never moved" is an overstatement. 6 of the 11 cases were pushed more than 5 cm in xy without being grasped.

## What I checked and how

* **Re-parse.** I re-parsed all 604 dev and eval raw outputs from x4 (the 4 smoke rows excluded) and the 186 frozen-v2 rows (150 eval + 36 dev) with x4's parsers. I re-derived rule r2 with my own code and recomputed every metric with independent code, including success, FP/FN, failure type on failures, Wilson intervals, the paired episode bootstrap and McNemar.
* **New checks:**
  * a clip-length baseline;
  * a checkpoint-cluster bootstrap;
  * the checkpoint view split by task;
  * per-truth-class paired tests;
  * the epoch-0 versus trained-checkpoint split;
  * a sim-state audit of the square "other" labels and of the "lift hallucination" cases (video frame k = sim step 2k);
  * a test for design-on-eval against `experiment2_error_analysis.json`.
* **Nebius probe (30 of my 30 allowed calls, tag `explore/x4_nemotron_modes_verify/...`):**
  * 10 eval success clips padded to the timeout length (201 frames, final frame repeated, prompt saying 20.1 s), sent through frozen v2;
  * the same 10 clips through C_vidcomp;
  * 10 eval failures re-run through A_track8 with byte-identical prompts (prompt SHA-256 matched 10/10), to measure run-to-run agreement.
* **No GPU use.**

**Label change during verification.** At 05:15 KST another session committed `baa3967` ("fix settled-height failure labels"). It ran `scripts/hackathon/refresh_truth.py`, which rewrote every `summary.json` and `data/outputs/hackathon_cosmos/*.jsonl`. On eval it changed 11 labels `other→no_grasp` and 1 `no_grasp→dropped`; on dev, 4 and 1. It also added 10 eval episodes (`can_s2_e0180`). x4's scratch rows keep the old labels. The main reproduction therefore uses the labels stored in x4's rows. A separate block re-scores against the current labels.

## Commands run

```
cd /path/to/repo
P="env PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 ~/.venvs/cosmos-plv/bin/python"
$P scripts/hackathon/explore/x4_nemotron_modes_verify.py rescore   # run 9x while adding checks; 4 runs exited 1
                                                                   # (my own bug: an empty subset after the label
                                                                   # refresh removed 'other'); final run exit 0
$P scripts/hackathon/explore/x4_nemotron_modes_verify.py plan      # exit 0 (deterministic probe plan)
$P - <<EOF ... (dry run: prompt-hash match 10/10, padded clip = 201 frames) EOF   # exit 0
$P scripts/hackathon/explore/x4_nemotron_modes_verify.py probe     # exit 0, 30 Nebius calls, 0 errors
$P scripts/hackathon/explore/x4_nemotron_modes_verify.py rescore   # exit 0 (final; includes the probe)
```

Probe rows are in `data/outputs/hackathon_explore/x4_nemotron_modes_verify/probe.jsonl`, with the log beside them in `probe.log`.

## Claim by claim

| x4 claim | verdict | evidence |
|---|---|---|
| Eval paired n=200 (20 ckpts), 59 failures; A_track8_r2 FT 0.390 [0.276, 0.517] vs v2 0.356 [0.246, 0.483], +0.034 [-0.119, +0.186], p=0.83 | **Reproduced exactly** | 0 parse, success or FT mismatches across all rows. Truth labels matched `summary.json` with 0 mismatches before the 05:15 refresh; afterwards the refreshed frozen file differs on exactly the 12 relabelled episodes. A cluster bootstrap over checkpoints gives [-0.064, +0.216]. |
| Success 0.920 vs 0.950, -0.030 [-0.075, +0.015], p=0.26; C 0.915, FT -0.051 | **Reproduced** | My episode bootstrap for C's FT gives [-0.203, +0.085] (x4: +0.102; RNG order differs). The cluster bootstrap for C's success difference is [-0.130, +0.035]. |
| Dev table, selection rule, B reasoning worse (-0.286 [-0.476, -0.095]), 25/36 repairs, 2.3x latency, 12x tokens | **Reproduced** | McNemar p=0.031 (B 0 only-correct vs v2 6). Without r2 the rule picks A_track8 (r1) + C, which are the same API calls. Eval started 13 s after the selection timestamp (19:55:19Z → 19:55:32Z), so there is no sign of eval peeking. |
| Wrong-compartment cans: A and C 5/5 vs v2 1/5 | **Numbers right; the evidence is weaker than presented** | This is design-on-eval: 4 of the 5 episodes are in `experiment2_error_analysis.json`, written 04:34 KST before x4 ran, which is where the blind spot was found and which motivated the compartment vocabulary. Held out, there is 1 eval episode (A and C 1/1, v2 0/1) and 2 dev episodes (2/2 vs 0/2), so 3/3 vs 0/3 in total. "5/5 vs 1/5" is the same fact as "can FP 0 vs 4": all 4 v2 can FPs are wrong-compartment. |
| C judges can success correctly on 150/150 (Wilson [0.975, 1.0]) | **Numbers right; the context is missing** | A free clip-length rule also scores 150/150. The Wilson interval treats 150 episodes from 15 checkpoints as independent. |
| C on square: 0.66, 17 FP ("nut on the table → on_square_peg_low") | **Mostly confirmed** | Truth types of the 17 FPs: other 8, dropped 8, no_grasp 1. 10/17 ended more than 17 cm from the peg. 3 ended within 6 mm of the peg axis (nut high on the peg), which is not "on the table". |
| no_grasp → dropped 12/28 is a hallucinated lift "when the gripper hides the object" | **The lift hallucination is confirmed; the mechanism is not shown** | At every keyframe the model called `in_gripper`, the object was raised less than 2 cm (0/12 raised ≥ 2 cm). In 9/12 cases the end effector was more than 8 cm from the object at all flagged keyframes. In the 4 flagged frames I viewed, the gripper is next to or over the can in the image, so the occlusion story is plausible but untested. 1/12 (`square_s1_e0030/200001`) did contain a real 9.9 cm lift, and that case is correct under the new labels. |
| 6 successful cans read as back-right | **Confirmed** | All 6 A false negatives on can have final location `right_bin_back_right`. |
| Square `other` is a spawn-height artifact (z 0.89 → 0.83 within about 5 steps); physically no_grasp | **Confirmed, with one overstatement** | Spawn minus settled height is 5.3–6.0 cm for all 60 square episodes, and 0/11 `other` cases were ever grasped. But "a nut that was never touched / never moved" is wrong for 6/11: they were pushed 5–29 cm in xy without a grasp, and the gripper came within 3.1 cm. The repo has since adopted the fix, and x4's harmonized labels match the new official labels on 59/59 failures. |
| PLV view: Spearman 0.893 (v2), 0.888 (A), 0.834 (C); "no gain"; "ranks checkpoints about as well as the simulator does" | **Numbers right; the interpretation is overstated** | The clip-length rule gives Spearman 1.0 and MAE 0. For v2 minus that rule, the bootstrap over checkpoints is [-0.34, -0.014]. The simulator is the target, not a competitor. Split by task: on can, C gives 1.0 vs v2 0.894 (diff CI [+0.003, +0.325], post hoc); on square (5 ckpts), C's MAE is 0.34. |
| 608/800 calls, 0 infrastructure errors, 0 repairs on eval | **Confirmed** | From the evidence log: smoke 4, dev 144, eval A 200, C 200, V2_frozen 60. |

## Issues x4 did not report

### 1. Clip-length label leakage (affects every video backend, not only x4)

Episodes stop 16 steps after the simulator's first success, so clip length encodes the success label. Every failure clip has 201 frames; success clips have 50–150. The rule "success ⇔ fewer than 201 frames" is correct on **220/220 eval** (Wilson [0.983, 1.0]) and **36/36 dev**.

Every prompt reveals the length:
* v2: "The clip lasts about X seconds."
* A: per-image timestamps.
* C: "(about X s)".

All VLM success accuracies (0.915–0.950) and checkpoint Spearmans (0.83–0.89) are therefore **below a zero-cost baseline**. Read this way, they measure what the VLM adds only in a setting without a success detector.

The padding probe did not show the models exploiting the cue: padded successes stayed successes for 10/10 in v2 and 10/10 in C. With 0/10 flips each, the Wilson upper bound is 0.28. The probe tests only one direction, because failures cannot be shortened without changing their content.

Recommendation: hide the duration in prompts and pad or trim clips to a fixed length before any VLM success number is reported.

### 2. Label version drift and a rerun hazard

Against the current official labels (same model outputs):

| | FT acc on failures [Wilson] | paired vs v2 [episode CI] [cluster CI] | McNemar p |
|---|---|---|---|
| frozen v2 | 0.356 [0.246, 0.483] | n/a | n/a |
| A_track8_r2 | 0.441 [0.322, 0.567] | +0.085 [-0.085, +0.254] [-0.033, +0.267] | 0.42 |
| A_track8 r1 | 0.407 [0.291, 0.534] | +0.051 [-0.119, +0.220] [-0.092, +0.275] | 0.69 |
| C_vidcomp | 0.322 [0.217, 0.449] | -0.034 [-0.186, +0.119] [-0.234, +0.250] | 0.82 |

x4's headline 0.390 vs 0.356 is now stale. Its "harmonized" numbers are now the official ones, and the conclusion (no significant gain) is unchanged.

x4's `score` command reads the refreshed `eval_nebius_v2.jsonl`, where 12 rows now carry labels that differ from x4's own variant rows. Each side is scored against its own row's label, so a rerun would mix two label versions. Today that reproduces 0.390 vs 0.356 only because v2's count does not change under either version. Before rerunning, refresh the labels in x4's scratch rows or score against `summary.json`.

### 3. The pooled null hides opposing per-class effects (post hoc)

Paired results by true failure type (labels as x4 had them, 6 tests, uncorrected):

| truth class (n) | v2 | A_track8_r2 | C_vidcomp |
|---|---|---|---|
| dropped (20) | 1/20 | 8/20 (8 vs 1 discordant, p=0.039) | 8/20 (p=0.039) |
| no_grasp (28) | 20/28 | 15/28 (p=0.27) | 10/28 (0 vs 10, p=0.002) |
| other (11, all square) | 0/11 | 0/11 | 0/11 |

30 of the 59 failures come from the three **epoch-0** checkpoints, and all of those are no_grasp or other:
* On the epoch-0 failures, v2 scores 16/30, A_r2 13/30 and C 7/30 (C vs v2 p=0.004).
* On the 29 failures from **trained** checkpoints (20 dropped, 8 no_grasp, 1 other), v2 scores only 0.172 [0.076, 0.345], against A_r2 0.345 and C 0.379. The paired differences are +0.17 (p=0.27) and +0.21 (p=0.11; cluster CI [+0.03, +0.455]).

None of this survives correction for multiple tests, and I chose the split after seeing the data. It does mean that "no gain" should read "not shown". The variants appear to trade no_grasp recall for dropped recall. Dropped is the failure mode of the trained checkpoints that PLV actually ranks.

### 4. Independence

The 59 failures come from 14 checkpoints: 3 contribute 10 each (the epoch-0 checkpoints), 3 contribute 4–6 and 8 contribute 1–3. Cluster bootstrap intervals are in the tables above; they leave every conclusion unchanged but widen C's success difference to [-0.130, +0.035]. The Wilson intervals in x4, such as [0.975, 1.0] for C on can, treat episodes of the same checkpoint as independent.

### 5. Run-to-run variance

A_track8 re-run on 10 eval failures with identical prompts (temperature 0.2):
* success agreement 10/10;
* r1 failure-type agreement 10/10;
* r2 agreement 9/10 (one flip, dropped → no_grasp).

Single-run noise is real but small compared with the effects discussed.

### 6. Minor

* `compact_video_bytes` keeps every second frame (5 fps of content) but encodes at 4 fps, so the model sees a clip 25% slower. The v2 prompt also says "4 frames per second". This applies to every video variant.
* The v2 latency (3.20 s) mixes 150 rows recorded earlier with 50 rerun rows, so the latency comparison is not time-matched.

## Corrected headline

EXPLORATORY and post-hoc. x4's numbers reproduce exactly from the saved outputs, and its main negative holds: no variant improves pooled failure-type accuracy significantly. A_track8_r2 gets 0.390 vs 0.356, +0.034 [-0.119, +0.186]; under the labels the repo adopted during verification it gets 0.441 vs 0.356, +0.085 [-0.085, +0.254].

Four qualifications apply:
* **Opposing effects.** The null averages a gain in recall for dropped episodes (8/20 vs 1/20, p=0.039 uncorrected) against a loss for no_grasp.
* **Design-on-eval.** The wrong-compartment win (5/5 vs 1/5) was designed after the same eval errors were seen; held out, it is 3/3 vs 0/3.
* **Clip-length baseline.** Every backend's success judgment is below a free clip-length rule (220/220) that all prompts disclose, although padding probes found no exploitation (0/20 flips).
* **Square label.** The square-label artifact is real, but the nuts were often pushed, not untouched.

## What to do next

1. Remove the duration cue from prompts and normalise clip length, then re-measure success accuracy for v2, local v6/v7 and x4 (the cost is unchanged).
2. Pre-register "dropped recall on trained checkpoints" as the target metric. Then test A_track8 against v2 on new eval checkpoints (e.g. can_s2_e0180 and later) that no error analysis has touched.
3. Fix x4's rerun path: score against `summary.json` labels, not per-row truth.
4. Add the clip-length baseline to `KNOWN_RISKS` as a structural risk for every VLM success claim in the hackathon write-up.
