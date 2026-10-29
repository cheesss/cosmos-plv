# x4_nemotron_modes: other ways to use Nemotron-3.5-Super-VL for failure diagnosis

**Status: EXPLORATORY / POST-HOC.** I chose the variant on the dev split (36 episodes, seeds 300000+) and then ran it unchanged on the eval split. Only the comparison against the frozen v2 prompt is new. The protocol in `06_experiment_protocol.md` is not amended.

> Summary: I tried Nemotron in 4 ways: tracking with 8 keyframes in one request (A), the v2 prompt with reasoning turned on (B), a video prompt that distinguishes compartments and pegs (C), and a single 2x4 grid image (D). The variant was chosen on dev only. The chosen A (r2 rule) and C were compared with the frozen v2 on the same 200 eval episodes (paired). The failure-type accuracy gain was not significant (0.390 vs 0.356, difference +0.034, 95% CI [-0.119, 0.186]). Success accuracy was slightly lower than v2 (0.920 vs 0.950). Of the 5 "can put into the wrong compartment" cases, A and C both caught 5/5, while v2 caught only 1/5. C judged success 150/150 on can but broke down on square (0.66). Reasoning (B) made every metric worse on dev, with 2.3x the latency and 12x the output tokens. The largest structural cause dragging down failure-type accuracy lies in the ground-truth labels. All 11/11 square 'other' labels are an artifact of the nut settling 6 cm down from its spawn height; physically they are no_grasp.

## Question

The frozen v2 video prompt has eval success accuracy 0.959 (n=145). On failed episodes, however, it gets the failure type right only 40% of the time, and it misses cans dropped into the wrong compartment. Can another way of applying the same Nemotron model give a better physical failure diagnosis at a similar cost?

## Why it could help PLV or make it more efficient

* PLV ranks checkpoints by cheap offline metrics. A VLM judge that names the *failure mode* (no grasp, dropped, not placed) for each checkpoint would explain a ranking, and on a real robot it would replace simulator instrumentation.
* Efficiency: every variant here costs one API call per episode. A cheaper input, such as the single grid image, would also cut tokens.

## Method

Every variant uses `nvidia/Nemotron-3.5-Super-VL` through `NebiusVLMClient(evidence_path=results/hackathon/explore/nebius_calls_explore.jsonl)`, temperature 0.2, with the tag `explore/x4_nemotron_modes/...`.

| id | input | output schema | success / failure type |
|---|---|---|---|
| A_track8 | 8 evenly spaced keyframes (448 px JPEG, including the last frame) as 8 images in ONE request | task-free per-image `{location, held}` with fine-grained locations (4 right-bin compartments, square vs round peg, low vs high on the peg, fell off) | fixed rule (below) |
| B_v2reason | frozen v2 video prompt text, unchanged | v2 schema | model's own; `enable_thinking=True`, max_tokens 4096 |
| C_vidcomp | compact video (the client's 4 fps / 320 px encoding) + full-resolution last frame | task-free facts: closed? lifted? left gripper? final fine-grained location, held at end | fixed rule |
| D_grid8 | the same 8 keyframes tiled into one 2x4 grid image (384 px panels with `n t=..s` labels) | per-panel `{location, held}` | fixed rule |

**Fixed rule.** It mirrors `failure_annotator.py` and was written before any data was seen:
* success means the final location is the goal: `right_bin_front_right` for can, `on_square_peg_low` for square;
* if it is not a success and the object was never lifted, the type is `other` when the object fell off (floor/table) and `no_grasp` otherwise;
* if it was lifted and is still held at the end, the type is `not_placed`; otherwise it is `dropped`.

**Rule r2.** I chose r2 on dev *after* looking at the dev errors. For A and D it counts the object as lifted only if some keyframe has location `in_gripper`. A `held` flag while the object still rests in place counts as contact only, which matches the annotator: a grasp without a 5 cm lift is `no_grasp`. r2 is post-processing of the same calls, so it needs no extra calls.

**Parser repair.** I added one recorded parser repair, chosen on the dev smoke test. Nemotron sometimes writes a string value without its opening quote (`"explanation": The robot ...",`). The repair adds the quote marks only and is applied to every variant. B needed it on 25/36 dev answers; the variants chosen for eval needed it 0 times on eval.

**Dev selection rule.** I declared this rule in code before the dev runs:
* maximise failure-type accuracy on failures, counting unparsed answers as wrong;
* require balanced success accuracy (unparsed counted as wrong) of at least the frozen v2 dev value minus 0.05, i.e. at least 0.821;
* keep the best two variants that use distinct API calls.

The chosen pair was A_track8_r2 and C_vidcomp. It is stored in `data/outputs/hackathon_explore/x4_nemotron_modes/dev_selection.json`.

**Eval.** This covers every eval episode with a video at run time: 200 episodes from 20 checkpoints (can_s1 e0–e100, can_s2 e0–e160, square_s1 e0–e120). The eval set grew to 210 while the runs were in progress. The last 10 episodes are scored by frozen v2 only and are excluded from the paired tables.
* The frozen v2 baseline comes from the rows of `eval_nebius_v2.jsonl` (150 episodes).
* I scored the 60 eval episodes missing from that file (checkpoints can_s1_e0080/e0100, can_s2_e0140/e0160, square_s1_e0120/e0150) with the *identical* frozen settings (variant `V2_frozen`: same prompt, reasoning off, max_tokens 700). They are in this exploration's scratch file.
* Every comparison is paired on the same 200 episodes.

**Metrics.** These follow `summarize_diagnostics.py`:
* success accuracy and balanced accuracy, counting unparsed answers as wrong;
* failure-type accuracy on truly failed episodes (unparsed counted as wrong);
* Wilson 95% CIs;
* a paired bootstrap (10 000 resamples over episodes) of the variant minus v2, with an exact McNemar p-value.

I also report:
* **Harmonized failure type.** This reruns `annotate_episode` on the trace from sim step 10 onward. It is a sim-state relabel that never uses model output; see the label audit below.
* **Hard cases** defined from sim state only:
  * `wrong_compartment`: the can ended inside the right bin but outside the can compartment;
  * `nut_high_on_peg`: the nut ended within 4 cm of the square peg axis, more than 3 cm above the table, and the episode is not a success.
* **PLV checkpoint view.** The Spearman correlation between the VLM-estimated and the true per-checkpoint success rate.

## Commands run (all exit 0)

```
# smoke test, 1 dev episode x 4 variants (4 calls; rows in data/outputs/hackathon_explore/x4_nemotron_modes/smoke.jsonl)
PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 ~/.venvs/cosmos-plv/bin/python -c "...m.run_one(c, ep, v, 'smoke')..."   # exit 0
# dev (36 episodes each)
for v in A_track8 B_v2reason C_vidcomp D_grid8; do
  PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/x4_nemotron_modes.py run --split dev --variant $v --workers 4; done   # exit 0 x4
~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/x4_nemotron_modes.py select          # exit 0 -> chosen [A_track8_r2, C_vidcomp]
# eval
... x4_nemotron_modes.py run --split eval --variant A_track8 --workers 4                               # exit 0 (200 episodes)
... x4_nemotron_modes.py run --split eval --variant C_vidcomp --workers 4                              # exit 0 (200 episodes)
... x4_nemotron_modes.py run --split eval --variant V2_frozen --only-missing-from-frozen --workers 4   # exit 0 (60 episodes)
... x4_nemotron_modes.py score                                                                         # exit 0
```

Nebius calls used: **608 / 800**: 4 smoke, 144 dev, 200 + 200 + 60 eval, 0 infrastructure errors. Logs are in `data/outputs/hackathon_explore/x4_nemotron_modes/logs/`.

## Results

### Dev (method choice only; n = 36 episodes, 21 failures)

| variant | success acc [Wilson 95%] | bal. acc | FT acc on failures [95%] | wrong-compartment caught | nut-high caught | latency s | prompt / completion tok |
|---|---|---|---|---|---|---|---|
| frozen v2 | 0.861 [0.713, 0.939] | 0.871 | 0.333 [0.172, 0.546] | 0/2 | 1/2 | 3.47 | 3533 / 130 |
| A_track8 (r1) | 0.972 [0.858, 0.995] | 0.976 | 0.333 [0.172, 0.546] | 2/2 | 1/2 | 3.71 | 2864 / 199 |
| **A_track8_r2** | 0.972 [0.858, 0.995] | 0.976 | 0.524 [0.324, 0.717] (r2 tuned on these errors) | 2/2 | 1/2 | 3.71 | 2864 / 199 |
| B_v2reason | 0.778 [0.619, 0.883] (1 unparsed) | 0.810 | 0.048 [0.008, 0.227] | 1/2 | 0/2 | 8.04 | 3533 / 1567 |
| **C_vidcomp** | 0.861 [0.713, 0.939] | 0.881 | 0.286 [0.138, 0.500] | 2/2 | 0/2 | 3.68 | 3879 / 86 |
| D_grid8 (r1 = r2) | 0.889 [0.747, 0.956] | 0.905 | 0.190 [0.077, 0.400] | 1/2 | 0/2 | 3.75 | 1953 / 203 |

On dev, reasoning (B) is clearly worse than the frozen v2 on failure type: paired difference -0.286, 95% CI [-0.476, -0.095]. It also costs 2.3x the latency and 12x the completion tokens. The grid (D) uses 32% fewer prompt tokens than A but loses accuracy.

### Eval (chosen variants, unchanged; paired on the same 200 episodes, 59 failures)

| | success acc [95%] | bal. acc | FP / FN | FT acc on failures [95%] | harmonized FT acc [95%] | wrong compartment (n=5) | nut high on peg (n=5) | latency s | prompt / completion tok |
|---|---|---|---|---|---|---|---|---|---|
| frozen v2 | **0.950** [0.910, 0.973] | **0.930** | 7 / 3 | 0.356 [0.246, 0.483] | 0.356 [0.246, 0.483] | 1/5 | 2/5 | 3.20 | 2675 / 129 |
| A_track8_r2 (chosen) | 0.920 [0.874, 0.950] | 0.899 | 9 / 7 | **0.390** [0.276, 0.517] | **0.441** [0.322, 0.567] | **5/5** | 1/5 | 3.69 | 2869 / 198 |
| A_track8 r1 (sensitivity) | 0.920 [0.874, 0.950] | 0.899 | 9 / 7 | 0.373 [0.261, 0.500] | 0.407 [0.291, 0.534] | 5/5 | 1/5 | 3.69 | 2869 / 198 |
| C_vidcomp (chosen) | 0.915 [0.868, 0.946] | 0.856 | 17 / 0 | 0.305 [0.203, 0.431] | 0.322 [0.217, 0.449] | **5/5** | 0/5 | 3.67 | 3027 / 87 |

Paired difference (variant minus frozen v2), bootstrap 95% CI, exact McNemar p:

| | success correct | FT correct on failures | harmonized FT correct |
|---|---|---|---|
| A_track8_r2 | -0.030 [-0.075, +0.015], p=0.26 | +0.034 [-0.119, +0.186], p=0.83 | +0.085 [-0.085, +0.254], p=0.42 |
| C_vidcomp | -0.035 [-0.080, +0.010], p=0.19 | -0.051 [-0.203, +0.102], p=0.65 | -0.034 [-0.186, +0.119], p=0.82 |

By task on eval (same episodes):

| | can success (n=150) | can FT on failures (n=37) | square success (n=50) | square FT on failures (n=22) |
|---|---|---|---|---|
| frozen v2 | 0.973 (FP 4, FN 0) | 0.541 | 0.880 (FP 3, FN 3) | 0.045 |
| A_track8_r2 | 0.960 (FP 0, FN 6) | 0.622 | 0.800 (FP 9, FN 1) | 0.000 (harmonized 0.136) |
| C_vidcomp | **1.000** [0.975, 1.000] (FP 0, FN 0) | 0.486 | 0.660 (FP 17, FN 0) | 0.000 (harmonized 0.045) |

Failure-type confusion on eval (truth → prediction):

| truth | frozen v2 | A_track8_r2 |
|---|---|---|
| no_grasp (28) | no_grasp 20, not_placed 8 | no_grasp 15, dropped 12, not_placed 1 |
| dropped (20) | dropped 1, not_placed 9, none 7, no_grasp 3 | dropped 8, not_placed 5, none 6, no_grasp 1 |
| other (11, all square) | not_placed 11 | dropped 6, none 3, no_grasp 2 |

PLV checkpoint view: the per-checkpoint success rate from the VLM against the simulator's success rate on the same episodes (20 checkpoints).

| | Spearman | MAE |
|---|---|---|
| frozen v2 | 0.893 | 0.040 |
| A_track8_r2 | 0.888 | 0.070 |
| C_vidcomp | 0.834 | 0.085 |

### Label audit (sim state only, all 210 eval episodes present at scoring)

| task | official label | harmonized label | n |
|---|---|---|---|
| square | other | no_grasp | 11 |
| square | no_grasp | dropped | 1 |
| everything else | unchanged | unchanged | 198 |

The square nut spawns at z = 0.89 and settles onto the table (z ≈ 0.83) within about 5 sim steps. `annotate_episode` measures heights from step 0, so a nut that is never touched ends "3 cm below its start" and is labelled `other` ("knocked off"). In fact it never moved. All 11 eval and 4 dev `other` labels come from this artifact. No VLM can match them visually, which caps square failure-type accuracy for every backend.

## Interpretation

1. **Negative on the main question.** None of the alternatives improves failure-type accuracy on eval in a way that can be told apart from noise. The best, A_track8_r2, gets 0.390 against 0.356, paired difference +0.034 with CI [-0.12, +0.19]. Its large dev gain (0.333 → 0.524) shrank on eval. That gain was partly from tuning the rule on dev errors, as expected.
2. **The errors change rather than shrink.** v2 calls drops "not placed" or "success". The keyframe tracker instead hallucinates a lift when the gripper hovers over and hides the object in one keyframe (no_grasp → dropped, 12/28). Eight keyframes, about 2–3 s apart, cannot tell "lifted 9 cm then dropped" from "touched".
3. **The compartment-aware vocabulary fixes the specific blind spot.** Both A and C flag 5/5 eval cans dropped into the wrong (front-left) compartment, against 1/5 for v2. On can, C (video plus a full-resolution last frame plus the 4-compartment vocabulary) judges success correctly on 150/150 episodes (Wilson [0.975, 1.0]), against 146/150 for v2. On square the same prompt collapses: 17 false positives, where a never-touched nut or a nut on the table is called `on_square_peg_low`. The keyframe tracker misreads front-right as back-right for 6 successful cans.
4. **Reasoning mode hurts.** With the frozen v2 text, `enable_thinking` lowers dev success and failure-type accuracy, more than doubles latency and multiplies completion tokens by 12. It also breaks JSON quoting in 25/36 answers.
5. **For PLV the per-checkpoint success rate is no better.** The Spearman correlation with the simulator success rate is 0.89 for both v2 and A, with a lower MAE for v2. The VLM judge already ranks checkpoints about as well as the simulator does, and these modes add nothing there.
6. **The biggest lever is the label, not the prompt.** The square `other` artifact means that 11 of the 22 square eval failures can never be matched. Any failure-type headline for square (including "Nebius 0.40") should be read with that in mind.

## Limitations

* The dev split is small: 36 episodes, 21 failures, only 2 wrong-compartment and 2 nut-high cases. I chose rule r2 after looking at the dev errors.
* The eval hard-case subsets have n=5 each, so their CIs are wide (5/5 gives Wilson [0.57, 1.0]).
* I ran each episode with one sampling seed at temperature 0.2. I did not measure run-to-run variance.
* `NebiusVLMClient` re-encodes video to about 5 fps at 320 px, which I could not change without editing it. C therefore adds a full-resolution last frame. A and D use 448 px stills.
* I did not try a true v7-style one-call-per-keyframe tracker (8 calls per episode), to stay within the call budget. A sends all 8 keyframes in one request instead.
* The frozen v2 baseline for 50 of the 200 paired episodes is my own rerun with the identical frozen settings, not the original file.
* The square task has only one run (seed 1) and 50 paired episodes.

## What to try next

* **Fix the annotator (structural risk).** Measure square heights from the settled pose, as `harmonized_failure_type` does here. Then re-score all backends and add the issue to KNOWN_RISKS.
* **Route by task, declared on a fresh dev split.** Use the compartment-aware video prompt (C) for can, and v2 or a peg-specific prompt for square. This is post-hoc here and must not be claimed from these numbers.
* **Denser track near contact.** Choose keyframes with the recorded gripper action, such as the frames just after each close command. That attacks the lift hallucination at the same call count, and the action stream is available offline without sim state.
* **Ask for per-frame visibility.** Add an "object occluded by gripper" field, so occluded frames drop out of the rule instead of counting as a lift.

## Files

* Script: `scripts/hackathon/explore/x4_nemotron_modes.py` (run / select / score; re-runnable, resumes and picks up new eval episodes)
* Results: `results/hackathon/explore/x4_nemotron_modes.json`
* Raw rows, frames, logs, dev selection: `data/outputs/hackathon_explore/x4_nemotron_modes/`
* Call evidence (no key): `results/hackathon/explore/nebius_calls_explore.jsonl` (tags `explore/x4_nemotron_modes/...`)
