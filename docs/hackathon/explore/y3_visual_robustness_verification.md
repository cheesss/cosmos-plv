# y3_visual_robustness adversarial verification (EXPLORATORY, post-hoc)

**Verdict: partially_confirmed.**

- **Confirmed.** The data, the state replay and the rescoring have no defects.
  - The 600 + 600 judge rows were rescored with independent code, and every published number matched.
  - The claim that state replay is bit-for-bit deterministic was reproduced independently on CPU (6/6, including negative controls).
  - The conclusion that image degradation has no detectable effect holds.
  - The conclusion that the judges are at chance level on the wrist camera also holds. This is the only effect that stays significant after multiple-comparison correction. The prompt confound, however, remains.
- **Partly refuted.** The interpretation of the far cameras. "Camera placement is decisive" and "far cameras inflate VLM-judged success by seeing failures as successes" **change direction and size with the backend and the sample composition.**
  - The error that reproduces for Nemotron birdview goes the opposite way. It is an FN that judges can successes as `no_grasp`, and it came back 4/4 on retest.
  - Reweighted to the eval composition, Nemotron birdview **lowers** the judged success rate from 0.77 to 0.62.
  - Under the same reweighting, "Nemotron ≥ Cosmos in every condition" flips for birdview and wrist.
- **Nature of the headline numbers.** "birdview −0.15 to −0.17" comes from a subset that oversampled failures. The Bonferroni-corrected (18 comparisons) intervals include 0.

## Question

We verified the y3 report (`docs/hackathon/explore/y3_visual_robustness.md`). The targets were:

- replay determinism
- no effect of image degradation
- size and direction of the camera effects
- error-mechanism numbers
- model ranking
- noise floor

We checked for leakage, dev/eval confusion, pairing, clustering, denominators, cherry-picking, and whether the claims are stronger than the numbers.

## Method

All work was on CPU. No new GPU jobs. 30 new Nebius calls were made.

1. **Independent rescoring** (`scripts/hackathon/explore/y3_visual_robustness_verify.py`). The saved `judge_local.jsonl` / `judge_nebius.jsonl` were rescored without importing y3_analyze.
   - Row integrity: duplicates, missing rows, and agreement of truth with summary.json and the npz.
   - Cross-check against the Nebius evidence log: request_id, video_sha256, prompt_sha256, temperature, pad.
   - acc / bal-acc / TPR / TNR / Wilson and the paired Δ were recomputed.
   - The checkpoint-cluster bootstrap ran 20000 times with seed 12345. The Bonferroni intervals use α = 0.05/18.
   - McNemar tests with Holm correction, plus LOCO and per-task Δ and the FN/FP breakdown, were computed.
2. **CPU state replay** (`y3_visual_robustness_verify_replay.py`, Env A, `MUJOCO_GL=osmesa`, `render=False`, `CUDA_VISIBLE_DEVICES=""`). Every 10th entry of the selection list was taken, 6 in total (can 3, square 3, a fixed rule independent of outcomes). 2 negative controls were run alongside: one adds +1e-3 to action x at step 20, the other adds +1 to the env seed.
3. **Nemotron birdview test-retest** (`y3_visual_robustness_verify_retest.py`, 30 calls). The 15 episodes where cam_birdview and cam_agentview disagreed on correctness in y3 were rerun in both conditions with the same frozen prompt and parameters. The tag is `explore/y3_visual_robustness_verification/retest/...`. Because the episodes were chosen by disagreement, pure noise would be biased toward a shrinking effect through regression to the mean.
4. **Strata / reweighting / alternative clusters** (`y3_visual_robustness_verify_strata.py`).
   - (a) Cherry-pick check: frozen accuracy per stratum was compared between the subset and the 250-episode universe.
   - (b) Post-stratification: the per-stratum Δ were reweighted to the universe composition (success 0.756), with CIs from a per-stratum episode bootstrap.
   - (c) Env seed cluster bootstrap: episodes that use the same seed share the initial scene even across checkpoints.

## Commands and exit codes

All commands were run from `/path/to/repo`. Logs are in `data/outputs/hackathon_explore/y3_visual_robustness_verification/`.

| # | command | exit |
|---|---|---|
| 1 | `PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/y3_visual_robustness_verify.py` | 0 |
| 2 | `MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa CUDA_VISIBLE_DEVICES="" bash -lc 'source scripts/hackathon/env.sh && python scripts/hackathon/explore/y3_visual_robustness_verify_replay.py'` | 0 |
| 3 | `PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/y3_visual_robustness_verify_retest.py` (30 calls, 0 API errors, 0 parse failures) | 0 |
| 4 | `PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/y3_visual_robustness_verify_strata.py` | 0 |
| 5 | 2 inline checks: input hash / pixel identity of orig vs cam_agentview, and the per-task distribution of re-render PSNR. The first attempt exited 1 because the system python lacks `imageio.v2`; rerun in Env B, it exited 0. | 1 → 0 |

`y3_analyze.py` was not rerun, because it overwrites the existing result files. Instead, step 1 recomputed the same numbers independently.

## Results

### 1. Integrity and reproduction

- **Row integrity.** Both local and nebius have 600 rows. 0 duplicates, 0 missing, 0 infra errors. Local parsing is 600/600 ok; nebius is 598 ok with 2 invalid_json (both eye-in-hand square failures).
- **Truth agreement.** 0 rows have a truth that disagrees with the current summary.json or the npz `success.any()`.
- **Nebius evidence.** There are 600 calls with the y3 tag and 600 tags, all unique. 0 errors, and every row's request_id is in the log. Temperature 0.2, max_tokens 700 and pad 20.1 s are identical in 600/600. There is only 1 prompt hash per task, so the frozen prompt was the same in every condition.
- **Recomputed numbers.** **0** values differ from the published JSON. The acc, bal, TPR, TNR and Wilson of every condition, every paired Δ and the number of discordant pairs all matched. The checkpoint-cluster CIs are also effectively the same (e.g. Nemotron birdview [−0.25, −0.04] vs published [−0.25, −0.04]).
- **No sign of cherry-picking.** Frozen accuracy per stratum is similar between the subset and the universe.

  | backend | stratum | universe accuracy (n) | subset accuracy (n) |
  |---|---|---|---|
  | local | can/dropped | 0.385 (13) | 0.375 (8) |
  | local | square/dropped | 0.50 (10) | 0.44 (9) |
  | local | square/no_grasp | 0.69 (13) | 0.67 (9) |
  | Nemotron | square/none | 0.979 (47) | 0.917 (12) |

  The remaining strata are 1.0 in both or nearly the same.
- **Selection rule.** It does not read VLM outputs (checked in code). The prompts and parameters are all frozen in the protocol, and there is no trace of anything being chosen on eval. No leakage or dev/eval confusion was found.

### 2. Replay determinism: confirmed

| check | result |
|---|---|
| y3 log 60/60 | eef / obj max error 0.0, grasped / success step agreement 1.0, success flag 60/60, frame count 60/60 match |
| this verification's CPU replay (no renderer) 6/6 | eef / obj error 0.0, success agreement 6/6 (can 3, square 3, lengths 130–400 steps) |
| negative control: action +1e-3 (step 20) | from step 21, eef error 1.5e-4 m, obj error 2.5e-4 m. The check has detection power. |
| negative control: seed +1 | obj error 0.68 m, success outcome flips |
| re-rendered mp4 byte-identical to the original | square 30/30 identical (Nemotron input hash and local last PNG are pixel-identical). can 30/30 differ slightly (PSNR of the re-encoded last frame 53–71 dB, median 67 dB). |

The interpretation is as follows.

- The claim that the *physical state* replay is bit-identical at the float32 probe level holds.
- The *rendering* is bit-identical only for square. can is near-identical. It had no effect on the judges (local predictions identical 60/60).
- The `env.close()` pitfall was contrasted on the same first 3 episodes. In the version that called close, 2/3 were 13–19 dB; in the debug version (no close) the first frame was 42.1–42.2 dB in 3/3. That is the extent of the evidence; the mechanism is not confirmed.
- For reference, `rollout_recorder.py` also calls `env.env.env.close()` at the end of a batch. However, the current eval and dev data have only 1 batch per checkpoint (batch_index 0), and the eval loop runs with `--max-checkpoints 1`. So the current data are most likely unaffected. The dev process layout was not checked.

### 3. Noise floor: confirmed, and even stronger

- The Nemotron orig rerun differed from the frozen pad201 run on 4 episodes (56/60). All 4 are actual failure episodes.
- 4 also flipped between orig and cam_agentview. **3 of them had compressed videos sent to Nemotron whose sha256 was byte-identical**. That is pure sampling noise at temperature 0.2.
- local has a fixed seed, so it was identical 60/60. The raw text was identical 57/60.

### 4. Paired Δ: corrected intervals, alternative clusters, reweighting

Column descriptions:

- Δ is condition accuracy − control accuracy on the subset. The control is orig for image-quality conditions and cam_agentview for camera conditions.
- The CI columns (ckpt 95%, Bonferroni, seed 95%) are intervals for this subset Δ. Bonferroni is the checkpoint-cluster interval corrected for 18 comparisons; seed 95% is the env seed cluster interval.
- "+FN / +FP" are the increases in FN and FP over the control. The failure-side counts include parse failures.
- "Reweighted Δ" is post-stratified to the eval universe composition (can/none 142, square/none 47, can/no_grasp 25, can/dropped 13, square/no_grasp 13, square/dropped 10; success 0.756). The reweighted acc and judged success rate use the same weights.

| backend | condition | Δ (subset) | ckpt 95% | Bonferroni | seed 95% | Holm p | +FN / +FP | reweighted Δ [95%] | reweighted acc | reweighted judged success rate |
|---|---|---|---|---|---|---|---|---|---|---|
| local | down224 | +0.050 | [0.00, 0.09] | [0.00, 0.11] | [0.00, 0.10] | 1.000 | −1 / −2 | +0.027 [0.00, 0.06] | 0.912 → 0.940 | 0.812 → 0.817 |
| local | down112 | −0.017 | [−0.07, 0.04] | [−0.10, 0.06] | [−0.08, 0.03] | 1.000 | −1 / +2 | +0.005 [−0.02, 0.04] | 0.912 → 0.918 | 0.812 → 0.838 |
| local | jpeg20 | −0.033 | [−0.11, 0.04] | [−0.15, 0.07] | [−0.10, 0.03] | 1.000 | +1 / +1 | −0.021 [−0.07, 0.03] | 0.912 → 0.891 | 0.812 → 0.802 |
| local | blur | +0.033 | [0.00, 0.07] | [0.00, 0.09] | [0.00, 0.08] | 1.000 | −1 / −1 | +0.021 [0.00, 0.06] | 0.912 → 0.934 | 0.812 → 0.822 |
| local | dim | −0.017 | [−0.06, 0.00] | [−0.09, 0.00] | [−0.06, 0.00] | 1.000 | 0 / +1 | −0.004 [−0.01, 0.00] | 0.912 → 0.908 | 0.812 → 0.817 |
| local | frontview | −0.100 | [−0.23, 0.02] | [−0.30, 0.06] | [−0.18, −0.03] | 1.000 | −1 / +7 | −0.020 [−0.05, 0.02] | 0.912 → 0.892 | 0.812 → 0.864 |
| local | birdview | −0.167 | [−0.33, 0.00] | [−0.43, 0.07] | [−0.24, −0.09] | 0.340 | −1 / +11 | −0.050 [−0.10, 0.00] | 0.912 → 0.862 | 0.812 → 0.894 |
| local | eye-in-hand | −0.283 | [−0.45, −0.15] | [−0.54, −0.09] | [−0.43, −0.10] | 0.026 | +10 / +7 | −0.229 [−0.29, −0.17] | 0.912 → 0.683 | 0.812 → 0.729 |
| Nemotron | down224 | −0.033 | [−0.11, 0.05] | [−0.15, 0.10] | [−0.12, 0.04] | 1.000 | −1 / +3 | +0.004 [−0.03, 0.05] | 0.946 → 0.950 | 0.779 → 0.806 |
| Nemotron | down112 | −0.033 | [−0.10, 0.03] | [−0.13, 0.07] | [−0.09, 0.03] | 1.000 | 0 / +2 | −0.005 [−0.03, 0.02] | 0.946 → 0.941 | 0.779 → 0.784 |
| Nemotron | jpeg20 | −0.083 | [−0.17, 0.00] | [−0.22, 0.05] | [−0.17, 0.00] | 1.000 | 0 / +5 | −0.026 [−0.06, 0.01] | 0.946 → 0.920 | 0.779 → 0.805 |
| Nemotron | blur | +0.017 | [−0.05, 0.09] | [−0.09, 0.13] | [−0.04, 0.10] | 1.000 | −1 / 0 | +0.019 [−0.02, 0.06] | 0.946 → 0.965 | 0.779 → 0.791 |
| Nemotron | dim | +0.017 | [−0.04, 0.08] | [−0.07, 0.11] | [−0.04, 0.08] | 1.000 | 0 / −1 | +0.010 [−0.01, 0.03] | 0.946 → 0.956 | 0.779 → 0.769 |
| Nemotron | frontview | −0.067 | [−0.16, 0.03] | [−0.20, 0.09] | [−0.21, 0.05] | 1.000 | 0 / +4 | −0.032 [−0.09, 0.03] | 0.952 → 0.920 | 0.773 → 0.805 |
| Nemotron | birdview | −0.150 | [−0.25, −0.04] | [−0.30, 0.02] | [−0.29, 0.00] | 0.527 | +4 / +5 | **−0.223 [−0.39, −0.08]** | 0.952 → **0.728** | 0.773 → **0.618** |
| Nemotron | eye-in-hand | −0.300 | [−0.45, −0.18] | [−0.55, −0.12] | [−0.34, −0.25] | 0.016 | +20 / −2 | −0.649 [−0.74, −0.53] | 0.952 → 0.302 | 0.773 → 0.088 |

Under LOCO (recompute leaving one checkpoint out), no camera effect changes sign. The ranges are:

| backend | frontview | birdview | eye-in-hand |
|---|---|---|---|
| local | [−0.12, −0.04] | [−0.20, −0.09] | [−0.33, −0.25] |
| Nemotron | [−0.09, −0.04] | [−0.17, −0.13] | [−0.32, −0.26] |

Split by task:

- The local frontview and birdview losses come almost entirely from square. The square Δ are −0.23 / −0.30 and the can Δ are +0.03 / −0.03.
- The Nemotron birdview loss comes from both can (−0.17) and square (−0.13).

### 5. Nemotron birdview retest (30 calls)

| disagreement type in y3 | n | still holds on retest | agentview rerun result | birdview rerun result |
|---|---|---|---|---|
| agentview right, birdview wrong | 12 | **7** | still right 11/12 | still wrong 8/12 |
| birdview right, agentview wrong | 3 | 2 | still wrong 3/3 | still right 2/3 |

- Of the 7 that held, **4 are FNs that saw a can success as `no_grasp`, and all 4/4 came back.** The explanation was also the same: "never closes around it… remains in the left bin".
- In contrast, of the 8 birdview FPs that saw a failure as a success, only 3 came back.
- Rerunning only the 15 discordant episodes, the net effect shrank from −9 to −5 episodes. Keeping the y3 values for the 45 concordant ones, it is roughly −0.08. This is a conservative estimate because of the selection bias.
- In conclusion, the direction of the Nemotron birdview effect reproduces. The component that reproduces is the **FN (success seen as failure)**, and the FP component is mostly indistinguishable from noise.

### 6. Counts cited in the report

| citation | verification | notes |
|---|---|---|
| local frontview failed square 16/18 `on_peg`, birdview 18/18 | matches | 8/18 were already `on_peg` in agentview. The increase is +8 / +10. In birdview it answered `on_peg` for all of square 30/30 (constant output, square bal acc 0.50). |
| local birdview failed can 10/18 `in_gripper` | matches | agentview 3/18 |
| local eye-in-hand failed can 14/18 `right_bin`, succeeded square 9/12 `not_visible` | matches | |
| Nemotron eye-in-hand judged 21/24 successes as failures (TPR 0.12) | matches | |
| "judged 88% of successes (21/24) as 'did not grasp / did not lift'" in the summary (originally in Korean) and the structured findings | **wrong** | The failure types of the 21 are 13 `no_grasp` and 8 `not_placed` (all square). "Did not grasp" is 13/24 (54%). |
| Nebius 600 calls, 0 API errors, 2 invalid JSON (eye-in-hand) / local 600 calls, 0 OOM, 0 parse failures | matches | There was a stretch where two Nebius client processes (4 + 3 threads) ran at the same time. That is more than the report's "3–5 threads", but not a constraint violation. |
| frozen accuracy local 0.767 / Nemotron 0.850, orig rerun agreement 60/60 / 56/60 | matches | |

## Interpretation: verdict per claim

1. **Replay determinism: confirmed.** Supported by independent CPU reproduction and negative controls. Rendering is not bit-identical for can, but it had no effect on the judges.
2. **Image degradation has almost no effect: confirmed.** It should be read as "no detectable effect", though. Single severity, n = 60, and Nemotron downsizes the input to 320 px. After reweighting, every Δ is within ±0.03. The sentence "the only meaningful loss is strong JPEG" is an overstatement. Nemotron jpeg20 has Bonferroni interval [−0.22, 0.05], Holm p = 1.0, and reweighted Δ −0.026.
3. **"Camera placement is decisive" (headline): partially confirmed.**
   - The wrist-camera effect stays strong in every analysis. It is, however, confounded with the frozen prompt.
   - For frontview, the checkpoint-cluster CI includes 0 for both backends. The report's limitations also say "no detectable effect". Yet the headline table groups it as evidence for "decisive".
   - Before correction, birdview excludes 0 only for Nemotron. After Bonferroni, it includes 0 for both backends.
   - Switching the cluster unit to env seed makes local frontview and birdview significant and Nemotron birdview borderline. Significance is sensitive to analysis choices.
4. **"Far cameras inflate VLM-judged success by seeing failures as successes": depends on the backend. Refuted for Nemotron birdview.**
   - It holds for Cosmos. The reweighted judged success rate rises from 0.81 to 0.86 (frontview) and 0.89 (birdview) (actual 0.756).
   - The Nemotron birdview increment is +4 FN / +5 FP. What reproduced on retest is the FN, and in the eval composition the judged success rate *falls* from 0.77 to 0.62.
   - Nobody measured whether a uniform inflation or deflation of the judged success rate changes the checkpoint *ranking*. So "directly biases the hybrid screening score" is an unsupported extension.
5. **Headline sizes ("birdview −0.15 to −0.17, wrist −0.28 to −0.30"): apply only to this subset.** Reweighted to the eval composition, the sizes change a lot.
   - Cosmos: frontview −0.02, birdview −0.05, wrist −0.23.
   - Nemotron: frontview −0.03, birdview −0.22, wrist −0.65.
   - The reweighted values are uncertain because the most heavily weighted stratum, can/none (weight 0.57), is estimated from 12 episodes. Still, the difference in direction is clear. The report said "paired differences are comparable", but because the effects differ across strata, the paired Δ also depends on the sample composition.
6. **"Nemotron ≥ Cosmos in every condition": true only on the subset.**
   - Reweighted to the eval composition, **Cosmos leads** at birdview, 0.728 vs 0.862, and at wrist, 0.302 vs 0.683.
   - Even on the subset, the wrist advantage (0.583 vs 0.483) comes from the failure share being 0.60, because Nemotron answered failure on 92%. In bal acc it is a tie, 0.507 vs 0.493.
   - In addition, on this subset a constant predictor that always answers "failure" has accuracy 0.60. Cosmos birdview (0.600) and both models' wrist accuracy are at or below this baseline.
7. **Error mechanism 1 (answering `on_peg` because the square nut is small): mostly confirmed.** In the figures the nut is also a few pixels in size. However, 8/18 were already `on_peg` in agentview. In birdview, the square output was constant.
8. **Error mechanism 2 (wrist): confirmed.** However, the statement that Nemotron judged '88% did not grasp' is wrong (see the table above).
9. **Noise-floor claim: confirmed, and strengthened.** Judgments flip even on byte-identical inputs.
10. **Claim about viewpoint-dependent wording: confirmed.** In the figures, the left/right bin layout of frontview and birdview matches the prompt.

## Limitations

- This verification is also post-hoc and exploratory. Multiple-comparison corrections are used as context only, not as a criterion for rejecting exploratory results.
- The reweighted CIs are a per-stratum episode bootstrap and ignore checkpoint clustering. The 12 can/none episodes represent weight 0.57.
- The retest reran only the discordant pairs (budget 30 calls). New flips that could occur in the 45 concordant pairs were not measured. Because discordant pairs were selected, the retest Δ is biased toward 0 by regression to the mean.
- Rendering determinism was not checked by re-rendering on CPU (EGL and osmesa differ in pixels). It was judged only from the y3 log and input hash identity.
- For the `env.close()` pitfall, y3's original check-version code no longer exists. So it cannot be checked whether anything other than the close call differed.
- Seed-to-seed variation of the local judge (other generation seeds) was not measured. The small local Δ (±0.05) are therefore results for a single seed.
- The repo AGENTS.md requires commit/push, but per the task constraint (no git commit) this was not done.

## Next steps

1. Fix the headline and usefulness wording of the y3 report.
   - State that "far camera = inflated success" applies only to Cosmos.
   - Write that Nemotron birdview misses can lifts, producing FNs.
   - Change "camera placement is decisive" to "the wrist camera (frozen prompt) is certain; far cameras range from −0.02 to −0.22 depending on the backend and the sample composition".
   - Change "88% no_grasp" to "21/24 judged failures (no_grasp 13, not_placed 8)".
2. Report camera effects as values reweighted to the eval composition and as checkpoint-level rankings by judged success rate (Spearman). The screening impact should be assessed by ranking.
3. For Nemotron, repeat all 60 cam_agentview and cam_birdview episodes 2 more times and compute the majority-vote Δ (about 240 calls). Keep only effects above the noise floor measured earlier. The noise floor is 4/60 (6.7%) versus the frozen run, and 3 (10%) flipped among the 30 byte-identical inputs.
4. Keep the judging-camera recommendation ("a close, oblique, fixed third-person camera"). Change its rationale to "far cameras create biases in different directions for each backend". Also state that changing the camera requires redoing dev selection per backend.
5. If `rollout_recorder` calls `close()` while running several batches or checkpoints in one process, rendering can break. Hand this risk over as a KNOWN_RISKS candidate. The current eval data have 1 batch and 1 process per checkpoint, so they are most likely unaffected.

## Files

- **Scripts:** `scripts/hackathon/explore/y3_visual_robustness_verify.py`, `y3_visual_robustness_verify_replay.py`, `y3_visual_robustness_verify_retest.py`, `y3_visual_robustness_verify_strata.py`
- **Results:** `results/hackathon/explore/y3_visual_robustness_verification.json` (rescoring, corrected intervals, cited counts), `results/hackathon/explore/y3_visual_robustness_verification_strata.json` (strata, reweighting, seed clusters, retest rows)
- **Nebius evidence:** `results/hackathon/explore/nebius_calls_explore.jsonl`, tag `explore/y3_visual_robustness_verification/retest/…` (30 calls)
- **Scratch:** `data/outputs/hackathon_explore/y3_visual_robustness_verification/` (`verify.log`, `replay_cpu.jsonl`, `replay_cpu.log`, `retest_nebius.jsonl`, `retest.log`, `strata.log`)
