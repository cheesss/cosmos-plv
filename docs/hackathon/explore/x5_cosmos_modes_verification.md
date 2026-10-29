# x5_cosmos_modes: adversarial verification

**Status: PARTIALLY CONFIRMED. EXPLORATORY / POST-HOC.** This note checks the x5 report
(`docs/hackathon/explore/x5_cosmos_modes.md`). Every new number below comes from
`results/hackathon/explore/x5_cosmos_modes_verification.json`, produced by
`scripts/hackathon/explore/x5_cosmos_modes_verify.py`, or from the two text files named next to the number. Other
numbers are quoted from x5's own results JSON, which I reproduced exactly. I changed none of x5's files.
I made **0 new Nebius calls** and **no GPU runs**. Every VLM result was re-scored from the saved raw outputs
(`data/outputs/hackathon_explore/x5_cosmos_modes/calls.jsonl`, 965 rows).

> Summary: x5's **core negative conclusion holds.** None of the 4 Cosmos3-Edge usage modes tested beats v6 (last frame, about 2 seconds). Rerunning the analysis script with only the output paths changed **reproduces with 0 differences** every number in the tables, the dev selection result and the rule ablation result. Re-parsing the 965 saved raw outputs with the current parser gives the same results. There is no eval leakage. Choices were made on dev only and then applied to eval unchanged, and the frame-level "held" conclusion holds when the ground-truth definition is changed or the frame↔step alignment is shifted by ±2 steps.
> However, the following need fixing.
> (1) **The square z0 label problem had already been fixed.** `failure_annotator` uses the settled height (minimum over steps 0–10) since 05:15 (commit `baa3967`). `refresh_truth.py` updated the labels at 05:15:26 (15 square other→no_grasp, 2 no_grasp→dropped). x5 analyzed at 05:51 with the already-updated labels, and the current labels contain 0 'other'. The statements "z0 is taken at step 0" and "the lift threshold is effectively 11cm", and the recommendation to add a KNOWN_RISKS entry, are out of date. The observation of a 6.57cm drop itself is correct (all 92 square episodes).
> (2) **The pairwise 'up' recall of 0/2 (dev) and 0/11 (eval) was computed only on the pairs that passed strict parsing (44/53).** The evaluated variant is the format-repaired one, so I recounted on all repaired pairs: dev 0/17, eval 2/35 (precision 2/2). The conclusion is the same, and even stronger.
> (3) **The pairwise prompt leaks success in its text.** The prompt's "X seconds apart" is always 2.8–2.9 seconds in failed episodes and 0.7–2.2 seconds in successful ones. This is because episodes end 16 steps after success; video length alone identifies success on dev 36/36 and eval 270/270. The model did not exploit this, so no result is inflated. But it is a design flaw, so recommendation 2 (Nebius video second-stage check) should only be done with the length-neutral version (pad201).
> (4) **The 'always no_grasp' baseline is an oracle that knows which episodes are failures.** A fair baseline that uses the same success judge is 0.576 on the 59 eval v7 failures (x5: 0.644). R2's 0.305 is still far lower, and in per-class mean recall it is 0.258, below the chance level of 0.5. The conclusion holds.
> (5) **I ran the Nebius majority-class check that x5 recommended on the same failure set.** On the 61 eval_nebius_v2 failures, Nebius scores 0.344 [0.24, 0.47] and always-no_grasp 0.623. For the pad201 version it is 0.444 vs 0.704. Replacing x5's "different sets 0.40 vs 0.64" comparison with a same-set comparison makes the conclusion clearer.
> (6) ROI's +0.08 on dev came from **2 episodes of one checkpoint (dev_square_s1_e0020)**. The CI of eval's −0.077 [−0.19, 0.00] also includes 0. The correct statement is not "worse than v6" but "no gain, at 8x the cost". The eval sample is a single env_seed, 200000, and this seed is hard for v6 too (0.80, other seeds 0.933).
> (7) Minor issues: the "earlier CPU check" 48.4% (eval 520 frames) is not saved in any file. The dev 47.1% reproduces exactly. The denominator of (b)'s "52.6% of midpoint frames" is 'not-held frames' (over all midpoint frames it is 46.3%). The "39-53%" in the summary at the top of x5 (originally written in Korean) differs from the "39-65%" in the body.

## What was checked and how

| check | method | outcome |
|---|---|---|
| Reproduction | `x5_analyze.main_with_frames()` and `x5_rule_ablation.main()` re-run with `OUT` paths redirected to `data/outputs/hackathon_explore/x5_cosmos_modes_verification/` | **0 metric diffs** (own-episode metrics, paired diffs, frame_level block, selection dev scores, rule-ablation dev/eval metrics and paired diff). `eval_local_v7.jsonl` has not grown since 05:20 (170 rows), so the 170-episode ablation reproduces exactly. |
| Raw re-score | every `calls.jsonl` row re-parsed with the current `parse_pair`, `parse_pair_repaired`, `parse_perception` | 0 mismatches for a/b/d. c: strict status identical (97 ok, 218 missing_fields, 41 no_json, 1 invalid_json). `x5_common.py` was edited at 05:12, after the GPU process started at 05:08, and this check shows the edit did not matter. |
| Leakage (score fit on target) | read the selection code and run_state | None. The choice used dev only (`selection_in_run` == recomputed). Eval was run after selection with the chosen variants unchanged. The mid-dev amendment (adding `c_pairwise8_fmtrepair`) used dev outputs only. |
| Truth labels | JSONL truth vs refreshed `summary.json` (`jsonl_truth_consistency.txt`) | 0 mismatches in v6 (250), v7 (170), dev_prompt_selection (228), nebius_v2 (250). Current annotator vs summary labels: 0 mismatches. |
| Frame/step alignment | `video_frames == (n_steps−1)//2 + 1` for all 270 eval episodes (`jsonl_truth_consistency.txt`), plus ±2-step offset sensitivity | Mapping k → 2k is consistent. Offsets change held precision by ≤ 0.05 (table below). |
| Clip-length leakage | length-only classifier; time text in the c prompt | Length alone gives success with 100% accuracy (dev 36/36, eval 270/270). The c prompt text separates the outcome perfectly. The model did not exploit it (see issue 3). a/b/d prompts contain no time text. |
| Dependence | checkpoint-cluster bootstrap; seed composition | Eval: 25 of 26 at env_seed 200000. Dev: 3 checkpoints. See issue 6. |
| Baseline fairness | same-detector majority baseline, per-class recall | See issue 4. |
| ROI bug? | rendered the last frames of the eval square misses with the ROI box (`square_fn_lastframes_with_roi.png`) and viewed the crop | The nut on the peg is fully inside the crop and clearly visible. a still answered `not_visible`. This is a perception error, not a clipping bug. |

## Claims: verdicts

| x5 claim | verdict | evidence |
|---|---|---|
| (a) ROI: dev 0.92 vs v6 0.84 (2 fixed / 0 broken); eval 0.73 vs 0.81, paired −0.077 [−0.19, 0.00], FP 4/8 both, FN 3/18 vs 1/18, 16.5 s vs 2.15 s | **Confirmed (numbers)**; wording too strong | Reproduced exactly. Both dev-fixed episodes are `dev_square_s1_e0020` (seeds 300200, 300207). Both eval-broken episodes are square successes (`square_s1_e0060`, `square_s1_e0090`) where the crop answer was `not_visible`. All 3 eval misses from a are square `not_visible`. The CIs include 0 on both splits, so "no measurable gain at about 8x cost" is supported and "worse than v6" is not. |
| (b) 15 keyframes: success = v6; failure type 0.19 vs v7 0.25; 31 s; 52.6% false held | **Confirmed**, with a denominator fix | 0 discordant with v6 on dev. 52.6% = 81/154 not-held midpoint frames. As a share of all 175 midpoint frames it is 46.3%. |
| (c) 260/357 (73%) not in the requested format, bracket lists; fmtrepair ft dev 0.625, eval 0.50 = majority; "up" 3/354; up recall 0/2 dev, 0/11 eval; success eval 0.54 | **Partially confirmed** | 260/357 confirmed. All 260 strict failures have `[` as the first payload character. 354 repaired, `up` 3, `stayed` 238 confirmed. The up-recall figures come from strictly parsed pairs only (44 dev / 53 eval). On all repaired pairs it is dev 0/17 and eval 2/35 (both eval "up" answers correct). On eval the rule's lift event never produced a non-no_grasp label (confusion: no_grasp→no_grasp 4, dropped→none 4). The prompt also leaks the outcome through its time gap (issue 3). |
| (d) reasoning: 9/25 unparsed, 3 at the 1024-token cap, 14/16 correct when parsed, 0.56, paired −0.28 [−0.48, −0.12], 18.2 s | **Confirmed** | 3 calls at exactly 1024 completion tokens. Status counts: ok 14, ok_repaired 2, missing_fields 5, no_json 4. 3-cluster bootstrap [−0.33, −0.25] (only 3 clusters, so indicative only). |
| Root cause: held precision 0.17–0.43, 39–65% false held on not-held frames; v7 47.1% dev (288) / 48.4% eval (520) | **Confirmed and robust**; one number unsupported | Holds under the alternative truth definitions below and on all repaired c pairs (eval image-2 false held 65.4% on 182 pairs vs 64.9% on 53). Dev 47.1% (115/244) reproduces exactly. The eval 48.4% / 520-frame figure is in no x5 file and cannot be reproduced now: on the current 170 v7 episodes (1360 frames) the rate is 49.2%. |
| Rule ablation: R2 chosen on dev, eval 0.186 → 0.305 (+0.119 [0.00, 0.24], 10/3), below always-no_grasp 0.644 | **Confirmed**; baseline is an oracle | Reproduced exactly. The same-detector baseline is 0.576 [0.45, 0.69]. Per-class (macro) recall: R0 0.198, R2 0.258, always-no_grasp 0.5. Every keyframe rule is below even the chance-level macro recall. |
| Square z0 label artifact: annotator takes z0 at step 0; never-grasped square can be `other`; lift threshold effectively about 11 cm; could not add to KNOWN_RISKS | **Refuted as a current claim** | Fixed in `cosmos_plv/failure_annotator.py` (commit `baa3967`, 05:15:38): `z0 = min(z[0:11])`, and lift is masked for the first 10 steps. Labels were refreshed at 05:15:26 (square `other→no_grasp` 15, `no_grasp→dropped` 2, from `summary.orig.json` vs `summary.json`). Current labels have 0 `other`. The spawn drop is 6.57 cm in all 92 square episodes, so the physical observation is right. The merged "square other = no_grasp" view in x5's JSON is a no-op. |
| Nebius v2 0.40 vs majority 0.64 on eval v7 failures (different sets) | **Superseded** by a same-set check | On the same failures: `eval_nebius_v2.jsonl` 61 failures, 0.344 [0.237, 0.470] vs always-no_grasp 0.623 [0.497, 0.734], macro recall 0.285. `eval_nebius_v2.orig.jsonl` keys (55 failures): 0.382 vs 0.691. Length-neutral `eval_nebius_v2_pad201.jsonl` (54 failures): 0.444 [0.320, 0.576] vs 0.704. Unscored rows count as wrong. |

### Frame-level "held" sensitivity (precision / false-held rate on not-held frames)

| truth definition | dev v7 full (200) | dev a crop (200) | eval v7 full (144) | eval a crop (208) | eval c image-2, all repaired (182) |
|---|---|---|---|---|---|
| x5: contact AND ≥ 2 cm above rest | 0.204 / 0.446 | 0.236 / 0.389 | 0.425 / 0.422 | 0.406 / 0.385 | 0.361 / 0.654 |
| contact only | 0.235 / 0.436 | 0.270 / 0.378 | 0.425 / 0.422 | 0.436 / 0.373 | 0.383 / 0.646 |
| ≥ 2 cm, no contact needed | 0.235 / 0.472 | 0.270 / 0.409 | 0.500 / 0.408 | 0.485 / 0.371 | 0.429 / 0.628 |
| x5 def, frame k → step 2k−2 | 0.163 / 0.458 | 0.191 / 0.402 | 0.438 / 0.417 | 0.436 / 0.373 | 0.383 / 0.646 |
| x5 def, frame k → step 2k+2 | 0.194 / 0.446 | 0.213 / 0.395 | 0.412 / 0.427 | 0.406 / 0.385 | 0.361 / 0.654 |

The held-perception failure is not an artifact of x5's truth definition or of a frame/step offset.

## Issues in detail

1. **Stale label-artifact finding (refuted).** See the claims table. The x5 report and its structured summary
   present this as an open risk that needs owner action. It was fixed and the labels re-annotated about 36 minutes
   before x5's analysis ran, and x5's own numbers already use the fixed labels. Recommended-next item 6 is moot.
2. **Biased subset for the c motion and held statistics.** `x5_analyze.frame_level` filters c pairs with the cached
   strict `parse_status`, which keeps 27% of pairs (97/357). The headline variant is the repaired parser. The
   corrected numbers point the same way (dev up recall 0/17, eval 2/35).
3. **Outcome leakage through the c prompt.** `build_pair_prompt(task, dt_s)` writes `(j1−j0)/fps` into the text.
   Failures always run 400 steps (201 frames), so dt is 2.8–2.9 s. Successes end 16 steps after success, so dt is
   0.7–2.2 s. No c result is inflated by this: its success accuracy is worse than v6 and its failure types never
   use it. But any future multi-frame or video prompt (including the recommended "Nebius video judge" cascade) must
   be length-neutral, as `eval_nebius_v2_pad201.jsonl` already is. Note also that, in simulation, success is free
   from the simulator flag or the clip length. A VLM success judge only matters for real-robot or unlabelled video.
4. **The majority reference is an oracle.** Always-no_grasp is scored on true failures, so it knows which episodes
   failed. A detector-matched reference (no_grasp whenever the variant predicts failure) is lower: eval v7 0.576 vs
   0.644, dev v7 0.476 vs 0.619. For a and c it is unchanged on eval (0.50). Conclusions survive. Every local Cosmos
   keyframe rule and Nebius v2 are also below the 0.5 macro-recall of always-no_grasp.
5. **Claim strength.** a vs v6: dev +0.08 [0.00, 0.20] (2 episodes, 1 checkpoint), eval −0.077 [−0.19, 0.00]
   (2 episodes, both square). Neither direction is established. The cost (16.5 s vs 2.15 s per episode on eval) is
   the decisive fact. The c "+0.57 over v7" is correctly flagged by x5 as a majority-class artifact.
6. **Representativeness and dependence.** The eval sample is 25 checkpoints × env_seed 200000 plus 1. v6 accuracy is
   0.80 [0.61, 0.91] on seed 200000 vs 0.933 [0.89, 0.96] on the other 225 eval episodes and 0.92 overall
   (`v6_seed_subset.txt`). The eval subset is harder than average. A checkpoint-cluster bootstrap on eval (25
   clusters) gives the same CI as x5's episode bootstrap. On dev, with 3 clusters, no valid cluster CI exists.
7. **Minor.** An unsaved number (48.4% / 520 frames). Inconsistent ranges between the Korean summary and the body
   ("39-53%" vs "39-65%"; "precision 0.2-0.4" vs 0.17-0.43). The (b) denominator wording.

## Commands run (cwd `/path/to/repo`) and exit codes

```
PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 ~/.venvs/cosmos-plv/bin/python \
  scripts/hackathon/explore/x5_cosmos_modes_verify.py \
  > data/outputs/hackathon_explore/x5_cosmos_modes_verification/verify_run.log 2>&1        # exit 0
# inline Env B snippets (read-only), outputs saved in the same scratch dir:
#   square miss last frames + ROI overlay -> square_fn_lastframes_with_roi.png              # exit 0
#   v6 accuracy by seed subset            -> v6_seed_subset.txt                              # exit 0
#   JSONL truth vs summary.json; frame-count identity -> jsonl_truth_consistency.txt         # exit 0
```

The verify script imports x5's modules and redirects their output paths, so x5's results files are not
overwritten. Its outputs: `results/hackathon/explore/x5_cosmos_modes_verification.json` and
`data/outputs/hackathon_explore/x5_cosmos_modes_verification/rerun_x5_cosmos_modes{,_episodes,_rule_ablation}.json[l]`.

## Corrected headline

(Exploratory, post-hoc analysis, reproduced) None of the 4 Cosmos3-Edge usage modes is clearly better than v6. ROI crop was +0.08 on dev. That difference came from 2 episodes and 1 checkpoint. On eval it was −0.077 [−0.19, 0.00]. Neither difference is significant, and the cost is 8x. The pairwise mode only imitates the majority class, and the time gap in the prompt leaks success. The reasoning mode failed to parse 9/25. Keyframe-based failure type is below the fair baseline (0.576), and also below answering only 'no_grasp' in per-class mean recall. Nebius v2 is also lower on the same failure set, 0.344 vs 0.623. The square z0 label problem had already been fixed before the x5 analysis.

## What to do with x5's recommendations

- #1 (majority check before using VLM failure types): **done here for Nebius v2**. It fails the check on the same
  failures (0.344 vs 0.623, pad201 0.444 vs 0.704). Report failure-type results with per-class recall.
- #2 (success cascade with a second-stage check): only with a length-neutral input (`pad201`-style). Otherwise the
  second stage can read success from the clip length.
- #3 (lift/drop from state): agreed. The verified annotator already does this.
- #4 (strict JSON for pairwise): agreed. Also remove the time gap from the prompt text.
- #5 (longer eval): agreed. Use checkpoint-major or seed-stratified order, since seed 200000 is atypically hard.
- #6 (square z0 into KNOWN_RISKS): **drop**. Already fixed in code and labels.
