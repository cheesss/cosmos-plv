# x5_cosmos_modes: other ways to use local Cosmos3-Edge for physical diagnosis (EXPLORATORY, post-hoc)

**Summary.** I compared 4 ways of using Cosmos3-Edge on 25 of the 36 dev episodes: (a) region-of-interest crop + 2x upsampling, (b) 15 keyframes, (c) questions on pairs of consecutive keyframes, (d) last frame + reasoning. The method chosen on dev did not reproduce on the 26 eval episodes. The 26 eval episodes are in effect a single env seed, 200000. On dev, (a) ranked 1st on success judgment (0.92, v6 0.84). On eval it fell to 0.73, while v6 was 0.81 (0 fixed, 2 broken). (c) only scores higher than v7 on failure type after a post-hoc format repair. But that number is an illusion. The model answered "up" only 3 times out of 354 pairs, so in effect it gives the same answer as "always no_grasp". It equals the majority-class baseline (eval 0.50 = 0.50). (d) failed to parse 36% of its outputs and took 18 seconds per episode (v6 1.8 seconds). (b) actually got worse. The cause is at the frame level. Among frames where the model answers "holding", the fraction actually held (precision) is only 0.2-0.4. It answers "held" on 39-53% of frames that are not held. So keyframe-based failure type is always below always-no_grasp (eval 0.64), whatever rule is used. There are two conclusions for PLV. Local Cosmos3-Edge is used only for last-frame success (v6). For failure type, use the simulator labels or Nebius.

## Question

Can a different way of feeding the same local model (`nvidia/Cosmos3-Edge` @ `344d602b128d1bbdacb43b08d0a3626f46343e29`, bf16, RTX 3090) diagnose rollouts better than the current v6 and v7 modes? v6 is last-frame perception: eval success accuracy 0.91, 12 false positives out of 53 failures. v7 is an 8-keyframe track: failure-type accuracy on dev failures 0.24. Four variants, chosen on dev only:

| id | what changes | model calls / episode |
|---|---|---|
| a_roi_track8 | The 8 v7 keyframes are cropped to a fixed region and upsampled 2x bicubic. Can box (x0,y0,x1,y1) = (0,40,448,336): both bins plus the carrying corridor. Square box = (16,96,432,448): the table with both pegs. Same v6 prompt and v7 rule. Vision tokens go from about 196 to about 600 per image (mean prompt tokens 763 vs 434). | 8 |
| b_track15 | 15 evenly spaced keyframes. That is v7's 8 plus the 7 midpoints. The 8 shared frames reuse the existing v7 per-frame outputs (identical frame index, prompt and seed 0). Same rule. | 15 (7 new) |
| c_pairwise8 | 7 questions, one per pair of consecutive v7 keyframes: object motion up/down/stayed/sideways, held in image 1/2, location in image 2. A rule fixed in advance turns the answers into lift and drop events. | 7 |
| d_reason_last | The v6 last-frame prompt with reasoning on (`<think>`), max 1024 new tokens. | 1 |

## Why it could help PLV or make it more efficient

PLV ranks checkpoints by cheap offline metrics and checks them against closed-loop success. A VLM that labels success and failure type from the recorded video could cut the labelling cost per rollout. It would also explain why a checkpoint fails (no grasp vs drop), which tells us which offline metric should track which failure. Each variant targets a known weakness of v6/v7: the object is too few pixels (a), events fall between keyframes (b), single-frame "held" judgements are unreliable (c), and the JSON contradicts the model's own explanation (d).

## Method

- **One GPU process** (`scripts/hackathon/explore/x5_cosmos_modes.py`). It ran under `flock /tmp/claude-1000/gpu_explore.lock` with `COSMOS_MAX_GPU_FRACTION=0.25`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and batch size 1. The model was loaded once, in 8.4 s. Free memory at start was 9609 MiB (threshold 9000). The lock was held by x6 from 04:50 to 05:08 KST.
- **Dev split.** Data: `hackathon_rollouts_dev`, seeds 300000+, used only for choosing a method. Episodes were processed in a seed-major round-robin across the 3 dev checkpoints. All 4 variants ran on each episode until a 28-minute cap. 25 of 36 episodes finished: 17 can, 8 square.
- **Selection rule (dev only).** The success choice is the variant with the best `balanced_acc_all`, where unparsed counts as wrong. The failure-type choice is the variant with the best failure-type accuracy on true failures. Ties go to lower latency.
  - **Rule amendment.** I changed the rule once, during the dev phase, before the selection step ran and without looking at eval. Cosmos3-Edge answered the pair prompt in a bracketed-list format most of the time (`[ "object_motion": "stayed", ... ]`), so I added `c_pairwise8_fmtrepair` to both candidate lists. It uses the same raw outputs and a post-hoc regex key-value extraction that changes format only, not values. This is recorded in the results JSON (`selection_in_run.rule`).
  - **Choices.** Success → `a_roi_track8`. Failure type → `c_pairwise8_fmtrepair`, which runs the `c_pairwise8` calls.
- **Eval split.** Data: `hackathon_rollouts_eval`, 250 episodes present at run time. The two chosen variants ran unchanged, in seed-major interleaved order, under a 14-minute cap. That covered **26 episodes**: env_seed 200000 for 25 checkpoints (can_s1 e0–e120, can_s2 e0–e200, square_s1 e0–e180) plus can_s1_e0000 seed 200001. The exact list is in `results/hackathon/explore/x5_cosmos_modes.json` → `eval_episodes_done`.
- **Scoring.** Labels are the simulator labels in `summary.json`. The conventions follow `summarize_diagnostics.py`. Every comparison is paired against v6 (`eval_local_v6.jsonl`, `dev_prompt_selection.jsonl`) or v7 (`eval_local_v7.jsonl`, in progress, 18 of the 26 eval episodes) on the same episodes. CIs are 95% Wilson; paired differences use an episode bootstrap (2000 resamples).
- **Frame-level check.** At each keyframe I compared the model's held / in_gripper answer with simulator truth. Truth "held" = finger-pad contact AND the object at least 2 cm above its resting height. Video frame k is sim step 2k.
- **CPU-only side analysis** (`x5_rule_ablation.py`, no new model calls). Three rules for combining frames were fixed in advance and applied to the existing v7 per-frame outputs: R0 is the v7 rule, R1 counts the object as held only when location is `in_gripper`, R2 requires held in at least 2 consecutive keyframes. The rule was chosen on dev v7 (36 episodes) and reported on eval v7 (170 episodes present).
- Nebius calls: 0. New Cosmos calls: 965 (25 reasoning, 408 crop, 357 pair, 175 midpoint). CUDA OOM events: 0.

## Commands run (cwd `/path/to/repo`) and exit codes

```
# fake-model smoke test of the pipeline in the scratchpad: first try ValueError (relative path for
# a scratchpad file), fixed with os.path.relpath; second try exit 0
flock /tmp/claude-1000/gpu_explore.lock env COSMOS_MAX_GPU_FRACTION=0.25 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 \
  bash -c '~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/x5_cosmos_modes.py --dev-minutes 28 --eval-minutes 14' \
  >> data/outputs/hackathon_explore/x5_cosmos_modes/run.log 2>&1          # EXIT 0 (05:08 -> 05:50 KST)
PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/x5_analyze.py        # exit 0
PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/x5_rule_ablation.py  # exit 0
PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/x5_report_tables.py  # exit 0
```

Re-run later (resumes from the call cache, then extends eval to new episodes):
`flock /tmp/claude-1000/gpu_explore.lock env COSMOS_MAX_GPU_FRACTION=0.25 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/x5_cosmos_modes.py --skip-dev --eval-minutes 30`, then the 3 CPU scripts above.

## Results

### Dev (25 episodes: 9 successes, 16 failures). This is the data the choice was made on.

| variant | unparsed | success acc (all) [Wilson] | false success on failures | missed successes | failure-type acc on failures [Wilson] | latency s/ep |
|---|---|---|---|---|---|---|
| v6 (existing) | 0 | 0.84 [0.65, 0.94] | 4/16 | 0/9 | - | 1.82 |
| v7 (existing) | 0 | 0.84 [0.65, 0.94] | 4/16 | 0/9 | 0.25 [0.10, 0.50] | 16.9 |
| **a_roi_track8** | 0 | **0.92** [0.75, 0.98] | 2/16 | 0/9 | 0.31 [0.14, 0.56] | 17.3 |
| b_track15 | 0 | 0.84 [0.65, 0.94] | 4/16 | 0/9 | 0.19 [0.07, 0.43] | 31.3 |
| c_pairwise8 (strict parser) | 17 | 0.24 [0.12, 0.43] | 12/16 | 7/9 | 0.19 [0.07, 0.43] | 15.9 |
| c_pairwise8_fmtrepair | 0 | 0.68 [0.48, 0.83] | 4/16 | 4/9 | **0.625** [0.39, 0.82] | 15.9 |
| d_reason_last | 9 | 0.56 [0.37, 0.73] | 7/16 | 4/9 | - | 18.2 |
| reference: always "no_grasp" on true failures | - | - | - | - | 0.688 [0.44, 0.86] | - |

Paired on dev:
- a − v6 success: +0.08, bootstrap CI [0.00, 0.20]. a fixed 2 episodes v6 got wrong and broke none.
- c_fmtrepair − v7 failure type: +0.375, CI [0.00, 0.75], 9 fixed / 3 broken.
- d − v6 success: −0.28, CI [−0.48, −0.12].

### Eval (26 episodes; 18 successes, 8 failures). The chosen variants were applied unchanged.

| variant | unparsed | success acc (all) [Wilson] | false success on failures | missed successes | failure-type acc on failures [Wilson] | latency s/ep |
|---|---|---|---|---|---|---|
| v6 (existing, same 26) | 0 | **0.81** [0.62, 0.92] | 4/8 | 1/18 | - | 2.15 |
| a_roi_track8 | 0 | 0.73 [0.54, 0.86] | 4/8 | 3/18 | 0.125 [0.02, 0.47] | 16.5 |
| c_pairwise8_fmtrepair | 0 | 0.54 [0.36, 0.71] | 4/8 | 8/18 | 0.50 [0.22, 0.78] | 14.5 |
| c_pairwise8 (strict parser) | 20 | 0.12 [0.04, 0.29] | 8/8 | 15/18 | 0.00 | 14.5 |
| reference: always "no_grasp" on true failures | - | - | - | - | 0.50 [0.22, 0.78] | - |
| v7 (existing; only 18 of the 26 have v7 rows) | 0 | 0.83 [0.61, 0.94] (n=18) | 3/7 | 0/11 | 0.00 [0.00, 0.35] (n=7) | 15.2 |

Paired on eval:
- a − v6 success: −0.077, CI [−0.19, 0.00]. a fixed 0 episodes and broke 2.
- c_fmtrepair − v6 success: −0.27, CI [−0.46, −0.12].
- c_fmtrepair − v7 failure type (7 shared failures): +0.57, CI [0.14, 0.86], 4 fixed / 0 broken. **But** c_fmtrepair scores exactly the always-"no_grasp" reference. Its confusion on eval is `no_grasp->no_grasp 4, dropped->none 4`.

### Why: per-keyframe "held" perception against the simulator

| source | frames | held precision [Wilson] | held recall | "held" on truly not-held frames |
|---|---|---|---|---|
| dev, v7 full frames | 200 | 0.20 [0.14, 0.29] | 0.80 | 44.6% |
| dev, a ROI crops (same frames) | 200 | 0.24 [0.16, 0.33] | 0.84 | 38.9% |
| dev, b midpoint frames | 175 | 0.17 [0.11, 0.26] | 0.81 | 52.6% |
| eval, v7 full frames | 144 | 0.43 [0.32, 0.53] | 0.97 | 42.2% |
| eval, a ROI crops | 208 | 0.41 [0.32, 0.50] | 0.79 | 38.5% |
| eval, c pair "held in image 2" | 53 | 0.37 [0.23, 0.53] | 0.88 | 64.9% |

The pairwise motion answer was "up" in only 3 of 354 parsed pairs, "stayed" in 238. Against simulator lifts of at least 3 cm between keyframes, recall of "up" was 0/2 on dev and 0/11 on eval. The lift detector in rule (c) never fires, so (c) labels every failure no_grasp.

Earlier CPU-only check on all existing v7 frames: false "held" on not-held frames was 47.1% on dev (288 frames) and 48.4% on eval (520 frames).

### CPU side analysis: rules for combining frames, on existing v7 outputs (no new calls)

| rule | dev failure-type acc (21 failures) | eval failure-type acc (59 failures) [Wilson] |
|---|---|---|
| R0 v7 rule | 0.238 | 0.186 [0.11, 0.30] |
| R1 held := location == in_gripper | 0.286 | 0.288 [0.19, 0.41] |
| **R2 held in ≥ 2 consecutive keyframes** (dev choice) | 0.381 | 0.305 [0.20, 0.43] |
| reference: always "no_grasp" | 0.619 | **0.644** [0.52, 0.75] |

R2 − R0 on eval: +0.119, bootstrap CI [0.00, 0.24], 10 fixed / 3 broken. It is a real improvement over the v7 rule, but it is still well below the trivial majority answer.

## Interpretation

- **ROI crop and upsampling (a): no reliable gain.** On dev it fixed 2 of v6's false successes. On eval it fixed none and missed 2 more successes. Cropping lowered false "held" only a little (about 39% vs 42–45% of not-held frames) and costs about 8x v6 latency. **Do not adopt it.**
- **More keyframes (b): worse.** Each extra frame is another chance for a false "held", and "ever held" saturates. Failure type fell from 0.25 to 0.19 on dev, at 2x v7 cost. **Negative.**
- **Pairwise questions (c): negative.** About 73% of raw answers break the requested JSON format (260 of 357 under the strict parser). The format repair recovers the values, but the model almost never reports upward motion, so the lift and drop logic is dead. The apparent failure-type gain over v7 (eval +0.57) only reproduces the majority class. It is an artifact and must not be reported as an improvement.
- **Reasoning on the last frame (d): negative.**
  - 9 of 25 dev outputs were unparsable. Three hit the 1024-token cap while looping over bounding-box coordinates. The rest returned JSON lists or mixed fragments instead of the requested object.
  - Latency was 18 s vs 1.8 s for v6.
  - On the 16 parsed outputs success accuracy was 14/16. Reasoning does not fix the perception; it only adds format risk.
- **Root cause:** Cosmos3-Edge's single-frame "is the object held" judgement is unreliable on this agentview. Precision is 0.2–0.4, and 39–65% of not-held frames are called held, whether the frame is full, cropped or paired. No way of combining frames tested here turns these judgements into failure types better than always answering no_grasp.
- **What this means for PLV:**
  - Use local Cosmos3-Edge only for v6-style last-frame success, which is cheap at about 2 s per episode.
  - Do not use local Cosmos failure types as PLV evidence. Take failure types from the simulator annotator, or from Nebius/Nemotron, which experiment 2 measured at 0.40 on failures. That is also below the 0.64 majority reference on eval v7 failures, so it too needs a majority-class check.
  - Efficiency result: the 8–15-call variants are 8–17x slower than v6 and none is better. Not running them saves GPU time.

## Limitations

- **Small eval sample, one initial condition.** The 14-minute cap with seed-major ordering covered only env_seed 200000 across 25 checkpoints, plus one 200001 episode. Results on other seeds may differ, and episodes sharing a seed are correlated. The bootstrap treats episodes as independent.
- **Dev selection on 25 of 36 episodes** (17 can, 8 square), with wide CIs. The dev ranking between a and v6 rests on 2 episodes.
- **Amended rule.** The selection candidate set was amended mid-dev (format repair). It is documented above, and eval was never used for it.
- **Partial v7 coverage.** v7 exists for only 18 of the 26 eval episodes, because `eval_local_v7.jsonl` was still being written. The v6/v7 baselines come from other processes; the same seed-0 settings were used, but GPU numerics could differ slightly.
- **b_track15 reuse.** It reuses v7 outputs for its 8 shared frames instead of re-running them.
- **Fixed ROI boxes.** The crop boxes were chosen by eye from 448x448 agentview frames and not tuned. A tighter crop could behave differently. The can crop is mostly vertical, so its gain comes mostly from upsampling.
- **Label artifact (worth adding to KNOWN_RISKS).** The square nut spawns about 6 cm above the table and settles. `failure_annotator` takes z0 from step 0, so never-grasped square failures can be labelled "other", and the 5 cm lift threshold is effectively about 11 cm for square. No visual variant can output "other". A merged-label view (square other = no_grasp) is reported in the JSON, but no eval square failure in this sample was affected.
- **Truth definition.** Frame-level "held" truth uses contact plus 2 cm, my own definition, and video frame k is mapped to sim step 2k.

## What to try next

1. **Majority-class check on failure types.** Before using any VLM failure type in PLV, compare it with the majority-class reference on the same failures. This applies to Nebius v2 too: 0.40 measured vs 0.64 majority on eval v7 failures.
2. **Success-only cascade.** Keep v6 (about 2 s per episode) and escalate only its "success" calls to a second check. That could be the Nebius video judge, or a right-bin compartment crop asking specifically about the compartment with the can marker. This targets the wrong-compartment false positives, which neither a nor d addressed.
3. **Lift and drop from state, not vision.** Take them from the recorded `object_pos`/`grasped` in the npz (already free), and reserve the VLM for what the simulator does not provide.
4. **Strict JSON.** If pairwise prompting is revisited, use constrained or JSON-mode decoding, or a single-token answer per question. The list-format failure here is a prompt-format problem, separate from the perception problem.
5. **Longer eval.** Extend the eval sample with the re-run command above, with `--eval-minutes` long enough to cover several seeds. Consider changing the eval order to checkpoint-major if seed diversity matters more than checkpoint coverage.

## Files

- Scripts: `scripts/hackathon/explore/x5_cosmos_modes.py` (GPU runner), `x5_common.py`, `x5_analyze.py`, `x5_rule_ablation.py`, `x5_report_tables.py`
- Results: `results/hackathon/explore/x5_cosmos_modes.json`, `x5_cosmos_modes_episodes.jsonl` (per-episode predictions), `x5_cosmos_modes_rule_ablation.json`
- Scratch (raw per-call outputs, prompts, frames, run log): `data/outputs/hackathon_explore/x5_cosmos_modes/{calls.jsonl, run_state.json, run.log, frames/}`
