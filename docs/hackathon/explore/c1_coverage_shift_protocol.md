# c1 — Coverage shift: do zero-shot VLM success judges miss more failures of stronger policies?

**Status:** pre-registered on 2026-10-29 (KST), before any rollout of this study was
recorded. This is an exploratory study, run after r1. Any later change is listed under
"Amendments" with its date.

## Motivation

There are two sources.

1. **A post-hoc look at the 330 Experiment 2 episodes.** Checkpoints are binned by
   training-time success over 50 rollouts. The table counts failed episodes that the judge
   called a success (false successes / failures).

   | checkpoint training success | Nemotron (v2) | Cosmos (v6) |
   |---|---|---|
   | ≤ 0.30 | 0/30 | 2/30 |
   | 0.30–0.70 | 2/15 | 7/15 |
   | 0.70–0.90 | 3/18 | 8/18 |
   | > 0.90 | 4/4 | 4/4 |

   * Every failure in the top bin is a Can "dropped" failure, and 3 of the 4 come from one
     initial state (seed 200000).
   * The top bin holds only Can checkpoints, so task and quality are confounded in this
     look.
2. **Guan et al. 2026, "When Better Policies Leave Harder Failures: Coverage Shift in VLM
   Success Monitoring"** ([arXiv:2610.24801](https://arxiv.org/abs/2610.24801)). They
   report the same effect for a tray-placement task.

This matters for anyone who uses a VLM judge to rank strong checkpoints or to filter
self-collected data. The judge's false successes would concentrate exactly where the
policy is good.

## Hypothesis

**H1.** Among **failed** episodes of the robosuite Can task, the probability that the judge
calls the episode a success rises with the checkpoint's quality.

Quality is `q` = the checkpoint's training-time success rate (`train_rollout_success`
in `results/hackathon/checkpoint_manifest.csv`). It comes from 50 training-time rollouts
on other seeds, recorded before this study.

## Data (new rollouts only)

All episodes come from the full-data Can pool (`data/outputs/hackathon_pool`). The
recorder settings are identical to the Experiment 2 eval set:
* agentview camera, 448 px, 10 fps;
* hold of 16 steps after success;
* sampling seed 0;
* max 400 steps.

The env seeds 600000+ are new, and every stratum uses the same leading seeds.

| stratum | q range | checkpoints | seeds per checkpoint | episodes |
|---|---|---|---|---|
| M | 0.38–0.70 | can_s1_e0015, can_s2_e0020, can_s1_e0020, can_s1_e0025, can_s1_e0030 | 12 (600000–600011) | 60 |
| U | 0.80–0.92 | can_s2_e0040, can_s2_e0030, can_s2_e0050, can_s1_e0040, can_s1_e0045, can_s1_e0095, can_s2_e0060, can_s1_e0035, can_s1_e0050, can_s1_e0055 | 16 (600000–600015) | 160 |
| T | 0.94–1.00 | can_s1_e0060, e0065, e0070, e0075, e0080, e0085, e0105, e0130, e0150, e0185; can_s2_e0070, e0080, e0090, e0110, e0130, e0150 | 25 (600000–600024) | 400 |

Altogether this is 620 episodes. At the training-time rates we expect roughly 27 / 20 /
15 failures in M / U / T.

## Judges (frozen; same settings as Experiment 2)

* **Nemotron-3.5-Super-VL on Nebius.** Prompt variant `v2_evidence_first`, no reasoning,
  max 700 tokens, 4 fps video.
* **Cosmos3-Edge, local.** Prompt variant `v6_perception_last`, no reasoning, max 300
  tokens.

Both run through `scripts/hackathon/run_cosmos_diagnostics.py`, unchanged. A parse
failure is missing data.

## Endpoints

**False success** means the judge's `task_success` is true for an episode whose simulator
truth is failure.

**Primary (Nemotron).** Over all failed episodes, compute the AUROC of `q` for "false
success".
* The 95% CI comes from a checkpoint-cluster bootstrap with 2,000 resamples (seed 0).
* The result is **positive** if the lower CI bound is above 0.5.
* If T has fewer than 8 failures, the result is reported as underpowered, whatever its
  value.

**Also reported:**
* the false-success rate per stratum, with Wilson CIs;
* Fisher's exact test of T vs M.

**Secondary analyses** (all reported, whatever their sign):
1. The same analyses for Cosmos.
2. Within the simulator failure type `dropped` only. This separates a shift in the
   failure mix from failures that are harder within a type.
3. Precision of claimed successes per stratum, over all episodes.
4. A seed-cluster bootstrap. Failures may concentrate on a few hard initial states.
5. A privileged mechanism check: the object's final position relative to the target bin,
   for false successes vs correctly detected failures.
6. The 330 exploratory episodes analysed the same way. They are reported separately and
   are not used for the primary endpoint.

## Budget

* 620 recorded simulator rollouts on the local GPU, with 4 recorder processes at low
  priority.
* 620 Nebius calls, about $0.45 at the catalog price.
* 620 local Cosmos calls.

## Amendments

**A1 (2026-10-29 16:39 KST, commit 669fafd).** Written while the recording was running, before any c1
judge call. At that point no c1 judge output existed, and only the recorder's per-checkpoint
success rates for stratum M had been printed.

*Trigger.* A check of the exploratory 330 eval episodes was run while writing
`c1_analyze.py`. On Can, all 5 Nemotron false successes (and 5 of Cosmos's 8) are failures
where the can ended **inside the goal bin but in the wrong compartment**. The other 33
failures produced 0 Nemotron false successes.

*Cause.* The frozen judge definitions do not contain the compartment criterion:
* v2's end-state sentence reads "rests inside the dark wooden bin on the right";
* v6's rule is `location == right_bin`.

The simulator, however, requires the front-right compartment. So part of what looks like
coverage shift may be a mismatch between our stated success criterion and the simulator's
criterion, not a perception failure.

*Changes.*
1. **New secondary 7 (privileged mechanism).** Report every false success by where the
   object came to rest: target compartment / other compartment of the goal bin /
   elsewhere. This sharpens the existing secondary 5.
2. **New secondary 8 (compartment-explicit prompt).** Nemotron is called again with
   variant `v2c_compartment`. It is v2 with the end-state sentence replaced by: "Success
   means that at the END of the attempt the red can rests inside the FRONT-RIGHT
   compartment of the dark wooden bin on the right, the compartment with the small grey
   can-shaped marker. A can resting in any other compartment of that bin is a failure
   (not_placed)."
   * Episodes: every failed c1 episode, plus 40 successes per stratum drawn at random
     (numpy seed 0).
   * Reported: false-success rate per stratum, and sensitivity on the sampled successes.
   * This tells whether Nemotron can see the compartment when it is asked to.
   * It is not a replacement for the primary, which stays on the frozen v2.
3. **Extra cost.** About 180 Nebius calls (about $0.13).

**Erratum to A1 "Cause" (2026-10-29, after verification).**
* v2's *goal* sentence does name the front-right compartment. Only its *success* sentence
  omits it.
* v6 cannot express a compartment by construction: its locations are `right_bin` etc.,
  and it is told to ignore the marker.

So v2c makes v2's success sentence consistent with its goal sentence.

A1's timing also needs a correction. A1 was written before any c1 judge call. By then
stratum M and one U checkpoint (can_s2_e0040, finished 7 s before the A1 commit) had
finished recording.

**Deviations found by the verification.**
* **Scale.** 620 rollouts and 809 Nebius calls (about $0.65). The user had approved about
  400 rollouts and about $0.3. The increase was disclosed in the conversation but not
  re-approved.
* **Recorder batch sizes.** They are 12 / 16 / 25 (Experiment 2: 10). Diffusion noise is
  seeded per batch, so per-env noise differs from Experiment 2.
* **Recorder code changed mid-run.** 13 of the 16 T checkpoints ran after commit 627063a
  changed `rollout_recorder.py`. The change adds an optional `save_obs` flag and has no
  effect when it is off.
* **Judge script.** `run_cosmos_diagnostics.py` gained `--episodes` and `--evidence`. The
  frozen variants' prompts are unchanged: prompt hashes were recomputed and match
  Experiment 2.
* **16 black videos.** All are T-stratum true successes at seed 600003, with frames after
  the first one black. This is an EGL artefact at batch index 3. The failure-only primary
  is unaffected. They are excluded from success-side metrics
  (`post_hoc_after_verification`).
* **Judge input.** Frames are subsampled to 5 Hz, resized to 320 px and played at 4 fps.
  Clips are unpadded, so clip length differs between successes and failures. This is
  inherited from Experiment 2's frozen condition.

## Results (2026-10-29; four independent verification lenses reproduced every number)

**Primary: positive under the pre-registered rule, but fragile.**
* Nemotron v2 false successes among failures: M 6/25, U 14/27, T 13/22.
* AUROC of q is 0.673. The checkpoint-cluster CI [0.556, 0.750] gives **positive**.
* Fisher T vs M p = 0.019.
* With 22 T failures the result is not underpowered.
* The other intervals include 0.5:
  * seed-cluster CI [0.399, 0.922];
  * crossed checkpoint × seed bootstrap [0.35, 0.95] (`post_hoc_after_verification`).
* Two initial states (600002, 600011) supply 23 of the 33 false successes.
* Leaving out seed 600011 alone gives AUROC 0.59 (post hoc).

**Cosmos v6:** 8/25, 14/27, 19/22; AUROC 0.739 [0.596, 0.837], seed CI [0.553, 0.927].

**Mechanism (secondaries 2, 7 and 8; interpretation is post hoc).**
* **It is a shift in the failure mix, not harder failures.**
  * No judge accepted any no-grasp failure (0/21). No-grasp failures disappear as quality
    rises: 15 / 6 / 0.
  * Judges accept failures that end at the goal bin at every quality level. Within
    "dropped" failures, the AUROC of q is 0.49.
* **Every Nemotron false success ended at the goal bin.** 32 of the 33 were in the wrong
  compartment, and 1 rested on the target compartment's divider.
  * The wrong compartment is visible in the 320 px input in 29 of the 33 cases.
* **Stating the compartment in the success sentence does not fix it (v2c).**
  * Nemotron still accepted 24 of 33 wrong-compartment cans.
  * Of its 9 rejections, only 2 cite the compartment. The other 7 falsely claim the can
    is still held.
  * So Nemotron essentially does not use compartment information.
* **Precision of claimed successes rises with quality:** Nemotron M 0.854, U 0.898,
  T 0.965. Data filtered from strong checkpoints is therefore *cleaner*. The
  contamination concern in the motivation does not hold at this task's failure rates.
* **Sensitivity on valid videos:** Nemotron 0.981. Cosmos answered "success" on all 16
  black frames, so its sensitivity of 1.0 is not grounded in the image.

**Actionable rule.**
* Judged success overstates true success by about the rate of near-goal failures: the
  object reaches the goal but violates a secondary condition, such as the wrong
  compartment or the rim.
* Both judges accept those failures almost always, at every checkpoint quality.
* Before trusting a judged-success gap between strong checkpoints, audit a sample of
  judge-accepted episodes for near-goal failures.
* Evaluate all candidates on the same, many, distinct initial states.

**Operational note (no effect on data).** At 16:37 the recorder gained a start-up guard. It
waits for free GPU memory so that 4 workers with 25-env batches do not run out of memory.
Recorder settings are unchanged.
