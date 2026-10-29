# r1 — Counterfactual replanning: was the plan change physically justified and useful?

**Status:** pre-registered on 2026-10-29 (KST), before any branch rollout or judge call.
This is an exploratory study, run after Experiments 1–3. Any change made after the dev
pilot is listed under "Amendments" with its date.

## Question

PLV's Level-3 diagnostic measures **how much** a Diffusion Policy changes its plan at a
replanning boundary. It is the disagreement between the old plan's unexecuted tail and
the new plan over the same 7 steps. It cannot say whether the change was **needed**
(the object slipped, so the old plan was invalid) or **harmful** (the policy switched
modes for no physical reason).

The question is whether NVIDIA judges, looking only at the camera, can tell these cases
apart when the disagreement magnitude alone cannot.

## Independent ground truth: branch rollouts in the simulator

Recorded rollouts replay bit-exactly in MuJoCo (11 y3: 60/60). For a boundary between
consecutive policy calls `c → c+1` at time `T = plan_t[c+1]`:

1. Rebuild the env with the same seed and replay the recorded executed actions `0..T-1`.
2. Branch.
   * **NEW:** execute the new plan's steps `action_pred[c+1][1:8]`, i.e. what the policy
     actually did. As a sanity check, these must equal the recorded actions `T..T+6`.
   * **OLD:** execute the old plan's steps `action_pred[c][9:16]` for the same 7 steps,
     i.e. ignore the replan.
3. From `T+7`, both branches run the same checkpoint closed-loop until success (+16 hold)
   or `max_steps`. There are K = 4 continuations per branch, with common random numbers:
   identical batch layout and `set_sampling_seed(seed_k)`, so replicate k of NEW and of
   OLD draw the same diffusion noise.

Outcome per branch: `p = mean_k success`.
* **benefit** `b = p_NEW − p_OLD` ∈ {−1, −0.75, …, 1}.
* **helpful:** `b > 0`. **harmful:** `b < 0`. **consequential:** `b ≠ 0`.

Secondary outcomes:
* steps to success;
* stage reached 40 steps after the branch (grasped / lifted / success).

## Events (selection uses no branch outcome)

* **Source:** the 330 eval episodes (33 checkpoints × seeds 200000–200009) and their
  recorded predicted plans. Epoch-0 checkpoints (untrained) are excluded, leaving 300
  episodes.
* **Candidate boundaries:** consecutive calls (`plan_t[c+1] − plan_t[c] = 8`) with
  `16 ≤ T ≤ len(actions) − 8`.
* **Disagreement** `d`: mean absolute difference between `old[9:16]` and `new[1:8]` over
  all 10 dims, each dim divided by the task's pooled std. This is the x3 per-boundary
  quantity behind `sim_replan_overlap_l1`.
* **HIGH event:** the boundary with the largest `d` in each episode (300 events).
* **CONTROL event:** for a random third of the episodes (numpy seed 0), one boundary
  drawn uniformly from those with `d ≤` the episode median (about 100 events).
* **Dev pilot:** the same rules on the 36 dev episodes (3 other checkpoints). It is used
  only for pipeline checks and base rates. Prompts are frozen after the pilot.

## Predictors (none is fitted on branch outcomes)

* **PLV:** `d`, with position / rotation / gripper sub-groups reported too.
* **Nemotron-3.5-Super-VL on Nebius, variant J1 "event".** Input: the recorded agentview
  clip from `T−40` to `T` steps (2 s, 10 fps). It is asked, task-free, whether anything
  unexpected happened to the object (slipped, dropped, pushed, missed grasp) so that the
  robot must change its plan. It answers with a JSON flag and a necessity rating of 1–5.
* **Nemotron variant J2 "plans".** The same clip, followed by 1 s of the frame at `T`
  with both 7-step end-effector paths drawn on it.
  * The old and new paths are labelled "1" / "2" in an order randomised per event
    (seed 0). Gripper-close points are marked.
  * It is asked which continuation better accomplishes the task: 1, 2, or "no
    meaningful difference".
* **Cosmos3-Edge local.** The frozen v6 perception prompt is run on the frames at `T−16`
  and `T`. The pre-specified rule: "unexpected" = the object state changes from
  `in_gripper` to anything else, or from `on_table` (start bin) to `fallen` / `moved`.
* **Reference only (privileged simulator state, not a deployable predictor):** grasp lost,
  or object height down by more than 2 cm, within `(plan_t[c], T]`.

## Endpoints (on eval; the dev pilot is not used)

**Primary.** Restrict to HIGH events, i.e. the boundaries PLV flags. Compute the AUROC
for helpful (`b > 0`) versus the rest, for two scores: the Nemotron J1 necessity rating
and `d`. Confidence intervals come from a bootstrap over episode clusters
(checkpoint × seed), 2,000 resamples.

**Positive result** requires both:
* Nemotron J1's AUROC lower bound is above 0.5;
* the paired difference (J1 − `d`) has a lower bound above 0.

**Secondary analyses.**
1. On consequential HIGH events, the accuracy of J2's choice versus the policy default
   ("always take the new plan").
2. The same AUROCs on HIGH + CONTROL events.
3. The Cosmos rule.
4. The privileged reference.
5. A lexicographic combination (J1 flag, then `d`) versus `d`.

All of these are reported, whatever their sign.

## Budget

* About 400 eval events × 2 branches × 4 continuations ≈ 3,200 closed-loop
  continuations on the local GPU.
* About 800 Nemotron calls (roughly $1) and 800 local Cosmos calls.

## Amendments

**A1 (2026-10-29, after the dev judge pilot, before any branch label was read and
before any eval judge call).** The dev pilot (48 events) showed two problems.
* J1 is degenerate. Nemotron answered `unexpected_event = false` with necessity 1 on
  47 of 47 parsed events, so it cannot rank anything.
* J2 has a strong label preference: "1" in 39 of 48, although the label order is random.

Changes:
1. **J1 stays as pre-registered and is reported as is.** We expect its AUROC to be
   0.5 by construction, so the primary endpoint is now computed on **J1b** instead.
   * J1b gives Nemotron the frozen v6 per-frame perception prompt (the "judge, don't
     plan" design of Experiment 2) on the frames at `T−16` and `T`.
   * The score is the same pre-specified rule as the Cosmos predictor:
     "unexpected" = held → not held, or (not held at either frame and the object
     location changes).
   * The primary test is otherwise unchanged: AUROC on HIGH events, J1b vs `d`.
2. **J2s** is J2 with the two labels swapped (same video, same prompt). The combined
   choice is "new" or "old" only when J2 and J2s agree on the same plan, and "tie"
   otherwise. J2 alone is reported too.
3. **The privileged reference no longer counts an intended release.** A grasp loss or
   fall is not counted if the episode is in its success state within 16 steps after
   `T` (an intended placement). The old rule is reported too.

Extra Nemotron calls: about 400 (J2s) + 800 (J1b) on eval.
4. **J1b / Cosmos parsing.** If an answer is invalid JSON only because the free-text
   `explanation` is unquoted (15 of 96 dev J1b answers), `object_location` and
   `held_by_gripper` are read by regex, and the repair is recorded. A remaining parse
   failure is missing data for that predictor.
