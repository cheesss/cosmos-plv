# 11 — Exploratory studies: other ways to use Cosmos / Nemotron for PLV

Everything on this page is **exploratory and post-hoc**. These studies came after
the pre-registered Experiments 1–3 in `06_experiment_protocol.md` and do not
replace them. Each study was implemented by one agent, and all except x0 (a deterministic rule) and w4 were checked by an
independent adversarial verifier. The verifier re-ran the analysis and looked for
leakage, wrong denominators, clustering and over-claiming. The verified (corrected)
conclusions are quoted below. Full reports are in `docs/hackathon/explore/`, data in
`results/hackathon/explore/`, and code in `scripts/hackathon/explore/`.

| id | question | verdict | helps PLV? | corrected conclusion |
|---|---|---|---|---|
| x0 | Can local Cosmos filter calls to Nebius? | not separately verified (simple, deterministic) | partially (cost) | Rule: if local Cosmos says failure, accept it; if it says success, ask Nemotron. On the full eval split (n=330) it scores 0.952 [0.923, 0.970] against 0.964 for Nemotron alone, using 84% of the Nebius calls (16% fewer); false negatives rise from 3 to 8. An earlier n=150 snapshot (0.960 with 71% of the calls) overstated the saving. On dev it scores 0.917 with 56% of the calls. |
| x1 | How many rollouts are needed to find a near-best checkpoint? | partially confirmed | partially | Adaptive allocation needs about 2–3x fewer rollouts than uniform allocation for the same excess regret (seed-bootstrap median ratio 3, 10–90% range 2–4). It uses the 50 training-time episodes per checkpoint: half as the pool, half held out. Absolute budgets are unstable as the pool grows. A noisy VLM judge raises the budget (Nemotron about 1.5x). Cosmos, once its within-checkpoint false-positive correlation is modelled, does not reach the target within 25N. A val-loss prior is worse than a random prior. A free "latest epoch" baseline is very strong in these short runs. |
| x2 | Can rollouts be aborted early once failure is evident? | partially confirmed | partially | Failing episodes use 54% of robot time. Held-out abort rules save 16% (time limit only) to 20% (with privileged simulator state) of total robot time at 2/141 false aborts, with no change in ranking (leave-one-checkpoint-out fit). Nearly all of the safe saving comes from a shorter time horizon. Cosmos v7 adds +1.5 points outside epoch-0 checkpoints. Nemotron on 4-s prefixes reaches AUROC 0.85 (0.76 without epoch 0) and adds +4.4 points to a time cut outside epoch 0. |
| x3 | Does high replanning disagreement (real feedback) point to specific failures? | partially confirmed | partially | Disagreement jumps at and after a drop, but shows no precursor before can drops. It is a post-hoc indicator, not an early warning. Gripper-dimension disagreement is higher for no_grasp than for drops (within-checkpoint AUROC 0.09, 11 pairs). Checkpoint-level disagreement adds nothing beyond epoch. |
| x4 | Do other Nemotron modes classify failures better? | partially confirmed | no | No mode improved pooled failure-type accuracy significantly (best 0.441 vs 0.356, CI includes 0). Dropped recall improved but no_grasp recall fell. The verifier also found **clip-length leakage**: successful clips stop 0.8 s after success, so length alone separates outcomes. See the length-neutral check below. |
| x5 | Do other Cosmos3-Edge modes help (ROI crop, 16 keyframes, frame pairs, reasoning)? | partially confirmed | no | None beat v6. ROI crop: +0.08 on dev (2 episodes), −0.077 on eval. Frame-pair failure types only match the majority-class baseline. Reasoning left 9/25 outputs unparsable. |
| x6 | Rollout-free "visual plan critic": draw the policy's first plan on the start image and let a VLM judge it | partially confirmed | no | It ranks checkpoints far worse than the free epoch baseline. Plan geometry of trained checkpoints is 86–98% determined by the env and sampling seed, not by the checkpoint. The VLM-free geometric check is enough to flag untrained checkpoints. |

## Length-neutral re-check (after the x4 finding)

The frozen Nemotron v2 prompt was first re-run on 250 eval episodes (the full 330 are below) with every clip
padded to the timeout length, holding the final frame, and with a fixed stated
duration (`eval_nebius_v2_pad201.jsonl`).

* Original condition: success accuracy 0.956 [0.923, 0.975], 8 false positives,
  3 false negatives.
* Length-neutral condition: success accuracy 0.956 [0.923, 0.975], 10 false
  positives, 1 false negative.
* 4/250 judgements changed between the two conditions.
* On all 330 eval episodes: original 0.964 [0.938, 0.979] (FP 9, FN 3) vs length-neutral 0.961 [0.934, 0.977] (FP 12, FN 1); 5/330 judgements changed.

Nemotron therefore does not exploit clip length. Two of the flips were square
successes that were missed before and found after padding: the held final frames
show the nut settled. This fits the "clip ended too early" explanation.

## Bugs found by verifiers and fixed

* **Square ground-truth labels** (x2 verifier): square nuts spawn about 6 cm above
  the table and settle. Episodes were labelled `other` ("fell") and lifts were
  under-detected. The annotator now uses the settled height, and every episode was
  re-annotated from its saved state trace: 15 `other` became `no_grasp`, 2
  `no_grasp` became `dropped`. Success labels are unchanged. Recorded in
  `06_experiment_protocol.md`.

## What we take from wave 1

1. **Success detection from video works.** The NVIDIA VLMs judge success from
   video alone with 0.92–0.96 accuracy (eval split). That matters most where no success
   detector exists, such as real robots.
2. **Failure taxonomy does not work.** Every variant fails to beat a
   majority-class baseline on failure types.
3. **The biggest efficiency lever is how rollouts are allocated**, not the VLM
   itself. Adaptive allocation plus early abort saves rollouts and robot time.
4. **PLV must beat the trivial epoch baseline.** Our short training runs improve
   almost monotonically, so "later epoch" is a strong baseline. See
   `07_experiment_results.md`.

## Wave 2

| id | question | verdict | helps PLV? | corrected conclusion |
|---|---|---|---|---|
| y3 | Is VLM success detection robust to camera view and image quality? | partially confirmed | partially | Recorded rollouts replay **bit-exactly** in MuJoCo (60/60, physical state error 0.0; a negative control detects 1e-3 action noise), so other cameras can be rendered without re-running the policy. Image degradation (down-scaling to 224/112, JPEG q20, blur, dimming) has no detectable effect on either judge (\|Δ\| ≤ 0.03 after re-weighting to the eval mix). With the frozen prompts both judges fall to chance on the wrist camera (balanced acc 0.49–0.51); this is the only effect that survives multiple-comparison correction, and it is confounded by the prompt saying "fixed third-person camera". Distant cameras (front, bird) change accuracy in a direction that depends on backend and sample mix (Cosmos adds square false positives; Nemotron birdview adds can false negatives). Practical rule: judge from a close, oblique fixed camera like agentview, and re-select the prompt on dev whenever the camera changes. |
| y4 | Can an NVIDIA text LLM on Nebius (Nemotron-3_5-Lightning) write grounded per-checkpoint evaluation reports? | partially confirmed | no (as is) | Given measured-facts-only sheets it rarely writes a number that appears nowhere in the sheet (1/723). A manual random audit nevertheless found a clear false statement in 5/16 reports, and the automatic checker caught only 2 of those 5. Its "spend more rollouts?" answer was a constant "no" (62/62). Cost: 118 calls, about $0.03. **LLM-written reports are therefore not shown as facts in the dashboard.** |

## Wave 3 (complete pool: 93 checkpoints)

| id | question | verdict | helps PLV? | corrected conclusion |
|---|---|---|---|---|
| y1 | Does a PLV-metric prior (top-30%) cut the rollout budget needed to find a near-best checkpoint? | partially confirmed | partially | Screening first beats no screening, clearly so with noisy VLM judges or truncated histories. **All of that gain is already delivered by the free latest-epoch prior.** 84.6–100% of every PLV top-30% set are the same checkpoints as the epoch top-30%. In 0 of 336 judge × strategy × prior cells per history level did a PLV prior beat epoch. Validation loss is a harmful prior. |
| y2 | On the plateau, can any signal rank checkpoints, and what is the noise ceiling? | partially confirmed | no | Two disjoint 25-rollout halves do not reproduce a ranking on the plateau (split-half ρ +0.02, CI [−0.16, +0.21]), so plateau ranking cannot be evaluated with 50 rollouts per checkpoint. Once split luck is controlled, no PLV, rollout or VLM signal shows robust value beyond epoch. |
| y5 | Does PLV help pick across two training seeds (where epoch cannot break ties)? | partially confirmed | no | Pooled Spearman: epoch 0.826, Action L1 0.790, rank-average 0.805. No PLV rule beats epoch. Only 4 of 21 epoch-matched seed pairs differ by ≥ 0.06 in success. |

## Wave 4: long low-data runs (Can 20%, Square 50% of demos, 1,000 epochs, 82 checkpoints)

| id | question | verdict | helps PLV? | corrected conclusion |
|---|---|---|---|---|
| w4 | Does PLV beat epoch when training is long and success plateaus or regresses? | no | no | Mean within-run Spearman over the two runs: epoch 0.56, open-loop smoothness 0.55, PLV-1 0.50, validation loss −0.38, replan overlap 0.08, boundary jump −0.01. Per run, epoch is 0.72 (Can) and 0.41 (Square); open-loop smoothness is 0.75 and 0.36. Within the late half nothing ranks: the two-run mean |ρ| is ≤ 0.15 for every metric except boundary jump (−0.27, driven by Square at −0.65). Regret@30 is ≤ 0.04 for every metric. `results/hackathon/explore/w4_lowdata/`. Not independently verified. |

## Wave 5

| id | question | verdict | helps PLV? | corrected conclusion |
|---|---|---|---|---|
| z1 | How many VLM-judged rollouts does a coarse BAD / OK / GOOD verdict need? | partially confirmed | partially | It is expensive. A perfect detector needs about 25–36 rollouts for 90% tier accuracy (sequential stopping: 25.1 on average). Nemotron is borderline at 50. Cosmos3-Edge does not reach 90% (it inflates OK to GOOD 46% of the time). The cost comes mostly from the 39% of checkpoints near the 0.9 boundary. Epoch/PLV priors only help confirm BAD early. |
| z2 | Can a cross-validated (leave-one-run-out) learned combination of signals beat epoch? | partially confirmed | no | No. The best fixed-split gain (+0.009) sits at the 98th percentile of random splits. Averaged over splits, every learned combination is slightly below epoch. Adding K=10 VLM features does not reach epoch (0.79–0.83 vs 0.85). |

## Wave 6: Cosmos world model, and counterfactual replanning

| id | question | verdict | helps PLV? | corrected conclusion |
|---|---|---|---|---|
| e4 / e4b | Does Cosmos3-Edge's action-conditioned forward dynamics follow the policy's actions on robomimic frames? | partially confirmed (4 verifier lenses) | not yet | E4 went through the untrained `libero` slot and is retracted. Through trained Franka slot 8 (RoboMIND encoding), each commanded axis moves the generated gripper consistently, 14–16 px (cos 0.87–0.94). On 15 recorded-rollout clips the action-dependent motion follows the true motion (cos 0.78–0.85), but with only 5 families the t(4) lower bounds are 0.15–0.63. Can is strong, Square weak (end error worse than zero actions). Motion is about 0.4x, and there is no PSNR gain over a frozen frame. See `07_experiment_results.md` §C3. |
| r1 | Was a large replan physically justified and useful, and can a VLM tell when PLV's magnitude cannot? | primary negative (4 verifier lenses; branch mechanics confirmed bit-exact) | no | Nemotron J1b 0.40 vs d 0.79 (AUROC helpful vs rest, 300 max-d boundaries). The 0.40 comes from intended releases after success: 77 of the 300 boundaries were already decided (a selection defect), and without them J1b is 0.49 and d 0.75. J1 0.49, Cosmos 0.54, privileged 0.49. J2 abstains on 29 of 52 consequential events; on the rest it is 0.61 vs always-new 0.70. d does not localize the replan that matters: same-episode low-d controls are consequential as often (19 vs 18 of 100). d mostly flags fragile episodes (episode-failure AUROC 0.87). At K=4 the labels are consistent with rollout noise (permutation P=0.26), so helpful vs harmful is unresolvable. Robust effect: replanning reaches success sooner (80 vs 51 all-success events, p=0.014). Deviations, including an interim look, are disclosed in the protocol. |
| c1 | Do zero-shot judges miss more failures of stronger policies (coverage shift)? | primary positive but fragile (pre-registered exploratory; 4 verifier lenses) | — | Nemotron false successes among failures: 6/25, 14/27, 13/22 (M/U/T). AUROC of quality 0.67 [0.56, 0.75] with checkpoint clusters; seed-cluster CI [0.40, 0.92] and two-way [0.35, 0.95] include 0.5. Two initial states give 23 of the 33 false successes. Cosmos 8/25, 14/27, 19/22. Mechanism: a shift in the failure mix. No-grasp failures are always rejected and vanish with quality; failures ending at the goal bin are accepted at every quality (dropped-only AUROC 0.49). 32 of 33 false successes are in the wrong compartment. Stating the compartment (v2c) still accepts 24/33, and only 2 rejections cite it, although it is visible at 320 px. Precision of judged successes rises with quality (0.85/0.90/0.97). 16 black success videos (EGL artefact, seed 600003) are excluded from success-side metrics. `explore/c1_coverage_shift_protocol.md`. |
| s1 | Can Nemotron-filtered self-training lift a plateaued Can policy (0.885)? | null (pre-registered; 4 verifier lenses, no bug) | — | Every arm scored 0.875–0.89 on 200 paired seeds: Nemotron −0.5 [−4.0, +3.0], simulator labels +1.0 [−2.5, +4.5]. Even unfiltered data with 32% failure frames was null, so this recipe cannot rank label sources. The always-fail initial states stayed failed. `explore/s1_self_improve_protocol.md`. |
| s2 | Redesign: weaker base (0.656), human-demo positive control, annealed fine-tune, 500 paired seeds | H1 negative (worse); positive control passed (1 verifier) | — | continue 0.856; human +8.2 [+5.2, +11.4]; oracle −3.2 [−7.0, +0.6]; Nemotron −10.4 [−14.8, −6.0]. Nemotron accepted 58/149 failures (44 in the wrong compartment), 40% of the self frames. Post hoc, Nemotron is 7.2 below oracle, mostly through extra goal-bin misplacements (46 vs 19). This is outside the pre-registered H2 gate. Caveats: one run per arm, demo dilution, clip length follows sim success. `explore/s2_self_improve_protocol.md`. |

## Running take-aways

1. **Success detection from video works** (0.92–0.96 on the eval split, robust to clip length and image quality) **if the camera is chosen well**. Failure-type taxonomy does not work.
2. **The biggest measured efficiency levers do not need a VLM**: adaptive rollout allocation (2–3x fewer rollouts) and early abort with a time limit (16% robot time; 20% with privileged simulator state). VLM judges replace a privileged success detector at a modest budget cost; a local-first cascade cuts the Nebius calls.
3. **PLV in this pool measures training progress.** Every PLV metric, prior, learned combination and VLM-augmented score is indistinguishable from, or worse than, the free epoch baseline (y1, y2, y5, z2). Near the plateau even 50 rollouts per checkpoint cannot rank. A fair test needs runs where "later" is not better. In wave 4 (long low-data runs, 82 checkpoints) epoch is still the strongest signal on average (Spearman 0.56; the best PLV metric, open-loop smoothness, is 0.55).
4. **Generated text must be checked against data.** Unconstrained LLM summaries contained false claims, so the application shows only measured values and authentic model outputs.
