# x6_visual_plan_critic: adversarial verification

**Status: PARTIALLY CONFIRMED. EXPLORATORY / POST-HOC.** This note checks the x6 report
(`docs/hackathon/explore/x6_visual_plan_critic.md`). Every new number below comes from
`results/hackathon/explore/x6_visual_plan_critic_verification.json`, which
`scripts/hackathon/explore/x6_visual_plan_critic_verify.py` produces. Other numbers are quoted from x6's own
`results/hackathon/explore/x6_visual_plan_critic.json`, which I reproduced bit for bit. I changed none of x6's
files. I made **0 new Nebius calls** and **no GPU runs**. Every VLM result was re-scored from the saved raw rows.

> Summary: x6's **core negative conclusion holds.** The visual plan critic, which attaches a VLM verdict to one initial plan, is useless for checkpoint screening and loses to the free epoch baseline. Rerunning `x6_analyze.py` with only the output paths changed **reproduces the results JSON completely, with 0 leaf differences**. There is no leakage. No score was fitted, the geometric heads give the same Spearman for thresholds 0–5 cm, and both fields and both variants were reported. All 375 analyzed Nemotron rows match the clean overlay images by SHA-256. I also confirmed the projection and the identity of the initial states.
> However, the following need fixing.
> (1) **plan3 is not a "24-step real rollout".** What actually ran was **16 steps** (2 chunks, plan_t = 0, 8, 16).
> (2) **"Geometric heads are all 1.0 at epoch≥10" is wrong.** `square_s1_e0010` is 0.8. The body's epoch≥20 statement is correct.
> (3) **"VLM heads is worse than geometric heads" is only a point estimate.** The paired bootstrap CIs of the difference are [−0.29, +0.71] / [−0.21, +0.81] and include 0.
> (4) **"The VLM metric is only a proxy for training progress" is not established.** At epoch≥20, the partial Spearman of VLM heads controlling for epoch is +0.41 [−0.47, +0.89] (permutation p=0.066) / +0.21 (p=0.26). This is evidence neither that it adds information nor that it does not. Also, at epoch≥20 the difference by which epoch beats VLM heads is not significant either (can_s1 [−0.02, +0.92], can_s2 [−0.15, +0.56]). It is significant only over all epochs.
> (5) **The η² figures are mixed with the task effect.** Computed within task, the seed η² of reaches is 0.77/0.81 and the checkpoint η² is 0.047/0.038 (chance about 0.19). The conclusion actually becomes stronger. And the plan geometry itself is 0.86–0.98 determined by seed. Seed dominance is a property of the plan, not a defect of the VLM, and because of CRN, seed means the scene and the sampling noise together.
> (6) **The test-retest heads 0.95 includes 10 square pairs that are always True.** On can alone it is 9/10 [0.60, 0.98], and the 1 flipped case is the ambiguous seed 210000.
> (7) **On square, "the sign of closer=better is reversed" is an overstatement.** It is +0.33 [−0.24, +0.71], a CI that includes 0, and square's minimum 3D distance has no epoch drift (+0.16 with epoch).
> (8) Beyond these there are minor issues: the raw control uses a different prompt, a degenerate bootstrap CI, and a wrong task-limit figure (written as 400 when it is 800).
> On the current pool (70) the epoch baseline is still +0.91 / +0.84 / +0.76. But square is weakening at epoch≥20, to +0.67 [+0.24, +0.90].

## 1. What I checked and how

* **Reproduction.** I imported `x6_analyze`, redirected `RESULT` to
  `data/outputs/hackathon_explore/x6_verify/x6_analyze_rerun.json` and ran `main()`. Then I diffed every JSON leaf
  against the published `x6_visual_plan_critic.json`.
* **Leakage and selection.** No score is fitted to outcomes. I checked the a-priori thresholds by sweeping
  the geometric-heads margin over 0, 1, 2, 5 and 8 cm. I also checked that both VLM fields (`heads`, `reaches`) and both
  variants (`plan1`, `plan3`) are reported, and that the 25 excluded calls were made on frames that really were
  corrupted.
* **Data integrity.**
  * I recomputed `pix_plan1` from `cam_P` and `action_pred`.
  * I checked that `eef0`, `obj0` and `cam_P` are bit-identical across all checkpoints for each seed. This is
    the redraw precondition.
  * I matched the `image_sha256_0` in the Nebius evidence log against the current clean image for every
    analysed row.
  * I checked that the policy is low-dim, so corrupted renders could not reach the policy input.
  * I inspected all 10 reference frames visually with the projected EEF and object markers.
* **Statistics.**
  * per-task AUCs, because pooled AUCs can show Simpson effects (square `heads` is always True);
  * within-task η² for the VLM verdicts and for the plan geometry itself;
  * paired bootstrap of Spearman differences;
  * partial Spearman given epoch, with bootstrap and permutation tests;
  * test-retest split by task;
  * plan3 step count, and plan3 AUC against the true 3-chunk progress;
  * the cost denominator;
  * the epoch baseline on the current, grown pool.
* **Not applicable here:**
  * Video frame/step alignment (frame k = step 2k) does not apply because x6 uses no video.
  * Clip-length leakage does not apply because there are no clips.
  * The plan index offset (index j ↔ step t+j−1) only affects the colour gradient and the `plan_t` bookkeeping.

## 2. Commands run (cwd `/path/to/repo`)

| # | Command | Exit |
|---|---|---|
| 1 | `PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 ~/.venvs/cosmos-plv/bin/python - <<EOF` (import `x6_analyze`, set `A.RESULT` to the scratch path, `A.main()`) | 0 |
| 2 | about 10 read-only interactive `python - <<EOF` checks (per-task AUC, η², partials, hashes, montage). All are folded into the script below | 0 (all) |
| 3 | `PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/x6_visual_plan_critic_verify.py` | 0 |

The pool grew from 69 to 70 checkpoints between step 2 and step 3, because training is still running. All
current-pool numbers below come from the step-3 JSON. The reproduction itself only uses the 51 checkpoints
that have x6 plans, so it is unaffected.

## 3. What is confirmed

| x6 claim | Verification | Verdict |
|---|---|---|
| All tables in §5 of the report | Re-run gives **0 differing leaves** against the published JSON | confirmed |
| 400 Nebius calls, 0 errors; 375 analysed, 25 excluded | Evidence log: 400 x6 calls (plan1 265, plan3 100, plan1_rep 20, raw 15), 0 errors. 375 clean rows, **375/375 image hashes match the current clean image**, 0 mismatches | confirmed |
| Frames after the first checkpoint corrupted or flipped; geometry fine | The originals of `can_s2_e0000` (vertically flipped), `can_s1_e0020`, `can_s2_e0080`, `square_s1_e0120` and `can_s1_e0095` are corrupted (teal noise or flip; montage `data/outputs/hackathon_explore/x6_verify/orig_frames.png`). The policy is low-dim (`low_dim_abs.hdf5`), so plans are unaffected | confirmed |
| Redraw valid because initial state per seed is identical | Max deviation of `eef0`, `obj0` and `cam_P` across 36 can and 15 square checkpoints is **0.0** for every seed. Recomputed `pix_plan1` max error is **0 px**. All 10 reference frames are upright, with the markers on the gripper and on the can or nut (`ref_frames_markers.png`) | confirmed (stronger: all 10 seeds, not only the one debug seed) |
| `reaches` is a distance threshold (pooled AUC 0.975) | Per task: can **0.982**, square **0.946**. On can epoch≥20 the AUC is **1.0**, with a sharp cut: the largest min-distance with `reaches=True` is 0.098 m and the smallest with False is 0.099 m | confirmed, not a pooling artefact |
| `heads` AUC vs 3D progress below chance (0.30) | Can alone: 0.33 overall, 0.248 at epoch≥20. Square is undefined (always True). On can epoch≥20, `heads` is lowest on seeds 210000 (0.233) and 210004 (0.633), which have the **largest** 3D progress (0.167 and 0.180 m) | confirmed |
| Epoch-0 heads=True 7/10 (can), 5/5 (square); geo heads 0.0/0.0/0.2 at epoch 0 | Reproduced. Geo heads is unchanged for margins of 0–5 cm (Spearman +0.53 / +0.44 / +0.60 at every margin) | confirmed, no threshold tuning |
| Epoch baseline beats the VLM (all epochs) | Paired bootstrap of epoch − VLM heads Spearman: can_s1 [+0.16, +1.06], can_s2 [+0.07, +1.01] | confirmed for all epochs on can |
| Cost about 28 s per checkpoint; eval 16.6 s per episode | `batch_wall_s` takes one value per summary (25/25), so dividing by 10 episodes is the right denominator | confirmed |
| Cosmos3-Edge not run | No `vlm_local_clean.jsonl`; no x6 Cosmos process is alive now | confirmed |

## 4. Issues found

1. **plan3 is a 16-step rollout, not 24.** In all 90 plan3 npz files, `plan_t` = (0, 8, 16) and `eef_traj` has
   17 entries. The loop in `x6_plan_render.py` breaks after the third `predict_action` and never executes it.
   So 2 chunks (16 sim steps, 0.8 s) were executed, and the third chunk is drawn but not executed. The report
   says "3 chunks (24 sim steps)", and the metric name `exec24_min_dist_m` repeats the error. This does not
   change the conclusion.
2. **Implementation slip in the plan3 AUC.** `auc_vlm_heads_from_3d_progress` for plan3 uses the end of the
   *first* chunk (`geo_end_dist_m`). Against the true 3-chunk end, the AUC is **0.959** instead of 0.969.
   Immaterial.
3. **"geometric heads = 1.0 at epoch ≥ 10"** (findings JSON, finding 2) is false. `square_s1_e0010` = 0.8, while
   `can_s1_e0010`, `can_s1_e0015` and `can_s2_e0010` = 1.0. (`can_s1_e0005` = 0.8 is below epoch 10.) The report
   body's "1.0 for all 43 checkpoints with epoch ≥ 20" is correct.
4. **"The VLM is worse than geometric heads" is a point-estimate claim.** The paired bootstrap of geo −
   VLM-heads Spearman over all epochs gives can_s1 [−0.29, +0.71] (1785/2000 non-degenerate resamples) and
   can_s2 [−0.21, +0.81]. Both include 0. The supportable wording is "not better than". In the epoch ≥ 20 regime,
   geometric heads is constant (undefined), while VLM heads has +0.53 / +0.63.
5. **"The VLM metrics are only a training-progress proxy" is not established for `heads`.**
   * On can_s1 epoch ≥ 20, VLM heads correlates more with success (+0.53) than with epoch (+0.40).
   * The partial Spearman of VLM heads with half-B success, given epoch, is **+0.41 [−0.47, +0.89]**
     (one-sided permutation p = 0.066) on can_s1 and **+0.21 [−0.43, +0.82]** (p = 0.26) on can_s2.
   * Over all epochs it is +0.29 (p = 0.12) and −0.01 (p = 0.53).
   * So there is no evidence of added information, and no proof that there is none, at n = 14–16 with 3
     distinct score values.
   * Also, in the epoch ≥ 20 regime the epoch-minus-VLM-heads difference is **not significant**: can_s1
     [−0.02, +0.92], can_s2 [−0.15, +0.56]. "Falls far behind" holds only for all epochs.
   * For `reaches`, the proxy reading is fair. The partial is negative (−0.25 / −0.52), which means it has the
     wrong sign beyond epoch.
6. **The η² figures are confounded with task.** x6 groups "seed" as seed × task (10 groups). Checkpoints are
   nested in task. Task alone explains η² = 0.20 of `reaches` at epoch ≥ 20. Within task (epoch ≥ 20):

   | quantity | can: by seed (chance) | can: by checkpoint (chance) | square: by seed (chance) | square: by checkpoint (chance) |
   |---|---|---|---|---|
   | VLM `reaches` | 0.766 (0.027) | 0.047 (0.195) | 0.809 (0.062) | 0.038 (0.188) |
   | VLM `heads` | 0.443 (0.027) | 0.084 (0.195) | undefined (all True) | undefined |
   | plan min 3D distance | 0.856 | 0.028 | 0.952 | 0.004 |
   | plan min pixel distance | 0.917 | 0.012 | 0.980 | 0.005 |
   | plan 3D progress | 0.826 | 0.034 | 0.961 | 0.003 |

   The conclusion "scene, not checkpoint" is **stronger** within task: the checkpoint explains less than chance.
   But the attribution needs correcting. The plan geometry itself is 86–98 % seed-determined, so the VLM is
   faithfully reading plans that barely differ between trained checkpoints. This is a property of one
   initial plan, not a VLM failure. Also, the sampling seed equals the env seed (common random numbers), so
   "seed" means scene **plus** diffusion noise. The report's "scene (seed)" cannot separate the two.
7. **The test-retest "heads 0.95" is padded.** 10 of the 20 pairs are square images, where `heads` is always True
   (base 10/10 True). On can, heads agrees **9/10 (Wilson [0.60, 0.98])**, and the one flip
   (`can_s1_e0040`, seed 210000) is on the ambiguous seed that drives most of the per-checkpoint `heads`
   variation. Reaches agrees 10/10 on both tasks. The repeat calls are genuinely independent: they have distinct
   request IDs, were made about 14–22 min after the base calls, and most have different raw text.
8. **"Sign opposite on square" and "monotonic geometry drift" are overstated.**
   * On square epoch ≥ 20, "closer = better" (min 3D distance) is +0.33 [−0.24, +0.71], so the CI includes 0.
   * The square min 3D distance has no epoch trend (oriented Spearman with epoch +0.16).
   * The monotonic drift (0.114–0.115 m → 0.128–0.131 m) is real only on can. On square, only the pixel
     distance drifts (−0.96 with epoch).
9. **The raw-frame control changes two things.** It removes the dots **and** switches to a different prompt
   (`PROMPT_RAW`: "is the gripper *currently* close"). So "reaches=True is induced by the drawn dots" is
   plausible, and consistent with the `reaches`-is-a-distance-threshold row of §3, but not isolated. The same control gives `heads=True` on 3/5
   raw can frames, which shows a sizeable `heads` prior even with no plan.
10. **On square, the geometric `reach` baseline uses the nut body centre.** `obj0` is the nut body. The trained
    plans visibly end on the **handle** for seeds 210000, 210002 and 210004 (`ref_frames_markers.png`, bottom
    row), where VLM `reaches` = 1.0 although the min distance to the centre is 0.091 and 0.070 m on two of them.
    So part of "the VLM threshold is looser than 5 cm" is an artefact of how the baseline is defined on square,
    not VLM looseness.
11. **Several bootstrap CIs are degenerate.** Examples: geo heads +0.44 [+0.43, +0.77], plan3 heads
    +0.71 [+0.71, +1.00], geo reach −0.44 [−0.78, −0.44]. `bootstrap_ci` silently drops all-tied resamples, so
    with 2–3 distinct values these intervals are conditional and not credible. In particular, "geo heads
    +0.53 [+0.38, +0.77]" should not be read as excluding 0 with confidence.
12. **Per-image Wilson intervals assume independent images.** "70 % [0.40, 0.89], n = 10" comes from 2
    checkpoints × 5 seeds. The pooled image AUCs (n = 255) are effectively clustered in about 10 seed × task
    cells, so their precision is overstated. No CI is given for them.
13. **Minor text errors.**
    * The report says "task limit 400". The task limit was 800; x6 capped itself at 400.
    * The headline lists heads/reaches with u = 1–3 next to geo heads without saying that geo heads also has
      only u = 2–3.

## 5. Current pool (training still running)

The epoch baseline on half-B success, from the verify JSON (70 checkpoints):

| run | max epoch | all epochs | epoch ≥ 20 |
|---|---|---|---|
| can_s1 (n=28) | 135 | +0.91 [+0.76, +0.98] | +0.86 [+0.64, +0.96] |
| can_s2 (n=21) | 200 | +0.84 [+0.60, +0.96] | +0.78 [+0.48, +0.93] |
| square_s1 (n=21) | 200 | +0.76 [+0.41, +0.93] | +0.67 [+0.24, +0.90] |

can_s2 has reached its final epoch 200 with no late collapse. Square is weakening: it was +0.85 / +0.77 on x6's
15 checkpoints. That is exactly the regime where x6 says a quality signal would matter. The x6 plan probe has
not been run on the 19 new checkpoints.

## 6. Corrected headline

> EXPLORATORY, negative. A Nemotron critic of the policy's first 16-step plan, drawn on the initial frame,
> does not help checkpoint screening in this pool. It is far below the free epoch baseline over all epochs
> (+0.38 / +0.34 / undefined vs +0.94 / +0.87 / +0.85). In the epoch ≥ 20 regime, its `heads` score is
> positive on can (+0.53 / +0.63). It stays below epoch (+0.89 / +0.80), although that gap is not significant
> in this regime, and it adds no demonstrated information beyond epoch (partial +0.41, p = 0.07; +0.21,
> p = 0.26). `reaches` is a sharp
> distance threshold (about 9.9 cm on trained can plans) and ranks with the wrong sign. The root cause is the
> probe, not the model: the plan geometry of trained checkpoints is 86–98 % determined by the env/sampling
> seed and at most 3 % by the checkpoint. A free geometric "plan heads to object" check is at least as good as
> the VLM at spotting untrained checkpoints. Its apparent advantage over the VLM is not significant.

## 7. Implications for PLV / what to do

* Keep x6's recommendation to drop VLM plan critics for screening *trained* checkpoints. The within-task η²
  result makes this stronger: one initial plan simply does not vary across trained checkpoints.
* Treat the geometric "plan heads to object" gate as a sanity check only, for untrained or broken
  checkpoints. Do not call it better than the VLM.
* Any rerun on the final pool should report the epoch ≥ 20 regime with paired CIs against epoch and partial
  correlations given epoch. That regime is now weakening on square.
* Fix the plan3 step count in the x6 text (16 executed steps). Define the square reach baseline against the
  handle site if the geometric reach is reused.
* The infrastructure lesson is confirmed (montage evidence): reusing one offscreen env per task across policy
  loads corrupts or flips frames.

## 8. Limits of this verification

* No new renders and no new VLM calls. I could not test whether a different prompt, Cosmos3-Edge, or a second
  camera view would change the per-checkpoint picture. The geometry η² argues that it would not.
* The "declared before looking" claims (the epoch ≥ 20 split, the thresholds) cannot be audited from files.
  Threshold robustness was checked instead.
* The partial-correlation tests have low power (n = 14–16, 3 distinct values). A null there is not evidence of
  absence.
