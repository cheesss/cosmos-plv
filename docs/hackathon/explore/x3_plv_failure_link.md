# x3_plv_failure_link: what failure a PLV disagreement signal points to

**EXPLORATORY / POST-HOC.** Nothing here was pre-registered. No score was fitted on outcomes: every feature is used raw, with a direction fixed in advance (higher = worse). The "high" thresholds are outcome-free quantiles. Every number below comes from `results/hackathon/explore/x3_plv_failure_link.json`, as generated at 2026-10-28T20:02Z (UTC), or from its side CSVs.

## 요약 (Korean TL;DR)

- **재계획 불일치가 높을 때 가장 먼저 의심할 실패는 no_grasp(파지 실패)입니다.** 특히 그리퍼 차원의 불일치가 높으면 no_grasp를 가리킵니다. 학습된(epoch>0) can 체크포인트에서 실패 에피소드만 놓고 dropped와 no_grasp를 비교한 결과입니다.
  - 그리퍼 차원 overlap의 AUROC(dropped > no_grasp)는 0.13 [0.00, 0.21]입니다. 이는 no_grasp 쪽이 더 높다는 뜻이며, dev split에서도 0.15로 재현됐습니다.
  - 위치 차원 불일치는 두 실패 유형을 구분하지 못합니다(0.37 [0.07, 0.89]).
- **drop(놓침)에 앞서 불일치가 튀는 현상은 없습니다(음성 결과).** 비교는 세 가지로 했습니다.
  - 같은 에피소드의 무작위 시점과 비교: p=0.95(단측)로 증가 없음.
  - 같은 체크포인트의 성공 에피소드에서 의도적으로 놓는 순간과 비교(위상 매칭): within-checkpoint AUROC는 can 0.41, square 0.24입니다.
  - 첫 들어올림 이전의 불일치만 사용: can 0.43 [0.25, 0.60]로 예측력이 없습니다.
  - 그리퍼 불일치 spike는 성공 에피소드의 release와 lift 시점에도 똑같이 크게 나타납니다. 즉 실패의 전조가 아니라 그리퍼 전환 구간의 표지입니다.
- **체크포인트 수준에서는 불일치가 클수록 실패 중 drop 비중이 낮습니다.** 학습된 can 체크포인트에서 평균 불일치와 실패 중 dropped 비율의 Spearman은 −0.79 [−0.95, −0.58]입니다.
  - 불일치가 큰 초기 체크포인트는 주로 no_grasp로 실패합니다.
  - 불일치가 낮은 후기 체크포인트에 남는 실패는 drop입니다. 이 drop은 PLV 불일치로는 보이지 않습니다.
- **Nemotron을 drop 전용 질문과 타임스탬프 오버레이로 쓰면 PLV의 사각지대를 일부 메웁니다.** 같은 120개 에피소드에서 비교했습니다.
  - drop 민감도는 기존 v2의 1/16에서 7/16으로 올랐습니다.
  - 특이도는 1.00에서 0.94로 내려갔습니다.
  - eval 180개 기준으로는 민감도 9/21(0.43 [0.24, 0.63]), 특이도 153/159(0.96 [0.92, 0.98])입니다.
  - 다만 drop 시각 오차가 중앙값 2.75 s로 커서, drop 시점에 정렬하는 분석에는 아직 쓸 수 없습니다.

## Question

PLV ranks checkpoints by cheap disagreement and smoothness metrics. When a checkpoint, or one of its episodes, shows high execution-matched replanning disagreement on real simulator feedback, which physical failure does that predict? The candidates are no_grasp, dropped and other. Does disagreement spike *before* a drop?

## Why it could help PLV or efficiency

- **Interpretability.** A PLV user currently sees only "metric X is bad". If high disagreement reliably means "cannot close the grasp" but never "will drop the object", the user knows which failure mode PLV can catch. The user also knows which mode needs another detector.
- **Efficiency.** If drops are invisible to PLV, then spending VLM or rollout budget only on checkpoints that look good on PLV is the right allocation. The leftover failure mode there is drops, and the VLM can be asked a narrow question about drops.

## Data snapshot (when the final run executed)

Rollouts were still being added while this ran. The script is re-runnable and picks up whatever exists.

| split | episodes | checkpoints | can (none / no_grasp / dropped) | square (none / no_grasp / dropped / other) |
|---|---|---|---|---|
| eval (seeds 200000-200009) | 210 | 21 (can_s1 e0-100, can_s2 e0-160, square_s1 e0-150) | 113 / 25 / 12 | 37 / 3 / 9 / 11 |
| dev (seeds 300000+, replication only) | 36 | 3 | 12 / 9 / 3 | 3 / 1 / 4 / 4 |

- `results/hackathon/baseline_metrics.csv` does not exist yet. The checkpoint-level "cheap metrics" are therefore val_loss and train_action_mse_error from `logs.json.txt`, plus the training-time success rate (`test/mean_score`, 50 episodes, seeds 100000-100049).
- The epoch-0 checkpoints fail every episode (can: all no_grasp; square: all other). The **primary cohort is `eval_trained_epoch_gt0`**: 18 checkpoints and 180 episodes. The all-checkpoint numbers are in the JSON.

## Method

1. **Episode-level features**, from the recorded plans (`action_pred`, `plan_t`; horizon 16, n_obs 2, n_action 8).
   - The features are built on `cosmos_plv.cheap_metrics.replan_metrics_from_plans`. The per-dimension scale is the std of the 10-D predicted actions, pooled over all eval episodes of the task, which uses no outcomes.
   - A per-boundary re-implementation reproduces the helper exactly (max |diff| = 0.0). It also adds:
     - dimension groups (pos 0:3, rot6d 3:9, gripper 9);
     - max and p90 over boundaries;
     - executed smoothness and the number of gripper sign flips (chatter);
     - time-local versions:
       - first 80 steps: the earliest success in any split is at step 82, so this window precedes every success;
       - boundaries up to the first lift: this window can never contain a drop;
       - ±16 steps around the first reach and the first lift.
   - Contrasts are computed within each task. AUROC uses a **checkpoint-clustered bootstrap** (2000 resamples of whole checkpoints).
   - A **within-checkpoint AUROC** is also reported. It compares only (positive, negative) pairs from the same checkpoint, so checkpoint quality and training progress cannot drive it.
   - "High vs low" composition uses a 75th-percentile threshold taken from the *other* can run (leave-one-run-out). For square, which has one run, the threshold is pooled.
2. **Temporal analysis.** Events come from the npz traces:
   - first lift: object ≥ 5 cm above its start;
   - drop: the first time after the lift that the object falls below 2 cm, or that grasp contact is absent for ≥ 3 consecutive steps while the object is still ≥ 2 cm up;
   - control: the intentional release in successful episodes, using the same contact-lost rule.

   An event-triggered average is computed on within-episode log-normalised disagreement (log d − median log d of that episode), in 0.4 s chunk bins. The pre-event statistic is the mean of the three boundaries before the event. It is compared with:
   - (a) a null that places the event at random times in the same episode (5000 permutations);
   - (b) a raw, phase-matched comparison: the three boundaries before a drop versus the three before a successful release, and the three before the first lift in dropped versus successful episodes. This comparison reports pooled and within-checkpoint AUROC.
3. **Checkpoint level.** For each checkpoint: the failure composition (simulator labels, and Nemotron v2 predictions from `eval_nebius_v2.jsonl`) against epoch, training-time success, val_loss, train_action_mse_error and the mean real-feedback disagreement. Spearman correlations within task, with a bootstrap over checkpoints.
4. **Nemotron drop timer** (new; 252 Nebius calls, 0 backend errors).
   - Videos are resampled to exactly 4 fps with a burnt-in `t=…s` stamp, so the Nebius client does not re-time them.
   - The prompt asks one narrow question: did the object leave the gripper before the goal, and at which time stamp?
   - The rule is fixed in advance: predicted drop = `left_gripper_before_goal ∧ ¬task_success`.
   - Two prompt variants were tried on **dev only**: v1 had balanced accuracy 0.594 and v2 had 0.539, so v1 was frozen.
   - v1 was then run on every eval episode of an epoch > 0 checkpoint (180 episodes). Selection used no outcomes.
   - The same comparison was run on the existing v2 diagnostic and on Cosmos v7 keyframe tracks.

## Commands run (Environment B, CPU, from the repo root)

| command | exit |
|---|---|
| `PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/x3_plv_failure_link.py --n-boot 100 --n-perm 200 --out data/outputs/hackathon_explore/x3_plv_failure_link/smoke.json --fig data/outputs/hackathon_explore/x3_plv_failure_link/smoke.png` | 1 (`omegaconf` is missing in Env B; fixed with a log-parsing fallback), then 0 on the smoke reruns |
| `… python scripts/hackathon/explore/x3_nemotron_drop_timer.py --split dev --limit 2 --workers 2` | 0 (2 calls) |
| `… x3_nemotron_drop_timer.py --split dev --workers 4` (v1) | 0 (34 calls) |
| `… x3_nemotron_drop_timer.py --split dev --workers 4 --variant droptimer_v2` | 0 (36 calls) |
| `… x3_nemotron_drop_timer.py --split eval --workers 4 --variant droptimer_v1` | 0 (170 calls) and 0 (10 calls, one new checkpoint) |
| `PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/x3_plv_failure_link.py` (final: 2000 bootstraps, 5000 permutations, 73 s) | 0 |

Nebius evidence: `results/hackathon/explore/nebius_calls_explore.jsonl`, tag `explore/x3_plv_failure_link/droptimer/<variant>/<split>/<ckpt>/<seed>`. The total is 252 calls against the 800 cap.

## Results

### R1. Episode level, trained checkpoints. AUROC is P(feature higher in the first group); [95% checkpoint-clustered CI]

| task | contrast (n) | replan overlap (episode mean) | position-dim overlap | gripper-dim overlap | overlap up to first lift | gripper flips, first 80 steps |
|---|---|---|---|---|---|---|
| can | failure vs success (17/113) | 0.91 [0.85, 0.96]; within-ckpt 0.88 | 0.95 [0.89, 0.99] | 0.69 [0.28, 0.86] | 0.70 [0.48, 0.85]; within 0.45 | 0.83 [0.68, 0.89]; within 0.75 [0.62, 0.82] |
| can | no_grasp vs success (5/113) | 0.99 [0.95, 1.00] | 0.97 [0.88, 1.00] | **0.99 [0.97, 1.00]** | n/a | 0.90 [0.81, 0.96] |
| can | dropped vs success (12/113) | 0.88 [0.82, 0.95]; within 0.85 | **0.94 [0.88, 0.99]** | 0.56 [0.23, 0.79] | 0.69 [0.45, 0.85]; **within 0.43 [0.25, 0.60]** | 0.80 [0.65, 0.89] |
| can | dropped vs no_grasp (12/5) | 0.18 [0.04, 0.36] | 0.37 [0.07, 0.89] | **0.13 [0.00, 0.21]** (dev: 0.15) | n/a | 0.38 [0.04, 0.69] |
| square | failure vs success (13/37) | 0.81 [0.55, 0.96]; within 0.92 | 0.87 [0.65, 0.98] | 0.53 [0.38, 0.73] | 0.59 [0.34, 0.78] | 0.57 [0.40, 0.82] |
| square | dropped vs success (9/37) | 0.73 [0.55, 0.93] | 0.82 [0.65, 0.98] | 0.58 [0.44, 0.77] | 0.59 [0.34, 0.78]; within 0.63 [0.19, 0.86] | 0.60 [0.38, 0.84] |

- **Caveat: the episode-mean features are partly post-hoc detectors.** A failed episode runs for 400 steps in off-distribution states, and episode length alone separates success from failure (AUROC 1.00).
  - Only the "up to first lift" column is free of post-failure behaviour.
  - The first-80-step window precedes every success, but it still contains the drop itself for 7 of the 12 can drops.
- Executed smoothness and the gripper-flip *rate* over the whole episode do **not** separate drops from successes (can, dropped vs success: 0.38 and 0.38).

### R2. What "high disagreement" means in practice (can, trained, threshold = q75 of the other run, outcome-free)

| feature | high: n (success / no_grasp / dropped) | low: n (success / no_grasp / dropped) |
|---|---|---|
| replan overlap (episode mean) | 35 (20 / 5 / 10) | 95 (93 / 0 / 2) |
| gripper-dim overlap | 30 (20 / 5 / 5) | 100 (93 / 0 / 7) |
| overlap, first 80 steps | 30 (16 / 5 / 9) | 100 (97 / 0 / 3) |
| gripper flips per 100 steps | 35 (26 / 5 / 4) | 95 (87 / 0 / 8) |

- Every no_grasp episode is flagged "high" by every feature.
- Drops are flagged by the overall and position disagreement, which is largely the post-drop tail. Most drops are *not* flagged by the gripper-dim or chatter features: 7 of 12 and 8 of 12 fall in "low".

### R3. Temporal: is there a precursor before drops? (eval, all checkpoints with drops, 21 events)

| test | statistic | result |
|---|---|---|
| ETA vs random time in the same episode, overall overlap, lags −3..−1 | pre = −0.053 [−0.201, 0.104]; null mean +0.091 | not elevated (p≥ = 0.95) |
| same, gripper dim | +0.452 vs null +0.416 | p≥ = 0.47 |
| same, gripper dim, can only (12 events) | +1.152 [0.59, 1.73] vs +0.309 | p≥ = 0.018, but see the controls in the next two rows |
| gripper dim before an **intentional release** in successes (150) | +1.541 [1.34, 1.76] vs +0.592 | p ≤ 0.0002: the same spike appears in successes |
| gripper dim before the **first lift** in successes (150) | +1.232 vs +0.592 | p ≤ 0.0002 |
| raw, before drop vs before release, within-checkpoint AUROC (trained) | can 0.41 (8 ckpts), square 0.24 (4 ckpts) | no drop-specific signal |
| raw, before first lift, dropped vs success, within-checkpoint AUROC | can 0.49, square 0.41 | none |
| dev replication (7 drops), overall overlap | +0.079 vs +0.129 | p≥ = 0.64 |

Figure: `docs/hackathon/explore/x3_plv_failure_link_eta.png`.

### R4. Checkpoint level (can, trained checkpoints, Spearman [bootstrap CI])

| target | mean replan overlap | epoch | training success (50 eps) | val_loss | train_action_mse | Nemotron v2 no_grasp rate |
|---|---|---|---|---|---|---|
| failure rate (13) | 0.91 [0.72, 0.97] | −0.88 | −0.83 | −0.49 [−0.81, 0.11] | 0.75 [0.36, 0.93] | 0.85 (9 ckpts) |
| no_grasp rate (13) | 0.73 [0.47, 0.88] | −0.71 | −0.69 | −0.08 | 0.55 | 0.90 [0.65, 1.00] (9) |
| dropped share of failures (8) | **−0.79 [−0.95, −0.58]** | 0.78 | 0.68 | −0.41 | −0.35 | −0.89 (7) |

- val_loss does not track any failure type (CIs include 0). It rises at late epochs while success saturates.
- Nemotron v2 checkpoint composition:
  - it tracks the no_grasp rate well (ρ = 0.95, all 11 can checkpoints with v2);
  - it predicts **zero** can drops, so it cannot estimate the drop rate (ρ undefined);
  - it files drops as `not_placed` instead.
- Square has only 5 trained checkpoints, so its checkpoint-level CIs span about [−1, 1].

### R5. Nemotron drop timer (frozen v1; eval = all 180 episodes of epoch > 0 checkpoints)

| | sensitivity (drops found) | specificity | PPV |
|---|---|---|---|
| drop timer v1, eval, all | 9/21 = 0.43 [0.24, 0.63] | 153/159 = 0.96 [0.92, 0.98] | 0.60 [0.36, 0.80] |
| drop timer v1, can / square | 4/12 = 0.33 / 5/9 = 0.56 | 0.97 / 0.93 | |
| same 120 episodes: drop timer v1 | 7/16 = 0.44 [0.23, 0.67] | 0.94 [0.88, 0.97] | |
| same 120 episodes: existing v2 whole-video diagnostic | 1/16 = 0.06 [0.01, 0.28] | 1.00 [0.96, 1.00] | |

- **Timing** is too coarse for event alignment.
  - Drop timer, true positives: median |error| 2.75 s, 3/9 within 1 s.
  - v2 `failure_time_s`: median |error| 3.7 s, 1/12 within 1 s. Many of its answers are 0.0 or the clip end.
  - Cosmos v7 keyframe tracks (2.9 s spacing): a "held → not held" transition appeared in 7 of 12 drops. Against the first keyframe after the simulator drop, 1 was on time, 3 were one keyframe late, 2 were two keyframes late, and 1 was about 8 s early.
  - An ETA on Nemotron-timed events shows nothing (14 flagged drops, overall overlap p≥ = 0.60).
- **Checkpoint level:** the Nemotron-predicted drop rate against the simulator drop rate gives can ρ = 0.60 [0.19, 0.87] (13 checkpoints) and square ρ = 0.55 [−0.97, 1.00] (5).
- **False positives** are mostly no_grasp episodes (5 of 6), where the can is pushed or tipped.

## Interpretation for a PLV user

1. **High real-feedback replanning disagreement means "this policy cannot commit to a grasp".**
   - The gripper-dimension disagreement, and gripper chatter in the first 4 s, point to **no_grasp**. Every no_grasp episode in trained checkpoints is in the top quartile.
   - Position-dimension disagreement is high in *any* failed episode, but largely *after* the failure.
2. **A low-disagreement checkpoint can still drop objects.** Drops are not preceded by policy indecision:
   - there is no pre-drop spike against random times or against the phase-matched successful release;
   - pre-lift disagreement does not predict a later drop within a checkpoint.

   In PLV terms, the residual failure mode of good checkpoints (drops) is invisible to disagreement metrics.
3. **Across checkpoints**, mean disagreement mostly tracks training progress. A high-disagreement checkpoint fails through no_grasp. A low-disagreement checkpoint that still fails, fails through drops (ρ = −0.79).
4. **Efficiency implication.** Use PLV disagreement to discard no_grasp-prone checkpoints cheaply. Spend VLM calls only on the surviving low-disagreement checkpoints, with a narrow drop question.
   - The focused, time-stamped Nemotron prompt raises drop sensitivity from 0.06 to 0.44 on the same episodes, at about 3 s per call.
   - It is still a weak detector: 0.43 sensitivity on eval, and it misses most fast can drops that happen within 0.5 s of lift-off.

## Limitations

- **Small numbers.** There are 21 eval drops in total (12 can, 9 square) and 5 no_grasp episodes from trained can checkpoints. Square has one run and 5 trained checkpoints, so its within-checkpoint CIs rest on 4 clusters.
- **Episode-length confound.** Failed episodes run to 400 steps. Episode-mean features include post-failure behaviour, so only the time-local, pre-lift windows are clean "predictive" tests. The 80-step prefix still contains about half of the can drops.
- **Drop-time definition.** Drop time is the first debounced contact loss while lifted. In at least 3 square drops, contact was regained afterwards, so the event time may be early. Several can "drops" are failed lift-offs within 1-9 steps of the first lift.
- **Normalisation.** Within-episode normalisation makes pre-drop windows look *lower* than random times, because post-drop flailing has high disagreement. The phase-matched raw comparison is the test to rely on.
- **Prompt variants.** The two drop-timer variants were chosen on 36 dev episodes. 148 of 180 eval outputs were malformed JSON (unquoted explanation) and were read with a field-level regex fallback.
- **Training still running.** New checkpoints are appearing, and all numbers are a snapshot.
- Post-hoc and exploratory; no multiple-comparison correction across the many features and contrasts.

## What to try next

- Let the drop timer run on the stronger checkpoints as they appear. Re-run `x3_nemotron_drop_timer.py --split eval` (it skips episodes it has already judged), then re-run the main script.
- Get a better drop timer: crop around the gripper, or use 8-10 fps clips limited to ±3 s around the first lift. The first-lift time is known on a real robot from gripper width plus force sensing.
- Model-feedback (paper PLV-1) per-boundary disagreement, as a check on whether the no_grasp signature exists offline without rollouts. This needs `baseline_metrics.csv` and the paper scripts.
- Redo R4 once `baseline_metrics.csv` exists (Action L1 and open-loop smoothness against failure composition).

## Files

- Scripts:
  - `scripts/hackathon/explore/x3_plv_failure_link.py` (analysis, CPU, no API)
  - `scripts/hackathon/explore/x3_nemotron_drop_timer.py` (Nemotron probe)
- Results:
  - `results/hackathon/explore/x3_plv_failure_link.json`
  - side CSVs `x3_plv_failure_link_{episodes,auroc,checkpoints,ckpt_correlations}.csv` in the same folder
- Scratch: `data/outputs/hackathon_explore/x3_plv_failure_link/` (time-stamped videos, Nemotron JSONL outputs, run logs)
- Rerun: `cd /path/to/repo && PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/x3_nemotron_drop_timer.py --split eval --workers 4 && PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/x3_plv_failure_link.py`
