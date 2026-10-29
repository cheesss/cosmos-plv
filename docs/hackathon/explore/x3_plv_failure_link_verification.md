# x3_plv_failure_link: adversarial verification

**Status: PARTIALLY CONFIRMED.** EXPLORATORY / post-hoc, like the work it checks. No GPU runs and no new Nebius calls (0 of the 30 allowed). Every number below comes from `results/hackathon/explore/x3_plv_failure_link_verification.json`, made by `scripts/hackathon/explore/x3_plv_failure_link_verify.py`, or from the explorer's own JSON.

## 요약 (Korean TL;DR)

- **재현은 완벽합니다.** 분석 스크립트를 다시 돌리면 AUROC, CI, 체크포인트 상관 690+행이 모두 같습니다(max |diff| 0.0). 에피소드 특징 246개도 explorer CSV와 같습니다. 프롬프트 해시 252/252 일치, dev→eval 호출 순서, 파서 fallback, 프레임 정렬(frame k = step 2k)도 모두 확인했습니다. 누설(outcome으로 점수를 만들어 같은 outcome에서 평가)은 없습니다.
- **헤드라인 1 (no_grasp > dropped, 그리퍼 차원 불일치): 방향은 맞지만 주장보다 약합니다.**
  - 학습된 can에서 no_grasp 5개는 체크포인트 3개에서 나왔고, 그중 4개가 epoch 20입니다.
  - 체크포인트 내부 비교 쌍은 11개뿐입니다. within-ckpt AUROC 0.09, permutation p = 0.036, dev exact p = 0.05입니다.
  - 이 값은 400 step 전체 평균, 즉 실패 *이후* 행동입니다. 처음 80 step에서는 within-ckpt 0.36(p = 0.34), dev 0.37(p = 0.30)로 유의하지 않습니다. 사후 설명 변수이지 조기 경보가 아닙니다.
- **헤드라인 2 (drop 전조 없음): can에서는 맞고, square는 판단 불가이며, 근거로 든 p = 0.95는 무효입니다.**
  - 무작위 시점 null의 평균 73%가 drop *이후* 시점이라 검정이 편향돼 있습니다. drop 이전 시점만 쓴 null로 바꾸면 p = 0.009로 뒤집힙니다(대신 위상 교란이 있음).
  - 위상 일치 대조(같은 체크포인트 성공 에피소드의 '첫 lift 후 같은 지연')로 보면 can은 0.58 [0.42, 0.71]로 전조가 없습니다.
  - square는 0.78 [0.52, 0.93](체크포인트 4개)로, 원래 보고된 0.24와 방향이 반대입니다.
  - 불일치는 drop *순간과 직후*에 크게 튑니다(can, 직후 0.96 [0.85, 1.00]). 따라서 "drops leave no trace"는 틀린 표현입니다.
- **헤드라인 3 (Nemotron drop timer, 1/16 → 7/16): 숫자는 맞지만 drop을 구분하지 못합니다.**
  - 실패(400 step) 에피소드 안에서만 보면 민감도 9/21, 특이도 3/9로 balanced accuracy 0.38입니다. 우연보다 낮습니다.
  - 특이도 0.96은 짧은 성공 클립을 성공으로 맞힌 결과입니다(149/150). 프롬프트에 클립 길이도 적혀 있습니다.
  - "클립이 400 step까지 갔으면 drop"이라는 outcome-free 규칙은 민감도 21/21, 특이도 0.94, PPV 0.70으로 drop timer보다 낫습니다.
  - 체크포인트 수준에서도 실패율 자체가 sim drop rate를 더 잘 따라갑니다(can ρ 0.99 vs drop timer 0.60).
  - v2 대비 늘어난 drop 6개에는 오경보 6개가 함께 붙습니다(McNemar 각각 p = 0.031).
- **체크포인트 수준(R4):** ρ 0.91과 −0.79는 재현됩니다. 하지만 epoch을 통제하면 0.53 [−0.42, 0.89], −0.23 [−0.98, 0.86]이 됩니다. 학습 진행도 이상의 정보는 보이지 않습니다.
  - dropped share 8개 값 중 5개는 실패 1개로 만든 1.0입니다.
  - 새 체크포인트 하나(can_s2_e0180)만 더해도 0.91이 0.78로 떨어집니다.
- **Cosmos v7 '7/12 drop 포착'은 drop에만 나타나는 신호가 아닙니다.** 같은 held→not-held 전환이 학습된 can no_grasp 5/5, can 성공 24/83, square 성공 28/28에서도 나타납니다.
- **데이터 변경:** 실행 도중(05:15) `refresh_truth.py`와 새 annotator가 square 라벨 17개(eval 12, dev 5)를 바꿨습니다. 검증은 explorer가 쓴 라벨로 했고, 현재 라벨 결과는 별도로 기록했습니다.

## What was checked and how

| check | method | outcome |
|---|---|---|
| Reproduction | Re-ran `x3_plv_failure_link.py` into scratch. Compared all AUROC rows, checkpoint correlations and drop-timer confusion with the published JSON. | Identical (max abs diff 0.0 on 690 AUROC rows and every checkpoint Spearman). Same 21 checkpoints and 210 + 36 episodes. |
| Feature identity | Recomputed features through the explorer's own functions, pinned to its 21-checkpoint snapshot and its action scale, then compared with its episode CSV. | 246/246 matched; max diff 0.0 for overlap, gripper overlap, prefix and pre-lift overlap, drop_t, first_lift, release_t. |
| Prompt integrity | sha256 of `build_prompt` / `build_prompt_v2` (task, duration) vs `prompt_sha256` in the Nebius evidence log. | 252/252 match. Order: dev v1 (19:55-19:56Z), dev v2 (19:56-19:57Z), eval v1 (19:57-20:01Z). No eval call for v2. 0 errors. 252 of 800 calls used. |
| Parser | Repaired the unquoted `explanation` with a regex, then strict `json.loads`; compared with the field-regex fallback. | 180/180 eval rows (and 36 + 36 dev rows) agree on all fields. The fallback is safe. |
| Frame/step alignment | `video_frames` vs `n_steps` in every summary; the drop-timer resampling map. | `video_frames = ceil(n_steps/2)` for 256/256 episodes, consistent with frame k = step 2k. The 4-fps stamp t = 0.25k s maps to sim step 2*round(2.5k), so the stamp is off by at most 0.05 s. Drop-time errors are measured correctly. |
| Leakage | Read both scripts. | No score is fitted on outcomes. Feature directions are fixed in advance and thresholds are outcome-free quantiles (other run for can). The drop-timer rule is fixed in advance and its variant was chosen on dev only. One mild point: the drop question itself was motivated by eval-split errors of the v2 diagnostic. That is hypothesis generation, not score fitting. |

## Findings by claim

### C1. "High gripper-dim disagreement points to no_grasp, not drops" (AUROC dropped > no_grasp 0.13 [0.00, 0.21], dev 0.15)

Can, trained (epoch > 0), snapshot labels: 12 dropped vs 5 no_grasp. All of these episodes are 401 steps long, so episode length does not confound this contrast.

| feature | pooled AUROC [cluster CI] | leave-one-ckpt-out range | within-ckpt (11 pairs, 3 ckpts) | within-ckpt permutation p (≤) | dev (3 vs 9) exact p |
|---|---|---|---|---|---|
| gripper overlap, episode mean | 0.13 [0.00, 0.21] | 0.06-0.15 | 0.09 | 0.036 | 0.15, p = 0.05 |
| overall overlap, episode mean | 0.18 [0.04, 0.36] | 0.08-0.25 | 0.27 | 0.20 | n/a |
| gripper overlap, first 80 steps | 0.27 [0.07, 0.40] | 0.17-0.29 | 0.36 | 0.34 | 0.37, p = 0.30 |
| overall overlap, first 80 steps | 0.30 [0.07, 0.50] | 0.14-0.33 | 0.45 | 0.51 | n/a |
| gripper flips, first 80 steps | 0.38 [0.04, 0.69] | 0.22-0.55 | 0.45 | 0.50 | 0.70, p = 0.89 |

- **The direction is robust.** No single checkpoint drives it, and the within-checkpoint permutation test and the dev split agree, each at about p = 0.04-0.05.
- **The clustered CI is optimistic.** The no_grasp class sits in only 3 checkpoints: can_s1_e0020 (3), can_s1_e0040 (1) and can_s2_e0020 (1). Four of the five are epoch 20, and only 11 within-checkpoint pairs exist.
- **It is a description of post-failure behaviour.** A no_grasp episode spends 400 steps on repeated grasp attempts. In a drop episode the gripper stays closed, transporting nothing. In the first 80 steps, which is mostly before the failure shows, the separation shrinks and is not significant within checkpoint or on dev.
- It also needs the rollout itself (real-feedback replanning), so it is not a cheap offline screen.
- **Verdict: confirmed as a post-hoc descriptor, not confirmed as a predictor.** "Points to no_grasp" should read "co-occurs with no_grasp over the failed episode".

### C2. "Drops have no precursor" (random-time null p = 0.95; pre-lift within-ckpt 0.43; before drop vs before release 0.41 can / 0.24 square)

- **The random-time null is invalid as evidence.** On average 73% of its admissible event times (minimum 6%) fall *after* the drop, into high-disagreement post-drop flailing. With the null restricted to times before the drop in the same episode, the same pre-drop window is *elevated*:
  - overall overlap: +0.19 log units, p = 0.009, 21 drops;
  - position overlap: p = 0.007;
  - can only, overall overlap: p = 0.013.

  This comparison is itself confounded by phase (grasp and lift versus approach), so neither null supports a conclusion. Only phase-matched tests count.
- **A better phase-matched control.** Each drop is compared with the successful episodes of the *same checkpoint*, evaluated at the same delay after *their* first lift. The control episode must still be holding the object.

| task | window | overall overlap | gripper overlap | position overlap | drops / pairs / ckpts |
|---|---|---|---|---|---|
| can | 3 boundaries before the drop | 0.58 [0.42, 0.71] | 0.49 [0.32, 0.63] | 0.53 [0.31, 0.74] | 12 / 72 / 8 |
| can | boundary at the drop (lag 0) | 0.65 [0.36, 0.91] | 0.72 [0.50, 0.92] | 0.69 [0.45, 0.91] | 12 / 68 / 8 |
| can | 3 boundaries after | 0.96 [0.85, 1.00] | 0.79 [0.42, 0.96] | 0.83 [0.47, 1.00] | 11 / 47 / 5 |
| square | 3 boundaries before the drop | 0.78 [0.52, 0.93] | 0.66 [0.52, 0.75] | 0.69 [0.33, 0.94] | 9 / 32 / 4 |
| square | boundary at the drop | 0.81 [0.70, 1.00] | 0.77 [0.53, 1.00] | 0.81 [0.57, 1.00] | 9 / 31 / 4 |

- **Can: confirmed.** There is no precursor before can drops; can drops occur 1-48 steps after the first lift, and 9 of 12 within 9 steps.
- **Square: not established.** With a lift-matched control the pre-drop disagreement is *higher* than in successes (0.78, CI above 0.5). That is the opposite of the reported release-matched 0.24. With 4 checkpoints, 9 drops and drop times that may be early, square is **inconclusive**, not "no signal".
- **"Drops leave no trace in it"** (usefulness text) **is false.** Disagreement rises at the drop boundary and strongly afterwards. The defensible claim is "no *early warning*".
- A side check supports the explorer's reading for can. An executed gripper-open command within 8 steps before the drop occurs in only 3/12 can drops [Wilson 0.09, 0.53], so most can drops are slips with the gripper commanded closed, not policy indecision.

### C3. Nemotron drop timer: "sensitivity 1/16 → 7/16 at specificity 0.94"; eval 9/21, specificity 0.96, PPV 0.60; checkpoint ρ 0.60

The numbers reproduce. Their interpretation does not hold.

| detector (trained eval, snapshot labels) | episodes | TP | FN | FP | TN | sensitivity | specificity | balanced acc. |
|---|---|---|---|---|---|---|---|---|
| drop timer v1 | all 180 | 9 | 12 | 6 | 153 | 0.43 [0.24, 0.63] | 0.96 [0.92, 0.98] | 0.70 |
| drop timer v1 | **failed only, 30** (21 drop / 8 no_grasp / 1 other) | 9 | 12 | 6 | **3** | 0.43 | **0.33 [0.12, 0.65]** | **0.38** |
| outcome-free "clip ran to the 400-step horizon ⇒ drop" | all 180 | 21 | 0 | 9 | 150 | 1.00 [0.85, 1.00] | 0.94 [0.90, 0.97] | 0.97 |
| v2 whole-video diagnostic | 120 common | 1 | 15 | 0 | 104 | 0.06 | 1.00 | 0.53 |
| drop timer v1 | 120 common | 7 | 9 | 6 | 98 | 0.44 | 0.94 | 0.69 |
| horizon rule | 120 common | 16 | 0 | 9 | 95 | 1.00 | 0.91 | 0.96 |

- **Specificity comes from clip length, not from drop recognition.**
  - Successful clips last at most 15.0 s and failed ones at least 20.25 s, and the prompt states the duration.
  - Nemotron calls 149/150 short clips successes and 7/30 long clips successes, and it never flags a drop on a success.
  - Among failed episodes, where the drop-vs-other question actually arises, the drop timer is below chance (balanced accuracy 0.38). The PPV of 0.60 is below the 0.70 base rate of drops among trained failures.
- **The gain over v2 is a trade, not an improvement.** On the 120 common episodes it finds 6 drops that v2 misses and 0 that only v2 finds (exact McNemar p = 0.031). It also raises 6 false alarms that v2 does not (p = 0.031). In failed episodes, v2's balanced accuracy is 0.53 and the drop timer's is 0.39.
- **Checkpoint level (can, 13 checkpoints):**

  | predictor of the sim drop rate | Spearman ρ [CI] |
  |---|---|
  | drop timer predicted drop rate | 0.60 [0.19, 0.87] |
  | true-positive part only | 0.59 |
  | false-positive part | 0.37 [0.20, 0.77] |
  | failure rate, i.e. horizon-length clips | **0.99 [0.90, 1.00]** |

  In these runs nearly every late can failure is a drop, so counting failures beats the drop timer.
- The prompt variant choice is not cherry-picked. v1 is chosen on dev under both the snapshot labels (balanced accuracy 0.594 vs 0.539) and the refreshed labels (0.643 vs 0.598). The margin is small, though: dev can sensitivity was 0/3 for both variants.
- **Verdict: refuted as evidence that a narrow Nemotron question "partly fills PLV's blind spot".** The prompt raises sensitivity at the cost of false alarms on the hard negatives, and adds nothing over the horizon-length rule on these data.

### C4. Checkpoint level: "mean overlap vs failure rate ρ 0.91; vs dropped share ρ −0.79"

| can, 13 trained checkpoints | ρ [bootstrap CI] | partial ρ given epoch |
|---|---|---|
| episode-mean overlap vs failure rate | 0.91 [0.72, 0.97] (reproduced) | 0.53 [−0.42, 0.89] |
| overlap, first 80 steps, vs failure rate | 0.88 [0.63, 0.96] | 0.10 [−0.82, 0.87] |
| overlap in *successful* episodes only, vs failure rate | 0.85 [0.54, 0.96] | −0.25 [−0.87, 0.62] |
| epoch vs failure rate | −0.88 [−0.97, −0.65] | n/a |
| overlap vs dropped share of failures (8 ckpts) | −0.79 [−0.95, −0.58] (reproduced) | −0.23 [−0.98, 0.86] |

- **The post-failure tail is not what drives ρ 0.91.** Prefix-only overlap and success-only overlap still correlate.
- **Disagreement carries nothing measurable beyond training progress.** All partial correlations given epoch have CIs spanning 0. The explorer's own text says it "mostly tracks training progress", but the headline and findings present ρ = −0.79 as a disagreement result.
- **The dropped-share variable is degenerate.** Five of its 8 values are 1.0, each built from a single failed episode. Only 3 checkpoints have any no_grasp at all: can_s1_e0020, can_s1_e0040 and can_s2_e0020. A bootstrap CI of [−0.95, −0.58] on such data is not trustworthy.
- **The result is fragile.** Adding the one checkpoint that has arrived since (can_s2_e0180: 1 drop, lowest overlap) moves overlap vs failure rate from 0.91 to 0.78 (14 checkpoints).

### C5. Other claims

- **Cosmos v7 "held → released in 7/12 drops": not drop-specific.** The same transition appears in:
  - 5/5 trained can no_grasp episodes;
  - 24/83 can successes;
  - 28/28 square successes (a release at the goal is also a release).

  v7 is still running; this was 170 joined rows at check time.
- **"Gripper flips, first 80 steps, failure vs success 0.83 [0.68, 0.89]"** partly contains the failure itself (7 of 12 can drops happen inside the window). Every first lift happens at step 44 or later, so a 40-step window precedes all of them. In that window, can gives 0.73 [0.59, 0.83] and 0.74 [0.55, 0.88] within checkpoint, while square gives 0.47. **The claim survives in weaker form, for can only.** The signal is early gripper closing: 13/17 failures vs 34/113 successes close before step 40.
- **"Nemotron v2 no_grasp rate tracks sim, ρ 0.95 (11 can ckpts)"** includes the two epoch-0 checkpoints, which are 100% no_grasp. On trained checkpoints only it is 0.90 (9). The claim "v2 predicts 0 can drops, files them as not_placed" is confirmed: the 11 joined can drops were labelled not_placed 5, no_grasp 3, none 3.
- **The usefulness claim "A PLV user can therefore discard no_grasp-prone checkpoints cheaply"** is unsupported. The disagreement used here needs the closed-loop rollouts, which also yield success directly. At checkpoint level it adds nothing beyond epoch (C4), and no offline PLV metric was tested.

## Data changed during verification (not the explorer's fault)

- **Labels.** At 05:15 local time (20:15Z), `scripts/hackathon/refresh_truth.py`, using an updated `cosmos_plv/failure_annotator.py`, re-annotated every `summary.json`. Success labels are unchanged. 17 failure types changed, all square:
  - eval square_s1_e0000: 10 episodes from other to no_grasp;
  - eval square_s1_e0030: seed 200001 from no_grasp to dropped, seed 200004 from other to no_grasp;
  - dev_square_s1_e0020: 5 episodes, one of them no_grasp to dropped.
- **Handling.** The verification uses the explorer's snapshot labels (`--labels snapshot`, taken from its episode CSV). Under the current labels, the drop timer on trained eval scores TP 10 / FN 12 / FP 5 / TN 153. In failed episodes it scores 10 / 12 / 5 / 3, a balanced accuracy of 0.41, so the conclusion is unchanged.
- **Definition mismatch.** The relabelled square_s1_e0030 seed 200001 rose only 3.9 cm above its start. x3's 5 cm-above-start lift rule therefore gives it no lift and no drop time, while the new annotator calls it lifted and dropped. The x3 event definitions and the refreshed taxonomy no longer match for square.
- **New checkpoint.** can_s2_e0180 (10 episodes, 1 drop) arrived after the snapshot. With it, C1 is unchanged (gripper overlap 0.12 [0.00, 0.20], permutation p = 0.036) and C4 weakens as shown above.

## Commands run (Environment B, CPU, repo root) and exit codes

| command | exit |
|---|---|
| `PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/x3_plv_failure_link.py --out data/outputs/hackathon_explore/x3_plv_failure_link_verification/rerun.json --fig data/outputs/hackathon_explore/x3_plv_failure_link_verification/rerun_eta.png` (its side CSVs also land in that scratch folder) | 0 |
| inline python: Nebius evidence-log tag, order and prompt-hash check (`x3_nemotron_drop_timer.build_prompt*` vs `prompt_sha256`) | 0 |
| `PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/x3_plv_failure_link_verify.py` (first version, unpinned; superseded because a new checkpoint and the label refresh landed mid-run) | 0 |
| `PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/x3_plv_failure_link_verify.py --labels snapshot` (final; 2000 bootstraps, 5000 permutations, 88 s) | 0 |

No GPU use, no Nebius calls, no existing file modified. Scratch is in `data/outputs/hackathon_explore/x3_plv_failure_link_verification/`.

## Corrected headline

EXPLORATORY, partially confirmed. The analysis reproduces exactly.
- **no_grasp vs drops.** Over whole 400-step failed episodes in trained can checkpoints, gripper-dim replanning disagreement is higher in no_grasp than in drops: AUROC 0.13, within-checkpoint 0.09 over 11 pairs in 3 checkpoints, permutation p = 0.036, dev exact p = 0.05. This is post-failure behaviour: in the first 80 steps the within-checkpoint value is 0.36 (p = 0.34).
- **Precursor.** Before can drops, a lift-offset-matched control shows no precursor (0.58 [0.42, 0.71]). Square is inconclusive (0.78 [0.52, 0.93], 4 checkpoints). Disagreement jumps at and after the drop.
- **Drop timer.** The time-stamped Nemotron prompt does not tell drops from other failures. In failed episodes it scores 9/21 sensitivity at 3/9 specificity (balanced accuracy 0.38), worse than the outcome-free rule "clip ran to the horizon" (21/21, overall specificity 0.94).
- **Checkpoint level.** Failure rate tracks the can drop rate better than the drop timer (ρ 0.99 vs 0.60). Mean disagreement adds nothing beyond epoch (partial ρ 0.53 [−0.42, 0.89]).

## What would settle the open points

- **Square precursor.** Rerun C2 with the refreshed annotator's drop times and more square checkpoints (more are being recorded). Report the lift-matched control alongside the release-matched one.
- **Drop timer.** Evaluate any drop detector **inside failed episodes only**, against the horizon-length baseline and the failure-rate baseline at checkpoint level. Remove the duration from the prompt, or pad clips to a fixed length, so that success cannot be read from clip length.
- **no_grasp signature.** Test it in an offline (model-feedback) PLV metric on held-out checkpoints, with no_grasp coming from more than 3 checkpoints, before calling it a screening rule.
