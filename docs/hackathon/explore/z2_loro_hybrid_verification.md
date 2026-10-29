# z2_loro_hybrid 적대적 검증 (EXPLORATORY, 사후 분석)

**판정: partially_confirmed (부분 확인).**

- **확인된 것.** 핵심 음성 결과는 유지되고, 오히려 더 강해졌다. 학습 조합도, VLM K=10 특징도 이 풀에서 epoch과 구별되지 않는다.
  - 원 스크립트를 다시 돌린 결과가 bit 단위로 같았다. `z2_loro_hybrid.json`의 숫자 16,985개를 비교했고 max abs diff는 0이다. 입력 sha256도 같다.
  - z2 코드를 import하지 않은 독립 구현(로그 parser, 기대 Spearman, ridge, backtracking Newton으로 푼 pairwise-logit, 중첩 λ, unitweight, select)이 held-out run별 ρ를 정확히 재현했다. all93 36개, eval33 53개, late93 36개 method 전부 차이 0이고, swap도 같다.
  - 누출은 찾지 못했다. fit과 inner CV는 훈련 run의 HA만 쓴다. held-out run의 HB는 채점에만 쓴다.
  - **새로 확인한 것.** 무작위 25/25 seed 분할 400개에서 epoch보다 평균 Δ가 양수인 method는 세 풀 모두 0개다.
- **반박되거나 약해진 것.**
  1. **헤드라인의 +0.009는 운 좋은 분할 하나의 값이다.** pairlogit_rank[all]의 +0.009는 400개 무작위 분할 분포의 98번째 백분위수다. 분할 평균 Δ는 −0.016이고, P(Δ>0)는 0.09다. ridge_rank[all]도 97번째 백분위수(평균 −0.018)다. RA(epoch, L1)는 95번째 백분위수(평균 −0.012)다. 원래 HB 절반은 epoch에게도 유리하다(epoch ρ가 88번째 백분위수).
  2. **plateau에서 Action L1 "+0.064, 3/3 run"은 분할 운이다.** 무작위 분할에서 이 값은 95번째 백분위수이고, 평균 Δ는 −0.025, P(Δ>0)는 0.32다. 50개 에피소드 전체를 타깃으로 쓰면 Δ는 −0.044 [−0.136, +0.056]이고, 3개 run 중 1개에서만 이긴다. 게다가 late can run의 타깃은 거의 퇴화되어 있다. can_seed2는 11개 중 10개, can_seed1은 21개 중 17개 체크포인트가 HB=1.0이다.
  3. **"모든 +vlm 모델이 VLM 없는 짝과 같거나 낮다"는 틀렸다.** 8쌍 중 3쌍에서 +vlm이 더 높다. pairlogit_rank[all] +0.015, ridge_rank[small] +0.010, unitweight[small] +0.004이고 1쌍은 같다. "+vlm 모델 0.815–0.833"도 틀렸고, 실제 범위는 0.793–0.833이다. 다만 결론(epoch을 넘지 못함)은 유지된다.
  4. **절대 ρ의 bootstrap CI는 유효한 CI가 아니다.** 블록 bootstrap에서는 36개 method 중 26개의 점추정치가 자기 95% CI 밖에 있다. epoch은 0.825인데 CI가 [0.370, 0.816]이다. 에피소드-only bootstrap에서도 20/36이 밖에 있다(epoch [0.637, 0.824]). 학습 모델의 −0.4 왼쪽 꼬리는 재표본 설계 탓이다(5절).
  5. **검정력 산술이 낙관적이다.** "8–17 run"은 대략 검정력 50%에 해당하는 수(9–18)다. 양측 α=0.05에서 검정력 80%를 얻으려면 17–34 run이 필요하다. 그런데 분할 평균 Δ가 음수라서 이 계산 자체가 의미가 없다.
  6. 작은 과장 몇 가지가 있다.
     - "run-level 95% CI"는 3개 fold의 [min, max]다. 모집단 중앙값을 덮을 확률은 75%뿐이다.
     - "pairlogit_z[all] = epoch 순위(3 fold 모두)"는 정확하지 않다. can_seed2에서 epoch 140과 160이 뒤바뀐다. 둘 다 HB=1.0이라 ρ는 같다.
     - late93 학습 모델 범위 "0.05–0.17"의 실제 값은 0.042–0.174다(unitweight는 0.025–0.145).

## 0. 사용자 질문에 대한 확인: "네모트론이나 코스모스 아웃풋 저장해놓은 거 있어?"

z2가 답한 내용을 직접 확인했고 맞다.

- 위치: `data/outputs/hackathon_cosmos/`
  - `eval_local_v6.jsonl`: Cosmos3-Edge, 마지막 프레임, 330행
  - `eval_local_v7.jsonl`: Cosmos3-Edge, 8 keyframe, 330행
  - `eval_nebius_v2.jsonl`: Nemotron(Nebius), 전체 영상, 330행
  - `eval_nebius_v2_pad201.jsonl`: Nemotron, 길이 중립 클립, 330행
- 이 밖에 `dev_prompt_selection.jsonl`, `dev_nebius_v2_pad201.jsonl`, `dev_infra_failures.jsonl`, 그리고 라벨 갱신 전 사본인 `*.orig.jsonl`(150/152/150/330행)이 있다.
- 각 행에는 다음 필드가 있다: `raw_text`, `cosmos_task_success`, `cosmos_failure_type`, `cosmos_explanation`, `latency_s`, `model_id`/`model_revision`, `request_id`, `truth_*`.
- `results/hackathon/raw_model_outputs/`에 사본 7개와 README가 있다. sha256이 원본과 모두 같다.
- Nebius 호출 로그는 `results/hackathon/nebius_calls.jsonl`(772행)과 `results/hackathon/explore/nebius_calls_explore.jsonl`(2380행)이다.
- Cosmos 모드 탐색 출력은 `results/hackathon/explore/x5_cosmos_modes_episodes.jsonl`(297행)이다.

## 1. 방법 (CPU만 사용, GPU와 Nebius 호출 없음)

| ID | 무엇을 | 스크립트 / 출력 |
|---|---|---|
| R | 원 명령을 그대로 재실행(`--n-boot 1000`)하고 JSON 전체를 수치 비교 | `data/outputs/hackathon_explore/z2_loro_hybrid_verify/rerun_main.json` |
| V1 | 독립 재구현으로 고정 HA/HB 분할과 swap의 점추정치를 구해 z2와 비교 | `scripts/hackathon/explore/z2_loro_hybrid_verify.py` |
| V2 | 50개 seed를 무작위로 25/25 분할(400회)하고, 분할마다 중첩 λ를 포함한 LORO 전체를 다시 fit | 같은 스크립트, `V2_random_splits` |
| V3 | LORO-50: 훈련 run은 50 에피소드로 fit하고, held-out run은 자기 50 에피소드로 채점. held-out 결과는 fit에 쓰지 않는다 | `V1_V3_point.*.loro50_*` |
| V4 | fit이 없는 점수를 50 에피소드 타깃에 대고, seed 군집 bootstrap(2000회)으로 Δ CI를 계산. refit이 없어 percentile CI를 쓸 수 있다 | `V4_free_scores_full50` |
| V5 | z2 JSON에서 bootstrap 편향을 진단하고, moving-block 끝점 효과를 측정. z2의 블록 bootstrap을 같은 RNG로 400회 다시 돌려 fold별 원인을 분해 | `V5_bootstrap_bias`, `z2_loro_hybrid_verify_blocktail.py` → `z2_loro_hybrid_verification_blocktail.json` |
| V6 | 주장과 산술 점검: 범위, +vlm 짝 비교, 검정력 | `V6_claims` |

```
cd /path/to/repo
PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python -W ignore scripts/hackathon/explore/z2_loro_hybrid.py --n-boot 1000 --workers 6 \
    --out data/outputs/hackathon_explore/z2_loro_hybrid_verify/rerun_main.json             # exit 0, 249 s
PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python -W ignore scripts/hackathon/explore/z2_loro_hybrid_verify.py --n-splits 400 --n-boot 2000 --workers 10   # exit 0, 105 s
PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python -W ignore scripts/hackathon/explore/z2_loro_hybrid_verify_blocktail.py                              # exit 0, 62 s
```

첫 verify 실행(400 분할)은 exit 0으로 끝났다. 하지만 late93의 V2/V4 요약이 NaN이었다. 일부 분할에서 can_seed2 late 타깃이 상수가 되어 ρ가 정의되지 않았기 때문이다. 정의된 run에 대해서만 평균하도록 고친 뒤 위 명령으로 다시 생성했다. all93와 eval33 수치는 바뀌지 않았다.

## 2. 재현 (R, V1)

- 재실행 JSON과 원본 JSON은 숫자 16,985개가 모두 같다(max abs diff 0). 입력 sha256도 같다.
- 독립 구현과 비교하면 held-out run별 ρ의 차이가 0인 method는 all93 36/36, eval33 53/53, late93 36/36이다. swap 평균 차이도 0이다.
- select_best_on_train의 선택도 z2 보고와 같다.
  - all93: RA(epoch,replan), RA(epoch,TF-MSE), RA(epoch,TF-MSE)
  - eval33: RA(epoch,dyn), plv1, RA(epoch,dyn)
  - late93: RA(epoch,replan), TF-MSE, epoch
- 보고서 표의 수치(Δ, run CI, block CI, P(Δ>0), swap)를 JSON과 대조했다. 5.1, 5.2, 5.3절 모두 일치한다. 예외는 아래 4절의 범위 표현이다.

## 3. 핵심: 고정 분할 하나 대신 무작위 분할 400개 (V2)

held-out ρ(HB 대신 무작위 25 에피소드)의 run 평균 Δ를 epoch과 비교했다. "원 분할 pct"는 z2가 쓴 분할의 Δ가 400개 분포에서 차지하는 백분위수다.

| pool | method | z2 원 분할 Δ | 원 분할 pct | 분할 평균 Δ | 중앙값 Δ | P(Δ>0) | 2.5–97.5% |
|---|---|---|---|---|---|---|---|
| all93 | pairlogit_rank[all] | +0.009 | 0.98 | −0.016 | −0.012 | 0.09 | [−0.060, +0.006] |
| all93 | ridge_rank[all] | +0.008 | 0.97 | −0.018 | −0.011 | 0.12 | [−0.075, +0.009] |
| all93 | pairlogit_z[small] | +0.005 | 0.85 | −0.005 | −0.001 | 0.47 | [−0.056, +0.010] |
| all93 | ridge_rank[small] | +0.004 | 0.87 | −0.004 | −0.004 | 0.29 | [−0.020, +0.009] |
| all93 | RA(epoch, L1) | +0.005 | 0.95 | −0.012 | −0.012 | 0.14 | [−0.032, +0.007] |
| all93 | unitweight[all] | +0.000 | 0.92 | −0.008 | −0.008 | 0.08 | [−0.023, +0.004] |
| eval33 | pairlogit_z[all] | 0.000 | — | −0.004 | 0.000 | 0.01 | [−0.056, 0.000] |
| eval33 | pairlogit_z[all+vlm] | −0.018 | 0.35 | −0.014 | −0.014 | 0.10 | [−0.039, +0.012] |
| eval33 | ridge_rank[all+vlm] | −0.022 | 0.25 | −0.020 | −0.012 | 0.17 | [−0.121, +0.019] |
| eval33 | RA(epoch, Nemotron) | −0.025 | 0.49 | −0.022 | −0.025 | 0.14 | [−0.058, +0.025] |
| late93 | Action L1 | +0.064 | 0.95 | −0.025 | −0.023 | 0.32 | [−0.131, +0.078] |
| late93 | RA(epoch, L1) | +0.064 | 0.97 | −0.010 | −0.011 | 0.35 | [−0.083, +0.064] |
| late93 | ridge_rank[small] | −0.193 | 0.22 | −0.106 | −0.020 | 0.36 | [−0.745, +0.160] |

- 평균 Δ가 0보다 큰 method 수는 all93 0/12, eval33 0/24, late93 0/12다. 가장 좋은 것은 train_loss(epoch과 사실상 같은 순서, −0.0004)다.
- "small 모델이 3/3 run 모두 승리"는 분할 400개 중 4–8%에서만 다시 나온다.
- late93에서는 13개 분할에서 run 하나의 타깃이 상수였다. 그 분할에서는 정의된 run만 평균했다.
- **해석.** z2의 "구별되지 않는다"는 맞다. 다만 고정 분할은 학습 조합에 유리한 쪽 꼬리였다. 분할을 평균하면 학습 조합은 epoch보다 약간(−0.004 ~ −0.018) 낮다.

## 4. 50 에피소드 타깃 (V3, V4)

**V3: LORO-50.** 훈련 run 50 에피소드로 fit하고 held-out run 50 에피소드로 채점했다.

| pool | epoch | 최고 non-epoch | pairlogit_rank[all] | ridge_rank[all] | RA(epoch, L1) | L1 |
|---|---|---|---|---|---|---|
| all93 | 0.815 | pairlogit_z[small] +0.003 | −0.023 | −0.022 | −0.015 | −0.043 |
| eval33 | 0.854 | epoch을 넘는 것 없음. pairlogit_z[all+vlm] −0.015, ridge_rank[all+vlm] −0.011, RA(epoch,Nemotron) −0.032 | — | — | — | — |
| late93 | 0.333 | unitweight[small] +0.010 | −0.182 | −0.227 | −0.015 | −0.044 |

**V4: fit 없는 점수, 50 에피소드 타깃, seed bootstrap 95% CI.**

| pool | score | Δ vs epoch | run별 Δ | 95% CI | P(Δ>0) |
|---|---|---|---|---|---|
| all93 | RA(epoch, L1) | −0.015 | −0.002 / −0.014 / −0.030 | [−0.033, +0.010] | 0.14 |
| eval33 | RA(epoch, L1) | −0.052 | −0.043 / −0.042 / −0.071 | [−0.077, −0.011] | 0.01 |
| eval33 | RA(epoch, Nemotron) | −0.032 | 0 / 0 / −0.096 | [−0.059, +0.024] | 0.17 |
| late93 | Action L1 | −0.044 | +0.050 / −0.095 / −0.087 | [−0.136, +0.056] | 0.27 |
| late93 | RA(epoch, L1) | −0.015 | +0.054 / −0.082 / −0.016 | [−0.079, +0.042] | 0.33 |

**late93 타깃의 퇴화.** late can run에서 HB 값은 서로 다른 값이 2개뿐이다. can_seed2는 11개 중 10개, can_seed1은 21개 중 17개가 1.0이다. 그래서 plateau Spearman은 can run에서 1–4개 체크포인트가 결정한다. z2는 이 사실을 보고하지 않았다. 50 에피소드 타깃에서는 "3/3 승리"가 "1/3 승리"로 바뀐다.

## 5. bootstrap 진단 (V5)

**절대 ρ CI는 편향되어 있다.**
- 블록 bootstrap(all93): 점추정치가 자기 95% CI 밖에 있는 method가 26/36이다.
  - epoch: 0.825, CI [0.370, 0.816]
  - pairlogit_rank[all]: 0.834, CI [0.095, 0.785]
- 에피소드-only bootstrap(all93): 20/36이 CI 밖이다. epoch은 [0.637, 0.824]다. 복원추출한 타깃이 Spearman을 깎기 때문이다.
- 따라서 `z2_loro_hybrid_tables.txt`의 "held-out rho [CI]" 열은 CI로 읽으면 안 된다.

**블록 bootstrap의 −0.4 꼬리는 설계에서 나온다.**
- 비순환 moving block은 끝점을 덜 뽑는다. can_seed1에서 첫 체크포인트의 기대 복사 수는 0.18이고, 가운데 체크포인트는 1.21이다. can_seed2는 0.34 대 1.17, square는 0.28 대 1.11이다.
- z2 블록 bootstrap을 같은 RNG로 400회 다시 돌렸다(fold 1,200개).
  - 56%의 fold에서 훈련 run 중 하나가 처음 10% epoch을 통째로 잃는다.
  - 그런 fold에서 학습 모델의 평균 Δ는 −0.089 ~ −0.099다(Δ<−0.1 비율 21–27%).
  - 두 훈련 run이 모두 초반 체크포인트를 가진 fold에서는 −0.013 ~ −0.037이다(Δ<−0.1 비율 7–17%).
  - fit이 없는 RA(epoch, L1)는 두 경우 모두 +0.005로 영향을 받지 않는다.
- 즉 z2가 limitation에 적은 추측(초반 학습곡선이 빠진 훈련 run)이 맞다. 그 꼬리는 학습 모델의 성질이라기보다 재표본 설계의 산물이다.
- 그래도 결론은 바뀌지 않는다. 중복과 블록이 없는 무작위 분할(V2)에서도 P(Δ>0)는 0.09–0.47이다.

**Δ CI의 위치.** 블록 bootstrap에서 학습 모델의 Δ 중앙값은 점추정치보다 낮다(pairlogit_rank[all]: 점 +0.009, 중앙값 −0.033). 그런데 V2의 분할 평균(−0.016)은 bootstrap 중앙값 쪽에 더 가깝다. 이 bootstrap의 위치 편향은 대체로 "원 분할이 운이 좋았다"는 사실을 반영한다.

## 6. 주장과 산술 점검 (V6)

| z2 주장 | 확인 결과 |
|---|---|
| all93 epoch 0.825 (0.853/0.838/0.784), Regret@30 = top-1 = 0 | 맞음. 다만 can run은 41개 중 21개, 21개 중 12개가 HB 최대값(1.0)에 동률이라 사실상 자명하다. 정보가 있는 run은 square(31개 중 5개)뿐이다 |
| 최고 학습 조합 0.834 (+0.009), run CI [−0.005, +0.030], block [−0.41, +0.06], P 0.24 | 수치는 맞음. 원 분할이 98번째 백분위수이고, 분할 평균은 −0.016 |
| swap 시 모든 학습 모델 0.669–0.686 < epoch 0.690 | 맞음. 단, fit 없는 RA(epoch, replan)는 swap에서 0.697로 epoch보다 높다 |
| eval33 "+vlm 모델 0.815–0.833, VLM 없는 짝과 같거나 낮음" | **틀림.** 범위는 0.793–0.833이고, 8쌍 중 3쌍은 +vlm이 더 높다 |
| eval33 pairlogit_z[all] = epoch 순위 | ρ는 같다. 순서는 can_seed2에서 140/160이 뒤바뀐다(동률 타깃) |
| late93 L1 0.431, Δ +0.064, 3/3 | 수치는 맞음. 분할 95번째 백분위수이고, 50 에피소드 타깃에서는 −0.044, 1/3 |
| late93 학습 모델 0.05–0.17 | 0.042–0.174 (unitweight 0.025–0.145) |
| 고정 λ cosine 0.96–0.99, val_loss·support_drift 부호 불안정(0.34–0.84), λ가 그리드 양 끝을 오감 | 맞음 |
| 3/3 단측 부호검정 p = 0.125 | 맞음 |
| "run-level 95% CI" | 3개 fold의 [min, max]다. 모집단 중앙값을 덮을 확률은 0.75다. 95%라는 이름은 과장이다 |
| 필요 run 수 "8–17" | fold Δ 3개(sd 추정 자체가 매우 불안정)로 계산하면 검정력 50%에 9–18, 80%에 17–34 run이 필요하다(pairlogit_rank/ridge_rank[all]). 분할 평균 Δ가 음수라서 의미가 없다 |

## 7. 점검 항목별 정리

- **누출.** 없다.
  - fit, inner λ, unitweight 부호, select는 모두 `yA[tr]`만 쓴다. 채점은 `yB[te]`로 한다.
  - within-run 정규화는 결과를 쓰지 않는다(transductive라는 점은 z2가 밝혔다).
  - 사소한 점이 하나 있다. eval33 select_best 후보에 특권 정보인 `k10_sim`과 `RA(epoch,k10_sim)`이 들어 있다. eval seed 200000+의 실제 결과라서 HB 누출은 아니다. 또 어느 fold에서도 선택되지 않아 수치에는 영향이 없다.
- **held-out 처리.** 올바르다. 훈련 run 2개로 하는 inner LORO도 HA만 쓴다.
- **군집과 독립성.** 결과 단위는 run이고 run은 3개뿐이다. 이 점을 z2는 정직하게 밝혔다. 같은 env seed가 체크포인트끼리, run끼리 공유되므로 seed를 같이 재표본한 것은 맞다. 그런데 블록 bootstrap에는 위의 끝점 편향이 있다.
- **다중비교.** 풀마다 36–53개 점수를 보고 최고치를 강조했다. 결론이 영가설 쪽이라 해는 작다. 다만 late93의 L1처럼 "35개 중 최고 + 3/3"인 경우가 선택 효과 그 자체라는 것이 V2와 V4로 확인됐다.
- **judge 타당성.**
  - VLM K=10 특징은 eval seed에서 나온 독립 결과 측정치다. Cosmos는 FP 21/67, Nemotron은 FP 9/67이다.
  - 무작위 분할(V2)에서도, 50 에피소드 타깃(V3·V4)에서도 이득이 없다. "VLM 특징이 epoch을 넘지 못함"은 견고하다.
  - "짝보다 항상 낮다"는 표현만 틀렸다.
- **수치보다 강한 주장.** 위 1–6번(반박 목록)에 정리했다.

## 정정 헤드라인

학습 조합도, VLM K=10 특징도 이 풀에서 epoch을 넘지 못한다. 재현은 bit 단위로 같았고, 독립 구현으로도 확인했다. z2의 최고값 +0.009(pairlogit_rank[all], held-out ρ 0.834 vs 0.825)는 z2가 고른 HA/HB 분할에서만 나온 값이다. 그 분할은 무작위 25/25 분할 400개의 98번째 백분위수에 해당한다. 분할을 평균하면 모든 학습 조합이 epoch보다 약간 낮다(pairlogit_rank[all] Δ −0.016, P(Δ>0) = 0.09; small 모델 −0.004 ~ −0.005). 50 에피소드 전체로 LORO를 하면 최고치는 +0.003이다. plateau에서 Action L1이 보인 +0.064(3/3)도 분할 운이다. 50 에피소드 타깃에서는 −0.044(1/3)이다. eval33의 +vlm 모델은 0.793–0.833으로 epoch(0.852) 아래다. 다만 VLM 없는 짝보다 항상 낮은 것은 아니다(8쌍 중 3쌍은 더 높다). 절대 ρ의 bootstrap CI는 점추정치를 담지 못하므로 쓰면 안 된다. 블록 bootstrap의 −0.4 꼬리는 moving-block의 끝점 under-sampling에서 나온 산물이다.

## 남은 위험 (열린 상태)

- run 3개, task 2개다. 모든 run-level 추론은 원리적으로 검정력이 없다.
- 훈련이 짧고 거의 단조롭다. plateau 타깃은 거의 퇴화되어 있다(can late HB 값이 2종류뿐이다).
- 고정 HA/HB 분할은 y1 검증에서도 유리한 쪽으로 치우쳐 있었다. 다른 z/y 결과도 고정 분할 하나에 기대고 있다면 무작위 분할로 다시 확인해야 한다.

## 파일

- 스크립트:
  - `scripts/hackathon/explore/z2_loro_hybrid_verify.py`
  - `scripts/hackathon/explore/z2_loro_hybrid_verify_blocktail.py`
- 결과:
  - `results/hackathon/explore/z2_loro_hybrid_verification.json`
  - `results/hackathon/explore/z2_loro_hybrid_verification_blocktail.json`
- 재실행 사본과 로그: `data/outputs/hackathon_explore/z2_loro_hybrid_verify/`
  - `rerun_main.json`
  - `rerun_main.log`
  - `verify_run2.log`
