# z1_tier_triage: BAD / OK / GOOD 판정에 rollout이 몇 개 필요한가 (EXPLORATORY, 사후 분석)

> 사후 탐색 분석이다. 사전 등록된 검정이 아니다. CPU만 썼고 Nebius나 GPU 호출은 없다.
> 결과: `results/hackathon/explore/z1_tier_triage.json` (main, quality-dependent judge world)
> 민감도: `results/hackathon/explore/z1_tier_triage_iidworld.json` (constant-FPR judge world)
> 그림: `docs/hackathon/explore/z1_tier_triage.png`
> 스크립트: `scripts/hackathon/explore/z1_tier_triage.py`, `z1_tier_triage_tables.py`

## 한 줄 결론

이 pool에서 **tier 정확도 90%는 비싸다.** 원인은 judge가 아니라 checkpoint 분포다. 39%의 checkpoint가 50-ep 성공률 0.80–0.95, 즉 0.9 경계 근처에 몰려 있다.

- **perfect detector:** 고정 K plug-in으로 K≈30–36이 필요하다. sequential 정지를 쓰면 평균 약 25 rollout이다.
- **Nemotron:** K=50에서도 0.878이다. sequential은 약 35 rollout에서 0.87이다.
- **Cosmos3-Edge:** K≤50에서 90%에 도달하지 못한다. 최선은 sequential 약 43 rollout에서 0.88이다.
- **경계에서 먼 checkpoint(74/93)만 보면** perfect는 약 14, Nemotron은 약 18이면 된다. Cosmos는 25 안에 도달하지 못한다.
- **epoch/PLV prior의 효과는 작다.** BAD를 일찍 확정해서 평균 rollout을 약 5–6% 줄인다. 하지만 90% 도달 K는 바꾸지 못한다. PLV 지표는 서로 구분되지 않고 epoch과도 구분되지 않는다(y1과 같은 결론).

## 설정

### tier와 truth

- tier 정의: BAD는 성공률 < 0.5, OK는 0.5–0.9, GOOD는 > 0.9다. 정수 비교로 계산하므로 k/K = 0.9는 OK다.
- **World A (주 분석, 비모수).** 93 ckpt × 50개 training-time rollout(seeds 100000–100049)을 쓴다.
  - replicate마다 25/25 seed split을 무작위로 뽑는다. 모든 ckpt가 같은 split을 공유한다.
  - **truth는 held-out 25 episode의 tier**다.
  - K개 rollout은 나머지 25개에서 비복원으로 뽑는다(K = 1..25).
  - 고정 split(100000–24 / 100025–49)은 따로 보고한다.
- **World B (보조, 모수).** truth는 50-ep 성공률 p_c의 tier다. rollout은 Bernoulli(p_c)이고 K는 50까지다. 25-ep truth label의 잡음을 제거한 설정이다.
- **World R (검증).** eval 33 ckpt × 10개 실제 녹화 rollout(seeds 200000–9)에 대한 **실제 VLM 판정**을 쓴다.
  - truth tier는 50 training-time episode에서 가져온다. seed가 서로 겹치지 않는다.
  - 같은 규칙을 두 가지 입력에 적용한다. (i) 실제 판정, (ii) 같은 실제 truth 위에 simulated judge를 씌운 판정.

### judge 오류 모델

- 330-episode eval JSONL로 fit했다. 이 seed는 A/B와 겹치지 않는다.
- FNR_task는 Beta posterior로 둔다.
- **quality-dependent FPR:** logit FPR = a_task + b·(s − 0.5)다. y1 verifier의 bbq 모델이며, κ는 상한에 붙으므로 binomial로 두었다. 파라미터에 N(0, 5²) MAP와 Laplace 근사를 썼다.
  - Nemotron: b = 4.19 (SE 1.52). implied FPR(can)은 s=0.5에서 0.13, s=0.9에서 0.44, s=1에서 0.54다.
  - Cosmos: b = 3.23 (SE 0.94). implied FPR(square)은 s=0.5에서 0.48, s=0.9에서 0.77이다.
- 시뮬레이션에서 world는 replicate마다 파라미터를 한 번 draw한다. 의사결정자는 point fit 또는 파라미터 marginalised likelihood를 쓴다. 따라서 현실적인 model mismatch가 들어 있다.

### 판정 규칙

- **plug-in:** k/K를 tier로 바꾼다. VLM 판정을 그대로 믿는 방식이다.
- **Bayes:** 400-point grid 위의 posterior를 쓴다. 우도는 q(p) = p(1−FNR) + (1−p)·FPR(p)다. 판정은 argmax tier다.
  - prior 종류:
    - `flat`: p에 대해 uniform이다. GOOD에는 질량이 10%만 간다.
    - `tierflat`: tier마다 1/3이다.
    - `pop`: 다른 run의 분포를 쓴다.
    - `kde_<m>`: 지표 percentile 근방에서 다른 run의 Beta 혼합을 쓴다.
    - `prog_<m>`: 지표로 P(BAD)만 학습한다. 나머지 질량은 OK/GOOD에 반씩 준다.
  - 지표 prior는 모두 **leave-one-run-out**이다. 다른 두 run의 50-ep 결과로만 만든다. 기본값은 고정해 두었고 결과를 보고 튜닝하지 않았다.
- **sequential:** K ≥ 1에서 max posterior tier 확률이 0.9 이상이면 멈춘다(0.95도 보고). 도달하지 못하면 K_max에서 판정한다.
- **CI:** 각 replicate는 무작위 split(A), run-stratified checkpoint bootstrap, judge 파라미터 draw 1회, ckpt마다 8개 episode 순서로 구성된다. 250 replicate의 percentile 95% CI이며, MC 잡음이 포함되어 보수적이다.

## 결과

### 1. 고정 K 정확도 (plug-in, 95% CI)

| judge | A: K=5 | A: K=10 | A: K=25 | A: 90% 도달 K | B: K=10 | B: K=25 | B: K=50 | B: 90% 도달 K (점추정; bootstrap median [2.5, 97.5]) |
|---|---|---|---|---|---|---|---|---|
| perfect | 0.724 [0.65, 0.79] | 0.758 [0.65, 0.85] | 0.840 [0.75, 0.93] | >25 (replicate의 14%만 도달) | 0.814 [0.77, 0.85] | 0.876 [0.84, 0.91] | **0.925 [0.89, 0.95]** | **36; 30 [18, >50]** |
| Nemotron | 0.722 [0.66, 0.79] | 0.764 [0.67, 0.85] | 0.814 [0.72, 0.90] | >25 | 0.803 [0.73, 0.87] | 0.837 [0.75, 0.91] | 0.878 [0.78, 0.95] | >50 (38% 도달) |
| Cosmos3-Edge | 0.698 [0.62, 0.77] | 0.739 [0.64, 0.83] | 0.771 [0.67, 0.87] | >25 | 0.765 [0.67, 0.84] | 0.769 [0.67, 0.88] | 0.806 [0.67, 0.92] | >50 (8% 도달) |

- **World A의 천장.** 25-ep truth 자체가 잡음이다. 고정 split에서 두 절반의 tier 일치율이 0.849다. 그래서 어떤 judge도 K≤25로는 90%에 도달하지 못한다.
- 80% 도달 K는 다음과 같다.
  - World A: perfect 14, Nemotron 17, Cosmos는 >25.
  - World B: perfect 9, Nemotron 10, Cosmos 29.
- 고정 split(perfect K=25: 0.850)은 무작위 split 평균(0.840)과 비슷하다. 이 분석에서는 y1에서 발견된 split 편향이 크지 않다.
- **톱니 모양.** plug-in 곡선은 K=10, 20, 30, …에서 떨어진다. 0.9K가 정수가 되는 K에서는 GOOD 판정에 k > 0.9K, 즉 실패 0이 필요하기 때문이다. 경계 규칙 하나가 정확도를 수 포인트 바꾼다.

### 2. 경계에서 먼 checkpoint만 볼 때 (보고용 층화)

- 층화 기준: 50-ep 성공률이 [0.45, 0.55]와 [0.85, 0.95] 밖에 있는 ckpt. 74/93개(can 50, square 24)다. 판정 규칙에는 쓰지 않았다.
- World A에서 90% 도달 K의 bootstrap median:
  - perfect 14 (K=25에서 [0.88, 1.00])
  - Nemotron 18
  - Cosmos는 25 안에 도달하지 못한다(K=25에서 [0.74, 0.93]).
- **실무용 예산 진술:** 경계 근처가 아닌 checkpoint는 perfect detector로 약 15, Nemotron으로 약 20 rollout이면 90% tier가 나온다. 0.85–0.95 구간의 checkpoint는 50 rollout으로도 확정하기 어렵다.

### 3. Sequential 정지 (P(tier) ≥ 0.9)

| world | 규칙 | 정확도 | 평균 rollout | BAD / OK / GOOD 평균 rollout | 같은 평균 예산의 고정 K plug-in |
|---|---|---|---|---|---|
| B | perfect, flat prior | **0.916 [0.88, 0.95]** | **25.1 [22.9, 27.3]** | 8.9 / 25.1 / 28.8 | 0.88 |
| B | perfect, prog_epoch | 0.845 [0.80, 0.90] | 18.9 | 6.8 / 25.6 / 15.5 | 0.82 |
| B | Nemotron, judge-corrected(marg.), flat | 0.873 [0.78, 0.93] | 35.0 [31.8, 39.0] | 8.5 / 33.4 / 42.7 | – |
| B | Cosmos, judge-corrected(marg.), flat | 0.881 [0.79, 0.94] | 42.9 [39.3, 46.3] | 9.4 / 44.8 / 48.8 | – |
| A | perfect, flat | 0.836 [0.72, 0.93] | 19.0 | 6.9 / 18.3 / 21.9 | 0.83 |

- flat-prior sequential은 World B에서 보정이 잘 맞는다(0.9 목표에 0.916). 같은 평균 예산의 고정 K보다 약 +4pt 높다.
- BAD는 약 7–9 rollout이면 끝난다. 예산은 OK/GOOD 경계에 쓰인다.
- Cosmos는 같은 정확도에 perfect의 약 1.7배 rollout이 필요하다.
- `tierflat`과 `prog_*` prior를 쓰면 sequential이 더 빨리 멈춘다. 대신 OK→GOOD 오판이 늘어 정확도가 0.84 수준으로 내려간다. prior가 GOOD에 1/3 질량을 줘서 경계 근처 OK를 GOOD으로 끌어가기 때문이다.

### 4. 실패 사례: 무엇이 헷갈리나

- **혼동은 거의 전부 OK와 GOOD 사이에서 일어난다.** World B, K=25, plug-in, 행 정규화 기준:

| truth OK의 판정 | OK | GOOD |
|---|---|---|
| perfect | 0.78 | 0.21 |
| Nemotron | 0.67 | 0.32 |
| Cosmos | 0.54 | 0.46 |

- truth GOOD의 recall은 0.95–0.98이다. GOOD을 놓치는 일은 드물고, OK를 GOOD으로 부풀리는 쪽이 문제다.
  - near-plateau FP가 원인이다. square의 0.80–0.88 plateau가 judged rate 약 0.9로 올라간다.
- **Cosmos는 BAD→OK 오판도 많다.** BAD 행에서 0.23이다. square의 FPR이 품질이 낮아도 높기 때문이다(s=0.3에서 약 0.33).
- 성공률 구간별 정확도(A, K=10, perfect plug-in):

| 50-ep 구간 | 정확도 |
|---|---|
| [0, 0.4) | 0.92 |
| [0.6, 0.8) | 0.94 |
| [0.8, 0.9) (27 ckpt) | 0.72 |
| [0.9, 0.95) (9 ckpt) | 0.35 |
| [0.95, 1] | 0.82 |

- 최악의 checkpoint는 square e100–e300(0.82–0.88)과 can e35–e95(0.84–0.90), can_s2_e20(0.48)이다.

### 5. epoch/PLV prior가 rollout을 줄이는가

- **`prog_<m>` (P(BAD)만 학습):**
  - K=0 prior만으로의 정확도는 0.56이다(tierflat은 0.38).
  - K=1 정확도를 0.58에서 0.62로 올린다.
  - sequential 평균 rollout을 줄인다. perfect 기준 A는 15.9에서 15.1, B는 20.1에서 18.9이고, BAD의 rollout은 7.3에서 5.1이다.
  - **90% 도달 K와 K≥10 정확도는 바뀌지 않는다.** 차이는 CI 안에 있다.
  - epoch, replan_overlap_h16, val_action_l1, plv1_smoothness는 서로 구분되지 않는다(K=0에서 0.54–0.56). `val_loss_logged`는 0.43으로 더 나쁘다.
- **`kde_<m>` (다른 run의 성공률 분포를 통째로 이전):** 해롭다.
  - truth OK인 ckpt에 prior P(GOOD) 평균 0.45를 준다. truth GOOD인 ckpt에는 P(OK) 평균 0.44를 준다. 즉 반대 방향의 정보다.
  - 원인은 task마다 plateau 수준이 다르다는 점이다. can은 0.96–1.0, square는 0.80–0.88이다.
  - World B K=25에서 perfect 기준 0.831로, plug-in 0.876보다 낮다.
  - **교훈:** cheap metric은 "학습이 진행됐다"는 정보는 이전할 수 있다. 하지만 plateau가 0.9를 넘는지는 이전하지 못한다. tier 판정에서 가장 비싼 OK/GOOD 구분이 바로 그 정보다.

### 6. judge 보정 (Bayes correction)

- 고정 K에서 quality-aware 보정은 plug-in을 이기지 못한다. 예시는 다음과 같다.
  - Nemotron, A, K=25: plug-in 0.814, 보정(tierflat) 0.774.
  - 파라미터 marginalised + flat prior는 큰 K에서만 따라붙는다. Cosmos, B, K=50에서 0.890으로 plug-in 0.806보다 높다. 대신 K≤10에서는 0.45–0.51로 낮다.
- **constant-FPR(iid) 보정은 Cosmos에서 해롭다.** B, K=50에서 0.698이다. 낮은 품질에서는 과보정하고 plateau에서는 과소보정하기 때문이다.
- **prior 선택이 보정보다 크게 작용한다.**
  - `flat`은 작은 K에서 GOOD을 거의 부르지 않는다. K≤5에서 0.39–0.50이다.
  - `tierflat`은 경계 근처 OK를 GOOD으로 부른다. OK recall이 K=10에서 0.43–0.53이다.
  - 실무 권장: 고정 K에서는 plug-in을 쓴다. 정지 규칙에는 flat-prior posterior를 쓴다.

### 7. simulated judge 검증 (33 eval checkpoint, leave-one-checkpoint-out)

| judge | 모델 | judged-rate MAE | 90% PI coverage | LOO log score (합) | FP 예측/실제, s_eval ≥ 0.85 | FP 예측/실제, s_eval < 0.5 |
|---|---|---|---|---|---|---|
| Nemotron | quality-dependent | **0.040** | 0.97 | **−22.5** | 3.8 / 4 | 1.6 / 1 |
| Nemotron | constant | 0.055 | 0.97 | −29.5 | 1.3 / 4 | 7.4 / 1 |
| Cosmos | quality-dependent | **0.049** | 1.00 | **−26.5** | 5.4 / 7 | 5.4 / 6 |
| Cosmos | constant | 0.072 | 0.91 | −34.5 | 2.6 / 7 | 14.1 / 6 |
| Nemotron pad201 | quality-dependent | 0.037 | 1.00 | −21.0 | 4.8 / 5 | 1.8 / 1 |

- **per-checkpoint judged rate 수준:** quality-dependent 모델이 일관되게 낫다. log score로 7–10 nat 차이다. constant 모델은 plateau FP를 과소예측하고 실패 많은 ckpt의 FP를 과대예측한다.
- **training-time 50-ep 품질만으로 judged rate를 예측하는 end-to-end 오차:** MAE 0.046이다. eval 10-ep truth 자체와 50-ep 사이의 MAE가 0.068이므로, 이 오차는 seed 차이가 대부분이다.
- **tier 수준에서는 맞지 않는다(그림 E).**
  - 실제 판정 plug-in의 K=10 정확도는 Nemotron 0.939 [0.85, 1.00], Cosmos 0.909 [0.82, 1.00]이다.
  - 같은 truth 위의 simulated 판정은 0.756, 0.764다. truth_success(perfect)는 0.758이다.
  - 실제 − sim 차이는 Nemotron **+0.18 [0.10, 0.28]**, Cosmos +0.15 [0.06, 0.23]이다.
- **원인:**
  - can의 50-ep GOOD ckpt 네 개(can_s1_e60/e80, can_s2_e100/e180)가 eval에서 실패를 1개씩 냈다. 실제 VLM은 그 실패를 모두 성공으로 판정했다. 실제 FP 4/4이고, 모델 예측은 각각 약 0.44다.
  - 그 결과 9/10이 10/10이 되었다. K=10에서 GOOD 판정에는 10/10이 필요하므로, plug-in이 우연히 고쳐졌다.
  - square_s1_e60(eval truth 10/10, 50-ep 0.68)에서는 Nemotron의 FN 2개가 우연히 OK로 맞혔다.
- **해석:** 33개 표본에서 실제 VLM의 near-plateau FP는 FPR(s=0.9)의 point fit보다 더 극단적이다(can 4/6). 그 방향이 can GOOD에 유리하게 작용했다.
  - 이 이득은 plateau가 0.9보다 아래인 task(square)에서는 오판으로 뒤집힌다.
  - 따라서 시뮬레이터는 judged rate 분포는 잘 재현하지만, tier 정확도를 이 표본의 실제값보다 **낮게** 본다.
  - "VLM이 perfect detector보다 tier를 잘 맞춘다"는 결론은 내리면 안 된다(33 ckpt, 10 ep, 경계 규칙에 의존).

### 8. 민감도: constant-FPR world

- world judge를 iid로 바꾸면 Nemotron plug-in의 B, K=50 정확도가 0.915가 되고 K=37에서 90%에 도달한다. Cosmos는 0.882로 여전히 미달이다.
- quality-dependent world가 더 비관적이다. 차이는 plateau FP 때문이다.

## 해석: 실로봇 평가 예산으로 옮기면

1. **tier 판정의 비용은 checkpoint가 경계에 얼마나 가까운지가 결정한다.**
   - 경계에서 먼 ckpt는 perfect detector나 Nemotron으로 15–20 rollout이면 된다.
   - 0.9 근처 plateau는 50 rollout으로도 판정이 흔들린다.
   - 실무에서는 고정 K보다 sequential 정지가 낫다. 이 pool에서 BAD는 약 7–9, OK/GOOD 경계는 25–49 rollout이 들었다.
2. **Nemotron은 perfect detector에 가깝다.**
   - 정확도 손실은 K=25에서 0.02–0.04다.
   - 90%에 가까운 정확도를 위한 sequential 예산은 perfect의 약 1.4배다(35 대 25).
3. **Cosmos3-Edge는 OK/GOOD 판정에 부적합하다.**
   - 고품질 checkpoint의 실패를 성공으로 부르는 경향(FPR 0.6–0.8)이 강하다. 그래서 truth OK의 46%가 GOOD으로 올라간다(K=25).
   - BAD와 not-BAD를 가르는 1차 triage로만 쓰는 것이 안전하다.
4. **cheap metric(epoch/PLV)은 BAD를 거르는 데만 쓸모가 있다.**
   - 평균 rollout을 약 5–6% 줄인다.
   - OK/GOOD 구분에는 도움이 되지 않는다. task 간에 plateau 수준이 이전되지 않기 때문이다.
   - 여기서도 PLV는 epoch과 구분되지 않는다.

## 한계

- **표본이 작다.** run 3개, task 2개, ckpt 93개다. GOOD은 전부 can이고 square에는 GOOD이 없다. 정확도는 tier 구성에 크게 좌우되며, GOOD 53%, OK 38%, BAD 10%다(A).
- **World A의 truth는 25-ep 추정치다.** truth label 잡음 때문에 천장이 생긴다. World B는 50-ep 성공률을 참값으로 둔 plug-in world model이고, 경계(0.90, 0.48) 위에 있는 ckpt는 원리적으로 맞히기 어렵다.
- **judge 오류 모델의 근거가 얇다.** 33 eval ckpt, 67개 실패로 fit했다.
  - 고품질 FPR은 실패 9개(s_eval ≥ 0.85)에서 외삽한 값이다.
  - 분석 대상 ckpt와 judge fit ckpt가 겹친다. seed는 겹치지 않지만 같은 checkpoint다.
  - square는 run이 하나뿐이라 task별 파라미터를 run 단위로 교차검증할 수 없다.
  - 실패 episode 사이의 상관(κ)은 모형화하지 않았다. y1에서 κ는 상한에 붙었다.
- **world R(실제 판정)은 ckpt당 10 rollout이라 K ≤ 10이다.** eval seed와 training seed는 성공률이 다르다(MAE 0.068).
- **prior 설정(bandwidth 0.1, 80/20·50/50 혼합, prog의 OK/GOOD 50/50 분할)은 사전에 정한 기본값이다.** 다른 설정은 탐색하지 않았다.
- 시뮬레이션 CI에는 MC 잡음이 포함되어 있다. 다중 비교(규칙 약 46개 × K)는 보정하지 않았다.
- **이미 알려진 미해결 위험을 다시 적는다.** y1 verifier가 지적했듯이 `*_h10`과 `*_h16` 열이 93/93행에서 같다. 따라서 replan_overlap_h16을 H=16 고유 지표로 해석하면 안 된다.

## 재실행

```
cd /path/to/repo
PY="env PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python"
$PY scripts/hackathon/explore/z1_tier_triage.py --reps 250 --orders 8                     # 153 s, exit 0
$PY scripts/hackathon/explore/z1_tier_triage.py --reps 100 --orders 8 --world-judge iid \
    --out results/hackathon/explore/z1_tier_triage_iidworld.json --fig ""                # sensitivity, exit 0
$PY scripts/hackathon/explore/z1_tier_triage_tables.py > data/outputs/hackathon_explore/z1_tier_triage/tables_main.md
```
