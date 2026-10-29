# y1_plv_prior_budget 적대적 검증 (EXPLORATORY, 사후 분석)

**판정: partially_confirmed (부분 확인).**

- **확인된 것.** 핵심 음성 결과는 유지된다. 이 3개 run에서 어떤 PLV prior도 무료 latest-epoch prior를 이기지 못한다.
  - 원 스크립트를 다시 돌린 결과가 bit 단위로 같았다. main과 splits 모두 max abs diff 0.0이다.
  - y1/x1 코드를 쓰지 않은 독립 구현(로그 parser, prior, uniform·TTTS 시뮬레이터, judge fit)도 MC 오차 안에서 같은 값을 냈다.
  - "beats both"는 4개 level × 336 cell에서 0이다. **epoch 하나만 상대로 해도 이기는 cell이 0**이다.
  - `val_loss_logged`는 해로운 prior다. full/trunc_2/trunc_3에서 28/28 cell 모두 epoch보다 유의하게 나쁘다.
  - 누출은 찾지 못했다. prior는 outcome-free 열만 쓴다. judge는 eval seed(200000+)로 fit했고, regret은 held-out 절반으로만 채점한다.
- **반박되거나 약해진 것.**
  1. **고정 25/25 split이 유리한 쪽에 치우쳐 있다.** held-out 절반(100025–100049)이 더 쉽다. 그래서 절대 regret 수치(reference 0.033, 1N 0.046→0.020, epoch top-1 0.000)는 무작위 split 기준의 약 절반이다. y1은 CI의 상향 이동을 "seed 중복" 탓으로만 설명했는데, 이는 부분적으로 틀렸다.
  2. **헤드라인 "적어도 2–8x(TTTS, perfect judge)"는 수치보다 강하다.** "적어도"는 grid floor 절단에서 나온 말이지 통계적 하한이 아니다. full/TTTS/perfect에서 epoch prior가 no-prior보다 먼저 도달하는 replicate는 59%뿐이다. thr 0.02에서는 median 비율이 1x다. 같은 조건의 regret 차이 CI도 0을 포함한다.
  3. "2–48x"의 분모는 모든 prior 팔에서 0.25N grid floor다. threshold(0.01/0.02)는 y1이 스스로 정한 노이즈 띠(0.02–0.03) 안에 있다. "최대 24x/48x"는 28개 cell 중 최댓값이고, re-split에서는 64x까지 나온다. 크기가 불안정하다.
- **정정 헤드라인**은 맨 아래에 있다.

## 질문

y1 보고서(`docs/hackathon/explore/y1_plv_prior_budget.md`), 결과(`results/hackathon/explore/y1_plv_prior_budget{,_splits}.json`), 스크립트(`y1_plv_prior_budget.py`, `_tables.py`)를 반박하려고 시도했다. 점검 항목은 다음과 같다.

- 누출
- held-out 처리
- clustering/독립성
- 다중비교
- 사후 부분집합 선택
- 분모
- 수치보다 강한 주장
- 산술

## 방법 (모두 CPU, Nebius/GPU 없음)

1. **재실행.** y1 main(seed bootstrap)과 `--resample split`을 scratch로 다시 돌렸다. 원 JSON과 row와 paired 값을 모두 비교했다.
2. **독립 재구현.** `scripts/hackathon/explore/y1_plv_prior_budget_verify.py`. y1/x1을 import하지 않는다.
   - `logs.json.txt`를 직접 파싱한다.
   - top-30% prior는 `baseline_metrics.csv`의 outcome-free 열로 직접 만들었다.
   - uniform과 TTTS(β=0.5, Beta(1,1), arm당 25 episode cap)는 직접 구현했고 perfect judge만 쓴다. 원 split에서 R=400이다.
   - **run별** paired Δ를 계산했다. seed bootstrap 200 × 20 reps와 무작위 re-split 200 × 20 reps 두 가지다. LORO(한 run 제외)도 함께 계산했다.
   - beta-binomial judge 모델(bb, bbq)을 연속 κ로 다시 fit했다.
   - y1 JSON의 주장 수치를 기계적으로 점검했다. 대상은 beats-both 계수, epoch 대비 Δ 범위, 도달 예산 비율, CI 밖 점추정 개수다.
3. **split 대표성 점검.** `scripts/hackathon/explore/y1_plv_prior_budget_verify_split.py`.
   - 두 절반의 성공률을 비교했다.
   - reference(perfect judge, no prior, 25N → pool-best)의 regret을 무작위 25/25 split 5000개 분포와 비교했다.
   - 0-rollout latest-epoch pick과 pool-best를 무작위 split 2000개에서 paired로 비교했다.
   - 각 PLV 지표와 epoch의 Spearman을 계산했다.

### 명령과 exit code

```
cd /path/to/repo
PY="env PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python"
S=data/outputs/hackathon_explore/y1_plv_prior_budget_verification
$PY scripts/hackathon/explore/y1_plv_prior_budget.py --workers 10 --out $S/rerun_main.json --fig ""                       # exit 0 (167 s)
$PY scripts/hackathon/explore/y1_plv_prior_budget.py --workers 10 --resample split --out $S/rerun_splits.json --fig ""    # exit 0 (184 s)
$PY scripts/hackathon/explore/y1_plv_prior_budget_tables.py $S/rerun_main.json > $S/rerun_tables_main.md                  # exit 0
$PY scripts/hackathon/explore/y1_plv_prior_budget_tables.py --headline $S/rerun_main.json > $S/rerun_headline_main.md     # exit 0
VERIFY_OUT=$S/smoke.json VERIFY_NB=4 VERIFY_R0=40 $PY scripts/hackathon/explore/y1_plv_prior_budget_verify.py             # exit 0 (smoke)
$PY scripts/hackathon/explore/y1_plv_prior_budget_verify.py          # exit 0 (139 s) -> results/hackathon/explore/y1_plv_prior_budget_verification.json
$PY scripts/hackathon/explore/y1_plv_prior_budget_verify_split.py    # exit 0        -> results/hackathon/explore/y1_plv_prior_budget_verification_split.json
```

## 결과

### 1. 재현성과 독립 재구현: 확인

- **재실행.** 원 JSON과 같았다. rows(mean, CI)와 paired(avg point, CI) 모두 max abs diff 0.0이고, judge fit도 같다. main과 splits 둘 다 그렇다.
- **데이터 정렬.** CSV `success`와 50-episode 로그 평균의 차이는 0.0이다. epoch도 일치한다.
- **독립 시뮬레이터(perfect judge, 원 split).** 4 level × 6 prior × 2 전략 = 48개 곡선이다. 대부분 MC SE의 1–2배 안에서 일치했다.
  - grid 전체 최대 차이는 0.013이다. 분산이 큰 random/none 팔에서 나왔다.
  - 대표 cell은 아래 표와 같다.

| level | prior | 전략 | 1N 독립 / y1 | avg 0.5–8N 독립 / y1 | 25N 독립 / y1 |
|---|---|---|---|---|---|
| full | none | TTTS | 0.048 / 0.046 | 0.042 / 0.042 | 0.033 / 0.033 |
| full | epoch | TTTS | 0.020 / 0.020 | 0.023 / 0.023 | 0.027 / 0.027 |
| full | replan_overlap_h16 | TTTS | 0.019 / 0.019 | 0.023 / 0.023 | 0.027 / 0.027 |
| full | val_loss_logged | TTTS | 0.103 / 0.103 | 0.105 / 0.105 | 0.106 / 0.107 |
| trunc_1 | epoch | TTTS | 0.049 / 0.048 | 0.052 / 0.051 | 0.040 / 0.040 |
| trunc_1 | val_action_l1 | TTTS | 0.040 / 0.040 | 0.035 / 0.035 | 0.014 / 0.014 |

### 2. PLV ≈ epoch (음성 결과): 확인. 단, 증거의 폭은 과대 표현됨

| level | cell | beats both (avg / any mult) | **epoch 단독 상대로 이김** | epoch보다 유의하게 나쁜 prior | 비-vll prior의 Δ vs epoch avg CI 외곽 |
|---|---|---|---|---|---|
| full | 336 | 0 / 0 | **0** | val_loss_logged (28/28) | [−0.012, +0.019] |
| trunc_1 | 336 | 0 / 0 | **0** | plv1_smoothness (SH 5 cell, 일부 mult) | [−0.068, +0.090] |
| trunc_2 | 336 | 0 / 0 | **0** | val_loss_logged | [−0.047, +0.043] |
| trunc_3 | 336 | 0 / 0 | **0** | val_loss_logged | [−0.023, +0.044] |

re-split JSON도 같다. epoch을 이기는 cell은 0이다. epoch보다 나쁜 prior로는 trunc_1에서 open_loop, plv1, val_action_l1이, trunc_2에서 val_action_l1이 추가로 나왔다.

- **부분집합 겹침 "85–100%".** 확인했다. run별 최소값은 84.6%다(can_seed1의 val_action_l1과 plv1, 11/13). run 합산은 86.7–100%다.
- **"336 cell"은 독립 증거 336개가 아니다.**
  - full에서 PLV prior 12개 중 5개는 3개 run 모두에서 top-30% 집합이 epoch과 같다. replan_overlap_h16, boundary_jump_h16, epoch+replan, epoch+open_loop, epoch+boundary가 그렇다. 이 cell들의 Δ는 MC 잡음을 빼면 0일 수밖에 없다.
  - replan_overlap_h16은 can_seed1에서 epoch과 순위가 같다(Spearman 1.000). plv1_smoothness는 square에서 epoch과 순위가 같다(1.000).
  - replan_h16과 boundary_h16은 run 안 Spearman이 0.99다.
  - 상류 데이터에서 `*_h10` 열과 `*_h16` 열이 93/93행에서 같은 값이다. y1의 오류는 아니지만, H=10/H=16을 구분한다는 Experiment 1 서술은 점검이 필요하다.
  - 실질적으로 서로 다른 prior는 3–4개 정도다.
- **PLV 지표의 epoch 상관.** 방향을 맞춘 Spearman은 0.86–1.00이다(val_loss_logged는 −0.62~−0.74). epoch을 통제한 rank-partial 상관은 −0.20~+0.19다(기술 통계이며 선택에 쓰지 않았다). "PLV는 epoch을 통해 success와 상관된다"는 해석을 지지한다.
- **run별 paired Δ(독립 구현, TTTS, perfect, avg 0.5–8N).**

| level | 비교 | can_seed1 | can_seed2 | square_seed1 | 3-run 평균 (bootstrap) | 3-run 평균 (re-split, Δ<0 비율) |
|---|---|---|---|---|---|---|
| full | replan_h16 − epoch | [−0.005, 0.005] | 0 (집합 동일) | [−0.010, 0.013] | [−0.004, 0.005] | [−0.004, 0.004], 0.49 |
| full | val_action_l1 − epoch | [−0.008, 0.050] | 0 | [−0.016, 0.019] | [−0.005, 0.021] | [−0.011, 0.019], 0.49 |
| trunc_1 | val_action_l1 − epoch | [−0.053, 0.091] | [−0.011, 0.012] | [−0.224, 0.095] | [−0.074, 0.031] | [−0.009, 0.042], **0.08** |

- **trunc_1 val_action_l1의 "split artifact" 주장.** 확인했다.
  - 독립 re-split에서도 val_action_l1이 epoch보다 나은 split은 8%다. y1은 8.5%를 보고했다. 3-run 평균의 median은 +0.016으로, 오히려 epoch보다 나쁜 쪽이다.
  - 메커니즘을 확인했다. square trunc_1에서 epoch 부분집합은 {70, 80, 90}이다. 이 중 e80이 pool 22/25로 pool 최고지만 held-out은 18/25로 가장 낮다. 50 episode 기준으로는 0.80 대 e70의 0.78이라 사실상 같다.
  - val_action_l1 부분집합 {60, 70, 90}은 e80을 빼므로, 원 split에서만 이득을 본다.
- **검정력.** trunc_1에서 PLV−epoch CI 폭은 ±0.05–0.07이다. 이 폭은 epoch−none 효과(약 −0.03~−0.05)와 같은 크기다. 따라서 trunc_1에서 "PLV adds nothing"은 동등성을 보인 것이 아니라 차이가 검출되지 않았다는 뜻이다.
- **문서에 빠진 불리한 결과.**
  - trunc_1에서 plv1_smoothness가 SH 5개 cell에서 epoch보다 유의하게 나쁘다.
  - trunc_2에서 val_action_l1이 thr 0.01에 도달하려면 모든 judge·전략에 걸쳐 1–6N이 필요하다. 문서는 "최대 4N"이라고 했다.
  - 다중비교 때문에 이것들도 단독으로 인용하면 안 된다.

### 3. prior vs no prior (예산 절감): 방향은 확인, 크기와 "적어도"는 과장

**(a) 고정 split의 대표성.** 가장 큰 정정 사항이다.

| 항목 | 원 split (pool 100000–24 / held-out 100025–49) | 무작위 25/25 split |
|---|---|---|
| 후반부 checkpoint 성공률, pool 절반 vs held-out 절반 | can1 0.962 / 0.992, can2 0.982 / 0.996, square 0.812 / 0.865 | (held-out 절반이 일관되게 더 쉽다) |
| reference regret (perfect, no prior, 25N pool-best) | **0.033** | mean 0.065, median 0.062, 95% [0.032, 0.107]. **원 split은 하위 3.3번째 백분위**다. 두 절반을 바꾸면 0.081 |
| y1 re-split JSON, full level | 1369/5460 row×mult의 점추정이 re-split 2.5% 하한보다 **아래**에 있다. 위에 있는 것은 0개 | |
| 0-rollout latest-epoch pick regret | 0.000 (3개 run 모두 0) | mean 0.027 [0, 0.067]. square의 마지막 epoch가 regret 0인 split은 23%뿐 |
| latest-epoch − pool-best(25N), paired | −0.033 | **−0.038 [−0.088, −0.001]**. latest-epoch가 나은 split 98.5% |

- y1은 seed-bootstrap CI가 점추정보다 위에 있는 이유를 "seed 중복이 pool을 약하게 하고 H에 잡음을 더하기 때문"이라고 했다.
- 그런데 중복이 없는 re-split에서도 같은 이동이 나타난다. 원래 split 자체가 낮은 regret 쪽 이상치다.
- 점추정이 자기 bootstrap CI 밖에 있는 경우도 있다(예: cosmos_bb epoch 1N 0.015, CI [0.016, 0.060]). 이것도 같은 증상이다.
- **정성적 결론은 re-split에서도 유지된다.** "0-rollout epoch pick이 25N full-pool pick보다 낫다"도 98.5%의 split에서 성립한다.
- 하지만 **절대 수치는 약 2배 낙관적이다.** 0.046 → 0.020, 0.000 < 0.033 같은 값을 인용하면 안 된다.

**(b) "적어도 2–8x(TTTS, perfect judge)".**

| full / TTTS / perfect | thr 0.0 | thr 0.01 (헤드라인) | thr 0.02 |
|---|---|---|---|
| 점추정 도달 예산 none / epoch | 20N / 0.25N | 2N / 0.25N | 0.5N / 0.25N |
| bootstrap 비율 median [p10, p90] | 4 [1, 48] | **2 [1, 24]** | **1 [1, 8]** |
| epoch가 strictly 먼저 도달한 replicate 비율 | 0.68 | **0.59** | 0.47 |
| re-split 비율 median, 먼저 도달한 비율 | | 2 [1, 12], 0.56 | 1 [1, 4], 0.44 |

- 점추정 2N은 경계선에 걸려 있다. 1.5N에서 none의 regret은 0.0439이고 목표는 0.0434다. 차이 0.0005로 아깝게 미달했고, 그 결과 점추정 비율이 8x가 됐다.
- 같은 조건의 regret 차이도 유의하지 않다.
  - epoch − none, TTTS, perfect, full: 1N은 −0.026 [−0.069, 0.004], avg는 −0.019 [−0.054, 0.0095]다.
  - full에서 CI가 0을 포함하는 judge×전략 cell은 28개 중 2개뿐이다(perfect TTTS와 perfect Thompson). 헤드라인이 인용한 바로 그 조건이다.
  - 독립 run별 bootstrap에서 square는 [−0.110, +0.063]이다.
  - LORO에서 can_seed1이나 can_seed2를 빼면 CI가 0을 포함한다. square를 빼야 [−0.035, −0.008]로 유의해진다.
  - re-split 3-run 평균은 [−0.040, −0.004]로 유의하다. 결론적으로 **perfect judge·full history에서는 경계선**이다.
- noisy judge나 truncated history에서는 효과가 견고하다.
  - trunc_1, cosmos_bb, 1N: epoch − none −0.098 [−0.195, −0.024], epoch − random −0.109 [−0.198, −0.021]이다. 둘 다 재현했다.
  - re-split에서는 trunc_1과 trunc_2의 28/28 cell에서 epoch − none CI가 0을 포함하지 않는다.

**(c) 비율의 크기에 대하여.**

- 모든 좋은 prior는 0.25N(grid floor)에서 이미 목표에 도달한다. 그래서 "none/epoch 비율"은 사실상 "none의 도달 예산 ÷ 0.25"다.
  - grid가 0.1N에서 시작했다면 모든 비율이 2.5배 커진다. 0N 점을 넣으면 무한대가 된다.
  - 의미가 있는 양은 비율이 아니라 none의 도달 예산(2–>25N)이다.
- threshold 0.01과 0.02는 y1이 "측정 잡음"이라고 부른 0.02–0.03보다 작다. held-out 1 episode는 run당 0.04이고, 3-run 평균에서는 약 0.013이다. 헤드라인 비율은 결국 잡음 띠 안에서의 도달 여부로 정해진다.
- "최대 24x(TTTS)/48x(uniform)"는 judge 7개 × level 4개, 즉 28개 cell 중 최댓값이다. re-split에서는 uniform이 64x까지 나온다. 이 크기는 안정적인 추정치가 아니다.

### 4. Judge 오류 모델: 확인 (해석 보완)

| judge | bb κ MLE, y1 grid / 독립 연속 | LR vs binomial (경계 보정 p) | bbq slope b (y1 SE) | bbq κ | quality slope LR (binomial) |
|---|---|---|---|---|---|
| Nemotron v2 | 1.6 / 1.42 | 4.66 / 4.67 (p≈0.015) | 4.77 (1.81) | → 상한 (1e4) | 12.9 |
| Cosmos3-Edge v6 | 3.5 / 4.0 | 3.29 / 3.31 (p≈0.034) | 3.35 (0.97) | → 상한 | 15.4 |

- "FP가 checkpoint 안에서 군집한다"는 표현은 정확하지 않다. 실제로는 checkpoint 사이의 이질성이다.
  - 실패가 2개 이하인 checkpoint에서 FP는 Nemotron 5/13, Cosmos 9/13이다.
  - 실패가 5개 이상인 checkpoint(주로 epoch 0)에서는 Nemotron 2/41, Cosmos 8/41이다.
  - y1도 "품질로 대부분 설명된다"고 썼으므로 결론과는 맞는다.
- checkpoint가 3개 run 안에 nested되어 있다는 점은 모델링되지 않았다.
- s=0.9의 FPR(0.40–0.78)은 실패가 1–2개뿐인 소수 checkpoint에서 외삽한 값이다.

### 5. 기타 주장

- "`val_loss_logged`의 top-30%는 epoch 20–110": 확인(run별 20–95, 20–80, 20–110).
- "Δ vs random +0.065 [0.005, 0.121]": 확인(TTTS, perfect, avg). 모든 judge에서 +0.060~+0.065이고 CI는 0을 포함하지 않는다.
- "can_seed1 41개 중 21개가 held-out 1.0에서 동률": 확인.
- "square epoch prior 0.045(0.25N) → 0.080(25N)": 확인. full square의 pool 최고는 e260(23/25)인데 held-out은 21/25다.
- "PLV vs epoch의 crossing median 비율 1 [1, 1], 먼저 도달 1–7%": 확인(full, 최대 6.5%/7%).
- "post-hoc best PLV per judge Δ vs epoch ±0.0003, CI ±0.006–0.010": 확인(full).

## 해석

1. **유지되는 것.** 이 pool에서 PLV top-30% prior는 latest-epoch prior와 거의 같은 집합이다. 어떤 셀에서도 epoch을 이기지 못한다. 따라서 "PLV가 rollout을 아낀다"는 epoch과 구분되는 주장으로 내세울 수 없다. `val_loss_logged`는 prior로 쓰면 안 된다. 이 두 결론은 재실행, 독립 재구현, re-split 모두에서 견고하다.
2. **약해지는 것.** "screen first"가 no-prior보다 낫다는 주장은 noisy judge와 truncated history에서는 견고하다. perfect judge·full history에서는 경계선이다(seed bootstrap CI가 0 포함, run 하나를 빼면 무너짐). 헤드라인의 "적어도 2–8x"는 grid 절단에서 나온 하한이다. 해당 조건에서 41%의 replicate는 절감이 없다(비율 1).
3. **수치 인용 주의.** 고정 split의 held-out 절반이 더 쉬워서 모든 절대 regret이 낮게 나왔다(reference는 무작위 split의 하위 3% 수준). 제출물에서는 상대 비교(paired Δ, re-split 비율)만 인용하는 것이 안전하다. 절대값을 쓰려면 re-split 평균을 함께 제시해야 한다.

## 한계 (이 검증의 한계)

- 독립 시뮬레이터는 perfect judge의 uniform과 TTTS만 다룬다. noisy judge 결과는 재실행 일치와 JSON 점검으로만 검증했다.
- run별 bootstrap은 replicate당 20 reps라서 MC 잡음이 구간에 섞여 있다. run이 3개뿐이므로 LORO는 민감도 점검일 뿐이다.
- 상관 기술 통계(epoch 통제 partial 상관)는 50 episode 결과 전체를 썼다. 선택이나 평가에는 쓰지 않았다.
- `*_h10 == *_h16` 동일성은 상류(Experiment 1) 관찰이다. 원인은 조사하지 않았다.
- 모두 사후 탐색적 분석이다. 사전 등록된 검정이 아니다.

## 정정 헤드라인

EXPLORATORY / post-hoc. 결과는 bit 단위 재현과 독립 재구현으로 확인했다.

- 이 3개의 거의 단조로운 run에서 무료 latest-epoch top-30% prior는 어떤 PLV prior만큼 좋다. PLV prior가 epoch을 이기는 cell은 level마다 336개 중 0개이고, PLV top-30% 집합의 84.6–100%가 epoch 집합과 같다. `val_loss_logged`는 해롭다.
- prior로 먼저 거르는 것은 noisy VLM judge와 truncated history에서 no-prior보다 분명히 낫다. perfect judge·full history에서는 경계선이다. TTTS 1N Δ는 −0.026 [−0.069, 0.004]이고, epoch가 먼저 도달하는 replicate는 59%이며, thr 0.02에서 median 비율은 1x다.
- "2–48x" 비율은 0.25N grid floor와 노이즈 띠 안의 threshold가 정한 값이다.
- 절대 regret은 유리한 고정 split(reference가 무작위 split의 하위 3%)에서 나와 약 2배 낙관적이다.

## 파일

- 검증 스크립트: `scripts/hackathon/explore/y1_plv_prior_budget_verify.py`, `scripts/hackathon/explore/y1_plv_prior_budget_verify_split.py`
- 결과: `results/hackathon/explore/y1_plv_prior_budget_verification.json`, `results/hackathon/explore/y1_plv_prior_budget_verification_split.json`
- scratch(재실행 JSON, 로그, 표): `data/outputs/hackathon_explore/y1_plv_prior_budget_verification/`
