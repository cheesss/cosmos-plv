# z1_tier_triage 검증 (적대적 검증, EXPLORATORY / 사후 분석)

> **판정: partially_confirmed.** 숫자는 재현된다. 다만 headline의 표현 몇 개가 숫자보다 강하고, 일부 수치는 MC 잡음이나 tie-break 때문에 생긴 값이다.
> CPU만 썼고 Nebius나 GPU 호출은 없다. 기존 파일은 고치지 않았다.
> - 스크립트: `scripts/hackathon/explore/z1_tier_triage_verify.py`, `z1_tier_triage_verify_frontier.py`
> - 결과: `results/hackathon/explore/z1_tier_triage_verification.json`, `z1_tier_triage_verification_frontier.json`
> - 재실행 산출물: `data/outputs/hackathon_explore/z1_tier_triage_verification/`

## 0. 사용자 질문: Nemotron·Cosmos 출력이 저장되어 있나

**저장되어 있다. 직접 확인했다.** 원본은 `data/outputs/hackathon_cosmos/`에 있고, 같은 파일의 사본이 `results/hackathon/raw_model_outputs/`에 있다. 사본 7개는 원본과 바이트 단위로 같다(`cmp`). 각 파일 설명은 `results/hackathon/raw_model_outputs/README.md`에 있다.

| 파일 | judge | 행 수 | 비고 |
|---|---|---|---|
| `eval_nebius_v2.jsonl` | Nemotron-3.5-Super-VL (Nebius), 영상 전체 | 330 (33 ckpt × 10) | raw_text 330/330, request_id 330/330 |
| `eval_nebius_v2_pad201.jsonl` | 위와 같음, 길이를 맞춘 clip | 330 | raw_text 330/330, request_id 330/330 |
| `eval_local_v6.jsonl` | Cosmos3-Edge (로컬), 마지막 frame | 330 | raw_text 330/330. 로컬 실행이라 request_id는 없다 |
| `eval_local_v7.jsonl` | Cosmos3-Edge, keyframe 8개 | 330 | frame별 `track.frames[*].raw_text` |
| `dev_prompt_selection.jsonl` | dev에서 시도한 모든 prompt 변형 | 228 | dev 36 episode |
| `dev_nebius_v2_pad201.jsonl` | Nemotron pad201, dev | 36 | |
| `dev_infra_failures.jsonl` | CUDA OOM으로 잃은 행 | 23 | 모델 출력이 아니다 |
| `results/hackathon/nebius_calls.jsonl` | Nebius 호출 로그 | 772 | |
| `results/hackathon/explore/nebius_calls_explore.jsonl` | 탐색 단계 Nebius 호출 로그 | 2380 | |

- 각 행에는 `raw_text`, 파싱된 `cosmos_*` 필드, `truth_*` 라벨, latency와 token 수, model id가 있다.
- z1이 언급하지 않은 파일도 있다. `data/outputs/hackathon_cosmos/*.orig.jsonl`은 settled-height로 다시 라벨링하기 전의 원본이다(eval v6/v2/v7은 150–152행).
- 영상은 git-ignore된 `data/outputs/` 아래에 있다.

## 1. 재현

- `z1_tier_triage.py --reps 250 --orders 8`을 다시 돌렸다. 결과는 기존 `z1_tier_triage.json`과 수치 차이가 0개다. 102초 걸렸고 exit 0이다.
- iid 민감도 실행(`--reps 100 --world-judge iid`)도 기존 `z1_tier_triage_iidworld.json`과 차이가 0개다.
- MC 없이 직접 계산한 값으로 독립 검산했다. World B는 binomial 합으로, World A는 hypergeometric으로 계산했고, sequential 규칙은 (K, k) 위 dynamic programming으로 계산했다.

| 항목 | z1 (MC) | 검증 (정확 계산) |
|---|---|---|
| B perfect plug-in, K=10 / 25 / 50 | 0.814 / 0.876 / 0.925 | 0.813 / 0.876 / 0.925 |
| A perfect plug-in, K=10 / 25 | 0.758 / 0.840 | 0.757 / 0.840 |
| B sequential, flat prior, P≥0.9 | 0.916 @ 25.1 | 0.9155 @ 25.07 |
| 같은 규칙, tier별 평균 rollout (BAD / OK / GOOD) | 8.9 / 25.1 / 28.8 | 9.0 / 25.1 / 28.8 |
| B Nemotron, qmarg-flat sequential | 0.873 @ 35.0 | 0.876 @ 34.9 |
| B Cosmos, qmarg-flat sequential | 0.881 @ 42.9 | 0.883 @ 42.9 |
| A sequential, flat prior | 0.836 @ 19.0 | 0.836 @ 19.0 |
| 0.80–0.95 구간의 checkpoint 비율 | 39% | 36/93 = 38.7% |

이 수치들 자체는 맞다. 문제는 아래와 같이 해석하고 표현한 방식에 있다.

## 2. 주장별 판정

| # | z1 주장 | 판정 | 근거 |
|---|---|---|---|
| 1 | perfect detector는 고정 K로 약 30–36개가 필요하다(bootstrap median 30 [18, >50], 96% 도달) | **부분 확인, 하한은 과소** | 정확 곡선은 K=36에서 처음 0.9를 넘는다(0.901). 그러나 K=41–42에서 0.9 아래로 다시 내려가고, **K≥43부터 계속 0.9 이상**이다. K=30은 0.899다. 잡음 없는 ckpt bootstrap의 median은 **35 [25, 50]**이고, 100% replicate가 50 안에 도달한다. z1의 "30 [18, >50]"은 replicate마다 들어간 MC 잡음(order 8개) 때문에 '처음 넘는 K'가 앞당겨진 값이다. World A의 "14% replicate 도달"도 정확 계산으로는 0.9%다 |
| 2 | Nemotron은 K=50에서도 0.878이고 50 안에 90%에 도달하지 못한다 | **judge 모형에 의존한다(경계선)** | 파라미터 draw 1500개 평균은 0.882, median 0.888, 5–95%는 [0.82, 0.92]다. draw의 35%는 K=50에서 0.9 이상이다. 다른 judge fit에서는 이렇다. point fit 0.897. can run 하나를 뺀 fit 0.908(K=48에서 도달) 또는 0.895. FPR 공변량을 50-ep 성공률로 바꾼 fit 0.907(K=40에서 도달). iid world 0.915(K=37에서 처음 넘지만 41–44에서 다시 아래). 정리하면 'K=50에서 0.88–0.91, 90%는 경계선'이다 |
| 3 | Cosmos는 K≤50에서 90%에 도달하지 못한다 | **확인** | 모든 변형에서 미달이다. point fit 0.823, run 제외 fit 0.837/0.844, 공변량 p50 fit 0.850, iid world 0.880. 파라미터 draw 중 0.9 이상은 4.5%뿐이다 |
| 4 | 비용이 큰 이유는 judge가 아니라 경계 근처 checkpoint 39%다 | **과장** | perfect와 Nemotron에는 맞다. Cosmos는 judge 자체가 크다. K=50에서 perfect 0.925, Cosmos 0.806이고, truth OK가 GOOD으로 판정되는 비율이 0.21 대 0.46이다 |
| 5 | 경계에서 먼 74개만 보면 perfect 약 14, Nemotron 약 18이면 된다 | **수치는 확인, 실무 진술로는 부적절** | A perfect의 point 최소 K는 14다. 그러나 층화를 50-ep **결과**로 정의했고, 이 50 episode에 truth와 K rollout이 모두 들어 있다. 층화 폭에 따라 값이 크게 바뀐다. 제외 구간을 [0.48,0.52]∪[0.88,0.92]로 좁히면(82개) 20, [0.4,0.6]∪[0.8,1.0)로 넓히면(28개) 6이다. 결과를 모르는 상태에서는 어떤 checkpoint가 이 층에 드는지 알 수 없으므로 예산 규칙으로 쓸 수 없다 |
| 6 | sequential은 같은 예산의 고정 K보다 약 4pt 높다 | **World B에서만 확인** | B에서 K=25 plug-in과 짝지어 비교하면 **+3.9pt [2.9, 4.9]**다. **World A에서는 +0.6pt [−0.7, +1.9]로 이득이 없다.** B에서는 rollout이 결정 모형과 같은 Bernoulli(p)로 생성되므로 sequential에 유리하다 |
| 7 | epoch/PLV prior는 sequential 평균 rollout을 5–6% 줄인다 | **수치는 맞지만 비교 기준이 약한 규칙이다** | 아래 3절 참조. prog(tierflat 기반)와 tierflat은 flat-prior frontier보다 B에서 4–6pt 낮다. 같은 예산의 고정 K plug-in보다도 2–3pt 낮은 쪽이다(CI가 0을 살짝 포함). flat 기반에서는 prior가 frontier를 개선하지 않는다 |
| 8 | prior만으로 K=0에서 0.56(tierflat 0.38), val_loss_logged는 0.43으로 더 나쁘다 | **tie-break 때문에 부풀려졌다** | prog prior는 79/93 ckpt에서 OK와 GOOD의 질량이 정확히 같다(설계상 50/50). argmax는 부동소수점 잡음으로 이 중 약 40%를 GOOD으로 부른다. tierflat은 같은 동점을 모두 OK로 부른다. tie를 일관되게 깨면 prog의 이득은 +18pt가 아니라 +6.5pt(tie→OK)에서 +10.8pt(tie→GOOD)다(50-ep truth 기준). val_loss_logged는 93/93이 동점이고 BAD를 하나도 부르지 않는다 |
| 9 | quality-dependent FPR 모델이 constant 모델보다 낫다 | **확인(더 강한 검정에서도)** | can run 두 개에서 run 하나를 빼고 fit해도 q 모델이 낫다. log score는 Nemotron −10.2 대 −14.6, Cosmos −11.3 대 −16.9다. MAE는 0.035 대 0.044, 0.039 대 0.055다. square는 run이 하나뿐이라 이 검정을 할 수 없다 |
| 10 | 실제 VLM의 tier 정확도가 시뮬레이션보다 높은 것은 can GOOD 4개의 '우연한' FP 때문이다 | **수치는 확인, '우연' 해석은 확립되지 않았다** | 아래 4절 참조 |
| 11 | 혼동은 거의 전부 OK→GOOD이다. GOOD recall은 0.95–0.98이고, Cosmos의 BAD→OK는 0.23이다 | 확인 | 수치가 모두 json과 일치한다. square Cosmos의 FPR은 s=0.3에서 0.33이다 |
| 12 | kde prior는 해롭다(B K=25에서 0.831, plug-in 0.876) | 확인 | |
| 13 | constant-FPR 보정은 Cosmos에 해롭다(0.698). marginalised flat 보정은 0.890 | 확인 | |

## 3. epoch/PLV prior: frontier로 보면 효과가 없다

z1은 prog_epoch(tierflat 기반)과 tierflat을 비교해서 '5–6% 절감'을 보고했다. 정확 계산으로 짝지어 비교하면 B는 −6.3%(평균 rollout −1.27 [−1.55, −1.02], 정확도 +0.4pt), A는 −5.6%로 수치는 맞다. 하지만 두 규칙 모두 열등하다.

- **두 규칙은 flat-prior sequential의 frontier보다 낮다.** 같은 평균 rollout에서 B는 −4~−6pt, A는 −3~−4pt다(임계값 0.85–0.975).
- **두 규칙은 고정 K plug-in보다도 낮은 쪽이다(CI가 0을 살짝 포함).**
  - prog_epoch sequential은 18.9개 rollout에서 0.845다. 같은 예산(K=19)의 plug-in은 0.867이다. 짝지은 차이는 −2.2pt [−5.1, +0.6]다.
  - tierflat sequential은 20.1개에서 0.841이다. K=20 plug-in은 0.871이고, 차이는 −3.0pt [−6.3, +0.1]다.
  - z1 md 표의 "같은 평균 예산의 고정 K plug-in" 열(0.82)은 이름이 잘못 붙었다. 코드(`fixed_K_acc_at_same_mean_budget`)는 plug-in이 아니라 **같은 Bayes 규칙의 고정 K 곡선**을 보간한다.
- **flat 기반에 P(BAD)만 붙이면(검증자 변형) 같은 frontier 위를 움직일 뿐이다.**
  - B에서 평균 rollout은 −18%(25.1→20.4)이지만 정확도는 −1.7pt [−2.7, −0.8]다. A도 −17%, −1.7pt다.
  - flat frontier와 비교한 이득은 임계값 0.85–0.975 전 구간에서 +0.7pt 이하다. 즉 임계값을 낮추는 것과 같은 효과이고, prior가 frontier를 개선하지 않는다.
  - val_loss_logged로 같은 prior를 만들어도 −13%, −1.1pt로 거의 같다. 정보는 '학습 초반이면 BAD'라는 epoch 진행 정보뿐이다.
- **예외로 아주 작은 예산(약 8 rollout)에서는 이득이 있다.** prog(tierflat)는 A에서 flat-seq보다 약 +5pt, B에서 약 +3pt 높다. B에서는 같은 예산의 고정 K plug-in(K=9에서 약 0.80)을 넘지 못한다.

**결론.** "PLV/epoch prior가 sequential 예산을 5–6% 줄인다"는 실무 진술로 쓰면 안 된다. 정확한 진술은 이렇다. *K≥10 예산에서 epoch/PLV prior는 정확도와 rollout의 frontier를 개선하지 않는다. PLV 지표는 epoch과 구분되지 않는다(prog_epoch과 prog_replan_overlap_h16의 결과가 소수점 넷째 자리까지 같다).* 이 결론은 z1과 y1의 "PLV는 epoch 이상의 정보가 없다"를 오히려 더 강하게 지지한다.

## 4. World R: near-plateau FP는 '우연'보다 구조로 보인다

- 수치는 확인했다. K=10에서 맞힌 checkpoint 수는 perfect 25/33, Nemotron 31/33, Cosmos 30/33이다.
- perfect와 Nemotron의 판정이 갈린 checkpoint는 6개이고, 6개 모두 Nemotron 쪽이 맞았다. can GOOD 4개, square_s1_e0030(FP 1개로 4/10→5/10, 즉 OK가 됨), square_s1_e0060(FN 2개)이다.
- can GOOD 4개의 실패 episode는 모두 `truth_failure_type = dropped`이고, grasped·lifted·dropped가 전부 True다. **두 VLM이 모두 성공으로 판정했다.** Nemotron은 4개 모두 "can을 오른쪽 bin 칸에 놓았고 끝까지 남아 있다"고 설명했다.
- **FP는 episode의 성질에 따라 생긴다.**
  - Nemotron FP 9개는 전부 grasp와 lift가 일어난 실패(dropped 8, not_placed 1)다. no_grasp 실패 38개 중에는 FP가 0개다.
  - Nemotron FP 9개 중 8개는 Cosmos도 FP다. 두 judge가 서로 독립이라면 기대값은 2.8개다. P(Nemotron FP | Cosmos FP)는 0.38이고 기저율은 0.13이다.
- **해석.**
  - GOOD checkpoint의 드문 실패는 거의 성공에 가까운 실패여서 영상에서 성공처럼 보인다. 이렇게 보면 "4/4 FP는 우연(모형 예측 약 0.44⁴ ≈ 0.04)"보다 실패 유형에서 오는 구조적 효과라는 설명이 데이터와 더 잘 맞는다.
  - 시뮬레이터는 실패마다 독립인 동전을 던지고, 그 확률은 checkpoint 품질에만 의존한다. 이 가정이 episode 수준에서 틀렸다.
  - 이 구조가 tier 정확도에 미치는 방향은 checkpoint 구성에 따라 다르다. GOOD checkpoint에서는 도움이 되고, 0.84–0.90의 OK checkpoint에서는 GOOD으로 부풀리는 쪽으로 해가 된다(can_s2_e0060: Cosmos FP→오판). 따라서 "시뮬레이터가 보수적"이라는 말은 이 33개 표본에 한해서만 맞다.
- 라벨 정의는 일관된다. eval truth와 training-time 결과 모두 robosuite의 success check를 쓴다(06 protocol). 따라서 이 차이는 라벨 버그가 아니라 시각적 모호성으로 보인다. 다만 영상을 직접 보지는 않았다.

## 5. 누설, held-out 처리, 독립성, 다중비교

- **누설.**
  - prior는 leave-one-run-out이다. 코드에서 `other = run_idx != run_idx[c]`를 확인했다. 지표 percentile은 결과를 쓰지 않는다.
  - World A에서 truth는 held-out 25, K rollout은 나머지 25에서 뽑는다. judge fit은 eval seed(200000+)를 쓰고, World R의 판정 규칙은 leave-one-checkpoint-out이다.
  - 명백한 누설은 없다. 남은 약한 누설은 두 가지다. ① judge fit에 쓴 33 ckpt가 분석 대상과 겹친다(seed는 다름). ② q 모델 자체를 y1 검증에서 같은 eval 데이터로 골랐다.
  - 경계 층화(5절 주장 5)는 결과로 정의한 층이다.
- **CI.**
  - bootstrap은 run 안에서 checkpoint를 다시 뽑는다. 따라서 run이나 task 사이의 분산(run 3개, task 2개)은 CI에 들어가지 않는다.
  - 같은 run의 인접 epoch은 서로 상관되어 있으므로 checkpoint를 독립으로 보는 bootstrap은 불확실성을 과소평가한다.
  - z1의 replicate CI에는 MC 잡음도 섞여 있다. MC 잡음은 구간을 넓히기도 하지만 '처음 넘는 K' 같은 통계량은 앞쪽으로 치우치게 만든다(주장 1).
- **다중비교.**
  - 규칙 약 46개 × K를 비교했고 보정은 없다. headline의 "sequential 0.916"은 여러 규칙 중 최고 규칙의 값이다.
  - 다만 flat prior는 사전에 정한 기본값이고, 정확 계산으로 다시 확인했다.
- **경계 규칙.**
  - 50-ep 성공률이 정확히 0.90인 checkpoint가 4개 있다(can_s1_e0035, can_s1_e0050, can_s2_e0060, square_s1_e0120). truth는 OK다. K가 무한히 커져도 이들의 정확도는 0.5로 수렴하므로 B의 이론적 상한은 0.978이다.
  - 판정 규칙을 GOOD: k/K ≥ 0.9로 바꾸면 B plug-in 정확도가 K=10에서 0.813→0.770, K=40에서 0.915→0.888, K=50에서 0.925→0.902로 떨어진다. 'GOOD 경계 규칙 하나가 정확도를 수 포인트 바꾼다'는 주장은 확인된다.
- **사소한 표기 오류.**
  - GOOD 비율이 findings에는 52%, md에는 53%로 적혀 있다. World A 기준으로 51.7%이고, World B(50-ep) 기준으로는 46%다.
  - log score 차이가 findings에는 "7–8 nat", md에는 "7–10 nat"이다(10은 pad201 포함).
  - "tierflat OK recall 0.43–0.53"에서 Cosmos는 0.40이다.

## 6. 수정된 headline

> 이 pool에서 BAD/OK/GOOD tier를 90%로 맞히는 비용은 크고, 그 크기는 경계 규칙과 checkpoint 분포에 좌우된다.
>
> - **perfect detector, 고정 K plug-in(World B, 정확 계산):** K=36에서 처음 0.9를 넘는다. K≥43부터는 계속 0.9 이상이다(ckpt bootstrap median 35 [25, 50]).
> - **sequential(flat prior, P≥0.9):** 평균 25.1개로 0.916이다. 같은 예산의 고정 K보다 +3.9pt [2.9, 4.9] 높지만, 이 이득은 World B에서만 나타난다(World A에서는 +0.6 [−0.7, 1.9]).
> - **Nemotron:** K=50 plug-in 정확도가 judge 모형에 따라 0.88–0.91이라 90% 도달 여부는 경계선이다. sequential은 35개에서 0.876이다.
> - **Cosmos3-Edge:** 모든 judge 모형 변형에서 K=50 정확도가 0.88 이하다. Cosmos에서는 judge 자체도 비용의 주요 원인이다(truth OK의 46%가 GOOD으로 판정됨).
> - **epoch/PLV prior:** K≥10에서 정확도–rollout frontier를 개선하지 않는다. z1이 보고한 5–6% 절감은 flat frontier보다 4–6pt 낮은 규칙과 비교한 수치다.
> - **'경계에서 먼 checkpoint는 14–18개'라는 진술:** 결과로 정의한 층에 대한 조건부 진술이다. 층의 폭에 따라 6에서 20까지 바뀐다.

## 7. 아직 열려 있는 위험

- run 3개, task 2개뿐이다. GOOD은 전부 can이다. 모든 정확도는 tier 구성에 크게 좌우된다.
- judge 오류 모델은 실패 67개(고품질 구간은 9개)로 fit했다. FP가 실패 유형에 따라 생긴다는 점(4절)은 모형화되어 있지 않다. square는 run 단위로 교차검증할 수 없다.
- 실로봇 예산으로 옮기려면 near-plateau checkpoint에서 checkpoint당 30개 이상의 녹화와 판정이 필요하다. 이를 위한 Nebius 호출은 사용자 승인이 있어야 한다.
- y1 검증에서 지적된 문제가 여전히 남아 있다. `*_h10` 열과 `*_h16` 열이 93/93행에서 같다. 따라서 replan_overlap_h16을 H=16 고유의 지표로 해석하면 안 된다.

## 재실행

```
cd /path/to/repo
PY="env PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python"
# z1 재현 (기존 결과와 수치 차이 0)
$PY scripts/hackathon/explore/z1_tier_triage.py --reps 250 --orders 8 \
    --out data/outputs/hackathon_explore/z1_tier_triage_verification/rerun_main.json --fig ""
$PY scripts/hackathon/explore/z1_tier_triage.py --reps 100 --orders 8 --world-judge iid \
    --out data/outputs/hackathon_explore/z1_tier_triage_verification/rerun_iid.json --fig ""
# 독립 정확 계산 (약 50초) + frontier (약 2분; import하면서 verify를 다시 실행)
$PY scripts/hackathon/explore/z1_tier_triage_verify.py
$PY scripts/hackathon/explore/z1_tier_triage_verify_frontier.py
```
