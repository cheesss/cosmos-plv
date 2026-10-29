# y3_visual_robustness 적대적 검증 (EXPLORATORY, 사후 분석)

**판정: partially_confirmed (부분 확인).**

- **확인된 것.** 데이터, 상태 재생, 재채점에는 결함이 없다.
  - 600 + 600개 judge 행을 독립 코드로 재채점했고, 발표된 수치와 모두 일치했다.
  - 상태 재생이 비트 단위로 결정적이라는 주장을 CPU에서 독립적으로 재현했다(6/6, 음성 대조 포함).
  - 화질 저하가 검출 가능한 영향을 주지 않는다는 결론은 유지된다.
  - 손목 카메라에서 judge가 우연 수준이라는 결론도 유지된다. 이 효과는 다중비교 보정 후에도 유의한 유일한 효과다. 다만 프롬프트 교란은 그대로 남아 있다.
- **부분 반박된 것.** 먼 카메라에 대한 해석이다. "카메라 위치는 결정적이다"와 "먼 카메라는 실패를 성공으로 보아 VLM-judged success를 부풀린다"는 **backend와 표본 구성에 따라 방향과 크기가 바뀐다.**
  - Nemotron birdview에서 재현되는 오류는 반대 방향이다. can 성공을 `no_grasp`로 판정하는 FN이며, retest에서 4/4가 다시 나왔다.
  - eval 구성으로 재가중하면 Nemotron birdview는 판정 성공률을 0.77에서 0.62로 **낮춘다.**
  - 같은 재가중에서 "어느 조건에서도 Nemotron ≥ Cosmos"는 birdview와 wrist에서 뒤집힌다.
- **헤드라인 수치의 성격.** "birdview −0.15~−0.17"은 실패를 많이 뽑은 부분집합에서 나온 값이다. Bonferroni(18개 비교) 보정 구간은 0을 포함한다.

## 질문

y3 보고서(`docs/hackathon/explore/y3_visual_robustness.md`)를 검증했다. 대상은 다음과 같다.

- 재생 결정성
- 화질 저하 무영향
- 카메라 효과의 크기와 방향
- 오류 메커니즘 수치
- 모델 순위
- 노이즈 바닥

확인한 항목은 누출, dev/eval 혼동, pairing, clustering, 분모, cherry-picking, 그리고 주장이 수치보다 강한지 여부다.

## 방법

모든 작업은 CPU에서 했다. 새 GPU 작업은 없었다. Nebius 호출은 30회를 새로 했다.

1. **독립 재채점** (`scripts/hackathon/explore/y3_visual_robustness_verify.py`). y3_analyze를 import하지 않고 저장된 `judge_local.jsonl` / `judge_nebius.jsonl`을 다시 채점했다.
   - 행 무결성: 중복, 누락, truth와 summary.json·npz의 일치 여부.
   - Nebius evidence log 대조: request_id, video_sha256, prompt_sha256, temperature, pad.
   - acc / bal-acc / TPR / TNR / Wilson과 paired Δ를 다시 계산했다.
   - 체크포인트 클러스터 bootstrap은 20000회, seed 12345로 돌렸다. Bonferroni 구간은 α = 0.05/18이다.
   - McNemar 검정에 Holm 보정을 적용했고, LOCO와 task별 Δ, FN/FP 분해도 계산했다.
2. **CPU 상태 재생** (`y3_visual_robustness_verify_replay.py`, Env A, `MUJOCO_GL=osmesa`, `render=False`, `CUDA_VISIBLE_DEVICES=""`). 선택 목록에서 10번째마다 하나씩 6개를 골랐다(can 3, square 3, 결과와 무관한 고정 규칙). 음성 대조 2개를 함께 돌렸다. 하나는 step 20의 action x에 +1e-3을 더한 것이고, 다른 하나는 env seed를 +1한 것이다.
3. **Nemotron birdview test-retest** (`y3_visual_robustness_verify_retest.py`, 30회 호출). y3에서 cam_birdview와 cam_agentview의 정오가 갈린 15개 에피소드를 두 조건 모두 같은 frozen 프롬프트와 파라미터로 다시 돌렸다. tag는 `explore/y3_visual_robustness_verification/retest/...`다. 불일치 쌍을 기준으로 골랐기 때문에, 순수 노이즈라면 평균으로의 회귀 때문에 효과가 줄어드는 방향으로 편향된다.
4. **층/가중/대안 클러스터** (`y3_visual_robustness_verify_strata.py`).
   - (a) cherry-pick 점검: frozen 정확도를 층별로 부분집합과 250개 universe에서 비교했다.
   - (b) post-stratification: 층별 Δ를 universe 구성(성공 0.756)으로 재가중했고, CI는 층별 episode bootstrap으로 구했다.
   - (c) env seed 클러스터 bootstrap: 같은 seed를 쓰는 에피소드는 체크포인트가 달라도 초기 장면이 같다.

## 명령과 종료 코드

모든 명령은 `/path/to/repo`에서 실행했다. 로그는 `data/outputs/hackathon_explore/y3_visual_robustness_verification/`에 있다.

| # | 명령 | exit |
|---|---|---|
| 1 | `PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/y3_visual_robustness_verify.py` | 0 |
| 2 | `MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa CUDA_VISIBLE_DEVICES="" bash -lc 'source scripts/hackathon/env.sh && python scripts/hackathon/explore/y3_visual_robustness_verify_replay.py'` | 0 |
| 3 | `PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/y3_visual_robustness_verify_retest.py` (30 calls, API 오류 0, 파싱 실패 0) | 0 |
| 4 | `PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/y3_visual_robustness_verify_strata.py` | 0 |
| 5 | 인라인 점검 2개: orig와 cam_agentview의 입력 hash/픽셀 동일성, 재렌더 PSNR의 task별 분포. 첫 시도는 시스템 python에 `imageio.v2`가 없어 exit 1이 났고, Env B로 다시 돌려 exit 0을 얻었다. | 1 → 0 |

`y3_analyze.py`는 다시 실행하지 않았다. 기존 결과 파일을 덮어쓰기 때문이다. 대신 1번에서 같은 수치를 독립적으로 재계산했다.

## 결과

### 1. 무결성과 재현

- **행 무결성.** local과 nebius 모두 600행이다. 중복 0, 누락 0, infra 오류 0이다. local 파싱은 600/600 ok, nebius는 598 ok에 invalid_json 2건이다(둘 다 eye-in-hand의 square 실패).
- **truth 일치.** truth가 현재 summary.json, npz `success.any()`와 어긋난 행은 0이다.
- **Nebius evidence.** y3 tag 호출이 600회 있고 tag도 600개로 모두 고유하다. 오류 0이며, 모든 행의 request_id가 log에 있다. temperature 0.2, max_tokens 700, pad 20.1 s가 600/600에서 같다. prompt hash는 task마다 1개뿐이라 모든 조건에서 frozen 프롬프트가 같았다.
- **수치 재계산.** 발표된 JSON과 다른 값이 **0건**이다. 모든 조건의 acc, bal, TPR, TNR, Wilson과 모든 paired Δ, 불일치 쌍 수가 일치했다. 체크포인트 클러스터 CI도 사실상 같다(예: Nemotron birdview [−0.25, −0.04] vs 발표값 [−0.25, −0.04]).
- **cherry-pick 징후는 없다.** frozen 정확도가 층별로 부분집합과 universe에서 비슷하다.

  | backend | 층 | universe 정확도 (n) | 부분집합 정확도 (n) |
  |---|---|---|---|
  | local | can/dropped | 0.385 (13) | 0.375 (8) |
  | local | square/dropped | 0.50 (10) | 0.44 (9) |
  | local | square/no_grasp | 0.69 (13) | 0.67 (9) |
  | Nemotron | square/none | 0.979 (47) | 0.917 (12) |

  나머지 층은 둘 다 1.0이거나 거의 같다.
- **선택 규칙.** VLM 출력을 읽지 않는다(코드로 확인). 프롬프트와 파라미터는 모두 protocol에서 frozen된 것이고, eval에서 무엇을 고른 흔적은 없다. 누출이나 dev/eval 혼동도 발견하지 못했다.

### 2. 재생 결정성: 확인

| 점검 | 결과 |
|---|---|
| y3 log 60/60 | eef / obj 최대 오차 0.0, grasped / success step 일치 1.0, 성공 플래그 60/60, 프레임 수 60/60 일치 |
| 본 검증의 CPU 재생(렌더러 없음) 6/6 | eef / obj 오차 0.0, 성공 일치 6/6 (can 3, square 3, 길이 130–400 step) |
| 음성 대조: action +1e-3 (step 20) | step 21부터 eef 오차 1.5e-4 m, obj 오차 2.5e-4 m. 점검에 검출력이 있다. |
| 음성 대조: seed +1 | obj 오차 0.68 m, 성공 판정이 뒤집힘 |
| 재렌더 mp4가 원본과 바이트 동일 | square 30/30이 동일(Nemotron 입력 hash와 local 마지막 PNG가 픽셀 단위로 같다). can 30/30은 미세하게 다르다(재인코딩한 마지막 프레임의 PSNR 53–71 dB, 중앙값 67 dB). |

해석은 이렇다.

- *물리 상태* 재생이 float32 probe 수준에서 비트 단위로 같다는 주장은 성립한다.
- *렌더링*은 square만 비트 단위로 같다. can은 거의 같은 수준이다. judge에는 영향이 없었다(local 예측 60/60 동일).
- `env.close()` 함정은 같은 첫 3개 에피소드로 대조했다. close를 부른 버전은 2/3이 13–19 dB였고, debug 버전(close 없음)의 첫 프레임은 3/3이 42.1–42.2 dB였다. 증거는 이 정도이며 메커니즘은 확인되지 않았다.
- 참고로 `rollout_recorder.py`도 batch가 끝날 때 `env.env.env.close()`를 부른다. 다만 현재 eval과 dev 데이터는 체크포인트마다 batch가 1개뿐이고(batch_index 0), eval loop는 `--max-checkpoints 1`로 돈다. 그래서 현재 데이터에는 영향이 없을 가능성이 높다. dev 프로세스 구성은 확인하지 않았다.

### 3. 노이즈 바닥: 확인, 오히려 더 강하다

- Nemotron orig 재실행은 frozen pad201 run과 4개가 달랐다(56/60). 4개 모두 실제 실패 에피소드다.
- orig와 cam_agentview 사이에서도 4개가 뒤집혔다. 그중 **3개는 Nemotron에 보낸 압축 영상의 sha256이 바이트 단위로 같은** 입력이었다. 즉 temperature 0.2에서 나오는 순수 샘플링 노이즈다.
- local은 seed가 고정되어 있어 60/60 같았다. raw text는 57/60이 같았다.

### 4. paired Δ: 보정 구간, 대안 클러스터, 재가중

열 설명:

- Δ는 부분집합에서 조건 정확도 − 대조 정확도다. 대조는 화질 조건이면 orig, 카메라 조건이면 cam_agentview다.
- CI 열(ckpt 95%, Bonferroni, seed 95%)은 이 부분집합 Δ의 구간이다. Bonferroni는 18개 비교를 보정한 체크포인트 클러스터 구간이고, seed 95%는 env seed 클러스터 구간이다.
- "+FN / +FP"는 대조 대비 늘어난 FN과 FP 수다. 실패 쪽 숫자에는 파싱 실패도 들어 있다.
- "재가중 Δ"는 eval universe 구성(can/none 142, square/none 47, can/no_grasp 25, can/dropped 13, square/no_grasp 13, square/dropped 10; 성공 0.756)으로 post-stratify한 값이다. 재가중 acc와 판정 성공률도 같은 가중을 쓴다.

| backend | 조건 | Δ (부분집합) | ckpt 95% | Bonferroni | seed 95% | Holm p | +FN / +FP | 재가중 Δ [95%] | 재가중 acc | 재가중 판정 성공률 |
|---|---|---|---|---|---|---|---|---|---|---|
| local | down224 | +0.050 | [0.00, 0.09] | [0.00, 0.11] | [0.00, 0.10] | 1.000 | −1 / −2 | +0.027 [0.00, 0.06] | 0.912 → 0.940 | 0.812 → 0.817 |
| local | down112 | −0.017 | [−0.07, 0.04] | [−0.10, 0.06] | [−0.08, 0.03] | 1.000 | −1 / +2 | +0.005 [−0.02, 0.04] | 0.912 → 0.918 | 0.812 → 0.838 |
| local | jpeg20 | −0.033 | [−0.11, 0.04] | [−0.15, 0.07] | [−0.10, 0.03] | 1.000 | +1 / +1 | −0.021 [−0.07, 0.03] | 0.912 → 0.891 | 0.812 → 0.802 |
| local | blur | +0.033 | [0.00, 0.07] | [0.00, 0.09] | [0.00, 0.08] | 1.000 | −1 / −1 | +0.021 [0.00, 0.06] | 0.912 → 0.934 | 0.812 → 0.822 |
| local | dim | −0.017 | [−0.06, 0.00] | [−0.09, 0.00] | [−0.06, 0.00] | 1.000 | 0 / +1 | −0.004 [−0.01, 0.00] | 0.912 → 0.908 | 0.812 → 0.817 |
| local | frontview | −0.100 | [−0.23, 0.02] | [−0.30, 0.06] | [−0.18, −0.03] | 1.000 | −1 / +7 | −0.020 [−0.05, 0.02] | 0.912 → 0.892 | 0.812 → 0.864 |
| local | birdview | −0.167 | [−0.33, 0.00] | [−0.43, 0.07] | [−0.24, −0.09] | 0.340 | −1 / +11 | −0.050 [−0.10, 0.00] | 0.912 → 0.862 | 0.812 → 0.894 |
| local | eye-in-hand | −0.283 | [−0.45, −0.15] | [−0.54, −0.09] | [−0.43, −0.10] | 0.026 | +10 / +7 | −0.229 [−0.29, −0.17] | 0.912 → 0.683 | 0.812 → 0.729 |
| Nemotron | down224 | −0.033 | [−0.11, 0.05] | [−0.15, 0.10] | [−0.12, 0.04] | 1.000 | −1 / +3 | +0.004 [−0.03, 0.05] | 0.946 → 0.950 | 0.779 → 0.806 |
| Nemotron | down112 | −0.033 | [−0.10, 0.03] | [−0.13, 0.07] | [−0.09, 0.03] | 1.000 | 0 / +2 | −0.005 [−0.03, 0.02] | 0.946 → 0.941 | 0.779 → 0.784 |
| Nemotron | jpeg20 | −0.083 | [−0.17, 0.00] | [−0.22, 0.05] | [−0.17, 0.00] | 1.000 | 0 / +5 | −0.026 [−0.06, 0.01] | 0.946 → 0.920 | 0.779 → 0.805 |
| Nemotron | blur | +0.017 | [−0.05, 0.09] | [−0.09, 0.13] | [−0.04, 0.10] | 1.000 | −1 / 0 | +0.019 [−0.02, 0.06] | 0.946 → 0.965 | 0.779 → 0.791 |
| Nemotron | dim | +0.017 | [−0.04, 0.08] | [−0.07, 0.11] | [−0.04, 0.08] | 1.000 | 0 / −1 | +0.010 [−0.01, 0.03] | 0.946 → 0.956 | 0.779 → 0.769 |
| Nemotron | frontview | −0.067 | [−0.16, 0.03] | [−0.20, 0.09] | [−0.21, 0.05] | 1.000 | 0 / +4 | −0.032 [−0.09, 0.03] | 0.952 → 0.920 | 0.773 → 0.805 |
| Nemotron | birdview | −0.150 | [−0.25, −0.04] | [−0.30, 0.02] | [−0.29, 0.00] | 0.527 | +4 / +5 | **−0.223 [−0.39, −0.08]** | 0.952 → **0.728** | 0.773 → **0.618** |
| Nemotron | eye-in-hand | −0.300 | [−0.45, −0.18] | [−0.55, −0.12] | [−0.34, −0.25] | 0.016 | +20 / −2 | −0.649 [−0.74, −0.53] | 0.952 → 0.302 | 0.773 → 0.088 |

LOCO(체크포인트 하나를 빼고 다시 계산)에서는 어떤 카메라 효과도 부호가 바뀌지 않았다. 범위는 다음과 같다.

| backend | frontview | birdview | eye-in-hand |
|---|---|---|---|
| local | [−0.12, −0.04] | [−0.20, −0.09] | [−0.33, −0.25] |
| Nemotron | [−0.09, −0.04] | [−0.17, −0.13] | [−0.32, −0.26] |

task별로 나누면 다음과 같다.

- local frontview와 birdview의 손실은 거의 모두 square에서 나온다. square Δ는 −0.23 / −0.30이고 can Δ는 +0.03 / −0.03이다.
- Nemotron birdview의 손실은 can(−0.17)과 square(−0.13) 양쪽에서 나온다.

### 5. Nemotron birdview retest (30회)

| y3의 불일치 유형 | n | retest에서도 유지 | agentview 재실행 결과 | birdview 재실행 결과 |
|---|---|---|---|---|
| agentview는 맞고 birdview는 틀림 | 12 | **7** | 여전히 맞음 11/12 | 여전히 틀림 8/12 |
| birdview는 맞고 agentview는 틀림 | 3 | 2 | 여전히 틀림 3/3 | 여전히 맞음 2/3 |

- 유지된 7개 중 **4개가 can 성공을 `no_grasp`로 본 FN이고, 4/4가 모두 다시 나왔다.** 설명도 "never closes around it… remains in the left bin"으로 같다.
- 반대로 실패를 성공으로 본 birdview FP 8개 중에서 다시 나온 것은 3개뿐이다.
- 불일치 15개만 다시 돌렸을 때 순 효과는 −9에서 −5 에피소드로 줄었다. 일치 45개는 y3 값을 그대로 두고 계산하면 대략 −0.08이다. 이 값은 선택 편향 때문에 보수적인 추정이다.
- 결론적으로 Nemotron birdview 효과의 방향은 재현된다. 재현되는 성분은 **FN(성공을 실패로 봄)** 이고, FP 성분은 대부분 노이즈와 구별되지 않는다.

### 6. 보고서가 인용한 개수

| 인용 | 검증 | 비고 |
|---|---|---|
| local frontview 실패 square 16/18 `on_peg`, birdview 18/18 | 일치 | agentview에서도 이미 8/18이 `on_peg`였다. 늘어난 것은 +8 / +10이다. birdview에서는 square 30/30 전부를 `on_peg`라고 답했다(상수 출력, square bal acc 0.50). |
| local birdview 실패 can 10/18 `in_gripper` | 일치 | agentview는 3/18 |
| local eye-in-hand 실패 can 14/18 `right_bin`, 성공 square 9/12 `not_visible` | 일치 | |
| Nemotron eye-in-hand 성공 21/24를 실패로 판정(TPR 0.12) | 일치 | |
| 한국어 요약·structured findings의 "성공 88%(21/24)를 '잡지 못함/들지 않음'으로 판정" | **틀림** | 21개의 failure type은 `no_grasp` 13개, `not_placed` 8개(모두 square)다. "잡지 못함"은 13/24(54%)다. |
| Nebius 600회, API 오류 0, invalid JSON 2(eye-in-hand) / local 600회, OOM 0, 파싱 실패 0 | 일치 | Nebius 클라이언트 두 프로세스(4 + 3 threads)가 동시에 돈 구간이 있다. 보고서의 "3–5 threads"보다 많지만 제약 위반은 아니다. |
| frozen 정확도 local 0.767 / Nemotron 0.850, orig 재실행 일치 60/60 / 56/60 | 일치 | |

## 해석: 주장별 판정

1. **재생 결정성: 확인.** 독립 CPU 재현과 음성 대조로 뒷받침된다. 렌더링은 can에서 비트 단위로 같지 않지만, judge에는 영향이 없었다.
2. **화질 저하는 거의 영향이 없다: 확인.** 단, "검출 가능한 효과 없음"으로 읽어야 한다. 단일 강도, n = 60이고, Nemotron은 입력을 320 px로 줄인다. 재가중하면 모든 Δ가 ±0.03 안에 있다. "의미 있는 손실은 강한 JPEG 하나뿐"이라는 문장은 과장이다. Nemotron jpeg20은 Bonferroni 구간 [−0.22, 0.05], Holm p = 1.0이고, 재가중 Δ는 −0.026이다.
3. **"카메라 위치가 결정적이다"(헤드라인): 부분 확인.**
   - 손목 카메라 효과는 모든 분석에서 강하게 남는다. 다만 frozen 프롬프트와 교란되어 있다.
   - frontview는 두 backend 모두 체크포인트 클러스터 CI가 0을 포함한다. 보고서 limitations도 "no detectable effect"라고 쓴다. 그런데 헤드라인 표는 이를 '결정적' 근거로 묶었다.
   - birdview는 보정 전에는 Nemotron에서만 0을 배제한다. Bonferroni 후에는 두 backend 모두 0을 포함한다.
   - 클러스터 단위를 env seed로 바꾸면 local frontview와 birdview는 유의하고 Nemotron birdview는 경계에 걸린다. 유의성이 분석 선택에 민감하다.
4. **"먼 카메라는 실패를 성공으로 보아 VLM-judged success를 부풀린다": backend에 따라 다르다. Nemotron birdview에서는 반박된다.**
   - Cosmos에서는 맞다. 재가중 판정 성공률이 0.81에서 0.86(frontview), 0.89(birdview)로 오른다(실제 0.756).
   - Nemotron birdview의 증분은 +4 FN / +5 FP다. retest에서 재현된 것은 FN이고, eval 구성에서는 판정 성공률이 0.77에서 0.62로 *떨어진다*.
   - 판정 성공률이 일정하게 부풀거나 줄어도 체크포인트 *순위*가 바뀌는지는 아무도 측정하지 않았다. 그래서 "hybrid screening 점수를 직접 편향시킨다"는 근거가 없는 확장이다.
5. **헤드라인 크기("birdview −0.15~−0.17, wrist −0.28~−0.30"): 이 부분집합에만 해당한다.** eval 구성으로 재가중하면 크기가 크게 바뀐다.
   - Cosmos는 frontview −0.02, birdview −0.05, wrist −0.23이다.
   - Nemotron은 frontview −0.03, birdview −0.22, wrist −0.65다.
   - 재가중값은 가중이 가장 큰 can/none 층(가중 0.57)을 12개 에피소드로 추정하므로 불확실하다. 그래도 방향 차이는 분명하다. 보고서는 "paired 차이는 비교 가능하다"고 했지만, 효과가 층마다 다르기 때문에 paired Δ도 표본 구성에 의존한다.
6. **"어느 조건에서도 Nemotron ≥ Cosmos": 부분집합에서만 참이다.**
   - eval 구성으로 재가중하면 birdview는 0.728 vs 0.862, wrist는 0.302 vs 0.683으로 **Cosmos가 앞선다.**
   - 부분집합에서도 wrist 우위(0.583 vs 0.483)는 실패 비율이 0.60인 데서 나온다. Nemotron이 92%를 실패로 답했기 때문이다. bal acc로 보면 0.507 vs 0.493으로 같다.
   - 덧붙여 이 부분집합에서는 항상 "실패"라고 답하는 상수 예측기의 정확도가 0.60이다. Cosmos birdview(0.600)와 두 모델의 wrist 정확도는 이 기준선 이하다.
7. **오류 메커니즘 1(작은 square nut 때문에 `on_peg`로 답함): 대체로 확인.** 그림에서도 nut은 수 픽셀 크기다. 다만 agentview에서 이미 8/18이 `on_peg`였다. birdview에서는 square 출력이 상수였다.
8. **오류 메커니즘 2(wrist): 확인.** 다만 Nemotron의 '잡지 못함 88%'라는 서술은 틀렸다(위 표 참조).
9. **노이즈 바닥 주장: 확인, 더 강화된다.** 같은 바이트 입력에서도 판정이 뒤집힌다.
10. **시점 의존 문구에 대한 주장: 확인.** 그림에서 frontview와 birdview의 bin 좌우 배치가 프롬프트와 맞는다.

## 한계

- 이 검증도 사후 탐색이다. 다중비교 보정은 맥락 정보로만 쓰며, 탐색 결과를 기각하는 기준으로 쓰지 않았다.
- 재가중 CI는 층별 episode bootstrap이라 체크포인트 클러스터링을 무시한다. can/none 12개가 가중 0.57을 대표한다.
- retest는 불일치 쌍만 다시 돌렸다(예산 30회). 일치 쌍 45개에서 생길 수 있는 새 뒤집힘은 측정하지 않았다. 불일치 쌍을 골라 돌렸으므로 retest Δ는 평균으로의 회귀 때문에 0 쪽으로 편향된다.
- 렌더링 결정성은 CPU에서 다시 렌더링하지 않았다(EGL과 osmesa는 픽셀이 다르다). y3 log와 입력 hash 동일성으로만 판단했다.
- `env.close()` 함정은 y3의 원래 check 버전 코드가 남아 있지 않다. 그래서 close 여부 외에 다른 차이가 있었는지 확인할 수 없다.
- local judge의 seed 간 변동(다른 generation seed)은 측정하지 않았다. 따라서 local의 작은 Δ(±0.05)는 seed 하나에 대한 결과다.
- repo AGENTS.md는 commit/push를 요구하지만, 작업 제약(git commit 금지)에 따라 하지 않았다.

## 다음 단계

1. y3 보고서의 헤드라인과 usefulness 문구를 고친다.
   - "먼 카메라 = success 부풀림"은 Cosmos에만 해당한다고 명시한다.
   - Nemotron birdview는 can lift를 놓쳐 FN이 생긴다고 쓴다.
   - "카메라 위치는 결정적"은 "손목 카메라(frozen 프롬프트)는 확실하고, 먼 카메라는 backend와 표본 구성에 따라 −0.02에서 −0.22"로 바꾼다.
   - "88% no_grasp"는 "21/24 실패 판정(no_grasp 13, not_placed 8)"으로 바꾼다.
2. 카메라 효과는 eval 구성으로 재가중한 값과 체크포인트 수준의 판정 성공률 순위(Spearman)로 보고한다. screening 영향은 순위로 봐야 한다.
3. Nemotron은 cam_agentview와 cam_birdview 60개 전체를 2회 더 반복해 majority vote Δ를 구한다(약 240회). 앞서 측정한 노이즈 바닥을 넘는 효과만 남긴다. 노이즈 바닥은 frozen run 대비 4/60(6.7%)이고, 바이트가 같은 입력 30개에서는 3개(10%)가 뒤집혔다.
4. 판정용 카메라 권고("가깝고 사선인 고정 3인칭 카메라")는 유지한다. 근거는 "먼 카메라는 backend마다 서로 다른 방향의 편향을 만든다"로 바꾼다. 또 카메라를 바꾸면 dev 선택을 backend별로 다시 해야 한다고 명시한다.
5. `rollout_recorder`가 여러 batch나 체크포인트를 한 프로세스에서 돌 때 `close()`를 부르면 렌더가 깨질 수 있다. 이 위험을 KNOWN_RISKS 후보로 넘긴다. 현재 eval 데이터는 체크포인트마다 batch 1개, 프로세스 1개라 영향이 없을 가능성이 높다.

## 파일

- **스크립트:** `scripts/hackathon/explore/y3_visual_robustness_verify.py`, `y3_visual_robustness_verify_replay.py`, `y3_visual_robustness_verify_retest.py`, `y3_visual_robustness_verify_strata.py`
- **결과:** `results/hackathon/explore/y3_visual_robustness_verification.json`(재채점, 보정 구간, 인용 개수), `results/hackathon/explore/y3_visual_robustness_verification_strata.json`(층, 재가중, seed 클러스터, retest 행)
- **Nebius evidence:** `results/hackathon/explore/nebius_calls_explore.jsonl`, tag `explore/y3_visual_robustness_verification/retest/…`(30회)
- **scratch:** `data/outputs/hackathon_explore/y3_visual_robustness_verification/`(`verify.log`, `replay_cpu.jsonl`, `replay_cpu.log`, `retest_nebius.jsonl`, `retest.log`, `strata.log`)
