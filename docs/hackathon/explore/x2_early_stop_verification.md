# x2_early_stop: adversarial verification (EXPLORATORY, post-hoc)

**Verdict: partially confirmed.** Every number in the x2 report reproduces exactly from code and data
(0 differences on the frozen snapshot). No outcome leakage, clip-length leakage, frame/step misalignment,
dev/eval confusion or variant cherry-picking was found. However, several claims are stronger than the
numbers support:

- the confidence intervals assume independent episodes, and they are about 2x too narrow once
  episodes are clustered by checkpoint;
- two of the "vision/state adds X points" gains are mostly epoch-0 effects, or their CIs include 0;
- the "4-7% false aborts" price and the square top-1 flip come from the small fitting sets, not from
  early stopping itself;
- one detail about the false failure calls is wrong;
- the side finding's implied impact on experiment 2 is nil.

**One-line summary:** Every x2 number reproduces exactly (0 differences). There was no leakage, no clip-length leakage and no frame-alignment problem. However, the following claims are stronger than the data.
- The confidence intervals ignore within-checkpoint correlation, so they are about 2x narrower than they should be.
- Most of Cosmos's +4.5pt gain comes from epoch-0. Without epoch-0 it is +1.5pt.
- The CI of the privileged-state dev→eval +3pt includes 0.
- The 4–7% false aborts and the square top-1 flip come from the small fitting sets. Fitting on the other 19 checkpoints (LOCO) gives 16–20% savings, 2/141 false aborts and no ranking change.
- The square label error is real, but relabeling leaves the Nebius failure-type accuracy unchanged.

Scope and budget: CPU only. No GPU runs and **0 new Nebius calls** were made. Everything VLM-related was
re-scored from the saved raw outputs.

## 1. What was checked and how (commands, exit codes)

All commands ran from `/path/to/repo` with
`PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 ~/.venvs/cosmos-plv/bin/python`.

| # | command | exit | output |
|---|---|---|---|
| 1 | `scripts/hackathon/explore/x2_early_stop_analyze.py --out data/outputs/hackathon_explore/x2_verify/rerun_current.json` (current data: 210 eval eps, 103 v7 rows) | 0 | scratch JSON |
| 2 | `scripts/hackathon/explore/x2_early_stop_verify.py` (frozen re-run plus all extra checks; new file) | 0 | `results/hackathon/explore/x2_early_stop_verification.json`, `data/outputs/hackathon_explore/x2_verify/rerun_frozen_snapshot.json` |
| 3 | `scripts/hackathon/explore/x2_early_stop_verify_loco.py` (leave-one-checkpoint-out and leave-one-run-out refits; new) | 0 | `results/hackathon/explore/x2_early_stop_verification_loco.json` |
| 4 | `scripts/hackathon/explore/x2_early_stop_tables.py data/outputs/hackathon_explore/x2_verify/rerun_frozen_snapshot.json` | 0 | `data/outputs/hackathon_explore/x2_verify/tables_frozen.md` |
| 5 | `scripts/hackathon/explore/x2_early_stop_verify_relabel.py` (re-annotation with settled height on a trace copy; annotator untouched; new) | 0 | `results/hackathon/explore/x2_early_stop_verification_relabel.json` |
| 6 | `scripts/hackathon/explore/x2_early_stop_verify_evidence.py` (Nemotron evidence fields vs state; new) | 0 | `results/hackathon/explore/x2_early_stop_verification_evidence.json` |

Two ad-hoc inline Python checks also ran before step 6, which now covers both. One of them (an
episode-trace printout) exited 1 with an index error after printing its full output. No existing file
was modified.

**Frozen snapshot.** The x2 JSON was built on 200 eval episodes from 20 checkpoints and 90 v7 rows from
9 checkpoints. Since then, `square_s1_e0150` and 13 more v7 rows have arrived. `x2_early_stop_verify.py`
monkeypatches the x2 loaders to the original checkpoint sets and re-runs `x2_early_stop_analyze.main()`.
A recursive diff against `results/hackathon/explore/x2_early_stop.json` gives **0 differences**,
bootstrap CIs included.

## 2. Checks that passed (no issue found)

| check | result |
|---|---|
| Reproduction | 0 differences (frozen snapshot); table spot-checks of 5.2/5.3 match the re-run tables. |
| Oracle ceiling | 23600 / 43374 steps = 0.544. All 59 failures run the full 400 steps; 29.5% of episodes fail. |
| Clip-length leakage (Nemotron) | All 252 clips at 4 s have 41 frames; all 47 clips at 8 s have 81 frames. In 20/20 sampled clips the last frame matches source video frame `cut/2` (best offset 0 of {-1, 0, +1}). Equal-length clips, prefix ends exactly at the cut step. |
| Frame/step alignment (v7) | All 36 failure tracks have keyframes at steps (0, 58, 114, 172, 228, 286, 342, 400) = `FAIL_KF_STEPS`; `times_s = idx / fps` and `step = 20 t` are consistent with frame k = step 2k. |
| Nemotron re-score from raw text | 0/299 mismatches with the x2 parser and 0/299 with an independent regex; 0 truth-label mismatches vs `summary.json`; evidence log has 299 `explore/x2_early_stop/` calls, all tags unique. |
| Dev → eval ordering / cherry-picking | Last dev call 19:48:59 UTC, first eval call 19:52:35 UTC; eval used only `B_evidence`. The eval job saw all 180 existing episodes (227 jobs, room 228), so nothing was truncated. |
| Part-1 selection | Parameters always fit on dev or on the other seed fold. The headline uses the `maxfp0` variants consistently, and `maxfp1` is reported as "lenient". |
| v7 snapshot timing bias | For successes, the latest keyframe before T is 8-21 steps stale, versus 0 for failures. Stale frames make "still at start" more likely, so the bias inflates false aborts. It is conservative for the precision claim. The history rules are correctly flagged `LEAKY_`. |
| Slow-success counts | can 7/113 successes past 8 s and 2 past 12 s; square 12/28 past 8 s and 1 past 12 s; last success at step 282 (can) and 255 (square). Confirmed. |
| Ranking numbers | `square_s1_e0060` 1.0 → 0.5 / 0.6 / 0.7 / 0.8; square Spearman 0.763 / 0.684; regret 0.3; can-run Spearman 0.98-1.00. Confirmed. |
| Cosmos `held` flag is noise | Robust to the truth definition. Against finger contact only, the flag says "held" in 293/572 frames without contact, versus 330/609 under x2's definition; it never says "not held" when the object is held. |
| Square annotator artifact | Confirmed in code: `failure_annotator.py` uses `z0 = obj[0, 2]`. Settled-height re-annotation turns all 11 eval square `other` labels into `no_grasp`, and 1 `no_grasp` into `dropped`. |
| Stability on current data (210 eps, 103 v7 rows) | time-only dev→eval 0.220 (11/150 false aborts), state 0.246 (10/150), cross-fit time 0.184 (6/150), cross-fit state 0.288 (8/150), Cosmos 0.228 vs time 0.187. All within about 1.5 points of the report. |

## 3. Issues found

### 3.1 CIs ignore clustering by checkpoint (independence assumption)

The Wilson and episode-bootstrap CIs treat the 200 episodes as independent. In fact, false aborts
concentrate in a few slow checkpoints: 5 of the 10 dev→eval time-only false aborts are in
`square_s1_e0060`. A checkpoint-cluster bootstrap (resampling checkpoints) widens every interval:

| policy (frozen, n=200 or 180) | saved, reported CI (episode) | saved, cluster CI | false-abort rate, Wilson | false-abort rate, cluster CI |
|---|---|---|---|---|
| time_only dev→eval | 0.223 [0.19, 0.25] | [0.14, 0.30] | 10/141 [0.04, 0.13] | [0.01, 0.16] |
| time_only seed cross-fit | 0.187 [0.16, 0.21] | [0.12, 0.24] | 6/141 [0.02, 0.09] | [0.00, 0.11] |
| state dev→eval | 0.254 [0.21, 0.30] | [0.14, 0.37] | 9/141 [0.03, 0.12] | [0.01, 0.13] |
| state seed cross-fit | 0.300 [0.26, 0.34] | [0.20, 0.38] | 8/141 [0.03, 0.11] | [0.01, 0.13] |
| Nemotron any confidence (180) | 0.287 [0.22, 0.35] | [0.15, 0.42] | 7/121 | [0.02, 0.10] |
| Nemotron high confidence (180) | 0.201 [0.14, 0.26] | [0.10, 0.30] | 1/121 | [0.00, 0.03] |
| Nemotron AUROC at 4 s (180) | 0.85 [0.78, 0.91] | [0.74, 0.91] | | |
| Nemotron AUROC at 4 s, excl. epoch 0 (150) | 0.76 [0.66, 0.85] | [0.68, 0.84] | | |

The excl.-epoch-0 AUROC also pools both tasks. Within task it is lower: can 0.74 (n=110, 17 failures)
and square 0.68 (n=40, 12 failures).

### 3.2 "Vision/state adds X points": paired differences, epoch-0 dependence

These are paired differences on the same episodes, against the time-only rule with the same T2. The
frozen data was used, and the cluster CI resamples checkpoints.

| comparison | Δ saved (all) [episode CI] [cluster CI] | Δ saved excl. epoch 0 [cluster CI] | false aborts (rule vs time) |
|---|---|---|---|
| state vs time, dev→eval | +0.030 [0.001, 0.058] [−0.003, 0.073] | +0.013 [−0.009, 0.038] | 9 vs 10 |
| state vs time, seed cross-fit | +0.113 [0.086, 0.142] [0.073, 0.146] | +0.100 [0.053, 0.143] | 8 vs 6 |
| Cosmos v7 snapshot vs time (can, 90 eps) | +0.045 [0.030, 0.060] [0.012, 0.072] | **+0.015 [0.003, 0.027]** (n=70) | 1 vs 1 |
| Nemotron high-conf + time vs time (180) | +0.107 [0.072, 0.144] [0.035, 0.180] | +0.044 [0.014, 0.081] | 11 vs 10 |
| Nemotron any-conf + time vs time (180) | +0.163 [0.124, 0.204] [0.069, 0.260] | +0.088 [0.044, 0.133] | 12 vs 10 |

- **The Cosmos gain is mostly epoch 0.** 15 of the 19 extra aborts by the Cosmos rule are in the
  two epoch-0 checkpoints, which are 20 of the 90 v7 rows. The report gives the epoch-0 caveat for
  Nemotron but not for Cosmos, and its "Cosmos snapshot about +4.5 points on can" becomes **+1.5
  points** outside epoch 0. Dense privileged `nolift` at the same T1/T2 saves 0.260 vs Cosmos 0.243.
- **"Privileged state adds 3 points (dev→eval)" is not distinguishable from 0** under the cluster
  bootstrap, and it falls to +1.3 points outside epoch 0. The cross-fit +11 points is robust, but it
  comes with 2 more false aborts.

### 3.3 The 4-7% false-abort price and the square top-1 flip come from small fitting sets

x2 fits on 36 dev episodes (3 square successes), or on 5 seeds of the same checkpoints. Refitting the
same rule families (maxfp0) **leave-one-checkpoint-out** on eval uses the other 19 checkpoints'
episodes, a held-out scheme at checkpoint level:

| scheme | saved [cluster CI] | false aborts | square Spearman / top-1 regret | chosen cuts |
|---|---|---|---|---|
| LOCO time_only | 0.162 [0.10, 0.21] | 2/141 [cluster 0.00, 0.04] | 1.00 / 0.0 | can 15 s (14 s for one fold), square 13 s (12 s for one fold) |
| LOCO state_combo | 0.201 [0.12, 0.28] | 2/141 | 1.00 / 0.0 | can: time only; square: nolift 10 s + drop5 + 13 s |
| leave-one-run-out time_only (can only) | 0.137 [0.07, 0.19] | 1/113 | – | 15 s / 14 s |
| leave-one-run-out state_combo (can only) | 0.216 [0.09, 0.35] | 4/113 | can_s2 0.91 | – |

So early stopping with a fit-set-sized safety margin gives 16-20% saving at 1.4% false aborts with no
ranking change on this snapshot. The "4-7%" price and the "can flip the top-1 choice" risk come from the
choice of fitting set; they are not inherent. The flip is also snapshot-fragile: with the newly arrived
`square_s1_e0150` (full 0.9, early-stopped 0.8), the dev→eval time-only square top-1 regret drops from
0.3 to **0.1**. The mechanism is real: slow, good checkpoints are the ones penalised.

### 3.4 Headline wording

- "held-out early-abort rules recover only 19-30% **of it**": the 19-30% is a fraction of **total**
  robot time. That is 34-55% of the 54.4% oracle ceiling, not 19-30% of it.
- The headline quotes Nemotron high confidence as "20.1% with 1/121 false aborts" without the caveat
  that this operating point was **not dev-validated**. On dev, the same operating point was wrong in
  1 of its 3 calls (1/15 dev successes falsely called failures). Also, a time-only cut tuned on the
  same 180 eval episodes to at most 1 false abort (can 14 s, square 13 s) saves 0.185, close to 0.201.
  High confidence alone is therefore not better than a horizon cut. Only its combination with the time
  cut adds +10.7 points (+4.4 points outside epoch 0).
- The Nemotron savings (n=180, 2041 s) and the Part-1 savings (n=200, 2169 s) use different
  denominators but sit side by side in the headline. This is minor.
- The KR summary says square e0060 drops to "0.5-0.7"; the findings say "0.5-0.8". The latter is right.

### 3.5 False failure calls: one detail is wrong

"All 7 false failure calls ... had not yet grasped at the cut." Six of the seven never had finger
contact before the cut. The seventh, `can_s1_e0020` seed 200004, is the **only high-confidence false
abort**:

- it made contact at step 46 and lifted the can 6.9 cm, then lost it before the cut;
- it re-grasped later and succeeded at step 266;
- Nemotron's 4 s reason, "the gripper is open and not touching the red can", is true at the cut;
- at 8 s it answered `will_succeed=true / high`, saying the can was "securely grasped and lifted",
  while state shows the can resting with no contact. The correct answer came from a hallucinated
  observation.

More broadly, the B_evidence evidence fields are unreliable, a point x2 did not analyse:

- `gripper_closed_on_object=true` in 51/95 eval prefixes with no finger contact in the last 3 steps;
- `object_lifted=true` in 38/97 prefixes where the object was below 5 cm at the cut.

### 3.6 Side finding: labels confirmed, implied impact on experiment 2 is nil

The square `other` labels are indeed artifacts. However, re-labelling with the settled height leaves
Nebius v2 failure-type accuracy **unchanged**: 0.382 on the 55 failures currently in
`eval_nebius_v2.jsonl` before and after, and square 1/19 either way. Nebius predicts `not_placed` for
those episodes, which is wrong under both labelings. "Partly mislabelled" is true, but the example
"Nebius failure-type accuracy 0.40" is not affected in practice. x2's count of 3 "lifted but annotated not
lifted" episodes is over dev+eval. On eval, besides the 11 `other` labels, the settled reference changes
1 more failure label (`no_grasp` → `dropped`).

### 3.7 Omissions (context, not errors)

- **Unreported policy.** The dev→eval `proprio_combo` policy (square `noclose` at 3 s + time 9 s)
  saved 0.251 with 12/141 false aborts and is missing from table 5.1. On square eval, `noclose` at
  3 s aborts 10 episodes, and 8 of them are successes. This supports "useless", but the selected
  policy should have been listed.
- **A free saving that is larger than the VLM gains.** Episodes are recorded for 16 steps after
  first success: 2256 steps, or **5.2%** of all eval steps. Stopping on the success signal itself
  saves this with zero false-abort risk. That is more than the Cosmos gain outside epoch 0 (+1.5
  points) and the dev→eval state gain (+3.0 points).
- **Weak dev selection.** A_plain and B_evidence gave identical dev confusion matrices (28/36 same
  binary calls), so the pick rests on an AUROC tie-break with overlapping CIs ([0.40, 0.74] vs
  [0.52, 0.84]). This is legitimate, but weak evidence that B is better.
- **Clip timing mismatch.** `compact_video_bytes` keeps every 2nd frame (`round(10/4) = 2`) and
  writes at 4 fps. The model therefore sees 21 frames played over 5.25 s while the prompt says
  "FIRST 4 seconds". This is the same for all clips, so there is no leakage, but the stated duration
  is off by 1.25x.

## 4. Corrected headline

EXPLORATORY: failing episodes are 54.4% of eval robot time. Held-out abort rules fit on small sets
(dev, or half the seeds) save 19-30% of **total** robot time (34-55% of that ceiling), at 4-7% false
aborts. Checkpoint-clustered CIs are roughly twice as wide as reported. Fitting on the other 19
checkpoints (LOCO) gives 16-20% saved at 2/141 false aborts with no ranking change. Nearly all of the
safe saving is a shorter horizon:

- privileged state adds +3 points dev→eval (CI includes 0) to +11 points cross-fit;
- Cosmos v7 adds +4.5 points on can, but only +1.5 points outside epoch-0 checkpoints;
- Nemotron high-confidence calls (operating point not dev-validated) alone save no more than a tuned
  time cut, but add +10.7 points on top of one (+4.4 points outside epoch 0) for 1 extra false abort.

## 5. Remaining limitations of this verification

- The LOCO refit still fits on eval outcomes of other checkpoints from the same runs. It is held out
  at checkpoint level, but not prospective, and square has a single run.
- The cluster bootstrap resamples whole checkpoints. With 5-9 checkpoints per run the intervals are
  themselves rough.
- No new VLM calls were made. The Nemotron generalisation issues (dev AUROC 0.68 vs eval 0.85) could
  only be re-scored, not re-tested.
