# x2_early_stop: abort a rollout once failure is evident (EXPLORATORY, post-hoc)

> **Label:** exploratory and post-hoc. None of these rules were pre-registered in
> `06_experiment_protocol.md`. All rule parameters were fit on held-out data (the dev split, or
> the other half of the eval seeds). Every number below comes from
> `results/hackathon/explore/x2_early_stop.json`, which was produced by the scripts listed here.
> Data snapshot at analysis time (2026-10-29 ~05:00 KST): 200 eval episodes from 20 checkpoints,
> 90 Cosmos v7 rows (can only), and 299 Nemotron calls. Rollouts and v7 rows are still arriving.
> Re-run the analysis with the command at the bottom.

**One-line summary:** Cutting a rollout off once failure is certain could in theory save up to 54% of robot time. The savings actually obtained on held-out data were 19–30%, at the cost of wrongly aborting 4–7% of successful episodes. Nemotron high-confidence failure calls saved 20% with almost no misjudgments (1/121), but most of those savings came from epoch-0 checkpoints. A time cut can drop the success rate of the top square checkpoint from 1.0 to 0.5–0.7 and so change the top-1 ranking.

## 1. Question

During closed-loop evaluation of a checkpoint, can an episode be stopped once failure is evident?
If so, how much robot or simulator time does that save, and how much does it bias the
per-checkpoint success estimate that PLV uses as its ranking target? Three signal sources were
tested:

1. privileged simulator state (upper bound),
2. Cosmos3-Edge per-keyframe perception (vision only),
3. Nemotron-3.5-Super-VL judging a video *prefix*.

## 2. Why it could help PLV and efficiency

The ground truth for PLV is closed-loop success, and that is the expensive part of the pipeline.
In this eval set, 54.4% of all control steps are spent inside episodes that end up failing (the
oracle ceiling below). The policy horizon is 400 steps (20 s), yet no eval success happened after
step 282 (can) or step 255 (square). On a real robot every aborted second is real time saved.
An abort only matters for ranking when it is a **false abort**: it turns a success into a failure.
Aborting a true failure leaves the estimate unchanged.

## 3. Method

**Online semantics.** A rule may abort an episode at step `a` only if two things hold: the episode
is still running at `a` (it has not reached its last step and has not succeeded), and the rule
uses only information up to `a`. The steps saved by an abort are `last_step - a`. The checkpoint
success estimate under early stopping is the fraction of episodes that succeeded and were not
aborted. 20 sim steps equal 1 s. Video frame k is sim step 2k.

**Height reference.** Heights are measured relative to the object's settled z (median of steps
5..10). The square nut spawns 6 cm above the table and drops in the first 5 steps (see §7).

**Part 1 rules (privileged state plus one deployable proprio rule):**

- `time_only(T)`: abort at T if the episode has not yet succeeded. This is the same as a shorter
  horizon.
- `nolift(T1)`: abort at T1 if the object was never 5 cm above its settled height.
- `drop(k)`: abort when the object, after having been lifted, has been below 2 cm and out of
  finger contact for k consecutive steps.
- `noclose(T1)`: the gripper was never commanded closed by T1. This uses executed actions only,
  so it is deployable.
- Combinations take the minimum abort step over their components. Every component can be
  switched off.

**Held-out selection.** Parameters are chosen per task to maximise steps saved, under the
constraint of at most `maxfp` false aborts (0 or 1) on the fitting set. Ties go to the most
conservative choice. Two fitting schemes:

- (a) fit on the **dev split** (36 episodes, seeds 300000+), apply unchanged to eval;
- (b) **2-fold cross-fit on eval by env seed** (200000–04 and 200005–09).

The training-time seeds 100000–100049 have no state traces, so they could not be used.

**Part 2 (Cosmos3-Edge, `eval_local_v7.jsonl`).** These are task-free perceptions of 8 keyframes
evenly spaced over the **whole** episode. That spacing leaks the outcome: successes are short, so
their keyframes are about 3x denser (mean spacing 19.8 steps for successes vs 57.1 for failures).
History rules such as "never held in any frame by T" therefore look optimistic, and they are
reported only under the label `LEAKY_`. The main rule is a time-matched **snapshot**: at the
failure-keyframe times (2.9, 5.7, ..., 17.1 s), look only at the single most recent keyframe at or
before T, and abort if it shows the object still at its start location (`left_bin` for can,
`on_table` for square). An optional variant also requires `held_by_gripper == false`. The
privileged twin applies the same snapshot to state at the same keyframe. The operating point
(T1, use_held, T2) is chosen on the dev v7 rows (36), then applied to eval.

**Part 3 (Nemotron via Nebius).** Prefixes were cut at **absolute** times of 4 s (step 80) and
8 s (step 160), not at fractions of the episode. Every episode still running at the cut gets a
clip of identical length, so clip length cannot leak the outcome. Episodes that had already
succeeded need no decision and were not sent.

- Clips are re-encoded with imageio into `data/outputs/hackathon_explore/x2_early_stop/clips/`.
  `NebiusVLMClient` then compacts them to 4 fps and 320 px.
- Two prompts were compared on dev at 4 s:
  - `A_plain`: the requested `{will_succeed, confidence, reason}`;
  - `B_evidence`: adds a check-what-you-see preamble and fields for gripper closed, lifted, and
    fell.
- The selection rule was fixed beforehand: highest dev balanced accuracy, with ties broken by
  AUROC. Both prompts tied on balanced accuracy (0.55), and `B_evidence` won on AUROC (0.68 vs
  0.58).
- `B_evidence` was then applied unchanged to eval:
  - at 4 s, all 180 eval episodes that existed when the job started;
  - at 8 s, the 47 episodes still running at 8 s, excluding epoch-0 checkpoints (budget).
- AUROC uses an ordinal score (true/high = 1.0 ... false/high = 0.0).
- Abort policy: abort at the first check where `will_succeed == false`. A `confidence == high`
  variant is also reported, but it was **not** validated on dev: on dev, B's high-confidence
  failure calls were 2 of 3 correct.

**Uncertainty.** Wilson 95% CIs are used for proportions. Percentile bootstrap (2000 resamples
over episodes) is used for the saved fraction and for AUROC.

## 4. Commands run (all from the repo root, Env B: `PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 ~/.venvs/cosmos-plv/bin/python`)

| # | command | exit |
|---|---|---|
| 1 | `scripts/hackathon/explore/x2_early_stop_nebius.py --split dev --variants A_plain,B_evidence --cuts 4 --dry-run` | 0 |
| 2 | `scripts/hackathon/explore/x2_early_stop_nebius.py --split eval --variants A_plain --cuts 4,8 --skip-epoch0-cuts 8 --dry-run` (count only) | 0 |
| 3 | `scripts/hackathon/explore/x2_early_stop_nebius.py --split dev --variants A_plain,B_evidence --cuts 4` (72 calls) | 0 |
| 4 | `scripts/hackathon/explore/x2_early_stop_nebius.py --split eval --variants B_evidence --cuts 4,8 --skip-epoch0-cuts 8 --budget 300` (227 calls) | 0 |
| 5 | `scripts/hackathon/explore/x2_early_stop_analyze.py` → `results/hackathon/explore/x2_early_stop.json` (final run) | 0 |
| 6 | `scripts/hackathon/explore/x2_early_stop_tables.py` (prints the tables below from the JSON) | 0 |

Nebius usage: 299 calls in total, 0 backend errors, 0 parse failures. Mean latency was 2.57 s,
with about 1209 prompt tokens and 72 completion tokens per call. The evidence log is
`results/hackathon/explore/nebius_calls_explore.jsonl` (tags `explore/x2_early_stop/...`). Raw
rows are in `data/outputs/hackathon_explore/x2_early_stop/nebius_prefix_calls.jsonl`.

The GPU was not used. The plan included a fixed-schedule Cosmos run, but `/tmp/claude-1000/gpu_explore.lock`
was held by x6, x5 was queued for about 40 minutes, and free memory was 5.9 GB (below 9 GB).
Part 2 therefore uses only the existing v7 rows.

Development runs of command 5 exposed three bugs before the final run, all fixed in the
committed script:

- the selection was infeasible because of a dev drop-and-recover success, and it degenerated to
  "abort everything" (the grid now allows components to be switched off);
- the square spawn-height artifact made `nolift` never fire and `drop` fire at step 1 (fixed with
  the settled reference);
- a missing `.desc` attribute.

## 5. Results

### 5.1 Headline: held-out abort policies (eval; "saved" = fraction of all control steps)

| policy (how parameters were chosen) | episodes | failures aborted | false aborts / successes [95% CI] | steps saved [95% CI] | robot time saved |
|---|---|---|---|---|---|
| **Oracle ceiling** (abort every failure at step 0) | 200 | 59/59 | 0 | **0.544** | – |
| time_only, dev→eval (can T=280 = 14 s, square T=180 = 9 s) | 200 | 59/59 | 10/141 = 0.071 [0.04, 0.13] | 0.223 [0.19, 0.25] | 484 of 2169 s |
| time_only, seed cross-fit | 200 | 59/59 | 6/141 = 0.043 [0.02, 0.09] | 0.187 [0.16, 0.21] | 405 s |
| state nolift+drop+time, dev→eval | 200 | 55/59 | 9/141 = 0.064 [0.03, 0.12] | 0.254 [0.21, 0.30] | 550 s |
| state nolift+drop+time, seed cross-fit | 200 | 59/59 | 8/141 = 0.057 [0.03, 0.11] | 0.300 [0.26, 0.34] | 651 s |
| proprio noclose+time, seed cross-fit (noclose was switched off in all folds) | 200 | 59/59 | 6/141 = 0.043 [0.02, 0.09] | 0.187 [0.16, 0.21] | 405 s |
| Cosmos v7 snapshot + time, dev→eval (can only) | 90 | 36/36 | 1/54 = 0.018 [0.00, 0.10] | 0.243 [0.20, 0.28] | 266 of 1096 s |
| ↳ time_only, same T2, same 90 episodes | 90 | 36/36 | 1/54 | 0.198 [0.17, 0.22] | 217 s |
| ↳ privileged snapshot twin, same T1/T2 | 90 | 36/36 | 2/54 | 0.283 [0.24, 0.32] | 311 s |
| Nemotron @4 s + @8 s, abort if `will_succeed=false` | 180 | 35/59 | 7/121 = 0.058 [0.03, 0.12] | 0.287 [0.22, 0.35] | 586 of 2041 s |
| Nemotron, abort only if false **and** high confidence (not dev-validated) | 180 | 27/59 | 1/121 = 0.008 [0.00, 0.04] | 0.201 [0.14, 0.26] | 410 s |
| ↳ same, **excluding epoch-0 checkpoints** | 150 | 10/29 | 1/121 = 0.008 [0.00, 0.04] | 0.096 [0.05, 0.15] | 138 of 1441 s |
| Nemotron (any confidence) + time_only dev T2 | 180 | 59/59 | 12/121 = 0.099 [0.06, 0.17] | 0.400 [0.35, 0.45] | 817 s |
| Reference: state nolift/drop rule checked only at 4 s and 8 s | 180 | 53/59 | 20/121 = 0.165 [0.11, 0.24] | 0.469 [0.41, 0.52] | 957 s |

### 5.2 Part 1: fixed rules vs check time (eval; descriptive, no fitting)

| task | rule | T | aborted | failures caught | false aborts | precision [CI] | recall [CI] | saved |
|---|---|---|---|---|---|---|---|---|
| can | time_only | 6 s | 65 | 37/37 | 28 | 0.57 [0.45, 0.68] | 1.00 | 0.394 |
| can | time_only | 8 s | 44 | 37 | 7 | 0.84 [0.71, 0.92] | 1.00 | 0.317 |
| can | time_only | 10 s | 41 | 37 | 4 | 0.90 [0.78, 0.96] | 1.00 | 0.258 |
| can | time_only | 12 s | 39 | 37 | 2 | 0.95 [0.83, 0.99] | 1.00 | 0.203 |
| can | time_only | 15 s | 37 | 37 | 0 | 1.00 [0.91, 1.00] | 1.00 | 0.125 |
| can | nolift | 4 s | 36 | 27 | 9 | 0.75 [0.59, 0.86] | 0.73 [0.57, 0.85] | 0.325 |
| can | nolift | 8 s | 28 | 26 | 2 | 0.93 [0.77, 0.98] | 0.70 | 0.218 |
| can | nolift | 12 s | 26 | 26 | 0 | 1.00 [0.87, 1.00] | 0.70 | 0.140 |
| can | nolift+drop10 | 4 s | 46 | 35 | 11 | 0.76 [0.62, 0.86] | 0.95 [0.82, 0.98] | 0.422 |
| can | nolift+drop10 | 8 s | 39 | 35 | 4 | 0.90 [0.76, 0.96] | 0.95 | 0.324 |
| can | nolift+drop10 | 12 s | 37 | 35 | 2 | 0.95 [0.82, 0.98] | 0.95 | 0.246 |
| can | noclose | 4 s | 1 | 0 | 1 | 0.00 | 0.00 | 0.005 |
| square | time_only | 8 s | 34 | 22/22 | 12 | 0.65 [0.48, 0.79] | 1.00 | 0.433 |
| square | time_only | 10 s | 27 | 22 | 5 | 0.81 [0.63, 0.92] | 1.00 | 0.337 |
| square | time_only | 12 s | 23 | 22 | 1 | 0.96 [0.79, 0.99] | 1.00 | 0.259 |
| square | time_only | 15 s | 22 | 22 | 0 | 1.00 [0.85, 1.00] | 1.00 | 0.161 |
| square | nolift+drop10 | 5 s | 21 | 18 | 3 | 0.86 [0.65, 0.95] | 0.82 [0.61, 0.93] | 0.422 |
| square | nolift+drop10 | 10 s | 17 | 17 | 0 | 1.00 [0.82, 1.00] | 0.77 [0.57, 0.90] | 0.266 |
| square | noclose | 4 s | 0 | 0 | 0 | – | 0.00 | 0.000 |

The complete curves (every 1 s, plus `nolift` alone for square) are in
`part1_state.curves_eval` of the JSON.

### 5.3 Part 2: Cosmos3-Edge v7 keyframes (eval; can only; 90 episodes from 9 checkpoints)

Frame-level perception vs state, over all keyframes:

- `held_by_gripper` is unreliable. When the object was **not** lifted and grasped, Cosmos still
  said "held" in 330/609 frames.
- `object_location == start location` is precise but incomplete. It never fired when the object
  had been lifted or the task had succeeded (0/227 frames). It caught 332/493 frames in which the
  object really was still at its start (recall 0.67).

| T | staleness of latest keyframe, succ/fail (steps) | rule | aborted | TP | FP | precision [CI] | recall [CI] | saved |
|---|---|---|---|---|---|---|---|---|
| 2.9 s | 8.1 / 0 | vis_snapshot_loc | 31 | 21 | 10 | 0.68 [0.50, 0.81] | 0.58 [0.42, 0.73] | 0.366 |
| 2.9 s | | vis_snapshot_loc_and_not_held | 15 | 14 | 1 | 0.93 [0.70, 0.99] | 0.39 [0.25, 0.55] | 0.224 |
| 2.9 s | | state snapshot (same frame) | 86 | 33 | 53 | 0.38 [0.29, 0.49] | 0.92 | 0.712 |
| 5.7 s | 9.3 / 0 | vis_snapshot_loc | 29 | 26 | 3 | 0.90 [0.74, 0.96] | 0.72 [0.56, 0.84] | 0.359 |
| 5.7 s | | state snapshot (same frame) | 39 | 33 | 6 | 0.85 [0.70, 0.93] | 0.92 [0.78, 0.97] | 0.463 |
| 5.7 s | | time_only | 61 | 36 | 25 | 0.59 [0.47, 0.70] | 1.00 | 0.525 |
| 8.6 s | 9.2 / 0 | vis_snapshot_loc | 23 | 21 | 2 | 0.91 [0.73, 0.98] | 0.58 [0.42, 0.73] | 0.229 |
| 8.6 s | | state snapshot (same frame) | 37 | 34 | 3 | 0.92 [0.79, 0.97] | 0.94 | 0.367 |
| 11.4 s | 21.0 / 0 | vis_snapshot_loc | 19 | 19 | 0 | 1.00 [0.83, 1.00] | 0.53 [0.37, 0.68] | 0.149 |
| 5.7 s | (leaky, history) | LEAKY_vis_nohold_history | 11 | 11 | 0 | 1.00 [0.74, 1.00] | 0.31 | 0.144 |

On dev, the selected operating point was: can, snapshot at 11.4 s (location only) plus a time cut
at 14 s; square, time cut at 9 s. There are no square v7 eval rows yet.

### 5.4 Part 3: Nemotron "will this attempt succeed?" on absolute-time prefixes

| split | prompt | cut | subset | n | succ/fail | accuracy [CI] | bal. acc | AUROC [CI] | precision of "fail" [CI] | recall of "fail" [CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| dev | A_plain | 4 s | all | 36 | 15/21 | 0.50 [0.34, 0.66] | 0.55 | 0.58 [0.40, 0.74] | 0.71 [0.36, 0.92] | 0.24 [0.11, 0.45] |
| dev | B_evidence | 4 s | all | 36 | 15/21 | 0.50 [0.34, 0.66] | 0.55 | 0.68 [0.52, 0.84] | 0.71 [0.36, 0.92] | 0.24 [0.11, 0.45] |
| eval | B_evidence | 4 s | all | 180 | 121/59 | 0.80 [0.74, 0.85] | 0.72 | **0.85 [0.78, 0.91]** | 0.83 [0.67, 0.92] | 0.49 [0.37, 0.62] |
| eval | B_evidence | 4 s | can | 130 | 93/37 | 0.88 [0.81, 0.92] | 0.81 | 0.86 [0.78, 0.93] | 0.89 [0.72, 0.96] | 0.65 [0.49, 0.78] |
| eval | B_evidence | 4 s | square | 50 | 28/22 | 0.60 [0.46, 0.72] | 0.56 | 0.72 [0.58, 0.84] | 0.62 [0.31, 0.86] | 0.23 [0.10, 0.43] |
| eval | B_evidence | 4 s | excl. epoch 0 | 150 | 121/29 | 0.81 [0.74, 0.87] | 0.60 | **0.76 [0.66, 0.85]** | 0.54 [0.29, 0.77] | 0.24 [0.12, 0.42] |
| eval | B_evidence | 8 s | running, epoch>0 | 47 | 18/29 | 0.55 [0.41, 0.69] | 0.63 | 0.68 [0.56, 0.79] | 0.90 [0.60, 0.98] | 0.31 [0.17, 0.49] |

Fraction of each true outcome type that Nemotron called a failure at 4 s:

| true outcome | called "fail" at 4 s |
|---|---|
| `no_grasp` | 0.79 |
| `dropped` | 0.20 (the drop usually happens later) |
| square `other` | 0.27 |
| success (`none`) | 0.05 |

All 7 false failure calls on eval (6 at 4 s, 1 at 8 s) were late successes, with first success
at steps 167–282, that had not yet grasped at the cut. The model wrote, for example, "the task
cannot succeed within the remaining time". Of the 28 high-confidence failure calls:

| where | count | all true failures? |
|---|---|---|
| 4 s, epoch-0 checkpoints | 17 | yes |
| 4 s, other checkpoints | 3 | no: 2 failures, 1 success |
| 8 s | 8 | yes |

### 5.5 Does early stopping change the checkpoint success estimate and ranking?

Only false aborts change the estimate, and they can only lower it. Under the strict (maxfp0)
held-out policies and the Nemotron policies, 3–7 of 18–20 checkpoints change, by at most 0.2–0.5
in success rate. The lenient (maxfp1) variants change up to 0.6. The Nemotron high-confidence
policy changes only 1 checkpoint, by 0.1.

The within-run Spearman correlation between the early-stopped and the full 10-episode success
estimates stays at 0.97–1.00 for both can runs under the strict and Nemotron policies, and drops
to 0.90 under the lenient ones. Square is fragile. The truly best square
checkpoint, `square_s1_e0060` (10/10), has slow successes, so time cuts drop it to:

- 0.5 (time_only dev→eval);
- 0.6 (time_only cross-fit);
- 0.7 (state rule, dev→eval);
- 0.8 (Nemotron, any confidence).

Under the first two, it falls to or below `e0090`/`e0120` (full 0.7 each), which gives a
pessimistic top-1 regret of 0.3 and square Spearman values of 0.68–0.76.

## 6. Interpretation

1. **Failures are rarely "evident" early with a safety margin.** Many eventual successes are
   slow (regrasp or retry), and the best square checkpoint is among them:
   - can: 7/113 successes are still running at 8 s, and 2 at 12 s;
   - square: 12/28 are still running at 8 s, and 1 at 12 s (`time_only` false aborts in 5.2).

   Any rule that fires before about 12–15 s therefore pays a false-abort rate of a few percent. Held-out
   policies saved **19–30%** of robot time, against an oracle ceiling of 54%, at false-abort rates
   of **4–7%**.
2. **Most of the safe saving is just a shorter horizon.** The time-only cut (no perception at all)
   captures 19–22%. Privileged state adds 3 points (dev→eval, 9 vs 10 false aborts) and 11 points
   (cross-fit, 8 vs 6 false aborts), mostly by catching never-lifted failures early.
3. **Proprio-only "gripper never closed" is useless here.** By 4 s, only 1 of 200 episodes had
   never commanded the gripper closed (`noclose` at 4 s).
4. **Cosmos3-Edge snapshot** ("object still in the left bin at time T") is about as precise as the
   privileged snapshot, but it misses about a third of the at-start frames. Its `held` flag is
   noise. On the same 90 can episodes, the dev-selected vision rule saved 24.3%, compared with
   19.8% for the time cut alone, at the same single false abort. The gain is small, rests on n=90
   with can only, and is still slightly affected by keyframe-timing differences.
5. **Nemotron prefix judgement** at 4 s works for the obvious "never grasped" failures
   (AUROC 0.85, but 0.76 excluding epoch 0) and does not work for drops. Dev AUROC was much lower
   (0.68). The model is optimistic and, without a horizon, it believes a missed grasp is
   unrecoverable. Its high-confidence failure calls are almost always right (27 of 28 aborts
   correct), but outside epoch 0 they cover only 10/29 failures and save 9.6%.
6. **For PLV efficiency**, the most useful direction is **horizon tuning per task with margin**,
   optionally plus a high-precision VLM or state veto. A VLM gate is not a large win in this data.
   Any early-stop policy must be checked for top-checkpoint bias, because slow but good
   checkpoints are the ones penalised.

## 7. Side finding (structural risk, not fixed)

`cosmos_plv/failure_annotator.py` uses the object height at **step 0** as its reference. For
square, the nut spawns at 0.89 m and settles to 0.83 m within 5 steps (medians over 62 square
dev+eval episodes). As a result:

- all 15 square episodes labelled `other` ("fell below start") end within 1 cm of their settled
  height, so they are really untouched or pushed (`no_grasp`-like) failures;
- 3 square episodes annotated as not lifted were in fact lifted 5 cm or more above the settled
  height.

The square failure-type ground truth used in experiment 2 (for example, Nebius failure-type
accuracy 0.40) is therefore partly mislabelled. Recorded in
`side_check_square_annotation`. The annotator was not modified (read-only constraint).

## 8. Limitations

- Small samples: 10 episodes per checkpoint, 59 failures, 3 runs, square from 1 run only. The
  dev split has 36 episodes (only 3 square successes, all by step 171), which made the dev-chosen
  square time cut (9 s) unsafe on eval.
- Cosmos part: can only, 90 episodes from 9 checkpoints. v7 keyframe timing depends on episode
  length. The snapshot rule mitigates this, but successes' latest frames are still 8–21 steps
  older than failures'. No fixed-schedule Cosmos run was possible (GPU lock busy).
- Nemotron: one prompt family, two cut times. 8 s covers only 47 running episodes (no epoch 0).
  The 20 new eval episodes from the two most recent checkpoints were not sent (budget). The
  high-confidence operating point was observed on eval, not validated on dev.
- Only steps are counted. Reset time per episode and VLM latency (2.6 s per call, could run
  asynchronously) are not included.
- The ranking effect is measured against the 10-episode eval success, not against the 50-episode
  training-time `test/mean_score`.

## 9. What to try next

1. A fixed-schedule Cosmos run: one frame every 1–2 s up to 8 s, through the existing
   perception prompt, under the GPU lock. This gives a leak-free vision-only curve and covers
   square too.
2. Per-task horizon tuning with an explicit safety margin, plus a "top-k guard": never early-stop
   episodes of the current top-k checkpoints. Evaluate it with the 50-seed training-time
   outcomes as the ranking target.
3. A Nemotron prompt that states the remaining time and that regrasps happen, or that asks
   "is the object irrecoverable?" rather than "will it succeed?". Use it only as a
   high-precision veto.
4. Checkpoint-level sequential stopping (stop sampling seeds for clearly bad checkpoints) using
   the 50 per-episode training-time outcomes, fit on seeds 100000–24 and tested on 100025–49.
   This is likely a bigger efficiency lever than within-episode aborts.
5. Fix the square spawn-height reference in the annotator (owner decision; this would change
   experiment-2 truth labels).

## Files

- Scripts: `scripts/hackathon/explore/x2_early_stop_nebius.py`, `x2_early_stop_analyze.py`,
  `x2_early_stop_tables.py`
- Results: `results/hackathon/explore/x2_early_stop.json`; Nebius evidence
  `results/hackathon/explore/nebius_calls_explore.jsonl`
- Scratch: `data/outputs/hackathon_explore/x2_early_stop/` (clips, `nebius_prefix_calls.jsonl`,
  logs, `tables.md`)
- Re-run on more data (CPU, about 10 s):
  `PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/x2_early_stop_analyze.py && PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/x2_early_stop_tables.py`.
  Nemotron calls for new episodes: re-run command 4 with a new `--budget` (it is resumable and
  skips rows already done).
