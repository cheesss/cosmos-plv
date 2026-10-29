# Final claim audit

This is the readable version of [`submission/claim_ledger.csv`](../../submission/claim_ledger.csv).
Every value below was recomputed from the raw result rows, or read from the result JSON where only a
summary exists, by [`scripts/hackathon/build_claim_ledger.py`](../../scripts/hackathon/build_claim_ledger.py).
The prose was not used as a source.

* **Snapshot:** 2026-10-29 14:52 UTC. The README and `configs/hackathon/demo_plan.json` were being edited
  while this audit ran, so line numbers can move. The script finds every quoted wording again on each
  run and prints the current `file:line`.
* **Regenerate:** `source scripts/hackathon/envb.sh && python scripts/hackathon/build_claim_ledger.py`.
  It needs CPU only and makes no network calls. It writes the CSV, then prints any mismatch between a
  recomputed value and its result file, and every document inconsistency that is still present.
* **Result at the snapshot:**
  * 48 ledger rows.
  * **0 recompute mismatches** against the result files.
  * The inconsistencies are listed at the end of this page.

Status values: **primary** = pre-registered Experiments 1–3; **secondary** = descriptive or
robustness; **exploratory** = post-hoc studies, or pre-registered exploratory audits (r1, c1, s1,
s2); **historical** = the prior paper, not re-run.

## Ledger (readable)

### Historical PLV paper (prior work, not reproduced)

Source: `docs/plv_arxiv/main.tex` (l.43, l.199, l.217, l.219). This is not a `results/hackathon` file.

| claim | value from source | note |
|---|---|---|
| Study size | 1,365 checkpoints from 45 runs (BC-RNN + Diffusion Policy), 50 rollouts per checkpoint | retrospective, not preregistered |
| Diffusion Policy cohort | PLV-1 ρ 0.556 vs Action L1 0.462; Regret@30 0.006 vs 0.017 | **24 DP runs**, not 45 |
| Pooled open-loop smoothness | Regret@30 0.019 | pooled over all 45 runs |

### Experiment 1: baseline reproduction (primary; protocol 06 with amendments)

* **Pool:** 93 checkpoints in 3 runs (can_seed1 41, can_seed2 21, square_seed1 31).
* **Target:** training-time success from 50 rollouts per checkpoint (4,650 rollouts).
* **Statistic:** oriented within-run Spearman, then the unweighted mean of the 3 runs. There is no CI
  for the mean; each run has a checkpoint-bootstrap CI.
* **Recomputed from:** `baseline_metrics.csv` with scipy. It matches `baseline_ranking.csv` exactly.

| metric | mean ρ | per run (CI) | late half | Regret@30 |
|---|---|---|---|---|
| epoch (free baseline) | **0.815** | 0.78 [0.55, 0.91], 0.88 [0.65, 0.98], 0.79 [0.52, 0.93] | 0.33 | 0.0067 |
| replan overlap | 0.801 | 0.78, 0.86, 0.76 | 0.24 | 0.0067 |
| open-loop smoothness | 0.788 | 0.76, 0.85, 0.75 | 0.26 | 0.0067 |
| boundary jump | 0.782 | 0.77, 0.85, 0.73 | 0.12 | 0.0067 |
| Action L1 | 0.772 | 0.75, 0.83, 0.73 | 0.29 | 0.0067 |
| PLV-1 | 0.759 | 0.73, 0.76, 0.79 | 0.20 | 0.0067 |
| validation loss | −0.481 | −0.46, −0.48, −0.51 | −0.33 | 0.0467 |

* **Regret@30 ≈ 0.007:**
  * The value is (0 + 0 + 0.02)/3, in success-fraction units.
  * Only square_seed1 is non-zero, by one 50-rollout step.
  * k = ceil(0.3N) = 13 / 7 / 10.
* **Replanning tracks training:** ρ(−replan overlap, epoch) is 1.000 / 0.988 / 0.969, and boundary
  jump is 0.991 / 0.977 / 0.956. H10 equals H16 in every run.

### Experiment 2: VLM success judging (primary)

* **Episodes:** 330 = 33 checkpoints × 10 seeds (200000–200009). 263 succeeded and 67 failed.
* **Recomputed from:** `cosmos_diagnostics.csv`, plus the raw jsonl files for the padded run and for v7.
* **CIs:** Wilson, with episodes treated as independent.

| judge | accuracy [95% CI] | balanced | FP (of 67) | FN (of 263) | Can / Square | latency |
|---|---|---|---|---|---|---|
| always "success" (base rate) | 0.797 [0.750, 0.837] | 0.500 | 67 | 0 | | |
| Nemotron-3.5-Super-VL (Nebius, v2) | **0.964** [0.938, 0.979] (318/330) | 0.927 | 9 (13%) | 3 | 0.977 / 0.936 | 3.09 s |
| Cosmos3-Edge (local, v6) | **0.918** [0.884, 0.943] (303/330) | 0.832 | 21 (31%) | 6 | 0.964 / 0.827 | 2.08 s |
| Nemotron, length-neutral (pad201) | 0.961 [0.934, 0.977] (317/330) | 0.909 | 12 | 1 | 0.973 / 0.936 | 3.58 s |

* **Length-neutral check:** 5 of 330 verdicts flip. The first-250 snapshot quoted in 07 and 11
  (4/250 flips; 0.956 in both conditions; FP 8→10, FN 3→1) is internally correct but superseded.
* **Failure types on the 67 failures:**
  * Nemotron 0.313. Length-neutral Nemotron 0.358. Cosmos v7 0.209.
  * A constant `no_grasp` guess scores 0.567 (38/67). Every judge is below it.
  * How Nemotron labels the 38 missed grasps: 20 `no_grasp`, 18 `not_placed`.
  * How it labels the 28 drops: 15 `not_placed`, 8 `none`, 4 `no_grasp`, 1 `dropped`.
* **Agreement:**
  * The judges agree on 309/330 episodes (0.936).
  * When they agree they are right on 300/309 (0.971 [0.946, 0.985]).
  * They disagree on 21 episodes, and Nemotron is right in 18 of them.
* **Dev split (36 episodes):** Cosmos 0.806, Nemotron 0.861.
* **Goal-prompted Cosmos v1:** it called 12/12 dev episodes successful, of which 3 had succeeded.
  * 9 outputs parsed as success.
  * 3 outputs were unparsable JSON whose text also says success.
  * Source: `data/outputs/hackathon_cosmos/dev_prompt_selection.jsonl`.

### Experiment 3: hybrid screening (primary)

Recomputed from `hybrid_screening.csv`; it matches `hybrid_ranking.csv`. The data are 33 checkpoints
(11 per run). The target is 50-rollout success. Pick regret = the best success in the run minus the
success of the single pick.

| score | mean ρ | pick regret (mean of 3 runs; per run) |
|---|---|---|
| latest epoch (free) | 0.854 | **0.0067** (0, 0, 0.02) |
| two-stage: top-30% Action L1 → Nemotron-judged K=10 | — | **0.040** (0.02, 0, 0.10); the Cosmos-judged variant is also 0.040 |
| Nemotron-judged K=10 | 0.743 | 0.060 |
| Cosmos-judged K=10 | 0.651 | 0.080 |
| privileged sim K=10 | 0.607 | 0.087 |
| Action L1 | 0.731 | 0.013 |

The two-stage rule does not beat the epoch pick: 0.040 vs 0.007. Regret@30 is 0.0067 for both.

### Exploratory efficiency claims: baseline, target, unit, prior

| claim as written | what was actually measured | baseline | target / unit | prior doing the work |
|---|---|---|---|---|
| "2–3x fewer rollouts" (x1) | Uniform/top-two-Thompson ratio of the budget needed to reach the target: **median 3, p10 2, p90 4** (B=100 seed bootstraps). Uniform reaches it at a median of 6N, TTTS at 2N. Successive halving: ratio median 2. Random splits: perfect labels 12N vs 3N; Nemotron-judged 12N vs 4N. | uniform (round-robin) allocation, no prior | budget = total rollouts / N checkpoints (mean rollouts per checkpoint) at which **excess regret over the full-pool floor (25 rollouts each) falls to ≤ 0.02**; regret on 25 held-out training-time seeds; **perfect simulator labels**; frozen **56-checkpoint snapshot** (22/18/16) | **none** in the 2–3x comparison; x1 never tested a PLV prior. With the free latest-epoch top-30% prior, TTTS reaches the target at the smallest budget tested (1N). y1: PLV priors beat the epoch prior in **0/336** cells. |
| "16–20% robot time" (x2) | Refitted leave-one-checkpoint-out with zero false aborts allowed on the other 19 checkpoints. **Time-only horizon cut** (Can 300 steps = 15 s, Square 260 steps = 13 s): **16.2%** [0.10, 0.21] at 2/141 false aborts. **Time + privileged state** (Square: no lift by 10 s, drop rule): **20.1%** [0.12, 0.28] at 2/141. Ranking unchanged. | running every episode to its recorded end | fraction of **total recorded eval robot steps** saved; 200 episodes from 20 checkpoints (snapshot); oracle ceiling (all failing time) 54.4% | **none** (no PLV or epoch prior). The 20% end needs privileged simulator state. |
| "16% fewer Nebius calls, 0.952 vs 0.964" (x0) | 278/330 calls = 84.2% (15.8% fewer); accuracy 0.952 [0.923, 0.970]; FN 8 vs 3; FP 8 vs 9; mean latency 4.57 s vs 3.09 s | Nemotron on every episode (330 calls) | **Nebius calls per episode** on the 330-episode eval split | none. The rule is deterministic: accept a local "failure", ask Nemotron otherwise. Not independently verified. |

### E4 / E4b: Cosmos world model (exploratory)

* **E4 is retracted.**
  * It ran 385 generations under 7 conditions through the `libero` slot (slot 5), which is untrained.
  * Trained slots: 1, 2, 3, 6, 7, 8, 12, 13, 15, 20 (10 slots). Slot 30 (`robocasa`) is also untrained.
  * Source: `e5_domain_weights.json`, `e4_cosmos_fd.json`.
* **E4b S3 (trained slot 8):**
  * Data: 15 moving clips of 22 (K=16), 5 initial-state families.
  * Drift-cancelled (main − reversed) cosine with the true motion: **0.776 / 0.792 / 0.845** for the
    Farneback / DIS / template trackers. Clip bootstrap CIs: [0.50, 0.93], [0.59, 0.92], [0.65, 0.98].
  * Mean of the 5 family means: 0.670 / 0.740 / 0.841.
  * **t(4) lower bounds 0.153 / 0.406 / 0.630.** Exact sign-flip p 0.0625 / 0.031 / 0.031, where 1/32
    is the smallest p possible with 5 families.
* **What weakens E4b:**
  * **Motion gain** is 0.389 / 0.390 / 0.351, i.e. about 0.4x the true motion.
  * **Can vs Square:** cosine 0.93 / 0.91 / 0.95 on Can vs 0.47 / 0.55 / 0.63 on Square.
  * **End-point error:** on Can it is 15.9–18.3 px vs 43.6–44.7 px with zero actions. On Square it is
    45.1–47.3 px, *worse* than 38.9–40.3 px with zero actions.
  * **PSNR vs a frozen frame:** 19.88 vs 20.42 dB (start 16: +0.69 dB; start 48: −1.77 dB).
  * **Action effect vs seed floor** at matched starts: 17.4–19.9 vs 9.1–10.3 px.
* **E4b S0 / S1:**
  * S0: PSNR vs the model-card output is 30.8 dB, against **26.2 dB** between seeds. There is one seed
    pair, so not a 26–27 range.
  * S1: best-rotation cosine 0.87–0.94, next-best 0.43–0.53, motion 14.1–15.5 px.
* Source: `e5_probe.json`.

### r1: counterfactual replanning (exploratory, pre-registered with amendments)

* **Pre-registration:** A1.1–A1.4 were added before the first eval call, and an interim look at
  309/400 events is disclosed.
* **Verification:** 4 verifier lenses.
* **Recomputed from:** `r1_counterfactual_replan_eval_events.csv`. It matches the JSON.

| claim | value |
|---|---|
| Design | 400 events: 300 HIGH (max-d boundary per episode, 30 checkpoints) + 100 CONTROL. Outcomes: HIGH 30 helpful / 22 harmful / 248 neutral; CONTROL 11 / 8 / 81 |
| **Primary** (AUROC, helpful vs rest, HIGH n=300) | J1b **0.404** [0.340, 0.477]; d **0.790** [0.710, 0.861]; J1b − d **−0.387** [−0.465, −0.303] → negative. J1 0.491, Cosmos 0.543, privileged 0.489 (raw 0.315) |
| Excluding the 77 decided events (n=223, post hoc) | J1b **0.486** [0.423, 0.554]; d **0.745** [0.660, 0.823]; diff −0.259 [−0.340, −0.172]; Cosmos 0.540; J1 0.495 |
| Localization (post hoc) | 100 paired episodes: HIGH **18** vs CONTROL **19** consequential (discordant 11/12, McNemar p 1.0); overall 52/300 vs 19/100 (Fisher p 0.76); d predicts recorded episode failure at AUROC 0.867 |
| Permutation null | consequential 52 vs 50.2 expected, **P 0.262**; helpful 30 vs 25.0, P 0.108 |
| Steps to success (pre-registered secondary) | n=135: NEW faster **80** vs OLD **51** (4 ties), sign test **p 0.014**, 3.45 steps saved; d AUROC 0.47 |
| J2 two-path choice | abstained 29/52; on the 23 decided events **0.609 vs 0.696** for always-new |

### c1: coverage shift (exploratory, pre-registered with amendment A1)

* **Data:** 620 Can episodes from 31 checkpoints:
  * M: 5 checkpoints, 60 episodes.
  * U: 10 checkpoints, 160 episodes.
  * T: 16 checkpoints, 400 episodes.
* **Failures:** 74.
* **Recomputed from:** `c1_coverage_shift_episodes.csv`.

| claim | value |
|---|---|
| Nemotron false successes | M **6/25**, U **14/27**, T **13/22**; Cosmos 8/25, 14/27, 19/22 |
| Primary AUROC of q | **0.673**, checkpoint-cluster [0.556, 0.750]; post hoc: seed-cluster [0.399, 0.922], two-way [0.354, 0.950]; Fisher T vs M p 0.019 |
| Where the 33 false successes ended | 32 in the wrong goal-bin compartment (of 33 such failures) + 1 on a rim/divider (of 1); 0 of the 40 failures that ended elsewhere. Both judges rejected all 21 no-grasp failures |
| v2c (compartment stated in the prompt) | still accepts **24/33** wrong-compartment cans |
| Precision of claimed success | M **0.854** (35/41), U **0.898** (123/137), T **0.965** (362/375) |
| Black videos | **16** (T stratum, seed 600003, all true successes); Nemotron called 0 of them success, Cosmos 16; Nemotron sensitivity on valid videos 0.981 |

There are two different "32 of 33" statements, and both are correct:
* 32 of the 33 *wrong-compartment failures* were accepted (README:22).
* 32 of the 33 *false successes* were wrong-compartment cans (09:172).

### s1 and s2: self-training (exploratory, pre-registered)

* **s1 (200 paired seeds):**
  * Base 0.885. The arms scored: continue 0.88, nemotron 0.875, oracle 0.89, all 0.88, v2c 0.875.
  * Nemotron − continue −0.005 [−0.040, +0.030].
  * Oracle − continue +0.010 [−0.025, +0.045].
* **s2 (500 paired seeds 810000–810499):**
  * The arm rates and paired differences were recomputed from the eval summaries.
  * The CIs are the JSON's 10,000-sample seed bootstrap.

| arm | success | vs continue [95% CI] | discordant |
|---|---|---|---|
| base B2 | 0.656 | | |
| continue | 0.856 | | |
| human (+36 demos) | **0.938** | **+0.082** [+0.052, +0.114] | 55/14 |
| oracle | 0.824 | −0.032 [−0.070, +0.006] | 40/56 |
| nemotron | **0.752** | **−0.104** [−0.148, −0.060] | 39/91 |
| nemotron − oracle (post hoc) | | −0.072 [−0.112, −0.032] | 33/69 |

* **What Nemotron let into the training data:**
  * It accepted **58 of 149** failures (precision 0.812, recall 1.0).
  * 44 of those cans were in the wrong compartment and 14 ended elsewhere.
  * The accepted failures are 23,200 of 58,250 self frames, i.e. **0.398**.
* **Post-hoc "46 vs 19":**
  * This counts eval failures resting over the goal-bin footprint (nemotron vs oracle; discordant 34/7).
  * Strictly inside a wrong compartment, the counts are 31 vs 12.
  * Recomputed here from the episode npz files.

#### Resolving the s2 base number (0.58 vs 0.656)

All three numbers describe the same checkpoint,
`data/outputs/hackathon_pool_lowdata/paper_dp_can_abs_ph_20_percent_seed1_e1000_r25/checkpoints/epoch=0125-test_mean_score=0.580.ckpt`.
They come from three different samples:

| definition | value | sample |
|---|---|---|
| **Training-time success** (filename `test_mean_score`, train log `logs.json.txt` epoch 125) | **0.580** = 29/50 [0.442, 0.706] | 50 env-runner rollouts, seeds 100000–100049. This is the "about 0.58" from earlier messages, and the protocol's "Training-time success: 0.58". |
| **Collection success** (episodes recorded for self-training) | **0.6275** = 251/400 [0.579, 0.673] | shard a 131/200 = **0.655** (seeds 710000–710199); shard b 120/200 = **0.600** (seeds 710200–710399) |
| **Fresh paired eval success** | **0.656** = 328/500 [0.613, 0.696] | the same 500 eval seeds (810000–810499) as every fine-tuned arm. This is the docs' 0.656, and the only base number comparable with the arm results. |

Shard a (0.655) is numerically close to the eval rate (0.656), but it is a different sample. Quote
0.656 only as the 500-seed eval success. s1 has the same pattern: its base is 0.860 in the filename,
0.86 in collection (172/200) and 0.885 in the fresh 200-seed eval.

### Nebius usage (estimate, not billing)

Counted from `results/hackathon/nebius_calls.jsonl` and
`results/hackathon/explore/nebius_calls_explore.jsonl`. The total cross-checks with
`make_figures._nebius_total()`.

* **Log cutoff:** last logged call 2026-10-29T14:39:41Z, in the main log (a dashboard call). The
  exploratory log ends at 2026-10-29T12:07:51Z. The first call was 2026-10-28T18:46:17Z.
* **Calls:** **7,004** = 775 main + 6,229 exploratory, with **0 failed** (`error` is null in every row).
  The log grew from 7,003 to 7,004 *during this audit*: a dashboard call was logged at 14:39:41Z.

| served model | calls | input tokens | output tokens | catalog price per 1M (in / out) | estimate |
|---|---|---|---|---|---|
| nvidia/Nemotron-3.5-Super-VL | 6,873 | 14,309,035 | 700,542 | $0.30 / $0.90 | $4.92 |
| nvidia/Nemotron-3_5-Lightning | 131 (118 y4 report calls + 13 verifier calls) | 269,930 | 81,603 | $0.06 / $0.24 | $0.04 |
| **total** | **7,004** | **14,578,965** | **782,145** | | **$4.96** |

* If every call were priced at the Super-VL rate, the total would be $5.08.
* Cached-input discounts are not applied.
* The billed amount is whatever the Nebius console shows.

### Other rows in the ledger

* **y1 (PLV prior):** 0 of 336 cells beat the epoch prior. The PLV top-30% sets overlap the epoch
  top-30% sets by 84.6–100% (minimum per run).
* **y3 (camera):**
  * Data: 60 replayed episodes from 22 checkpoints.
  * Wrist camera balanced accuracy: Cosmos 0.493, Nemotron 0.507 (agentview 0.799 / 0.896).
  * Raw accuracy change under the image degradations: −0.083 to +0.050. The docs' |Δ| ≤ 0.03 is after
    re-weighting to the eval mix.
* **y4 (LLM reports):** 5 of 16 randomly audited reports contained a clear false statement
  ([0.142, 0.556]). The automatic checker caught 2 of the 5.
* **Demo video:**
  * The committed `media/hackathon/demo.mp4` is **143.2 s (2:23), 1600x992** (ffprobe).
  * The current `demo_plan.json` sums to **153.0 s** (91.0 s of real footage).
  * `submission/demo_final.mp4` was still being written and could not be read at the snapshot.

## Inconsistencies found

The line numbers are as of the snapshot. The script re-locates every item. "Correct" values come
from the source files named in the ledger.

### Stale numbers

1. **Nebius snapshot in 05 is stale** (`docs/hackathon/05_nebius_execution.md`, section "Measured
   usage (generated 2026-10-29T02:25:19Z)", lines 67–188). It should be regenerated with the cutoff
   stated.

   | location | says | correct at the 2026-10-29T14:39:41Z cutoff |
   |---|---|---|
   | 05:73–74 | main `calls_total` / `calls_succeeded` 772 | **775** |
   | 05:77 | main `last_call_utc` 02:15:23Z | **2026-10-29T14:39:41Z** |
   | 05:99–100 | main tokens 2,598,343 / 96,356 | **2,611,780 / 96,750** |
   | 05:102 | main $0.8662 | **$0.8706** |
   | 05:151–152 | explore calls 2380 | **6,229** |
   | 05:155 | explore `last_call_utc` 2026-10-28T21:56:47Z | **2026-10-29T12:07:51Z** |
   | 05:179–180 | explore tokens 6,172,302 / 367,511 | **11,967,185 / 685,395** |
   | 05:182 | explore $2.1825 (Lightning priced at the Super-VL rate) | **$4.0884** at per-model prices |
   | 05:184 | `calls_by_tag_prefix` `"explore": 2380` | newer rows use r1_*, s1_base, s2_base and similar tags |

   The "2:15" the brief mentions is only this `last_call_utc` timestamp (02:15:23Z) and the old
   video length. No doc still gives the video as 2:15.
2. **The total call count is one behind the live logs.** 09:201 "(7,003, 0 failed)", 09:262
   "Over 7,003 calls" and 10:14 "7,003 logged calls" should read **7,004** at the 14:39:41Z cutoff.
   `demo_plan.json:127` already says 7,004. `submission/devpost_final.md`,
   `nebius_nvidia_feedback.md` and `requirements_checklist.md` still say 7,003. Freeze the logs (no
   more dashboard calls), or state "as of <cutoff>".
3. **Old totals are gone from the audited docs.** No audited doc or figure contains "3,152", "5,392",
   "$3.60" or "10.6 M input"; they were replaced in commit 7d7da95.
4. **The length-neutral check is quoted from a 250-episode snapshot.**
   * 07:90–91 says "changed 4 of 250 judgements (first 250 episodes)".
   * 11:23–31 says "250 eval episodes … 0.956 [0.923, 0.975], 8 FP, 3 FN vs 0.956, 10 FP, 1 FN; 4/250".
   * `experiment2_backend_comparison_pad201.json` now covers all 330 episodes. The original is
     **0.964 [0.938, 0.979], FP 9, FN 3**; the length-neutral run is **0.961 [0.934, 0.977], FP 12,
     FN 1**; **5/330** verdicts flip.
5. **Figure E prints a labelled stale snapshot** (`scripts/hackathon/make_figures.py:358–361`). The
   "Main pipeline only" line reads `results/hackathon/cost_latency.json` (2026-10-29T02:09:57Z:
   2,593,864 / 96,224 tokens, $0.86). The main log now holds 775 calls, 2,611,780 / 96,750 tokens,
   $0.87. The line is labelled as a snapshot, so this is minor.

### Wrong or overstated

6. **07:84 "Nemotron calls most missed grasps and drops 'not_placed'" is wrong for missed grasps.**
   * Of the 38 missed grasps, 20 were labelled `no_grasp` (correct) and **18** `not_placed`.
   * Of the 28 drops, **15** were `not_placed`, 8 `none`, 4 `no_grasp` and 1 `dropped`.
   * Correct wording: "about half of missed grasps and most drops".
7. **09:201 gives the wrong reason for "upper bound".** It says $4.96 is an upper bound "since some
   exploratory calls went to the cheaper Lightning model".
   * $4.96 already prices the 131 Lightning calls at Lightning's rate ($0.06 / $0.24).
   * The total would be $5.08 at the Super-VL rate.
   * It is an upper bound only because cached-input discounts are not applied.
8. **09:135 "Accuracy is unchanged when clip length is neutralised" is overstated.** Accuracy is
   0.961 vs 0.964, but false positives rise **9 → 12** and balanced accuracy falls **0.927 → 0.909**.
   Better: "essentially unchanged (0.961 vs 0.964)".
9. **09:209 contradicts 09 itself.** It says "The exploratory studies are post-hoc, except r1 and c1,
   which were pre-registered". s1 and s2 were also pre-registered: 09:179 and 09:186 say so, and so
   do their protocols.
10. **The 11:49 and 11:99 accuracy range is too wide.** They say "0.90–0.96 accuracy (on the eval
    split)", but the eval-split accuracies are **0.918** (Cosmos) and **0.964** (Nemotron), i.e.
    0.92–0.96 as `demo_plan.json` says.
11. **The 07:16–17 historical scope is too broad.** It says these numbers come "from the primary
    Diffusion Policy cohort of the paper's 1,365-checkpoint study (45 runs, …)".
    * The DP cohort is **24 runs**; 45 is the whole study (21 BC-RNN + 24 DP).
    * The "Pooled open-loop smoothness 0.019" bullet is pooled over all 45 runs, not the DP cohort
      (main.tex:217, 219).
12. **10:14 says only Super-VL served the 7,003 calls** ("served model
    `nvidia/Nemotron-3.5-Super-VL`"). In fact 6,873 calls went to Super-VL and **131** to
    `nvidia/Nemotron-3_5-Lightning`; 10:15 mentions Lightning separately.
13. **07:12 overstates its sources.** It says "Every number below is read from a file in
    `results/hackathon/`", but several numbers come from elsewhere:
    * §A comes from `docs/plv_arxiv/main.tex`.
    * The 12/12 narration count and the s2 46 vs 19 come from `data/outputs/`.
    * "3e-8", "800/800" and "only 2 v2c rejections cite the compartment" exist only in verifier
      reports.
14. **11:5–6 overstates verification.** It says "Each study was … checked by an independent
    adversarial verifier", but x0 (11:13) and w4 (11:78) were not independently verified.

### Not traceable to a result file

15. **07:160 "against 0.10–0.19 for mismatched clips".** No mismatched-clip control exists in
    `e5_probe.json`, `e5_probe_s3_units.csv` or `e5_analyze.py`. Either add the computation or drop
    the number.
16. **07:158 "30.8 dB vs 26–27 dB between seeds".** `e5_probe.json` S0 holds one between-seed
    comparison, **26.2 dB**. The 30.8 dB is correct.

### Missing qualifiers on the efficiency headlines

The numbers are correct; the baseline, target and unit are missing.

17. **"2–3x fewer rollouts" / "about 3x fewer rollouts".**
    * Locations: README:21 ("About **3x**"), README:172 ("about 2–3x"), 09:143, 07:302, 11:14, 11:100,
      `demo_plan.json:101` ("about 3x").
    * Two inconsistencies:
      * Within the README, 3x and 2–3x appear for the same claim.
      * The 09 / 07 / 11 wording (2–3x) differs from the newer README / demo / devpost_final wording
        (3x).
    * Source: median ratio 3, 10–90% range 2–4.
    * What it is: a post-hoc simulation on logged training-time rollouts, with perfect labels, on a
      56-checkpoint snapshot.
    * Baseline: uniform allocation. Target: excess regret ≤ 0.02 over the full-pool floor.
    * No PLV prior is involved.
18. **"16–20% robot time".**
    * Locations: README:21, README:174 ("a **time-horizon** early abort saves 16–20%"), 09:144,
      07:303, 11:15, 11:100, `demo_plan.json:101`.
    * The time-horizon rule alone saves **16.2%**. The **20.1%** rule also uses privileged simulator
      state (no lift / drop).
    * Both are on a 200-episode, 20-checkpoint snapshot, at 2/141 false aborts.
    * "Time-horizon … 16–20%" therefore mislabels the upper end.

### Demo video length and format (re-check after the re-render)

19. The length and format of the demo video disagree across documents.

    | location | says |
    |---|---|
    | README:10 | "2:25, 1080p" |
    | README:127 | "2 min 23 s" |
    | 10:20 | "(2:23, 1600x992)" |
    | 10:21 | "About 84 s of the 143 s video" |

    Measured at the snapshot:
    * The committed `media/hackathon/demo.mp4` is **143.2 s, 1600x992**, which matches README:127,
      10:20 and 10:21 but not README:10.
    * The current `demo_plan.json` sums to **153.0 s**, with 91.0 s of real footage.
    * `submission/demo_final.mp4` was not yet readable.

    After the final render, run `ffprobe` on the uploaded file. Then make README:10, README:127,
    10:20 and 10:21 agree with it, including the real-footage seconds.

### Checked and consistent

These were checked and found consistent with the source:
* **Experiment 1:** 93 / 3 / 4,650; every ρ; Regret@30 0.007; ρ with epoch 0.96–1.00.
* **Experiment 2:** 330 / 263 / 67; base rate 0.797; 0.964 / 0.927 / 9 of 67 / 3; 0.918 / 0.832 /
  21 of 67; 0.961; 94% / 97%; failure types below the majority baseline; dev 0.806 / 0.861; the
  12/12 narration.
* **Experiment 3:** 0.040 vs 0.007; 0.06–0.09.
* **x0:** 16% / 0.952 / 3 → 8.
* **E4b:** cos 0.78–0.85; t(4) bounds 0.15 / 0.41 / 0.63; 0.4x; PSNR; Can vs Square; the retraction.
* **r1:** every number in 07 §C4, README:200 and 09:147–163.
* **c1:** every number in 07 §C5, README:22, README:201 and 09:164–177.
* **s1:** 0.875–0.89 and both CIs.
* **s2:** 0.856 / 0.938 / 0.824 / 0.752, every CI, 58/149, 44, 40%, −7.2, 46 vs 19.
* **Nebius:** 0 failures; 14.6 M / 0.78 M tokens; $4.96.
* **Figures and demo:** the figure titles in `s2_figure.py`; demo captions 95, 115, 121 and 133.

## Claims not ledgered

Some detailed numbers in 11 and the `docs/hackathon/explore/*` reports were not given their own
ledger row. Each is quoted from its own verification report:
* x3, x4 (0.441 vs 0.356), x5 and x6.
* y2 (split-half ρ 0.02), y5 and z1 / z2.
* w4, which is not independently verified.

The test count in 10 ("38 pass in each environment"; 45 `test_` functions exist) was not re-run.
