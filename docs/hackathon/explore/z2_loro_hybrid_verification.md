# z2_loro_hybrid adversarial verification (EXPLORATORY, post-hoc analysis)

**Verdict: partially_confirmed.**

- **What is confirmed.** The core negative result holds, and in fact becomes stronger. Neither the learned combinations nor the VLM K=10 features are distinguishable from epoch in this pool.
  - Rerunning the original script gave bit-identical results. We compared 16,985 numbers in `z2_loro_hybrid.json` and the max abs diff is 0. The input sha256 values are also the same.
  - An independent implementation that does not import z2 code (log parser, expected Spearman, ridge, pairwise-logit solved with backtracking Newton, nested λ, unitweight, select) reproduced the per-held-out-run ρ exactly. All 36 all93 methods, 53 eval33 methods and 36 late93 methods differ by 0, and swap is also the same.
  - We found no leakage. The fit and inner CV use only HA of the training runs. HB of the held-out run is used only for scoring.
  - **Newly confirmed.** Over 400 random 25/25 seed splits, the number of methods with a positive mean Δ over epoch is 0 in all three pools.
- **What is refuted or weakened.**
  1. **The headline +0.009 is the value of one lucky split.** The +0.009 of pairlogit_rank[all] is the 98th percentile of the distribution over 400 random splits. The split-mean Δ is −0.016, and P(Δ>0) is 0.09. ridge_rank[all] is also at the 97th percentile (mean −0.018). RA(epoch, L1) is at the 95th percentile (mean −0.012). The original HB half is also favorable to epoch (epoch ρ is at the 88th percentile).
  2. **The plateau Action L1 "+0.064, 3/3 runs" is split luck.** Over random splits this value is at the 95th percentile, the mean Δ is −0.025, and P(Δ>0) is 0.32. Using all 50 episodes as the target, Δ is −0.044 [−0.136, +0.056], and it wins in only 1 of the 3 runs. Moreover, the targets of the late can runs are nearly degenerate. 10 of 11 checkpoints in can_seed2 and 17 of 21 in can_seed1 have HB=1.0.
  3. **"Every +vlm model is equal to or lower than its pair without VLM" is wrong.** In 3 of 8 pairs, +vlm is higher: pairlogit_rank[all] +0.015, ridge_rank[small] +0.010, unitweight[small] +0.004, and 1 pair is equal. "+vlm models 0.815–0.833" is also wrong; the actual range is 0.793–0.833. The conclusion (does not beat epoch) still holds.
  4. **The bootstrap CIs of absolute ρ are not valid CIs.** In the block bootstrap, for 26 of 36 methods the point estimate lies outside its own 95% CI. Epoch is 0.825, yet its CI is [0.370, 0.816]. Even in the episode-only bootstrap, 20/36 lie outside (epoch [0.637, 0.824]). The −0.4 left tail of the learned models is caused by the resampling design (Section 5).
  5. **The power arithmetic is optimistic.** "8–17 runs" corresponds roughly to the number for 50% power (9–18). Getting 80% power at two-sided α=0.05 requires 17–34 runs. But because the split-mean Δ is negative, this calculation itself is meaningless.
  6. There are a few small overstatements.
     - The "run-level 95% CI" is the [min, max] of 3 folds. Its probability of covering the population median is only 75%.
     - "pairlogit_z[all] = epoch ranking (all 3 folds)" is not exact. In can_seed2, epochs 140 and 160 are swapped. Both have HB=1.0, so ρ is the same.
     - The late93 learned-model range "0.05–0.17" is actually 0.042–0.174 (unitweight is 0.025–0.145).

## 0. Check of the user's question: "Do we have the Nemotron or Cosmos outputs saved anywhere?"

We checked z2's answer directly, and it is correct.

- Location: `data/outputs/hackathon_cosmos/`
  - `eval_local_v6.jsonl`: Cosmos3-Edge, last frame, 330 rows
  - `eval_local_v7.jsonl`: Cosmos3-Edge, 8 keyframes, 330 rows
  - `eval_nebius_v2.jsonl`: Nemotron (Nebius), full video, 330 rows
  - `eval_nebius_v2_pad201.jsonl`: Nemotron, length-neutral clip, 330 rows
- Besides these there are `dev_prompt_selection.jsonl`, `dev_nebius_v2_pad201.jsonl`, `dev_infra_failures.jsonl`, and the pre-label-refresh copies `*.orig.jsonl` (150/152/150/330 rows).
- Each row has the following fields: `raw_text`, `cosmos_task_success`, `cosmos_failure_type`, `cosmos_explanation`, `latency_s`, `model_id`/`model_revision`, `request_id`, `truth_*`.
- `results/hackathon/raw_model_outputs/` holds 7 copies and a README. All sha256 values match the originals.
- The Nebius call logs are `results/hackathon/nebius_calls.jsonl` (772 rows) and `results/hackathon/explore/nebius_calls_explore.jsonl` (2380 rows).
- The Cosmos mode-exploration output is `results/hackathon/explore/x5_cosmos_modes_episodes.jsonl` (297 rows).

## 1. Method (CPU only, no GPU or Nebius calls)

| ID | What | Script / output |
|---|---|---|
| R | Rerun the original command as is (`--n-boot 1000`) and compare the whole JSON numerically | `data/outputs/hackathon_explore/z2_loro_hybrid_verify/rerun_main.json` |
| V1 | Compute point estimates for the fixed HA/HB split and the swap with an independent reimplementation, and compare with z2 | `scripts/hackathon/explore/z2_loro_hybrid_verify.py` |
| V2 | Split the 50 seeds randomly 25/25 (400 times), and refit the whole LORO, including nested λ, for each split | same script, `V2_random_splits` |
| V3 | LORO-50: fit training runs on 50 episodes, and score the held-out run on its own 50 episodes. Held-out results are not used in the fit | `V1_V3_point.*.loro50_*` |
| V4 | Apply fit-free scores to the 50-episode target and compute Δ CIs with a seed-cluster bootstrap (2000 times). With no refit, percentile CIs can be used | `V4_free_scores_full50` |
| V5 | Diagnose bootstrap bias from the z2 JSON and measure the moving-block endpoint effect. Rerun z2's block bootstrap 400 times with the same RNG and decompose the causes per fold | `V5_bootstrap_bias`, `z2_loro_hybrid_verify_blocktail.py` → `z2_loro_hybrid_verification_blocktail.json` |
| V6 | Check claims and arithmetic: ranges, +vlm pair comparisons, power | `V6_claims` |

```
cd /path/to/repo
PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python -W ignore scripts/hackathon/explore/z2_loro_hybrid.py --n-boot 1000 --workers 6 \
    --out data/outputs/hackathon_explore/z2_loro_hybrid_verify/rerun_main.json             # exit 0, 249 s
PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python -W ignore scripts/hackathon/explore/z2_loro_hybrid_verify.py --n-splits 400 --n-boot 2000 --workers 10   # exit 0, 105 s
PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python -W ignore scripts/hackathon/explore/z2_loro_hybrid_verify_blocktail.py                              # exit 0, 62 s
```

The first verify run (400 splits) finished with exit 0. But the late93 V2/V4 summaries were NaN. In some splits the can_seed2 late target became constant, so ρ was undefined. We fixed this to average only over runs where ρ is defined, and regenerated the outputs with the commands above. The all93 and eval33 numbers did not change.

## 2. Reproduction (R, V1)

- The rerun JSON and the original JSON agree on all 16,985 numbers (max abs diff 0). The input sha256 values are also the same.
- Compared with the independent implementation, the methods whose per-held-out-run ρ differs by 0 are all93 36/36, eval33 53/53, late93 36/36. The swap mean difference is also 0.
- The selections of select_best_on_train also match the z2 report.
  - all93: RA(epoch,replan), RA(epoch,TF-MSE), RA(epoch,TF-MSE)
  - eval33: RA(epoch,dyn), plv1, RA(epoch,dyn)
  - late93: RA(epoch,replan), TF-MSE, epoch
- We checked the numbers in the report tables (Δ, run CI, block CI, P(Δ>0), swap) against the JSON. Sections 5.1, 5.2 and 5.3 all match. The exception is the range statements in Section 4 below.

## 3. Key point: 400 random splits instead of one fixed split (V2)

We compared the run-mean Δ of held-out ρ (25 random episodes instead of HB) against epoch. "Original split pct" is the percentile that the Δ of the split z2 used occupies in the distribution of 400.

| pool | method | z2 original split Δ | original split pct | split-mean Δ | median Δ | P(Δ>0) | 2.5–97.5% |
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

- The number of methods with mean Δ above 0 is all93 0/12, eval33 0/24, late93 0/12. The best is train_loss (effectively the same ordering as epoch, −0.0004).
- "The small models win in all 3/3 runs" recurs in only 4–8% of the 400 splits.
- In late93, one run's target was constant in 13 splits. In those splits we averaged only over the runs where ρ is defined.
- **Interpretation.** z2's "not distinguishable" is correct. But the fixed split was in the tail favorable to the learned combinations. Averaged over splits, the learned combinations are slightly (−0.004 to −0.018) below epoch.

## 4. 50-episode target (V3, V4)

**V3: LORO-50.** Fit on the 50 episodes of the training runs and scored on the 50 episodes of the held-out run.

| pool | epoch | best non-epoch | pairlogit_rank[all] | ridge_rank[all] | RA(epoch, L1) | L1 |
|---|---|---|---|---|---|---|
| all93 | 0.815 | pairlogit_z[small] +0.003 | −0.023 | −0.022 | −0.015 | −0.043 |
| eval33 | 0.854 | nothing beats epoch. pairlogit_z[all+vlm] −0.015, ridge_rank[all+vlm] −0.011, RA(epoch,Nemotron) −0.032 | — | — | — | — |
| late93 | 0.333 | unitweight[small] +0.010 | −0.182 | −0.227 | −0.015 | −0.044 |

**V4: fit-free scores, 50-episode target, seed-bootstrap 95% CI.**

| pool | score | Δ vs epoch | per-run Δ | 95% CI | P(Δ>0) |
|---|---|---|---|---|---|
| all93 | RA(epoch, L1) | −0.015 | −0.002 / −0.014 / −0.030 | [−0.033, +0.010] | 0.14 |
| eval33 | RA(epoch, L1) | −0.052 | −0.043 / −0.042 / −0.071 | [−0.077, −0.011] | 0.01 |
| eval33 | RA(epoch, Nemotron) | −0.032 | 0 / 0 / −0.096 | [−0.059, +0.024] | 0.17 |
| late93 | Action L1 | −0.044 | +0.050 / −0.095 / −0.087 | [−0.136, +0.056] | 0.27 |
| late93 | RA(epoch, L1) | −0.015 | +0.054 / −0.082 / −0.016 | [−0.079, +0.042] | 0.33 |

**Degeneracy of the late93 target.** In the late can runs, HB takes only 2 distinct values. 10 of 11 in can_seed2 and 17 of 21 in can_seed1 are 1.0. So the plateau Spearman in the can runs is decided by 1–4 checkpoints. z2 did not report this. With the 50-episode target, "3/3 wins" becomes "1/3 wins".

## 5. Bootstrap diagnostics (V5)

**The absolute-ρ CIs are biased.**
- Block bootstrap (all93): for 26/36 methods the point estimate lies outside its own 95% CI.
  - epoch: 0.825, CI [0.370, 0.816]
  - pairlogit_rank[all]: 0.834, CI [0.095, 0.785]
- Episode-only bootstrap (all93): 20/36 lie outside the CI. Epoch is [0.637, 0.824]. This is because the target resampled with replacement lowers the Spearman.
- So the "held-out rho [CI]" column of `z2_loro_hybrid_tables.txt` must not be read as a CI.

**The −0.4 tail of the block bootstrap comes from the design.**
- A non-circular moving block under-samples the endpoints. In can_seed1 the expected number of copies of the first checkpoint is 0.18, versus 1.21 for a middle checkpoint. can_seed2 is 0.34 vs 1.17, and square is 0.28 vs 1.11.
- We reran z2's block bootstrap 400 times with the same RNG (1,200 folds).
  - In 56% of folds, one of the training runs loses its first 10% of epochs entirely.
  - In those folds the mean Δ of the learned models is −0.089 to −0.099 (share with Δ<−0.1: 21–27%).
  - In folds where both training runs keep their early checkpoints, it is −0.013 to −0.037 (share with Δ<−0.1: 7–17%).
  - The fit-free RA(epoch, L1) is unaffected, at +0.005 in both cases.
- So the guess z2 wrote in its limitations (a training run missing the early learning curve) is correct. That tail is more an artifact of the resampling design than a property of the learned models.
- Still, the conclusion does not change. Even in random splits with no duplication and no blocks (V2), P(Δ>0) is 0.09–0.47.

**Location of the Δ CI.** In the block bootstrap, the median Δ of the learned models is lower than the point estimate (pairlogit_rank[all]: point +0.009, median −0.033). Yet the V2 split mean (−0.016) is closer to the bootstrap median. This location bias of the bootstrap largely reflects the fact that "the original split was lucky".

## 6. Claim and arithmetic checks (V6)

| z2 claim | Check result |
|---|---|
| all93 epoch 0.825 (0.853/0.838/0.784), Regret@30 = top-1 = 0 | Correct. But in the can runs 21 of 41 and 12 of 21 are tied at the HB maximum (1.0), so it is effectively trivial. The only informative run is square (5 of 31) |
| best learned combination 0.834 (+0.009), run CI [−0.005, +0.030], block [−0.41, +0.06], P 0.24 | Numbers correct. The original split is at the 98th percentile, and the split mean is −0.016 |
| under swap all learned models 0.669–0.686 < epoch 0.690 | Correct. However, the fit-free RA(epoch, replan) is 0.697 under swap, higher than epoch |
| eval33 "+vlm models 0.815–0.833, equal to or lower than their pairs without VLM" | **Wrong.** The range is 0.793–0.833, and in 3 of 8 pairs +vlm is higher |
| eval33 pairlogit_z[all] = epoch ranking | ρ is the same. The ordering has 140/160 swapped in can_seed2 (tied targets) |
| late93 L1 0.431, Δ +0.064, 3/3 | Numbers correct. 95th percentile over splits, and −0.044, 1/3 with the 50-episode target |
| late93 learned models 0.05–0.17 | 0.042–0.174 (unitweight 0.025–0.145) |
| fixed λ cosine 0.96–0.99, val_loss·support_drift sign unstable (0.34–0.84), λ moves between both ends of the grid | Correct |
| 3/3 one-sided sign test p = 0.125 | Correct |
| "run-level 95% CI" | It is the [min, max] of 3 folds. Its probability of covering the population median is 0.75. Calling it 95% is an overstatement |
| required number of runs "8–17" | Computed from 3 fold Δs (the sd estimate itself is very unstable), 50% power needs 9–18 runs and 80% needs 17–34 (pairlogit_rank/ridge_rank[all]). Meaningless because the split-mean Δ is negative |

## 7. Summary by check item

- **Leakage.** None.
  - The fit, inner λ, unitweight signs and select all use only `yA[tr]`. Scoring uses `yB[te]`.
  - Within-run normalization does not use outcomes (z2 disclosed that it is transductive).
  - There is one minor point. The eval33 select_best candidates include the privileged `k10_sim` and `RA(epoch,k10_sim)`. These are actual outcomes on eval seeds 200000+, so they are not HB leakage. They are also never selected in any fold, so the numbers are unaffected.
- **Held-out handling.** Correct. The inner LORO over the 2 training runs also uses only HA.
- **Clustering and independence.** The outcome unit is the run, and there are only 3 runs. z2 disclosed this honestly. The same env seeds are shared across checkpoints and across runs, so resampling seeds jointly is correct. But the block bootstrap has the endpoint bias described above.
- **Multiple comparisons.** Each pool reports 36–53 scores and highlights the best. Since the conclusion is on the null side, the harm is small. However, V2 and V4 confirmed that cases like late93 L1, "best of 35 + 3/3", are the selection effect itself.
- **Judge validity.**
  - The VLM K=10 features are independent outcome measurements from eval seeds. Cosmos has FP 21/67 and Nemotron FP 9/67.
  - There is no gain either in random splits (V2) or with the 50-episode target (V3·V4). "VLM features do not beat epoch" is robust.
  - Only the wording "always lower than its pair" is wrong.
- **Claims stronger than the numbers.** Listed in items 1–6 above (the refutation list).

## Corrected headline

Neither the learned combinations nor the VLM K=10 features beat epoch in this pool. The reproduction was bit-identical and was also confirmed with an independent implementation. z2's best value +0.009 (pairlogit_rank[all], held-out ρ 0.834 vs 0.825) appears only on the HA/HB split z2 chose. That split corresponds to the 98th percentile of 400 random 25/25 splits. Averaged over splits, every learned combination is slightly below epoch (pairlogit_rank[all] Δ −0.016, P(Δ>0) = 0.09; small models −0.004 to −0.005). With LORO over all 50 episodes, the best value is +0.003. The +0.064 (3/3) of Action L1 on the plateau is also split luck. With the 50-episode target it is −0.044 (1/3). The eval33 +vlm models are at 0.793–0.833, below epoch (0.852). But they are not always below their pairs without VLM (in 3 of 8 pairs they are higher). The bootstrap CIs of absolute ρ do not contain the point estimates and must not be used. The −0.4 tail of the block bootstrap is an artifact of moving-block endpoint under-sampling.

## Remaining risks (open)

- 3 runs, 2 tasks. All run-level inference has no power in principle.
- Training is short and nearly monotonic. The plateau target is nearly degenerate (can late HB takes only 2 values).
- The fixed HA/HB split was also skewed in the favorable direction in the y1 verification. If other z/y results also rely on one fixed split, they should be rechecked with random splits.

## Files

- Scripts:
  - `scripts/hackathon/explore/z2_loro_hybrid_verify.py`
  - `scripts/hackathon/explore/z2_loro_hybrid_verify_blocktail.py`
- Results:
  - `results/hackathon/explore/z2_loro_hybrid_verification.json`
  - `results/hackathon/explore/z2_loro_hybrid_verification_blocktail.json`
- Rerun copies and logs: `data/outputs/hackathon_explore/z2_loro_hybrid_verify/`
  - `rerun_main.json`
  - `rerun_main.log`
  - `verify_run2.log`
