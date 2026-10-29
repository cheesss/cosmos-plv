# z2_loro_hybrid: can a cross-validated combination of outcome-free signals beat "later epoch"? (EXPLORATORY, post-hoc)

> **Label:** exploratory and post-hoc. Nothing here was pre-registered in `06_experiment_protocol.md`.
> Every number comes from `results/hackathon/explore/z2_loro_hybrid.json` (main, checkpoint-block bootstrap),
> `z2_loro_hybrid_episode_boot.json` (episode-only bootstrap) and `z2_loro_hybrid_late.json` (late-half pool), all written by
> `scripts/hackathon/explore/z2_loro_hybrid.py`. Tables were rendered by `z2_loro_hybrid_tables.py` into
> `results/hackathon/explore/z2_loro_hybrid_tables.txt` and `z2_loro_hybrid_late_tables.txt`. CPU only. No GPU and no Nebius calls.

**한 줄 요약 (KR):** 학습된 조합 중 어느 것도 epoch과 구별되지 않았다. 93개 체크포인트를 leave-one-run-out으로 돌렸을 때 가장 좋은 학습 조합(pairwise-logit, 랭크 특징)의 held-out Spearman은 0.834였고 epoch은 0.825였다(Δ +0.009). 그러나 run-level CI는 [−0.005, +0.030], 체크포인트 블록 bootstrap CI는 [−0.41, +0.06], P(Δ>0)=0.24였다. 절반을 바꿔 fit하면(HB로 fit, HA로 평가) 순서가 뒤집혔다(0.672 vs 0.690). 33개 eval 체크포인트에서는 VLM(K=10) 특징을 넣어도 epoch(0.852)을 넘는 모델이 없었다. 사후에 λ를 가장 유리하게 골라도 넘지 못했다. Regret@30과 top-1 regret은 epoch이 세 run 모두에서 이미 0이어서 개선할 여지가 없다. 학습된 가중치는 "학습이 진행될수록 줄어드는 모든 지표"에 비슷한 양의 가중치를 주는 쪽으로 수렴했다. 반면 val_loss와 support_drift의 부호, 그리고 λ 선택은 fold마다 불안정했다. run이 3개뿐이라 run-level 검정은 원리적으로 p<0.05에 도달할 수 없다(3/3 승리여도 단측 부호검정 p=0.125).

## 1. Question

Experiment 1 and the wave-1/2 studies found that the free "later epoch" baseline is as good as any single PLV
metric in this pool (`07_experiment_results.md`, `11_explorations.md`). A natural rescue is to **learn** a combination of
outcome-free signals. Can any learned combination beat epoch when it is cross-validated across runs and evaluated on
outcomes it never saw? The signals are epoch, the cheap PLV metrics, the losses and, on the 33 eval checkpoints, K=10
VLM-judged success rates.

## 2. Data

* 93 Diffusion Policy checkpoints in 3 runs: `can_seed1` (41, every 5 epochs), `can_seed2` (21, every 10), `square_seed1`
  (31, every 10, to epoch 300). Metrics come from `results/hackathon/baseline_metrics.csv`.
* Outcomes are the per-episode training-time results `test/sim_max_reward_<seed>` from each run's `logs.json.txt`. The
  script asserts that their means reproduce `success` exactly.
  * **HA** = seeds 100000–100024. These are the only labels used for fitting.
  * **HB** = seeds 100025–100049. This is the target, used only on the held-out run.
* Eval checkpoints: 33 (11 per run, every 20 or 30 epochs), each with 10 recorded rollouts (seeds 200000–200009,
  disjoint from HA/HB). They are judged by:
  * Cosmos3-Edge (`eval_local_v6.jsonl`);
  * Nemotron via Nebius (`eval_nebius_v2.jsonl`);
  * Nemotron on length-neutral clips (`eval_nebius_v2_pad201.jsonl`).

  The instrumented K=10 success is kept only as a privileged reference. The script asserts that every VLM row is scored
  and that its `truth_success` matches the rollout summary.

## 3. Method

**Features** (outcome-free; all oriented so that higher = better a priori):
* epoch;
* −log(cost) for 10 cheap metrics: val_action_l1, open_loop_smoothness, plv1_smoothness, replan_overlap_h16,
  boundary_jump_h16, val_loss, train_loss, support_drift, dynamics_uncertainty, teacher_forced_latent_mse;
* on eval33 only: the K=10 VLM success rates.

Omitted as duplicates:
* H=10 replan and boundary jump are identical to H=16 here;
* val_loss_logged equals val_loss;
* epoch/max-epoch and rank-within-run are the same affine transform of epoch, because the checkpoints are uniformly
  spaced.

Each feature is normalised **within run**, without outcomes, either as a z-score or as a rank in [0, 1].

**Feature sets:**
* `all`: epoch + 10 metrics;
* `noepoch`: the 10 metrics only;
* `small`: epoch, Action L1, open-loop smoothness, replan overlap, val_loss;
* on eval33 also `all+vlm` and `small+vlm` (adding Cosmos v6 and Nemotron v2), and `epoch+vlm`.

**Models:**
* `pairlogit_{z,rank}`: linear RankNet. This is an L2-penalised logistic loss on all within-run checkpoint pairs with
  different HA success. Pairs are weighted by |ΔS|, each run counts equally, and there is no intercept.
* `ridge_rank`: ridge regression of the within-run rank of HA success on within-run centred rank features.
* `unitweight`: gradient-free. Each feature's rank gets weight ±1, with the sign learned on the training runs.
* `select_best_on_train`: picks the best of 31 fixed scores by mean within-run Spearman on the training runs. The 31 are
  epoch, the 10 single metrics, rank-average(epoch, metric) for each metric, the two all-metric rank-averages and, on
  eval33, the VLM scores.
* Fixed scores that need no fitting: every single metric, rank-average(epoch, m), RA(all cheap), RA(epoch + all cheap),
  VLM alone, and RA(epoch, VLM).
* References:
  * `oracle_HA25`: the held-out run's own 25 HA rollouts. It uses outcomes, but never HB.
  * `random`.

**Leave-one-run-out:** 3 folds. Fit on two runs using HA, then score the third run against HB.
* λ is chosen from {1e-3, …, 1e3} by an *inner* leave-one-run-out over the two training runs, again using HA only.
  Ties go to the larger λ.
* As a robustness check, the halves are swapped: fit on HB, evaluate on HA.

**Statistics on the held-out run:**
* Spearman, computed as an exact expectation over random tie-breaks;
* Regret@30 = max S − max S among the top ⌈0.3n⌉;
* top-1 regret.

Each is averaged over the three held-out runs.

**CIs:**
* **Run-level:** exact enumeration of all 27 ordered resamples of the 3 fold results. With 3 runs the 95% interval is
  effectively [min fold, max fold].
* **Checkpoint-block bootstrap:** 1000 replicates with **refitting**. Each replicate resamples, jointly:
  * moving blocks of 40 epochs of checkpoints inside every run (block length 8/4/4 checkpoints on all93 and 2 on
    eval33);
  * the HA seeds;
  * the HB seeds;
  * the eval seeds, with the VLM rates recomputed.

  Δ vs epoch is paired inside each replicate.
* **Episode-only bootstrap:** the same, but with the checkpoint sets held fixed.

**Late-half pool** (`late93`): checkpoints with epoch ≥ the run's median epoch (21/11/16), a subset chosen without
outcomes. This is the plateau, where epoch order should be weakest.

## 4. Commands and exit codes

```
PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/z2_loro_hybrid.py --n-boot 5  --out data/outputs/hackathon_explore/z2_loro_hybrid/smoke.json   # exit 0
...                                                   --n-boot 16/32 (smoke2): exit 1 (no informative pair in a block-resampled run -> np.vstack on []), fixed, exit 0
...                                                   --pools late93: exit 1 (KeyError: no finite training rho for select_best in a plateau replicate), fixed (fallback = epoch), exit 0
PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/z2_loro_hybrid.py --n-boot 1000 --workers 8                    # exit 0, 283 s
PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/z2_loro_hybrid.py --no-blocks --n-boot 1000 --workers 8 \
    --out results/hackathon/explore/z2_loro_hybrid_episode_boot.json                                                                             # exit 0, 283 s
PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/z2_loro_hybrid.py --pools late93 --n-boot 1000 --workers 6 \
    --out results/hackathon/explore/z2_loro_hybrid_late.json                                                                                     # exit 0, 143 s
PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/z2_loro_hybrid_tables.py                                      # exit 0
PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/z2_loro_hybrid_tables.py \
    results/hackathon/explore/z2_loro_hybrid_late.json results/hackathon/explore/z2_loro_hybrid_late_tables.txt                                  # exit 0
```
All three final JSONs were produced by the final script. Logs are in `data/outputs/hackathon_explore/z2_loro_hybrid/`.

## 5. Results

### 5.1 All 93 checkpoints (cheap features only)

Epoch alone gives Spearman 0.853 / 0.838 / 0.784 (mean 0.825), with Regret@30 = 0 and top-1 regret = 0 **in every held-out
run**. On the regret metrics nothing can do better than tie epoch.

| score | held-out ρ (HB) | Δρ vs epoch | run-level CI of Δ | block-boot CI of Δ | P(Δ>0) block | episode-only CI of Δ | runs better/worse | top-1 regret | swap ρ (epoch 0.690) |
|---|---|---|---|---|---|---|---|---|---|
| **epoch** | 0.825 | 0 | — | — | — | — | — | 0.000 | 0.690 |
| pairlogit_rank[all] | 0.834 | +0.009 | [−0.005, +0.030] | [−0.407, +0.059] | 0.24 | [−0.105, +0.013] | 2/1 | 0.000 | 0.672 |
| ridge_rank[all] | 0.833 | +0.008 | [−0.002, +0.020] | [−0.405, +0.055] | 0.22 | [−0.126, +0.011] | 2/1 | 0.000 | 0.677 |
| pairlogit_z[small] | 0.830 | +0.005 | [+0.001, +0.010] | [−0.441, +0.057] | 0.40 | [−0.126, +0.020] | 3/0 | 0.027 | 0.680 |
| ridge_rank[small] | 0.829 | +0.004 | [+0.001, +0.007] | [−0.505, +0.054] | 0.34 | [−0.018, +0.012] | 3/0 | 0.000 | 0.677 |
| RA(epoch, Action L1) (no fitting) | 0.830 | +0.005 | [−0.014, +0.026] | [−0.085, +0.097] | 0.54 | [−0.020, +0.021] | 2/1 | 0.000 | 0.664 |
| unitweight[all] | 0.825 | +0.000 | [−0.015, +0.013] | [−0.512, +0.057] | 0.44 | [−0.022, +0.010] | 2/1 | 0.000 | 0.681 |
| train_loss | 0.824 | −0.001 | [−0.004, 0.000] | [−0.023, +0.019] | 0.41 | — | 0/1 | 0.000 | 0.690 |
| replan_overlap_h16 | 0.817 | −0.009 | [−0.026, 0.000] | [−0.043, +0.017] | 0.26 | [−0.018, +0.004] | 0/1 | 0.027 | 0.680 |
| val_action_l1 | 0.815 | −0.010 | [−0.030, +0.031] | [−0.161, +0.118] | 0.39 | [−0.050, +0.023] | 1/2 | 0.000 | 0.637 |
| RA(epoch + all cheap) | 0.809 | −0.016 | [−0.032, −0.005] | [−0.114, +0.055] | 0.25 | [−0.035, +0.007] | 0/3 | 0.000 | 0.672 |
| pairlogit_z[all] | 0.793 | −0.033 | [−0.078, −0.010] | [−0.470, +0.036] | 0.21 | [−0.117, +0.012] | 0/3 | 0.000 | 0.675 |
| select_best_on_train | 0.783 | −0.042 | [−0.150, +0.024] | [−0.420, +0.066] | 0.15 | [−0.234, +0.007] | 1/1 | 0.027 | 0.685 |
| oracle_HA25 (25 real rollouts) | 0.588 | −0.237 | [−0.412, −0.050] | [−0.492, +0.010] | 0.03 | [−0.337, −0.081] | 0/3 | 0.033 | 0.562 |
| val_loss | −0.456 | −1.281 | [−1.366, −1.192] | [−1.363, −0.593] | 0.00 | — | 0/3 | 0.133 | −0.329 |

The full table, covering all 38 scores, is in `z2_loro_hybrid_tables.txt`.

* **Point estimates:** the best learned combinations come out +0.004 to +0.009 ahead of epoch. That is smaller than the
  episode-label noise alone. Under the episode-only bootstrap, ridge_rank[small] has a CI of [−0.018, +0.012] and
  P(Δ>0) = 0.52.
* **Swapped halves:** fitting on HB and evaluating on HA, **every** learned model falls below epoch (0.669–0.686 vs
  0.690).
* **Block bootstrap:**
  * the median Δ of the learned models is negative (pairlogit_rank[all] −0.033, ridge_rank[all] −0.039);
  * the lower tails are long (≈ −0.4). When the resampled training runs miss the steep early part of the learning curve,
    the fit is driven by plateau noise.
  * epoch has no such failure mode.
* **3/3 runs:** the two `small` models beat epoch in 3/3 runs, by +0.001 to +0.010. With 3 runs this has a one-sided
  sign-test p = 0.125, and it reverses under the half swap.
* **Post-hoc λ:** choosing λ on the test folds (an optimistic upper bound) gives at most 0.833 (pairlogit_z[small],
  λ = 0.01), which is +0.008.
* **Choosing a score on the training runs:** this is *worse* than epoch (−0.042). Twice it picks RA(epoch,
  teacher-forced latent MSE), which wins on can_seed2 and fails on square.
* **Real rollouts:** epoch also beats 25 real rollouts of the held-out run (oracle_HA25 0.588). The within-run
  Spearman(HA, HB) is 0.60 / 0.79 / 0.37. Most of the remaining rank disagreement sits inside binomial noise.

### 5.2 The 33 eval checkpoints (adding K=10 VLM-judged success)

Epoch gives 0.882 / 0.795 / 0.878 (mean 0.852), again with zero Regret@30 and zero top-1 regret in every run.

| score | held-out ρ (HB) | Δρ vs epoch | run-level CI of Δ | block-boot CI of Δ | P(Δ>0) block | episode-only CI of Δ | swap ρ (epoch 0.783) |
|---|---|---|---|---|---|---|---|
| **epoch** | 0.852 | 0 | — | — | — | — | 0.783 |
| pairlogit_z[all] (= epoch ranking in all 3 folds) | 0.852 | 0.000 | [0, 0] | [−0.480, +0.012] | 0.04 | [−0.101, 0.000] | 0.783 |
| pairlogit_z[small] | 0.846 | −0.006 | [−0.018, 0.000] | [−0.538, +0.011] | 0.04 | [−0.133, +0.009] | 0.777 |
| ridge_rank[all] | 0.835 | −0.017 | [−0.042, 0.000] | [−0.562, +0.054] | 0.10 | [−0.216, +0.013] | 0.777 |
| pairlogit_z[all+vlm] | 0.833 | −0.018 | [−0.055, 0.000] | [−0.529, +0.073] | 0.12 | [−0.111, +0.017] | 0.772 |
| ridge_rank[all+vlm] | 0.830 | −0.022 | [−0.065, 0.000] | [−0.596, +0.083] | 0.10 | [−0.190, +0.011] | 0.786 |
| RA(epoch, Nemotron K10) (no fitting) | 0.827 | −0.025 | [−0.074, 0.000] | [−0.253, +0.104] | 0.18 | [−0.111, +0.024] | 0.757 |
| unitweight[all+vlm] | 0.823 | −0.029 | [−0.042, −0.010] | [−0.535, +0.045] | 0.08 | [−0.063, +0.007] | 0.788 |
| pairlogit_z[small+vlm] | 0.815 | −0.037 | [−0.111, 0.000] | [−0.595, +0.073] | 0.12 | [−0.153, +0.019] | 0.755 |
| RA(epoch, Cosmos K10) | 0.795 | −0.057 | [−0.171, 0.000] | [−0.277, +0.127] | 0.19 | [−0.140, +0.031] | 0.720 |
| ridge_rank[epoch+vlm] | 0.793 | −0.059 | [−0.176, 0.000] | [−0.585, +0.138] | 0.16 | [−0.163, +0.034] | 0.734 |
| select_best_on_train | 0.730 | −0.122 | [−0.219, −0.022] | [−0.503, +0.098] | 0.08 | [−0.276, +0.016] | 0.783 |
| Nemotron K10 alone | 0.636 | −0.215 | [−0.252, −0.186] | [−0.558, +0.077] | 0.06 | [−0.375, −0.047] | 0.656 |
| Cosmos K10 alone | 0.571 | −0.281 | [−0.510, −0.119] | [−0.635, +0.082] | 0.06 | [−0.458, −0.052] | 0.515 |
| instrumented K10 (privileged) | 0.555 | −0.297 | [−0.462, −0.109] | [−0.655, +0.110] | 0.06 | [−0.437, −0.067] | 0.540 |

* **Adding VLM features:** this never helped. Every `+vlm` model is at or below its VLM-free counterpart.
* **Post-hoc λ:** the best post-hoc λ for any `+vlm` model reaches 0.833, below epoch.
* **K=10 alone:** this ranks far worse than epoch, whether instrumented or VLM-judged. That matches Experiment 3: 10
  rollouts are too noisy among near-equal checkpoints.

### 5.3 Late half only (plateau regime, 48 checkpoints)

* Epoch drops to 0.367 (0.481 / 0.300 / 0.320).
* The held-out run's own 25 rollouts give **−0.179**, so HA and HB rankings disagree inside the plateau: what remains to
  rank is noise.
* Action L1 alone and RA(epoch, Action L1) reach 0.431, Δ +0.064 in 3/3 runs:

  | score | run-level CI of Δ | block-boot CI of Δ | P(Δ>0) |
  |---|---|---|---|
  | Action L1 alone | [+0.012, +0.100] | [−0.198, +0.420] | 0.76 |
  | RA(epoch, Action L1) | [+0.032, +0.100] | [−0.141, +0.322] | 0.79 |

  **The half swap reverses it** (epoch 0.153, L1 0.075, RA 0.108).
* Every learned model collapses (0.05–0.17), because it is fitted to plateau noise.
* Treat the L1 plateau signal as a hypothesis for a larger pool, not a finding.

### 5.4 Feature weights and their stability

The weights below are normalised to unit length and come from `weight_stability` and `weight_bootstrap` in the JSON.

* **What the models learn:** a near-uniform positive weight on everything that falls monotonically with training:
  * epoch;
  * train_loss;
  * replan overlap;
  * boundary jump;
  * Action L1;
  * open-loop and PLV-1 smoothness;
  * dynamics uncertainty;
  * teacher-forced latent MSE.

  For example, pairlogit_rank[all] puts 0.11–0.46 on each of these in every fold, most of them 0.28–0.35.

  At fixed λ = 0.1 the cross-fold cosine similarity is 0.96–0.99 (all93). The weight vector is therefore stable, but it is
  essentially "the average of training-progress proxies", which ranks almost exactly like epoch. In the pairlogit_rank[all]
  bootstrap, epoch, train_loss, replan overlap and teacher-forced latent MSE keep their sign in ≥ 0.92 of replicates.
* **Unstable features:**
  * **val_loss** changes sign across folds in most models (pairlogit_rank[all]: −0.005 / +0.380 / +0.033; bootstrap
    same-sign 0.42–0.65).
  * **support_drift** behaves the same way.
  * On eval33, Action L1 flips sign on the can_seed2 fold whenever the inner CV picks a tiny λ there:
    * ridge_rank[all]: +0.31 / −0.65 / +0.29, nested cross-fold cosine 0.57;
    * pairlogit_rank[all]: +0.31 / −0.55 / +0.31, cosine 0.62.
* **VLM weights** (eval33 `all+vlm`): both are positive in every fold. Nemotron is the more stable of the two (bootstrap
  same-sign 0.81–0.99). In ridge_rank[epoch+vlm], Cosmos v6 flips sign on the can_seed1 fold (−0.12; same-sign 0.31).
* **Regularisation is not identifiable from 2 training runs:**
  * The inner leave-one-run-out picks λ from opposite ends of the grid in different folds. For example, pairlogit_z[all]
    picks 1000 / 10 / 0.001 and ridge_rank[all] picks 1000 / 0.1 / 1000.
  * The post-hoc sensitivity table shows held-out ρ is flat for λ ≥ 0.1 and worse for small λ. Strong shrinkage
    (≈ correlation-weighted averaging) is the safe choice.

## 6. Power and what can be claimed

* **Run-level CIs:** with 3 runs the run-level 95% interval is the [min, max] of the three fold results. An exact
  sign test cannot reach p < 0.05 even if a method wins in 3/3 runs (one-sided p = 0.125).
* **Effect size versus noise:** the observed best advantages (+0.004 to +0.009 in ρ) are an order of magnitude smaller
  than:
  * the checkpoint-block bootstrap spread (≈ ±0.05 upper, −0.4 lower);
  * the label-noise spread (episode-only CI half-widths 0.015–0.12).
* **Required runs:** a rough paired-t extrapolation from the 3 fold deltas would need about 8–17 runs for
  pairlogit_rank / ridge_rank[all]. That holds only if the effect is real, and the half-swap reversal suggests it is not.
* **Regret metrics:** epoch is already at zero Regret@30 and zero top-1 regret in all three runs, so no score can show an
  improvement there in this pool.

**Conclusion:** in this pool no learned combination of outcome-free signals is distinguishable from the free epoch
baseline, with or without VLM-judged K=10 rates. The point estimates are within label noise and reverse under a
half swap.

## 7. Limitations

* Three runs and two tasks only. LORO means fitting on 2 runs, and the can-vs-square heterogeneity is part of what is
  being tested. The runs are short (200–300 epochs) and nearly monotone, so this pool cannot probe the plateau or
  overfitting regime where the paper motivates PLV.
* **The HB target is noisy** (25 episodes per checkpoint; within-run HA–HB Spearman 0.37–0.79, and −0.18 in the late
  half).
* **Within-run normalisation is transductive.** It uses the whole held-out run's features, though no outcomes, so it
  presumes the full run is available at screening time.
* **The block bootstrap is pessimistic for learned models.** It can hand them training runs without the early learning
  curve, which a practitioner would normally have. The episode-only bootstrap is the less pessimistic companion and
  reaches the same conclusion.
* **Only linear models were tried.** No interactions were used and no non-linear learners, which would be even less
  identifiable with 3 runs.
* **Same data as earlier waves.** This is post-hoc on data that motivated the question ("epoch is hard to beat").

## 8. Files

* `scripts/hackathon/explore/z2_loro_hybrid.py`, `scripts/hackathon/explore/z2_loro_hybrid_tables.py`
* `results/hackathon/explore/z2_loro_hybrid.json`, `z2_loro_hybrid_episode_boot.json`, `z2_loro_hybrid_late.json`
* `results/hackathon/explore/z2_loro_hybrid_tables.txt`, `z2_loro_hybrid_late_tables.txt`
* logs: `data/outputs/hackathon_explore/z2_loro_hybrid/run_{main,epboot,late}.log`
