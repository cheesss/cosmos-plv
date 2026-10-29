# Cosmos-PLV: World-Model-Guided Robot Policy Evaluation

**Nebius x NVIDIA Global AI Hackathon 2026 — Physical AI track**

Cosmos-PLV helps a robotics researcher decide **which imitation-learning
checkpoint to evaluate next, and why**. It combines lightweight offline
checkpoint screening (PLV, Policy-in-the-Loop Latent Validation, from our prior
research) with real simulator rollouts and physical-outcome judgements from NVIDIA
models: **NVIDIA Cosmos3-Edge** on a local GPU, and **NVIDIA
Nemotron-3.5-Super-VL** on **Nebius Token Factory**.

**TL;DR**
* **Workflow:** free offline PLV metrics pick which checkpoints get robot time.
  Zero-shot NVIDIA judges then replace the hand-coded success detector, and every
  call is logged.
* **Measured:** on 330 recorded simulator rollouts, success accuracy is 0.964 for Nemotron on
  Nebius and 0.918 for Cosmos3-Edge running locally, with no fine-tuning. Ranking
  checkpoints with VLM verdicts is as good as with privileged simulator labels.
  Adaptive allocation needs 2–3x fewer rollouts, and early abort saves 16–20% of
  robot time.
* **Not (yet):** it does not beat "pick the latest epoch" in these short runs, and
  it does not classify failure types. It cannot yet replace rollouts with
  predicted video. In an exploratory test, Cosmos3-Edge's trained Franka action
  slot moved the generated gripper roughly along the recorded motion (cos ≈ 0.8 on
  15 clips from 5 initial states; strong on Can, weak on Square). The motion is
  compressed, though, and pixel fidelity is no better than a frozen frame
  (Experiment 4b).
* **Replanning audit (r1, pre-registered):** we restored the exact simulator state at 400
  replanning boundaries and executed both the old and the new plan. Zero-shot judges
  were at chance at telling when a replan helped. PLV's disagreement score flagged
  fragile episodes, not the replan that mattered.
* **Judge blind spot (c1, pre-registered):** zero-shot judges accept almost every failure
  that ends at the goal bin, here a can in the wrong compartment, even when told the
  compartment. Strong checkpoints fail mostly that way, so their judged success is
  inflated.
* **Self-training from judged successes (s1, s2, pre-registered):** this did not improve
  the policy. Filtering by Nemotron made it 10 points *worse* than plain continued
  training, because the blind spot let failures into the data. 36 extra human demos
  gave +8 points.

![architecture](media/hackathon/architecture.png)

## The problem

Low training or validation error does not imply high closed-loop task success.
Teacher-forced validation scores actions on recorded expert observations; at
deployment the policy's own actions change what it sees next. Checkpoints with
similar offline losses can behave very differently on the robot, and every
real evaluation rollout is expensive. Our earlier study ("Cheap Metrics First,
Replanning Diagnostics Second", 1,365 checkpoints, see
[docs/hackathon/08_original_vs_new_work.md](docs/hackathon/08_original_vs_new_work.md))
found that cheap metrics such as Action L1 and open-loop action smoothness are
useful screens, but that a bigger learned dynamics model did not make ranking
more reliable.

The question for this hackathon: **can NVIDIA physical-AI models add useful
diagnostic information beyond cheap offline metrics, at practical cost?**

## Related work and what is new here

Prior work this builds on:
* **VLM success detection:** SuccessVQA ([Du et al. 2023](https://arxiv.org/abs/2303.07280)).
* **Autonomous real-robot policy evaluation:** [AutoEval](https://arxiv.org/abs/2503.24278)
  (CoRL 2025) fine-tunes a PaliGemma success classifier per scene on about 1,000
  labelled images.
* **Simulation as an evaluation proxy:** [SIMPLER](https://arxiv.org/abs/2405.05941).
* **Policy evaluation inside learned world models:**
  [WorldGym](https://arxiv.org/abs/2506.00613) and
  [Ctrl-World](https://arxiv.org/abs/2510.10125), both trained on robot data.
* **Failure reasoning, which these works find hard:**
  [REFLECT](https://arxiv.org/abs/2306.15724), [AHA](https://arxiv.org/abs/2410.00371),
  and RoboFail in [Cosmos-Reason1](https://arxiv.org/abs/2503.15558).
* **Reliability of VLM judges:** [FailBench](https://arxiv.org/abs/2609.03611) (near
  chance on contact-rich assembly, biased toward "success"),
  [PathEval](https://arxiv.org/abs/2411.18711) (choosing between drawn paths),
  [Robometer](https://arxiv.org/abs/2603.02115), and coverage shift in
  [Guan et al. 2026](https://arxiv.org/abs/2610.24801) (better policies leave failures
  that are harder to see).
* **Chunk-consistency signals, the closest prior art to PLV Level 3:**
  [Sentinel/STAC](https://arxiv.org/abs/2410.04640) (CoRL 2024),
  [VLA-FAIL/ACC](https://arxiv.org/abs/2606.21386) (2026) and
  [ActProbe/TCE](https://arxiv.org/abs/2606.08508) (2026). All three compare the
  overlapping part of consecutive action chunks and raise runtime failure alarms.
  None screens checkpoints. [SAFE](https://arxiv.org/abs/2506.09937) finds the
  single-sample STAC variant weak on a real robot.
* **Replan-or-keep rules:** [BID](https://arxiv.org/abs/2408.17355),
  [SGAC](https://arxiv.org/abs/2510.12392) (Diffusion Policy on robomimic Can and Square),
  [CARE](https://arxiv.org/abs/2609.32550) and [RTC](https://arxiv.org/abs/2506.07339).
  They are evaluated by overall success, not per boundary.
* **Same-state counterfactual evaluation:**
  [Beyond Visual Quality](https://arxiv.org/abs/2609.24745),
  [SVA](https://arxiv.org/abs/2607.03751), [CMA](https://arxiv.org/abs/2609.27247), and
  the common-random-numbers vine estimator in [TRPO](https://arxiv.org/abs/1502.05477).
* **Budget-aware policy selection:**
  [Active Offline Policy Selection](https://arxiv.org/abs/2106.10251),
  [Anwar et al. 2025](https://arxiv.org/abs/2502.09829),
  [Snyder et al. 2025](https://arxiv.org/abs/2503.10966),
  [RoboART](https://arxiv.org/abs/2502.06575).
* **Rollout-free checkpoint ranking:** [WorldEval](https://arxiv.org/abs/2505.19017),
  [RoboFQE](https://arxiv.org/abs/2610.18699), [CI-MSE](https://arxiv.org/abs/2606.29898).

We do not claim the idea of a VLM judge, a world-model evaluator, or a chunk-consistency
signal. Other projects also use Nemotron and Cosmos to judge rollout videos. What is new:
1. **Integration.** The judges sit after an offline checkpoint screen (PLV) in one
   logged workflow, and that workflow runs in the app.
2. **Zero-shot, open NVIDIA models.** There is no per-scene fine-tuning or labelling.
   Note that this is simulation, while AutoEval runs on a real robot.
3. **Effect on the decision itself.** We measure how zero-shot VLM verdicts change the
   checkpoint pick at an equal rollout budget, not only judge accuracy. Budget-aware
   selection itself is established (see above). What we add is VLM-judged outcomes in
   place of privileged success labels.
4. **Pitfalls that we found and controlled for.**
   * Clip-length leakage.
   * A goal-prompted model narrating its plan instead of the video ("judge, don't
     plan").
   * Camera choice mattering more than image quality.
   * An untrained action slot in the released Cosmos3-Edge checkpoint.
5. **A zero-shot check of Cosmos3-Edge action conditioning on another simulator's
   robot** (Experiment 4b).
6. **A same-state audit of the policy's own replans (r1).** We did not find prior work
   that does all of the following:
   * restores the exact simulator state at a replanning boundary;
   * executes both the old plan's unexecuted tail and the new plan, with shared random
     numbers;
   * uses the outcome difference to score a chunk-consistency metric and zero-shot VLM
     judges on the same events.

   The result is negative for the judges (see Results).

The earlier paper and its public site:
[cheesss.github.io/plv-checkpoint-screening](https://cheesss.github.io/plv-checkpoint-screening/).

## What the application does

1. **Load a Diffusion Policy checkpoint** and read its execution contract from the
   checkpoint itself (horizon 16, 2 observation steps, 8 executed actions).
2. **Cheap screening** with the original PLV scripts: Action L1 (Level 1),
   open-loop smoothness and PLV-1 model-feedback smoothness (Level 2),
   execution-matched replanning overlap / boundary jump (Level 3).
   Level 3 is a single-sample, per-dimension-normalized relative of the chunk-consistency
   scores used for runtime failure detection (STAC in
   [Sentinel](https://arxiv.org/abs/2410.04640), ACC in
   [VLA-FAIL](https://arxiv.org/abs/2606.21386)). PLV aggregates it offline per checkpoint
   for screening, instead of raising per-episode alarms.
3. **Real rollouts** in robosuite 1.5 (MuJoCo): agentview video, privileged
   simulator state and every predicted action chunk, with verified outcome labels
   (`success`, `no_grasp`, `dropped`, `not_placed`, `other`).
4. **Physical diagnostics** from video:
   * Cosmos3-Edge answers task-free questions about the object ("where is the
     can?"). Success is judged from the last frame. For the failure type, 8 keyframes
     go through a fixed grasp / lift / drop rule.
   * Nemotron-3.5-Super-VL on Nebius watches the episode video and returns a
     schema-validated failure report.
5. **Recommendation**: keep the cheap-metric top 30 %, then rank by VLM-judged
   success of recorded attempts. The free "latest epoch" pick and each rule's
   measured regret are shown next to it. In this pool the two-stage rule does
   **not** beat the epoch pick (Experiment 3), so the app says so. Verified
   simulator success is shown separately and is never used to choose.

Everything is traceable: checkpoint sha256, env seed, diffusion sampling seed,
model id + revision, Nebius request id, latency and token usage.

## NVIDIA and Nebius usage (exact)

| Component | What | Where it runs | Purpose |
|---|---|---|---|
| `nvidia/Cosmos3-Edge` rev `344d602b128d1bbdacb43b08d0a3626f46343e29` (OpenMDW-1.1) | Reasoner tower, bf16, Transformers 5.21 | local RTX 3090 | per-keyframe physical state of the manipulated object |
| `nvidia/Nemotron-3.5-Super-VL` (OpenMDW-1.1) | OpenAI-compatible chat completions with `video_url` | **Nebius Token Factory** (`api.tokenfactory.nebius.com`) | video-level failure diagnosis (second opinion) |

Runtime evidence of every Nebius call (no secrets): `results/hackathon/nebius_calls.jsonl`,
summary in [docs/hackathon/05_nebius_execution.md](docs/hackathon/05_nebius_execution.md).
Why Cosmos is not called through Nebius: no Cosmos model is served on Token
Factory (catalog checked 2026-10-29); see
[docs/hackathon/04_cosmos_integration.md](docs/hackathon/04_cosmos_integration.md).

## Results

All numbers are measured on this machine; details and CIs in
[docs/hackathon/07_experiment_results.md](docs/hackathon/07_experiment_results.md).
Demo video: `media/hackathon/demo.mp4` (2 min 23 s; the dashboard clips are real screen recordings).

**Pool.** 93 retrained Diffusion Policy checkpoints (Can PH seeds 1 and 2, Square
PH seed 1), each with 50 training-time simulator rollouts (4,650 rollouts), plus
330 recorded evaluation rollouts with video (33 checkpoints x 10 seeds).

**Experiment 1 — cheap metrics (original PLV scripts) vs verified success**, mean
within-run Spearman over 3 runs:

| training epoch (free baseline) | replan overlap | open-loop smoothness | boundary jump | Action L1 | PLV-1 | validation loss |
|---|---|---|---|---|---|---|
| **0.815** | 0.801 | 0.788 | 0.782 | 0.772 | 0.759 | −0.481 |

* Every PLV-family metric keeps a near-best checkpoint in its top 30% (Regret@30 0.007).
* Validation loss is anti-correlated with closed-loop success, which reproduces
  the paper's motivation.
* **No PLV metric beats the free "later epoch is better" baseline in these short,
  nearly monotone runs.** Replanning disagreement falls almost monotonically with
  training (ρ with epoch 0.96–1.00).

**Experiment 2 — can an NVIDIA VLM tell success from failure in a real rollout video?**
(330 episodes, privileged simulator labels as ground truth, prompts frozen on a
separate dev split)

| judge | accuracy [95% CI] | balanced | latency |
|---|---|---|---|
| NVIDIA Cosmos3-Edge, local RTX 3090 (last-frame perception) | 0.918 [0.884, 0.943] | 0.832 | 2.1 s |
| NVIDIA Nemotron-3.5-Super-VL via **Nebius Token Factory** (video) | 0.964 [0.938, 0.979] | 0.927 | 3.1 s |
| same, length-neutral clips (robustness) | 0.961 [0.934, 0.977] | 0.909 | 3.6 s |

* The two judges agree on 94% of episodes, and are 97% accurate when they agree.
* Failure *types* (no grasp / dropped / not placed) are **not** solved: both judges
  score below a majority-class baseline.
* Misses include cans dropped into the wrong compartment of the right bin.

**Experiment 3 — hybrid screening.**
* Scores from 10 extra rollouts rank near-equal checkpoints worse than the free
  cheap scores (pick regret 0.06–0.09 vs 0.007). This holds whether success comes
  from the simulator or from a VLM.
* VLM-judged rollouts are as good as privileged simulator labels. So the visual
  judge can stand in for a success detector where none exists, such as on a real
  robot.

**Exploratory follow-ups** (post-hoc; all except the simple cascade rule were checked by an independent verifier;
[docs/hackathon/11_explorations.md](docs/hackathon/11_explorations.md)):
* adaptive rollout allocation needs about 2–3x fewer rollouts than uniform on this pool (the idea follows
  active policy evaluation such as [AOPS](https://arxiv.org/abs/2106.10251));
* a time-horizon early abort saves 16–20% of robot time;
* a local-first cascade (accept local Cosmos failures, ask Nemotron otherwise) uses 16% fewer Nebius calls
  at a small accuracy cost (0.952 vs 0.964; missed successes 3 → 8);
* camera placement matters, image quality does not;
* LLM-written reports were not reliable, so the app shows only measured values.

**Experiment 4 — Cosmos world model (exploratory).**
* Our first run went through Cosmos3-Edge's `libero` action slot, which turned out to
  be untrained in the released checkpoint. That run is retracted.
* Through the trained Franka slot, ±x/y/z commands move the generated gripper in a
  consistent direction per axis.
* On 15 clips from recorded policy rollouts, the action-dependent part of the generated
  motion points along the true motion (cos 0.78–0.85).
* Can is strong. Square is weak: its end-point error is worse than with no actions.
* Motion is compressed to about 0.4x, the evidence covers only 5 initial states, and
  pixel fidelity is no better than a frozen frame.
* So it cannot yet replace rollouts. Details:
  [07 §C3](docs/hackathon/07_experiment_results.md),
  [04](docs/hackathon/04_cosmos_integration.md).

![E4b](media/hackathon/cosmos_fd_axis_sweep.png)

**r1 — can a judge tell when a replan helps? (exploratory, pre-registered).**

*Setup.*
* 300 replanning boundaries: the one with the largest disagreement in each eval
  episode, plus 100 low-disagreement controls.
* At each boundary we restore the exact simulator state.
* Two branches:
  * the new plan's 7 steps;
  * the old plan's unexecuted 7 steps.
* Each branch is continued 4 times with the same checkpoint and shared random numbers.
* Benefit = success rate of the new branch − success rate of the old branch.
* Protocol, amendments and disclosed deviations:
  [r1 protocol](docs/hackathon/explore/r1_counterfactual_replan_protocol.md).

*Results.*
* **The pre-registered primary is negative.**
  * The Nemotron perception rule does not beat PLV's own disagreement score d (AUROC 0.40
    vs 0.79, difference −0.39 [−0.47, −0.30]).
  * The 0.40 comes from intended releases after success. On the 223 boundaries whose
    outcome was not already decided, it is 0.49, i.e. chance.
  * Nemotron's event question (0.49) and the Cosmos rule (0.54) are at chance too.
  * Asked to choose between the two drawn paths, Nemotron abstained on 29 of 52 events
    that mattered. When it did choose, it was right 61% of the time; always taking the
    new plan was right 70% of the time on those events.
* **d does not localize the replan that matters.**
  * In the same episodes, a low-d control boundary changed the outcome as often as the
    max-d boundary (19 vs 18 of 100).
  * d mostly flags fragile episodes: it predicts the recorded episode's failure at
    AUROC 0.87.
* **With 4 continuations per branch, good and bad replans are indistinguishable from
  rollout noise for any method** (permutation test P = 0.26).
* **One robust effect.** When all 8 continuations succeed, the new plan gets there
  sooner in 80 vs 51 events (p = 0.014, about 3.5 steps).

**c1 — do judges miss more failures of stronger checkpoints? (exploratory, pre-registered).**

*Setup.* 620 new Can rollouts from 31 checkpoints in three quality strata.

*Results.*
* **The pre-registered primary is positive, but fragile.**
  * Nemotron accepted 6/25, 14/27 and 13/22 failures as successes, from the weakest to the
    strongest stratum.
  * AUROC of checkpoint quality is 0.67 [0.56, 0.75] with checkpoint clusters. Cosmos:
    8/25, 14/27, 19/22.
  * With seed clusters the CI includes 0.5, because two initial states supply 23 of the
    33 false successes.
* **Cause: the failure mix shifts. Failures do not get harder to see.**
  * The judges rejected every no-grasp failure (21/21). Those failures vanish as quality
    rises (15, 6, 0).
  * Failures that end at the goal bin are accepted at every quality level.
  * All 33 Nemotron false successes ended at the goal bin; 32 were in the wrong
    compartment.
* **Telling Nemotron the compartment did not help.** It still accepted 24 of 33 such
  cans. Only 2 of its rejections cited the compartment, although the difference is
  visible in its 320 px input.
* **Precision of judged successes rises with quality** (0.85 → 0.90 → 0.97).
* **Rule we take from it:** judged success overstates true success by about the rate of
  near-goal failures. Before trusting a gap between strong checkpoints, audit a sample of
  accepted episodes.
* Protocol, amendment and deviations:
  [c1 protocol](docs/hackathon/explore/c1_coverage_shift_protocol.md).

**s1 — can judged successes improve a plateaued policy? (exploratory, pre-registered).**

*Setup.* A Can policy trained on 20% of the demos (success 0.885) was fine-tuned on its
own 200 rollouts. The rollouts were selected by Nemotron, by simulator labels, or not
filtered at all.

*Results.*
* **No gain:** all arms scored 0.875–0.89 on 200 paired seeds.
  * Nemotron vs continuing on the original demos: −0.5 points [−4.0, +3.0].
  * Simulator labels: +1.0 [−2.5, +4.5].
* **No bug found.** Four verifiers checked the pipeline. The fine-tune changed behaviour
  (weights moved 5–9%, and failure types shifted), but not success.
* **It cannot rank label sources.** Even unfiltered data, with 32% failure frames, did not
  hurt.

**s2 — the same question, redesigned (exploratory, pre-registered).**

*Setup.*
* Base: an earlier checkpoint of the same run (success 0.656).
* Positive control: 36 extra human demos.
* Fine-tune with an annealed learning rate. 500 paired eval seeds.

*Results.*

| arm | success | vs continue [95% CI] |
|---|---|---|
| continue on the 36 demos | 0.856 | — |
| + 36 human demos (positive control) | **0.938** | +8.2 points [+5.2, +11.4] |
| + own successes, simulator labels | 0.824 | −3.2 [−7.0, +0.6] |
| + own episodes Nemotron calls successes | **0.752** | **−10.4 [−14.8, −6.0]** |

* **The recipe can raise success, but the policy's own successes did not.** Human demos
  did. With simulator labels the comparison stays inconclusive.
* **Nemotron-filtered self-training was clearly worse.**
  * Nemotron accepted 58 of 149 failures. 44 of those had the can in the wrong
    compartment, the c1 blind spot. Kept whole, they were 40% of the self frames.
  * Post hoc: the Nemotron arm ends 7.2 points below the simulator-labelled arm
    [−11.2, −3.2]. Its extra eval failures are mostly cans in the wrong place in the goal
    bin (46 vs 19).
  * The protocol made this comparison interpretable only if simulator-labelled
    self-training beat continue, which it did not.
* **Caveats.**
  * One training run per arm.
  * A fixed step budget dilutes the original demos (10% of samples in the oracle arm).
  * Judge clip length follows the simulator's success signal.
* Protocol and results: [s2 protocol](docs/hackathon/explore/s2_self_improve_protocol.md).
* Protocol and results: [s1 protocol](docs/hackathon/explore/s1_self_improve_protocol.md).

| Figure | |
|---|---|
| ![B](media/hackathon/checkpoint_comparison.png) | B — cheap metrics vs measured success per run |
| ![C](media/hackathon/robot_execution.png) | C — real rollouts with synchronized state |
| ![D](media/hackathon/cosmos_example.png) | D — actual Cosmos / Nemotron inputs and outputs |
| ![E](media/hackathon/cost_latency.png) | E — measured cost and latency |


## Install

Tested on Ubuntu 20.04, 1x RTX 3090 (24 GB), NVIDIA driver 570 (CUDA 12.8).
Isolated environments are used (the PLV stack and the Cosmos stack have
incompatible dependencies); they exchange MP4 / NPZ / JSON files. Environment C
is needed only for the exploratory world-model experiment.

```bash
git clone https://github.com/cheesss/cosmos-plv.git && cd cosmos-plv

# Environment A (PLV: training, metrics, robosuite rollouts) — Python 3.10
conda env create -f conda_environment.yaml   # original definition; or reuse an env matching
pip install -e .                              # configs/hackathon/requirements-plv-lock.txt
source scripts/hackathon/env.sh               # sets MUJOCO_GL=egl, LD_PRELOAD libffi fix, W&B offline

# Environment B (Cosmos + Nebius client + dashboard) — Python 3.11
python3.11 -m venv ~/.venvs/cosmos-plv
~/.venvs/cosmos-plv/bin/pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
~/.venvs/cosmos-plv/bin/pip install -r requirements-cosmos.txt
source scripts/hackathon/envb.sh

# Environment C (Cosmos3-Edge generator: action-conditioned forward dynamics, exploratory
# Experiment 4) — Python 3.12, diffusers main pinned to the commit in the lock file
python3.12 -m venv ~/.venvs/cosmos-gen
~/.venvs/cosmos-gen/bin/pip install -r configs/hackathon/requirements-cosmos-gen-lock.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128

# Nebius Token Factory key (never commit it)
mkdir -p ~/.config/cosmos-plv && cp .env.example ~/.config/cosmos-plv/nebius.env  # then fill NEBIUS_API_KEY
```

## Data and checkpoints

```bash
bash scripts/hackathon/download_datasets.sh          # robomimic v1.5 can/square PH, pinned revision + sha256
bash scripts/hackathon/train_checkpoint_pool.sh      # 3 runs with the ORIGINAL DP training script
python scripts/hackathon/make_pool_manifest.py       # results/hackathon/checkpoint_manifest.{csv,json}
```

The original paper checkpoints no longer exist (deleted 2026-09-28), so a reduced
pool was retrained: Can PH seeds 1 and 2, Square PH seed 1, a checkpoint every
5–10 epochs, each with 50 simulator rollouts on seeds 100000–100049 (the
repository's own protocol). See
[docs/hackathon/03_model_recovery.md](docs/hackathon/03_model_recovery.md).

## Smoke test (one command, ~2 min)

```bash
bash scripts/hackathon/smoke_test.sh
```

Runs the unit tests in both environments, a 1-step DP training smoke run, one
recorded rollout, one Cosmos3-Edge call and (if a key is configured) one Nebius
call.

## Demo (one command)

```bash
source scripts/hackathon/envb.sh && streamlit run dashboard/app.py
```

Tabs: **Recommendation** (evaluate-next answer), **Checkpoint & cheap metrics**
(re-run the original metric script live), **Rollouts & Cosmos** (watch real
rollouts, record a new one, run Cosmos3-Edge, ask Nemotron on Nebius),
**Compare**, **Cost & latency**.

## Reproduce the experiments

```bash
bash scripts/hackathon/compute_baseline_metrics.sh            # Experiment 1 (original scripts, paper flags)
python scripts/hackathon/analyze_baseline.py                  # -> results/hackathon/baseline_*.csv
python scripts/hackathon/record_eval_set.py --wait-for-pool   # 10 eval rollouts per eval checkpoint
python scripts/hackathon/run_cosmos_diagnostics.py --rollouts data/outputs/hackathon_rollouts_eval \
    --variant v7_perception_track --no-reasoning --keyframes 8 --out data/outputs/hackathon_cosmos/eval_local_v7.jsonl
python scripts/hackathon/run_cosmos_diagnostics.py --rollouts data/outputs/hackathon_rollouts_eval \
    --backend nebius --variant v2_evidence_first --no-reasoning --max-new-tokens 700 \
    --out data/outputs/hackathon_cosmos/eval_nebius_v2.jsonl
python scripts/hackathon/analyze_hybrid.py                    # Experiments 2 + 3
python scripts/hackathon/make_figures.py                      # media/hackathon/*.png
```

The protocol (splits, seeds, prompt-selection rule) was written before any
evaluation episode was scored: [docs/hackathon/06_experiment_protocol.md](docs/hackathon/06_experiment_protocol.md).

## Repository layout (hackathon additions)

```
cosmos_plv/            integration package (stages A-F)
scripts/hackathon/     setup, training, recording, diagnostics, analysis, figures, demo
dashboard/app.py       Streamlit application
tests/hackathon/       unit tests (no GPU needed)
docs/hackathon/        00-10 audit, protocol, results, status
results/hackathon/     machine-readable results (CSV / JSON / JSONL)
media/hackathon/       figures and demo video
configs/hackathon/     dataset manifest, environment locks
```

## Hardware requirements

* Training + rollouts: one CUDA GPU (we used an RTX 3090; each DP lowdim run
  needs ~2.3 GB) and a multi-core CPU (robosuite rollouts are CPU-bound).
* Cosmos3-Edge Reasoner: ~5 GB GPU memory for single images, ~8.4 GB peak
  for a 20 s video, bf16 (Ampere or newer).
* Nebius: an API key; no local GPU needed for that backend.

## License and attribution

* Code in this repository: MIT (see `LICENSE`; upstream Diffusion Policy
  © 2023 Columbia AI and Robotics Lab). The hackathon additions are released
  under the same MIT terms.
* Third-party components keep their own licenses: robomimic / robosuite (MIT),
  MuJoCo (Apache-2.0), Diffusion Policy (MIT), datasets from robomimic (not
  redistributed here).
* Model weights are not redistributed. `nvidia/Cosmos3-Edge` and
  `nvidia/Nemotron-3.5-Super-VL` are used under OpenMDW-1.1 as published by NVIDIA.
* Research: Hyun Joon Cho and Taehwan Yun, Sungkyunkwan University. Please cite
  Diffusion Policy (Chi et al., RSS 2023) and robomimic (Mandlekar et al., CoRL 2021).
