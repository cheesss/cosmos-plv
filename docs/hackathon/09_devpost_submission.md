# 09 — Devpost submission text (ready to paste)

## Project name

Cosmos-PLV: World-Model-Guided Robot Policy Evaluation

## Elevator pitch

Pick the robot checkpoint worth testing next. Free offline PLV screening comes
first; zero-shot NVIDIA judges then replace the hand-coded success detector
(Nemotron on Nebius Token Factory, Cosmos3-Edge on a local GPU).

## Track

Physical AI

## About the project

### Inspiration

Training a robot imitation-learning policy produces dozens of checkpoints, and
the validation loss barely tells them apart. The only reliable test is to run
each checkpoint on the robot, which is slow and on real hardware expensive. Our
earlier study of 1,365 checkpoints ("Cheap Metrics First, Replanning Diagnostics
Second") found that cheap offline metrics are useful screens. It also found that
a larger learned dynamics model did not make the ranking more reliable. So we
asked: can NVIDIA's physical-AI models see something in the robot's behaviour
that the cheap metrics miss, and at what cost?

### What it does

Cosmos-PLV is a working checkpoint-evaluation application for robot learning researchers:

1. **Screens every checkpoint cheaply** with the original PLV metrics. These are
   Action L1, open-loop smoothness, PLV-1 model-feedback smoothness and
   execution-matched replanning, each read from the checkpoint's own chunking
   configuration. The replanning metric is a single-sample relative of the
   chunk-consistency scores used for runtime failure detection (Sentinel's STAC,
   VLA-FAIL's ACC). PLV uses it offline to screen checkpoints.
2. **Runs real robot simulations** in robosuite/MuJoCo. Each rollout records
   synchronized video, simulator state and every action chunk the policy planned,
   and labels the outcome automatically: success, missed grasp, dropped,
   not placed.
3. **Asks NVIDIA models what physically happened:**
   * **NVIDIA Cosmos3-Edge** runs on a local RTX 3090. It answers task-free
     questions about the object, for example "the can is in the gripper" or "the
     can is back in the left bin". Success comes from the last frame; failure types
     come from eight keyframes and a fixed rule.
   * **NVIDIA Nemotron-3.5-Super-VL**, called live through **Nebius Token
     Factory**, watches the whole episode and returns a validated JSON failure
     report.
4. **Recommends which checkpoint to evaluate next and why.** It keeps the
   cheap-metric top 30% and ranks them by how often the models judged the recorded
   attempts successful.
   * The free "latest epoch" pick and each rule's measured regret are shown next to
     the recommendation. In this pool the two-stage rule does not beat the epoch
     pick, and the app says so.
   * Predicted failure modes are flagged as unreliable.
   * Verified simulator success is shown separately and is never used to choose.

A Streamlit dashboard exposes all of this live:
* re-run the original metric script on a checkpoint
* record a new rollout
* run Cosmos3-Edge on the GPU
* call Nemotron on Nebius
* compare checkpoints and inspect cost and latency

### How we built it

* **Two isolated Python environments**, which exchange MP4, NPZ and JSON files:
  * Environment A: PyTorch 2.1, robosuite 1.5, robomimic, the original PLV code.
  * Environment B: PyTorch 2.8, Transformers 5.21, the OpenAI client and Streamlit.
* **Checkpoints:** the paper's checkpoints were deleted after the study. We
  retrained a reduced Diffusion Policy pool with the unchanged original script:
  Can and Square PH, three runs, a checkpoint every 5–10 epochs, and 50 simulator
  rollouts per checkpoint.
* **`cosmos_plv` package (stages A–F):** checkpoint adapter, cheap metrics,
  rollout recorder, Cosmos and Nebius backends, a strict JSON validator, a
  state-based failure annotator, ranking statistics, and the recommendation step.
* **Protocol:** written down before any evaluation episode was scored. Prompt
  development used separate seeds and checkpoints, and no score was fitted on
  outcomes.

### NVIDIA models used

* `nvidia/Cosmos3-Edge`, revision `344d602b`, Reasoner tower, bf16, local GPU.
  It extracts per-keyframe physical object state.
* `nvidia/Nemotron-3.5-Super-VL` on Nebius Token Factory. It produces the
  video-level failure diagnosis.

### Nebius usage

Nebius Token Factory serverless inference:
* Endpoint: `https://api.tokenfactory.nebius.com/v1/`, OpenAI-compatible
  `chat.completions` with `video_url` inputs.
* Every call is logged with its request id, served model, token usage and
  latency in `results/hackathon/nebius_calls.jsonl`. The key is never logged.
* The dashboard makes live calls.

### Challenges

* **Cosmos narrated the plan instead of the video.** We gave Cosmos3-Edge a full
  rollout video together with the task goal. It answered "success" for 12 out of
  12 development episodes, although only 3 had succeeded. Its reasoning read like
  a robot narrating its intended plan ("I close my gripper… and carefully lower
  the can"). We fixed this with a design choice rather than prompt tweaking:
  * ask only task-free perception questions per frame ("where is the red can?");
  * derive grasp, lift, drop and success with a fixed rule.
* **A silent dead end in the world model.** Our first forward-dynamics run (385
  generations) showed no effect of the actions at all.
  * Reading the checkpoint's weights showed why: the `libero` action slot we had forced
    through the diffusers guard is **untrained** in the released Cosmos3-Edge. Only the
    10 slots on the model card are trained.
  * We retracted the run and switched to the trained Franka slot. A positive control
    first reproduced the model card's own example.
  * After that, the generated gripper does follow commanded and recorded motion
    directionally (Experiment 4b).
* **One consumer GPU for everything.** Training, rollouts and Cosmos all shared a
  single 24 GB RTX 3090. Out-of-memory events forced careful scheduling. Failed
  calls are logged as infrastructure failures, never as model answers.

### Results

Everything was measured on one RTX 3090. Details are in `docs/hackathon/07_experiment_results.md`.

* **Success detection from video works.** On 330 real rollout videos (33 checkpoints x 10 seeds), compared with the simulator's privileged labels:

  | judge | accuracy [95% CI] | balanced accuracy | failures called "success" |
  |---|---|---|---|
  | NVIDIA Nemotron-3.5-Super-VL via Nebius Token Factory | **0.964** [0.938, 0.979] | 0.927 | 9 of 67 (13%) |
  | NVIDIA Cosmos3-Edge, locally | **0.918** [0.884, 0.943] | 0.832 | 21 of 67 (31%) |
  | always answer "success" (base rate) | 0.797 | 0.500 | 67 of 67 |

  * Latency is 2–3 s per episode.
  * Accuracy is unchanged when clip length is neutralised.
  * When the two judges agree (94% of episodes), they are right 97% of the time.
* **Failure taxonomy does not work yet.** Both models score below a majority-class baseline on failure types, and they miss cans dropped into the wrong compartment.
* **PLV reproduction on a new pool.** We retrained 93 Diffusion Policy checkpoints with 4,650 simulator rollouts and ran the original PLV metrics.
  * Every metric ranks checkpoints well (within-run Spearman 0.76–0.80) and keeps a near-best checkpoint in its top 30%.
  * Validation loss is anti-correlated with success (−0.48), which is the original motivation.
  * Honest negative result: in these short, nearly monotone runs, no PLV metric beats the free "later epoch is better" baseline (0.815).
* **Where the evaluation cost really drops** (exploratory; every finding except the deterministic cascade rule was checked by an independent verifier):
  * adaptive rollout allocation needs about 2–3x fewer rollouts than uniform;
  * a time-horizon early abort saves 16–20% of robot time;
  * a local-Cosmos-first cascade uses 16% fewer Nebius calls at a small accuracy cost (0.952 vs 0.964 on 330 episodes; missed successes 3 → 8).
* **Cosmos world model (exploratory, Experiment 4b).** Through the trained Franka action slot, ±x/y/z commands move the generated gripper in a consistent direction per axis. On recorded policy rollouts, the action-dependent part of the generated motion points along the true motion (cos 0.78–0.85, 15 clips, 5 initial states). Can is strong; Square is weak.
* **Can a judge tell when a replan helps? (exploratory, pre-registered; r1).**
  * Setup:
    * At 400 replanning boundaries we restored the exact simulator state.
    * We executed both the new plan and the old plan's unexecuted tail.
    * Each branch was continued 4 times with shared random numbers.
  * The pre-registered primary was negative. Nemotron and Cosmos were at chance at
    predicting when the new plan helped (AUROC 0.49–0.54 on boundaries whose outcome was
    not already decided).
  * Choosing between the two drawn paths did worse than always replanning.
  * PLV's disagreement score flagged fragile episodes, not the replan that mattered. A
    low-disagreement boundary in the same episode changed the outcome just as often.
  * With 4 continuations, good and bad replans could not be told apart from rollout noise
    by any method.
  * The one robust effect: replanning reaches success about 3.5 steps sooner (80 vs 51
    events, p = 0.014).
  * We did not find prior work that audits a policy's own replans this way. The closest
    are same-state branching studies (Beyond Visual Quality, SVA, CMA).
* **Judge blind spot on strong checkpoints (exploratory, pre-registered; c1).**
  * Setup: 620 new Can rollouts from 31 checkpoints in three quality strata.
  * The share of failures Nemotron called "success" rose from 6/25 to 14/27 to 13/22.
    That is positive under the pre-registered rule, but fragile: two initial states
    supply most of the effect.
  * The cause is a change in the failure mix:
    * The judges never miss a failed grasp (21/21 caught).
    * They accept almost every failure that ends at the goal bin. All 33 Nemotron false
      successes ended there, 32 of them with the can in the wrong compartment.
    * Strong checkpoints fail mostly that way, so their judged success is inflated.
  * Telling Nemotron the compartment did not fix it (24/33 still accepted), although the
    difference is visible in its input.
  * Rule for users: before trusting a judged-success gap between strong checkpoints,
    audit a sample of accepted episodes for near-goal failures.
* **Self-training from judged successes did not improve a plateaued policy
  (exploratory, pre-registered; s1).**
  * We fine-tuned a 0.885-success Can policy on its own rollouts, selected by Nemotron,
    by simulator labels, or not filtered at all.
  * All arms stayed at 0.875–0.89 on 200 paired seeds: Nemotron −0.5 points
    [−4.0, +3.0], simulator labels +1.0 [−2.5, +4.5].
  * Four verifiers found no bug. The recipe did change the policy's behaviour, but not
    its success, so this run cannot rank label sources.
* **The redesigned follow-up (s2, pre-registered).**
  * Setup: weaker base (0.656), a human-demo positive control, an annealed fine-tune and
    500 paired seeds.
  * Results against continued training on the original demos:
    * 36 extra human demos: 0.938, +8.2 points [+5.2, +11.4]. So the recipe works.
    * The policy's own simulator-verified successes: 0.824, −3.2 [−7.0, +0.6].
    * The episodes Nemotron called successes: 0.752, −10.4 [−14.8, −6.0].
  * Nemotron accepted 58 of 149 failures, 44 of them with the can in the wrong
    compartment. Those whole failed episodes became 40% of the self-collected frames.
  * Post hoc, the Nemotron arm finished 7.2 points below the simulator-labelled arm, mostly
    through extra cans left in the wrong place in the goal bin.
  * Caveats: one training run per arm, dilution of the original demos under a fixed step
    budget, and judge clip lengths that follow the simulator's success signal.
  * Lesson: audit a zero-shot judge's false successes before using its verdicts as a
    training filter.
* **Cost.** All Nebius calls in the study (7,003, 0 failed) used about 14.6 M input tokens and 0.78 M output tokens. That is about $4.96 at the public catalog price; this is an upper bound, since some exploratory calls went to the cheaper Lightning model.

### Limitations

* This is simulation only: robomimic Can and Square, 3 short training runs. The paper's 1,365-checkpoint study was not re-run.
* Both NVIDIA models were used **zero-shot**, with no fine-tuning on our tasks. The scope tested is one simulator (robosuite), one robot (Franka Panda, which is common in robot-learning datasets, so not an unseen embodiment), two tasks and one main camera. We did not test whether this transfers to other robots or to real-robot video.
* VLM-based diagnosis still needs rollouts to produce videos. It replaces the success *detector*, not the rollouts.
* Rollout-free world-model feedback is not ready. Through its trained Franka slot, Cosmos3-Edge moves the generated gripper roughly along the recorded motion: cos ≈ 0.8 on 15 clips from 5 initial states, strong on Can and weak on Square. But the motion is compressed and pixel fidelity is no better than a frozen frame (exploratory Experiment 4b).
* Prompt variants were chosen on a small development split (36 episodes). The exploratory studies are post-hoc, except r1 and c1, which were pre-registered (r1 with disclosed amendments and an interim look).
* VLM success judging is known to degrade on contact-rich assembly and to lean toward "success" when evidence is ambiguous (FailBench, 2026). Square is such a task.
* Prior art. PLV's replanning metric is close to published chunk-consistency monitors (Sentinel's STAC, 2024; VLA-FAIL's ACC and ActProbe's TCE, 2026). Our use differs: offline checkpoint screening, not runtime alarms.
* Cosmos runs locally, not on Nebius: no Cosmos model is served on Token Factory, and we did not provision paid AI Cloud GPUs.

### What's next

* Post-training the Cosmos3 generator on robosuite / robomimic data, or using a
  LIBERO-post-trained checkpoint, so that predicted futures can replace some
  rollouts. Zero-shot, the trained Franka slot is directionally right but too
  compressed (Experiment 4b).
* Real-robot videos, where no privileged success signal exists. This is where
  visual diagnosis matters most.
* Running Cosmos itself on Nebius AI Cloud GPUs.
* Larger checkpoint pools and more tasks.

## Built with

python, pytorch, nvidia-cosmos (Cosmos3-Edge), nvidia-nemotron
(Nemotron-3.5-Super-VL), nebius-token-factory, hugging-face-transformers,
robosuite, mujoco, robomimic, diffusion-policy, streamlit, playwright, ffmpeg,
numpy, pandas, matplotlib

## Try it out

* Repository: https://github.com/cheesss/cosmos-plv (MIT license)
* `bash scripts/hackathon/smoke_test.sh` (end-to-end smoke test)
* `streamlit run dashboard/app.py` (demo)

## What changed after 2026-08-26

Before the submission period the repository held the PLV research code and the
manuscript. The commits between 2026-08-26 and 2026-10-02 changed only the
manuscript, figures and project site. Everything Cosmos-, Nebius- and
application-related was built during the hackathon on branch
`hackathon/cosmos-plv-2026`:
* the `cosmos_plv` package
* the rollout recorder with verified outcome labels
* the NVIDIA Cosmos3-Edge integration and the Nebius Token Factory integration
* the validated diagnostic schema and prompt protocol
* real-feedback replanning diagnostics
* the hybrid screening experiment
* the dashboard, tests, figures and the demo video

The checkpoint pool was also newly retrained. Historical paper numbers are quoted
only as history. Details: `docs/hackathon/08_original_vs_new_work.md`.

## Feedback on Nebius and NVIDIA tools

**Nebius Token Factory**
* What worked:
  * The OpenAI-compatible API worked on the first try with the standard `openai` SDK. We moved from key to first successful call in minutes.
  * The public model catalog endpoint (`/api/public/models_info`) let us check model ids, licenses, regions and prices programmatically.
  * Video input as a `video_url` data URI worked for 20-second MP4 clips, with about 3 s latency for Nemotron-3.5-Super-VL. Over 7,003 calls we saw 0 API errors.
* What could be better:
  * For a Physical AI track, the main gap is that no Cosmos model is served. The catalog lists Cosmos3-Super-Reasoner as retired on 2026-08-31, so we had to run Cosmos locally.
  * The VL model sometimes returned invalid JSON. For example, it left a string value unquoted in 6 of 36 image calls. A JSON-schema or structured-output mode would help.
  * The model is listed in us-north1 only. It was not obvious from the docs whether the global endpoint would serve it (it did).

**NVIDIA Cosmos3-Edge**
* What worked:
  * It is small, ungated and actually runs on a consumer Ampere GPU in bf16, using about 5 GB for an image call. Transformers 5.x support worked out of the box.
  * The per-image, task-free perception answers ("where is the can?") are good, and robust to JPEG compression and down-scaling.
* What could be better:
  * When the prompt also stated the task goal, the model narrated the *intended* plan in the first person ("I close my gripper… and lower the can") instead of describing the video. It called 12 of 12 dev episodes successful when only 3 were. Good prompting guidance for "judge, don't plan" would help robot evaluators.
  * Thinking mode often used up the token budget without producing the final JSON.
  * There were recurring small JSON slips, such as closing an object with `]`.
  * Some docs disagreed with the Hub: docs.nvidia.com describes the checkpoints as gated, while Hugging Face does not gate them.
  * Which action slots are trained is not documented, and nothing fails when an untrained one is used. The released Cosmos3-Edge has untrained `libero` and `robocasa` slots, which we found only by reading the weights. A trained-domain list in the model card, plus a hard error on untrained slots, would have saved a day. The action frame and normalization for robosuite-style Panda setups are also undocumented.

**NVIDIA Nemotron-3.5-Super-VL**
* What worked: it was the most reliable judge of whole-episode outcomes. Its explanations are often specific ("the gripper touched the can but never closed around it"). They are not always right, though: in one no-grasp episode it described a pick-up that never happened. So the app shows the simulator truth next to every explanation and trusts only the binary verdict.
* What could be better: fine-grained failure categories (dropped vs. not placed) remain weak.

**Nemotron-3_5-Lightning**
* We tried it for auto-written evaluation reports. It rarely invented numbers, but a manual audit found false statements in 5 of 16 reports. Its recommendation answer was constant, so we kept it out of the app.
