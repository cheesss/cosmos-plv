# 09 — Devpost submission text (ready to paste)

The same text is in `submission/devpost_final.md`; numbers are traced in `submission/claim_ledger.csv`.

# Cosmos-PLV: World-Model-Guided Robot Policy Evaluation

**Track:** Physical AI

**Short description (Devpost tagline):** Evaluate robot imitation-learning checkpoints with cheap screening and NVIDIA visual judges, running on a local GPU and on Nebius Token Factory, and audit when those judges cannot be trusted.

**Cosmos-PLV is for robot-learning researchers who train many checkpoints and cannot afford to test them all on a robot.**
* It screens every checkpoint cheaply.
* It runs real robosuite rollouts of the shortlisted ones.
* It judges each rollout video with two NVIDIA models: **Cosmos3-Edge** on a local RTX 3090, and **Nemotron-3.5-Super-VL** called live on **Nebius Token Factory**.
* It recommends what to test next, next to the free "latest epoch" baseline and each rule's measured regret.
* In the app you pick a checkpoint, watch the rollout, run both judges live, and see whether each verdict matches the simulator.

**What we found matters as much as what we built.**
* The judges are accurate overall: Nemotron 96.4% on 330 recorded rollouts, though it still called 9 of 67 failures a success.
* Pre-registered audits show where they fail. Nemotron accepted 32 of 33 failures in which the can ended in the wrong compartment of the goal bin.
* When we let it filter self-collected training data, it accepted 58 of 149 failures. The resulting policy reached 75.2% success, against 85.6% for plain continued training.
* The lesson for anyone using AI judges in robot learning: **audit what the judge systematically misses before you trust it to rank strong checkpoints or to label training data.**

**Demo video (2:45, narrated):** _[YouTube URL — added after the owner uploads `submission/demo_final.mp4`]_ · **Code:** https://github.com/cheesss/cosmos-plv (MIT) · **Try it without a GPU or API key:** `COSMOS_PLV_REPLAY=1 streamlit run dashboard/app.py` (recorded replay of real rollouts and real model answers)

*About the name.* Cosmos3-Edge serves as a visual judge. Its world model (action-conditioned generation) was explored, but it does not drive checkpoint choice.

![results at a glance](https://raw.githubusercontent.com/cheesss/cosmos-plv/main/submission/results.png)
![accurate overall, wrong where it matters](https://raw.githubusercontent.com/cheesss/cosmos-plv/main/submission/figures/accuracy_vs_blind_spot.png)

## Inspiration

Training a robot imitation-learning policy produces dozens of checkpoints, and validation loss barely tells them apart. The reliable test is to run each checkpoint on the robot, which is slow and, on real hardware, expensive. On a real robot there is also usually no programmable success detector: someone has to watch the videos.

Our earlier research (PLV, "Cheap Metrics First, Replanning Diagnostics Second", 1,365 checkpoints in 45 runs) found that cheap offline metrics are useful screens. We asked two questions:
- Can NVIDIA's physical-AI models take over the expensive part, judging what happened in each rollout, at practical cost?
- How far can their judgments be trusted?

## What it does

Cosmos-PLV is a working checkpoint-evaluation application, a Streamlit dashboard plus a logged pipeline, for robot-learning researchers.

1. **Screens every checkpoint cheaply** with the original PLV scripts: Action L1, open-loop smoothness, PLV-1 model feedback and execution-matched replanning overlap. No rollouts are needed.
2. **Runs real robot simulations** of the shortlisted checkpoints in robosuite / MuJoCo. Each run records video, simulator state and every action chunk the policy planned.
3. **Asks NVIDIA models what physically happened:**
   * **NVIDIA Cosmos3-Edge** runs on a local RTX 3090. It answers task-free per-frame questions ("where is the can?"), and a fixed rule turns those answers into grasp / drop / success.
   * **NVIDIA Nemotron-3.5-Super-VL**, called live through **Nebius Token Factory**, watches the episode video. It returns a schema-validated JSON verdict, and its request id, latency and tokens are logged.
4. **Recommends what to evaluate next**, and shows the free "latest epoch" baseline and each rule's measured regret beside it. In our pool the PLV-based rule does not beat that baseline, and the app says so.
5. **Warns when the judge cannot be trusted.** For strong checkpoints a "judge trust" warning appears, and an audit panel shows where the judges fail. Both come from our pre-registered audits.

Simulator ground truth is used only to score the judges. It never chooses a checkpoint.

## How we built it

* **Two isolated Python environments** that exchange MP4, NPZ and JSON files:
  * Environment A: PyTorch 2.1, robosuite 1.5, robomimic, the original PLV code.
  * Environment B: Transformers 5.21 for Cosmos3-Edge, the OpenAI client for Nebius, and Streamlit.
* **A retrained pool:** 93 Diffusion Policy checkpoints (Can PH seeds 1 and 2, Square PH seed 1), trained with the unchanged original script. Each checkpoint has 50 training-time rollouts (4,650 in all), and there are 330 recorded evaluation rollouts with video.
* **`cosmos_plv` package:**
  * checkpoint adapter and rollout recorder with verified outcome labels;
  * Cosmos and Nebius backends with strict JSON validation;
  * ranking statistics and the recommendation step.
* **Process:** every exploratory analysis was checked by an independent verifier. The four follow-up audits (r1, c1, s1, s2) were pre-registered before any data was collected, and their amendments and deviations are disclosed in `docs/hackathon/explore/`.

## What we measured

**Positive results**
* **Success judging from video works, zero-shot.** On 330 recorded rollouts (base rate 79.7% successes):

  | judge | accuracy [95% CI] | balanced accuracy | failures called "success" |
  |---|---|---|---|
  | Nemotron-3.5-Super-VL on Nebius | 0.964 [0.938, 0.979] | 0.927 | 9 of 67 |
  | Cosmos3-Edge, local | 0.918 [0.884, 0.943] | 0.832 | 21 of 67 |

  * With length-neutral clips accuracy is similar (0.961), but more failures are called success (12 vs 9 of 67).
  * Failure *types* (no grasp / dropped / not placed) are not solved: both judges score below a majority-class baseline.
* **Cheaper evaluation (exploratory, post-hoc, on logged rollouts):**
  * Adaptive allocation needs 2–3x fewer rollouts per checkpoint than uniform allocation to reach the same pick quality (excess regret ≤ 0.02 vs uniform allocation; post-hoc simulation on a 56-checkpoint snapshot with simulator labels; median ratio 3, range 2–4; the epoch prior, not PLV, is what helps).
  * Aborting failing rollouts at a time limit saves 16% of rollout execution time (summed simulated robot time over 200 evaluation episodes), with 2 of 141 successes aborted (leave-one-checkpoint-out); 20% if privileged simulator state is also used, which a real robot would not have.
  * A local-Cosmos-first cascade uses 16% fewer Nebius calls (278 vs 330) at a small accuracy cost (0.952 vs 0.964).
  * These save rollouts, robot time and API calls. They do not make training faster.

**Honest negatives**
* **PLV does not beat the free baseline here.** In these short, nearly monotone runs, no PLV metric beats "pick the latest epoch" (mean within-run Spearman 0.815 for epoch vs 0.759–0.801 for PLV metrics). PLV metrics still keep a near-best checkpoint in their top 30% (regret about 0.007). Here, later checkpoints were almost always better (replanning disagreement falls with epoch, ρ 0.96–1.00), so epoch is hard to beat. The earlier 1,365-checkpoint study was not re-run.
* **Cosmos is not a rollout replacement.** Through its trained Franka action slot, Cosmos3-Edge's generated motion follows the recorded motion only in direction (cos 0.78–0.85 on 15 clips from 5 initial states). Motion is compressed, Square is weak, and pixel fidelity is no better than a frozen frame. An earlier run through an untrained action slot is retracted.

## What surprised us: an accurate judge is not a safe source of labels

* **r1: replans (pre-registered).** We restored the exact simulator state at 400 replanning boundaries and executed both the old and the new plan.
  * The judges could not tell when a replan helped: AUROC 0.40–0.54, about 0.49 once already-decided events are excluded.
  * PLV's disagreement score flagged fragile episodes, not the replan that mattered. Low-disagreement boundaries in the same episodes changed the outcome just as often (19 vs 18 of 100).
* **c1: a systematic blind spot (pre-registered; 620 new Can rollouts, 31 checkpoints).**
  * Every Nemotron false success ended with the can in the goal bin: 32 of 33 in the wrong compartment and 1 on a divider.
  * Telling Nemotron the compartment still left 24 of 33 accepted.
  * Strong checkpoints fail mostly that way. The share of failures accepted rose from 6/25 to 14/27 to 13/22 with checkpoint quality. This is positive under our pre-registered rule, but fragile: two initial states supply most of the false successes.
* **s2: the consequence (pre-registered).** We fine-tuned a Can policy (0.656 on 500 paired eval seeds) on different data:

  | added data | success | vs continued training [95% CI] |
  |---|---|---|
  | none (continued training on the 36 demos) | 0.856 | — |
  | 36 held-out human demos | 0.938 | +8.2 points [+5.2, +11.4] |
  | own successes, simulator labels | 0.824 | −3.2 [−7.0, +0.6] |
  | own episodes Nemotron called successes | **0.752** | **−10.4 [−14.8, −6.0]** |

  * Nemotron accepted 58 of 149 failures, 44 of them in the wrong compartment, and they entered the training set.
  * Caveats:
    * One training run per arm.
    * The original demos are diluted under a fixed step budget.
    * Judge clip length follows the simulator's success signal.
    * The Nemotron-vs-simulator-labels gap (−7.2 points) is a post-hoc comparison.
  * An earlier attempt on a plateaued policy (s1) showed no change from any data source.

The practical conclusion: **visual success classification can be accurate overall and still be unsafe as an unverified source of robot-learning labels.** Audit what the judge systematically misses before using it to rank strong checkpoints or to train on its verdicts.

## Challenges

* **Cosmos narrated the plan instead of the video.** When told the task goal, Cosmos3-Edge described the robot's intended plan and called 12 of 12 dev episodes successful, although only 3 were. Asking task-free perception questions and applying a fixed rule fixed it.
* **A silent dead end in the world model.** The `libero` action slot we first used is untrained in the released Cosmos3-Edge. We found this only by reading the weights.
* **The judge blind spot.** It only showed up when we compared the judges against simulator labels in a pre-registered audit.
* **One 24 GB GPU for everything:** training, rollouts and Cosmos.
* **Simulation only.** robomimic Can and Square, three training runs, one camera.

## What is new during the hackathon (since 2026-08-26)

Before the submission period the repository held the PLV research code and manuscript. Everything Cosmos-, Nebius- and application-related was built during the hackathon:
* the `cosmos_plv` package and the rollout recorder with verified labels;
* the Cosmos3-Edge and Nebius integrations, with the validated diagnostic schema;
* the dashboard, including the judge-trust warning and the audit panel;
* the retrained 93-checkpoint pool;
* Experiments 1–3, the exploratory efficiency studies, the Cosmos world-model probe;
* the pre-registered audits r1, c1, s1 and s2.

Historical paper numbers are quoted only as history. See `docs/hackathon/08_original_vs_new_work.md`.

We do not claim the ideas of a VLM success judge, chunk-consistency monitoring, active evaluation or self-improvement. All have prior work: SuccessVQA, AutoEval, Sentinel, VLA-FAIL, Active Offline Policy Selection, SOAR, RoboCat, Large Reward Models, and Guan et al. 2026 on coverage shift. Our contribution is the integrated, logged workflow and the measured audits of when its judges fail.

## What's next (future work, not built)

* Real-robot videos, where no privileged success signal exists.
* More tasks and cameras.
* Spatially grounded judge prompts or verifiers for near-goal failures.
* Post-training a Cosmos world model on robosuite data, so predicted futures could replace some rollouts.

## Built with

python, pytorch, nvidia-cosmos (Cosmos3-Edge), nvidia-nemotron (Nemotron-3.5-Super-VL), nebius-token-factory, hugging-face-transformers, robosuite, mujoco, robomimic, diffusion-policy, streamlit, playwright, ffmpeg

## Nebius and NVIDIA usage

* **Nebius Token Factory** (`https://api.tokenfactory.nebius.com/v1/`, OpenAI-compatible `chat.completions` with `video_url`):
  * 7,005 logged calls with 0 failures;
  * about $4.96 estimated at the public list price (an estimate, not an invoice);
  * every request id is in `results/hackathon/nebius_calls.jsonl` and `results/hackathon/explore/nebius_calls_explore.jsonl`.
* **NVIDIA open models:**
  * `nvidia/Cosmos3-Edge` (revision `344d602b`), running locally;
  * `nvidia/Nemotron-3.5-Super-VL`, served by Nebius.
* No Cosmos model is served on Token Factory, so Cosmos runs locally.

## Try it out

* Repository: https://github.com/cheesss/cosmos-plv (MIT license)
* `bash scripts/hackathon/smoke_test.sh` (end-to-end smoke test)
* `streamlit run dashboard/app.py` (live demo: needs the checkpoint pool, the local Cosmos3-Edge weights and a Nebius key)
* `COSMOS_PLV_REPLAY=1 streamlit run dashboard/app.py` (recorded replay: no GPU, no API key; real recorded rollouts and model answers only)

## Feedback on Nebius and NVIDIA tools

**Nebius Token Factory**
* What worked:
  * The OpenAI-compatible API worked on the first try with the standard `openai` SDK. We moved from key to first successful call in minutes.
  * The public model catalog endpoint (`/api/public/models_info`) let us check model ids, licenses, regions and prices programmatically.
  * Video input as a `video_url` data URI worked for 20-second MP4 clips, with about 3 s latency for Nemotron-3.5-Super-VL. Over 7,005 calls we saw 0 API errors.
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

**Additional findings from the pre-registered judge audits (c1, s2)**
* **Nemotron-3.5-Super-VL blind spot.**
  * Of the c1 failures where the can ended in the wrong compartment of the goal bin,
    Nemotron called 32 of 33 a success.
  * Writing the compartment requirement into the prompt still left 24 of 33 accepted. Only
    2 of its rejections cited the compartment, although the difference is visible in the
    320 px input.
  * In s2, Nemotron accepted 58 of 149 failures into a self-training set; 44 of them
    were wrong-compartment cans.
  * A way to ask for spatially grounded checks would help robot evaluators, for example
    structured "which compartment" questions or region references.
* **Clip length can leak the outcome.** Our recorder stops 16 steps after success, and
  the prompt states the clip duration. A length-neutral variant scored about the same
  (0.961 vs 0.964), but evaluators should pad clips. This is our pipeline's issue, not
  the model's.
* **Cosmos3-Edge on black frames.**
  * An EGL rendering artefact in our recorder made 16 videos black after the first frame.
  * On those black last frames Cosmos3-Edge answered "right_bin" (success) every time,
    so its location answers can be ungrounded.
  * A "not visible" answer is available in our prompt and was not used.

**Nebius AI Cloud (GPU)**
* We did not use AI Cloud GPUs. Our runs fit on one local RTX 3090, and we chose not to provision paid GPU capacity. We cannot give feedback on it beyond that. Because no Cosmos model is served on Token Factory, a one-click way to run a Cosmos checkpoint on AI Cloud would have been the most useful next step for us.

**Would we build with these tools again?**
* **Nebius Token Factory: yes.** An OpenAI-compatible endpoint with video input, 0 failed calls out of about 7,000, and about 3 s per episode let us add a second, stronger judge without a second GPU.
* **Nemotron-3.5-Super-VL: yes, for binary success judging with an audit.**
  * It was our most accurate judge (0.964 on 330 rollouts).
  * We would not use its verdicts as training labels without checking near-goal failures.
* **Cosmos3-Edge: yes, as a free local first-pass judge** (0.918; a local-first cascade saves 16% of the Nebius calls). As a rollout-free world model it is not yet usable for this task without post-training.
