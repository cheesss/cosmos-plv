# Feedback for Nebius and NVIDIA (Cosmos-PLV, Nebius x NVIDIA Global AI Hackathon 2026)

This is grounded in our logs and tests:
* `results/hackathon/nebius_calls.jsonl`
* `results/hackathon/explore/nebius_calls_explore.jsonl`
* `docs/hackathon/05_nebius_execution.md`
* `docs/hackathon/04_cosmos_integration.md`
* the c1 and s2 protocols in `docs/hackathon/explore/`

Totals are from the call logs: 7,005 logged Nebius calls with 0 failed. The estimated
cost is about $4.96 at the public list price; this is an estimate from logged tokens, not
an invoice.

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
