# Related work and what is new here (Cosmos-PLV)

Moved here from the README on 2026-10-30 so the README stays short. Prior-art search: `docs/hackathon/explore/` and the r1 / c1 protocols.


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

