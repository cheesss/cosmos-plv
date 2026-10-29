# s1 — Can zero-shot Nemotron verdicts improve a plateaued policy? (VLM-filtered self-training)

**Status:** pre-registered on 2026-10-29 (KST). Written before any s1 rollout, label or
training step. This is an exploratory study. Any later change is listed under
"Amendments" with its date.

**Prior art.** The method is not new. The same loop (run the policy, judge success, keep
successes, retrain) appears in:
* SOAR ([2407.20635](https://arxiv.org/abs/2407.20635));
* RoboCat ([2306.11706](https://arxiv.org/abs/2306.11706));
* Large Reward Models ([2603.16065](https://arxiv.org/abs/2603.16065)), which uses
  progress-weighted behaviour cloning.

What is tested here is narrower: whether a **zero-shot** NVIDIA judge, with the label
errors measured in c1, is good enough to drive that loop on a policy that has stopped
improving, compared with the simulator's own labels.

## Base policy and why it was chosen

* **Base B:** Diffusion Policy, robomimic Can PH, the `20_percent_train` subset (36
  demos), seed 1, epoch 1000.
* **File:** `data/outputs/hackathon_pool_lowdata/paper_dp_can_abs_ph_20_percent_seed1_e1000_r25/checkpoints/epoch=1000-*.ckpt`.
* **Why this checkpoint:** training-time success stays between 0.80 and 0.90 from epoch
  425 to 1000, so more training on the same demos does not help. That leaves headroom
  that only new data could fill.

## Hypotheses

**H1 (primary).** Fine-tuning B on its own successes, as selected by Nemotron, gives
higher closed-loop success than the same fine-tuning on the original demos only.

**H2 (secondary).** Nemotron-selected data is worse than simulator-selected data. The
reason would be false successes (see c1), for example a can left in the wrong compartment
of the goal bin.

## Steps

1. **Collect.**
   * B runs for 200 episodes on new seeds 700000–700199.
   * The recorder settings match the Experiment 2 eval set: agentview at 448 px, 10 fps
     video, 16-step hold after success, sampling seed 0, max 400 steps.
   * Low-dim observations and executed absolute actions are saved too.
2. **Label every collected episode.**
   * Simulator truth (privileged).
   * Nemotron-3.5-Super-VL on Nebius, frozen `v2_evidence_first`. Settings are as in
     Experiment 2: no reasoning, max 700 tokens, 4 fps.
   * Nemotron, compartment-explicit `v2c_compartment` (c1 amendment A1). Same settings.
   * Cosmos3-Edge, frozen `v6_perception_last`. Its labels are reported but not used for
     any arm.
3. **Training data.**
   * Each selected episode is truncated at its first simulator success step (inclusive).
     Real demos also end at task completion.
   * An episode that a judge calls a success but that never succeeded is kept whole, up
     to 400 steps.

   Everything goes into one hdf5. It holds the original 36 train and 4 valid demos and
   the 200 collected episodes. Each arm is a filter key over that file:

   | arm | training episodes |
   |---|---|
   | `continue` | 36 original demos only |
   | `nemotron` (**H1 arm**) | 36 demos + episodes Nemotron v2 calls a success |
   | `oracle` | 36 demos + episodes the simulator calls a success |
   | `all` | 36 demos + all 200 episodes, unfiltered |
   | `nemotron_v2c` | 36 demos + episodes Nemotron v2c calls a success |

4. **Fine-tune** every arm with the same recipe.
   * Start from B's weights. The normalizer is kept fixed from B.
   * 8,000 gradient steps, batch 128, AdamW at a constant learning rate of 3e-5, weight
     decay 1e-6. The EMA update is that of the original config.
   * Training seed 1.
   * All arms train concurrently on the same GPU.
   * The EMA weights after step 8,000 are evaluated. There is no checkpoint selection.
5. **Evaluate** B and the five fine-tuned checkpoints.
   * 200 new seeds, 800000–800199, the same for every arm. No video.
   * Same recorder; success is the simulator's.

## Endpoints

**Primary.**
* The quantity is paired success(`nemotron`) − success(`continue`) on the 200 eval
  seeds.
* The 95% CI comes from a bootstrap over seeds (10,000 resamples, seed 0).
* The exact McNemar p is reported too.
* H1 is **positive** if the lower CI bound is above 0.

**Secondary** (all reported, whatever their sign):
1. `oracle` − `continue`. Does self-training help at all here?
2. `nemotron` − `oracle` (H2), and `nemotron_v2c` − `nemotron`.
3. `all` − `continue`. Does filtering matter?
4. B vs `continue`. The effect of the fine-tuning recipe itself.
5. Label quality on the 200 collected episodes, for Nemotron v2, Nemotron v2c and
   Cosmos v6:
   * precision and recall of "success";
   * false successes by failure type and by where the can came to rest.

## Budget

* About 400 Nebius calls (v2 and v2c on 200 episodes), about $0.30.
* 200 local Cosmos calls.
* 200 collection episodes and 1,200 evaluation episodes.
* Five 8,000-step fine-tunes.

## Amendments

None during the run.

**Deviations found by the verification (2026-10-29).** None changes an outcome.
* **Optimizer.** AdamW starts fresh, with betas (0.95, 0.999) and eps 1e-8 taken from B's
  config.
* **EMA.** It is re-initialised from B's *raw* weights, which are 0.18% from B's EMA. Its
  decay warm-up restarts: decay 0 at step 1, about 0.9988 at step 8,000.
* **DataLoader.** `drop_last=True`, `num_workers=0`.
* **Unparsed judge rows** count as "not selected". This affected 0 of 600 rows.
* **Batches and sampling seeds.**
  * Collection ran in batches of 25 and eval in batches of 50.
  * The diffusion sampling seed is set per batch (0 + 7919·b), so the eval noise seeds are
    shared across arms.
* **Post-hoc contrast.** `v2c_minus_continue` is post hoc.

## Results (2026-10-29; four verification lenses, no bug found that could explain the null)

| arm | eval success (200 paired seeds) | vs `continue` [95% CI], discordant, McNemar p |
|---|---|---|
| B (base) | 0.885 | — |
| continue | 0.880 | — |
| **nemotron (H1)** | 0.875 | −0.005 [−0.040, +0.030], 6/7, 1.0 |
| oracle (sim labels) | 0.890 | +0.010 [−0.025, +0.045], 7/5, 0.77 |
| all (unfiltered) | 0.880 | 0.000 [−0.035, +0.035], 6/6, 1.0 |
| nemotron_v2c | 0.875 | −0.005 [−0.030, +0.020] (post hoc) |

**H1: not supported.** The study only rules out large gains:
* With about 11 discordant seeds per pair, 80% power is reached only for gains of about
  6.5–7 points.
* The 95% upper bounds are +3.0 (nemotron vs continue) and +4.5 (oracle vs continue).

**H2 cannot be tested here, because there is no working positive control.**
* Even simulator-filtered data did not help.
* Adding all 28 failed rollouts did not hurt either, although they make up 32% of the
  training frames.
* So this recipe is insensitive to what the data contain. The run does **not** show that
  judge errors are harmless.
* Nemotron's 10 false successes were 5.5% of the selected episodes but 14% of the training
  frames.

**The fine-tune did change the policy, just not its success.**
* EMA weights moved 5–9%.
* Every arm, including `continue`, traded no-grasp failures for drops: B had 16 no-grasp
  and 7 dropped failures; the arms have 9–11 and 12–15.
* Success is fixed by initial state, with noise shared across arms: 13 seeds fail and 163
  succeed in all six arms.
* The always-failing seeds are hover-without-grasp stalls at the edges of the placement
  range. Nearby collected successes (within 3–5 cm) did not fix them.

**Labels on the 200 collected episodes (secondary 5).**

| judge | precision | recall | false successes |
|---|---|---|---|
| Nemotron v2 | 0.945 | 0.994 | 10, of which 8 in the wrong compartment |
| Nemotron v2c | 0.955 | 0.983 | 8 |
| Cosmos v6 | 0.945 | 1.0 | 10 |

**What a fair follow-up needs.**
* A pre-registered positive control at the same recipe, for example held-out human demos.
* A base with real headroom.
* Self-data that adds coverage (more exploration, hard initial states).
* A recipe that anneals the learning rate.
* About 1,000 paired seeds for a 3-point MDE.

**Cost.** 400 Nebius calls, 0 errors, about $0.31.
