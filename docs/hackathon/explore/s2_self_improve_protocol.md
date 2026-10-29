# s2 — VLM-filtered self-training, redesigned after the s1 null

**Status:** pre-registered on 2026-10-29 (KST). Written before any s2 rollout, label or
training step. This is an exploratory study. Any later change is listed under
"Amendments" with its date.

## Why s2

s1 was null: verified, with no bug that could explain it. The verification also showed
that s1 could not rank label sources, because the recipe did not respond to data content
at all. Even simulator-filtered data was null, and 32% failure frames did not hurt. s2
changes the four things the verification named:
1. **A base with headroom.** B2 comes from an early, still-improving checkpoint of the
   same run, not from the plateau.
2. **A positive control.** An arm with 36 extra held-out human demos shows whether the
   recipe can raise success at all.
3. **A recipe that trains and then anneals.** It starts from B2's EMA weights, and the
   learning rate is cosine-annealed to 0.
4. **More power.** 400 collected rollouts, and 500 paired eval seeds.

The method is not new (SOAR, RoboCat, Large Reward Models). The question stays narrow:
does a zero-shot Nemotron filter give the same gain as simulator labels on a policy with
headroom?

## Base

* **B2:** Diffusion Policy, robomimic Can PH, the `20_percent_train` subset (36 demos),
  seed 1, epoch 125.
* **File:** `data/outputs/hackathon_pool_lowdata/paper_dp_can_abs_ph_20_percent_seed1_e1000_r25/checkpoints/epoch=0125-test_mean_score=0.580.ckpt`.
* **Training-time success:** 0.58.
* **Why epoch 125:** it is the last checkpoint below 0.6. The same run later reaches
  0.80–0.90, so the 36 demos alone also leave room to improve, and the `continue` control
  can gain too. Only the gain *beyond* `continue` counts.

## Hypotheses

* **H1 (primary).** Fine-tuning B2 on its own successes as selected by Nemotron gives
  higher closed-loop success than the same fine-tuning on the original 36 demos only.
* **Positive control (precondition for interpreting label quality).** Adding 36 held-out
  human demos beats `continue`.
* **H2 (secondary).** `nemotron` is worse than `oracle`. This is interpreted only if
  `oracle` − `continue` is positive.

## Steps

1. **Collect.**
   * B2 runs for 400 episodes on seeds 710000–710399, in two parallel processes.
   * Recorder settings as in s1: agentview 448 px, 10 fps, 16-step hold, batches of 25,
     max 400 steps. Low-dim observations and absolute actions are saved.
2. **Label.**
   * Simulator truth.
   * Nemotron-3.5-Super-VL on Nebius, frozen `v2_evidence_first`, with Experiment 2
     settings, run in two parallel processes over the seed halves.
   * Unparsed rows are "not selected".
3. **Training data.**
   * Selected episodes are truncated at their first simulator success. A judge-accepted
     failure is kept whole.
   * One hdf5 holds the original 36 train + 4 valid demos, 36 held-out human demos, and
     the 400 collected episodes.
   * The held-out human demos are drawn with numpy seed 0 from the 144 PH `train` demos
     that are in neither `20_percent_train` nor `20_percent_valid`.

   | arm | training episodes |
   |---|---|
   | `continue` | 36 original demos |
   | `nemotron` (**H1**) | 36 + episodes Nemotron v2 calls a success |
   | `oracle` | 36 + episodes the simulator calls a success |
   | `human` (positive control) | 36 + 36 held-out human demos |

4. **Fine-tune** every arm with the same recipe.
   * Start from B2's EMA weights. The normalizer is kept fixed from B2.
   * 8,000 steps, batch 128, AdamW (betas 0.95/0.999, eps 1e-8, weight decay 1e-6).
   * Learning rate: 200-step linear warm-up to 1e-4, then cosine to 0 at step 8,000. This
     follows the shape of the original training schedule.
   * EMA as configured, re-initialised from the starting weights.
   * Training seed 1. The four arms train concurrently.
   * The EMA weights after step 8,000 are evaluated. There is no checkpoint selection.
5. **Evaluate** B2 and the four arms.
   * 500 seeds, 810000–810499, the same for every arm. No video. Batches of 50.
   * Simulator success.

## Endpoints

**Primary.**
* The quantity is paired success(`nemotron`) − success(`continue`) over the 500 seeds.
* The 95% CI comes from a bootstrap over seeds (10,000 resamples, seed 0).
* The exact McNemar p is reported too.
* H1 is **positive** if the lower CI bound is above 0.

**Secondary** (all reported, whatever their sign):
1. `human` − `continue`. This is the positive control.
2. `oracle` − `continue`.
3. `nemotron` − `oracle` (H2), interpreted only if secondary 2 is positive.
4. B2 vs every arm. This includes the recipe effect.
5. Nemotron label quality on the 400 collected episodes:
   * precision and recall;
   * false successes by failure type and by where the can came to rest;
   * contamination weighted by episodes and by frames.
6. Failure types by arm.

## Budget

* 400 Nebius calls, about $0.30.
* 400 collection episodes and 2,500 eval episodes.
* Four 8,000-step fine-tunes.
* Expected wall clock: about 3 hours.

## Amendments

None during the run.

## Results (2026-10-29; an independent verifier reproduced every number)

Eval: 500 paired seeds, the same for every arm. CIs come from a seed bootstrap
(10,000 resamples); p values are exact McNemar.

| arm | success | vs `continue` [95% CI] | discordant | McNemar p |
|---|---|---|---|---|
| B2 (base, epoch 125) | 0.656 | — | — | — |
| continue | 0.856 | — | — | — |
| **human** (positive control) | **0.938** | **+0.082 [+0.052, +0.114]** | 55/14 | 7e-7 |
| oracle | 0.824 | −0.032 [−0.070, +0.006] | 40/56 | 0.13 |
| **nemotron** (H1) | **0.752** | **−0.104 [−0.148, −0.060]** | 39/91 | 6e-6 |

**What each comparison shows.**
* **The positive control passed.** The recipe can raise success.
* **H1 is not supported, and the effect goes the other way.** Self-training on
  Nemotron-selected episodes is clearly *worse* than continuing on the original demos.
* **Simulator-filtered self-data did not beat `continue`** (secondary 2). So, by the
  protocol's rule, H2 is **not interpretable as pre-registered**.
* **Post hoc only: nemotron − oracle = −0.072 [−0.112, −0.032], p = 5e-4.**
  * The nemotron mask is exactly the oracle mask plus the 58 failures Nemotron accepted.
  * Most of the extra eval failures in the nemotron arm are cans left in the wrong place
    in the goal bin: 46 vs 19 (discordant 34/7). That is the same signature as the
    accepted failures.

**Nemotron labels on the 400 collected episodes.**
* Precision 0.812 and recall 1.0: it accepted 58 of 149 failures.
* Where the 58 came to rest:

  | resting place | count |
  |---|---|
  | wrong compartment of the goal bin (the c1 blind spot) | 44 |
  | on a divider or rim | 8 |
  | still held over the goal bin | 2 |
  | never grasped (plain hallucination) | 4 |

* They are kept whole, so they make up 40% of that arm's self frames (38% of its
  training windows).

**Caveats.**
* **One training run per arm.** The CIs cover eval seeds only. Run-to-run variance was
  not measured.
* **Dilution under the fixed 8,000-step budget.** The 36 original demos make up 100% /
  50% / 10.5% / 6.5% of samples in continue / human / oracle / nemotron. The oracle arm
  barely re-fit them (fixed-noise loss on those demos: 0.0100, vs 0.0035 for continue).
  This may explain part of oracle ≤ continue. No arm separates content from size.
* **The Nemotron arm is not fully simulator-free.** Clip length follows the simulator's
  success signal (recording stops 16 steps after success), and the prompt states that
  length. Recall 1.0 may partly be a length cue. This is inherited from Experiment 2,
  whose length-neutral check gave 0.961 vs 0.964.
* **Process.** The collection and judge records show git commit 2108ca7, a dashboard and
  docs commit made during the run. No s2, `cosmos_plv` or `diffusion_policy` code changed
  between e754568 and 2108ca7.

**Cost.** 400 Nebius calls, about $0.3.
