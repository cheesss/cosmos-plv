# y4_grounded_reports: adversarial verification (EXPLORATORY, post-hoc)

Verdict: **partially confirmed**. The bookkeeping is right. Every count in the y4 result JSON reproduces exactly, and the evidence log, the prompts and the raw outputs all match. The "always no" recommendation also holds up and replicates on fresh samples. The headline grounding claim does not hold up. "Almost never invents a number (1/723, Wilson upper bound 0.8%)" comes from a check that can hardly fail on these reports. Reports that contain a false number or claim are common, and the y4 verifier does not catch them.

All numbers below come from `results/hackathon/explore/y4_grounded_reports_verification.json` or from the scratch files in `data/outputs/hackathon_explore/y4_grounded_reports_verification/`. Labels marked "manual" are my own reading of the saved reports against the fact sheets.

## 1. Question

Do y4's claims survive re-scoring of its saved outputs, plus a small fresh replication? I looked for weak checks, wrong denominators, dev/test confusion, leakage and claims stronger than the numbers.

## 2. Method

- Reproduction:
  - Rebuilt the fact sheets into scratch and diffed them against y4's copies.
  - Re-ran `y4_analyze.py` into scratch. It was monkeypatched so it would not overwrite y4's files.
  - Checked all 118 tagged evidence rows against `generations.jsonl`: request id, raw-text sha256, and the prompt sha256 recomputed from the current code and fact sheets.
- Power of the lenient number check (`y4v_number_check_power.py`):
  - Counted which numbers the 723 counted tokens actually are.
  - Counted how many of the integers 0..10 each sheet "supports".
  - Placed hypothetical wrong numbers (+1, -1, a random integer from 0..10) at every real number position and checked whether y4's check would flag them.
- Strict claim check (`y4v_strict_success_claims.py`): success counts attributed to exactly one source, compared with that source's value. Each flag was then hand-checked.
- Manual audits:
  - a seeded random sample of 16 parsed test checkpoint reports (seed 20261029; 13 checkpoints);
  - all 6 run-level test reports;
  - all 9 missing-diagnostics reports;
  - all 12 y4 "miscited" flags.
- Fresh replication (`y4v_replicate.py`): 13 Nebius calls with y4's frozen v2 prompt code, unchanged, at temperature 0.2. That is one new sample for each of the 10 test checkpoints where RULE_CKPT says "yes", plus 3 new can_s1 run reports. Tag: `explore/y4_grounded_reports_verification/...`.
- Noise check on the post-hoc regret comparison: S_i comes from 50 training-time rollouts, so I put a Newcombe 95% CI on each difference.

## 3. Commands and exit codes

All commands ran in Env B (`PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python`) from the repo root. No GPU was used.

| step | command | exit |
|---|---|---|
| rebuild fact sheets into scratch | inline: `y4_build_facts.OUT=<scratch>/rebuild; main()` | 0 |
| re-run analysis into scratch | inline: `y4_analyze.RES=<scratch>/reanalysis; main()` | 0 |
| evidence, hash and prompt checks | inline python | 0 |
| power test | `scripts/hackathon/explore/y4v_number_check_power.py` | 0 |
| strict success-count claims | `scripts/hackathon/explore/y4v_strict_success_claims.py` | 0 |
| fresh replication (13 Nebius calls) | `HF_HUB_OFFLINE=1 ... scripts/hackathon/explore/y4v_replicate.py` | 0 |
| consolidate | `scripts/hackathon/explore/y4v_summarize.py` | 1 (my sample-order assertion), then 0 after the fix |

## 4. What was confirmed

| y4 claim | check | result |
|---|---|---|
| 118 calls, 0 errors, 245,513 + 77,234 tokens, about $0.033 | evidence log | exact; ids and sha256 match 118/118 |
| fact sheets and prompts are what was sent | rebuild + prompt sha256 | identical for the 28 original checkpoints; prompt sha 118/118 |
| all y4 metrics | analysis re-run | `y4_grounded_reports.json` and `_reports.json` reproduced field for field |
| no S_i in the sheets | grep for `test_mean_score`, `checkpoint_path`, `.ckpt` | none present |
| dev errors v1 5, v2 3, v3 8, v3+thinking 50; thinking runs truncated | re-score; thinking runs used completion = max_tokens 10/10 | confirmed; median 22.1 s vs 2.5 s |
| LLM says "no" 62/62; rule says "yes" 30/62; agreement 32/62 | re-score | confirmed |
| always "no" is systematic | fresh samples on the 10 rule-"yes" checkpoints | **0/10 "yes"** (Wilson [0, 0.278]) |
| self-contradictory "no" reasons | regex for "contender / top / already perfect" wording in a "no" reason | 43/62 reports; 26/30 of the rule-"yes" ones (heuristic, not checked one by one) |
| 13 auto-flagged failure-type claims: 2 real, 11 false positives (7 list negations, 3 count parses, 1 source) | manual | confirmed |
| run picks consistent with the rule 4/6; one template echo | re-score + 3 fresh can_s1 samples | confirmed. The template echo did not recur: 3/3 fresh picks were valid and in the accepted set, so 7/9 overall |

## 5. What does not hold

### 5.1 The lenient number check has almost no power, so "1/723" says very little

| quantity | value |
|---|---|
| counted number tokens that are exactly "10" | 363/723 (50.2%), mostly "k/10" denominators |
| tokens that are integers 0..10 | 94.1% |
| integers 0..10 "supported" by a sheet (min / median / max over sheets) | 10 / 11 / 11 out of 11 |
| hypothetical wrong integer from 0..10 flagged | **1/696 (0.1%)**, naive Wilson [0.000, 0.008] |
| hypothetical value -1 flagged | 26/680 (3.8%) |
| hypothetical value +1 flagged | 219/710 (30.8%), mostly because 10+1 = 11 is absent |

An integer also counts as matched to any sheet value within ±0.5, for example a success time of 9.4 s.

y4's single "unsupported" number is "30% of evaluations fail" (can_s1_e0040 s2). It is a correct derived value: 3 of 10 failed.

So the observed rate of numbers that are both invented and wrong is 0/723, and the check could not have detected most wrong counts anyway. The Wilson upper bound of 0.8% in the headline also treats 723 numbers inside 62 correlated reports (21 checkpoints) as independent.

### 5.2 Wrong numbers and false claims are common; the verifier misses most of them

All of these passed the "unsupported number" check:

| measure | k / n | rate | 95% CI |
|---|---|---|---|
| test checkpoint reports with ≥1 confirmed **wrong number** (lower bound, not exhaustive) | 5 / 62 | 0.081 | Wilson [0.035, 0.175] |
| random-audit reports with ≥1 clear factual or semantic error (manual) | 5 / 16 | 0.31 | Wilson [0.142, 0.556]; checkpoint-cluster bootstrap [0.111, 0.538] |
| same, including minor misstatements | 7 / 16 | 0.44 | Wilson [0.231, 0.668] |
| clear errors in the random audit that y4's verifier flagged | 2 / 5 | | |
| run-level test reports with a false statement (manual) | 4 / 6 (5 non-template) | | y4 auto: 0/128 unsupported, 0/30 miscited |
| missing-diagnostics reports that invent a rank (manual) | 2 / 9 | | Wilson [0.063, 0.547] |

The wrong numbers:

- can_s2_e0020 s1: "Nemotron judges all 10 rollouts as successful" (the sheet says 6).
- can_s2_e0040 s0: "10 other checkpoints also achieve 10/10" (7 do).
- can_s2_e0060 s0: a trailing "10 [nemotron.judged_successes]" (the value is 9).
- can_s1_e0000 s2: "Nemotron judged all 10 ... no_grasp" (9).
- can_s2_e0020 s0: "Cosmos3-Edge correctly judges 7 of 10 ... missing 3 failures". The simulator recorded 6 successes; Cosmos judged 3 failures and missed 1.

Run-level examples:

- "square_s1_e0180 is the only checkpoint with perfect 10/10". e0060 is also 10/10.
- "Mid-training (e0040-e0080) eliminates no_grasp ... achieving 9/10 or 10/10". e0040 is 7/10 and has 1 no_grasp.
- "the final perfect epoch 100". e0120 to e0160 are also 10/10.
- "Nemotron matches this with 8/10" for a simulator result of 10/10.

Missing-diagnostics set: can_s1_e0160 is "the top-ranked checkpoint in the run context", but every rank for e0160 is null.

Of y4's 12 "miscited" flags, 5 are real wrong numbers (4 distinct claims; one claim is counted twice). One has the right number under the wrong key, and 6 are verifier false positives. The 12/198 = 0.061 figure therefore mixes real and spurious flags.

### 5.3 Dominant failure mode 62/62 is a copy task

The sheet gives `sim.failure_types` directly, and RULE_DOMINANT is its argmax. Of the 21 test checkpoints:

- 8 have no failures;
- 9 have only one failure type;
- 4 have more than one type (can_s1_e0020, can_s1_e0040, can_s2_e0020, square_s1_e0030), one of them a tie.

At checkpoint level, 21/21 gives Wilson [0.845, 1.0], not [0.942, 1.0]. The 62/62 also leaves out the one unparsed report (62/63 overall). None of this shows physical understanding; it shows the model can read one field.

### 5.4 The post-hoc regret claim is within noise and partly mislabelled

| comparison (S_i from 50 rollouts) | difference in S | Newcombe 95% CI |
|---|---|---|
| can_s1: LLM e0080 vs latest-with-target e0140 | -0.02 | [-0.116, 0.070] |
| can_s2 (dev run): LLM e0080 vs e0200 | -0.04 | [-0.135, 0.037] |
| square_s1: LLM-majority / rule e0180 vs e0210 | -0.06 | [-0.208, 0.090] |
| square_s1: single LLM pick e0060 vs e0210 | -0.18 | [-0.335, -0.014] |

- **Within noise.** "Worse regret than the latest-epoch baseline in all 3 runs" rests on differences of 1 to 3 rollouts out of 50, except for the one e0060 pick.
- **Mislabelled baseline.** For can_s1, the "latest epoch" baseline is e0140, the latest checkpoint *with a target*. The actual latest checkpoint, e0160, has no S_i.
- **No edge for the rule.** On square_s1, the deterministic rule's pick has the same regret as the LLM's majority pick (0.10), so this evidence does not favour "use the rule" either.

### 5.5 Dev and test are not cleanly separated

- **can_s2.** The dev run is can_s2, yet 9 of its checkpoints make up 27 of the 63 test checkpoint reports. The dev run-level prompt iteration (v1 to v3) saw all of their facts and outputs.
- **square_s1.** The test run sheet contains the dev checkpoints e0090 and e0120.

The prompt edits are generic, so the leak is mild. Still, the split is by checkpoint, not by run. Per run, the test results are:

| run | unsupported | miscited | failure flags | rule agreement |
|---|---|---|---|---|
| can_s1 | 1/231 | 4/52 | 3/78 | 9/21 |
| can_s2 | 0/344 | 5/112 | 6/104 | 18/27 |
| square_s1 | 0/148 | 3/34 | 4/61 | 5/14 |

Rule agreement only tracks each run's share of rule-"no" checkpoints.

### 5.6 Smaller points

- **Freezing v2 over v1 is noise-level.** The dev scores were 5 vs 3 automatic errors, with about 2 real errors each. v2's rule 6 targeted the "Nemotron judged all 10 as no_grasp" error, which still appears in v2 dev (can_s2_e0000) and in test (can_s1_e0000 s2).
- **Timeline.** The verifier file was saved after the v1 and v2 dev outputs but before test (06:15 vs 06:17). That is allowed, but the freeze metric was edited after seeing candidate outputs. `y4_analyze.py` was saved after test (06:19).
- **"Only 27% of numbers cited" is partly an artefact.** The "10" denominators of "k/10" are counted as uncited. Without them, 198/561 = 35% are cited.
- **A small y4 audit slip.** For "3 other checkpoints also reach 9 or 10", the correct count is 8 others, not 9.
- **The data has moved on.** A new eval checkpoint, square_s1_e0240 (no VLM diagnostics), now exists. Rebuilding the fact sheets today would change the square_s1 run context.

## 6. Interpretation

- **Useful:**
  - y4's cost and latency figures;
  - the finding that "spend more rollouts?" is a constant "no" (it replicates);
  - the design advice: compute the decision deterministically and let the LLM only explain it.
- **Overstated:** the grounding headline. The right statement is:

  > With measured-facts sheets, Lightning rarely writes a number that appears nowhere in the sheet. That check cannot catch wrong small counts. In a random audit, about 1 in 3 reports (5/16, Wilson [0.14, 0.56]) contained a clear false statement, and the regex verifier caught 2 of those 5. Run-level summaries were worse (4/5 non-template reports had errors).

- **Consequence for the dashboard:** free-text reports from this setup should not be shown as "verified". Structured claims with exact checks (y4's own next step 1) are needed first.

## 7. Limitations

- **Manual audits are small and done by one person, without blinding:** 16 + 6 + 9 reports; the "error" and "minor" labels are judgement calls.
- **The wrong-number count is a lower bound.** I did not read all 62 reports.
- **The contradiction count is a regex heuristic.**
- **The replication is small:** 13 calls, temperature 0.2, one model.
- **The power test only places wrong numbers where real numbers already are.** It says nothing about wrong numbers that appear elsewhere in a report.
- **All of this remains post-hoc and exploratory**, on 21 test checkpoints from 3 runs and 2 tasks.

## 8. Next steps

1. Use a structured-claim output: {source, key, value, text} triples, with prose rendered from them. Re-measure using the manual error definition from this note.
2. Replace the "number appears anywhere" check with a key-bound check, report its power using the perturbation test above, and hand-label 30 random reports for precision and recall.
3. Re-test dominant-mode extraction on a sheet that gives per-episode labels only, without the `failure_types` summary, so the field is not a copy.
4. If the regret comparison is kept, state the 50-rollout noise, use the real latest checkpoint, and say "not distinguishable".
5. Split dev and test by run, not by checkpoint.

## Files

- Scripts:
  - `scripts/hackathon/explore/y4v_number_check_power.py`
  - `scripts/hackathon/explore/y4v_strict_success_claims.py`
  - `scripts/hackathon/explore/y4v_replicate.py`
  - `scripts/hackathon/explore/y4v_summarize.py`
- Results: `results/hackathon/explore/y4_grounded_reports_verification.json`
- Scratch (`data/outputs/hackathon_explore/y4_grounded_reports_verification/`):
  - `rebuild/` and `reanalysis/`
  - `number_power.json`, `strict_success_claims.json`
  - `replicate.jsonl`, `audit_sample.txt`
- Nebius evidence: `results/hackathon/explore/nebius_calls_explore.jsonl`, tag `explore/y4_grounded_reports_verification/` (13 calls, 0 errors)
