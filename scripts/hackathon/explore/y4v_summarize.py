#!/usr/bin/env python
"""Verification of y4_grounded_reports (EXPLORATORY, post-hoc): collect all re-checks into
results/hackathon/explore/y4_grounded_reports_verification.json.

Automatic parts are recomputed here from saved files; MANUAL audit verdicts (verifier's own reading
of saved reports against the fact sheets) are stored as data below and labelled as such.
Run after y4v_number_check_power.py, y4v_strict_success_claims.py, y4v_replicate.py.
"""
from __future__ import annotations
import json, math, random, re, sys
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/hackathon/explore"))
import y4_verify as V  # noqa: E402
import y4_grounded_reports as G  # noqa: E402
VD = ROOT / "data/outputs/hackathon_explore/y4_grounded_reports_verification"
RES = ROOT / "results/hackathon/explore/y4_grounded_reports_verification.json"
Z = 1.959964


def wilson(k, n):
    if n == 0:
        return [None, None]
    p = k / n; d = 1 + Z * Z / n; c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return [round(max(0, c - h), 3), round(min(1, c + h), 3)]


def newcombe(k1, n1, k2, n2):
    """95% CI for p1 - p2 (Newcombe hybrid score)."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1); l2, u2 = wilson(k2, n2)
    d = p1 - p2
    return [round(d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), 3), round(d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2), 3)]


def cluster_boot_prop(items, B=2000, seed=0):
    """items: list of (cluster, 0/1)."""
    g = defaultdict(list)
    for c, x in items:
        g[c].append(x)
    ks = sorted(g); rng = random.Random(seed); vals = []
    for _ in range(B):
        s = [x for k in (rng.choice(ks) for _ in ks) for x in g[k]]
        vals.append(sum(s) / len(s))
    vals.sort()
    return [round(vals[int(0.025 * B)], 3), round(vals[int(0.975 * B) - 1], 3)]


# ---------------- MANUAL audit verdicts (verifier's reading; see verification note) ----------------
# random sample: random.Random(20261029).sample(range(62), 16) over parsed test ckpt reports in y4_grounded_reports_reports.json order
RANDOM_AUDIT = [
    ("can_s1_e0020", 1, "error", "'5/10 simulator success rate is near the run median of 10/10 judged successes'; 'close success counts across the run'", False),
    ("can_s1_e0060", 1, "ok_contradictory_reason", "", False),
    ("can_s1_e0080", 2, "ok_contradictory_reason", "auto miscite flag is a verifier false positive", True),
    ("can_s1_e0100", 1, "ok", "", False),
    ("can_s1_e0120", 1, "ok_contradictory_reason", "", False),
    ("can_s2_e0020", 0, "error", "'Cosmos3-Edge correctly judges 7 of 10 last frames as successful, missing 3 failures' (sim 6 successes; Cosmos judged 3 failures, missed 1); '6/10 ... close to the run median of 10/10'", False),
    ("can_s2_e0060", 0, "error", "'3 other checkpoints ... also achieve 9 or 10 judged successes' (8 others); trailing '10 [nemotron.judged_successes]' (value 9)", True),
    ("can_s2_e0060", 1, "ok_contradictory_reason", "", False),
    ("can_s2_e0060", 2, "ok", "", False),
    ("can_s2_e0100", 1, "error", "'9/10 simulator success, tying for the top performance' / 'already at the run's peak' (5 can_s2 checkpoints have sim 10/10); 'Nemotron judged all 10 as successful ... confirming the simulator's drop' (non sequitur)", False),
    ("can_s2_e0140", 1, "ok_contradictory_reason", "auto failure-type flag is a list-negation false positive", True),
    ("can_s2_e0160", 1, "ok_contradictory_reason", "'10/10 judged successes [sim.successes]' source/key mismatch, both values 10", False),
    ("can_s2_e0160", 2, "minor", "val_action_l1 rank 3 described as 'the top of the run'", False),
    ("can_s2_e0180", 1, "minor", "'1st for smoothness metrics' (plv1 rank 1, open_loop_smoothness rank 2)", False),
    ("square_s1_e0030", 0, "error", "'when it does succeed, it often drops the object' (flagged; also in y4 audit)", True),
    ("square_s1_e0150", 0, "ok_contradictory_reason", "", False),
]
RUN_TEST_AUDIT = [
    ("square_s1", 0, "error", "'square_s1_e0180 is the only checkpoint with perfect 10/10 [sim_successes]' (e0060 also 10/10); 'e0000-e0030 fails almost all 10 rollouts' (e0030 is 4/10)"),
    ("square_s1", 1, "error", "'simulator success reaches 10/10 ... and Nemotron matches this with 8/10'"),
    ("square_s1", 2, "ok", "selective ('maintain high sim_successes' skips e0090/e0120 at 7/10) but no false statement found"),
    ("can_s1", 1, "error", "'the last checkpoint before the final perfect epoch 100' (e0120-e0160 are also 10/10)"),
    ("can_s1", 2, "error", "'Mid-training (e0040-e0080) eliminates no_grasp failures ... achieving 9/10 or 10/10' (e0040: 7/10 with 1 no_grasp)"),
    ("can_s1", 0, "template_echo", "returned the output template verbatim"),
]
MISSING_AUDIT = [
    ("can_s1_e0160", 0, "error", "'is the top-ranked checkpoint in the run context [run_context]' although every rank of e0160 is null"),
    ("can_s1_e0160", 1, "error", "'with 9 checkpoints in the run and this one at the top rank' (no rank exists)"),
]
MISCITE_AUDIT = {  # the 12 y4 'miscited' flags in test ckpt reports
    "real_wrong_number": [("can_s1_e0000", 2, "all 10 no_grasp vs 9"), ("can_s2_e0020", 1, "Nemotron judges all 10 successful vs 6 (counted twice)"),
                          ("can_s2_e0020", 1, "duplicate of previous"), ("can_s2_e0040", 0, "'10 other checkpoints also achieve 10/10' vs 7 with J=10"),
                          ("can_s2_e0060", 0, "trailing '10 [nemotron.judged_successes]' vs 9")],
    "right_number_wrong_key": [("can_s2_e0160", 0, "val_action_l1 rank 3 cited as [cheap_metrics.val_action_l1]")],
    "verifier_false_positive": [("can_s1_e0020", 0, "number words 'three ... two ... out of 10'"), ("can_s1_e0020", 0, "same"),
                                ("can_s1_e0080", 2, "'Nemotron judged 0 failures' correct"), ("square_s1_e0060", 0, "'zero failures ... across 10'"),
                                ("square_s1_e0180", 0, "'10/10 [..] and no failures [sim.failures]'"), ("square_s1_e0180", 1, "same")],
}
# reports (test ckpt) known to contain >=1 WRONG NUMBER (lower bound; union of y4 audit and this audit)
WRONG_NUMBER_REPORTS = [("can_s1_e0000", 2), ("can_s2_e0020", 1), ("can_s2_e0040", 0), ("can_s2_e0060", 0), ("can_s2_e0020", 0)]


def main():
    facts, runs, truth = G.load()
    parsed = []
    for rec in G.load_generations("test", "v2"):
        rep, v = G.verify_rec(rec, facts, runs, truth)
        if v.get("parse_status") in ("ok", "ok_extracted"):
            parsed.append((rec["key"], rec["sample"], rep, v))
    # the audit sample was drawn over the dashboard file's order (sorted checkpoint ids, entries in stored order)
    dash = json.loads((ROOT / "results/hackathon/explore/y4_grounded_reports_reports.json").read_text())
    order = [(k, e["sample"]) for k, L in sorted(dash["checkpoint_reports"].items()) for e in L
             if e["phase"] == "test" and e["report"] is not None]
    assert len(order) == len(parsed) == 62
    idx = sorted(random.Random(20261029).sample(range(len(order)), 16))
    drawn = [order[i] for i in idx]
    assert drawn == [(k, s) for k, s, *_ in RANDOM_AUDIT], (drawn, [(k, s) for k, s, *_ in RANDOM_AUDIT])
    n_err = sum(1 for a in RANDOM_AUDIT if a[2] == "error")
    n_err_soft = sum(1 for a in RANDOM_AUDIT if a[2] in ("error", "minor"))
    n_err_flagged = sum(1 for a in RANDOM_AUDIT if a[2] == "error" and a[4])
    n_contra = sum(1 for a in RANDOM_AUDIT if a[2] == "ok_contradictory_reason")
    # contradiction heuristic over all 62
    pat = re.compile(r"(strong|clear|top|leading)[- ]?\w*\s+contender|contender for the best|among the (best|top)|already[- ](perfect|leading|competitive|at the top)|tie[sd]? for (the )?(top|highest|best)|top of the run|at the run's peak|clearly best", re.I)
    contra = [(k, s, truth[k]["rule_ckpt"]) for k, s, rep, v in parsed if pat.search(str(rep.get("recommendation_reason", "")))]
    # dominant-mode difficulty
    test_ck = sorted({k for k, *_ in parsed})
    diff = {"none": 0, "single_type": 0, "multi_type_unique_max": 0, "tie": 0}
    for c in test_ck:
        acc = truth[c]["dominant_failure_mode_accepted"]; ft = facts[c]["sim"]["failure_types"]
        diff["none" if acc == ["none"] else ("tie" if len(acc) > 1 else ("single_type" if len(ft) == 1 else "multi_type_unique_max"))] += 1
    nontriv = [c for c in test_ck if diff and truth[c]["dominant_failure_mode_accepted"] != ["none"] and len(facts[c]["sim"]["failure_types"]) > 1]
    # replication
    rep_rows = [json.loads(l) for l in open(VD / "replicate.jsonl")] if (VD / "replicate.jsonl").is_file() else []
    rep_ck = []; rep_run = []
    for r in rep_rows:
        obj, st = V.parse_json_obj(r["raw_text"])
        if r["kind"] == "ckpt":
            rep_ck.append({"key": r["key"], "parse": st, "rec": V.norm_choice((obj or {}).get("spend_more_rollouts"), ["yes", "no"]), "rule": truth[r["key"]]["rule_ckpt"]})
        else:
            vr = V.verify_run_report(obj, runs[r["key"]], truth[f"RUN::{r['key']}"], facts) if obj else {}
            rep_run.append({"key": r["key"], "parse": st, "pick": (obj or {}).get("evaluate_next"), "rule_consistent": vr.get("rule_consistent"),
                            "n_unsupported": vr.get("n_unsupported")})
    yes_rep = sum(1 for r in rep_ck if r["rec"] == "yes")
    # regret noise (S_i from 50 training-time rollouts)
    S = {"can_s1_e0080": 0.96, "can_s1_e0140": 0.98, "can_s1_e0120": 0.98, "can_s2_e0080": 0.96, "can_s2_e0200": 1.0,
         "square_s1_e0180": 0.80, "square_s1_e0210": 0.86, "square_s1_e0060": 0.68}
    import csv
    Sc = {r["checkpoint_id"]: (float(r["success"]), int(r["n_train_rollouts"])) for r in csv.DictReader(open(ROOT / "results/hackathon/baseline_metrics.csv"))}
    for k, v in S.items():
        assert abs(Sc[k][0] - v) < 1e-9 and Sc[k][1] == 50, k
    pairs = {"can_s1: LLM e0080 vs latest-with-target e0140": ("can_s1_e0080", "can_s1_e0140"),
             "can_s2 (dev run): LLM e0080 vs latest e0200": ("can_s2_e0080", "can_s2_e0200"),
             "square_s1: LLM/rule e0180 vs latest e0210": ("square_s1_e0180", "square_s1_e0210"),
             "square_s1: LLM e0060 vs latest e0210": ("square_s1_e0060", "square_s1_e0210")}
    regret_noise = {name: {"S_pick": S[a], "S_latest": S[b], "diff": round(S[a] - S[b], 2),
                           "newcombe95_diff": newcombe(round(S[a] * 50), 50, round(S[b] * 50), 50)} for name, (a, b) in pairs.items()}
    power = json.loads((VD / "number_power.json").read_text())
    strict = json.loads((VD / "strict_success_claims.json").read_text())
    out = {
        "name": "y4_grounded_reports_verification", "label": "EXPLORATORY / post-hoc adversarial verification",
        "reproduction": {
            "fact_sheets_rebuilt": "identical for the 28 original checkpoints; a new eval checkpoint square_s1_e0240 (no VLM diagnostics) now exists and changes only square_s1 run_context.n_eval_checkpoints_in_run/epochs_in_run",
            "analysis_rerun_into_scratch": "y4_grounded_reports.json and y4_grounded_reports_reports.json reproduced exactly (no differing field)",
            "evidence_log": "118/118 tagged calls, 0 errors, request ids match generations.jsonl, raw_text sha256 match 118/118, prompt sha256 recomputed from current code + fact sheets match 118/118",
            "tokens": {"prompt": 245513, "completion": 77234, "cost_usd_catalog": 0.0333},
            "timeline": "fact sheets 06:09; dev v1 06:12, v2 06:14 (prompt file saved 06:14:37), verifier saved 06:15:04 (after v1/v2 dev, before test 06:17:46), analysis script saved 06:19:35 (after test)",
        },
        "number_check_power": power,
        "strict_success_count_claims": {k: strict[k] for k in ("n_reports", "n_claims_checked", "n_flagged", "reports_with_flag")} |
                                        {"hand_check": "1 of 3 flags real (can_s2_e0020 s1 'Nemotron judges all 10 rollouts as successful', true 6); 2 are failure-type counts misread as success counts"},
        "cited_share_excluding_k_over_10_denominators": {"k": 198, "n": 561, "rate": 0.353},
        "miscitation_audit_manual": {k: len(v) for k, v in MISCITE_AUDIT.items()} | {"detail": MISCITE_AUDIT},
        "reports_with_wrong_number_lower_bound": {"k": len(WRONG_NUMBER_REPORTS), "n": 62, "rate": round(len(WRONG_NUMBER_REPORTS) / 62, 3),
                                                   "wilson95": wilson(len(WRONG_NUMBER_REPORTS), 62), "reports": WRONG_NUMBER_REPORTS,
                                                   "note": "all these numbers passed y4's 'unsupported number' check; y4's single unsupported number ('30% of evaluations fail', can_s1_e0040 s2) is a CORRECT derived value (3/10 failures)"},
        "random_audit_manual": {"n_reports": 16, "n_checkpoints": len({a[0] for a in RANDOM_AUDIT}), "seed": 20261029,
                                "reports_with_clear_error": n_err, "rate": round(n_err / 16, 3), "wilson95": wilson(n_err, 16),
                                "cluster_boot95": cluster_boot_prop([(a[0], int(a[2] == "error")) for a in RANDOM_AUDIT]),
                                "incl_minor": n_err_soft, "incl_minor_wilson95": wilson(n_err_soft, 16),
                                "clear_errors_flagged_by_y4_verifier": n_err_flagged,
                                "ok_but_self_contradictory_reason": n_contra, "verdicts": RANDOM_AUDIT},
        "run_test_audit_manual": {"n": 6, "error": sum(a[2] == "error" for a in RUN_TEST_AUDIT), "template_echo": 1,
                                  "y4_auto_flags_on_these": "0 unsupported, 0 miscited", "verdicts": RUN_TEST_AUDIT},
        "missing_set_audit_manual": {"n": 9, "fabricated_rank_claims": len(MISSING_AUDIT), "wilson95": wilson(len(MISSING_AUDIT), 9), "verdicts": MISSING_AUDIT},
        "dominant_mode_difficulty": {"test_checkpoints": len(test_ck), **diff,
                                     "nontrivial_checkpoints(>1 type present)": nontriv,
                                     "wilson95_checkpoint_level_21_of_21": wilson(21, 21),
                                     "note": "the sheet hands the LLM sim.failure_types; RULE_DOMINANT is its argmax"},
        "recommendation_reason_contradiction_regex": {"k": len(contra), "n": len(parsed), "among_rule_yes": sum(1 for c in contra if c[2] == "yes"),
                                                       "rule_yes_total": sum(1 for k, s, rep, v in parsed if truth[k]["rule_ckpt"] == "yes"),
                                                       "note": "regex for 'contender/top/already perfect' language in a 'no' reason; not hand-verified one by one"},
        "replication_fresh_samples": {"ckpt_rule_yes": {"n": len(rep_ck), "llm_yes": yes_rep, "wilson95_yes": wilson(yes_rep, len(rep_ck)), "rows": rep_ck},
                                      "run_can_s1": {"n": len(rep_run), "rows": rep_run}, "nebius_calls": len(rep_rows)},
        "regret_noise_check": regret_noise,
    }
    RES.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("random_audit_manual", "reports_with_wrong_number_lower_bound", "replication_fresh_samples", "regret_noise_check", "dominant_mode_difficulty", "recommendation_reason_contradiction_regex")}, indent=1)[:6000])


if __name__ == "__main__":
    main()
