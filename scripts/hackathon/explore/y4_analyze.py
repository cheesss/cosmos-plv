#!/usr/bin/env python
"""y4_grounded_reports, step 3: score all generations with the verifier, pick the frozen dev config
(pre-declared criterion), and summarize test grounding error rates with Wilson and
checkpoint-clustered bootstrap CIs. Also writes the dashboard-ready report JSON.

EXPLORATORY / post-hoc. Usage: y4_analyze.py [--frozen v2]   (if omitted, the dev criterion picks it)
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/hackathon/explore"))
import y4_grounded_reports as G  # noqa: E402

RES = ROOT / "results/hackathon/explore"
PRICE_IN, PRICE_OUT = 0.06, 0.24  # USD per 1M tokens, public catalog snapshot (models_info_snapshot.json)


def wilson(k, n, z=1.959964):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0, c - h), 3), round(min(1, c + h), 3)]


def cluster_boot(rows, fn, key="key", B=2000, seed=0):
    """rows: list of dicts; fn(rows) -> float. Resample clusters (checkpoints) with replacement."""
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    ks = sorted(groups)
    if len(ks) < 2:
        return [None, None]
    rng = random.Random(seed)
    vals = []
    for _ in range(B):
        samp = [r for k in (rng.choice(ks) for _ in ks) for r in groups[k]]
        v = fn(samp)
        if v is not None:
            vals.append(v)
    vals.sort()
    return [round(vals[int(0.025 * len(vals))], 3), round(vals[int(0.975 * len(vals)) - 1], 3)]


def config_of(rec):
    return rec["variant"] + ("_think" if rec.get("reasoning") else "")


def score_all():
    facts, runs, truth = G.load()
    out = []
    for rec in G.load_generations():
        rep, v = G.verify_rec(rec, facts, runs, truth)
        out.append({"rec": rec, "report": rep, "v": v, "config": config_of(rec)})
    return out, facts, runs, truth


def err_count(v):
    if v.get("parse_status") not in ("ok", "ok_extracted"):
        return None
    e = v["n_unsupported"] + v["n_miscited"] + v["n_wrong_failure_claims"] + v.get("n_misattributed_counts", 0)
    if "dominant_correct" in v:
        e += int(not v["dominant_correct"])
    return e


def summarize(items, kind):
    """items: scored rows of one kind ('ckpt' or 'run')."""
    rows = []
    for it in items:
        v, rec = it["v"], it["rec"]
        ok = v.get("parse_status") in ("ok", "ok_extracted")
        r = {"key": rec["key"], "parsed": ok, "latency_s": rec["latency_s"], "pt": rec["prompt_tokens"] or 0,
             "ct": rec["completion_tokens"] or 0}
        if ok:
            r.update({"n_numbers": v["n_numbers"], "n_unsupported": v["n_unsupported"],
                      "n_cited": v["n_cited_numbers"], "n_miscited": v["n_miscited"], "n_bad_key": v["n_bad_key"],
                      "n_uncited": v["n_uncited_numbers"], "n_fmentions": v["n_failure_mentions"],
                      "n_wrong_f": v["n_wrong_failure_claims"], "n_count_claims": v["n_count_claims"],
                      "n_misattr": v["n_misattributed_counts"], "err": err_count(v),
                      "rule_consistent": v.get("rule_consistent")})
            if kind == "ckpt":
                r["dominant_correct"] = v["dominant_correct"]
                r["recommend"] = v["recommend_reported"]
                r["rule"] = v["rule"]
            else:
                r["evaluate_next"] = v["evaluate_next"]
                r["pick_valid"] = v["pick_valid_id"]
                r["invented_ids"] = v["invented_checkpoint_ids"]
        rows.append(r)
    P = [r for r in rows if r["parsed"]]
    n = len(rows)
    s = {"n_reports": n, "n_parsed": len(P), "parse_rate": round(len(P) / n, 3) if n else None,
         "parse_rate_wilson95": wilson(len(P), n)}
    if not P:
        return s, rows

    def prop(pred, sub=None):
        sub = P if sub is None else sub
        k = sum(1 for r in sub if pred(r))
        return {"k": k, "n": len(sub), "rate": round(k / len(sub), 3) if sub else None, "wilson95": wilson(k, len(sub)),
                "cluster_boot95": cluster_boot(sub, lambda rr: (sum(1 for r in rr if pred(r)) / len(rr)) if rr else None)}

    def ratio(num, den):
        N = sum(r[num] for r in P)
        D = sum(r[den] for r in P)
        return {"k": N, "n": D, "rate": round(N / D, 4) if D else None, "wilson95_naive": wilson(N, D),
                "cluster_boot95": cluster_boot(P, lambda rr: (sum(r[num] for r in rr) / max(1, sum(r[den] for r in rr))))}

    s["reports_with_any_unsupported_number"] = prop(lambda r: r["n_unsupported"] > 0)
    s["unsupported_numbers_per_number"] = ratio("n_unsupported", "n_numbers")
    s["miscited_per_cited_number"] = ratio("n_miscited", "n_cited")
    s["cited_fraction_of_numbers"] = {"k": sum(r["n_cited"] for r in P), "n": sum(r["n_numbers"] for r in P)}
    s["bad_citation_keys"] = sum(r["n_bad_key"] for r in P)
    s["wrong_failure_claims_per_mention"] = ratio("n_wrong_f", "n_fmentions")
    s["misattributed_count_claims_per_claim"] = ratio("n_misattr", "n_count_claims")
    s["reports_with_any_automatic_error"] = prop(lambda r: (r["err"] or 0) > 0)
    s["mean_errors_per_report"] = round(statistics.mean(r["err"] for r in P), 3)
    if kind == "ckpt":
        s["dominant_failure_mode_correct"] = prop(lambda r: r["dominant_correct"])
        sub = [r for r in P if r["rule"] is not None]
        s["recommendation_consistent_with_rule"] = prop(lambda r: r["rule_consistent"], sub)
        s["recommend_yes_rate"] = prop(lambda r: r["recommend"] == "yes", sub)
        s["rule_yes_rate"] = prop(lambda r: r["rule"] == "yes", sub)
        s["recommendation_confusion"] = {f"llm_{a}_rule_{b}": sum(1 for r in sub if r["recommend"] == a and r["rule"] == b)
                                         for a in ("yes", "no") for b in ("yes", "no")}
    else:
        s["pick_consistent_with_rule"] = prop(lambda r: bool(r["rule_consistent"]))
        s["picks"] = [r.get("evaluate_next") for r in P]
        s["invented_ids"] = sorted({i for r in P for i in r["invented_ids"]})
    lat = [r["latency_s"] for r in rows if r["latency_s"]]
    s["latency_s"] = {"median": round(statistics.median(lat), 3), "mean": round(statistics.mean(lat), 3),
                      "max": round(max(lat), 3)} if lat else None
    pt = sum(r["pt"] for r in rows)
    ct = sum(r["ct"] for r in rows)
    s["tokens"] = {"prompt_total": pt, "completion_total": ct, "prompt_mean": round(pt / n, 1),
                   "completion_mean": round(ct / n, 1)}
    s["cost_usd_estimate"] = {"total": round(pt / 1e6 * PRICE_IN + ct / 1e6 * PRICE_OUT, 5),
                              "per_report": round((pt / 1e6 * PRICE_IN + ct / 1e6 * PRICE_OUT) / n, 6)}
    return s, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", default=None)
    a = ap.parse_args()
    scored, facts, runs, truth = score_all()
    # ---- dev selection (pre-declared): lowest total dev error count over ckpt + run reports;
    #      unparsed reports count as 5 errors; ties -> fewer completion tokens
    dev = defaultdict(list)
    for it in scored:
        if it["rec"]["phase"] in ("dev", "run_dev"):
            dev[it["config"]].append(it)
    dev_table = {}
    for cfg, its in sorted(dev.items()):
        errs = [err_count(it["v"]) for it in its]
        tot = sum(5 if e is None else e for e in errs)
        ck = [it for it in its if it["rec"]["kind"] == "ckpt"]
        rn = [it for it in its if it["rec"]["kind"] == "run"]
        dev_table[cfg] = {"total_errors": tot, "n_reports": len(its), "n_unparsed": sum(e is None for e in errs),
                          "completion_tokens": sum(it["rec"]["completion_tokens"] or 0 for it in its),
                          "ckpt": summarize(ck, "ckpt")[0], "run": summarize(rn, "run")[0]}
    best = sorted(dev_table, key=lambda c: (dev_table[c]["total_errors"], dev_table[c]["completion_tokens"]))[0]
    frozen = a.frozen or best
    # ---- test
    test_ck = [it for it in scored if it["rec"]["phase"] == "test" and it["config"] == frozen]
    miss_ck = [it for it in scored if it["rec"]["phase"] == "missing" and it["config"] == frozen]
    test_run = [it for it in scored if it["rec"]["phase"] == "run_test" and it["config"] == frozen]
    s_test, rows_test = summarize(test_ck, "ckpt")
    s_miss, _ = summarize(miss_ck, "ckpt")
    s_run, rows_run = summarize(test_run, "run")
    # per-task breakdown of test ckpt reports
    by_task = {}
    for task in ("can", "square"):
        sub = [it for it in test_ck if facts[it["rec"]["key"]]["task"] == task]
        if sub:
            by_task[task] = summarize(sub, "ckpt")[0]
    # outcome check of run picks against the training-time target S_i (post hoc, NOT used anywhere upstream)
    import csv
    S = {}
    with open(ROOT / "results/hackathon/baseline_metrics.csv") as f:
        for r in csv.DictReader(f):
            S[r["checkpoint_id"]] = float(r["success"])
    pick_eval = {}
    for run in sorted({it["rec"]["key"] for it in scored if it["rec"]["kind"] == "run"}):
        ids = list(runs[run]["checkpoints"].keys())
        best_S = max(S[c] for c in ids if c in S)
        latest = max((c for c in ids if c in S), key=lambda c: runs[run]["checkpoints"][c]["epoch"])  # latest WITH a target
        acc = truth[f"RUN::{run}"]["evaluate_next_accepted"]
        rule_pick = max(acc, key=lambda c: runs[run]["checkpoints"][c]["epoch"])
        picks = [it["v"].get("evaluate_next") for it in scored if it["rec"]["kind"] == "run" and it["rec"]["key"] == run
                 and it["config"] == frozen and it["v"].get("parse_status") in ("ok", "ok_extracted")]
        pick_eval[run] = {"best_S_among_eval_ckpts": best_S,
                          "latest_epoch_with_target": {"id": latest, "S": S.get(latest), "regret": round(best_S - S[latest], 3)},
                          "rule_pick_latest_tied": {"id": rule_pick, "S": S.get(rule_pick), "regret": (round(best_S - S[rule_pick], 3) if rule_pick in S else None)},
                          "llm_picks": [{"id": p, "S": S.get(p), "regret": (round(best_S - S[p], 3) if p in S else None)} for p in picks]}
    # ---- dashboard JSON
    dash = {"model_id": G.MODEL, "frozen_config": frozen, "note": "EXPLORATORY. Reports generated from fact sheets; "
            "'verification' lists automatic grounding-check results. Simulator labels are ground truth.",
            "checkpoint_reports": {}, "run_reports": {}}
    for it in scored:
        if it["config"] != frozen and not (it["rec"]["phase"] in ("dev", "run_dev")):
            continue
        rec = it["rec"]
        entry = {"phase": rec["phase"], "config": it["config"], "sample": rec["sample"], "report": it["report"],
                 "verification": {k: it["v"].get(k) for k in ("parse_status", "n_numbers", "n_unsupported", "unsupported",
                                                                "n_cited_numbers", "n_miscited", "miscited", "n_bad_key",
                                                                "n_wrong_failure_claims", "wrong_failure_claims",
                                                                "n_misattributed_counts", "misattributed_counts",
                                                                "dominant_reported", "dominant_correct",
                                                                "recommend_reported", "rule", "rule_consistent",
                                                                "evaluate_next", "rule_accepted", "invented_checkpoint_ids")
                                  if k in it["v"]},
                 "latency_s": rec["latency_s"], "prompt_tokens": rec["prompt_tokens"],
                 "completion_tokens": rec["completion_tokens"], "request_id": rec["request_id"]}
        tgt = dash["checkpoint_reports"] if rec["kind"] == "ckpt" else dash["run_reports"]
        tgt.setdefault(rec["key"], []).append(entry)
    (RES / "y4_grounded_reports_reports.json").write_text(json.dumps(dash, indent=1))
    n_calls = sum(1 for line in open(G.EVIDENCE) if f'"{G.TAG}/' in line)
    result = {
        "name": "y4_grounded_reports", "label": "EXPLORATORY / post-hoc",
        "model_id": G.MODEL, "price_usd_per_1M": {"input": PRICE_IN, "output": PRICE_OUT},
        "dev_checkpoints": G.DEV_CKPTS, "dev_runs": G.DEV_RUNS, "test_runs": G.TEST_RUNS,
        "selection_rule": "lowest total dev automatic-error count (unsupported+miscited+wrong failure claims+"
                          "misattributed counts+wrong dominant mode; unparsed=5), ties -> fewer completion tokens",
        "dev_table": dev_table, "dev_selected": best, "frozen_config": frozen,
        "test_ckpt": s_test, "test_ckpt_by_task": by_task, "missing_diagnostics_ckpt": s_miss, "test_run": s_run,
        "run_pick_outcome_check_posthoc": pick_eval,
        "nebius_calls_total_tagged": n_calls,
        "files": {"generations": str(G.GEN.relative_to(ROOT)), "fact_sheets": "data/outputs/hackathon_explore/y4_grounded_reports/fact_sheets_ckpt.json",
                  "run_fact_sheets": "data/outputs/hackathon_explore/y4_grounded_reports/fact_sheets_run.json",
                  "truth_rules": "data/outputs/hackathon_explore/y4_grounded_reports/truth_rules.json",
                  "dashboard_reports": "results/hackathon/explore/y4_grounded_reports_reports.json"},
    }
    (RES / "y4_grounded_reports.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: (v if k != "dev_table" else {c: {kk: vv for kk, vv in t.items() if kk in ("total_errors", "n_unparsed", "completion_tokens")}
                                                      for c, t in v.items()}) for k, v in result.items()
                      if k in ("dev_table", "dev_selected", "frozen_config")}, indent=1))


if __name__ == "__main__":
    main()
