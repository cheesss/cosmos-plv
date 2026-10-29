#!/usr/bin/env python
"""y4_grounded_reports, step 1: build compact per-checkpoint and per-run FACT SHEETS.

EXPLORATORY / post-hoc. Only measured facts go in, plus deterministic derived statistics
(counts, Wilson CIs, within-run ranks). The training-time target S_i (50-rollout success,
also encoded in checkpoint filenames as test_mean_score) is deliberately NOT included,
since it is the quantity PLV is supposed to predict.

PRE-REGISTERED RULES (written here before any LLM report was generated):

RULE_CKPT ("spend more evaluation rollouts on this checkpoint?"):
  J = Nemotron-3.5-Super-VL v2 judged success count over the 10 eval episodes.
  med = median of J over the run's checkpoints that have Nemotron diagnostics.
  If the run has PLV-1 smoothness (plv1_smoothness, lower = better) for its eval checkpoints:
      YES iff J >= med AND plv1 rank (1 = lowest) <= ceil(0.3 * n_with_metric)
  else (only val_loss available, which is not a validated screening metric):
      YES iff J >= med
  Checkpoints without Nemotron diagnostics: rule undefined (reported separately).
  PLV-1 smoothness was chosen because it is the headline DP metric of the original paper
  (docs/hackathon/00_repository_audit.md), not because of its correlation on this data.

RULE_DOMINANT (dominant failure mode):
  most frequent simulator failure_type among failed episodes; ties -> any tied type accepted;
  no failures -> "none".

RULE_RUN ("which checkpoint to evaluate next"):
  accepted set = checkpoints whose J equals the run maximum of J (ties all accepted).
"""
from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = ROOT / "data/outputs/hackathon_rollouts_eval"
DIAG = ROOT / "data/outputs/hackathon_cosmos"
OUT = ROOT / "data/outputs/hackathon_explore/y4_grounded_reports"
FTYPES = ["no_grasp", "dropped", "not_placed", "other"]
CHEAP_COLS = ["val_loss", "val_action_l1", "plv1_smoothness", "open_loop_smoothness",
              "boundary_jump_h10", "replan_overlap_h10"]  # all lower = better
CONTROL_HZ = 20.0


def wilson(k: int, n: int, z: float = 1.959964):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 3), round(min(1.0, c + h), 3))


def load_jsonl(name):
    rows = {}
    p = DIAG / f"{name}.jsonl"
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        rows[(d["checkpoint_id"], int(d["env_seed"]))] = d
    return rows


def load_cheap():
    import csv
    out = {}
    with open(ROOT / "results/hackathon/baseline_metrics.csv") as f:
        for r in csv.DictReader(f):
            vals = {}
            for c in CHEAP_COLS:
                v = r.get(c, "")
                vals[c] = float(v) if v not in ("", None) else None
            out[r["checkpoint_id"]] = vals
    return out


def run_of(cid: str) -> str:
    return cid.rsplit("_e", 1)[0]


def rank_lower_better(values: dict):
    """values: id -> float|None. Competition rank (1 = lowest); ties share the best rank."""
    vv = {k: v for k, v in values.items() if v is not None}
    out = {}
    for k, v in vv.items():
        out[k] = 1 + sum(1 for w in vv.values() if w < v)
    return out, len(vv)


def rank_higher_better(values: dict):
    vv = {k: v for k, v in values.items() if v is not None}
    return {k: 1 + sum(1 for w in vv.values() if w > v) for k, v in vv.items()}, len(vv)


def r4(x):
    return None if x is None else float(f"{x:.4g}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    neb = load_jsonl("eval_nebius_v2")
    nebp = load_jsonl("eval_nebius_v2_pad201")
    cos = load_jsonl("eval_local_v6")
    cheap = load_cheap()

    ckpts = sorted(p.name for p in EVAL_DIR.iterdir() if (p / "summary.json").is_file())
    base = {}
    for cid in ckpts:
        s = json.loads((EVAL_DIR / cid / "summary.json").read_text())
        eps = sorted(s["episodes"], key=lambda e: e["env_seed"])
        epoch = int(s["checkpoint"]["epoch"])
        task = s["checkpoint"]["task"]
        ep_facts = {}
        sim_succ = 0
        ftc = Counter()
        n_neb = n_cos = n_nebp = 0
        neb_succ = neb_agree = cos_succ = cos_agree = nebp_succ = nebp_agree = 0
        neb_ftc = Counter()
        ft_cmp = ft_match = 0
        succ_times = []
        for e in eps:
            seed = int(e["env_seed"])
            ok = bool(e["success"])
            sim_succ += ok
            ft = e["failure_type"] if not ok else "none"
            if not ok:
                ftc[ft] += 1
            rec = {"sim_success": ok, "sim_failure_type": ft,
                   "sim_grasped": bool(e.get("grasped")), "sim_lifted": bool(e.get("lifted"))}
            if ok and e.get("first_success_step", -1) >= 0:
                t = round(e["first_success_step"] / CONTROL_HZ, 1)
                rec["sim_success_time_s"] = t
                succ_times.append(t)
            d = neb.get((cid, seed))
            if d is not None and d.get("parse_status") == "ok" and d.get("cosmos_task_success") is not None:
                n_neb += 1
                js = bool(d["cosmos_task_success"])
                neb_succ += js
                neb_agree += (js == ok)
                jft = "none" if js else (d.get("cosmos_failure_type") or "unparsed")
                rec["nemotron_success"] = js
                rec["nemotron_failure_type"] = jft
                if not js:
                    neb_ftc[jft] += 1
                if not ok:
                    ft_cmp += 1
                    ft_match += (jft == ft)
                if (not ok) or (not js):
                    expl = (d.get("cosmos_explanation") or "").strip()
                    if len(expl) > 170:
                        expl = expl[:167].rsplit(" ", 1)[0] + "..."
                    rec["nemotron_note"] = expl
            elif d is not None:
                rec["nemotron_success"] = None
            dp = nebp.get((cid, seed))
            if dp is not None and dp.get("parse_status") == "ok" and dp.get("cosmos_task_success") is not None:
                n_nebp += 1
                nebp_succ += bool(dp["cosmos_task_success"])
                nebp_agree += (bool(dp["cosmos_task_success"]) == ok)
            c = cos.get((cid, seed))
            if c is not None and c.get("parse_status") == "ok" and c.get("cosmos_task_success") is not None:
                n_cos += 1
                cs = bool(c["cosmos_task_success"])
                cos_succ += cs
                cos_agree += (cs == ok)
                rec["cosmos_success"] = cs
            ep_facts[str(seed)] = rec
        n = len(eps)
        lo, hi = wilson(sim_succ, n)
        sim = {"n_episodes": n, "successes": sim_succ, "failures": n - sim_succ,
               "success_rate": round(sim_succ / n, 3) if n else None,
               "success_rate_wilson95_lo": lo, "success_rate_wilson95_hi": hi,
               "failure_types": {k: ftc[k] for k in FTYPES if ftc[k]}}
        if succ_times:
            sim["median_success_time_s"] = round(statistics.median(succ_times), 1)
        f = {"checkpoint_id": cid, "run": run_of(cid), "task": task, "epoch": epoch,
             "eval_env_seeds": f"{eps[0]['env_seed']}-{eps[-1]['env_seed']}",
             "sim": sim}
        if n_neb:
            f["nemotron"] = {"n_judged": n_neb, "judged_successes": neb_succ,
                             "judged_failures": n_neb - neb_succ,
                             "agree_with_sim": neb_agree,
                             "judged_failure_types": dict(sorted(neb_ftc.items())),
                             "failure_type_matches_sim": ft_match,
                             "failure_type_compared_on_sim_failures": ft_cmp}
        else:
            f["nemotron"] = None
        f["nemotron_padded"] = ({"n_judged": n_nebp, "judged_successes": nebp_succ, "agree_with_sim": nebp_agree}
                                if n_nebp else None)
        f["cosmos"] = ({"n_judged": n_cos, "judged_successes": cos_succ, "agree_with_sim": cos_agree}
                       if n_cos else None)
        cm = cheap.get(cid, {})
        f["cheap_metrics"] = {k: r4(v) for k, v in cm.items() if v is not None}
        f["episodes"] = ep_facts
        base[cid] = f

    # within-run context (ranks over the run's eval checkpoints)
    runs = sorted({run_of(c) for c in base})
    truth = {}
    run_sheets = {}
    for run in runs:
        ids = [c for c in base if run_of(c) == run]
        J = {c: (base[c]["nemotron"]["judged_successes"] if base[c]["nemotron"] else None) for c in ids}
        Jv = [v for v in J.values() if v is not None]
        med = statistics.median(Jv) if Jv else None
        jrank, nj = rank_higher_better(J)
        cheap_ranks = {}
        for col in CHEAP_COLS:
            rk, nn = rank_lower_better({c: base[c]["cheap_metrics"].get(col) for c in ids})
            cheap_ranks[col] = (rk, nn)
        has_plv1 = cheap_ranks["plv1_smoothness"][1] > 0
        for c in ids:
            ctx = {"n_eval_checkpoints_in_run": len(ids),
                   "n_checkpoints_with_nemotron": nj,
                   "run_median_nemotron_judged_successes": med,
                   "run_max_nemotron_judged_successes": max(Jv) if Jv else None,
                   "nemotron_judged_rank_in_run": jrank.get(c),
                   "epochs_in_run": sorted(base[x]["epoch"] for x in ids)}
            for col in CHEAP_COLS:
                rk, nn = cheap_ranks[col]
                if c in rk:
                    ctx[f"{col}_rank_in_run"] = rk[c]
                    ctx[f"{col}_n_ranked"] = nn
            base[c]["run_context"] = ctx
            # ---- pre-registered truth for verification (NOT shown to the LLM) ----
            ft = base[c]["sim"]["failure_types"]
            if ft:
                m = max(ft.values())
                dom = sorted(k for k, v in ft.items() if v == m)
            else:
                dom = ["none"]
            if J[c] is None:
                rule = None
            elif has_plv1:
                n30 = math.ceil(0.3 * cheap_ranks["plv1_smoothness"][1])
                rk = cheap_ranks["plv1_smoothness"][0].get(c)
                rule = "yes" if (J[c] >= med and rk is not None and rk <= n30) else "no"
            else:
                rule = "yes" if J[c] >= med else "no"
            truth[c] = {"dominant_failure_mode_accepted": dom, "rule_ckpt": rule,
                        "rule_uses_plv1": has_plv1, "J": J[c], "run_median_J": med}
        mx = max(Jv) if Jv else None
        run_sheets[run] = {
            "run": run, "task": base[ids[0]]["task"], "n_eval_checkpoints": len(ids),
            "eval_env_seeds": base[ids[0]]["eval_env_seeds"], "episodes_per_checkpoint": 10,
            "run_median_nemotron_judged_successes": med, "run_max_nemotron_judged_successes": mx,
            "cheap_metrics_available": sorted({k for c in ids for k in base[c]["cheap_metrics"]}),
            "checkpoints": {
                c: {"epoch": base[c]["epoch"],
                    "sim_successes": base[c]["sim"]["successes"],
                    "sim_failure_types": base[c]["sim"]["failure_types"],
                    "nemotron_judged_successes": J[c],
                    "nemotron_agree_with_sim": base[c]["nemotron"]["agree_with_sim"] if base[c]["nemotron"] else None,
                    "cosmos_judged_successes": base[c]["cosmos"]["judged_successes"] if base[c]["cosmos"] else None,
                    "nemotron_judged_rank_in_run": jrank.get(c),
                    **{k: v for k, v in base[c]["cheap_metrics"].items()},
                    **{f"{col}_rank_in_run": base[c]["run_context"][f"{col}_rank_in_run"]
                       for col in CHEAP_COLS if f"{col}_rank_in_run" in base[c]["run_context"]}}
                for c in sorted(ids, key=lambda x: base[x]["epoch"])},
        }
        truth[f"RUN::{run}"] = {"evaluate_next_accepted": sorted(c for c in ids if J[c] is not None and J[c] == mx)}

    (OUT / "fact_sheets_ckpt.json").write_text(json.dumps(base, indent=1, sort_keys=False))
    (OUT / "fact_sheets_run.json").write_text(json.dumps(run_sheets, indent=1))
    (OUT / "truth_rules.json").write_text(json.dumps(truth, indent=1))
    print("checkpoints", len(base), "runs", runs)
    print("with nemotron", sum(1 for c in base if base[c]["nemotron"]))
    lens = [len(json.dumps(base[c], separators=(",", ":"))) for c in base]
    print("fact sheet chars min/median/max", min(lens), statistics.median(lens), max(lens))


if __name__ == "__main__":
    main()
