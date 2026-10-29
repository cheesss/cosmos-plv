#!/usr/bin/env python3
"""r1 counterfactual replanning, step 4 — labels from branch rollouts and the pre-registered
endpoints (Environment B, CPU). Protocol: docs/hackathon/explore/r1_counterfactual_replan_protocol.md

Output: results/hackathon/explore/r1_counterfactual_replan_<split>.json and _events.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from cosmos_plv.visual_diagnostics import _first_json_object, parse_perception, split_reasoning  # noqa: E402

BR = ROOT / "data/outputs/hackathon_explore/r1_branch"
JD = ROOT / "data/outputs/hackathon_explore/r1_judge"


def key(ev):
    return f"{ev['checkpoint_id']}_{ev['env_seed']}_{ev['c']}"


def parse_json(raw):
    _, ans = split_reasoning(raw or "")
    blob = _first_json_object(ans)
    try:
        return json.loads(blob) if blob else None
    except json.JSONDecodeError:
        return None


def auroc(score, y):
    s, y = np.asarray(score, float), np.asarray(y, bool)
    ok = ~np.isnan(s)
    s, y = s[ok], y[ok]
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def boot(df, fn, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    cl = df.cluster.unique()
    groups = {c: df[df.cluster == c] for c in cl}
    vals = []
    for _ in range(n):
        pick = rng.choice(cl, len(cl))
        vals.append(fn(pd.concat([groups[c] for c in pick])))
    vals = np.asarray(vals, float)
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


def privileged(ev, exclude_release=False):
    d = np.load(ROOT / ev["episode_npz"])
    t0, T = int(d["plan_t"][ev["c"]]), ev["T"]
    g, z = d["grasped"].astype(bool), d["object_pos"][:, 2].astype(float)
    lost = bool((g[t0:T] & ~g[t0 + 1:T + 1]).any())
    fell = bool(z[t0] - z[t0:T + 1].min() > 0.02)
    if exclude_release and bool(np.asarray(d["success"])[T:T + 17].any()):
        return False  # amendment A1.3: an intended placement is not a surprise
    return lost or fell


def cosmos_rule(row, task):
    """Pre-specified perception-change rule on frames T-16 (a) and T (b); used for Cosmos and Nemotron J1b."""
    if row is None:
        return np.nan
    sa, sb = perception_fields(row["raw_a"], task), perception_fields(row["raw_b"], task)
    if sa is None or sb is None:
        return np.nan
    (la, ha), (lb, hb) = sa, sb
    lost = ha and not hb
    moved = (not ha) and (not hb) and la != lb
    return float(lost or moved)


def perception_fields(raw, task):
    """(object_location, held_by_gripper) from a v6 answer. Amendment A1.4: when the JSON is invalid only
    because the free-text explanation is unquoted (a recurring Nemotron slip), read the two fields by regex."""
    import re

    p = parse_perception(raw, task)
    if p.parse_status in ("ok", "ok_repaired"):
        return p.fields["object_location"], bool(p.fields["held_by_gripper"])
    m_loc = re.search(r'"object_location"\s*:\s*"([a-z_]+)"', raw or "")
    m_held = re.search(r'"held_by_gripper"\s*:\s*(true|false)', raw or "")
    if p.parse_status == "invalid_json" and m_loc and m_held:
        return m_loc.group(1), m_held.group(1) == "true"
    return None


def build(split):
    evs = json.loads((ROOT / f"results/hackathon/explore/r1_events_{split}.json").read_text())["events"]
    by = {}
    for ev in evs:
        by.setdefault(ev["checkpoint_id"], []).append(ev)
    p = {}
    for ck, lst in by.items():
        for br in ("NEW", "OLD"):
            f = BR / split / f"{ck}__{br}.json"
            if not f.exists():
                continue
            r = json.loads(f.read_text())
            for i, ev in enumerate(lst):
                reps = [x for x in r["replicates"] if x["event"] == i]
                p[(key(ev), br)] = {"p": float(np.mean([x["success"] for x in reps])),
                                    "steps": [x["first_success_t"] for x in reps if x["success"]],
                                    "succ40": float(np.mean([bool(x["success40"]) or (x["success"] and x["first_success_t"] <= ev["T"] + 47)
                                                             for x in reps]))}
    j1b = {}
    f = JD / split / "J1b.jsonl"
    if f.exists():
        for l in f.read_text().splitlines():
            r = json.loads(l)
            j1b[r["key"]] = r
    j = {}
    for v in ("J1", "J2", "J2s"):
        f = JD / split / f"{v}.jsonl"
        if f.exists():
            for l in f.read_text().splitlines():
                r = json.loads(l)
                j[(r["key"], v)] = parse_json(r["raw_text"])
    cf = {}
    f = JD / split / "CF.jsonl"
    if f.exists():
        for l in f.read_text().splitlines():
            r = json.loads(l)
            cf[r["key"]] = r
    rows = []
    for ev in evs:
        k = key(ev)
        if (k, "NEW") not in p or (k, "OLD") not in p:
            continue
        row = {"key": k, "kind": ev["kind"], "task": ev["task"], "checkpoint_id": ev["checkpoint_id"],
               "cluster": f"{ev['checkpoint_id']}_{ev['env_seed']}", "T": ev["T"],
               "d_all": ev["d_all"], "d_pos": ev["d_pos"], "d_rot": ev["d_rot"], "d_grip": ev["d_grip"],
               "p_new": p[(k, "NEW")]["p"], "p_old": p[(k, "OLD")]["p"], "episode_success": ev["episode_success"]}
        row["benefit"] = row["p_new"] - row["p_old"]
        j1, j2 = j.get((k, "J1")), j.get((k, "J2"))
        row["j1_parsed"] = j1 is not None
        row["j1_necessity"] = float(j1["replan_necessity"]) if j1 and isinstance(j1.get("replan_necessity"), (int, float)) else np.nan
        row["j1_flag"] = float(bool(j1["unexpected_event"])) if j1 and isinstance(j1.get("unexpected_event"), bool) else np.nan
        row["j1_event"] = j1.get("event") if j1 else None
        def choice(js, new_is_1):
            b = str(js.get("better")) if js else None
            return (None if b is None else "tie" if b == "no_meaningful_difference"
                    else ("new" if (b == "1") == new_is_1 else "old") if b in ("1", "2") else None)
        row["j2_choice"] = choice(j2, ev["new_is_1"])
        row["j2s_choice"] = choice(j.get((k, "J2s")), not ev["new_is_1"])
        row["j2_combined"] = (None if row["j2_choice"] is None or row["j2s_choice"] is None
                              else row["j2_choice"] if row["j2_choice"] == row["j2s_choice"] and row["j2_choice"] != "tie" else "tie")
        row["cosmos_rule"] = cosmos_rule(cf.get(k), ev["task"])
        row["j1b_rule"] = cosmos_rule(j1b.get(k), ev["task"])
        row["privileged_raw"] = float(privileged(ev))
        row["privileged"] = float(privileged(ev, exclude_release=True))
        rows.append(row)
    return pd.DataFrame(rows)


def endpoints(df):
    df = df.copy()
    df["helpful"] = df.benefit > 0
    df["consequential"] = df.benefit != 0
    df["lex"] = df.j1_flag.fillna(0) * 10 + df.d_all  # flag first, then d (pre-specified, not fitted)
    out = {"n_events": int(len(df)), "counts": {}}
    for kind in ("HIGH", "CONTROL"):
        q = df[df.kind == kind]
        out["counts"][kind] = {"n": int(len(q)), "helpful": int((q.benefit > 0).sum()), "harmful": int((q.benefit < 0).sum()),
                               "neutral": int((q.benefit == 0).sum()), "mean_benefit": float(q.benefit.mean()) if len(q) else None,
                               "by_task": q.groupby("task").benefit.agg(["count", "mean", lambda s: int((s > 0).sum()), lambda s: int((s < 0).sum())]).rename(
                                   columns={"<lambda_0>": "helpful", "<lambda_1>": "harmful"}).round(3).to_dict("index")}
    hi = df[df.kind == "HIGH"]
    df["lex_b"] = df.j1b_rule.fillna(0) * 10 + df.d_all
    hi = df[df.kind == "HIGH"]
    scores = ["j1b_rule", "d_all", "d_pos", "d_grip", "j1_necessity", "j1_flag", "cosmos_rule", "privileged", "privileged_raw",
              "lex", "lex_b"]
    out["primary_HIGH"] = {s: {"auroc": auroc(hi[s], hi.helpful), "ci": boot(hi, lambda x, s=s: auroc(x[s], x.helpful))} for s in scores}
    for a, b in (("j1b_rule", "d_all"), ("j1_necessity", "d_all"), ("lex_b", "d_all"), ("cosmos_rule", "d_all")):
        out["primary_HIGH"][f"{a}_minus_{b}"] = {
            "diff": auroc(hi[a], hi.helpful) - auroc(hi[b], hi.helpful),
            "ci": boot(hi, lambda x, a=a, b=b: auroc(x[a], x.helpful) - auroc(x[b], x.helpful))}
    lo = out["primary_HIGH"]["j1b_rule"]["ci"][0]  # amendment A1.1: primary score is J1b
    out["primary_positive"] = bool(lo > 0.5 and out["primary_HIGH"]["j1b_rule_minus_d_all"]["ci"][0] > 0)
    out["all_events"] = {s: {"auroc": auroc(df[s], df.helpful), "ci": boot(df, lambda x, s=s: auroc(x[s], x.helpful))} for s in scores}
    out["all_events_consequential_vs_not"] = {s: auroc(df[s], df.consequential) for s in scores}
    for col in ("j2_combined", "j2_choice", "j2s_choice"):
        c = hi[hi.consequential & hi[col].notna()]
        if len(c):
            better = np.where(c.benefit > 0, "new", "old")
            out[f"{col}_on_consequential_HIGH"] = {
                "n": int(len(c)), "accuracy": float((c[col].values == better).mean()),
                "always_new_accuracy": float((better == "new").mean()),
                "choice_counts": c[col].value_counts().to_dict(),
                "diff_ci": boot(c.assign(better=better), lambda x, col=col: float((x[col].values == x.better.values).mean()
                                                                                 - (x.better.values == "new").mean()))}
    out["parse_rates"] = {"j1": float(df.j1_parsed.mean()), "j2": float(df.j2_choice.notna().mean()),
                          "j2s": float(df.j2s_choice.notna().mean()), "j1b": float(df.j1b_rule.notna().mean()),
                          "cosmos": float(df.cosmos_rule.notna().mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "eval"], required=True)
    a = ap.parse_args()
    df = build(a.split)
    df.to_csv(ROOT / f"results/hackathon/explore/r1_counterfactual_replan_{a.split}_events.csv", index=False)
    res = {"label": "EXPLORATORY; pre-registered in docs/hackathon/explore/r1_counterfactual_replan_protocol.md",
           "split": a.split, **endpoints(df)}
    (ROOT / f"results/hackathon/explore/r1_counterfactual_replan_{a.split}.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: res[k] for k in ("n_events", "counts", "parse_rates")}, indent=1, default=str))
    for k in ("primary_HIGH", "j2_combined_on_consequential_HIGH", "primary_positive"):
        if k in res:
            print(k, json.dumps(res[k], indent=1, default=str))


if __name__ == "__main__":
    main()
