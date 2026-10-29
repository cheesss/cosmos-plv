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


def boot(df, fn, n=2000, seed=0, by="cluster"):
    rng = np.random.default_rng(seed)
    cl = df[by].unique()
    pos = {c: np.flatnonzero((df[by] == c).values) for c in cl}
    vals = []
    for _ in range(n):
        pick = rng.choice(cl, len(cl))
        vals.append(fn(df.iloc[np.concatenate([pos[c] for c in pick])]))
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
    (la, ha, _), (lb, hb, _) = sa, sb
    lost = ha and not hb
    moved = (not ha) and (not hb) and la != lb
    return float(lost or moved)


def perception_fields(raw, task):
    """(object_location, held_by_gripper) from a v6 answer. Amendment A1.4: when the JSON is invalid only
    because the free-text explanation is unquoted (a recurring Nemotron slip), read the two fields by regex."""
    import re

    p = parse_perception(raw, task)
    if p.parse_status in ("ok", "ok_repaired"):
        return p.fields["object_location"], bool(p.fields["held_by_gripper"]), False
    m_loc = re.search(r'"object_location"\s*:\s*"([a-z_]+)"', raw or "")
    m_held = re.search(r'"held_by_gripper"\s*:\s*(true|false)', raw or "")
    if p.parse_status == "invalid_json" and m_loc and m_held:
        return m_loc.group(1), m_held.group(1) == "true", True
    return None


def regex_repairs(row, task):
    """Number of the two frames read by the A1.4 regex repair (0-2), or NaN when the judge row is missing."""
    if row is None:
        return np.nan
    fs = [perception_fields(row[f"raw_{s}"], task) for s in ("a", "b")]
    return float(sum(1 for f in fs if f is not None and f[2]))


def stage40(x, T):
    """Ordinal stage 40 steps after the branch: 0 none, 1 grasped, 2 lifted (> 2 cm, the privileged-reference
    threshold), 3 success. Lift threshold chosen after the protocol (it names the stages but not the cut).
    A replicate that stopped earlier has no stage sample: success by then counts as 3, a max_steps stop is NaN."""
    if x["success40"] or (x["success"] and x["first_success_t"] <= T + 47):
        return 3
    if x["objz40"] is None:
        return np.nan
    if x["objz40"] - x["z0"] > 0.02:
        return 2
    return 1 if x["grasped40"] else 0


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
                reps = sorted((x for x in r["replicates"] if x["event"] == i), key=lambda x: x["k"])
                p[(key(ev), br)] = {"p": float(np.mean([x["success"] for x in reps])),
                                    "succ": [bool(x["success"]) for x in reps],
                                    "steps": [x["first_success_t"] for x in reps if x["success"]],
                                    "stage40": float(np.nanmean([stage40(x, ev["T"]) for x in reps])),
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
    missing = [key(ev) for ev in evs if (key(ev), "NEW") not in p or (key(ev), "OLD") not in p]
    if missing:
        print(f"WARNING: {len(missing)} events have no branch result and are left out")
    for ev in evs:
        k = key(ev)
        if (k, "NEW") not in p or (k, "OLD") not in p:
            continue
        rec_success = np.asarray(np.load(ROOT / ev["episode_npz"])["success"]).astype(bool)
        first = int(np.argmax(rec_success)) if rec_success.any() else -1
        row = {"key": k, "kind": ev["kind"], "task": ev["task"], "checkpoint_id": ev["checkpoint_id"],
               "cluster": f"{ev['checkpoint_id']}_{ev['env_seed']}", "T": ev["T"],
               "d_all": ev["d_all"], "d_pos": ev["d_pos"], "d_rot": ev["d_rot"], "d_grip": ev["d_grip"],
               "p_new": p[(k, "NEW")]["p"], "p_old": p[(k, "OLD")]["p"], "episode_success": ev["episode_success"],
               "succ_new": p[(k, "NEW")]["succ"], "succ_old": p[(k, "OLD")]["succ"],
               "steps_new": float(np.mean(p[(k, "NEW")]["steps"])) if len(p[(k, "NEW")]["steps"]) == 4 else np.nan,
               "steps_old": float(np.mean(p[(k, "OLD")]["steps"])) if len(p[(k, "OLD")]["steps"]) == 4 else np.nan,
               "stage40_new": p[(k, "NEW")]["stage40"], "stage40_old": p[(k, "OLD")]["stage40"],
               "recorded_first_success": first,
               # sensitivity (post hoc, after verification): the recorded episode had already succeeded, or succeeded
               # inside the bit-exact 7-step NEW segment, so both branches are decided by construction
               "determined": bool(0 <= first <= ev["T"] + 7)}
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
        row["new_is_1"] = bool(ev["new_is_1"])
        row["j2_choice"] = choice(j2, ev["new_is_1"])
        row["j2s_choice"] = choice(j.get((k, "J2s")), not ev["new_is_1"])
        row["j2_combined"] = (None if row["j2_choice"] is None or row["j2s_choice"] is None
                              else row["j2_choice"] if row["j2_choice"] == row["j2s_choice"] and row["j2_choice"] != "tie" else "tie")
        row["cosmos_rule"] = cosmos_rule(cf.get(k), ev["task"])
        row["j1b_rule"] = cosmos_rule(j1b.get(k), ev["task"])
        row["j1b_regex_frames"] = regex_repairs(j1b.get(k), ev["task"])
        row["cosmos_regex_frames"] = regex_repairs(cf.get(k), ev["task"])
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
    scores = ["j1b_rule", "d_all", "d_pos", "d_rot", "d_grip", "j1_necessity", "j1_flag", "cosmos_rule", "privileged",
              "privileged_raw", "lex", "lex_b"]
    out["primary_HIGH"] = {s: {"auroc": auroc(hi[s], hi.helpful), "ci": boot(hi, lambda x, s=s: auroc(x[s], x.helpful))} for s in scores}
    for a, b in (("j1b_rule", "d_all"), ("j1_necessity", "d_all"), ("lex", "d_all"), ("lex_b", "d_all"), ("cosmos_rule", "d_all")):
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
                          "cosmos": float(df.cosmos_rule.notna().mean()),
                          "j1b_regex_repaired_frames": int(df.j1b_regex_frames.sum()),
                          "cosmos_regex_repaired_frames": int(df.cosmos_regex_frames.sum())}
    out["secondary_outcomes_HIGH_not_determined"] = secondary_outcomes(df[(df.kind == "HIGH") & ~df.determined])
    out["post_hoc_after_verification"] = post_hoc(df)
    return out


def secondary_outcomes(q):
    """Pre-registered secondary outcomes (steps to success, stage 40 steps after the branch), on HIGH events that were
    not already decided by construction (see post_hoc)."""
    from scipy.stats import binomtest

    both = q[q.steps_new.notna() & q.steps_old.notna()]
    faster_new, faster_old = int((both.steps_new < both.steps_old).sum()), int((both.steps_new > both.steps_old).sum())
    st = q.stage40_new - q.stage40_old
    return {"steps_to_success_all_8_succeed": {"n": int(len(both)), "new_faster": faster_new, "old_faster": faster_old,
                                               "ties": int(len(both) - faster_new - faster_old),
                                               "mean_steps_saved_by_new": float((both.steps_old - both.steps_new).mean()),
                                               "sign_test_p": float(binomtest(faster_new, faster_new + faster_old).pvalue)
                                               if faster_new + faster_old else None,
                                               "auroc_d_all_new_faster": auroc(both.d_all, both.steps_new < both.steps_old)},
            "stage40_mean_new_minus_old": {"n": int(len(q)), "mean": float(st.mean()), "new_higher": int((st > 0).sum()),
                                           "old_higher": int((st < 0).sum()),
                                           "sign_test_p": float(binomtest(int((st > 0).sum()), int((st != 0).sum())).pvalue)
                                           if (st != 0).any() else None,
                                           "note": "stage 0 none / 1 grasped / 2 lifted > 2 cm / 3 success; lift cut chosen post hoc"}}


def post_hoc(df):
    """Analyses added after the adversarial verification (2026-10-29), not pre-registered. Reported as such."""
    from scipy.stats import binomtest, fisher_exact

    df = df.copy()
    df["helpful"], df["consequential"] = df.benefit > 0, df.benefit != 0
    hi = df[df.kind == "HIGH"]
    res = {"label": "POST HOC (after verification); not pre-registered"}

    # 1. events decided by construction: recorded success at or before T+7 (post-success boundary in the recorder's
    #    16-step hold, or success inside the bit-exact NEW segment)
    und = hi[~hi.determined]
    res["determined_events"] = {"HIGH": int(hi.determined.sum()), "CONTROL": int(df[df.kind == "CONTROL"].determined.sum()),
                                "consequential_among_determined": int(hi[hi.determined].consequential.sum())}
    sc = ["j1b_rule", "d_all", "d_grip", "cosmos_rule", "j1_necessity", "privileged", "privileged_raw"]
    res["HIGH_excluding_determined"] = {
        "n": int(len(und)), "helpful": int(und.helpful.sum()), "harmful": int((und.benefit < 0).sum()),
        **{s: {"auroc": auroc(und[s], und.helpful), "ci": boot(und, lambda x, s=s: auroc(x[s], x.helpful))} for s in sc},
        "j1b_rule_minus_d_all": {"diff": auroc(und.j1b_rule, und.helpful) - auroc(und.d_all, und.helpful),
                                 "ci": boot(und, lambda x: auroc(x.j1b_rule, x.helpful) - auroc(x.d_all, x.helpful))}}
    ok = hi[hi.j1b_regex_frames == 0]
    res["j1b_rule_regex_frames_as_missing"] = {"n": int(len(ok)), "auroc": auroc(ok.j1b_rule, ok.helpful)}
    res["j1b_flags_on_determined"] = {"flags_HIGH": int(hi.j1b_rule.sum()), "flags_on_determined": int(hi[hi.determined].j1b_rule.sum())}

    # 2. good vs bad replans: helpful vs harmful among consequential HIGH events
    c = hi[hi.consequential]
    res["helpful_vs_harmful_consequential_HIGH"] = {
        "n": int(len(c)), "helpful": int(c.helpful.sum()),
        "abs_benefit_counts": c.benefit.abs().round(2).value_counts().sort_index().to_dict(),
        **{s: {"auroc": auroc(c[s], c.helpful), "ci_episode": boot(c, lambda x, s=s: auroc(x[s], x.helpful)),
               "ci_checkpoint": boot(c, lambda x, s=s: auroc(x[s], x.helpful), by="checkpoint_id")}
           for s in ("d_all", "d_grip", "j1b_rule", "cosmos_rule")}}
    full = c[c.benefit.abs() == 1]
    res["full_flips"] = {"n": int(len(full)), "helpful": int(full.helpful.sum()),
                         "j1b_flagged": int(full.j1b_rule.fillna(0).sum()), "cosmos_flagged": int(full.cosmos_rule.fillna(0).sum()),
                         "j2_combined": full.j2_combined.value_counts(dropna=False).to_dict(), "keys": full.key.tolist()}

    # 3. does d localize the replan that matters? same-episode CONTROL boundaries have small d
    res["consequential_vs_neutral_HIGH_d_all"] = {"auroc": auroc(hi.d_all, hi.consequential),
                                                  "ci": boot(hi, lambda x: auroc(x.d_all, x.consequential))}
    rate = {k: (int(df[df.kind == k].consequential.sum()), int((df.kind == k).sum())) for k in ("HIGH", "CONTROL")}
    tab = [[rate["HIGH"][0], rate["HIGH"][1] - rate["HIGH"][0]], [rate["CONTROL"][0], rate["CONTROL"][1] - rate["CONTROL"][0]]]
    pair = df.pivot_table(index="cluster", columns="kind", values="consequential", aggfunc="first").dropna()
    h_only, c_only = int((pair.HIGH.astype(bool) & ~pair.CONTROL.astype(bool)).sum()), int((~pair.HIGH.astype(bool) & pair.CONTROL.astype(bool)).sum())
    res["HIGH_vs_CONTROL_consequential"] = {
        "rates": {k: v[0] / v[1] for k, v in rate.items()}, "counts": rate, "fisher_p": float(fisher_exact(tab)[1]),
        "paired_episodes": int(len(pair)), "paired_HIGH": int(pair.HIGH.sum()), "paired_CONTROL": int(pair.CONTROL.sum()),
        "discordant_HIGH_only": h_only, "discordant_CONTROL_only": c_only,
        "mcnemar_exact_p": float(binomtest(h_only, h_only + c_only).pvalue) if h_only + c_only else None,
        "median_d": {k: float(df[df.kind == k].d_all.median()) for k in ("HIGH", "CONTROL")}}
    res["d_all_within_checkpoint_concordance_consequential"] = within_group_concordance(df, "d_all", "consequential", "checkpoint_id")
    res["d_all_predicts_recorded_episode_failure_HIGH"] = auroc(hi.d_all, ~hi.episode_success.astype(bool))

    # 4. are the K=4 labels distinguishable from replicate noise? reshuffle the 8 outcomes between the branches
    rng = np.random.default_rng(0)
    obs = {"consequential": int(hi.consequential.sum()), "helpful": int(hi.helpful.sum()),
           "sum_abs_benefit": float(hi.benefit.abs().sum())}
    pooled = [np.r_[a, b].astype(float) for a, b in zip(hi.succ_new, hi.succ_old)]
    sims = {k: [] for k in obs}
    for _ in range(5000):
        b = np.array([(lambda s: s[:4].mean() - s[4:].mean())(rng.permutation(x)) for x in pooled])
        sims["consequential"].append((b != 0).sum())
        sims["helpful"].append((b > 0).sum())
        sims["sum_abs_benefit"].append(np.abs(b).sum())
    res["permutation_null_HIGH"] = {k: {"observed": obs[k], "null_mean": float(np.mean(v)),
                                        "p_ge_observed": float(np.mean(np.asarray(v) >= obs[k] - 1e-9))} for k, v in sims.items()}

    # 5. J2: coverage and accuracy on decided (non-tie) combined choices; label-position bias
    dec = c[c.j2_combined.isin(["new", "old"])]
    better = np.where(dec.benefit > 0, "new", "old")
    lab = lambda ch, new_is_1: None if ch not in ("new", "old") else ("1" if (ch == "new") == new_is_1 else "2")
    j2_lab = [lab(ch, n1) for ch, n1 in zip(df.j2_choice, df.new_is_1)]
    j2s_lab = [lab(ch, not n1) for ch, n1 in zip(df.j2s_choice, df.new_is_1)]
    res["j2_combined_decided_consequential_HIGH"] = {
        "coverage": f"{len(dec)}/{len(c)}", "accuracy": float((dec.j2_combined.values == better).mean()) if len(dec) else None,
        "always_new_accuracy_same_events": float((better == "new").mean()) if len(dec) else None,
        "label_1_share_all_events": {"J2": float(np.mean([x == "1" for x in j2_lab if x])),
                                     "J2s": float(np.mean([x == "1" for x in j2s_lab if x]))}}
    return res


def within_group_concordance(df, score, y, group):
    """AUROC restricted to positive/negative pairs from the same group (pooled over groups)."""
    num = den = 0.0
    for _, g in df.groupby(group):
        s, t = g[score].values.astype(float), g[y].values.astype(bool)
        pos, neg = s[t], s[~t]
        if len(pos) and len(neg):
            num += (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
            den += len(pos) * len(neg)
    return float(num / den) if den else float("nan")


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
