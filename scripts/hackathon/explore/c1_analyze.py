#!/usr/bin/env python3
"""c1 coverage shift, step 3 — pre-registered endpoints (Environment B, CPU).
Protocol: docs/hackathon/explore/c1_coverage_shift_protocol.md

Inputs: data/outputs/hackathon_explore/c1_coverage/*/summary.json (simulator truth),
        data/outputs/hackathon_explore/c1_judge/{nebius_v2,local_v6}.jsonl (frozen Experiment 2 judges),
        results/hackathon/checkpoint_manifest.csv (q = training-time success over 50 rollouts).
Output: results/hackathon/explore/c1_coverage_shift.json and c1_coverage_shift_episodes.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
REC = ROOT / "data/outputs/hackathon_explore/c1_coverage"
JD = ROOT / "data/outputs/hackathon_explore/c1_judge"
JUDGES = {"nemotron": "nebius_v2.jsonl", "cosmos": "local_v6.jsonl", "nemotron_v2c": "nebius_v2c.jsonl"}
# robosuite PickPlaceCan goal bin (bin2_pos (0.1, 0.28, 0.8), BinsArena size (0.39, 0.49)); the can's compartment is
# the +x/+y quarter (object id 3)
BIN2, SIZE = np.array([0.1, 0.28, 0.8]), np.array([0.39, 0.49])


def auroc(score, y):
    s, y = np.asarray(score, float), np.asarray(y, bool)
    ok = ~np.isnan(s)
    s, y = s[ok], y[ok]
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def boot(df, fn, by, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    cl = df[by].unique()
    pos = {c: np.flatnonzero((df[by] == c).values) for c in cl}
    vals = [fn(df.iloc[np.concatenate([pos[c] for c in rng.choice(cl, len(cl))])]) for _ in range(n)]
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(c - h), float(c + h)]


def final_place(npz):
    """Where the object came to rest: target compartment / goal bin, other compartment / elsewhere (privileged)."""
    p = np.load(ROOT / npz)["object_pos"][-1]
    in_bin = abs(p[0] - BIN2[0]) < SIZE[0] / 2 and abs(p[1] - BIN2[1]) < SIZE[1] / 2 and BIN2[2] < p[2] < BIN2[2] + 0.1
    in_target = BIN2[0] < p[0] < BIN2[0] + SIZE[0] / 2 and BIN2[1] < p[1] < BIN2[1] + SIZE[1] / 2 and in_bin
    return "target_compartment" if in_target else "goal_bin_other_compartment" if in_bin else "elsewhere"


def final_place_fine(npz):
    """Post hoc (after verification): split 'elsewhere' into cans resting on a goal-bin rim or divider and the rest."""
    p = np.load(ROOT / npz)["object_pos"][-1]
    coarse = final_place(npz)
    over_bin = abs(p[0] - BIN2[0]) < SIZE[0] / 2 and abs(p[1] - BIN2[1]) < SIZE[1] / 2
    return "goal_bin_rim_or_divider" if coarse == "elsewhere" and over_bin and p[2] >= BIN2[2] + 0.1 else coarse


def black_video(path):
    """Post hoc: recorder/EGL artefact found by the verifier (frames after the first are black)."""
    from cosmos_plv.video_recorder import read_mp4

    m = np.array([f.mean() for f in read_mp4(ROOT / path)])
    return bool(len(m) > 1 and (m[1:] < 5).mean() > 0.5)


def load(rec_dir, judge_files, manifest):
    q = manifest.set_index("checkpoint_id").train_rollout_success
    rows = []
    for summ in sorted(Path(rec_dir).glob("*/summary.json")):
        s = json.loads(summ.read_text())
        for ep in s["episodes"]:
            rows.append({"checkpoint_id": s["checkpoint_id"], "stratum": s.get("stratum"), "env_seed": ep["env_seed"],
                         "task": ep["task"], "q": float(q[s["checkpoint_id"]]), "success": bool(ep["success"]),
                         "failure_type": ep["failure_type"], "grasped": ep["grasped"], "lifted": ep["lifted"],
                         "final_place": final_place(ep["episode_npz"]),
                         "final_place_fine": final_place_fine(ep["episode_npz"]),
                         "black_video": black_video(ep["video"]) if ep.get("video") else None})
    df = pd.DataFrame(rows)
    for name, f in judge_files.items():
        verdict = {}
        if Path(f).exists():
            for line in Path(f).read_text().splitlines():
                r = json.loads(line)
                ok = r.get("parse_status") in ("ok", "ok_repaired") and isinstance(r.get("cosmos_task_success"), bool)
                verdict[(r["checkpoint_id"], int(r["env_seed"]))] = float(r["cosmos_task_success"]) if ok else np.nan
        df[f"{name}_says_success"] = [verdict.get((c, int(s)), np.nan) for c, s in zip(df.checkpoint_id, df.env_seed)]
    return df


def judge_block(df, col, strata=True):
    f = df[~df.success & df[col].notna()].copy()
    f["fs"] = f[col] == 1
    out = {"n_failures": int(len(f)), "false_successes": int(f.fs.sum()),
           "auroc_q": auroc(f.q, f.fs),
           "ci_checkpoint": boot(f, lambda x: auroc(x.q, x.fs), "checkpoint_id"),
           "ci_seed": boot(f, lambda x: auroc(x.q, x.fs), "env_seed")}
    if strata and "stratum" in f and f.stratum.notna().all():
        out["by_stratum"] = {}
        for s, g in f.groupby("stratum"):
            k, n = int(g.fs.sum()), int(len(g))
            out["by_stratum"][s] = {"false_successes": k, "failures": n, "rate": k / n if n else None, "wilson": wilson(k, n),
                                    "failure_types": g.failure_type.value_counts().to_dict(),
                                    "distinct_seeds": int(g.env_seed.nunique())}
        if {"T", "M"} <= set(out["by_stratum"]):
            from scipy.stats import fisher_exact
            t, m = out["by_stratum"]["T"], out["by_stratum"]["M"]
            out["fisher_T_vs_M_p"] = float(fisher_exact([[t["false_successes"], t["failures"] - t["false_successes"]],
                                                         [m["false_successes"], m["failures"] - m["false_successes"]]])[1])
    d = f[f.failure_type == "dropped"]
    out["dropped_only"] = {"n": int(len(d)), "false_successes": int(d.fs.sum()), "auroc_q": auroc(d.q, d.fs),
                           "ci_checkpoint": boot(d, lambda x: auroc(x.q, x.fs), "checkpoint_id") if len(d) else None}
    out["false_success_by_final_place"] = {p: {"failures": int(len(g)), "false_successes": int(g.fs.sum())}
                                           for p, g in f.groupby("final_place")}
    a = df[df[col].notna()]
    claimed = a[a[col] == 1]
    out["precision_of_claimed_success"] = {
        "all": float(claimed.success.mean()) if len(claimed) else None,
        **({s: float(g[g[col] == 1].success.mean()) if (g[col] == 1).any() else None for s, g in a.groupby("stratum")}
           if a.stratum.notna().all() else {})}
    out["sensitivity_on_successes"] = float((a[a.success][col] == 1).mean()) if a.success.any() else None
    return out


def select_v2c(df):
    """Amendment A1, secondary 8: every failure plus 40 random successes per stratum (numpy seed 0)."""
    rng = np.random.default_rng(0)
    pick = [df[~df.success]]
    for s, g in df[df.success].groupby("stratum"):
        pick.append(g.iloc[np.sort(rng.choice(len(g), min(40, len(g)), replace=False))])
    sel = pd.concat(pick)
    (JD / "v2c_episodes.json").write_text(json.dumps([[c, int(e)] for c, e in zip(sel.checkpoint_id, sel.env_seed)]))
    print("v2c episodes:", len(sel), "failures:", int((~sel.success).sum()))


def post_hoc(df):
    """Added after the adversarial verification (2026-10-29); not pre-registered."""
    out = {"label": "POST HOC (after verification); not pre-registered"}
    bl = df[df.black_video == True]  # noqa: E712
    out["black_videos"] = {"n": int(len(bl)), "strata": bl.stratum.value_counts().to_dict(), "seeds": sorted(bl.env_seed.unique().tolist()),
                           "all_true_successes": bool(bl.success.all()) if len(bl) else None,
                           "judged_success": {j: int((bl[f"{j}_says_success"] == 1).sum()) for j in ("nemotron", "cosmos")}}
    ok = df[df.black_video != True]  # noqa: E712
    out["sensitivity_valid_videos"] = {j: float((ok[ok.success][f"{j}_says_success"] == 1).mean()) for j in ("nemotron", "cosmos")}
    f = df[~df.success].copy()
    f["fs_n"], f["fs_c"] = f.nemotron_says_success == 1, f.cosmos_says_success == 1
    out["failures_by_place_and_stratum"] = {
        pl: {s: {"failures": int(len(g)), "nemotron_fs": int(g.fs_n.sum()), "cosmos_fs": int(g.fs_c.sum())}
             for s, g in q.groupby("stratum")} for pl, q in f.groupby("final_place_fine")}
    out["failure_types_by_stratum"] = f.groupby(["stratum", "failure_type"]).size().unstack(fill_value=0).to_dict("index")
    out["wrong_compartment_rate_per_episode"] = {s: float((g.final_place == "goal_bin_other_compartment").mean())
                                                 for s, g in df.groupby("stratum")}
    seeds = f.env_seed.value_counts()
    out["false_successes_by_seed_nemotron"] = f[f.fs_n].env_seed.value_counts().to_dict()
    out["leave_one_seed_out_nemotron"] = {}
    for sd in seeds.index[:6]:
        g = f[f.env_seed != sd]
        out["leave_one_seed_out_nemotron"][int(sd)] = {"auroc_q": auroc(g.q, g.fs_n),
                                                       "ci_checkpoint": boot(g, lambda x: auroc(x.q, x.fs_n), "checkpoint_id")}
    common = f[f.env_seed < 600012]  # seeds shared by all three strata
    out["common_seeds_600000_600011"] = {s: {"failures": int(len(g)), "nemotron_fs": int(g.fs_n.sum())}
                                         for s, g in common.groupby("stratum")}
    rng = np.random.default_rng(0)
    cks, sds = f.checkpoint_id.unique(), f.env_seed.unique()
    vals = []
    for _ in range(2000):  # crossed checkpoint x seed bootstrap: an episode is drawn as often as its checkpoint times its seed
        wc = pd.Series(rng.multinomial(len(cks), np.ones(len(cks)) / len(cks)), index=cks)
        ws = pd.Series(rng.multinomial(len(sds), np.ones(len(sds)) / len(sds)), index=sds)
        w = (wc[f.checkpoint_id].values * ws[f.env_seed].values).astype(int)
        g = f.loc[f.index.repeat(w)]
        vals.append(auroc(g.q, g.fs_n))
    out["two_way_bootstrap_nemotron"] = [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]
    return out


def main():
    manifest = pd.read_csv(ROOT / "results/hackathon/checkpoint_manifest.csv")
    df = load(REC, {k: JD / v for k, v in JUDGES.items()}, manifest)
    if "--select-v2c" in sys.argv:
        return select_v2c(df)
    df.to_csv(ROOT / "results/hackathon/explore/c1_coverage_shift_episodes.csv", index=False)
    res = {"label": "EXPLORATORY; pre-registered in docs/hackathon/explore/c1_coverage_shift_protocol.md",
           "n_episodes": int(len(df)), "n_checkpoints": int(df.checkpoint_id.nunique()),
           "episodes_by_stratum": df.groupby("stratum").size().to_dict(),
           "success_by_stratum": df.groupby("stratum").success.mean().round(3).to_dict(),
           "parse_rates": {k: float(df[f"{k}_says_success"].notna().mean()) for k in JUDGES}}
    res["nemotron"] = judge_block(df, "nemotron_says_success")
    res["cosmos"] = judge_block(df, "cosmos_says_success")
    if df.nemotron_v2c_says_success.notna().any():  # amendment A1, secondary 8 (failures + sampled successes only)
        res["nemotron_v2c_compartment"] = judge_block(df, "nemotron_v2c_says_success")
        res["nemotron_v2c_compartment"]["note"] = ("all failures + 40 sampled successes per stratum; precision is not "
                                                   "comparable to the other judges because successes are subsampled")
    t_fail = res["nemotron"].get("by_stratum", {}).get("T", {}).get("failures", 0)
    res["primary"] = {"judge": "nemotron", "auroc_q": res["nemotron"]["auroc_q"], "ci": res["nemotron"]["ci_checkpoint"],
                      "positive": bool(res["nemotron"]["ci_checkpoint"][0] > 0.5), "T_failures": t_fail,
                      "underpowered": bool(t_fail < 8)}
    res["post_hoc_after_verification"] = post_hoc(df)
    # secondary 6: the exploratory Experiment 2 eval episodes (Can only), analysed the same way, reported separately
    old = load(ROOT / "data/outputs/hackathon_rollouts_eval",
               {"nemotron": ROOT / "data/outputs/hackathon_cosmos/eval_nebius_v2.jsonl",
                "cosmos": ROOT / "data/outputs/hackathon_cosmos/eval_local_v6.jsonl"}, manifest)
    old = old[old.task == "can"]
    res["exploratory_eval330_can"] = {k: judge_block(old, f"{k}_says_success", strata=False) for k in ("nemotron", "cosmos")}
    out = ROOT / "results/hackathon/explore/c1_coverage_shift.json"
    out.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: res[k] for k in ("n_episodes", "episodes_by_stratum", "success_by_stratum", "parse_rates", "primary")},
                     indent=1, default=str))
    for k in ("nemotron", "cosmos", "nemotron_v2c_compartment"):
        if k in res:
            print(k, json.dumps(res[k].get("by_stratum"), default=str))


if __name__ == "__main__":
    sys.exit(main())
