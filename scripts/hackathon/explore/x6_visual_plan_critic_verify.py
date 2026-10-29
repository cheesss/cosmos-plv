#!/usr/bin/env python3
"""Adversarial verification of x6_visual_plan_critic (Environment B, CPU only, no VLM calls).

Re-scores everything from the SAVED x6 artefacts (plan npz files, clean overlays, saved Nemotron rows,
Nebius evidence log). It never overwrites x6 outputs: x6_analyze.main() is re-run with RESULT redirected
into the verification scratch dir and diffed against the published JSON.

  PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/x6_visual_plan_critic_verify.py

Writes results/hackathon/explore/x6_visual_plan_critic_verification.json and two montage PNGs in
data/outputs/hackathon_explore/x6_verify/.
"""
from __future__ import annotations

import collections
import glob
import hashlib
import json
import math
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/hackathon/explore"))
os.chdir(ROOT)
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import x6_analyze as A  # noqa: E402
from cosmos_plv.ranking import _rankdata, bootstrap_ci, spearman  # noqa: E402

X6 = ROOT / "data/outputs/hackathon_explore/x6_visual_plan_critic"
SCR = ROOT / "data/outputs/hackathon_explore/x6_verify"
ORIG = ROOT / "results/hackathon/explore/x6_visual_plan_critic.json"
OUT = ROOT / "results/hackathon/explore/x6_visual_plan_critic_verification.json"
EVID = ROOT / "results/hackathon/explore/nebius_calls_explore.jsonl"
REF = {"can": "can_s1_e0000", "square": "square_s1_e0000"}


def r3(x):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), 3)


def diff_json(a, b, path=""):
    out = []
    if isinstance(a, dict):
        for k in set(a) | set(b):
            if k not in a or k not in b:
                out.append(path + "/" + k)
            else:
                out += diff_json(a[k], b[k], path + "/" + k)
    elif isinstance(a, list):
        if len(a) != len(b):
            return [path]
        for i, (u, v) in enumerate(zip(a, b)):
            out += diff_json(u, v, f"{path}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        try:
            if not ((math.isnan(a) and math.isnan(b)) or abs(a - b) < 1e-9):
                out.append(path)
        except TypeError:
            if a != b:
                out.append(path)
    elif a != b:
        out.append(path)
    return out


def eta2(y, groups):
    y = pd.Series(np.asarray(y, float))
    g = pd.Series(list(zip(*groups)) if isinstance(groups, tuple) else list(groups))
    m = y.groupby(g.values).transform("mean")
    tot = float(((y - y.mean()) ** 2).sum())
    return float(((m - y.mean()) ** 2).sum() / tot) if tot > 0 else float("nan")


def partial_spearman(x, y, z):
    rx, ry, rz = (_rankdata(np.asarray(v, float)) for v in (x, y, z))

    def res(a, b):
        B = np.c_[np.ones(len(b)), b]
        return a - B @ np.linalg.lstsq(B, a, rcond=None)[0]

    ex, ey = res(rx, rz), res(ry, rz)
    if ex.std() < 1e-12 or ey.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])


def sha(f):
    return hashlib.sha256(open(f, "rb").read()).hexdigest()


def main():
    SCR.mkdir(parents=True, exist_ok=True)
    res = {"name": "x6_visual_plan_critic_verification", "label": "verification of EXPLORATORY / post-hoc x6",
           "new_nebius_calls": 0, "gpu_used": False}

    # 1. reproduce x6_analyze without touching the published JSON
    A.RESULT = SCR / "x6_analyze_rerun.json"
    A.main()
    rerun = json.loads(A.RESULT.read_text())
    orig = json.loads(ORIG.read_text())
    d = diff_json(orig, rerun)
    res["reproduction"] = {"rerun_path": str(A.RESULT.relative_to(ROOT)), "n_differing_leaves": len(d),
                           "examples": d[:10]}

    pc = pd.DataFrame(rerun["per_checkpoint"])
    pf = A.plan_features()
    pf["epoch"] = pf.checkpoint_id.str[-4:].astype(int)
    pf["task"] = pf.checkpoint_id.str.split("_").str[0]
    pf["progress"] = pf.geo_start_dist_m - pf.geo_end_dist_m
    vl = A.vlm_rows()
    vl = vl[vl.parse_status == "ok"]
    p1 = vl[vl.variant == "plan1"].drop(columns=["task"]).merge(pf, on=["checkpoint_id", "env_seed"])

    # 2. per-task AUCs (pooled AUC could be a Simpson artefact: square heads is always True)
    per_task = {}
    for task, g in p1.groupby("task"):
        gt = g[g.epoch >= 20]
        per_task[task] = {
            "n": int(len(g)), "vlm_heads_rate": r3(g.heads_to_object.mean()),
            "vlm_reaches_rate": r3(g.reaches_object.mean()),
            "auc_reaches_from_neg_3d_min_dist": r3(A.auc(-g.geo_min_dist_m, g.reaches_object)),
            "auc_reaches_from_neg_pix_dist": r3(A.auc(-g.geo_min_pix, g.reaches_object)),
            "auc_heads_from_3d_progress": r3(A.auc(g.progress, g.heads_to_object)),
            "kappa_heads_vs_geo_heads": r3(A.kappa(g.heads_to_object, g.geo_heads)),
            "epoch_ge20_auc_reaches_from_neg_3d_min_dist": r3(A.auc(-gt.geo_min_dist_m, gt.reaches_object)),
            "epoch_ge20_auc_heads_from_3d_progress": r3(A.auc(gt.progress, gt.heads_to_object)),
            "epoch_ge20_reaches_true_max_min_dist_m": r3(gt[gt.reaches_object].geo_min_dist_m.max()),
            "epoch_ge20_reaches_false_min_min_dist_m": r3(gt[~gt.reaches_object].geo_min_dist_m.min()),
        }
    res["per_task_image_auc_plan1"] = per_task

    # 3. eta^2: pooled (as reported, task-confounded) vs within task; VLM verdicts and plan geometry
    mt = p1[p1.epoch >= 20]
    eta = {}
    for col in ("reaches_object", "heads_to_object", "geo_min_dist_m", "geo_min_pix", "progress"):
        e = {"pooled_by_task": r3(eta2(mt[col], mt.task)),
             "pooled_by_seed_x_task": r3(eta2(mt[col], (mt.env_seed, mt.task))),
             "pooled_by_checkpoint": r3(eta2(mt[col], mt.checkpoint_id))}
        for task, g in mt.groupby("task"):
            n = len(g)
            e[task] = {"n": int(n), "by_seed": r3(eta2(g[col], g.env_seed)),
                       "chance_seed": r3((g.env_seed.nunique() - 1) / (n - 1)),
                       "by_checkpoint": r3(eta2(g[col], g.checkpoint_id)),
                       "chance_checkpoint": r3((g.checkpoint_id.nunique() - 1) / (n - 1))}
        eta[col] = e
    res["eta2_epoch_ge20_plan1"] = eta
    res["per_seed_rates_epoch_ge20"] = {
        task: g.groupby("env_seed")[["heads_to_object", "reaches_object", "geo_start_dist_m", "progress",
                                     "geo_min_dist_m"]].mean().round(3).reset_index().to_dict(orient="records")
        for task, g in mt.groupby("task")}

    # 4. paired bootstrap of Spearman differences; partial Spearman given epoch
    rng = np.random.default_rng(0)
    comp = []
    for regime, sub in (("all", pc), ("epoch_ge20", pc[pc.epoch >= 20])):
        for run, g in sub.groupby("run_name"):
            y = g.train_success_B.values
            e = g.epoch.values.astype(float)
            vh, gh = g.vlm_nebius_plan1_heads.values, g.geo_heads_frac.values
            vr = g.vlm_nebius_plan1_reaches.values
            n = len(g)
            dg, de = [], []
            for _ in range(2000):
                ix = rng.integers(0, n, n)
                a, b, c = spearman(vh[ix], y[ix]), spearman(gh[ix], y[ix]), spearman(e[ix], y[ix])
                if np.isfinite(a) and np.isfinite(b):
                    dg.append(b - a)
                if np.isfinite(a) and np.isfinite(c):
                    de.append(c - a)
            pv = partial_spearman(vh, y, e)
            bs = [partial_spearman(vh[ix], y[ix], e[ix]) for ix in (rng.integers(0, n, n) for _ in range(2000))]
            bs = [v for v in bs if np.isfinite(v)]
            perm = [partial_spearman(rng.permutation(vh), y, e) for _ in range(5000)]
            perm = np.array([v for v in perm if np.isfinite(v)])
            comp.append({
                "regime": regime, "run": run, "n": int(n),
                "rho_vlm_heads": r3(spearman(vh, y)), "rho_geo_heads": r3(spearman(gh, y)),
                "rho_epoch": r3(spearman(e, y)),
                "diff_geo_minus_vlm_heads_ci95": [r3(np.percentile(dg, 2.5)), r3(np.percentile(dg, 97.5))] if dg else None,
                "diff_geo_minus_vlm_heads_valid_boot": len(dg),
                "diff_epoch_minus_vlm_heads_ci95": [r3(np.percentile(de, 2.5)), r3(np.percentile(de, 97.5))] if de else None,
                "partial_spearman_vlm_heads_success_given_epoch": r3(pv),
                "partial_ci95": [r3(np.percentile(bs, 2.5)), r3(np.percentile(bs, 97.5))] if bs else None,
                "partial_perm_p_one_sided": r3(np.mean(perm >= pv)) if np.isfinite(pv) and len(perm) else None,
                "partial_spearman_vlm_reaches_success_given_epoch": r3(partial_spearman(vr, y, e)),
            })
    res["paired_comparisons_halfB"] = comp

    # 5. plan3: executed steps and AUC vs true 3-chunk progress
    rows = []
    for r in vl[vl.variant == "plan3"].itertuples():
        dd = np.load(X6 / "plans" / r.checkpoint_id / f"seed{r.env_seed}.npz")
        pos = dd["action_pred"][:, :, :3].astype(float)
        ds = float(np.linalg.norm(dd["eef0"] - dd["obj0"]))
        rows.append({"task": r.task, "heads": r.heads_to_object, "n_exec": len(dd["eef_traj"]) - 1,
                     "plan_t": tuple(dd["plan_t"].tolist()),
                     "prog1": ds - np.linalg.norm(pos[0, -1] - dd["obj0"]),
                     "prog3": ds - np.linalg.norm(pos[-1, -1] - dd["obj0"])})
    p3 = pd.DataFrame(rows)
    res["plan3"] = {"n_images": int(len(p3)), "executed_sim_steps": sorted(p3.n_exec.unique().tolist()),
                    "plan_t": sorted(set(p3.plan_t)),
                    "auc_heads_vs_plan1_progress_as_reported": r3(A.auc(p3.prog1, p3.heads)),
                    "auc_heads_vs_3chunk_progress": r3(A.auc(p3.prog3, p3.heads))}

    # 6. projection + initial-state identity (precondition of x6_redraw)
    mx = 0.0
    for f in glob.glob(str(X6 / "plans/*/seed*.npz")):
        dd = np.load(f)
        pos = dd["action_pred"][0, :, :3].astype(np.float64)
        pix = (dd["cam_P"] @ np.c_[pos, np.ones(len(pos))].T).T
        rc = np.stack([pix[:, 1] / pix[:, 2], pix[:, 0] / pix[:, 2]], 1)
        mx = max(mx, float(np.abs(rc - dd["pix_plan1"]).max()))
    ident = {}
    for task in ("can", "square"):
        for s in range(210000, 210005):
            ds_ = [np.load(f) for f in sorted(glob.glob(str(X6 / f"plans/{task}_*/seed{s}.npz")))]
            ident[f"{task}_{s}"] = {"n_ckpts": len(ds_), "max_dev_eef0_obj0_camP": max(
                float(np.abs(x[k] - ds_[0][k]).max()) for x in ds_ for k in ("eef0", "obj0", "cam_P")),
                "start_dist_m": r3(np.linalg.norm(ds_[0]["eef0"] - ds_[0]["obj0"]))}
    res["projection_recompute_max_abs_px"] = mx
    res["initial_state_identity"] = ident

    # 7. evidence: call count, errors, image hash of every analysed row == current clean image
    ev = [json.loads(l) for l in EVID.read_text().splitlines() if '"explore/x6_visual_plan_critic' in l]
    by = collections.defaultdict(list)
    for e in ev:
        by[e["tag"]].append(e["params"].get("image_sha256_0"))
    cl = [json.loads(l) for l in (X6 / "vlm_nebius_clean.jsonl").read_text().splitlines() if l.strip()]
    ok = bad = 0
    for r in cl:
        tag = f"explore/x6_visual_plan_critic/{r['variant']}/{r['checkpoint_id']}/seed{r['env_seed']}"
        vimg = r["variant"][:-4] if r["variant"].endswith("_rep") else r["variant"]
        img = (X6 / "plans" / REF[r["task"]] / f"seed{r['env_seed']}_raw.png" if vimg == "raw"
               else X6 / "plans_clean" / r["checkpoint_id"] / f"seed{r['env_seed']}_{vimg}.png")
        if sha(img) in by[tag]:
            ok += 1
        else:
            bad += 1
    base = {(r["checkpoint_id"], r["env_seed"]): r for r in cl if r["variant"] == "plan1"}
    rt = {}
    for task in ("can", "square"):
        rr = [r for r in cl if r["variant"] == "plan1_rep" and r["task"] == task]
        kh = sum(r["heads_to_object"] == base[(r["checkpoint_id"], r["env_seed"])]["heads_to_object"] for r in rr)
        kr = sum(r["reaches_object"] == base[(r["checkpoint_id"], r["env_seed"])]["reaches_object"] for r in rr)
        rt[task] = {"n": len(rr), "heads_agree": int(kh), "heads_wilson95": [r3(v) for v in A.wilson(kh, len(rr))],
                    "reaches_agree": int(kr),
                    "base_heads_true": int(sum(base[(r["checkpoint_id"], r["env_seed"])]["heads_to_object"] for r in rr))}
    res["evidence"] = {"x6_calls_logged": len(ev), "x6_call_errors": sum(1 for e in ev if e.get("error")),
                       "calls_by_variant": dict(collections.Counter(e["tag"].split("/")[2] for e in ev)),
                       "analysed_rows": len(cl), "analysed_rows_hash_match_clean_image": ok,
                       "analysed_rows_hash_mismatch": bad,
                       "retest_by_task": rt}

    # 8. geometric-heads threshold sensitivity (a-priori 2 cm)
    sens = {}
    for hm in (0.0, 0.01, 0.02, 0.05, 0.08):
        pf["gh"] = (pf.geo_end_dist_m < pf.geo_start_dist_m - hm).astype(float)
        m = pc.merge(pf.groupby("checkpoint_id").gh.mean().rename("gh").reset_index(), on="checkpoint_id")
        sens[str(hm)] = {"rhoB_per_run": [r3(spearman(g.gh, g.train_success_B)) for _, g in m.groupby("run_name")],
                         "epoch0_values": m[m.epoch == 0].gh.round(2).tolist(),
                         "min_over_epoch_ge20": r3(m[m.epoch >= 20].gh.min())}
    res["geo_heads_threshold_sensitivity"] = sens
    res["geo_heads_epoch5_15"] = pc[(pc.epoch > 0) & (pc.epoch < 20)][["checkpoint_id", "geo_heads_frac"]].to_dict(orient="records")

    # 9. epoch baseline on the CURRENT (grown) pool
    ck = pd.DataFrame(A.discover(A.POOL))
    tro = A.training_outcomes()
    ck["B"] = [tro.get((r, e), (np.nan, np.nan))[1] for r, e in zip(ck.run_name, ck.epoch)]
    cur = {"n_checkpoints": int(len(ck))}
    for run, g in ck.groupby("run_name"):
        for lab, sub in (("all", g), ("epoch_ge20", g[g.epoch >= 20])):
            s, y = sub.epoch.values.astype(float), sub.B.values
            m = np.isfinite(y)
            cur[f"{run}|{lab}"] = {"n": int(m.sum()), "max_epoch": int(g.epoch.max()),
                                   "rho": r3(spearman(s[m], y[m])),
                                   "ci95": [r3(v) for v in bootstrap_ci(lambda ix: spearman(s[m][ix], y[m][ix]), int(m.sum()))]}
    res["epoch_baseline_current_pool"] = cur

    # 10. cost denominator: batch_wall_s is per batch of all episodes in a summary?
    uniq = []
    for s in sorted(glob.glob(str(A.EVAL / "*/summary.json"))):
        eps = json.loads(Path(s).read_text())["episodes"]
        uniq.append(len(set(e["batch_wall_s"] for e in eps)))
    res["eval_summaries_distinct_batch_wall_values"] = dict(collections.Counter(uniq))

    # 11. montages for the visual check
    tiles = []
    for task, trained in (("can", "can_s2_e0080"), ("square", "square_s1_e0120")):
        row = []
        for s in range(210000, 210005):
            dd = np.load(X6 / "plans" / REF[task] / f"seed{s}.npz")
            im = Image.open(X6 / "plans" / REF[task] / f"seed{s}_raw.png").convert("RGB")
            dr = ImageDraw.Draw(im)
            for (rr, cc), col in ((dd["pix_eef0"], (0, 255, 0)), (dd["pix_obj0"], (255, 0, 255))):
                dr.line([(cc - 12, rr), (cc + 12, rr)], fill=col, width=3)
                dr.line([(cc, rr - 12), (cc, rr + 12)], fill=col, width=3)
            row.append(im.resize((224, 224)))
        tiles.append(row)
        tiles.append([Image.open(X6 / "plans_clean" / trained / f"seed{s}_plan1.png").convert("RGB").resize((224, 224))
                      for s in range(210000, 210005)])
    W = Image.new("RGB", (224 * 5, 224 * 4), "white")
    for i, row in enumerate(tiles):
        for j, t in enumerate(row):
            W.paste(t, (224 * j, 224 * i))
    W.save(SCR / "ref_frames_markers.png")
    row = [Image.open(X6 / "plans" / c / "seed210000_plan1.png").convert("RGB").resize((224, 224))
           for c in ("can_s2_e0000", "can_s1_e0020", "can_s2_e0080", "square_s1_e0120", "can_s1_e0095")]
    W2 = Image.new("RGB", (224 * 5, 224), "white")
    for j, t in enumerate(row):
        W2.paste(t, (224 * j, 0))
    W2.save(SCR / "orig_frames.png")
    res["montages"] = [str((SCR / "ref_frames_markers.png").relative_to(ROOT)),
                       str((SCR / "orig_frames.png").relative_to(ROOT))]

    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("reproduction", "plan3", "evidence", "projection_recompute_max_abs_px")},
                     indent=1, default=float))
    print(f"[x6-verify] wrote {OUT}")


if __name__ == "__main__":
    main()
