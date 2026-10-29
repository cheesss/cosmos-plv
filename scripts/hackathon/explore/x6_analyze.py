#!/usr/bin/env python3
"""x6_visual_plan_critic, stage 3 (Environment B, EXPLORATORY / post-hoc).

Per checkpoint: VLM "plan reaches the object" fractions (Nemotron / Cosmos) and
purely geometric plan baselines (3D min distance plan->object, 2D pixel
distance, plan jitter), compared WITHIN RUN with training-time success.

No score is fitted.  Every metric has an a-priori direction.  The training-time
episodes are split: seeds 100000-100024 (half A, used only to pick which VLM
field/variant is called "primary") and 100025-100049 (half B, held-out report).

  PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/x6_analyze.py
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cosmos_plv.checkpoint_adapter import CKPT_RE, _read_logs  # noqa: E402
from cosmos_plv.ranking import bootstrap_ci, regret_at_q, spearman  # noqa: E402

NAME = "x6_visual_plan_critic"
OUTD = ROOT / f"data/outputs/hackathon_explore/{NAME}"
PLANS = OUTD / "plans"
POOL = ROOT / "data/outputs/hackathon_pool"
EVAL = ROOT / "data/outputs/hackathon_rollouts_eval"
RESULT = ROOT / f"results/hackathon/explore/{NAME}.json"
REACH_M = 0.05  # a-priori geometric "reaches" threshold (not tuned)
HEADS_M = 0.02  # a-priori: plan end at least 2 cm closer than the start
SEEDS_A = range(100000, 100025)
SEEDS_B = range(100025, 100050)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def discover(pool: Path):
    """Env-B twin of checkpoint_adapter.discover_checkpoints (that one needs omegaconf).
    Same checkpoint_id convention: <task>_s<training.seed>_e<epoch:04d>."""
    import yaml

    out = []
    for run_dir in sorted(p for p in pool.iterdir() if p.is_dir()):
        if not (run_dir / "checkpoints").is_dir():
            continue
        cfg = yaml.safe_load((run_dir / ".hydra" / "config.yaml").read_text())
        task, seed = str(cfg["task"]["task_name"]), int(cfg["training"]["seed"])
        logs = _read_logs(run_dir)
        for ckpt in sorted((run_dir / "checkpoints").glob("epoch=*.ckpt")):
            m = CKPT_RE.search(ckpt.name)
            if not m:
                continue
            ep = int(m.group("epoch"))
            row = logs.get(ep, {})
            out.append({"checkpoint_id": f"{task}_s{seed}_e{ep:04d}", "run_name": run_dir.name, "task": task,
                        "epoch": ep, "checkpoint_path": str(ckpt.relative_to(ROOT)),
                        "train_success_50": float(row.get("test/mean_score", float(m.group("score")))),
                        "val_loss": row.get("val_loss", np.nan)})
    return out


def training_outcomes():
    out = {}
    for run_dir in sorted(p for p in POOL.iterdir() if p.is_dir()):
        f = run_dir / "logs.json.txt"
        if not f.is_file():
            continue
        for line in f.read_text().splitlines():
            if "test/mean_score" not in line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            a = [r.get(f"test/sim_max_reward_{s}") for s in SEEDS_A]
            b = [r.get(f"test/sim_max_reward_{s}") for s in SEEDS_B]
            if any(v is None for v in a + b):
                continue
            out[(run_dir.name, int(r["epoch"]))] = (float(np.mean(a)), float(np.mean(b)))
    return out


def eval_outcomes():
    out = {}
    for s in sorted(EVAL.glob("*/summary.json")):
        try:
            eps = json.loads(s.read_text())["episodes"]
        except Exception:
            continue
        if eps:
            out[s.parent.name] = {"eval_success": float(np.mean([e["success"] for e in eps])),
                                  "eval_n": len(eps),
                                  "eval_wall_s_per_episode": float(np.mean([e["batch_wall_s"] for e in eps])) / max(1, len(eps))}
    return out


def plan_features():
    rows = []
    for cdir in sorted(p for p in PLANS.iterdir() if p.is_dir()):
        for npz in sorted(cdir.glob("seed*.npz")):
            d = np.load(npz)
            pos = d["action_pred"][:, :, :3].astype(float)
            obj, eef = d["obj0"].astype(float), d["eef0"].astype(float)
            dist1 = np.linalg.norm(pos[0] - obj, axis=1)
            d_start = float(np.linalg.norm(eef - obj))
            px = d["pix_plan1"].astype(float)
            d2 = np.linalg.norm(px - d["pix_obj0"].astype(float)[None], axis=1)
            jit = float(np.mean(np.linalg.norm(np.diff(pos[0], 2, axis=0), axis=1)))
            row = {"checkpoint_id": cdir.name, "env_seed": int(npz.stem[4:]), "n_chunks": int(pos.shape[0]),
                   "geo_min_dist_m": float(dist1.min()), "geo_end_dist_m": float(dist1[-1]),
                   "geo_start_dist_m": d_start, "geo_min_pix": float(d2.min()),
                   "geo_reach": float(dist1.min() < REACH_M), "geo_heads": float(dist1[-1] < d_start - HEADS_M),
                   "plan_jitter_m": jit, "frac_plan_in_frame": float(np.mean(d["inframe_plan1"])),
                   "load_s": float(d["load_s"]), "reset_render_s": float(d["reset_s"]),
                   "call1_s": float(d["call_s"][0])}
            if pos.shape[0] >= 3:
                dall = np.linalg.norm(pos.reshape(-1, 3) - obj, axis=1)
                ex = np.linalg.norm(d["eef_traj"].astype(float) - obj, axis=1)
                row.update({"geo3_min_dist_m": float(dall.min()), "exec24_min_dist_m": float(ex.min()),
                            "geo3_reach": float(dall.min() < REACH_M),
                            "geo3_heads": float(np.linalg.norm(pos[-1, -1] - obj) < d_start - HEADS_M)})
            rows.append(row)
    return pd.DataFrame(rows)


def vlm_rows():
    frames = []
    for backend in ("nebius", "local"):
        f = OUTD / f"vlm_{backend}_clean.jsonl"  # vlm_nebius.jsonl holds calls on corrupted frames (excluded)
        if not f.is_file():
            continue
        rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        df = pd.DataFrame(rows)
        if df.empty:
            continue
        df = df[df["error"].isna()].drop_duplicates(["checkpoint_id", "env_seed", "variant"], keep="last")
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def rank_eval(df, metric, higher_better, target, n_boot=2000, q=0.3, rng_seed=0):
    out = {}
    per_run = []
    for run, g in df.groupby("run_name", sort=True):
        g = g[np.isfinite(g[metric].astype(float)) & np.isfinite(g[target].astype(float))]
        if len(g) < 4:
            continue
        s = g[metric].values.astype(float) * (1 if higher_better else -1)
        y = g[target].values.astype(float)
        rho = spearman(s, y)
        ci = bootstrap_ci(lambda ix: spearman(s[ix], y[ix]), len(g), n_boot=n_boot)
        reg = regret_at_q(s, y, q, g["checkpoint_path"].values)
        # expected regret under random tie-breaking (the path tie-break favours early epochs)
        rng = np.random.default_rng(rng_seed)
        k = max(1, int(math.ceil(q * len(g))))
        rr = []
        for _ in range(500):
            jitter = rng.random(len(g)) * 1e-9
            order = np.lexsort((jitter, -s))
            rr.append(float(y.max() - y[order[:k]].max()))
        per_run.append({"run": run, "n": int(len(g)), "spearman": rho, "ci95": list(ci),
                        "regret@30_path_tiebreak": reg["regret"], "regret@30_random_tiebreak": float(np.mean(rr)),
                        "k@30": k, "n_unique_scores": int(len(np.unique(s)))})
    out["per_run"] = per_run
    if per_run:
        out["mean_spearman"] = float(np.nanmean([r["spearman"] for r in per_run]))
        out["mean_regret@30_path"] = float(np.nanmean([r["regret@30_path_tiebreak"] for r in per_run]))
        out["mean_regret@30_random"] = float(np.nanmean([r["regret@30_random_tiebreak"] for r in per_run]))
    return out


def auc(score, label):
    """P(score_pos > score_neg) (Mann-Whitney), ties count 1/2."""
    score, label = np.asarray(score, float), np.asarray(label, bool)
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def kappa(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    po = np.mean(a == b)
    pe = np.mean(a) * np.mean(b) + (1 - np.mean(a)) * (1 - np.mean(b))
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def main():
    ck = pd.DataFrame(discover(POOL))
    tro = training_outcomes()
    ck["train_success_A"] = [tro.get((r, e), (np.nan, np.nan))[0] for r, e in zip(ck.run_name, ck.epoch)]
    ck["train_success_B"] = [tro.get((r, e), (np.nan, np.nan))[1] for r, e in zip(ck.run_name, ck.epoch)]
    ev = eval_outcomes()
    for k in ("eval_success", "eval_n", "eval_wall_s_per_episode"):
        ck[k] = [ev.get(c, {}).get(k, np.nan) for c in ck.checkpoint_id]

    pf = plan_features()
    vl = vlm_rows()
    result = {"name": NAME, "label": "EXPLORATORY / post-hoc",
              "infrastructure_notes": [
                  "Env A reused one offscreen env per task: frames after the first checkpoint were corrupted or "
                  "vertically flipped although sim state was bit-identical; overlays were redrawn by x6_redraw.py "
                  "on the validated first frames (can_s1_e0000, square_s1_e0000) after asserting eef0/obj0/cam_P "
                  "equality. Analysed VLM rows: vlm_<backend>_clean.jsonl only.",
                  "25 Nebius calls on corrupted/flipped frames (can_s2_e0000, can_s1_e0020) are excluded "
                  "(kept in vlm_nebius.jsonl).",
                  "Cosmos3-Edge (local) rows are included only if vlm_local_clean.jsonl exists; in the 2026-10-29 "
                  "run the GPU was full and then the shared GPU lock was held by another exploration, so none exist.",
                  "Training-time success halves: A = seeds 100000-100024, B = 100025-100049 (held-out report)."],
              "reach_threshold_m": REACH_M,
              "heads_threshold_m": HEADS_M, "env_seeds": sorted(pf.env_seed.unique().tolist()) if len(pf) else []}

    agg = pf.groupby("checkpoint_id").agg(
        n_seeds=("env_seed", "count"), geo_min_dist_m=("geo_min_dist_m", "mean"),
        geo_end_dist_m=("geo_end_dist_m", "mean"), geo_min_pix=("geo_min_pix", "mean"),
        geo_reach_frac=("geo_reach", "mean"), geo_heads_frac=("geo_heads", "mean"),
        plan_jitter_m=("plan_jitter_m", "mean"), load_s=("load_s", "first"),
        reset_render_s=("reset_render_s", "mean"), call1_s=("call1_s", "mean")).reset_index()
    if "geo3_min_dist_m" in pf:
        agg = agg.merge(pf.groupby("checkpoint_id").agg(geo3_min_dist_m=("geo3_min_dist_m", "mean"),
                                                        exec24_min_dist_m=("exec24_min_dist_m", "mean"),
                                                        geo3_reach_frac=("geo3_reach", "mean")).reset_index(),
                        on="checkpoint_id", how="left")
    vlm_metrics = []
    image_agreement = {}
    if len(vl):
        vl = vl[vl["parse_status"] == "ok"].copy() if "parse_status" in vl else vl
        # test-retest: "<variant>_rep" rows are independent repeat calls on the same image
        retest = {}
        for (backend, variant), g in vl.groupby(["backend", "variant"]):
            if not variant.endswith("_rep"):
                continue
            base = vl[(vl.backend == backend) & (vl.variant == variant[:-4])]
            mm = g.merge(base, on=["checkpoint_id", "env_seed"], suffixes=("_rep", ""))
            retest[f"{backend}_{variant[:-4]}"] = {
                "n_pairs": int(len(mm)),
                "checkpoints": sorted(mm.checkpoint_id.unique().tolist()),
                "agree_heads": float(np.mean(mm.heads_to_object_rep == mm.heads_to_object)),
                "agree_reaches": float(np.mean(mm.reaches_object_rep == mm.reaches_object)),
                "agree_heads_wilson95": wilson(int((mm.heads_to_object_rep == mm.heads_to_object).sum()), len(mm)),
                "agree_reaches_wilson95": wilson(int((mm.reaches_object_rep == mm.reaches_object).sum()), len(mm))}
        result["test_retest"] = retest
        for (backend, variant), g in vl.groupby(["backend", "variant"]):
            if variant == "raw" or variant.endswith("_rep"):
                continue
            tag = f"{backend}_{variant}"
            a = g.groupby("checkpoint_id").agg(**{f"vlm_{tag}_heads": ("heads_to_object", "mean"),
                                                  f"vlm_{tag}_reaches": ("reaches_object", "mean"),
                                                  f"vlm_{tag}_n": ("env_seed", "count"),
                                                  f"vlm_{tag}_latency_s": ("latency_s", "mean")}).reset_index()
            agg = agg.merge(a, on="checkpoint_id", how="left")
            vlm_metrics += [f"vlm_{tag}_heads", f"vlm_{tag}_reaches"]
            m = g.merge(pf, on=["checkpoint_id", "env_seed"])
            if variant == "plan3" and "geo3_reach" in m:
                geo_r, geo_h = m["geo3_reach"], m["geo3_heads"]
            else:
                geo_r, geo_h = m["geo_reach"], m["geo_heads"]
            # variance decomposition among trained checkpoints (epoch >= 20): is the verdict about the
            # checkpoint or about the scene (env seed)?
            mt = m[m.checkpoint_id.str.slice(-4).astype(int) >= 20]
            eta = {}
            for col in ("heads_to_object", "reaches_object"):
                y = mt[col].astype(float)
                tot = float(((y - y.mean()) ** 2).sum())
                for fac, key in (("env_seed", "seed"), ("checkpoint_id", "checkpoint")):
                    grp = mt.groupby([fac, "task"])[col].transform("mean").astype(float) if fac == "env_seed" else mt.groupby(fac)[col].transform("mean").astype(float)
                    eta[f"eta2_{col}_by_{key}"] = float(((grp - y.mean()) ** 2).sum() / tot) if tot > 0 else float("nan")
            eta["n_images_epoch_ge20"] = int(len(mt))
            # expected eta^2 for a factor with no effect is about (groups-1)/(n-1)
            n_t = max(1, len(mt) - 1)
            eta["eta2_chance_seed"] = float((mt.groupby(["env_seed", "task"]).ngroups - 1) / n_t)
            eta["eta2_chance_checkpoint"] = float((mt["checkpoint_id"].nunique() - 1) / n_t)
            image_agreement[tag] = {**eta,
                "n_images": int(len(m)),
                "vlm_heads_rate": float(m["heads_to_object"].mean()), "vlm_reaches_rate": float(m["reaches_object"].mean()),
                "geo_heads_rate": float(geo_h.mean()), "geo_reach_rate": float(geo_r.mean()),
                "agree_heads": float(np.mean(m["heads_to_object"].astype(float) == geo_h)),
                "kappa_heads": kappa(m["heads_to_object"], geo_h),
                "agree_reaches": float(np.mean(m["reaches_object"].astype(float) == geo_r)),
                "kappa_reaches": kappa(m["reaches_object"], geo_r),
                "vlm_reaches_rate_when_geo_reach": float(m.loc[geo_r == 1, "reaches_object"].mean()) if (geo_r == 1).any() else None,
                "vlm_reaches_rate_when_geo_noreach": float(m.loc[geo_r == 0, "reaches_object"].mean()) if (geo_r == 0).any() else None,
                "vlm_heads_rate_epoch0": float(m[m.checkpoint_id.str.endswith("e0000")]["heads_to_object"].mean())
                if m.checkpoint_id.str.endswith("e0000").any() else None,
                "vlm_heads_rate_epoch_ge10": float(m[~m.checkpoint_id.str.endswith(("e0000", "e0005"))]["heads_to_object"].mean()),
                # what does the VLM verdict track?  AUC of (-distance) for predicting the VLM's own label
                "auc_vlm_reaches_from_neg_pix_dist": auc(-m["geo_min_pix"], m["reaches_object"]),
                "auc_vlm_reaches_from_neg_3d_dist": auc(-(m["geo3_min_dist_m"] if variant == "plan3" and "geo3_min_dist_m" in m else m["geo_min_dist_m"]), m["reaches_object"]),
                "auc_vlm_heads_from_3d_progress": auc(m["geo_start_dist_m"] - m["geo_end_dist_m"], m["heads_to_object"]),
            }
        for (backend, variant), g in vl.groupby(["backend", "variant"]):
            if variant == "raw":
                result.setdefault("raw_control", {})[backend] = [
                    {"checkpoint_id": r.checkpoint_id, "env_seed": int(r.env_seed),
                     "heads": bool(r.heads_to_object), "reaches": bool(r.reaches_object), "explanation": r.explanation}
                    for r in g.itertuples()]
    allv = vlm_rows()
    if len(allv):
        result["parse_status_counts"] = {f"{b}_{v}": g["parse_status"].value_counts().to_dict()
                                         for (b, v), g in allv.groupby(["backend", "variant"])}
    df = ck.merge(agg, on="checkpoint_id", how="inner")
    result["n_checkpoints"] = int(len(df))
    result["checkpoints_per_run"] = df.groupby("run_name").size().to_dict()

    df["random_baseline_const"] = 0.0  # all ties -> regret@30_random_tiebreak = expected regret of random ranking
    df["epoch_baseline"] = df["epoch"].astype(float)  # free baseline: "later checkpoint = better"
    metrics = [("random_baseline_const", True), ("epoch_baseline", True)] + [(m, True) for m in vlm_metrics] + [
        ("geo_reach_frac", True), ("geo_heads_frac", True), ("geo_min_dist_m", False), ("geo_end_dist_m", False),
        ("geo_min_pix", False), ("plan_jitter_m", False), ("val_loss", False)]
    if "geo3_min_dist_m" in df:
        metrics += [("geo3_min_dist_m", False), ("exec24_min_dist_m", False), ("geo3_reach_frac", True)]
    targets = ["train_success_B", "train_success_A", "train_success_50"]
    table = {}
    for m, hb in metrics:
        if m not in df or df[m].notna().sum() < 4:
            continue
        sub = df[df[m].notna()]
        table[m] = {"higher_is_better": hb, "n_checkpoints": int(len(sub))}
        for t in targets:
            table[m][t] = rank_eval(sub, m, hb, t)
        # selection regime: drop the trivially bad early checkpoints (declared before looking at results)
        late = sub[sub["epoch"] >= 20]
        table[m]["train_success_B_epoch_ge20"] = rank_eval(late, m, hb, "train_success_B")
        table[m]["train_success_50_epoch_ge20"] = rank_eval(late, m, hb, "train_success_50")
        # is the metric just a training-progress proxy?  oriented Spearman with epoch, epoch >= 20
        table[m]["spearman_with_epoch_epoch_ge20"] = [
            {"run": run, "spearman": spearman(g[m].values * (1 if hb else -1), g["epoch"].values), "n": int(len(g))}
            for run, g in late.groupby("run_name") if len(g) >= 4]
        # secondary: eval-rollout success (held-out env seeds 200000..), few checkpoints per run
        es = sub[sub["eval_success"].notna()]
        if len(es) >= 4:
            table[m]["eval_success"] = rank_eval(es, m, hb, "eval_success")
    result["ranking"] = table
    result["image_agreement_vlm_vs_geometry"] = image_agreement

    # VLM vs geometric baseline at checkpoint level (same checkpoints)
    vg = {}
    for m in vlm_metrics:
        for gm, hb in (("geo_reach_frac", True), ("geo_min_dist_m", False), ("geo_min_pix", False)):
            rhos = []
            for run, g in df.groupby("run_name"):
                g = g[g[m].notna()]
                if len(g) >= 4:
                    rhos.append({"run": run, "spearman": spearman(g[m], g[gm] * (1 if hb else -1)), "n": int(len(g))})
            vg[f"{m}__vs__{gm}"] = rhos
    result["vlm_vs_geometry_checkpoint_spearman"] = vg

    # cost per checkpoint
    cost = {"policy_load_s_mean": float(df["load_s"].mean()),
            "reset_render_s_per_seed_mean": float(df["reset_render_s"].mean()),
            "policy_call_s_per_seed_mean": float(df["call1_s"].mean()),
            "n_seeds": int(df["n_seeds"].max())}
    lat = {c: float(df[c].mean()) for c in df.columns if c.endswith("_latency_s") and df[c].notna().any()}
    cost["vlm_latency_s_per_call_mean"] = lat
    cost["visual_probe_s_per_checkpoint_excl_vlm"] = cost["policy_load_s_mean"] + cost["n_seeds"] * (
        cost["reset_render_s_per_seed_mean"] + cost["policy_call_s_per_seed_mean"])
    if df["eval_wall_s_per_episode"].notna().any():
        cost["closed_loop_eval_wall_s_per_episode_mean"] = float(df["eval_wall_s_per_episode"].mean())
        cost["closed_loop_eval_note"] = ("batch_wall_s / n_episodes from hackathon_rollouts_eval summaries "
                                         "(batched envs, so per-episode cost is amortised)")
    result["cost"] = cost

    # Wilson CIs for the per-image VLM positive rates by epoch bucket
    buckets = {}
    if len(vl):
        tmp = vl.merge(ck[["checkpoint_id", "epoch"]], on="checkpoint_id")
        tmp["bucket"] = np.where(tmp.epoch == 0, "epoch0", np.where(tmp.epoch < 20, "epoch5-15", "epoch>=20"))
        for (b, v, task, bk), g in tmp[~tmp.variant.str.endswith("_rep")].groupby(["backend", "variant", "task", "bucket"]):
            k = int(g["reaches_object"].sum()); n = int(len(g)); kh = int(g["heads_to_object"].sum())
            buckets[f"{b}|{v}|{task}|{bk}"] = {"n": n, "reaches_rate": k / n, "reaches_wilson95": wilson(k, n),
                                               "heads_rate": kh / n, "heads_wilson95": wilson(kh, n)}
    result["vlm_rates_by_epoch_bucket"] = buckets
    result["per_checkpoint"] = json.loads(df.drop(columns=["checkpoint_path"]).to_json(orient="records"))
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, indent=1, default=float))
    # console summary
    print(f"[x6-an] checkpoints {len(df)} per run {result['checkpoints_per_run']}")
    for m, v in table.items():
        tb = v["train_success_B"]
        print(f"{m:34s} n={v['n_checkpoints']:3d} rhoB={tb.get('mean_spearman', float('nan')):+.3f} "
              f"regB(path)={tb.get('mean_regret@30_path', float('nan')):.3f} regB(rand)={tb.get('mean_regret@30_random', float('nan')):.3f} "
              + " | ".join(f"{r['run'][8:22]} {r['spearman']:+.2f} [{r['ci95'][0]:+.2f},{r['ci95'][1]:+.2f}] u={r['n_unique_scores']}" for r in tb.get("per_run", []))
              + f" || e>=20 rhoB={v['train_success_B_epoch_ge20'].get('mean_spearman', float('nan')):+.3f}")
    for k, v in image_agreement.items():
        print(k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()})
    print("cost", json.dumps(cost))
    print(f"[x6-an] wrote {RESULT}")


if __name__ == "__main__":
    main()
