#!/usr/bin/env python3
"""Build results/hackathon/{experiment_manifest,execution_summary,cost_latency,budget_curve}.json
from the files produced by the pipeline (Environment A or B; numpy + pandas only).

Every number is read from an existing artifact; missing inputs are reported as
null with a reason, never filled in.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "hackathon"))

from cosmos_plv.provenance import environment_info, git_commit, utc_now, write_json  # noqa: E402
from cosmos_plv.ranking import regret_at_q, spearman  # noqa: E402
from nebius_evidence_summary import summarize as nebius_summary  # noqa: E402

RES = ROOT / "results/hackathon"
LOGS = ROOT / "data/outputs/hackathon_logs"
MET = ROOT / "data/outputs/hackathon_metrics"


def time_log(name):
    """Sum of wall-clock seconds over all metric batches (data/outputs/hackathon_metrics/*/<name>.log)."""
    vals = []
    for p in MET.glob(f"*/{name}.log"):
        m = re.findall(rf"{name} ([0-9.]+) s", p.read_text(errors="ignore"))
        if m:
            vals.append(float(m[-1]))
    return sum(vals) if vals else None


def pct(a):
    a = np.asarray([x for x in a if x is not None and np.isfinite(x)], float)
    if not len(a):
        return None
    return {"mean": float(a.mean()), "p10": float(np.percentile(a, 10)), "p50": float(np.median(a)),
            "p90": float(np.percentile(a, 90)), "n": int(len(a))}


def budget_curve(screen: pd.DataFrame, eps: pd.DataFrame, diag: pd.DataFrame, n_rep=200, seed=0):
    """Ranking quality vs rollout budget K per checkpoint (subsampling the 10 recorded episodes)."""
    rng = np.random.default_rng(seed)
    out = []
    pred = {}
    for b in sorted(diag.backend.unique()) if len(diag) else []:
        d = diag[(diag.backend == b) & diag.scored.fillna(False).astype(bool)]
        pred[b] = {(r.checkpoint_id, r.env_seed): float(bool(r.cosmos_task_success)) for r in d.itertuples()}
    sources = {"sim_instrumented": {(r.checkpoint_id, r.env_seed): float(r.success) for r in eps.itertuples()}}
    sources.update({f"vlm_{b}": v for b, v in pred.items()})
    for K in (1, 2, 4, 6, 8, 10):
        for name, lab in sources.items():
            rhos, regs = [], []
            for _ in range(n_rep):
                per_run_rho, per_run_reg = [], []
                for run, g in screen.groupby("run_id"):
                    est = []
                    for ck in g.checkpoint_id:
                        seeds = [s for (c, s) in lab if c == ck]
                        if len(seeds) < K:
                            est.append(np.nan); continue
                        pick = rng.choice(seeds, size=K, replace=False)
                        est.append(np.mean([lab[(ck, s)] for s in pick]))
                    est = np.asarray(est)
                    ok = np.isfinite(est)
                    if ok.sum() < 3:
                        continue
                    jit = est + 1e-6 * rng.random(len(est))  # random tie-breaking
                    per_run_rho.append(spearman(jit[ok], g.success.values[ok]))
                    per_run_reg.append(regret_at_q(jit[ok], g.success.values[ok], 1e-9)["regret"])
                if per_run_rho:
                    rhos.append(np.nanmean(per_run_rho)); regs.append(np.nanmean(per_run_reg))
            if rhos:
                out.append({"K": K, "source": name, "spearman_mean": float(np.mean(rhos)),
                            "spearman_p10": float(np.percentile(rhos, 10)), "spearman_p90": float(np.percentile(rhos, 90)),
                            "pick_regret_mean": float(np.mean(regs))})
    return out


def main():
    base = pd.read_csv(RES / "baseline_metrics.csv") if (RES / "baseline_metrics.csv").exists() else pd.DataFrame()
    screen = pd.read_csv(RES / "hybrid_screening.csv") if (RES / "hybrid_screening.csv").exists() else pd.DataFrame()
    diag = pd.read_csv(RES / "cosmos_diagnostics.csv") if (RES / "cosmos_diagnostics.csv").exists() else pd.DataFrame()
    eps_rows, wall = [], []
    for s in sorted((ROOT / "data/outputs/hackathon_rollouts_eval").glob("*/summary.json")):
        js = json.loads(s.read_text())
        eps_rows += js["episodes"]
        wall.append(js["episodes"][0]["batch_wall_s"] + js["episodes"][0].get("load_s", 0))
    eps = pd.DataFrame(eps_rows)
    manifest = json.loads((RES / "checkpoint_manifest.json").read_text()) if (RES / "checkpoint_manifest.json").exists() else {}
    neb = nebius_summary(RES / "nebius_calls.jsonl") if (RES / "nebius_calls.jsonl").exists() else None

    n_ck = max(1, len(base))
    per_ck = {
        "Action L1 + PLV-1 (original script, incl. f_psi training)": (time_log("plv1") or np.nan) / n_ck,
        "Open-loop smoothness (original script)": (time_log("open_loop") or np.nan) / n_ck,
        "Replanning, 2 horizons x 4 seeds (original script)": (time_log("chunk") or np.nan) / n_ck,
        "10 recorded rollouts with video (robosuite)": float(np.mean(wall)) if wall else np.nan,
    }
    lat = {}
    if len(diag):
        loc = diag[diag.backend == "local"].latency_s
        nb = diag[diag.backend == "nebius"].latency_s
        if len(loc):
            per_ck["Cosmos3-Edge last-frame judge, 10 episodes (local)"] = float(loc.mean() * 10)
            lat["Cosmos3-Edge last frame (local GPU)"] = pct(loc.values)
        v7p = ROOT / "data/outputs/hackathon_cosmos/eval_local_v7.jsonl"
        if v7p.exists():
            v7lat = [json.loads(l)["latency_s"] for l in open(v7p)]
            per_ck["Cosmos3-Edge 8-keyframe track, 10 episodes (local)"] = float(np.mean(v7lat) * 10)
            lat["Cosmos3-Edge 8 keyframes (local GPU)"] = pct(v7lat)
        if len(nb):
            per_ck["Nemotron VL on Nebius, 10 episodes"] = float(nb.mean() * 10)
            lat["Nemotron-3.5-Super-VL (video, Nebius)"] = pct(nb.values)
    per_ck = {k: (None if (v is None or not np.isfinite(v)) else round(float(v), 1)) for k, v in per_ck.items()}
    cost = {
        "generated_utc": utc_now(),
        "hardware": "1x RTX 3090 24 GB shared by up to 3 training runs and 3 rollout recorders (timings are upper bounds)",
        "per_checkpoint_seconds": per_ck,
        "per_episode_latency_s": lat,
        "nebius_tokens": neb["tokens"] if neb else None,
        "nebius_usd_estimate": neb["usd_estimate_at_catalog_price"] if neb else None,
        "notes": "cheap-metric times are whole-pool wall-clock divided by the number of checkpoints, "
                 "the three scripts ran concurrently on the same GPU.",
    }
    write_json(RES / "cost_latency.json", cost)

    if len(screen) and len(eps):
        write_json(RES / "budget_curve.json", {"generated_utc": utc_now(), "rows": budget_curve(screen, eps, diag)})

    exp = {
        "generated_utc": utc_now(), "git_commit": git_commit(),
        "experiments": {
            "E1_baseline": {"checkpoints": int(len(base)), "runs": sorted(base.run_id.unique().tolist()) if len(base) else [],
                            "target": "training-time success, 50 rollouts per checkpoint, env seeds 100000-100049",
                            "metrics": ["val_action_l1", "open_loop_smoothness", "plv1_smoothness", "boundary_jump_h16",
                                        "replan_overlap_h16", "boundary_jump_h10", "replan_overlap_h10", "val_loss_logged"]},
            "E2_visual_failure": {"episodes": int(len(eps)), "env_seeds": "200000-200009",
                                  "diagnostic_rows": int(len(diag)),
                                  "backends": sorted(diag.backend.unique().tolist()) if len(diag) else []},
            "E3_hybrid": {"checkpoints": int(len(screen)), "q": 0.3, "K": 10},
            "E4_action_conditioned": {"status": "blocked", "reason": "no faithful robosuite -> Cosmos action adapter"},
            "dev_prompt_selection": {"episodes": 36, "env_seeds": "300000-300011, 300100-300111, 300200-300211"},
        },
        "models": {"local": {"id": "nvidia/Cosmos3-Edge", "revision": "344d602b128d1bbdacb43b08d0a3626f46343e29"},
                   "nebius": {"id": "nvidia/Nemotron-3.5-Super-VL", "endpoint": "https://api.tokenfactory.nebius.com/v1/"}},
        "dataset": json.loads((ROOT / "configs/hackathon/dataset_manifest.json").read_text()),
        "checkpoint_pool": {k: manifest.get(k) for k in ("n_checkpoints", "runs", "validation_problems")},
        "diffusion_sampling_seed": "per recorder batch: 0 + 7919 * batch_index; metric scripts: policy seed 0, sampling seeds 0-3",
    }
    write_json(RES / "experiment_manifest.json", exp)
    ex = {
        "generated_utc": utc_now(), "git_commit": git_commit(), "environment": environment_info(),
        "nebius": neb,
        "artifacts": sorted(str(p.relative_to(ROOT)) for p in RES.glob("*") if p.is_file()),
        "media": sorted(str(p.relative_to(ROOT)) for p in (ROOT / "media/hackathon").glob("*") if p.is_file()),
        "commands": [json.loads(l) for l in (RES / "commands_log.jsonl").read_text().splitlines()] if (RES / "commands_log.jsonl").exists() else None,
    }
    write_json(RES / "execution_summary.json", ex)
    print(json.dumps(cost, indent=2))


if __name__ == "__main__":
    main()
