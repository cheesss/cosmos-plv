#!/usr/bin/env python3
"""Generate the hackathon figures from REAL result files only (Environment B).

media/hackathon/architecture.png          Figure A (diagram, no data)
media/hackathon/checkpoint_comparison.png Figure B (baseline metrics vs measured success)
media/hackathon/robot_execution.png       Figure C (synchronized frames + state from real rollouts)
media/hackathon/cosmos_example.png        Figure D (actual model input + actual output)
media/hackathon/cost_latency.png          Figure E (measured cost / latency)
A figure whose inputs are missing is skipped (and reported), never faked.
"""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results/hackathon"
OUT = ROOT / "media/hackathon"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
C = {"cheap": "#4C72B0", "sim": "#55A868", "cosmos": "#76B900", "nebius": "#7E57C2", "grey": "#8C8C8C", "bad": "#C44E52"}


def fig_architecture():
    fig, ax = plt.subplots(figsize=(16, 7.4))
    ax.set_xlim(0, 16); ax.set_ylim(0, 7.6); ax.axis("off")

    def box(x, y, w, h, title, body, color, tsize=12):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", fc=color, ec="none", alpha=0.18))
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", fc="none", ec=color, lw=2))
        ax.text(x + w / 2, y + h - 0.25, title, ha="center", va="top", fontsize=tsize, weight="bold")
        ax.text(x + w / 2, y + h - 0.8, body, ha="center", va="top", fontsize=9.5, linespacing=1.4)

    def arrow(x0, y0, x1, y1, label="", rad=0.0, color="#333"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=18, lw=1.8, color=color,
                                     connectionstyle=f"arc3,rad={rad}"))
        if label:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.15, label, ha="center", fontsize=9, color=color)

    box(0.1, 3.4, 2.4, 2.6, "Checkpoint", "Diffusion Policy\n(robomimic lowdim,\nabsolute OSC actions)\nH=16, n_obs=2, n_act=8", C["grey"])
    # cheap box kept narrow enough that the "top-q" label fits in the gap before the rollouts box
    box(2.95, 3.4, 2.55, 2.6, "Cheap screening", "Action L1 (L1)\nopen-loop smoothness (L2)\nPLV-1 via learned f_psi (L2)\nreplan overlap / jump (L3)\noriginal PLV scripts", C["cheap"])
    box(6.3, 3.4, 2.8, 2.6, "Real rollouts", "robosuite 1.5 / MuJoCo\nagentview video, 10 fps\nstate trace + every\npredicted action chunk\nverified sim labels", C["sim"])
    box(9.55, 4.35, 3.35, 1.9, "NVIDIA Cosmos3-Edge", "local RTX 3090 (bf16)\nper-keyframe object state\n-> rule: grasp / lift / drop", C["cosmos"])
    box(9.55, 1.75, 3.35, 1.9, "Nemotron-3.5-Super-VL", "NVIDIA VLM, served by\nNebius Token Factory\nepisode video -> failure JSON\n(second opinion)", C["nebius"])
    box(13.35, 2.6, 2.5, 3.4, "Report & dashboard", "validated JSON\nhybrid screening\n\"evaluate next, why\"\nprovenance:\nckpt sha, seed,\nmodel revision", C["grey"], tsize=11)
    arrow(2.6, 4.7, 2.9, 4.7)
    arrow(5.62, 4.7, 6.25, 4.7)
    ax.text(5.9, 4.85, "top-q", ha="center", fontsize=9, color="#333")  # centred in the gap, clear of both boxes
    arrow(9.2, 5.0, 9.5, 5.3)
    arrow(9.2, 4.2, 9.5, 2.9)
    arrow(13.0, 5.3, 13.3, 4.9)
    arrow(13.0, 2.7, 13.3, 3.3)
    arrow(4.22, 6.15, 14.6, 6.15, rad=-0.12, color=C["cheap"])
    ax.text(9.5, 7.2, "cheap metrics for every checkpoint", ha="center", fontsize=9, color=C["cheap"])
    ax.add_patch(FancyBboxPatch((9.55, 1.15), 3.35, 0.38, boxstyle="round,pad=0.03", fc=C["nebius"], ec="none"))
    ax.text(11.22, 1.34, "runs on Nebius Token Factory", ha="center", va="center", color="white", fontsize=9.5, weight="bold")
    ax.text(0.1, 0.65, "Environment A (py3.10, torch 2.1): training, PLV metrics, robosuite rollouts.   "
            "Environment B (py3.11, torch 2.8): Cosmos, Nebius client, dashboard.   Exchange: MP4 + NPZ + JSON.",
            fontsize=9.5, color="#444")
    ax.text(0.1, 0.2, "Mode B: exploratory — Cosmos3-Edge forward dynamics through the trained Franka action slot follows "
            "commanded gripper motion (Experiment 4b); not used for ranking",
            fontsize=9.5, color="#444")
    ax.set_title("Cosmos-PLV architecture", fontsize=16, weight="bold", loc="left")
    fig.savefig(OUT / "architecture.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return "architecture.png"


def fig_checkpoint_comparison():
    p = RES / "baseline_metrics.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    scr = pd.read_csv(RES / "hybrid_screening.csv") if (RES / "hybrid_screening.csv").exists() else pd.DataFrame()
    runs = sorted(df.run_id.unique())
    metrics = [("val_action_l1", "Action L1"), ("open_loop_smoothness", "Open-loop smoothness"), ("plv1_smoothness", "PLV-1 smoothness")]
    fig, axes = plt.subplots(len(metrics) + 1, len(runs), figsize=(5.4 * len(runs), 11), squeeze=False, sharex="col")
    for j, run in enumerate(runs):
        g = df[df.run_id == run].sort_values("epoch")
        ax = axes[0, j]
        ax.plot(g.epoch, g.success, "-o", ms=3, color=C["sim"], label="50-rollout success (training-time)")
        if len(scr):
            s = scr[scr.run_id == run].sort_values("epoch")
            ax.plot(s.epoch, s.k_sim_success, "s", ms=5, mfc="none", color=C["sim"], label="10-rollout success (eval seeds)")
            for col, color, lab in (("vlm_success_local", C["cosmos"], "Cosmos3-Edge judged (10)"),
                                    ("vlm_success_nebius", C["nebius"], "Nemotron VL judged (10)")):
                if col in s:
                    ax.plot(s.epoch, s[col], "^", ms=5, color=color, label=lab)
        ax.set_ylim(-0.05, 1.05); ax.set_title(run, weight="bold"); ax.set_ylabel("success")
        if j == 0:
            ax.legend(fontsize=8, loc="lower right")
        for i, (col, lab) in enumerate(metrics, start=1):
            ax = axes[i, j]
            if col in g and g[col].notna().any():
                ax.plot(g.epoch, g[col], "-", color=C["cheap"])
                ax.set_yscale("log")
            ax.set_ylabel(lab)
        axes[-1, j].set_xlabel("training epoch")
    fig.suptitle("Figure B - cheap offline metrics (lower = better) vs measured simulator success, per training run", x=0.01, ha="left", weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "checkpoint_comparison.png", dpi=130)
    plt.close(fig)
    return "checkpoint_comparison.png"


def _episodes():
    rows = []
    for s in sorted((ROOT / "data/outputs/hackathon_rollouts_eval").glob("*/summary.json")):
        rows += json.loads(s.read_text())["episodes"]
    return pd.DataFrame(rows)


def _frames(video):
    import imageio.v2 as imageio
    return [f for f in imageio.get_reader(str(ROOT / video))]


def fig_robot_execution():
    ep = _episodes()
    if ep.empty:
        return None
    picks = []
    for ft in ("none", "dropped", "no_grasp"):
        c = ep[(ep.failure_type == ft) & ep.video.notna()]
        if len(c):
            picks.append(c.sort_values(["checkpoint_id", "env_seed"]).iloc[len(c) // 2])
    fig, axes = plt.subplots(len(picks), 6, figsize=(16, 3.0 * len(picks)), squeeze=False)
    for i, e in enumerate(picks):
        fr = _frames(e.video)
        d = np.load(ROOT / e.episode_npz)
        idx = [int(round(x)) for x in np.linspace(0, len(fr) - 1, 6)]
        for j, k in enumerate(idx):
            ax = axes[i, j]
            ax.imshow(fr[k]); ax.set_xticks([]); ax.set_yticks([])
            step = min(k * 2, len(d["object_pos"]) - 1)
            h = d["object_pos"][step, 2] - d["object_pos"][:11, 2].min()  # settled reference height
            ax.set_title(f"t={k / e.video_fps:.1f}s  obj dz={h * 100:+.0f}cm  grasp={'Y' if d['grasped'][step] else 'n'}", fontsize=8)
            if j == 0:
                sim = "success" if e.failure_type == "none" else e.failure_type
                ax.set_ylabel(f"{e.checkpoint_id}\nseed {e.env_seed}\nsim: {sim}", fontsize=9)
    fig.suptitle("Figure C - real robosuite rollouts: agentview frames synchronized with privileged state (object height above its settled height, finger-pad grasp)",
                 x=0.01, ha="left", weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "robot_execution.png", dpi=120)
    plt.close(fig)
    return "robot_execution.png"


# Simulator check of the per-keyframe Cosmos labels in Figure D, robosuite default layout:
# pegs_arena.xml peg1 (square) / peg2 (round); PickPlace bin1 (source) / bin2 (target), 0.39 x 0.49 m.
PEGS_XY = np.array([(0.23, 0.10), (0.23, -0.10)])
BINS_XY = {"left_bin": (0.1, -0.25), "right_bin": (0.1, 0.28)}
BIN_HALF = (0.195 + 0.02, 0.245 + 0.02)  # 2 cm margin on the bin walls
FLOOR_Z = 0.75  # table / bin tops are at 0.80-0.82 m; below this the object lies on the floor
REGION_TEXT = {"floor": "on the floor", "on_peg": "at a peg", "on_table": "on the table", "left_bin": "in the left bin",
               "right_bin": "in the right bin", "outside_bins": "outside both bins"}


def _keyframe_error(task, z, k, frame):
    """Why the Cosmos label of video frame k contradicts the simulator, or None if it does not.

    Video frame k = sim step 2k. A label is called wrong only if it is contradicted at every sim step
    within +-1 video frame, so frame/step alignment alone cannot make an error.
    held = finger-pad contact and object >= 2 cm above its settled height (as in explore/x5_analyze.py).
    'not_visible' is wrong whenever the object is not on the floor: agentview frames the whole table and
    both bins (occlusion by the arm is not modelled).
    """
    if frame.get("parse_status") != "ok":
        return None
    pos, grasp = z["object_pos"], z["grasped"].astype(bool)
    T = len(pos) - 1
    rest = pos[min(10, T), 2]
    loc = frame.get("p_object_location")
    held = bool(frame.get("p_held_by_gripper")) or loc == "in_gripper"

    def region(p):
        if p[2] < FLOOR_Z:
            return "floor"
        if task == "square":
            return "on_peg" if np.hypot(*(PEGS_XY - p[:2]).T).min() <= 0.06 else "on_table"
        for name, (cx, cy) in BINS_XY.items():
            if abs(p[0] - cx) <= BIN_HALF[0] and abs(p[1] - cy) <= BIN_HALF[1]:
                return name
        return "outside_bins"

    def why(s):
        p = pos[s]
        if loc == "not_visible":
            return "object is in view" if region(p) != "floor" else None
        if held and not grasp[s]:
            return "not grasped"
        if not held and grasp[s] and p[2] - rest >= 0.02:
            return "object is held"
        ok = {"floor": ("off_table", "outside_bins"), "on_peg": ("on_peg", "on_table")}.get(region(p), (region(p),))
        return None if loc in ok + ("in_gripper",) else f"object is {REGION_TEXT[region(p)]}"
    steps = range(max(0, 2 * k - 2), min(T, 2 * k + 2) + 1)
    return why(min(2 * k, T)) if all(why(s) for s in steps) else None


def fig_cosmos_example():
    p = RES / "cosmos_diagnostics.csv"
    v7p = ROOT / "data/outputs/hackathon_cosmos/eval_local_v7.jsonl"
    if not p.exists() or not v7p.exists():
        return None
    d = pd.read_csv(p)
    neb = d[(d.backend == "nebius") & d.scored.fillna(False).astype(bool)]
    v7 = {(r["checkpoint_id"], r["env_seed"]): r for r in map(json.loads, open(v7p)) if r.get("track")}
    ep = _episodes().set_index(["checkpoint_id", "env_seed"])
    cases = []
    for want in ("none", "dropped", "no_grasp"):
        c = [r for r in v7.values() if r["truth_failure_type"] == want and r.get("scored")]
        c = sorted(c, key=lambda r: (r["checkpoint_id"], r["env_seed"]))
        if c:
            cases.append(c[len(c) // 2])
        if len(cases) == 2:
            break
    if not cases:
        return None
    import imageio.v2 as imageio
    fig = plt.figure(figsize=(16, 5.6 * len(cases)))
    gs = fig.add_gridspec(len(cases) * 2, 4, height_ratios=[1, 1] * len(cases), hspace=0.25, wspace=0.03)

    def verdict(r, success, ftype):  # episode-level output vs simulator truth
        ok = bool(success) == bool(r["truth_success"]) and ftype == r["truth_failure_type"]
        return "[matches simulator]" if ok else "[WRONG vs simulator]"
    for i, r in enumerate(cases):
        frames = r["track"]["frames"]
        times = r["track"]["times_s"]
        z = np.load(ROOT / ep.loc[(r["checkpoint_id"], r["env_seed"]), "episode_npz"])
        kf = [int(round(x)) for x in np.linspace(0, r["video_frames"] - 1, len(frames))]  # as run_cosmos_diagnostics.py
        errs = [_keyframe_error(r["task"], z, k, f) for k, f in zip(kf, frames)]
        for j, (f, t, err) in enumerate(zip(frames[::2], times[::2], errs[::2])):
            ax = fig.add_subplot(gs[2 * i, j])
            ax.imshow(imageio.imread(ROOT / f["image"])); ax.set_xticks([]); ax.set_yticks([])
            title = f"t={t:.1f}s  Cosmos: {f.get('p_object_location')}{' (held)' if f.get('p_held_by_gripper') else ''}"
            if err:  # never show a contradicted label as if it were right
                for s in ax.spines.values():
                    s.set_visible(True); s.set_edgecolor(C["bad"]); s.set_linewidth(3)
                ax.text(0.03, 0.97, "model error", transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
                        color="white", weight="bold", bbox=dict(boxstyle="square,pad=0.25", fc=C["bad"], ec="none"))
                title += f"\nsimulator: {err}"
            ax.set_title(title, fontsize=9, color=C["bad"] if err else "black")
        ax = fig.add_subplot(gs[2 * i + 1, :]); ax.axis("off")
        bad_t = [f"{t:.1f}" for t, e in zip(times, errs) if e]
        lines = [f"EPISODE {r['checkpoint_id']} seed {r['env_seed']}  |  simulator truth: success={r['truth_success']}, "
                 f"failure_type={r['truth_failure_type']}",
                 f"INPUT  8 agentview keyframes (4 shown) + task-free question 'where is the object? is it held?'",
                 f"OUTPUT nvidia/Cosmos3-Edge @{str(r['model_revision'])[:8]} (local RTX 3090, {r['latency_s']:.1f} s for 8 frames): "
                 + ", ".join(f"{x.get('p_object_location')}{'*' if x.get('p_held_by_gripper') else ''}" for x in frames),
                 f"       rule-derived -> success={r.get('cosmos_task_success')}, failure_type={r.get('cosmos_failure_type')}   (* = held)   "
                 + verdict(r, r.get("cosmos_task_success"), r.get("cosmos_failure_type")),
                 (f"       simulator check: {len(bad_t)} of {len(frames)} keyframe labels contradict the simulator state "
                  f"(t={', '.join(bad_t)} s; shown ones outlined in red)") if bad_t else
                 f"       simulator check: no keyframe label contradicts the simulator state"]
        n = neb[(neb.checkpoint_id == r["checkpoint_id"]) & (neb.env_seed == r["env_seed"])]
        if len(n):
            n = n.iloc[0]
            lines += [f"OUTPUT nvidia/Nemotron-3.5-Super-VL via Nebius Token Factory ({n.latency_s:.1f} s, request {n.request_id}): "
                      f"success={n.cosmos_task_success}, failure_type={n.cosmos_failure_type}   "
                      + verdict(r, n.cosmos_task_success, n.cosmos_failure_type)]
            lines += ["       " + x for x in textwrap.wrap(f'"{n.cosmos_explanation}"', 150)[:3]]
        ax.text(0, 0.95, "\n".join(lines), va="top", family="monospace", fontsize=9.5)
    fig.suptitle("Figure D - actual model inputs and actual outputs (one successful and one failed real rollout); "
                 "red = model label contradicted by the simulator state at that frame",
                 x=0.01, ha="left", weight="bold")
    fig.savefig(OUT / "cosmos_example.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return "cosmos_example.png"


# USD per 1M (input, output) tokens, public Nebius catalog snapshot checked 2026-10-29
# (data/outputs/hackathon_explore/y4_grounded_reports/models_info_snapshot.json); the same prices are used
# by nebius_evidence_summary.py (Super-VL) and explore/y4_analyze.py (Lightning). Uncached list prices.
CATALOG_USD_PER_1M = {"nvidia/Nemotron-3.5-Super-VL": (0.30, 0.90), "nvidia/Nemotron-3_5-Lightning": (0.06, 0.24)}
CALL_LOGS = {"main pipeline": RES / "nebius_calls.jsonl", "exploratory": RES / "explore/nebius_calls_explore.jsonl"}


def _nebius_total():
    """Calls, tokens and catalog-price cost per served model over every logged Nebius call (both logs)."""
    if not all(p.exists() for p in CALL_LOGS.values()):
        return None
    out = {"calls": {}, "failed": 0, "models": {}, "unpriced": 0}
    for name, p in CALL_LOGS.items():
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        out["calls"][name] = len(rows)
        for r in rows:
            out["failed"] += bool(r.get("error"))
            m, pin, pout = r.get("served_model") or r["model_id"], r.get("prompt_tokens"), r.get("completion_tokens")
            if m not in CATALOG_USD_PER_1M or pin is None or pout is None:
                out["unpriced"] += 1
                continue
            t = out["models"].setdefault(m, {"calls": 0, "prompt": 0, "completion": 0})
            t["calls"] += 1; t["prompt"] += pin; t["completion"] += pout
    for m, t in out["models"].items():
        t["usd"] = t["prompt"] / 1e6 * CATALOG_USD_PER_1M[m][0] + t["completion"] / 1e6 * CATALOG_USD_PER_1M[m][1]
    return out


def fig_cost_latency():
    p = RES / "cost_latency.json"
    if not p.exists():
        return None
    c = json.loads(p.read_text())
    items = [(k, v) for k, v in c["per_checkpoint_seconds"].items() if v is not None]
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.6), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    names = [k for k, _ in items]; vals = [v for _, v in items]

    def colour(k):
        if "original script" in k:
            return C["cheap"]
        if "rollout" in k:
            return C["sim"]
        if "Cosmos" in k:
            return C["cosmos"]
        return C["nebius"]
    ax.barh(names, vals, color=[colour(k) for k in names])
    for y, v in enumerate(vals):
        ax.text(v, y, f" {v:.0f}s", va="center", fontsize=9)
    ax.set_xlim(0, max(vals) * 1.1)  # linear: the values span less than a decade, a log axis hid the differences
    ax.set_xlabel("measured wall-clock seconds per checkpoint (upper bounds: GPU shared with training)")
    ax.set_title("Per-checkpoint wall-clock time (1x RTX 3090; Nemotron on Nebius)")
    ax = axes[1]
    lat = c.get("per_episode_latency_s", {})
    labs = list(lat.keys())
    ax.barh(labs, [v["mean"] for v in lat.values()],
            xerr=[[v["mean"] - v["p10"] for v in lat.values()], [v["p90"] - v["mean"] for v in lat.values()]],
            color=[C["cosmos"] if "Cosmos" in k else C["nebius"] for k in labs], capsize=4)
    ax.set_xlabel("seconds per episode (mean, p10-p90 whiskers)")
    ax.set_title("Diagnostic latency per episode")
    lines = []
    tot = _nebius_total()
    if tot is not None:
        n_calls = sum(tot["calls"].values())
        head = (f"Nebius Token Factory, all {n_calls:,} logged calls ("
                + " + ".join(f"{v:,} {k}" for k, v in tot["calls"].items()) + f"; {tot['failed']} failed): ")
        if tot["unpriced"]:
            lines.append(head + f"total cost NOT computable - {tot['unpriced']} calls lack logged tokens or a documented price")
        else:
            ms = tot["models"]
            lines.append(head + f"{sum(t['prompt'] for t in ms.values()):,} input / "
                         f"{sum(t['completion'] for t in ms.values()):,} output tokens "
                         f"-> TOTAL ~ ${sum(t['usd'] for t in ms.values()):.2f} estimated at catalog list price")
            lines.append("    = " + ";  ".join(
                f"{m.split('/')[-1]}: {t['calls']:,} calls, {t['prompt']:,} in / {t['completion']:,} out at "
                f"${CATALOG_USD_PER_1M[m][0]:.2f} / ${CATALOG_USD_PER_1M[m][1]:.2f} per 1M = ${t['usd']:.2f}" for m, t in ms.items()))
    if c.get("nebius_usd_estimate") is not None:
        lines.append(f"Main pipeline only (results/hackathon/cost_latency.json, snapshot {c['generated_utc'][:16].replace('T', ' ')} UTC): "
                     f"{c['nebius_tokens']['prompt']:,} in / {c['nebius_tokens']['completion']:,} out = "
                     f"${c['nebius_usd_estimate']:.2f} at the Super-VL catalog price.")
    if lines:
        lines.append("Estimates from logged tokens; cached-input discounts not applied; the billed amount is what the Nebius console shows.")
        fig.text(0.01, 0.015, "\n".join(lines).replace("$", r"\$"),  # literal dollars, not mathtext
                 ha="left", va="bottom", fontsize=9.5, linespacing=1.5)
    fig.suptitle("Figure E - measured cost and latency", x=0.01, ha="left", weight="bold")
    fig.tight_layout(rect=(0, 0.04 + 0.032 * len(lines), 1, 1))
    fig.savefig(OUT / "cost_latency.png", dpi=130)
    plt.close(fig)
    return "cost_latency.png"


if __name__ == "__main__":
    made, skipped = [], []
    for fn in (fig_architecture, fig_checkpoint_comparison, fig_robot_execution, fig_cosmos_example, fig_cost_latency):
        try:
            r = fn()
        except Exception as e:  # report, do not fake
            r = None
            print(f"{fn.__name__} failed: {type(e).__name__}: {e}", file=sys.stderr)
        (made if r else skipped).append(r or fn.__name__)
    print("made:", made)
    print("skipped (inputs missing):", skipped)
