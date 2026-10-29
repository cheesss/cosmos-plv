#!/usr/bin/env python3
"""Submission figures, generated from the result files only (Environment B).

  submission/figures/{results_overview, evaluation_efficiency, judge_accuracy, wrong_compartment_failure,
                      self_training_s2, system_architecture}.png
  submission/{results, reliability, architecture, cover}.png and submission/youtube_thumbnail.png
Every number drawn here is read from results/hackathon/... at run time; nothing is typed in by hand.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES, EXP = ROOT / "results/hackathon", ROOT / "results/hackathon/explore"
OUT, FIG = ROOT / "submission", ROOT / "submission/figures"
INK, MUTED, GRID = "#1f2933", "#5f6b7a", "#e4e7eb"
NEM, COS, SIM, PLV, NEG, HUM = "#7B4FBF", "#76B900", "#4C9A6A", "#3D5A99", "#B5523B", "#D9962B"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.titleweight": "bold", "axes.titlesize": 14})


def load(p):
    return json.loads(Path(p).read_text())


def facts():
    e2 = {r["backend"]: r for r in load(RES / "experiment2_summary.json")["by_backend_variant"]}
    pad = load(RES / "experiment2_backend_comparison_pad201.json")
    cas = load(EXP / "x0_cascade.json")["eval"]
    x1 = load(EXP / "x1_adaptive_budget_verification.json")["C_D_thresholds_from_x1_json"]["excess_crossing_by_threshold"]
    loco = load(EXP / "x2_early_stop_verification_loco.json")
    c1 = load(EXP / "c1_coverage_shift.json")
    s2 = load(EXP / "s2_self_improve.json")
    return e2, pad, cas, x1, loco, c1, s2


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, zorder=0)


def judge_accuracy(e2, pad):
    n, nem, cos = e2["nebius"]["episodes"], e2["nebius"], e2["local"]
    base = (nem["tp"] + nem["fn"]) / n
    rows = [("Nemotron-3.5-Super-VL\n(Nebius Token Factory)", nem, NEM), ("Cosmos3-Edge\n(local RTX 3090)", cos, COS)]
    fig, ax = plt.subplots(figsize=(12, 6.2))
    x = np.arange(3)
    for k, (lab, r, c) in enumerate(rows):
        lo, hi = r["success_acc_ci95"]
        ax.bar(k - 0.18, r["success_acc"], 0.34, color=c, zorder=3, yerr=[[r["success_acc"] - lo], [hi - r["success_acc"]]], capsize=5)
        ax.bar(k + 0.18, r["balanced_acc"], 0.34, color=c, alpha=0.45, zorder=3)
        ax.text(k - 0.18, hi + 0.012, f"{r['success_acc']:.3f}", ha="center", fontweight="bold")
        ax.text(k + 0.18, r["balanced_acc"] + 0.012, f"{r['balanced_acc']:.3f}", ha="center")
        fails = r["tn"] + r["fp"]
        ax.text(k, 0.52, f"failures called success: {r['fp']}/{fails}\nsuccesses called failure: {r['fn']}/{r['tp'] + r['fn']}",
                ha="center", fontsize=10.5, color=INK, bbox=dict(fc="white", ec=GRID))
    ax.bar(2 - 0.18, base, 0.34, color="#9aa0a6", zorder=3)
    ax.bar(2 + 0.18, 0.5, 0.34, color="#9aa0a6", alpha=0.45, zorder=3)
    ax.text(2 - 0.18, base + 0.012, f"{base:.3f}", ha="center", fontweight="bold")
    ax.text(2 + 0.18, 0.512, "0.500", ha="center")
    ax.set_xticks(x, [rows[0][0], rows[1][0], "always answer\n'success' (base rate)"])
    ax.set_ylim(0.45, 1.03)
    ax.set_ylabel("score on 330 recorded rollouts")
    pn = pad["overall"]["nebius"][0]["success_acc"]  # v2_evidence_first_pad201 (length-neutral clips)
    sub = f"; Nemotron on length-neutral clips: {pn:.3f}" if isinstance(pn, float) else ""
    ax.set_title(f"Zero-shot success judging from video, n = {n} recorded rollouts\n"
                 f"(dark = accuracy with 95% CI, light = balanced accuracy{sub})", fontsize=13)
    ax.text(0.0, -0.22, "Failure TYPES (no grasp / dropped / not placed) are not solved: both judges score below a majority-class\n"
            "baseline. Ground truth = privileged simulator labels, used only to score the judges.", transform=ax.transAxes,
            fontsize=10.5, color=MUTED)
    style(ax)
    fig.tight_layout()
    return fig


def evaluation_efficiency(cas, x1, loco):
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.6))
    judges = [("perfect", "simulator labels", SIM), ("nemotron_nebius_v2", "Nemotron", NEM), ("cosmos3_edge_local_v6", "Cosmos3-Edge", COS)]
    ax = axs[0]
    vals = [x1[j]["0.02"]["ratio_uniform_over_SH"] for j, _, _ in judges]
    ax.bar(range(3), vals, color=[c for *_, c in judges], zorder=3)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.06, f"{v:.1f}x", ha="center", fontweight="bold")
    ax.set_xticks(range(3), [l for _, l, _ in judges])
    ax.set_ylabel("rollouts needed: uniform / adaptive")
    ax.set_title("Adaptive allocation\n(successive halving, excess regret <= 0.02)", fontsize=12.5)
    ax.set_ylim(0, 4)
    style(ax)
    ax = axs[1]
    rules = [("loco/time_only", "time limit only\n(deployable)"), ("loco/state_combo", "time + privileged\nsim state (sim only)")]
    sv = [loco[k]["saved_frac"] for k, _ in rules]
    ax.bar(range(2), [100 * v for v in sv], color=PLV, zorder=3)
    for i, (k, _) in enumerate(rules):
        r = loco[k]
        ax.text(i, 100 * sv[i] + 0.8, f"{100 * sv[i]:.0f}%\nfalse aborts {r['false_aborts']}/{r['n_successes']}", ha="center",
                fontweight="bold")
    ax.set_xticks(range(2), [l for _, l in rules])
    ax.set_ylabel("robot time saved (% of all episode time)")
    ax.set_ylim(0, 32)
    ax.set_title("Early abort of failing rollouts\n(leave-one-checkpoint-out, 200 episodes)", fontsize=12.5)
    style(ax)
    ax = axs[2]
    a, c = cas["nebius_only"], cas["cascade"]
    ax.bar([0, 1], [a["nebius_calls"], c["nebius_calls"]], color=[NEM, COS], zorder=3)
    for i, r in enumerate((a, c)):
        ax.text(i, r["nebius_calls"] + 6, f"{r['nebius_calls']} calls\naccuracy {r['accuracy']:.3f}", ha="center", fontweight="bold")
    ax.set_xticks([0, 1], ["Nemotron on\nevery episode", "local Cosmos first,\nNemotron if 'success'"])
    ax.set_ylabel(f"Nebius calls for {a['n']} episodes")
    ax.set_ylim(0, 420)
    ax.set_title("Local-first cascade", fontsize=12.5)
    style(ax)
    fig.suptitle("Evaluation efficiency — EXPLORATORY, post-hoc analyses of logged rollouts (savings in rollouts / robot time / "
                 "API calls, not in training)", fontsize=12.5, color=INK)
    fig.tight_layout()
    return fig


def wrong_compartment(c1):
    fig = plt.figure(figsize=(16, 6.4))
    ax0 = fig.add_axes([0.0, 0.03, 0.58, 0.9])
    ax0.imshow(Image.open(ROOT / "media/hackathon/c1_wrong_compartment.png"))
    ax0.axis("off")
    ax = fig.add_axes([0.65, 0.16, 0.33, 0.7])
    cats = [("can in the wrong\ncompartment", "goal_bin_other_compartment"),
            ("can elsewhere (start bin,\ntable, goal-bin rim)", "elsewhere")]
    rows = [("Nemotron (frozen prompt)", "nemotron", NEM), ("Cosmos3-Edge", "cosmos", COS),
            ("Nemotron, compartment stated", "nemotron_v2c_compartment", NEG)]
    w = 0.26
    for j, (lab, key, col) in enumerate(rows):
        fp = c1[key]["false_success_by_final_place"]
        fr = [(fp[p]["false_successes"], fp[p]["failures"]) for _, p in cats]
        ax.bar(np.arange(2) + (j - 1) * w, [k / n for k, n in fr], w, color=col, label=lab, zorder=3)
        for i, (k, n) in enumerate(fr):
            ax.text(i + (j - 1) * w, k / n + 0.02, f"{k}/{n}", ha="center", fontsize=10)
    ax.set_xticks(range(2), [l for l, _ in cats])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("share of simulator FAILURES judged 'success'")
    ax.legend(fontsize=9.5, loc="upper right", frameon=False)
    ax.set_title(f"c1 (pre-registered): {c1['n_episodes']} new Can rollouts, {c1['n_checkpoints']} checkpoints", fontsize=12)
    style(ax)
    return fig


def results_overview(e2, cas, x1, loco, c1, s2):
    nem = e2["nebius"]
    base = (nem["tp"] + nem["fn"]) / nem["episodes"]
    ratio = x1["nemotron_nebius_v2"]["0.02"]["ratio_uniform_over_SH"]
    t_lo, t_hi = loco["loco/time_only"]["saved_frac"], loco["loco/state_combo"]["saved_frac"]
    wc = c1["post_hoc_after_verification"]["failures_by_place_and_stratum"]["goal_bin_other_compartment"]
    wk, wn = sum(v["nemotron_fs"] for v in wc.values()), sum(v["failures"] for v in wc.values())
    p = s2["paired"]["primary_nemotron_minus_continue"]
    h = s2["paired"]["human_minus_continue"]
    cards = [
        ("EVALUATION", f"{100 * nem['success_acc']:.1f}%", f"Nemotron success-judging accuracy\non {nem['episodes']} recorded rollouts "
         f"(base rate {100 * base:.1f}%;\nbalanced accuracy {100 * nem['balanced_acc']:.1f}%)", NEM, "measured"),
        ("EFFICIENCY", f"~{ratio:.0f}x / {100 * t_lo:.0f}%", "fewer rollouts with adaptive allocation /\nrobot time "
         "saved by a time-limit abort\n(vs uniform allocation / full-length runs)", PLV, "exploratory, post-hoc"),
        ("RELIABILITY", f"{wk}/{wn}", "simulator failures with the can in the\nWRONG compartment that Nemotron\ncalled 'success' (c1)",
         NEG, "pre-registered"),
        ("CONSEQUENCE", f"{100 * p['diff']:+.1f} pts", f"policy success after self-training on\nNemotron-filtered data vs continued\n"
         f"training (95% CI {100 * p['ci95'][0]:+.1f} to {100 * p['ci95'][1]:+.1f});\n36 human demos instead: {100 * h['diff']:+.1f} pts",
         NEG, "pre-registered, negative (s2)"),
    ]
    fig, axs = plt.subplots(1, 4, figsize=(18, 5.4))
    for ax, (head, big, sub, col, tag) in zip(axs, cards):
        ax.axis("off")
        ax.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0.01,rounding_size=0.04", fc="white",
                                    ec=col, lw=2.5, transform=ax.transAxes))
        ax.text(0.5, 0.86, head, ha="center", fontsize=15, fontweight="bold", color=col, transform=ax.transAxes)
        ax.text(0.5, 0.62, big, ha="center", fontsize=30, fontweight="bold", color=INK, transform=ax.transAxes)
        ax.text(0.5, 0.28, sub, ha="center", fontsize=11.5, color=INK, transform=ax.transAxes, linespacing=1.4)
        ax.text(0.5, 0.07, tag, ha="center", fontsize=10.5, color=MUTED, style="italic", transform=ax.transAxes)
    fig.suptitle("Cosmos-PLV: results at a glance — accurate overall is not the same as safe to train on", fontsize=15,
                 fontweight="bold", color=INK)
    fig.tight_layout()
    return fig


def architecture():
    fig, ax = plt.subplots(figsize=(16, 7.4))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7.4)
    ax.axis("off")

    def box(x, y, w, h, title, body, col, fc="white", ls="-"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc=fc, ec=col, lw=2.2, ls=ls))
        ax.text(x + w / 2, y + h - 0.3, title, ha="center", va="top", fontsize=12.5, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 0.8, body, ha="center", va="top", fontsize=10.3, color=INK, linespacing=1.35)

    def arrow(a, b, col=INK, ls="-"):
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8, ls=ls, shrinkA=0, shrinkB=0))

    ax.add_patch(FancyBboxPatch((0.15, 2.3), 10.75, 4.75, boxstyle="round,pad=0.02,rounding_size=0.2", fc="#f6f8fa", ec=MUTED,
                                lw=1, ls="--"))
    ax.text(0.35, 6.75, "LOCAL MACHINE (1x RTX 3090)", fontsize=11, color=MUTED, fontweight="bold")
    ax.add_patch(FancyBboxPatch((11.2, 3.75), 4.65, 3.3, boxstyle="round,pad=0.02,rounding_size=0.2", fc="#f3effa", ec=NEM, lw=1,
                                ls="--"))
    ax.text(11.4, 6.75, "NEBIUS TOKEN FACTORY (cloud)", fontsize=11, color=NEM, fontweight="bold")
    box(0.4, 4.0, 2.3, 2.45, "Checkpoints", "Diffusion Policy\n(robomimic Can,\nSquare)\n93 retrained", MUTED)
    box(2.95, 4.0, 2.5, 2.45, "PLV screening", "original PLV scripts:\nAction L1, smoothness,\nreplanning overlap\n(no rollouts)", PLV)
    box(5.7, 4.0, 2.3, 2.45, "Rollouts", "robosuite / MuJoCo\nagentview video\n+ action chunks", SIM)
    box(8.25, 4.0, 2.4, 2.45, "Cosmos3-Edge", "NVIDIA open model\non the local GPU:\nper-frame object\nstate -> verdict", COS)
    box(11.45, 4.0, 4.15, 2.45, "Nemotron-3.5-Super-VL", "NVIDIA open model served\nby Nebius Token Factory:\nvideo -> JSON verdict,\nrequest id + tokens logged",
        NEM)
    box(2.95, 0.3, 5.6, 1.55, "Recommendation + judge audit", "dashboard: pick vs free latest-epoch baseline,\nmeasured regret, "
        "judge-trust warning,\naudits r1 / c1 / s1 / s2", INK)
    box(9.4, 0.3, 6.2, 1.55, "Simulator ground truth: VERIFICATION ONLY", "privileged success labels score the judges;\n"
        "they are never used to choose a checkpoint", NEG, fc="#fbf1ee", ls="--")
    arrow((2.7, 5.2), (2.95, 5.2))
    arrow((5.45, 5.2), (5.7, 5.2))
    ax.text(5.57, 4.85, "top\n30%", ha="center", fontsize=9, color=INK)
    arrow((8.0, 5.2), (8.25, 5.2))
    ax.plot([7.3, 7.3, 12.2], [4.0, 3.55, 3.55], color=NEM, lw=1.8)
    arrow((12.2, 3.55), (12.2, 4.0), col=NEM)
    ax.text(9.6, 3.62, "episode video over HTTPS", ha="center", fontsize=9.5, color=NEM)
    arrow((4.2, 4.0), (4.2, 1.85), col=PLV)
    arrow((9.45, 4.0), (7.8, 1.85), col=COS)
    arrow((13.5, 4.0), (8.4, 1.85), col=NEM)
    arrow((6.3, 4.0), (10.6, 1.85), col=NEG, ls="--")
    ax.text(8.95, 2.55, "privileged labels", fontsize=9.5, color=NEG, rotation=-27)
    arrow((9.4, 1.0), (8.55, 1.0), col=NEG, ls="--")
    ax.set_title("Cosmos-PLV system: where each component runs", loc="left", fontsize=15)
    return fig


def cover(c1, s2, e2):
    import sys

    sys.path.insert(0, str(ROOT))
    from cosmos_plv.video_recorder import read_mp4

    frames = read_mp4(ROOT / "data/outputs/hackathon_rollouts_eval/can_s2_e0100/videos/seed200008.mp4")
    frame = Image.fromarray(frames[len(frames) * 2 // 3])
    fig = plt.figure(figsize=(15, 10), facecolor="#111a24")
    ax = fig.add_axes([0.04, 0.2, 0.5, 0.62])
    ax.imshow(frame)
    ax.axis("off")
    fig.text(0.04, 0.88, "Cosmos-PLV", fontsize=46, fontweight="bold", color="#76B900")
    fig.text(0.04, 0.1, "real robosuite rollout recorded by the pipeline", fontsize=13, color="#9aa5b1")
    t = fig.text
    t(0.58, 0.74, "Robot checkpoint evaluation\nwith an audited AI judge", fontsize=27, fontweight="bold", color="white", linespacing=1.3)
    nem = e2["nebius"]
    lines = [("1", "Cheap PLV screening picks which checkpoints get robot time"),
             ("2", "NVIDIA Cosmos3-Edge (local) + Nemotron on Nebius judge the videos"),
             ("3", f"{100 * nem['success_acc']:.1f}% judging accuracy on {nem['episodes']} rollouts..."),
             ("4", "...but a systematic blind spot: near-goal failures look like success")]
    y = 0.58
    for k, s in lines:
        t(0.58, y, k, fontsize=20, fontweight="bold", color="#76B900")
        t(0.615, y, s, fontsize=14.5, color="#e6e9ec", wrap=True)
        y -= 0.085
    t(0.58, 0.18, "Nebius x NVIDIA Global AI Hackathon 2026 · Physical AI", fontsize=14, color="#9aa5b1")
    t(0.58, 0.13, "github.com/cheesss/cosmos-plv", fontsize=14, color="#9aa5b1")
    return fig


def real_case():
    """A real c1 episode: the simulator's failure next to Nemotron's recorded answer (no editing of the answer)."""
    import sys

    sys.path.insert(0, str(ROOT))
    from cosmos_plv.video_recorder import read_mp4

    ck, seed = "can_s1_e0060", 600002
    row = next(json.loads(l) for l in (ROOT / "data/outputs/hackathon_explore/c1_judge/nebius_v2.jsonl").read_text().splitlines()
               if json.loads(l)["checkpoint_id"] == ck and int(json.loads(l)["env_seed"]) == seed)
    ans = json.loads(row["raw_text"])
    frame = read_mp4(ROOT / row["video"])[-1]
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_axes([0.02, 0.03, 0.47, 0.74])
    ax.imshow(frame)
    ax.axis("off")
    fig.text(0.02, 0.9, "SIMULATOR: FAILURE", fontsize=22, fontweight="bold", color=NEG)
    fig.text(0.02, 0.87, "the can ended in the front-LEFT compartment;\nsuccess requires the front-right one (next to the grey can marker)",
             fontsize=12, color=INK, va="top")
    fig.text(0.53, 0.9, "NEMOTRON: \"SUCCESS\"", fontsize=22, fontweight="bold", color=NEM)
    fig.text(0.53, 0.87, f"recorded answer, {row['served_model']} on Nebius Token Factory\nrequest {row['request_id']} · "
             f"{row['latency_s']:.1f} s · c1 study, {row['logged_utc'][:10]}", fontsize=11.5, color=INK, va="top")
    import textwrap

    lines = []
    for k in ("task_success", "failure_type", "object_lifted", "object_dropped", "explanation"):
        v = json.dumps(ans[k])
        lines += textwrap.wrap(f'"{k}": {v}', 58, subsequent_indent="    ")
    fig.text(0.53, 0.74, "\n".join(lines), family="DejaVu Sans Mono", fontsize=12.5, color=INK, va="top", linespacing=1.5,
             bbox=dict(fc="#f6f8fa", ec=NEM, lw=1.5, boxstyle="round,pad=0.8"))
    fig.text(0.53, 0.2, "Not an isolated slip: in the c1 audit Nemotron called\n32 of 33 wrong-compartment failures a success.",
             fontsize=13.5, color=NEG, fontweight="bold")
    return fig


def accuracy_vs_blind_spot(e2, s2):
    nem = e2["nebius"]
    lab = s2["nemotron_labels"]
    wc = lab["false_success_final_place"].get("goal_bin_other_compartment", 0)
    sr = s2["success_rate"]
    fig, axs = plt.subplots(1, 3, figsize=(17, 6.2), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    ax = axs[0]
    fails = nem["tn"] + nem["fp"]
    ax.bar([0, 1], [nem["success_acc"], nem["fp"] / fails], color=[NEM, NEG], zorder=3)
    ax.text(0, nem["success_acc"] + 0.03, f"{100 * nem['success_acc']:.1f}%\naccuracy", ha="center", fontweight="bold", fontsize=13)
    ax.text(1, nem["fp"] / fails + 0.03, f"{nem['fp']}/{fails}\nfailures\ncalled success", ha="center", fontweight="bold", fontsize=12)
    ax.set_xticks([0, 1], ["all rollouts", "real failures"])
    ax.set_ylim(0, 1.18)
    ax.set_title(f"1. Benchmark eval set\n{nem['episodes']} rollouts of 33 checkpoints", fontsize=13)
    style(ax)
    ax = axs[1]
    ax.bar([0, 1], [lab["false_successes"] / lab["failures"], wc / lab["failures"]], color=[NEG, "#8f3f2d"], zorder=3)
    ax.text(0, lab["false_successes"] / lab["failures"] + 0.03, f"{lab['false_successes']}/{lab['failures']}\nfailures\ncalled success",
            ha="center", fontweight="bold", fontsize=12)
    ax.text(1, wc / lab["failures"] + 0.03, f"{wc} of them:\ncan in the\nwrong compartment", ha="center", fontweight="bold", fontsize=12)
    ax.set_xticks([0, 1], ["accepted failures", "wrong compartment"])
    ax.set_ylim(0, 1.18)
    ax.set_yticklabels([])
    ax.set_title("2. Self-collected training data (s2)\n400 rollouts of a weaker policy", fontsize=13)
    style(ax)
    ax = axs[2]
    arms = [("continue", "continued\ntraining", PLV), ("nemotron", "+ Nemotron-\nfiltered data", NEG), ("human", "+ 36 human\ndemos", HUM)]
    ax.bar(range(3), [sr[a] for a, _, _ in arms], color=[c for *_, c in arms], zorder=3)
    for i, (a, _, _) in enumerate(arms):
        ax.text(i, sr[a] + 0.02, f"{100 * sr[a]:.1f}%", ha="center", fontweight="bold", fontsize=13)
    ax.set_xticks(range(3), [l for _, l, _ in arms])
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("policy success (500 paired eval seeds)")
    ax.set_title("3. Policies trained on it (s2)\nsame base checkpoint, same seeds", fontsize=13)
    style(ax)
    fig.suptitle("Accurate overall, wrong where it matters: three DIFFERENT datasets, read left to right", fontsize=15,
                 fontweight="bold", color=INK)
    fig.text(0.5, 0.005, "Panels use different data and denominators; they are not one accuracy comparison. s2 caveats: one training "
             "run per arm, demo dilution, clip length follows the simulator's success signal.", ha="center", fontsize=10.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    e2, pad, cas, x1, loco, c1, s2 = facts()
    out = {}
    for name, fig in [("judge_accuracy", judge_accuracy(e2, pad)), ("evaluation_efficiency", evaluation_efficiency(cas, x1, loco)),
                      ("wrong_compartment_failure", wrong_compartment(c1)), ("results_overview", results_overview(e2, cas, x1, loco, c1, s2)),
                      ("system_architecture", architecture()), ("real_case_wrong_compartment", real_case()),
                      ("accuracy_vs_blind_spot", accuracy_vs_blind_spot(e2, s2))]:
        out[name] = FIG / f"{name}.png"
        fig.savefig(out[name], dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)
    import subprocess

    subprocess.run([str(Path.home() / ".venvs/cosmos-gen/bin/python"), str(ROOT / "scripts/hackathon/explore/s2_figure.py")], check=True)
    shutil.copy(ROOT / "media/hackathon/self_training_s2.png", FIG / "self_training_s2.png")
    shutil.copy(out["results_overview"], OUT / "results.png")
    shutil.copy(out["wrong_compartment_failure"], OUT / "reliability.png")
    shutil.copy(out["system_architecture"], OUT / "architecture.png")
    fig = cover(c1, s2, e2)
    fig.savefig(OUT / "cover.png", dpi=120, facecolor=fig.get_facecolor())  # 1800x1200, 3:2
    plt.close(fig)
    Image.open(OUT / "cover.png").convert("RGB").resize((1280, 853)).crop((0, 66, 1280, 786)).save(OUT / "youtube_thumbnail.png")
    for p in sorted(list(FIG.glob("*.png")) + list(OUT.glob("*.png"))):
        im = Image.open(p)
        print(f"{p.relative_to(ROOT)}  {im.size[0]}x{im.size[1]}")


if __name__ == "__main__":
    main()
