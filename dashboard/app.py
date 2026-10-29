"""Cosmos-PLV dashboard (Environment B):  streamlit run dashboard/app.py

Shows ONLY measured or generated results from results/hackathon/ and the recorded
rollouts; missing values are shown as "unavailable".  Live actions:
  * recompute cheap metrics for a checkpoint (original repo script, Environment A)
  * record a new simulator rollout (Environment A)
  * run NVIDIA Cosmos3-Edge on an episode (local GPU, this process)
  * ask NVIDIA Nemotron-3.5-Super-VL on Nebius Token Factory (live API call)
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from cosmos_plv.evaluation_pipeline import (  # noqa: E402
    claims_grasp, failure_type_reliability, latest_epoch_pick, recommend, rule_regrets)
from cosmos_plv.visual_diagnostics import (  # noqa: E402
    aggregate_track, build_perception_prompt, build_prompt, parse_diagnostic, parse_perception)

RES = ROOT / "results" / "hackathon"
# raw model outputs: the full local run if present, else the committed copy (fresh clone)
DIAG_DIRS = [ROOT / "data" / "outputs" / "hackathon_cosmos", RES / "raw_model_outputs"]
NEBIUS_LOGS = [RES / "nebius_calls.jsonl", RES / "explore" / "nebius_calls_explore.jsonl"]
ROLL_DIRS = [ROOT / "data/outputs/hackathon_rollouts_eval", ROOT / "data/outputs/hackathon_live",
             ROOT / "media/hackathon/episodes"]
NEEDS_POOL = "needs the checkpoint pool (see README: Data and checkpoints)"
ENV_A = os.environ.get("PLV_ENV_SH", str(ROOT / "scripts/hackathon/env.sh"))
GREEN, PURPLE, BLUE, GREY = "#76B900", "#7E57C2", "#4C72B0", "#8C8C8C"
OK_C, BAD_C = "#55A868", "#C44E52"
JUDGE = {"local": "Cosmos3-Edge", "nebius": "Nemotron VL"}
FT_KEY = {"local": "local_failure", "nebius": "nebius"}  # experiment2 key holding each judge's failure types

st.set_page_config(page_title="Cosmos-PLV", page_icon="🤖", layout="wide")
st.markdown("""<style>
.block-container {padding-top: 1.2rem; padding-bottom: 1rem;}
div[data-testid="stMetricValue"] {font-size: 1.6rem;}
.small {color:#666; font-size:0.85rem;}
.tag {display:inline-block; padding:1px 8px; border-radius:10px; font-size:0.8rem; color:white; margin-right:6px;}
</style>""", unsafe_allow_html=True)


@st.cache_data(ttl=30)
def load_csv(name):
    p = RES / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data(ttl=30)
def load_json(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data(ttl=30)
def load_jsonl(path):
    p = Path(path)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


@st.cache_data(ttl=30)
def load_episodes():
    rows = []
    for d in ROLL_DIRS:
        for s in sorted(d.glob("*/summary.json")):
            js = json.loads(s.read_text())
            for e in js["episodes"]:
                e["_dir"] = rel(s.parent)
                rows.append(e)
    df = pd.DataFrame(rows)
    # the full recordings (data/) take precedence over the public 320-px samples (media/)
    return df.drop_duplicates(subset=["checkpoint_id", "env_seed"], keep="first") if len(df) else df


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "unavailable"
    return f"{x:.{nd}f}" if isinstance(x, (float, np.floating)) else str(x)


def rel(text):
    """Repo-relative paths for display: never show the absolute local checkout or home directory."""
    return str(text).replace(str(ROOT) + "/", "").replace(str(ROOT), ".").replace(str(Path.home()), "~")


def diag_path(name):
    """First existing copy of a raw model-output file (DIAG_DIRS order)."""
    return next((d / name for d in DIAG_DIRS if (d / name).exists()), DIAG_DIRS[-1] / name)


def is_true(x):
    return str(x).strip().lower() in ("true", "1", "1.0")


def is_false(x):
    return str(x).strip().lower() in ("false", "0", "0.0")


def outcome(success, ftype=None):
    """'success' / 'failure (type)' for a simulator label or a model verdict."""
    if is_true(success):
        return "success"
    if not is_false(success):
        return "no verdict"
    return f"failure ({ftype})" if isinstance(ftype, str) and ftype not in ("", "none") else "failure"


def tag(text, ok):
    return f'<span class="tag" style="background:{OK_C if ok else BAD_C}">{"✓" if ok else "✗"} {text}</span>'


def grasp_claims(r, e):
    """Why a saved verdict claims a grasp the simulator never registered (empty list = no conflict)."""
    if not is_false(e.get("grasped")):
        return []
    why = ["explanation"] if claims_grasp(r.get("cosmos_explanation")) else []
    return why + (["object_lifted = true"] if is_true(r.get("cosmos_object_lifted")) else [])


@st.cache_data
def quick_settings():
    """The flags scripts/hackathon/quick_cheap_metrics.py passes to the original script (read from it)."""
    p = ROOT / "scripts/hackathon/quick_cheap_metrics.py"
    f = dict(re.findall(r'"--(max-eval-samples|horizon|policy-seed)", "([^"]+)"', p.read_text())) if p.exists() else {}
    if len(f) < 3:
        return "settings unavailable"
    return f"{f['max-eval-samples']} validation windows, H = {f['horizon']}, policy seed {f['policy-seed']}, EMA weights"


def run_env_a(args, log_box):
    cmd = f"source {ENV_A} && cd {ROOT} && " + " ".join(args)
    p = subprocess.Popen(["bash", "-lc", cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    for line in p.stdout:
        if "WARNING" in line or "it/s]" in line or "macro" in line:
            continue
        lines.append(rel(line.rstrip()))
        log_box.code("\n".join(lines[-12:]))
    return p.wait()


@st.cache_resource
def cosmos_model():
    from cosmos_plv.cosmos_backend import CosmosReasonerLocal
    return CosmosReasonerLocal()


baseline = load_csv("baseline_metrics.csv")
screen = load_csv("hybrid_screening.csv")
diag = load_csv("cosmos_diagnostics.csv")
ranking = load_csv("hybrid_ranking.csv")
base_rank = load_csv("baseline_ranking.csv")
exp2 = load_json("experiment2_backend_comparison.json")
exp3 = load_json("experiment3_summary.json")
cost = load_json("cost_latency.json")
episodes = load_episodes()
v6_list = load_jsonl(diag_path("eval_local_v6.jsonl"))
v7_list = load_jsonl(diag_path("eval_local_v7.jsonl"))
v7_rows = {(r["checkpoint_id"], r["env_seed"]): r for r in v7_list}
neb_logs = {p: load_jsonl(p) for p in NEBIUS_LOGS}
nebius_log = [r for rows in neb_logs.values() for r in rows]
ft_rel = failure_type_reliability(exp2)


def ft_caveat(backend):
    """One-line warning that predicted failure types are below a majority-class guess (Experiment 2)."""
    f = ft_rel.get(FT_KEY.get(backend, ""))
    if not f:
        return "Predicted failure types are unreliable (below a majority-class guess in Experiment 2)."
    return (f"Predicted failure types are unreliable: {JUDGE.get(backend, backend)} failure-type accuracy on the "
            f"{f['n_failures']} failed eval episodes is {f['acc']:.2f}, below always guessing "
            f"'{f['majority_type']}' ({f['majority_acc']:.2f}; Experiment 2).")

# ------------------------------------------------------------------ header
st.markdown("## 🤖 Cosmos-PLV — which robot checkpoint should I evaluate next?")
st.markdown(
    '<span class="tag" style="background:#4C72B0">cheap PLV screening</span>'
    '<span class="tag" style="background:#55A868">real robosuite rollouts</span>'
    f'<span class="tag" style="background:{GREEN}">NVIDIA Cosmos3-Edge (local GPU)</span>'
    f'<span class="tag" style="background:{PURPLE}">NVIDIA Nemotron VL on Nebius Token Factory</span>'
    '<span class="small">All numbers are measured; unavailable values are labelled.</span>',
    unsafe_allow_html=True)
k = st.columns(6)
k[0].metric("Checkpoints trained", len(baseline) if len(baseline) else "–")
n_train = baseline.n_train_rollouts.sum() if "n_train_rollouts" in baseline else 50 * len(baseline)
k[1].metric("Training-time rollouts", f"{int(n_train):,}" if len(baseline) else "–")
# eval episodes = recorded videos here plus every episode a judge scored (cosmos_diagnostics.csv), so a
# fresh clone without data/outputs still counts the full eval set
ep_keys = set(zip(episodes.checkpoint_id, episodes.env_seed.astype(int))) if len(episodes) else set()
ev_keys = set(zip(diag.checkpoint_id, diag.env_seed.astype(int))) if len(diag) else set()
k[2].metric("Recorded eval rollouts", f"{len(ep_keys | ev_keys):,}" if (ep_keys or ev_keys) else "–",
            help=f"videos in this checkout: {len(episodes):,}" + ("" if len(episodes) >= len(ev_keys) else
                 " (public samples in media/hackathon/episodes; full recordings: see README, Data and checkpoints)"))
# Cosmos calls from the raw model outputs (v6: 1 call per episode, v7: 1 per keyframe);
# without the raw logs, v6 falls back to the local rows of cosmos_diagnostics.csv
n_v6 = len(v6_list) if v6_list else (int((diag.backend == "local").sum()) if len(diag) else 0)
n_v7 = sum(len((r.get("track") or {}).get("frames", [])) for r in v7_list)
src_v6 = rel(diag_path("eval_local_v6.jsonl")) if v6_list else "results/hackathon/cosmos_diagnostics.csv"
k[3].metric("Cosmos3-Edge calls (eval)", f"{n_v6 + n_v7:,}" if n_v6 + n_v7 else "–",
            help=f"{n_v6:,} last-frame calls (v6, {src_v6}) + " +
                 (f"{n_v7:,} keyframe calls (v7, {rel(diag_path('eval_local_v7.jsonl'))})" if v7_list
                  else "v7 keyframe calls not counted (eval_local_v7.jsonl missing)"))
k[4].metric("Nebius API calls", f"{len(nebius_log):,}" if nebius_log else "–",
            help=" + ".join(f"{len(v):,} in {rel(p)}" for p, v in neb_logs.items()) + " (one logged line per call)")
if exp2 and exp2.get("overall"):
    loc = exp2["overall"].get("local_success", [{}])[0].get("success_acc_all")
    neb = exp2["overall"].get("nebius", [{}])[0].get("success_acc_all")
    k[5].metric("Success-judge accuracy", f"{loc:.2f} / {neb:.2f}" if loc and neb else "–",
                help="vs privileged simulator labels on the eval split: local Cosmos / Nebius Nemotron")

with st.sidebar:
    st.header("Selection")
    tasks = sorted(baseline.task.unique()) if len(baseline) else ["can"]
    task = st.selectbox("Robot task", tasks)
    runs = sorted(baseline[baseline.task == task].run_id.unique()) if len(baseline) else []
    run = st.selectbox("Training run", runs) if runs else None
    cks = baseline[baseline.run_id == run].sort_values("epoch") if run else pd.DataFrame()
    ev_ids = set(episodes.checkpoint_id) if len(episodes) else set()
    labels = [f"{c}{'  🎥' if c in ev_ids else ''}" for c in cks.checkpoint_id] if len(cks) else []
    pick = st.selectbox("Checkpoint (🎥 = recorded rollouts)", labels) if labels else None
    ck_id = pick.split()[0] if pick else None
    show_truth = st.toggle("Reveal verified success (never used to choose)", value=True)
    st.caption("Models: `nvidia/Cosmos3-Edge` @344d602b (local), `nvidia/Nemotron-3.5-Super-VL` (Nebius).")

tab_rec, tab_ck, tab_roll, tab_res, tab_exp, tab_cost = st.tabs(
    ["① Recommend", "② Cheap screening", "③ Rollouts & Cosmos", "④ Results", "⑤ Explorations", "⑥ Cost & provenance"])

# ------------------------------------------------------------------ recommendation
with tab_rec:
    if run is None or screen.empty:
        st.info("Hybrid screening results unavailable (run scripts/hackathon/analyze_hybrid.py).")
    else:
        vcols = [c for c in screen.columns if c.startswith("vlm_success_")]
        c1, c2 = st.columns([1, 2])
        with c1:
            vlm_col = st.radio("Stage-2 judge", vcols, horizontal=False,
                               format_func=lambda c: {"vlm_success_local": "Cosmos3-Edge (local)",
                                                      "vlm_success_nebius": "Nemotron VL (Nebius)"}.get(c, c)) if vcols else None
            q = st.slider("Stage-1: keep top fraction by Action L1", 0.1, 1.0, 0.3, 0.1)
        bk = vlm_col.replace("vlm_success_", "") if vlm_col else None
        fails = diag[diag.backend == bk] if (vlm_col and len(diag)) else None
        g = screen[screen.run_id == run].sort_values("epoch")
        if g.val_action_l1.notna().any():
            rec = recommend(screen[screen.val_action_l1.notna()], run, q=q, vlm_col=vlm_col, failures=fails)
            free = latest_epoch_pick(screen, run)
            with c2:
                st.success(f"### Evaluate next: **{rec['recommended_checkpoint']}**\n{rec['basis']}")
                for r in rec["reasons"]:
                    st.write("•", r)
                if rec["predicted_failure_modes"]:
                    st.caption("⚠ " + ft_caveat(bk))
                same = free["recommended_checkpoint"] == rec["recommended_checkpoint"]
                st.info(f"Free baseline, latest epoch: **{free['recommended_checkpoint']}** (epoch {free['epoch']}; "
                        f"no metric, no rollouts) — {'the same checkpoint' if same else 'a different checkpoint'}.")
                # measured regret of each rule's single pick in this pool (Experiment 3, hybrid_ranking.csv)
                rr = rule_regrets(ranking)
                if len(rr):
                    by = dict(zip(rr.metric, rr.mean_pick_regret))
                    ts, ep0 = by.get(f"hybrid_twostage_l1_{bk}"), by.get("epoch")
                    n_runs = int(rr.n_runs.max())
                    q0 = (exp3 or {}).get("q")
                    if ts is not None and ep0 is not None:
                        msg = (f"two-stage {ts:.3f} vs latest epoch {ep0:.3f} (regret of the single pick, "
                               f"mean of {n_runs} runs{f', q = {q0}' if q0 is not None else ''}; Experiment 3).")
                        if ts >= ep0:
                            st.warning(f"**In this pool the two-stage rule does not beat the free latest-epoch "
                                       f"pick:** {msg}")
                        else:
                            st.info(f"In this pool the two-stage rule beats the latest-epoch pick: {msg}")
                    if q0 is not None and abs(q - q0) > 1e-9:
                        st.caption(f"The measured regrets are for q = {q0}; the slider is at q = {q:.1f}.")
                if show_truth:
                    best = g.success.max()
                    got = float(g[g.checkpoint_id == rec["recommended_checkpoint"]].success.iloc[0])
                    got0 = float(g[g.checkpoint_id == free["recommended_checkpoint"]].success.iloc[0])
                    st.caption(f"Verification only — 50-rollout success of the pick {got:.2f}, best in run {best:.2f}, "
                               f"regret {best - got:.2f}; latest-epoch pick {got0:.2f}, regret {best - got0:.2f}.")
                if len(rr):
                    other = {"local": "_nebius", "nebius": "_local"}.get(bk, "\0")
                    t = rr[~rr.metric.str.endswith(other)]
                    shown = {f"hybrid_twostage_l1_{bk}": "← recommendation above", "epoch": "← free baseline"}
                    tbl = pd.DataFrame({"rule": t.label, "regret of the single pick (mean of runs)": t.mean_pick_regret.round(3)})
                    if show_truth and run in t:
                        tbl[f"in {run}"] = t[run].round(3)
                    tbl[""] = t.metric.map(shown).fillna("")
                    st.markdown("**Measured regret of each rule in this pool** (`results/hackathon/hybrid_ranking.csv`, "
                                "lower is better)")
                    st.dataframe(tbl, hide_index=True, use_container_width=True)
            g = g.assign(retained=g.checkpoint_id.isin(rec["retained"]),
                         recommended=g.checkpoint_id == rec["recommended_checkpoint"])
        else:
            with c2:
                st.warning("Cheap metrics for this run are still being computed.")
            g = g.assign(retained=False, recommended=False)
        long = []
        for col, lab in [("success", "verified 50-rollout success"), ("k_sim_success", "10 eval rollouts (sim)")] + \
                [(c, {"vlm_success_local": "Cosmos-judged (10)", "vlm_success_nebius": "Nemotron-judged (10)"}[c]) for c in vcols]:
            if col == "success" and not show_truth:
                continue
            if col in g:
                long += [{"epoch": e, "value": v, "series": lab, "retained": r} for e, v, r in zip(g.epoch, g[col], g.retained)]
        if long:
            ch = alt.Chart(pd.DataFrame(long)).mark_line(point=True).encode(
                x="epoch:Q", y=alt.Y("value:Q", title="success rate", scale=alt.Scale(domain=[0, 1.02])),
                color=alt.Color("series:N", scale=alt.Scale(
                    domain=["verified 50-rollout success", "10 eval rollouts (sim)", "Cosmos-judged (10)", "Nemotron-judged (10)"],
                    range=["#2F4B7C", "#55A868", GREEN, PURPLE])),
                tooltip=["epoch", "series", "value"]).properties(height=260)
            rules = alt.Chart(g[g.retained]).mark_rule(color="#f5a623", opacity=0.35, size=8).encode(x="epoch:Q")
            st.altair_chart(rules + ch, use_container_width=True)
            st.caption("Orange bands = checkpoints kept by the cheap stage-1 screen.")
        cols = ["checkpoint_id", "retained", "epoch", "val_action_l1", "open_loop_smoothness", "plv1_smoothness"] + vcols + ["k_sim_success"]
        if show_truth:
            cols.append("success")
        st.dataframe(g[[c for c in cols if c in g]], hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ checkpoint
with tab_ck:
    if ck_id is None:
        st.info("No checkpoints found.")
    else:
        row = cks[cks.checkpoint_id == ck_id].iloc[0]
        st.markdown(f"#### {ck_id} · epoch {int(row.epoch)}")
        c = st.columns(5)
        c[0].metric("Action L1 (L1)", fmt(row.get("val_action_l1")))
        c[1].metric("Open-loop smoothness (L2)", fmt(row.get("open_loop_smoothness")))
        c[2].metric("PLV-1 smoothness (L2)", fmt(row.get("plv1_smoothness")))
        c[3].metric("Replan overlap (L3)", fmt(row.get("replan_overlap_h16")))
        c[4].metric("Verified success", fmt(row.get("success"), 2) if show_truth else "hidden")
        st.code(rel(row.checkpoint_path))
        have_ck = (ROOT / str(row.checkpoint_path)).exists()
        st.markdown(f"**Recompute now — quick mode** (`scripts/hackathon/quick_cheap_metrics.py` → original "
                    f"`plv_open_loop_smoothness_baseline.py`): Action L1 + open-loop smoothness on {quick_settings()}, "
                    f"this checkpoint only, ~1 min")
        st.caption("Quick mode is a live check, not the reported numbers: the reported Action L1 comes from the PLV-1 "
                   "metric script, so the two can differ slightly.")
        if not have_ck:
            st.caption(f"Disabled: {NEEDS_POOL}.")
        if st.button("▶ Run cheap metrics (quick mode)", key="cheap", disabled=not have_ck):
            out = ROOT / "data/outputs/hackathon_live" / f"cheap_{ck_id}.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            box = st.empty()
            t0 = time.time()
            code = run_env_a(["python", "scripts/hackathon/quick_cheap_metrics.py", "--checkpoint",
                              f"'{row.checkpoint_path}'", "--out", rel(out)], box)
            if code == 0 and out.exists():
                r = pd.read_csv(out).iloc[0]
                st.success(f"quick mode done in {time.time() - t0:.0f}s: Action L1 = {r.val_action_l1:.4f} "
                           f"(reported {fmt(row.get('val_action_l1'), 4)}), open-loop smoothness = "
                           f"{r.open_loop_smoothness:.4f} (reported {fmt(row.get('open_loop_smoothness'), 4)})")
            else:
                st.error(f"exit code {code}")
        if len(cks) and "val_action_l1" in cks and cks.val_action_l1.notna().any():
            m = cks.melt(id_vars=["epoch"], value_vars=[c for c in ("val_action_l1", "open_loop_smoothness", "plv1_smoothness")
                                                         if c in cks], var_name="metric", value_name="value")
            st.altair_chart(alt.Chart(m).mark_line(point=True).encode(
                x="epoch:Q", y=alt.Y("value:Q", scale=alt.Scale(type="log"), title="cost (log, lower = better)"),
                color="metric:N").properties(height=230), use_container_width=True)

# ------------------------------------------------------------------ rollouts & cosmos
with tab_roll:
    ep = episodes[episodes.checkpoint_id == ck_id] if (ck_id and len(episodes)) else pd.DataFrame()
    d_ck = diag[diag.checkpoint_id == ck_id] if (ck_id and len(diag)) else pd.DataFrame()

    def episode_label(s):
        """Episode picker entry: simulator label, then ✓/✗ per saved model verdict (⚠ = grasp claim vs sim)."""
        x = ep[ep.env_seed == s].iloc[0]
        marks = [f"{JUDGE.get(r.backend, r.backend)} {'✓' if outcome(r.cosmos_task_success) == outcome(x.success) else '✗'}"
                 f"{' ⚠' if grasp_claims(r, x) else ''}" for _, r in d_ck[d_ck.env_seed == s].iterrows()]
        return f"{s} — sim: {x.failure_type}" + ("".join(f" · {m}" for m in marks))

    if ep.empty:
        st.info("No recorded rollouts for this checkpoint (choose one marked 🎥, or record one below).")
        seed = None
    else:
        seed = st.selectbox("Episode (env seed) — sim label · saved model verdicts (✓ agrees with sim, "
                            "⚠ claims a grasp the sim never registered)", ep.env_seed.tolist(), format_func=episode_label)
    left, right = st.columns([3, 2])
    with left:
        if seed is not None:
            e = ep[ep.env_seed == seed].iloc[0]
            if e.get("video") and (ROOT / e["video"]).exists():
                st.video(str(ROOT / e["video"]))
            st.markdown(f"**Verified simulator outcome:** success = `{e.success}`, failure type = `{e.failure_type}`, "
                        f"grasped = `{e.grasped}`, lifted = `{e.lifted}`, max object height = {e.max_object_height_m:.3f} m")
            tr = v7_rows.get((ck_id, int(seed)))
            if tr and tr.get("track"):
                st.markdown("**Cosmos3-Edge keyframe perception** (task-free: *where is the object?*)")
                fr = tr["track"]["frames"]
                cols = st.columns(len(fr))
                for col, f, t in zip(cols, fr, tr["track"]["times_s"]):
                    if (ROOT / f["image"]).exists():
                        col.image(str(ROOT / f["image"]), use_container_width=True)
                    col.caption(f"t={t:.1f}s\n{f.get('p_object_location')}{' ✋' if f.get('p_held_by_gripper') else ''}")
                vtype = tr.get("cosmos_failure_type") if isinstance(tr.get("cosmos_failure_type"), str) else None
                v_badge = tag("success/failure", outcome(tr.get("cosmos_task_success")) == outcome(e.success))
                if vtype and vtype != "none" and is_false(tr.get("cosmos_task_success")) and is_false(e.success):
                    v_badge += tag("failure type", vtype == e.failure_type) + '<span class="small">(types unreliable)</span>'
                st.markdown(f"Rule-derived: **{outcome(tr.get('cosmos_task_success'), vtype)}** "
                            f"· sim: **{outcome(e.success, e.failure_type)}** {v_badge} "
                            f"· {tr.get('latency_s', 0):.1f} s for {len(fr)} frames", unsafe_allow_html=True)
                if vtype and vtype != "none":
                    st.caption("⚠ " + ft_caveat("local"))
                held = sum(1 for f in fr if is_true(f.get("p_held_by_gripper")))
                if held and is_false(e.get("grasped")):
                    st.warning(f"Cosmos3-Edge reports the object held by the gripper (✋) in {held}/{len(fr)} keyframes, "
                               f"but the simulator never registered a grasp in this episode (grasped = False throughout).")
        with st.expander("Record a new rollout (real robosuite execution, ~1–2 min)"):
            new_seed = st.number_input("env seed", value=210000, step=1)
            have_ck = ck_id is not None and (ROOT / str(cks[cks.checkpoint_id == ck_id].checkpoint_path.iloc[0])).exists()
            if not have_ck:
                st.caption(f"Disabled: {NEEDS_POOL}.")
            if st.button("▶ Record rollout", key="rec", disabled=not have_ck) and ck_id is not None:
                row = cks[cks.checkpoint_id == ck_id].iloc[0]
                box = st.empty()
                code = run_env_a(["python", "scripts/hackathon/record_rollouts.py", "--checkpoint", f"'{row.checkpoint_path}'",
                                  "--checkpoint-id", ck_id, "--out", "data/outputs/hackathon_live",
                                  "--seeds", f"{int(new_seed)}:{int(new_seed) + 1}", "--batch-size", "1"], box)
                st.cache_data.clear()
                st.success("recorded — reselect the checkpoint to view it" if code == 0 else f"exit {code}")
    with right:
        if seed is not None:
            st.markdown("**Saved diagnostics** (authentic model outputs)")
            d = diag[(diag.checkpoint_id == ck_id) & (diag.env_seed == seed)] if len(diag) else pd.DataFrame()
            if d.empty:
                st.write("unavailable for this episode")
            e = ep[ep.env_seed == seed].iloc[0]
            typed = set()
            for _, r in d.iterrows():
                colr = GREEN if r.backend == "local" else PURPLE
                st.markdown(f'<span class="tag" style="background:{colr}">{r.backend}</span> {r.model_id} · '
                            f'{r.variant} · {fmt(r.latency_s, 1)} s · parse={r.parse_status}', unsafe_allow_html=True)
                # truth-vs-model badge: the verdict next to the simulator label it is scored against
                mtype = r.get("cosmos_failure_type") if isinstance(r.get("cosmos_failure_type"), str) else None
                badge = tag("success/failure", outcome(r.cosmos_task_success) == outcome(e.success))
                if mtype and mtype != "none" and is_false(r.cosmos_task_success) and is_false(e.success):
                    badge += tag("failure type", mtype == e.failure_type) + '<span class="small">(types unreliable)</span>'
                st.markdown(f"model: **{outcome(r.cosmos_task_success, mtype)}** · sim: "
                            f"**{outcome(e.success, e.failure_type)}** {badge}", unsafe_allow_html=True)
                why = grasp_claims(r, e)
                if why:
                    st.warning(f"{JUDGE.get(r.backend, r.backend)} claims the object was grasped ({' and '.join(why)}), "
                               f"but the simulator never registered a grasp in this episode (grasped = False throughout).")
                shown = {"task_success": r.get("cosmos_task_success")}
                if mtype:
                    shown["failure_type (unreliable)"] = mtype
                    if mtype != "none":
                        typed.add(r.backend)
                shown["explanation"] = r.get("cosmos_explanation")
                st.json(shown)
            for b in sorted(typed):
                st.caption("⚠ " + ft_caveat(b))
            video = ROOT / e["video"] if e.get("video") else None
            video = video if (video is not None and video.exists()) else None
            st.divider()
            if video is None:
                st.caption("Live model buttons disabled: this episode's video is not in this checkout.")
            if st.button("▶ Run Cosmos3-Edge now (local GPU)", key="cos", disabled=video is None) and video is not None:
                import imageio.v2 as imageio
                try:
                    frames = [f for f in imageio.get_reader(str(video))]
                    idx = [int(round(x)) for x in np.linspace(0, len(frames) - 1, 8)]
                    tmpd = ROOT / "data/outputs/hackathon_live/keyframes"
                    tmpd.mkdir(parents=True, exist_ok=True)
                    with st.spinner("loading NVIDIA Cosmos3-Edge and judging 8 keyframes…"):
                        m = cosmos_model()
                        parsed, t0 = [], time.time()
                        prompt = build_perception_prompt(e["task"], reasoning=False)
                        for i in idx:
                            pth = tmpd / f"{ck_id}_{seed}_{i}.png"
                            imageio.imwrite(pth, frames[i])
                            res = m.generate(prompt, images=[str(pth)], max_new_tokens=300, reasoning=False)
                            parsed.append(parse_perception(res.raw_text, e["task"]))
                        agg = aggregate_track(e["task"], parsed, [i / float(e["video_fps"]) for i in idx])
                    st.write(f"Cosmos3-Edge ({m.model_id}@{m.revision[:8]}) · 8 keyframes · {time.time() - t0:.1f} s")
                    st.json({"task_success": agg.fields.get("task_success"),
                             "failure_type (unreliable)": agg.fields.get("failure_type"), "per_frame": agg.answer_text})
                except Exception as ex:  # no GPU / weights not downloaded etc.
                    st.error(rel(ex))
            if st.button("▶ Ask Nemotron-3.5-Super-VL on Nebius", key="neb", disabled=video is None) and video is not None:
                from cosmos_plv.nebius_backend import NebiusVLMClient
                try:
                    c = NebiusVLMClient(evidence_path=str(RES / "nebius_calls.jsonl"))
                    with st.spinner("calling Nebius Token Factory…"):
                        res = c.generate(build_prompt(e["task"], 4, 20.1, False, "v2_evidence_first"), video=str(video),
                                         max_tokens=700, tag=f"dashboard/{ck_id}/seed{seed}", pad_video_to_s=20.1)
                    p = parse_diagnostic(res.raw_text)
                    st.write(f"Nebius Token Factory · {res.served_model} · request `{res.request_id}` · "
                             f"{res.latency_s:.1f} s · tokens {res.prompt_tokens}/{res.completion_tokens}")
                    st.json({"parse_status": p.parse_status, **p.fields} if not res.error else {"error": res.error})
                    if not res.error and isinstance(p.fields.get("failure_type"), str):
                        st.caption("⚠ " + ft_caveat("nebius"))
                except Exception as ex:  # missing key etc.
                    st.error(rel(ex))

# ------------------------------------------------------------------ results
with tab_res:
    st.markdown("#### Experiment 1 — cheap metrics vs verified success (within-run Spearman, original PLV scripts)")
    if len(base_rank):
        br = base_rank[(base_rank.run_id == "mean_over_runs") & ~base_rank.metric.str.endswith("_h10")].copy()
        br["subset"] = br.get("subset", "all")
        st.altair_chart(alt.Chart(br).mark_bar().encode(
            y=alt.Y("label:N", sort="-x", title=None, axis=alt.Axis(labelLimit=420)),
            x=alt.X("spearman:Q", title="mean within-run Spearman"),
            color=alt.Color("subset:N"), row=alt.Row("subset:N", title=None),
            tooltip=["label", "subset", "spearman", "regret@30", "n_checkpoints"]).properties(height=170, width=700))
        st.dataframe(base_rank, hide_index=True, use_container_width=True)
    else:
        st.info("baseline_ranking.csv unavailable")
    st.markdown("#### Experiment 2 — can an NVIDIA VLM tell success from failure in a rollout video?")
    if exp2:
        rows = []
        for kname, lst in exp2["overall"].items():
            for r in lst:
                rows.append({"judge": kname, "variant": r["variant"], "episodes": r["episodes"],
                             "accuracy": r["success_acc_all"], "ci95": r["success_acc_ci95"], "balanced": r["balanced_acc_all"],
                             "false pos.": r["fp"], "false neg.": r["fn"], "failure-type acc. (failures)": r["failure_type_acc_on_failures"],
                             "latency s": r["mean_latency_s"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        for b in ("nebius", "local"):
            if FT_KEY[b] in ft_rel:
                st.caption("⚠ " + ft_caveat(b))
        if exp2.get("agreement"):
            a = exp2["agreement"]
            st.caption(f"Cosmos and Nemotron agree on {a['agree_rate']:.0%} of episodes; when they agree accuracy is "
                       f"{a['accuracy_when_agree']:.3f}. Disagreements: Nemotron right {a['nebius_right_when_disagree']}/{a['disagree']}.")
    st.markdown("#### Experiment 3 — hybrid screening (cheap screen → VLM-judged rollouts)")
    if len(ranking):
        st.dataframe(ranking, hide_index=True, use_container_width=True)
    if len(baseline):
        metric = st.selectbox("Scatter: metric vs verified success", [c for c in ("epoch", "val_action_l1", "open_loop_smoothness",
                              "plv1_smoothness", "replan_overlap_h16", "val_loss_logged") if c in baseline])
        st.altair_chart(alt.Chart(baseline).mark_circle(size=60).encode(
            x=alt.X(f"{metric}:Q", scale=alt.Scale(type="log") if metric not in ("epoch",) else alt.Scale()),
            y="success:Q", color="run_id:N", tooltip=["checkpoint_id", metric, "success"]).properties(height=300),
            use_container_width=True)

# ------------------------------------------------------------------ explorations
with tab_exp:
    p = ROOT / "docs/hackathon/11_explorations.md"
    st.markdown(p.read_text() if p.exists() else "unavailable")

# ------------------------------------------------------------------ cost
with tab_cost:
    if cost:
        c = st.columns(3)
        c[0].metric("Nebius tokens (prompt / completion)",
                    f"{cost['nebius_tokens']['prompt']:,} / {cost['nebius_tokens']['completion']:,}" if cost.get("nebius_tokens") else "–")
        c[1].metric("Nebius cost (catalog price)", f"${cost['nebius_usd_estimate']:.3f}" if cost.get("nebius_usd_estimate") is not None else "–")
        c[2].metric("Hardware", "1× RTX 3090 (shared)")
        pcs = pd.DataFrame([{"step": k2, "seconds per checkpoint": v} for k2, v in cost["per_checkpoint_seconds"].items() if v])
        if len(pcs):
            st.altair_chart(alt.Chart(pcs).mark_bar().encode(
                y=alt.Y("step:N", sort="-x", title=None), x=alt.X("seconds per checkpoint:Q", scale=alt.Scale(type="log"))
            ).properties(height=220), use_container_width=True)
    else:
        st.info("cost_latency.json unavailable (run scripts/hackathon/finalize_results.py)")
    if nebius_log:
        calls = pd.DataFrame(nebius_log)
        st.markdown(f"**Nebius Token Factory calls logged:** {len(calls):,} = "
                    + " + ".join(f"{len(v):,} in `{rel(p)}`" for p, v in neb_logs.items())
                    + " (sanitized evidence; no secrets)")
        st.dataframe(calls[[c for c in ("started_utc", "served_model", "request_id", "latency_s", "prompt_tokens",
                                        "completion_tokens", "error", "tag") if c in calls]].tail(200),
                     hide_index=True, use_container_width=True)
