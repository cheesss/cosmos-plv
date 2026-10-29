#!/usr/bin/env python3
"""Experiment 4b (exploratory) — analysis of the e5_probe stages (Environment C, CPU only).

S0  reproduction of the model card's UMI output and action sensitivity vs seed noise.
S1  slot 8 axis sweeps: track the gripper from the true EEF pixel; per axis i
      r_i   = (d(+i) - d(-i)) / 2          antisymmetric (action-following) response
      sym_i = (d(+i) + d(-i)) / 2 - d(hold) symmetric part (not explained by the sign)
    and fit the action frame M over the 24 axis-aligned rotations: predicted image
    displacement p_i(M) = proj(P + 0.16 R_site M e_i) - proj(P).
S2  untrained slots 5 vs 9 with the LIBERO encoding: agreement of their r_i, and
    agreement with the LIBERO world-frame geometry.
S3  slot 8 (R_q, fitted frame) on real replays: generated vs projected-true EEF
    displacement over the chunk; drift-cancelled direction cos(d_main - d_rev, d_real);
    seed floor |d_main - d_main_seed2|; PSNR vs the frozen-frame baseline.
Pass for S1 (fixed before running): best mean cos >= 0.7, median |r_i| >= 8 px and
median |sym_i| < median |r_i|.
Pass for S3 (written about 20 s after S3 was launched, before any S3 output was read):
the 95% CI of cos(d_main - d_rev, d_real) at K=16 on moving units (true EEF image
displacement >= 10 px) is above 0 for all three trackers, with a unit-level bootstrap
resampling the 5 initial-state families (task, env seed). Family-level statistics (mean
of family means, t(4) interval, exact sign-flip p; its floor is 1/32 with 5 families) and
task / start splits are reported next to it (added after the E4b verification).
The 'reversed' arm replays the path backwards from the current pose, so at K=8 the
drift-cancelled target is proj(P16) - proj(P0), not the K=8 real displacement.
Output: results/hackathon/explore/e5_probe.json
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from e5_track import cos, track_flow, track_template  # noqa: E402

OUT = ROOT / "data/outputs/hackathon_explore/e5_probe"
REPLAY = ROOT / "data/outputs/hackathon_explore/e4_cosmos_fd/replay"
SNAP = Path.home() / ".cache/huggingface/hub/models--nvidia--Cosmos3-Edge/snapshots/344d602b128d1bbdacb43b08d0a3626f46343e29"
S_START = 32
TRACKERS = {"farn": lambda v, p: track_flow(v, p, method="farneback", h=12, weight="mag"),
            "dis": lambda v, p: track_flow(v, p, method="dis", h=12, weight="mag"),
            "tmpl": lambda v, p: track_template(v, p, half=20, search=12)}
AXES = ("x", "y", "z")


def psnr(a, b):
    m = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(255.0 ** 2 / max(m, 1e-9))


def proj(T, P):
    h = np.r_[P, 1.0] @ T.T
    return np.array([h[1] / h[2], h[0] / h[2]])


def rotations24(proper=True):
    """The 24 axis-aligned rotations (proper=True) or all 48 signed permutations."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3))
            for i, (p, s) in enumerate(zip(perm, signs)):
                M[p, i] = s
            if not proper or np.isclose(np.linalg.det(M), 1):
                out.append(M)
    return out


def jobs():
    return [json.loads(l) for l in (OUT / "jobs.jsonl").read_text().splitlines()]


def vid(j):
    return np.load(ROOT / j["file"])["video"]


def s0(js):
    import cv2
    J = {(j["arm"], j["seed"]): vid(j) for j in js if j["stage"] == "S0"}
    if not J:
        return None
    cap = cv2.VideoCapture(str(SNAP / "assets/diffusers_outputs/edge_action_fd_diffusers_chunk_00.mp4"))
    ref = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        ref.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    ref = np.asarray(ref)
    m0 = J[("main", 0)]
    n = min(len(ref), len(m0))
    if ref.shape[1:3] != m0.shape[1:3]:
        ref = np.asarray([cv2.resize(f, (m0.shape[2], m0.shape[1]), interpolation=cv2.INTER_AREA) for f in ref])

    def mad(a, b):
        return float(np.mean(np.abs(a[1:].astype(np.float64) - b[1:].astype(np.float64))))
    seeds = sorted(sd for (arm, sd) in J if arm == "main")
    seed_noise = float(np.mean([mad(J[("main", a)], J[("main", b)]) for a, b in itertools.combinations(seeds, 2)]))
    res = {"psnr_main0_vs_card_output": float(np.mean([psnr(m0[k], ref[k]) for k in range(1, n)])),
           "psnr_main0_vs_main1": float(np.mean([psnr(m0[k], J[("main", 1)][k]) for k in range(1, n)])),
           "seed_noise_mad": seed_noise,
           "action_effect_mad": {arm: float(np.mean([mad(J[("main", sd)], J[(arm, sd)]) for sd in seeds if (arm, sd) in J]))
                                 for arm in ("negpos", "hold", "other")}}
    res["pass"] = bool(res["psnr_main0_vs_card_output"] >= 30 and
                       all(res["action_effect_mad"][a] >= 2 * seed_noise for a in ("negpos", "hold")))
    return res


def responses(js, stage, frame_key):
    """Per (unit, enc, tracker): d(arm) at k=16 and pixel-level |video(+i)-video(-i)|."""
    by = {}
    for j in js:
        if j["stage"] == stage:
            by.setdefault((j["unit"], j["enc"]), {})[j["arm"]] = j
    rows = []
    for (unit, enc), arms in sorted(by.items()):
        if len(arms) < 7:
            continue
        d = np.load(REPLAY / f"{unit}.npz", allow_pickle=True)
        p0 = d["eef_px_raw"][frame_key].astype(float)
        V = {a: vid(j) for a, j in arms.items()}
        for tn, fn in TRACKERS.items():
            disp = {a: (lambda tr: tr[16] - tr[0])(fn(v, p0)) for a, v in V.items()}
            for i, ax in enumerate(AXES):
                r = (disp["+" + ax] - disp["-" + ax]) / 2
                sym = (disp["+" + ax] + disp["-" + ax]) / 2 - disp["hold"]
                rows.append({"unit": unit, "enc": enc, "tracker": tn, "axis": i, "r": r.tolist(), "sym": sym.tolist(),
                             "pix_pm": float(np.mean(np.abs(V["+" + ax][1:].astype(float) - V["-" + ax][1:].astype(float))))})
    return rows


def s1(js):
    rows = responses(js, "S1", S_START)
    if not rows:
        return None
    R24 = rotations24()
    out = {}
    for enc in sorted({r["enc"] for r in rows}):
        res = {}
        for tn in TRACKERS:
            q = [r for r in rows if r["enc"] == enc and r["tracker"] == tn]
            scores = []
            for M in R24:
                cs = []
                for r in q:
                    d = np.load(REPLAY / f"{r['unit']}.npz", allow_pickle=True)
                    T, P, Rs = d["cam_T"], d["eef_pos"][S_START], d["eef_mat"][S_START]
                    p = proj(T, P + 0.16 * Rs @ M[:, r["axis"]]) - proj(T, P)
                    cs.append(cos(np.asarray(r["r"]), p))
                scores.append(float(np.nanmean(cs)))
            order = np.argsort(scores)[::-1]
            rmag = [np.linalg.norm(r["r"]) for r in q]
            smag = [np.linalg.norm(r["sym"]) for r in q]
            res[tn] = {"best_cos": scores[order[0]], "second_cos": scores[order[1]], "best_M": R24[order[0]].tolist(),
                       "median_abs_r_px": float(np.median(rmag)), "median_abs_sym_px": float(np.median(smag)),
                       "pix_plusminus_mad": float(np.mean([r["pix_pm"] for r in q]))}
            res[tn]["pass"] = bool(res[tn]["best_cos"] >= 0.7 and res[tn]["median_abs_r_px"] >= 8
                                   and res[tn]["median_abs_sym_px"] < res[tn]["median_abs_r_px"])
            sc48 = []
            for M in rotations24(proper=False):  # reflections too: the 3-D frame is not identified by S1 alone
                cs = []
                for r in q:
                    d = np.load(REPLAY / f"{r['unit']}.npz", allow_pickle=True)
                    T, P, Rs = d["cam_T"], d["eef_pos"][S_START], d["eef_mat"][S_START]
                    cs.append(cos(np.asarray(r["r"]), proj(T, P + 0.16 * Rs @ M[:, r["axis"]]) - proj(T, P)))
                sc48.append(float(np.nanmean(cs)))
            o48 = np.argsort(sc48)[::-1]
            res[tn]["best_cos_incl_reflections"] = sc48[o48[0]]
            res[tn]["second_cos_incl_reflections"] = sc48[o48[1]]
        Ms = {json.dumps(v["best_M"]) for v in res.values()}
        out[enc] = {"trackers": res, "best_M_agrees_across_trackers": len(Ms) == 1,
                    "pass": all(v["pass"] for v in res.values()) and len(Ms) == 1}
    return {"by_encoding": out, "rows": rows}


def s2(js):
    rows = responses(js, "S2", S_START)
    if not rows:
        return None
    res = {}
    for tn in TRACKERS:
        q = {(r["unit"], r["axis"], r["enc"]): r for r in rows if r["tracker"] == tn}
        agree, geo5, geo9, mag = [], [], [], {"libero": [], "libero_slot9": []}
        for (unit, ax, enc), r in q.items():
            mag[enc].append(np.linalg.norm(r["r"]))
            if enc != "libero":
                continue
            r9 = q.get((unit, ax, "libero_slot9"))
            d = np.load(REPLAY / f"{unit}.npz", allow_pickle=True)
            T, P = d["cam_T"], d["eef_pos"][S_START]
            p = proj(T, P + 0.32 * np.eye(3)[ax]) - proj(T, P)  # LIBERO deltas are world-frame, 2 cm x 16
            geo5.append(cos(np.asarray(r["r"]), p))
            if r9 is not None:
                agree.append(cos(np.asarray(r["r"]), np.asarray(r9["r"])))
                geo9.append(cos(np.asarray(r9["r"]), p))
        res[tn] = {"cos_slot5_vs_slot9": float(np.nanmean(agree)), "cos_slot5_vs_geometry": float(np.nanmean(geo5)),
                   "cos_slot9_vs_geometry": float(np.nanmean(geo9)),
                   "median_abs_r_px": {k: float(np.median(v)) for k, v in mag.items()}, "n": len(agree)}
    return {"trackers": res, "rows": rows}


def family(ep):
    task, seed = ep.split("_")[0], ep.split("_")[-1]
    return f"{task}_{seed}"


def s3(js):
    by = {}
    for j in js:
        if j["stage"] == "S3":
            by.setdefault(j["unit"], {})[j["arm"]] = j
    rows = []
    for unit, arms in sorted(by.items()):
        if not {"main", "zero", "reversed"} <= set(arms):
            continue
        ep, s = unit.split("|")
        d = np.load(REPLAY / f"{ep}.npz", allow_pickle=True)
        idx = np.asarray(arms["main"]["real_idx"])
        px = d["eef_px_raw"].astype(float)
        p0 = px[idx[0]]
        real = d["frames"][idx]
        V = {a: vid(j) for a, j in arms.items()}
        for tn, fn in TRACKERS.items():
            tr = {a: fn(v, p0) for a, v in V.items()}
            for K in (8, 16):
                dr = px[idx[K]] - p0
                dmr = px[idx[16]] - p0  # target of d_main - d_rev at any K (reversed starts from the path end)
                dg = {a: t[K] - t[0] for a, t in tr.items()}
                row = {"unit": unit, "episode": ep, "family": family(ep), "start": int(s), "tracker": tn, "K": K,
                       "real_px": float(np.linalg.norm(dr)),
                       "cos_main": cos(dg["main"], dr), "cos_rev": cos(dg["reversed"], dr), "cos_zero": cos(dg["zero"], dr),
                       "cos_main_minus_rev": cos(dg["main"] - dg["reversed"], dmr),
                       "gain_main_minus_rev": float((dg["main"] - dg["reversed"]) @ dmr / 2 / max(dmr @ dmr, 1e-9)) if K == 16 else float("nan"),
                       "end_err_main": float(np.linalg.norm(tr["main"][K] - px[idx[K]])),
                       "end_err_zero": float(np.linalg.norm(tr["zero"][K] - px[idx[K]])),
                       "end_err_frozen": float(np.linalg.norm(dr)),
                       "main_minus_rev_px": float(np.linalg.norm(dg["main"] - dg["reversed"]))}
                if "main_seed2" in dg:
                    row["seed_floor_px"] = float(np.linalg.norm(dg["main"] - dg["main_seed2"]))
                if "gripflip" in dg:
                    row["gripflip_px"] = float(np.linalg.norm(dg["main"] - dg["gripflip"]))
                rows.append(row)
        for a, v in V.items():
            n = min(len(v), len(real))
            rows.append({"unit": unit, "episode": ep, "family": family(ep), "start": int(s), "tracker": "psnr", "arm": a,
                         "psnr_full": float(np.mean([psnr(v[k], real[k]) for k in range(1, n)])),
                         "psnr_frozen": float(np.mean([psnr(real[0], real[k]) for k in range(1, n)]))})
    if not rows:
        return None
    import pandas as pd
    df = pd.DataFrame(rows)
    rng = np.random.default_rng(0)

    def boot(q, col):
        fams = q.family.unique()
        vals = [np.nanmean(np.concatenate([q[q.family == f][col].values for f in rng.choice(fams, len(fams))])) for _ in range(4000)]
        return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]
    summ = {}
    for tn in TRACKERS:
        for K in (8, 16):
            q = df[(df.tracker == tn) & (df.K == K)]
            mv = q[q.real_px >= 10]
            ent = {"units": int(len(q)), "moving_units": int(len(mv))}
            for col in ("cos_main", "cos_rev", "cos_zero", "cos_main_minus_rev", "gain_main_minus_rev"):
                ent[col] = float(mv[col].mean())
                ent[col + "_ci"] = boot(mv, col)
            for col in ("end_err_main", "end_err_zero", "end_err_frozen", "main_minus_rev_px", "seed_floor_px", "gripflip_px"):
                ent[col] = float(mv[col].mean()) if col in mv else None
            summ[f"{tn}_K{K}"] = ent
    fam = {}
    from scipy import stats
    for tn in TRACKERS:
        mv = df[(df.tracker == tn) & (df.K == 16) & (df.real_px >= 10)]
        fm = mv.groupby("family").cos_main_minus_rev.mean()
        n = len(fm)
        signs = np.array(list(itertools.product((1, -1), repeat=n)))
        null = (signs * np.abs(fm.values)).mean(1)
        lo = fm.mean() - stats.t.ppf(0.975, n - 1) * fm.std(ddof=1) / np.sqrt(n)
        fam[tn] = {"family_means": fm.round(3).to_dict(), "mean_of_family_means": float(fm.mean()),
                   "t_ci_lower": float(lo), "sign_flip_p_one_sided": float((null >= fm.mean() - 1e-12).mean()),
                   "by_task": mv.assign(task=mv.episode.str.split("_").str[0]).groupby("task").cos_main_minus_rev.mean().round(3).to_dict(),
                   "by_start": mv.groupby("start").cos_main_minus_rev.mean().round(3).to_dict(),
                   "end_err_by_task": mv.assign(task=mv.episode.str.split("_").str[0]).groupby("task")[["end_err_main", "end_err_zero", "end_err_frozen"]].mean().round(1).to_dict(),
                   "all_units_end_err": df[(df.tracker == tn) & (df.K == 16)][["end_err_main", "end_err_zero", "end_err_frozen"]].mean().round(1).to_dict(),
                   "matched_s16_main_minus_rev_px": float(df[(df.tracker == tn) & (df.K == 16) & (df.start == 16) & (df.real_px >= 10)].main_minus_rev_px.mean()),
                   "matched_s16_seed_floor_px": float(df[(df.tracker == tn) & (df.K == 16) & (df.start == 16) & (df.real_px >= 10)].seed_floor_px.mean())}
    ps = df[df.tracker == "psnr"]
    psnr_tab = ps.groupby("arm")[["psnr_full", "psnr_frozen"]].mean().round(3).reset_index().to_dict("records")
    passed = all(summ[f"{tn}_K16"]["cos_main_minus_rev_ci"][0] > 0 for tn in TRACKERS)
    df.to_csv(ROOT / "results/hackathon/explore/e5_probe_s3_units.csv", index=False)
    psnr_start = ps.groupby(["start", "arm"])[["psnr_full", "psnr_frozen"]].mean().round(3).reset_index().to_dict("records")
    return {"summary": summ, "psnr": psnr_tab, "psnr_by_start": psnr_start, "family_level": fam, "pass": bool(passed),
            "families": sorted(df.family.unique().tolist())}


def main():
    js = jobs()
    res = {"label": "EXPLORATORY (post-hoc probes after the E4 null; S0-S2 criteria fixed before running, "
                    "S3 criteria fixed before any S3 output was read)",
           "n_jobs": len(js), "mean_gen_seconds": float(np.mean([j["seconds"] for j in js])),
           "S0": s0(js), "S1": s1(js), "S2": s2(js), "S3": s3(js)}
    out = ROOT / "results/hackathon/explore/e5_probe.json"
    out.write_text(json.dumps(res, indent=1))
    show = {k: v for k, v in res.items() if k not in ("S1", "S2")}
    print(json.dumps(show, indent=1))
    for st in ("S1", "S2"):
        if res[st]:
            print(st, json.dumps(res[st].get("by_encoding", res[st].get("trackers")), indent=1))


if __name__ == "__main__":
    main()
