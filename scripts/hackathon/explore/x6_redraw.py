#!/usr/bin/env python3
"""x6_visual_plan_critic, stage 1b (Environment B, CPU): redraw plan overlays on
validated reference frames.

Why: re-using one offscreen-rendering env per task inside one process produced
corrupted (magenta noise) or vertically flipped agentview frames after the
first checkpoint, although the simulator state was bit-identical.  The initial
state for a given env seed is identical for every checkpoint (same seeded
reset), so each overlay is redrawn on the FIRST rendered frame of that seed
(can_s1_e0000 / square_s1_e0000), whose projection was checked on debug images
(EEF and object markers land on the gripper and the object).  The script
asserts eef0 / obj0 / camera matrix equality before drawing.

  PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/x6_redraw.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/hackathon/explore"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from x6_plan_render import draw_paths  # noqa: E402

NAME = "x6_visual_plan_critic"
PLANS = ROOT / f"data/outputs/hackathon_explore/{NAME}/plans"
CLEAN = ROOT / f"data/outputs/hackathon_explore/{NAME}/plans_clean"
REF = {"can": "can_s1_e0000", "square": "square_s1_e0000"}


def main():
    n_new = n_skip = 0
    for cdir in sorted(p for p in PLANS.iterdir() if p.is_dir()):
        task = cdir.name.split("_")[0]
        for npz in sorted(cdir.glob("seed*.npz")):
            out1 = CLEAN / cdir.name / f"{npz.stem}_plan1.png"
            d = np.load(npz)
            n_chunks = d["action_pred"].shape[0]
            out3 = CLEAN / cdir.name / f"{npz.stem}_plan{n_chunks}.png"
            if out1.is_file() and (n_chunks == 1 or out3.is_file()):
                n_skip += 1
                continue
            ref_npz = PLANS / REF[task] / npz.name
            ref_png = PLANS / REF[task] / f"{npz.stem}_raw.png"
            if not (ref_npz.is_file() and ref_png.is_file()):
                print(f"[x6-redraw] no reference frame for {task} {npz.stem}; skip")
                continue
            r = np.load(ref_npz)
            for k in ("eef0", "obj0", "cam_P"):
                if not np.allclose(d[k], r[k], atol=1e-6):
                    raise SystemExit(f"[x6-redraw] {cdir.name}/{npz.name}: {k} differs from reference")
            img0 = np.asarray(Image.open(ref_png).convert("RGB"))
            pos_len = d["action_pred"].shape[1]
            out1.parent.mkdir(parents=True, exist_ok=True)
            draw_paths(img0, [d["pix_plan1"]], [np.arange(pos_len, dtype=float)], float(pos_len - 1)).save(out1)
            if n_chunks > 1:
                n_obs = 2
                t_idx = [np.asarray(d["plan_t"][c] + np.arange(pos_len) - (n_obs - 1), dtype=float)
                         for c in range(n_chunks)]
                tmin = min(x.min() for x in t_idx)
                t_idx = [x - tmin for x in t_idx]
                tmax = max(x.max() for x in t_idx)
                pix = d["pix_all"]
                draw_paths(img0, [pix[c * pos_len:(c + 1) * pos_len] for c in range(n_chunks)], t_idx,
                           float(tmax)).save(out3)
            n_new += 1
    print(f"[x6-redraw] drew {n_new} seeds, {n_skip} already present -> {CLEAN}")


if __name__ == "__main__":
    main()
