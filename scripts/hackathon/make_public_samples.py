#!/usr/bin/env python3
"""Copy a few real eval checkpoints' rollout videos (re-encoded to 320 px) and their
summary.json into media/hackathon/episodes/, rewriting paths, so the dashboard can
show real rollouts from a fresh clone without the git-ignored data/ tree."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/outputs/hackathon_rollouts_eval"
DST = ROOT / "media/hackathon/episodes"
CKPTS = sys.argv[1:] or ["can_s1_e0000", "can_s1_e0020", "can_s2_e0100", "square_s1_e0030", "square_s1_e0120"]

for ck in CKPTS:
    js = json.loads((SRC / ck / "summary.json").read_text())
    out = DST / ck
    (out / "videos").mkdir(parents=True, exist_ok=True)
    for ep in js["episodes"]:
        src = ROOT / ep["video"]
        dst = out / "videos" / src.name
        frames = [np.asarray(Image.fromarray(f).resize((320, 320), Image.BILINEAR)) for f in imageio.get_reader(str(src))]
        with imageio.get_writer(str(dst), fps=ep["video_fps"], codec="libx264", quality=6, macro_block_size=16,
                                pixelformat="yuv420p") as w:
            for f in frames:
                w.append_data(f)
        ep["video"] = str(dst.relative_to(ROOT))
        ep["render_hw"] = [320, 320]
        ep["episode_npz"] = None  # state traces stay in data/ (not redistributed)
    js["note"] = "public sample: videos re-encoded to 320x320 from the 448x448 originals"
    (out / "summary.json").write_text(json.dumps(js, indent=1, default=float))
    print(ck, len(js["episodes"]))
