#!/usr/bin/env python3
"""y3_visual_robustness Step 3 (Environment B, CPU): image-quality degradations of the
ORIGINAL agentview frames (decoded from the recorded mp4).  EXPLORATORY.

Fixed parameters (set before any judge output was seen):
  down224  448 -> 224 (PIL bilinear, antialiased) -> 448 (bilinear)
  down112  448 -> 112 -> 448
  jpeg20   JPEG quality 20 (PIL), decoded back
  blur     Gaussian blur, PIL radius (sigma) 1.5 px
  dim      brightness x0.75 then contrast x0.75 (PIL ImageEnhance)
Each condition writes <out>/<cond>/<ckpt>/seed<seed>.mp4 (same encoder settings as the
recorder; every frame degraded) and seed<seed>_last.png = the degraded LAST decoded
frame (the v6 input; PNG so no further loss).  The 'orig' condition links the original
last frame (decoded like run_cosmos_diagnostics.frames_png) and the original mp4.

  PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/y3_degrade.py
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def down(img: Image.Image, s: int) -> Image.Image:
    w, h = img.size
    return img.resize((s, s), Image.BILINEAR).resize((w, h), Image.BILINEAR)


def jpeg(img: Image.Image, q: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


DEGRADATIONS = {
    "down224": lambda im: down(im, 224),
    "down112": lambda im: down(im, 112),
    "jpeg20": lambda im: jpeg(im, 20),
    "blur": lambda im: im.filter(ImageFilter.GaussianBlur(radius=1.5)),
    "dim": lambda im: ImageEnhance.Contrast(ImageEnhance.Brightness(im).enhance(0.75)).enhance(0.75),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default=str(ROOT / "results/hackathon/explore/y3_visual_robustness_episodes.json"))
    ap.add_argument("--out", default=str(ROOT / "data/outputs/hackathon_explore/y3_visual_robustness/degrade"))
    args = ap.parse_args()
    import imageio.v2 as imageio

    from cosmos_plv.video_recorder import read_mp4, write_mp4

    eps = json.loads(Path(args.episodes).read_text())["episodes"]
    out = Path(args.out)
    manifest = []
    for ep in eps:
        frames = read_mp4(ROOT / ep["video"])
        stem = f"seed{ep['env_seed']}"
        d = out / "orig" / ep["checkpoint_id"]
        d.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(d / f"{stem}_last.png", frames[-1])
        manifest.append({"condition": "orig", "checkpoint_id": ep["checkpoint_id"], "env_seed": ep["env_seed"],
                         "mp4": ep["video"], "last_png": str((d / f"{stem}_last.png").relative_to(ROOT))})
        for name, fn in DEGRADATIONS.items():
            d = out / name / ep["checkpoint_id"]
            d.mkdir(parents=True, exist_ok=True)
            deg = [np.asarray(fn(Image.fromarray(f).convert("RGB"))) for f in frames]
            write_mp4(d / f"{stem}.mp4", deg, fps=10)
            imageio.imwrite(d / f"{stem}_last.png", deg[-1])
            manifest.append({"condition": name, "checkpoint_id": ep["checkpoint_id"], "env_seed": ep["env_seed"],
                             "mp4": str((d / f"{stem}.mp4").relative_to(ROOT)),
                             "last_png": str((d / f"{stem}_last.png").relative_to(ROOT))})
        print(ep["checkpoint_id"], ep["env_seed"], flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("manifest rows", len(manifest))


if __name__ == "__main__":
    main()
