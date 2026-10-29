#!/usr/bin/env python3
"""Assemble media/hackathon/demo.mp4 (<= 180 s) from a REAL recorded dashboard
session plus real rollout videos and generated figures.

The edit plan is a JSON list of segments:
  {"kind": "card", "seconds": 6, "title": "...", "body": "..."}
  {"kind": "image", "path": "media/hackathon/architecture.png", "seconds": 8, "caption": "..."}
  {"kind": "clip", "path": "...session.webm", "start": 12.0, "end": 40.0, "speed": 2.0, "caption": "..."}
  {"kind": "sidebyside", "paths": [a.mp4, b.mp4], "labels": [..], "seconds": 10, "caption": "..."}
Speed-ups are announced in the caption ("2x") so nothing is passed off as real time.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
W, H, FPS = 1600, 992, 25  # 992 = 900 dashboard rows + caption bar; divisible by 16
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def font(size, bold=False):
    try:
        return ImageFont.truetype(FONT_B if bold else FONT, size)
    except OSError:
        return ImageFont.load_default()


def fit(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    s = min(W / img.width, (H - 90) / img.height)
    img = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.LANCZOS)
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    canvas.paste(img, ((W - img.width) // 2, (H - 90 - img.height) // 2))
    return canvas


def caption(img: Image.Image, text: str) -> np.ndarray:
    if text:
        d = ImageDraw.Draw(img)
        d.rectangle([0, H - 90, W, H], fill=(20, 20, 20))
        f = font(26)
        lines, cur = [], ""
        for w in text.split():
            if d.textlength(cur + " " + w, font=f) > W - 60:
                lines.append(cur); cur = w
            else:
                cur = (cur + " " + w).strip()
        lines.append(cur)
        for i, ln in enumerate(lines[:2]):
            d.text((30, H - 80 + i * 36), ln, font=f, fill=(255, 255, 255))
    return np.asarray(img)


def card(title, body, seconds):
    img = Image.new("RGB", (W, H), (18, 24, 33))
    d = ImageDraw.Draw(img)
    d.text((80, 260), title, font=font(54, True), fill=(118, 185, 0))
    y = 360
    for para in body.split("\n"):
        d.text((80, y), para, font=font(30), fill=(235, 235, 235)); y += 50
    fr = np.asarray(img)
    return [fr] * int(seconds * FPS)


def clip(path, start, end, speed, text):
    r = imageio.get_reader(str(ROOT / path) if not Path(path).is_absolute() else path)
    src_fps = r.get_meta_data().get("fps", FPS) or FPS
    out, step = [], speed * src_fps / FPS
    i0, i1 = int(start * src_fps), int(end * src_fps)
    nxt = float(i0)
    for i, f in enumerate(r):
        if i < i0:
            continue
        if i > i1:
            break
        while i >= nxt:
            out.append(caption(fit(Image.fromarray(f)), text)); nxt += step
    r.close()
    return out


def side_by_side(paths, labels, seconds, text):
    readers = [[f for f in imageio.get_reader(str(ROOT / p))] for p in paths]
    n = int(seconds * FPS)
    out = []
    for k in range(n):
        tiles = []
        for fr, lab in zip(readers, labels):
            im = Image.fromarray(fr[min(len(fr) - 1, int(k / n * len(fr)))]).resize((700, 700))
            ImageDraw.Draw(im).rectangle([0, 0, 700, 46], fill=(20, 20, 20))
            ImageDraw.Draw(im).text((12, 8), lab, font=font(24, True), fill=(255, 255, 255))
            tiles.append(im)
        canvas = Image.new("RGB", (W, H), (255, 255, 255))
        for j, t in enumerate(tiles):
            canvas.paste(t, (60 + j * 760, 60))
        out.append(caption(canvas, text))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", default=str(ROOT / "media/hackathon/demo.mp4"))
    a = ap.parse_args()
    plan = json.loads(Path(a.plan).read_text())
    budget = sum(seg.get("seconds", 0) for seg in plan) + sum(
        (seg["end"] - seg["start"]) / seg.get("speed", 1.0) for seg in plan if seg["kind"] == "clip")
    if budget > 180:
        raise SystemExit(f"plan too long: {budget:.1f}s > 180s")
    n = 0
    with imageio.get_writer(a.out, fps=FPS, codec="libx264", quality=7, macro_block_size=16,
                            pixelformat="yuv420p") as w:
        for seg in plan:
            k = seg["kind"]
            if k == "card":
                fr = card(seg["title"], seg.get("body", ""), seg["seconds"])
            elif k == "image":
                fr = [caption(fit(Image.open(ROOT / seg["path"])), seg.get("caption", ""))] * int(seg["seconds"] * FPS)
            elif k == "clip":
                fr = clip(seg["path"], seg["start"], seg["end"], seg.get("speed", 1.0), seg.get("caption", ""))
            elif k == "sidebyside":
                fr = side_by_side(seg["paths"], seg["labels"], seg["seconds"], seg.get("caption", ""))
            else:
                raise SystemExit(f"unknown segment kind {k}")
            for f in fr:
                w.append_data(f)
            n += len(fr)
            print(f"{k:10s} -> total {n / FPS:6.1f}s", flush=True)
    print(f"wrote {a.out} ({n / FPS:.1f}s)")
    if n / FPS > 180:
        raise SystemExit("video longer than 180 s")


if __name__ == "__main__":
    main()
