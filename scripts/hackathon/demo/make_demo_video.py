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
W, H, FPS = 1600, 992, 25  # default: 992 = 900 dashboard rows + caption bar; divisible by 16 (--size/--fps override)
CAP = 90  # caption bar height, scaled with --size
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def font(size, bold=False):
    try:
        return ImageFont.truetype(FONT_B if bold else FONT, size)
    except OSError:
        return ImageFont.load_default()


def fit(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    top = int(64 * W / 1600) if TAG else 0  # keep the provenance tag clear of the picture
    s = min(W / img.width, (H - CAP - top) / img.height)
    img = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.LANCZOS)
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    canvas.paste(img, ((W - img.width) // 2, top + (H - CAP - top - img.height) // 2))
    return canvas


TAG = ""  # per-segment provenance label drawn top-left (e.g. "LIVE screen recording · 2x"), set in main()
STRIP = False  # --strip: always draw the dark caption strip (narration subtitles are burned into it later)


def caption(img: Image.Image, text: str) -> np.ndarray:
    if TAG:
        d = ImageDraw.Draw(img)
        f = font(int(22 * W / 1600), True)
        tw = d.textlength(TAG, font=f)
        d.rounded_rectangle([14, 14, 14 + tw + 28, 14 + int(40 * W / 1600)], radius=8, fill=(20, 20, 20))
        d.text((28, 20), TAG, font=f, fill=(255, 255, 255))
    if STRIP and not text:
        ImageDraw.Draw(img).rectangle([0, H - CAP, W, H], fill=(20, 20, 20))
    if text:
        d = ImageDraw.Draw(img)
        d.rectangle([0, H - CAP, W, H], fill=(20, 20, 20))
        f = font(int(26 * W / 1600))
        lines, cur = [], ""
        for w in text.split():
            if d.textlength(cur + " " + w, font=f) > W - 60:
                lines.append(cur); cur = w
            else:
                cur = (cur + " " + w).strip()
        lines.append(cur)
        for i, ln in enumerate(lines[:2]):
            d.text((30, H - CAP + 10 + i * int(36 * W / 1600)), ln, font=f, fill=(255, 255, 255))
    return np.asarray(img)


def card(title, body, seconds):
    img = Image.new("RGB", (W, H), (18, 24, 33))
    d = ImageDraw.Draw(img)
    k = W / 1600
    d.text((int(80 * k), int(260 * k)), title, font=font(int(54 * k), True), fill=(118, 185, 0))
    y = int(360 * k)
    for para in body.split("\n"):
        d.text((int(80 * k), y), para, font=font(int(30 * k)), fill=(235, 235, 235)); y += int(50 * k)
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
            ts = min((W - 180) // 2, H - CAP - 100)
            im = Image.fromarray(fr[min(len(fr) - 1, int(k / n * len(fr)))]).resize((ts, ts))
            ImageDraw.Draw(im).rectangle([0, 0, ts, 46], fill=(20, 20, 20))
            ImageDraw.Draw(im).text((12, 8), lab, font=font(24, True), fill=(255, 255, 255))
            tiles.append(im)
        canvas = Image.new("RGB", (W, H), (255, 255, 255))
        x0 = (W - 2 * ts - 60) // 2
        for j, t in enumerate(tiles):
            canvas.paste(t, (x0 + j * (ts + 60), (H - CAP - ts) // 2))
        out.append(caption(canvas, text))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", default=str(ROOT / "media/hackathon/demo.mp4"))
    ap.add_argument("--size", default="", help="WxH, e.g. 1920x1080 (default 1600x992)")
    ap.add_argument("--fps", type=int, default=0)
    ap.add_argument("--srt", default="", help="also write the captions as an .srt subtitle file")
    ap.add_argument("--strip", action="store_true", help="always draw the caption strip (for burned-in narration subtitles)")
    a = ap.parse_args()
    global W, H, FPS, CAP, TAG, STRIP
    STRIP = a.strip
    if a.size:
        W, H = (int(x) for x in a.size.lower().split("x"))
        CAP = int(90 * H / 992) if H != 992 else 90
    if a.fps:
        FPS = a.fps
    plan = json.loads(Path(a.plan).read_text())
    budget = sum(seg.get("seconds", 0) for seg in plan) + sum(
        (seg["end"] - seg["start"]) / seg.get("speed", 1.0) for seg in plan if seg["kind"] == "clip")
    if budget > 180:
        raise SystemExit(f"plan too long: {budget:.1f}s > 180s")
    n = 0
    cues, times = [], []
    with imageio.get_writer(a.out, fps=FPS, codec="libx264", quality=7, macro_block_size=8,
                            pixelformat="yuv420p") as w:
        for seg in plan:
            k = seg["kind"]
            TAG = seg.get("tag", "")
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
            if seg.get("caption"):
                cues.append((n / FPS, (n + len(fr)) / FPS, seg["caption"]))
            times.append({"index": len(times), "kind": k, "start_s": n / FPS, "end_s": (n + len(fr)) / FPS})
            n += len(fr)
            print(f"{k:10s} -> total {n / FPS:6.1f}s", flush=True)
    print(f"wrote {a.out} ({n / FPS:.1f}s)")
    Path(a.out).with_suffix(".timing.json").write_text(json.dumps(times, indent=1))  # per-segment times (narration)
    if a.srt:
        ts = lambda x: f"{int(x // 3600):02d}:{int(x % 3600 // 60):02d}:{int(x % 60):02d},{int(round(x % 1 * 1000)) % 1000:03d}"  # noqa: E731
        Path(a.srt).write_text("".join(f"{i}\n{ts(t0)} --> {ts(t1)}\n{txt}\n\n" for i, (t0, t1, txt) in enumerate(cues, 1)))
        print(f"wrote {a.srt} ({len(cues)} cues)")
    if n / FPS > 180:
        raise SystemExit("video longer than 180 s")


if __name__ == "__main__":
    main()
