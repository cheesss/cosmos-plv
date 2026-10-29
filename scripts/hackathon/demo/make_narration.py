#!/usr/bin/env python3
"""Synthetic English narration with subtitles synced to the speech (TTS environment: ~/.venvs/tts: kokoro, soundfile).

  synth  speak every line of configs/hackathon/demo_narration.json locally with Kokoro-82M (Apache-2.0) and write
         submission/narration/line_XX.wav plus durations.json (build_demo_plan.py sizes segments from it)
  mux    place each line 0.3 s into its first segment (times from <video>.timing.json, written by make_demo_video.py),
         write sentence-level subtitles timed to the speech, burn them into the caption strip below the picture, and
         mux a loudness-normalized AAC track:  make_narration.py mux --video X_silent.mp4 --out X.mp4
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[3]
CFG = ROOT / "configs/hackathon/demo_narration.json"
NAR = ROOT / "submission/narration"
SR = 24000
LEAD = 0.3


def synth():
    from kokoro import KPipeline

    cfg = json.loads(CFG.read_text())
    NAR.mkdir(parents=True, exist_ok=True)
    pipe = KPipeline(lang_code="a")
    out = []
    for i, ln in enumerate(cfg["lines"]):
        speed = ln.get("speed", 1.0)
        audio = np.concatenate([np.asarray(r.audio, dtype=np.float32) for r in pipe(ln["say"], voice=cfg["voice"], speed=speed)])
        sf.write(NAR / f"line_{i:02d}.wav", audio, SR)
        out.append({"line": i, "segments": ln["segments"], "duration_s": round(len(audio) / SR, 3), "speed": speed})
        print(out[-1], flush=True)
    (NAR / "durations.json").write_text(json.dumps({"voice": cfg["voice"], "model": "hexgrad/Kokoro-82M", "lines": out}, indent=1))


def sentences(text):
    parts = re.split(r"(?<=[.?!:])\s+", text.strip())
    return [p for p in parts if p]


def srt_time(x):
    return f"{int(x // 3600):02d}:{int(x % 3600 // 60):02d}:{int(x % 60):02d},{int(round(x % 1 * 1000)) % 1000:03d}"


def mux(video, out):
    cfg = json.loads(CFG.read_text())
    dur = {d["line"]: d for d in json.loads((NAR / "durations.json").read_text())["lines"]}
    times = json.loads(Path(video).with_suffix(".timing.json").read_text())
    total = times[-1]["end_s"]
    track = np.zeros(int((total + 1) * SR), dtype=np.float32)
    cues, report = [], []
    for i, ln in enumerate(cfg["lines"]):
        t0 = times[ln["segments"][0]]["start_s"] + LEAD
        t_end = times[ln["segments"][-1]]["end_s"]
        audio, _ = sf.read(NAR / f"line_{i:02d}.wav", dtype="float32")
        d = len(audio) / SR
        k = int(t0 * SR)
        track[k:k + len(audio)] += audio[: len(track) - k]
        report.append({"line": i, "start_s": round(t0, 2), "end_s": round(t0 + d, 2), "segment_end_s": round(t_end, 2),
                       "fits": t0 + d <= t_end + 1e-6})
        # sentence cues: split the line's speech time in proportion to the spoken characters of each sentence
        shown, said = sentences(ln["text"]), sentences(ln["say"])
        if len(shown) != len(said):
            shown, said = [ln["text"]], [ln["say"]]
        w = np.array([len(s) for s in said], float)
        edges = t0 + d * np.r_[0, np.cumsum(w) / w.sum()]
        for s, a, b in zip(shown, edges[:-1], edges[1:]):
            cues.append((a, b, s))
    for r in report:
        print(r)
    srt = Path(out).with_suffix(".srt")
    srt.write_text("".join(f"{j}\n{srt_time(a)} --> {srt_time(b)}\n{s}\n\n" for j, (a, b, s) in enumerate(cues, 1)))
    wav = NAR / "narration_track.wav"
    sf.write(wav, track[: int(total * SR)], SR)
    style = ("FontName=DejaVu Sans,FontSize=13,PrimaryColour=&H00FFFFFF,OutlineColour=&H00141414,BackColour=&H00141414,"
             "BorderStyle=4,Outline=0,Shadow=0,Alignment=2,MarginV=6,MarginL=40,MarginR=40")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", video, "-i", str(wav), "-i", str(srt),
                    "-filter_complex", f"[0:v]subtitles={srt}:force_style='{style}'[v];[1:a]loudnorm=I=-16:TP=-1.5:LRA=11[a]",
                    "-map", "[v]", "-map", "[a]", "-map", "2:s", "-c:v", "libx264", "-crf", "20", "-preset", "medium",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-c:s", "mov_text",
                    "-metadata:s:s:0", "language=eng", "-movflags", "+faststart", out], check=True)
    Path(out).with_suffix(".narration.json").write_text(json.dumps({"voice": cfg["voice"], "model": "hexgrad/Kokoro-82M",
                                                                   "lines": report, "cues": len(cues)}, indent=1))
    print("wrote", out, "| all lines fit:", all(r["fits"] for r in report), "| cues:", len(cues))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["synth", "mux"])
    ap.add_argument("--video")
    ap.add_argument("--out")
    a = ap.parse_args()
    synth() if a.step == "synth" else mux(a.video, a.out)


if __name__ == "__main__":
    main()
