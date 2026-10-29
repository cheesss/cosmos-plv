#!/usr/bin/env python3
"""Drive the REAL dashboard with Playwright (system Chrome, H.264 video support)
and record the browser session.

Writes <out>/session.webm and <out>/timeline.json (step -> seconds since the
recording started); make_demo_video.py uses the timeline for captions and
speed-ups. Every button press does real work: the original metric script,
Cosmos3-Edge on the local GPU, and a live Nebius Token Factory call.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data/outputs/hackathon_demo"))
    ap.add_argument("--port", type=int, default=8601)
    ap.add_argument("--run", default="can_seed1")
    ap.add_argument("--checkpoint", default="can_s1_e0020")
    ap.add_argument("--episode", default="200006")
    ap.add_argument("--skip-cheap", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    st = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "dashboard/app.py"),
                           "--server.headless", "true", "--server.port", str(a.port),
                           "--server.fileWatcherType", "none", "--browser.gatherUsageStats", "false"],
                          cwd=ROOT, stdout=open(out / "streamlit.log", "w"), stderr=subprocess.STDOUT)
    time.sleep(10)
    tl = {}
    try:
        with sync_playwright() as p:
            b = p.chromium.launch(channel="chrome")
            ctx = b.new_context(viewport={"width": 1600, "height": 900}, record_video_dir=str(out / "raw"),
                                record_video_size={"width": 1600, "height": 900})
            pg = ctx.new_page()
            t0 = time.time()

            def mark(k):
                tl[k] = round(time.time() - t0, 2)

            def select(label, value):
                pg.locator('div[data-testid="stSelectbox"]').filter(has_text=label).click()
                pg.wait_for_timeout(600)
                pg.get_by_role("option", name=value).first.click()
                pg.wait_for_timeout(2000)

            def tab(name):
                pg.get_by_role("tab", name=name).click()
                pg.wait_for_timeout(2500)

            def scroll(dy, wait=2500):
                pg.mouse.move(900, 600)
                pg.mouse.wheel(0, dy)
                pg.wait_for_timeout(wait)

            pg.goto(f"http://localhost:{a.port}", wait_until="networkidle")
            pg.wait_for_timeout(6000); mark("open")
            select("Training run", a.run)
            select("Checkpoint", a.checkpoint); mark("selected")
            pg.wait_for_timeout(3000); mark("recommend_view")
            scroll(450, 4000); scroll(500, 3500); scroll(-950, 1500); mark("recommend_done")
            tab("② Cheap screening"); mark("cheap_tab")
            if not a.skip_cheap:
                pg.get_by_role("button", name="▶ Run cheap metrics").click(); mark("cheap_start")
                pg.get_by_text("done in").wait_for(timeout=600_000); mark("cheap_done")
                pg.wait_for_timeout(3000)
            scroll(500, 3000); scroll(-500, 1000)
            tab("③ Rollouts & Cosmos"); mark("roll_tab")
            select("Episode (env seed)", a.episode)
            pg.evaluate("() => { const v = document.querySelector('video'); if (v) { v.muted = true; v.playbackRate = 2.0; v.play(); } }")
            mark("video_play")
            pg.wait_for_timeout(10500); mark("video_done")
            scroll(600, 4500); mark("keyframes_shown")
            pg.get_by_role("button", name="▶ Run Cosmos3-Edge now (local GPU)").click(); mark("cosmos_start")
            pg.get_by_text("Cosmos3-Edge (nvidia/Cosmos3-Edge").wait_for(timeout=600_000); mark("cosmos_done")
            pg.wait_for_timeout(1500); scroll(400, 4000)
            pg.get_by_role("button", name="▶ Ask Nemotron-3.5-Super-VL on Nebius").click(); mark("nebius_start")
            pg.get_by_text("Nebius Token Factory ·").wait_for(timeout=300_000); mark("nebius_done")
            pg.wait_for_timeout(1000); scroll(500, 5000)
            pg.mouse.wheel(0, -4000); pg.wait_for_timeout(800)
            tab("④ Results"); mark("results")
            pg.wait_for_timeout(3000); scroll(700, 4500); scroll(700, 4500); scroll(700, 4000); mark("results_done")
            pg.mouse.wheel(0, -5000); pg.wait_for_timeout(800)
            tab("⑤ Explorations"); mark("audits"); pg.wait_for_timeout(4000)
            scroll(500, 5000); scroll(500, 5000); mark("audits_done")
            pg.mouse.wheel(0, -3000); pg.wait_for_timeout(800)
            tab("⑥ Cost & provenance"); mark("cost"); pg.wait_for_timeout(3500); scroll(600, 3500)
            pg.mouse.wheel(0, -2000); pg.wait_for_timeout(500)
            tab("① Recommend"); pg.wait_for_timeout(3000); mark("end")
            vid = pg.video.path()
            ctx.close(); b.close()
            shutil.move(vid, out / "session.webm")
    finally:
        st.terminate()
        (out / "timeline.json").write_text(json.dumps(tl, indent=2))
    print(json.dumps(tl, indent=2))


if __name__ == "__main__":
    main()
