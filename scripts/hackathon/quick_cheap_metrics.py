#!/usr/bin/env python3
"""Cheap metrics for ONE checkpoint via the ORIGINAL repo script (Environment A).

Builds a throw-away run directory that links the checkpoint, its .hydra config and
logs.json.txt, then runs diffusion_policy/scripts/plv_open_loop_smoothness_baseline.py
(Action L1 on 64 validation windows + open-loop smoothness, H=10, policy seed 0).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    ckpt = (ROOT / a.checkpoint).resolve() if not os.path.isabs(a.checkpoint) else Path(a.checkpoint)
    run = ckpt.parent.parent
    with tempfile.TemporaryDirectory(prefix="plv_quick_") as tmp:
        t = Path(tmp) / run.name
        (t / "checkpoints").mkdir(parents=True)
        os.symlink(ckpt, t / "checkpoints" / ckpt.name)
        os.symlink(run / ".hydra", t / ".hydra")
        if (run / "logs.json.txt").exists():
            os.symlink(run / "logs.json.txt", t / "logs.json.txt")
        cmd = [sys.executable, str(ROOT / "diffusion_policy/scripts/plv_open_loop_smoothness_baseline.py"),
               "--roots", str(t), "--output", str(Path(a.out).resolve()), "--device", a.device,
               "--max-eval-samples", "64", "--horizon", "10", "--policy-seed", "0", "--skip-latest"]
        print("RUN", " ".join(cmd), flush=True)
        sys.exit(subprocess.call(cmd, cwd=ROOT))


if __name__ == "__main__":
    main()
