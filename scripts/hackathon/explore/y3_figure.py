#!/usr/bin/env python3
"""y3_visual_robustness figure: last frame of one successful and one failed episode
under every condition (first success / first dropped episode of the fixed selection
order).  Writes docs/hackathon/explore/y3_visual_robustness_views.png.

  PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/y3_figure.py
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
SCR = ROOT / "data/outputs/hackathon_explore/y3_visual_robustness"
CONDS = [("orig", "degrade/orig"), ("down224", "degrade/down224"), ("down112", "degrade/down112"),
         ("jpeg20", "degrade/jpeg20"), ("blur", "degrade/blur"), ("dim", "degrade/dim"),
         ("re-render agentview", "render/agentview"), ("frontview", "render/frontview"),
         ("birdview", "render/birdview"), ("eye_in_hand", "render/robot0_eye_in_hand")]


def main():
    eps = json.loads((ROOT / "results/hackathon/explore/y3_visual_robustness_episodes.json").read_text())["episodes"]
    pick = []
    for want in (("can", "none"), ("square", "none"), ("square", "dropped")):
        for e in eps:
            if (e["task"], e["failure_type"]) == want and (SCR / "render/agentview" / e["checkpoint_id"] /
                                                          f"seed{e['env_seed']}_last.png").exists():
                pick.append(e)
                break
    T = 180
    sheet = Image.new("RGB", (T * len(CONDS), (T + 16) * len(pick) + 16), "white")
    d = ImageDraw.Draw(sheet)
    for j, (name, _) in enumerate(CONDS):
        d.text((j * T + 4, 2), name, fill="black")
    for i, e in enumerate(pick):
        for j, (name, sub) in enumerate(CONDS):
            p = SCR / sub / e["checkpoint_id"] / f"seed{e['env_seed']}_last.png"
            if p.exists():
                sheet.paste(Image.open(p).convert("RGB").resize((T, T), Image.BILINEAR), (j * T, 16 + i * (T + 16)))
        d.text((4, 16 + i * (T + 16) + T + 2), f"{e['checkpoint_id']} seed{e['env_seed']} truth={e['failure_type']}",
               fill="black")
    out = ROOT / "docs/hackathon/explore/y3_visual_robustness_views.png"
    sheet.save(out)
    print(out)


if __name__ == "__main__":
    main()
