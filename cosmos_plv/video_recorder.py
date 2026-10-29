"""MP4 writing / reading and frame sampling for Cosmos inputs."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np


def write_mp4(path: Path, frames: Sequence[np.ndarray], fps: int = 10) -> None:
    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        str(path), fps=fps, codec="libx264", quality=8, macro_block_size=16, pixelformat="yuv420p"
    ) as w:
        for f in frames:
            w.append_data(np.asarray(f, dtype=np.uint8))


def read_mp4(path: Path) -> List[np.ndarray]:
    import imageio.v2 as imageio

    with imageio.get_reader(str(path)) as r:
        return [np.asarray(f) for f in r]


def sample_frames(frames: Sequence[np.ndarray], n: int) -> List[int]:
    """Evenly spaced frame indices including first and last."""
    if len(frames) == 0:
        return []
    if n >= len(frames):
        return list(range(len(frames)))
    return [int(round(x)) for x in np.linspace(0, len(frames) - 1, n)]


def contact_sheet(frames: Sequence[np.ndarray], idx: Sequence[int], cols: int = 4) -> np.ndarray:
    """Tile selected frames into one image (for figures)."""
    tiles = [np.asarray(frames[i]) for i in idx]
    h, w = tiles[0].shape[:2]
    rows = int(np.ceil(len(tiles) / cols))
    sheet = np.full((rows * h, cols * w, 3), 255, dtype=np.uint8)
    for k, tile in enumerate(tiles):
        r, c = divmod(k, cols)
        sheet[r * h : (r + 1) * h, c * w : (c + 1) * w] = tile[..., :3]
    return sheet
