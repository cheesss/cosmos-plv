"""Provenance helpers: every result row must be traceable to its inputs."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: os.PathLike, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_commit(repo: Path = REPO_ROOT) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        )
        return out.strip()
    except Exception:
        return None


def git_dirty(repo: Path = REPO_ROOT) -> Optional[bool]:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(out.strip())
    except Exception:
        return None


def environment_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname_hash": sha256_bytes(platform.node().encode())[:12],
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["torch_cuda"] = torch.version.cuda
    except Exception:
        pass
    for mod in ("robosuite", "robomimic", "mujoco", "transformers", "vllm"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            continue
    return info


def repo_relative(path: os.PathLike) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def write_json(path: os.PathLike, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=_json_default)
    os.replace(tmp, path)


def _json_default(obj: Any) -> Any:
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)}")
