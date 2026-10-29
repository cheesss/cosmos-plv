"""Nebius Token Factory backend (OpenAI-compatible API) for an NVIDIA open VLM.

Default model: ``nvidia/Nemotron-3.5-Super-VL`` (NVIDIA, OpenMDW-1.1), listed as
an active image/video model in the public Token Factory catalog
(https://tokenfactory.nebius.com/api/public/models_info, checked 2026-10-29).

The API key is read from ``NEBIUS_API_KEY`` or from the first existing file of
``$COSMOS_PLV_SECRETS`` / ``~/.config/cosmos-plv/nebius.env`` (``KEY=value``
lines).  The key is never logged; every call appends a sanitized evidence
record (model, request id, token usage, latency, input hashes) to a JSONL file.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .cosmos_backend import BackendResult
from .provenance import sha256_bytes, utc_now

NEBIUS_MODEL_ID = os.environ.get("NEBIUS_VLM_MODEL", "nvidia/Nemotron-3.5-Super-VL")
BASE_URLS = [
    os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/"),
    # the catalog lists this model in us-north1 only; regional host as fallback
    "https://api.tokenfactory.us-north1.nebius.com/v1/",
]
SECRET_FILES = [
    os.environ.get("COSMOS_PLV_SECRETS", ""),
    str(Path.home() / ".config" / "cosmos-plv" / "nebius.env"),
]


def load_api_key() -> Optional[str]:
    key = os.environ.get("NEBIUS_API_KEY")
    if key:
        return key.strip()
    for f in SECRET_FILES:
        if f and Path(f).is_file():
            for line in Path(f).read_text().splitlines():
                line = line.strip()
                if line.startswith("NEBIUS_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def compact_video_bytes(src: str, fps: float = 4.0, size: int = 320, max_s: float = 30.0,
                        pad_to_s: Optional[float] = None) -> bytes:
    """Re-encode a rollout mp4 to a small mp4 (subsampled fps/resolution).

    ``pad_to_s``: repeat the final frame until the clip lasts this long, so that the
    clip length cannot reveal the outcome (successful rollouts stop early).
    """
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image

    reader = imageio.get_reader(src)
    meta = reader.get_meta_data()
    src_fps = float(meta.get("fps", 10.0))
    step = max(1, int(round(src_fps / fps)))
    frames = []
    for i, f in enumerate(reader):
        if i % step:
            continue
        if len(frames) >= int(max_s * fps):
            break
        frames.append(np.asarray(Image.fromarray(f).resize((size, size), Image.BILINEAR)))
    reader.close()
    if pad_to_s is not None and frames:
        # target = number of frames a full-length source clip of pad_to_s seconds
        # yields with the same subsampling, so every padded clip has the same length
        target = min(int(max_s * fps), len(range(0, int(round(pad_to_s * src_fps)), step)))
        while len(frames) < target:
            frames.append(frames[-1])
    buf = io.BytesIO()
    with imageio.get_writer(buf, format="mp4", fps=fps, codec="libx264", quality=6,
                            macro_block_size=16, pixelformat="yuv420p") as w:
        for f in frames:
            w.append_data(f)
    return buf.getvalue()


class NebiusVLMClient:
    backend_name = "nebius-token-factory"

    def __init__(self, model_id: str = NEBIUS_MODEL_ID, evidence_path: Optional[str] = None,
                 api_key: Optional[str] = None, timeout_s: float = 180.0):
        from openai import OpenAI

        key = api_key or load_api_key()
        if not key:
            raise RuntimeError(
                "NEBIUS_API_KEY not set (env var or ~/.config/cosmos-plv/nebius.env)"
            )
        self._key = key
        self.model_id = model_id
        self.timeout_s = timeout_s
        self.evidence_path = Path(evidence_path) if evidence_path else None
        self._OpenAI = OpenAI
        self.base_url = None

    def _client(self, base_url: str):
        return self._OpenAI(base_url=base_url, api_key=self._key, timeout=self.timeout_s, max_retries=1)

    def list_models(self) -> Dict[str, Any]:
        errors = []
        for url in BASE_URLS:
            try:
                ids = [m.id for m in self._client(url).models.list().data]
                return {"base_url": url, "models": ids}
            except Exception as e:
                errors.append(f"{url}: {type(e).__name__}: {str(e)[:200]}")
        return {"base_url": None, "models": [], "errors": errors}

    def generate(self, prompt: str, video: Optional[str] = None, images: Optional[List[str]] = None,
                 max_tokens: int = 1024, reasoning: bool = False, temperature: float = 0.2,
                 system: str = "You are a helpful assistant.", tag: str = "",
                 pad_video_to_s: Optional[float] = None) -> BackendResult:
        content: List[Dict[str, Any]] = []
        input_hashes: Dict[str, str] = {}
        if video is not None:
            data = compact_video_bytes(video, pad_to_s=pad_video_to_s)
            input_hashes["video_sha256"] = sha256_bytes(data)
            input_hashes["video_bytes"] = str(len(data))
            if pad_video_to_s is not None:
                input_hashes["video_padded_to_s"] = str(pad_video_to_s)
            content.append({"type": "video_url",
                            "video_url": {"url": "data:video/mp4;base64," + base64.b64encode(data).decode()}})
        for im in images or []:
            raw = Path(im).read_bytes()
            input_hashes[f"image_sha256_{len(input_hashes)}"] = sha256_bytes(raw)
            mime = "image/png" if str(im).endswith(".png") else "image/jpeg"
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:{mime};base64," + base64.b64encode(raw).decode()}})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
        started = utc_now()
        last_err = None
        for url in ([self.base_url] if self.base_url else BASE_URLS):
            t0 = time.time()
            try:
                resp = self._client(url).chat.completions.create(
                    model=self.model_id, messages=messages, max_tokens=max_tokens,
                    temperature=temperature,
                    extra_body={"chat_template_kwargs": {"enable_thinking": bool(reasoning)}},
                )
                latency = time.time() - t0
                self.base_url = url
                usage = getattr(resp, "usage", None)
                result = BackendResult(
                    backend=self.backend_name, model_id=self.model_id, model_revision=None,
                    raw_text=resp.choices[0].message.content or "", latency_s=round(latency, 3),
                    prompt_tokens=getattr(usage, "prompt_tokens", None),
                    completion_tokens=getattr(usage, "completion_tokens", None),
                    request_id=getattr(resp, "id", None), served_model=getattr(resp, "model", None),
                    started_utc=started,
                    params={"base_url": url, "max_tokens": max_tokens, "temperature": temperature,
                            "reasoning": reasoning, **input_hashes, "prompt_sha256": sha256_bytes(prompt.encode())},
                )
                self._log(result, tag)
                return result
            except Exception as e:
                last_err = f"{url}: {type(e).__name__}: {str(e)[:300]}"
        result = BackendResult(backend=self.backend_name, model_id=self.model_id, model_revision=None,
                               raw_text="", latency_s=0.0, started_utc=started, error=last_err,
                               params={**input_hashes, "prompt_sha256": sha256_bytes(prompt.encode())})
        self._log(result, tag)
        return result

    def _log(self, result: BackendResult, tag: str) -> None:
        if not self.evidence_path:
            return
        rec = {k: v for k, v in result.to_dict().items() if k != "raw_text"}
        rec["raw_text_sha256"] = sha256_bytes(result.raw_text.encode())
        rec["raw_text_chars"] = len(result.raw_text)
        rec["tag"] = tag
        rec["logged_utc"] = utc_now()
        self.evidence_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.evidence_path, "a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
