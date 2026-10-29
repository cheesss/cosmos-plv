"""Stage D: NVIDIA Cosmos inference backends (Environment B: py3.11, torch>=2.6).

``CosmosReasonerLocal`` runs the Reasoner tower of ``nvidia/Cosmos3-Edge``
through Hugging Face Transformers (AutoModelForImageTextToText, bfloat16).
The model id and revision are pinned and recorded with every call.

Every call returns a ``BackendResult`` holding the raw generated text, timing,
token counts and full provenance; parsing happens in visual_diagnostics.py.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

COSMOS_MODEL_ID = os.environ.get("COSMOS_MODEL_ID", "nvidia/Cosmos3-Edge")
COSMOS_REVISION = os.environ.get("COSMOS_REVISION", "344d602b128d1bbdacb43b08d0a3626f46343e29")


@dataclass
class BackendResult:
    backend: str
    model_id: str
    model_revision: Optional[str]
    raw_text: str
    latency_s: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    request_id: Optional[str] = None
    served_model: Optional[str] = None
    started_utc: Optional[str] = None
    error: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CosmosReasonerLocal:
    """In-process Cosmos3-Edge Reasoner on a local CUDA GPU."""

    backend_name = "local-transformers"

    def __init__(
        self,
        model_id: str = COSMOS_MODEL_ID,
        revision: str = COSMOS_REVISION,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
    ):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model_id = model_id
        self.revision = revision
        self.device = device
        # Optional hard cap so that a shared GPU (training / rollouts) is never starved:
        # this process OOMs instead of the others.
        frac = os.environ.get("COSMOS_MAX_GPU_FRACTION")
        if frac:
            torch.cuda.set_per_process_memory_fraction(float(frac), torch.device(device))
        t0 = time.time()
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, revision=revision, dtype=getattr(torch, dtype), device_map=device
        )
        self.model.eval()
        self.load_s = time.time() - t0
        self.dtype = dtype

    def _messages(self, prompt: str, video: Optional[str], images: Optional[List[str]], system: str):
        content: List[Dict[str, Any]] = []
        # Cosmos guide: always place media before the text prompt.
        if video is not None:
            content.append({"type": "video", "path": str(video)})
        for im in images or []:
            content.append({"type": "image", "path": str(im)})
        content.append({"type": "text", "text": prompt})
        return [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": content},
        ]

    def generate(
        self,
        prompt: str,
        video: Optional[str] = None,
        images: Optional[List[str]] = None,
        fps: float = 4.0,
        max_new_tokens: int = 1536,
        reasoning: bool = True,
        seed: int = 0,
        system: str = "You are a helpful assistant.",
        do_sample: bool = True,
    ) -> BackendResult:
        import torch

        from .provenance import utc_now

        started = utc_now()
        messages = self._messages(prompt, video, images, system)
        kwargs: Dict[str, Any] = dict(
            tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )
        if video is not None:
            kwargs["fps"] = fps
        kwargs["enable_thinking"] = bool(reasoning)
        # Official Cosmos3 sampling settings (reasoner_prompt_guide.md).
        if reasoning:
            gen = dict(temperature=0.6, top_p=0.95, top_k=20, repetition_penalty=1.0)
        else:
            gen = dict(temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.0)
        t0 = time.time()
        try:
            inputs = self.processor.apply_chat_template(messages, **kwargs).to(self.model.device)
            for k, v in list(inputs.items()):
                if hasattr(v, "dtype") and v.dtype == torch.float32:
                    inputs[k] = v.to(torch.bfloat16)
            torch.manual_seed(seed)
            with torch.no_grad():
                out = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, **(gen if do_sample else {})
                )
            n_in = int(inputs["input_ids"].shape[1])
            new = out[0, n_in:]
            text = self.processor.decode(new, skip_special_tokens=True)
            err = None
            n_out = int(new.shape[0])
        except Exception as e:  # keep failures as data, never fabricate output
            text, err, n_in, n_out = "", f"{type(e).__name__}: {e}", None, None
        latency = time.time() - t0
        return BackendResult(
            backend=self.backend_name,
            model_id=self.model_id,
            model_revision=self.revision,
            raw_text=text,
            latency_s=round(latency, 3),
            prompt_tokens=n_in,
            completion_tokens=n_out,
            started_utc=started,
            error=err,
            params={
                "fps": fps if video is not None else None,
                "max_new_tokens": max_new_tokens,
                "reasoning": reasoning,
                "seed": seed,
                "do_sample": do_sample,
                "dtype": self.dtype,
                "device": self.device,
                **(gen if do_sample else {}),
            },
        )
