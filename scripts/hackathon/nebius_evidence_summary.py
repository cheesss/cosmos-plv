#!/usr/bin/env python3
"""Summarize the sanitized Nebius Token Factory call log (results/hackathon/nebius_calls.jsonl).

Cost is an ESTIMATE from the public catalog price of nvidia/Nemotron-3.5-Super-VL
(checked 2026-10-29: $0.30 / 1M input tokens, $0.90 / 1M output tokens); the
billed amount is whatever the Nebius console shows.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PRICE_IN, PRICE_OUT = 0.30, 0.90  # USD per 1M tokens (public catalog, 2026-10-29)


def summarize(path: Path) -> dict:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    ok = [r for r in rows if not r.get("error")]
    lat = np.array([r["latency_s"] for r in ok], dtype=float)
    pin = int(sum(r.get("prompt_tokens") or 0 for r in ok))
    pout = int(sum(r.get("completion_tokens") or 0 for r in ok))
    tags = Counter(str(r.get("tag", "")).split("/")[0] for r in rows)
    return {
        "calls_total": len(rows),
        "calls_succeeded": len(ok),
        "calls_failed": len(rows) - len(ok),
        "first_call_utc": min((r["started_utc"] for r in rows), default=None),
        "last_call_utc": max((r["started_utc"] for r in rows), default=None),
        "requested_models": sorted({r["model_id"] for r in rows}),
        "served_models": sorted({r["served_model"] for r in ok if r.get("served_model")}),
        "base_urls": sorted({r.get("params", {}).get("base_url") for r in ok if r.get("params")} - {None}),
        "example_request_ids": [r["request_id"] for r in ok[:3]],
        "latency_s": {"mean": float(lat.mean()), "p50": float(np.median(lat)), "p90": float(np.percentile(lat, 90)),
                      "max": float(lat.max())} if len(lat) else None,
        "tokens": {"prompt": pin, "completion": pout},
        "usd_estimate_at_catalog_price": round(pin / 1e6 * PRICE_IN + pout / 1e6 * PRICE_OUT, 4),
        "calls_by_tag_prefix": dict(tags),
        "errors": Counter((r.get("error") or "")[:80] for r in rows if r.get("error")).most_common(5),
    }


if __name__ == "__main__":
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/hackathon/nebius_calls.jsonl"
    print(json.dumps(summarize(p), indent=2))
