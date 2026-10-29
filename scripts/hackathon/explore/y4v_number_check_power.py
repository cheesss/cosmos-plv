#!/usr/bin/env python
"""Verification of y4_grounded_reports (EXPLORATORY, post-hoc): how much power does the lenient
'number appears somewhere in the fact sheet' check have?

Re-scores the SAVED test generations only (no new model calls). For every number token that the
y4 verifier counted in a parsed test checkpoint report we ask: if the model had written a WRONG
number instead (true value +1, -1, or a value drawn from 0..10), would the verifier have flagged it?
Also: distribution of the 723 counted numbers, and fraction of the integers 0..10 that are
'supported' by each sheet (i.e. can never be flagged as unsupported).
Writes data/outputs/hackathon_explore/y4_grounded_reports_verification/number_power.json
"""
from __future__ import annotations
import json, sys, random, statistics
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/hackathon/explore"))
import y4_verify as V  # noqa: E402
import y4_grounded_reports as G  # noqa: E402

OUT = ROOT / "data/outputs/hackathon_explore/y4_grounded_reports_verification"


def wilson(k, n, z=1.959964):
    import math
    if n == 0:
        return [None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0, c - h), 3), round(min(1, c + h), 3)]


def main():
    facts, runs, truth = G.load()
    recs = [r for r in G.load_generations("test", "v2")]
    rng = random.Random(0)
    tokens = Counter()
    per_sheet_int_cov = {}
    pert = {"plus1": [0, 0], "minus1": [0, 0], "rand0_10": [0, 0]}
    pert_by_ckpt = {}
    for rec in recs:
        rep, v = G.verify_rec(rec, facts, runs, truth)
        if v.get("parse_status") not in ("ok", "ok_extracted"):
            continue
        f = facts[rec["key"]]
        allowed = V.numeric_values(f)
        cov = [i for i in range(0, 11) if any(V.num_matches(str(i), False, a) for a in allowed)]
        per_sheet_int_cov[rec["key"]] = cov
        full = " ".join(t for _, t in V.text_fields(rep))
        nrec = V.check_numbers(full, f, V.flatten(f), extra_allowed=[])
        nums = [r for k, r in nrec if k == "num"]
        for r in nums:
            s = r["number"]
            tokens[s] += 1
            pct = s.endswith("%")
            core = s.rstrip("%")
            if r.get("word") or not core.replace(".", "").isdigit():
                continue
            x = float(core)
            dec = V.decimals(core)
            for name, alt in (("plus1", x + (10 ** -dec if dec else 1)), ("minus1", x - (10 ** -dec if dec else 1)),
                              ("rand0_10", None)):
                if alt is None:
                    if dec:
                        continue
                    choices = [i for i in range(0, 11) if i != int(x)]
                    alt = rng.choice(choices)
                if alt < 0:
                    continue
                alt_s = (f"{alt:.{dec}f}" if dec else str(int(round(alt))))
                flagged = not any(V.num_matches(alt_s, pct, a) for a in allowed)
                pert[name][0] += int(flagged)
                pert[name][1] += 1
                pb = pert_by_ckpt.setdefault(rec["key"], {"plus1": [0, 0]})
                if name == "plus1":
                    pb["plus1"][0] += int(flagged); pb["plus1"][1] += 1
    total = sum(tokens.values())
    res = {
        "n_number_tokens": total,
        "top_tokens": tokens.most_common(15),
        "share_tokens_equal_10": round(tokens["10"] / total, 3),
        "share_tokens_integer_0_10": round(sum(c for t, c in tokens.items() if t.isdigit() and int(t) <= 10) / total, 3),
        "integers_0_10_supported_per_sheet": {"min": min(len(v) for v in per_sheet_int_cov.values()),
                                               "median": statistics.median(len(v) for v in per_sheet_int_cov.values()),
                                               "max": max(len(v) for v in per_sheet_int_cov.values()),
                                               "out_of": 11},
        "perturbation_detection_rate": {k: {"flagged": a, "n": b, "rate": round(a / b, 3) if b else None,
                                            "wilson95_naive": wilson(a, b)} for k, (a, b) in pert.items()},
        "note": "detection rate = fraction of perturbed numbers the lenient y4 check would flag as unsupported. "
                "Perturbations are hypothetical wrong numbers placed at the positions of real numbers.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "number_power.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
