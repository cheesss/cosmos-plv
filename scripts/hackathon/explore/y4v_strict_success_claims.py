#!/usr/bin/env python
"""Verification of y4_grounded_reports (EXPLORATORY, post-hoc): a STRICT check of one claim type
on the saved test checkpoint reports (no new model calls).

Claim type: a clause that names exactly one source (simulator / Nemotron / Cosmos), talks about
success(es)/successful, does not talk about agreement, and states a count as "k/10", "k of 10",
"k out of 10" or "all 10". The count must equal THAT source's success count for THIS checkpoint
(sim.successes, nemotron.judged_successes, cosmos.judged_successes). Every flag is printed so it
can be hand-checked; the hand-check verdicts are recorded in the verification note.
Writes data/outputs/hackathon_explore/y4_grounded_reports_verification/strict_success_claims.json
"""
from __future__ import annotations
import json, re, sys, math
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/hackathon/explore"))
import y4_verify as V  # noqa: E402
import y4_grounded_reports as G  # noqa: E402
OUT = ROOT / "data/outputs/hackathon_explore/y4_grounded_reports_verification"
SRC = {"sim": re.compile(r"simulat|\bsim\b|sim\.successes", re.I),
       "nemotron": re.compile(r"nemotron|\bvlm\b|vision[- ]language|judged success|judge[sd]?\b", re.I),
       "cosmos": re.compile(r"cosmos", re.I)}
CNT = re.compile(r"\b(?:(\d+)\s*(?:/|out of|of)\s*10\b|all\s+10\b)", re.I)


def wilson(k, n, z=1.959964):
    if n == 0:
        return [None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0, c - h), 3), round(min(1, c + h), 3)]


def main():
    facts, runs, truth = G.load()
    claims, flagged = 0, []
    reports_with_flag = set()
    n_reports = 0
    for rec in G.load_generations("test", "v2"):
        rep, v = G.verify_rec(rec, facts, runs, truth)
        if v.get("parse_status") not in ("ok", "ok_extracted"):
            continue
        n_reports += 1
        f = facts[rec["key"]]
        truthc = {"sim": f["sim"]["successes"], "nemotron": f["nemotron"]["judged_successes"],
                  "cosmos": (f["cosmos"] or {}).get("judged_successes")}
        full = " ".join(t for _, t in V.text_fields(rep))
        full = V.CITE_RE.sub("", full)
        for clause in re.split(r"(?<=[.;])\s+|,\s+(?:and|but|while|whereas)\s+|\s+while\s+|\s+whereas\s+|\s+but\s+", full):
            if not re.search(r"success", clause, re.I) or re.search(r"agree|match|median|rank|other checkpoints|run", clause, re.I):
                continue
            srcs = [s for s, r in SRC.items() if r.search(clause)]
            if len(srcs) != 1:
                continue
            for m in CNT.finditer(clause):
                k = 10 if m.group(1) is None else int(m.group(1))
                claims += 1
                if truthc[srcs[0]] is not None and k != truthc[srcs[0]]:
                    flagged.append({"key": rec["key"], "sample": rec["sample"], "source": srcs[0], "claimed": k,
                                    "true": truthc[srcs[0]], "clause": clause[:300]})
                    reports_with_flag.add((rec["key"], rec["sample"]))
    res = {"n_reports": n_reports, "n_claims_checked": claims, "n_flagged": len(flagged),
           "flag_rate_per_claim": round(len(flagged) / claims, 3) if claims else None,
           "flag_rate_per_claim_wilson95_naive": wilson(len(flagged), claims),
           "reports_with_flag": len(reports_with_flag),
           "reports_with_flag_wilson95": wilson(len(reports_with_flag), n_reports),
           "flagged": flagged}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "strict_success_claims.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
