"""y4_grounded_reports: automatic grounding verifier for LLM-written checkpoint/run reports.

EXPLORATORY. Checks, per report:
  * numbers: every number in the report text (digits, plus number words before count nouns)
    must equal some value in the fact sheet (exact, or within the rounding implied by the number
    of written decimals; fractions may be written as percentages). -> unsupported_numbers
  * citations: a number followed by "[key]" must match the value stored under that key.
    -> miscited (wrong value) / bad_key (key does not exist)
  * failure-type claims: a non-negated mention of a failure type must refer to a type that occurs
    (simulator or Nemotron-judged) for the checkpoint(s) in scope; "N <type>" counts must equal the
    simulator or judged count. -> wrong_failure_claims
  * structured fields: dominant_failure_mode vs RULE_DOMINANT, spend_more_rollouts vs RULE_CKPT,
    evaluate_next vs RULE_RUN (see y4_build_facts.py).
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

FTYPES = ["no_grasp", "dropped", "not_placed", "other"]
FT_PATTERNS = {
    "no_grasp": r"no[_\s-]grasp\w*|fail(?:ed|s|ure|ing)?\s+to\s+grasp|never\s+grasp\w*|grasp\s+failure\w*"
                r"|miss(?:ed|es)?\s+(?:the\s+)?grasp|without\s+grasping|not\s+grasp\w*",
    "dropped": r"\bdrop(?:ped|s|ping)?\b(?!\s+(?:in|to|from|of)\b)",
    "not_placed": r"not[_\s-]placed|fail(?:ed|s|ing)?\s+to\s+place|placement\s+failure\w*|never\s+plac\w+|misplac\w+",
    "other": r"[`'\"]other[`'\"]|\bother\s+failure",
}
NEG_CUE = re.compile(r"\b(no|zero|none|never|without|absent|absence|not\s+any|did\s+not|didn't|nor|0|rather\s+than|instead\s+of|not\s+(?:a|an|due\s+to|from|by|caused\s+by))\b[^.;]{0,40}$",
                     re.I)
WORDNUM = {"zero": 0, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
           "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
WORDNUM_RE = re.compile(r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b"
                        r"(?=\s+(?:\w+\s+)?(?:of\b|out\b|episodes?|rollouts?|failures?|successes|trials?|checkpoints?|runs?))",
                        re.I)
ID_PATTERNS = [
    r"\b(?:can|square)_s\d+(?:_e\d{3,4})?\b",
    r"Cosmos[-\s]?3[-\w]*", r"Nemotron[-\s]?3(?:[._]5)?[-\w]*", r"Super[-\s]VL", r"PLV-\d",
    r"\b[vV]\d+(?:_\w+)?\b", r"\b[hH]=?\d+\b", r"_h\d+", r"pad\d+", r"\bseed\s*[12]\b",
    r"\bs[12]\b", r"\be\d{4}\b", r"robomimic", r"\b3090\b",
]
NUM_RE = re.compile(r"(?<![A-Za-z_\d.])(\d+(?:\.\d+)?)(\s*(?:%|percent\b))?")
CITE_RE = re.compile(r"\[([^\[\]]{0,120}[A-Za-z_][^\[\]]{0,120})\]")
FRAC_RE = re.compile(r"(\d+)\s*(?:/|out\s+of|of)\s*(\d+)\s*(?:\w+\s*){0,2}$")


def flatten(obj: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(obj, dict):
        if prefix:
            out[prefix] = obj
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        out[prefix] = obj
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


def numeric_values(facts: Any) -> List[float]:
    vals: List[float] = []
    for k, v in flatten(facts).items():
        for seg in re.findall(r"\d+(?:\.\d+)?", k):  # keys such as episode seeds
            vals.append(float(seg))
        if isinstance(v, bool) or v is None:
            continue
        if isinstance(v, (int, float)):
            vals.append(float(v))
        elif isinstance(v, str):
            for s in re.findall(r"(?<![A-Za-z_])\d+(?:\.\d+)?", v):
                vals.append(float(s))
    return vals


def decimals(s: str) -> int:
    return len(s.split(".")[1]) if "." in s else 0


def num_matches(xs: str, is_pct: bool, v: float) -> bool:
    x = float(xs)
    tol = 0.5 * 10 ** (-decimals(xs)) + 1e-9
    if abs(x - v) <= tol:
        return True
    if (is_pct or 0 < v <= 1) and abs(x - 100 * v) <= tol and (is_pct or decimals(xs) == 0 and x > 1):
        # 60 / 60% for 0.6 (without % only if it is an integer > 1, i.e. clearly a percentage)
        return True if is_pct else False
    if is_pct and abs(x / 100 - v) <= tol / 100:
        return True
    return False


def mask_ids(text: str) -> str:
    for p in ID_PATTERNS:
        text = re.sub(p, lambda m: "#" * len(m.group(0)), text)
    return text


def resolve(flat: Dict[str, Any], key: str):
    """resolve a citation; supports "a.b", "a.b, c.d", "a.b=3", "a.b: 3". Returns (ok, value or list)."""
    parts = [p for p in re.split(r"[,;]|\band\b", key) if p.strip()]
    if len(parts) > 1:
        vals, ok_any = [], False
        for p_ in parts:
            ok, v = resolve(flat, p_)
            if ok:
                ok_any = True
                vals.append(v)
        return ok_any, {"__multi__": vals}
    key = re.split(r"\s*(?:=|:)\s*", key.strip(), maxsplit=1)[0]
    return _resolve1(flat, key)


def _resolve1(flat: Dict[str, Any], key: str):
    k = key.strip().strip("`'\" ")
    k = re.sub(r"^(?:facts?|fact_sheet|sheet)\.", "", k)
    k = re.sub(r"\[(\d{4,})\]", r".\1", k)          # episodes[200001] -> episodes.200001
    k = re.sub(r"\[['\"]?([A-Za-z_]\w*)['\"]?\]", r".\1", k)
    k = k.replace(" ", "")
    if k in flat:
        return True, flat[k]
    return False, None


def text_fields(report: Dict[str, Any], skip=("dominant_failure_mode", "spend_more_rollouts", "evaluate_next",
                                              "alternatives", "checkpoint_id")) -> List[Tuple[str, str]]:
    out = []
    for k, v in report.items():
        if k in skip:
            continue
        if isinstance(v, str):
            out.append((k, v))
        elif isinstance(v, list):
            out.extend((k, x) for x in v if isinstance(x, str))
    return out


def check_numbers(text: str, facts: Any, flat: Dict[str, Any], extra_allowed: List[float]):
    """returns list of number records."""
    allowed = numeric_values(facts) + list(extra_allowed)
    recs = []
    # 1) citations: associate with the closest preceding number
    masked = mask_ids(text)
    cites = [(m.start(), m.end(), m.group(1)) for m in CITE_RE.finditer(masked)]
    cited_spans = {}
    prev_end = 0
    for (s, e, _k) in cites:
        key = text[s + 1:e - 1]
        before = masked[prev_end:s]
        fr = FRAC_RE.search(before[-40:])
        nums = list(NUM_RE.finditer(before))
        target = None
        if fr:
            # numerator is the cited value
            off = prev_end + (len(before) - len(before[-40:])) + fr.start(1)
            target = (off, off + len(fr.group(1)), fr.group(1), False)
        elif nums and (len(before) - nums[-1].end()) <= 30:
            m = nums[-1]
            target = (prev_end + m.start(1), prev_end + m.end(1), m.group(1), bool(m.group(2)))
        ok_key, val = resolve(flat, key)
        rec = {"cite_key": key, "key_exists": ok_key}
        if target is not None:
            rec.update({"number": target[2], "pct": target[3]})
            cited_spans[target[0]] = rec
            if ok_key:
                vals = []
                if isinstance(val, dict) and "__multi__" in val:
                    for vv in val["__multi__"]:
                        if isinstance(vv, dict):
                            vals += [x for x in vv.values() if isinstance(x, (int, float)) and not isinstance(x, bool)]
                        elif isinstance(vv, list):
                            vals += [x for x in vv if isinstance(x, (int, float)) and not isinstance(x, bool)]
                        elif isinstance(vv, (int, float)) and not isinstance(vv, bool):
                            vals.append(vv)
                elif isinstance(val, dict):
                    vals = [v for v in val.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
                elif isinstance(val, list):
                    vals = [v for v in val if isinstance(v, (int, float)) and not isinstance(v, bool)]
                elif isinstance(val, (int, float)) and not isinstance(val, bool):
                    vals = [val]
                in_key = target[2] in re.findall(r"\d+(?:\.\d+)?", key)
                rec["cite_match"] = bool(in_key or any(num_matches(target[2], target[3], float(v)) for v in vals))
                if not rec["cite_match"]:
                    # "3 dropped failures out of 10 episodes [key]": accept if another number in the
                    # same stretch (<= 60 chars, since the previous citation) matches the key's value
                    for m2 in NUM_RE.finditer(before[-60:]):
                        if any(num_matches(m2.group(1), bool(m2.group(2)), float(v)) for v in vals):
                            rec["cite_match"] = True
                            rec["cite_match_loose"] = True
                            rec["number"] = m2.group(1)
                            break
            else:
                rec["cite_match"] = False
        else:
            rec["number"] = None
            rec["cite_match"] = None  # citation with no number in front (e.g. citing a string fact)
        recs.append(("cite", rec))
        prev_end = e
    # 2) all numbers (citations removed)
    stripped = CITE_RE.sub(lambda m: " " * len(m.group(0)), masked)
    for m in NUM_RE.finditer(stripped):
        xs, pct = m.group(1), bool(m.group(2))
        sup = any(num_matches(xs, pct, v) for v in allowed)
        recs.append(("num", {"number": xs + ("%" if pct else ""), "supported": sup, "pos": m.start(1),
                             "context": text[max(0, m.start() - 50):m.end() + 30],
                             "cited": m.start(1) in cited_spans}))
    for m in WORDNUM_RE.finditer(stripped):
        w = m.group(1).lower()
        if w == "one":
            continue
        xs = str(WORDNUM[w])
        sup = any(num_matches(xs, False, v) for v in allowed)
        recs.append(("num", {"number": w, "supported": sup, "pos": m.start(1),
                             "context": text[max(0, m.start() - 50):m.end() + 30], "cited": False,
                             "word": True}))
    return recs


def source_counts(f: Dict[str, Any]) -> Dict[str, Any]:
    """count facts per source for one checkpoint: sim / nemotron / cosmos (sets of allowed counts,
    and per failure type)."""
    sim = f.get("sim") or {}
    neb = f.get("nemotron") if isinstance(f.get("nemotron"), dict) else {}
    nebp = f.get("nemotron_padded") if isinstance(f.get("nemotron_padded"), dict) else {}
    cos = f.get("cosmos") if isinstance(f.get("cosmos"), dict) else {}
    sft = sim.get("failure_types") or {}
    jft = (neb or {}).get("judged_failure_types") or {}
    def ints(d, keys):
        return {int(d[k]) for k in keys if isinstance(d.get(k), (int, float)) and not isinstance(d.get(k), bool)}
    return {
        "sim": ints(sim, ["n_episodes", "successes", "failures"]) | {int(v) for v in sft.values()},
        "nemotron": ints(neb or {}, ["n_judged", "judged_successes", "judged_failures", "agree_with_sim",
                                     "failure_type_matches_sim", "failure_type_compared_on_sim_failures"])
                    | {int(v) for v in jft.values()} | ints(nebp or {}, ["n_judged", "judged_successes", "agree_with_sim"]),
        "cosmos": ints(cos or {}, ["n_judged", "judged_successes", "agree_with_sim"]),
        "type_sim": {t: int(sft.get(t, 0)) for t in FTYPES},
        "type_judged": {t: int(jft.get(t, 0)) for t in FTYPES},
    }


SRC_RE = {"nemotron": re.compile(r"nemotron|\bvlm\b|vision[- ]language|video judge|\bjudge[sd]?\b|judg(?:ment|ement)", re.I),
          "cosmos": re.compile(r"cosmos", re.I),
          "sim": re.compile(r"simulat|\bsim\b|\bsim[._]|ground[- ]truth", re.I)}
CLAUSE_SPLIT = re.compile(r"(?<=[.;!?])\s+|;\s*|,\s+(?:and|but|while|whereas|with)\s+|\s+while\s+|\s+whereas\s+|\s+but\s+")
ID_RE = re.compile(r"\b(?:can|square)_s\d+_e\d{3,4}\b")


def clause_sources(clause: str) -> List[str]:
    src = [k for k, r in SRC_RE.items() if r.search(clause)]
    return src or ["sim", "nemotron", "cosmos"]


def check_failure_claims(text: str, scopes: Dict[str, Dict[str, Any]], default_scope: List[str]):
    """scopes: ckpt_id -> source_counts(). Sentences naming checkpoint ids are scoped to them.
    Returns (failure-type claim records, attributed count-claim records)."""
    recs, crecs = [], []
    for sent in re.split(r"(?<=[.;!?])\s+", text):
        ids = ID_RE.findall(sent)
        scope = [i for i in ids if i in scopes] or default_scope
        for clause in CLAUSE_SPLIT.split(sent):
            if not clause:
                continue
            cl_ids = [i for i in ID_RE.findall(clause) if i in scopes]
            cscope = cl_ids or scope
            srcs = clause_sources(clause)
            mclause = mask_ids(clause)
            # --- attributed "k/10", "k of 10", "k out of 10" claims
            for m in re.finditer(r"(?<![\d./#])(\d+)\s*(?:/|out\s+of|of)\s*(10)\b", mclause):
                k = int(m.group(1))
                allowed = set()
                for sc in cscope:
                    for so in srcs:
                        allowed |= scopes[sc][so]
                crecs.append({"claim": m.group(0), "sources": srcs, "scope": cscope if len(cscope) <= 3 else f"{len(cscope)} ckpts",
                              "wrong": k not in allowed, "allowed": sorted(allowed), "clause": clause[:250]})
            # --- failure-type mentions
            for t, pat in FT_PATTERNS.items():
                for m in re.finditer(pat, mclause, re.I):
                    before = mclause[:m.start()]
                    negated = bool(NEG_CUE.search(before))
                    sim_c = {scopes[sc]["type_sim"][t] for sc in cscope}
                    jud_c = {scopes[sc]["type_judged"][t] for sc in cscope}
                    if srcs == ["nemotron"]:
                        att = jud_c
                    elif srcs == ["sim"]:
                        att = sim_c
                    else:
                        att = sim_c | jud_c
                    occurs = any(c > 0 for c in sim_c | jud_c)
                    cnt = None
                    mb = re.search(r"(?<![\d./#])(\d+|" + "|".join(WORDNUM) + r")\s+(?:[A-Za-z']+\s+){0,2}$", before, re.I)
                    ma = re.match(r"\s*(?:\(|:|=)\s*(\d+)(?![.\d/])", mclause[m.end():m.end() + 20])
                    if mb:
                        tok = mb.group(1).lower()
                        cnt = int(tok) if tok.isdigit() else WORDNUM.get(tok)
                    elif ma:
                        cnt = int(ma.group(1))
                    if negated:
                        wrong = len(cscope) == 1 and max(sim_c) > 0 and "nemotron" not in srcs
                        reason = "negates a type the simulator recorded" if wrong else ""
                    elif not occurs:
                        wrong, reason = True, "type occurs neither in simulator nor Nemotron labels in scope"
                    elif cnt is not None and cnt not in att:
                        wrong, reason = True, f"count {cnt} not in {srcs} counts {sorted(att)}"
                    else:
                        wrong, reason = False, ""
                    recs.append({"type": t, "span": m.group(0), "negated": negated, "count": cnt, "sources": srcs,
                                 "scope": cscope if len(cscope) <= 3 else f"{len(cscope)} ckpts",
                                 "wrong": wrong, "reason": reason, "sentence": clause[:300]})
    return recs, crecs


def norm_choice(x: Any, choices: List[str]) -> str:
    s = str(x or "").strip().lower().replace(" ", "_").replace("-", "_")
    for c in choices:
        if s == c:
            return c
    for c in choices:
        if c in s:
            return c
    return s or "missing"


def parse_json_obj(raw: str):
    if not raw:
        return None, "empty"
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    try:
        return json.loads(s), "ok"
    except Exception:
        pass
    i, j = s.find("{"), s.rfind("}")
    if i >= 0 and j > i:
        try:
            return json.loads(s[i:j + 1]), "ok_extracted"
        except Exception as e:
            return None, f"invalid_json: {type(e).__name__}"
    return None, "no_json"


def verify_ckpt_report(report: Dict[str, Any], facts: Dict[str, Any], truth: Dict[str, Any],
                       prompt_constants: List[float]):
    flat = flatten(facts)
    texts = text_fields(report)
    full = " ".join(t for _, t in texts)
    nrec = check_numbers(full, facts, flat, extra_allowed=[])
    nums = [r for k, r in nrec if k == "num"]
    cites = [r for k, r in nrec if k == "cite"]
    unsupported = [r for r in nums if not r["supported"]]
    # numbers that only appear in the prompt instructions (e.g. "10 episodes") are tracked separately
    unsup_prompt = [r for r in unsupported if any(num_matches(r["number"].rstrip("%"), r["number"].endswith("%"), v)
                                                  for v in prompt_constants)]
    scopes = {facts["checkpoint_id"]: source_counts(facts)}
    fclaims, cclaims = check_failure_claims(full, scopes, [facts["checkpoint_id"]])
    dom = norm_choice(report.get("dominant_failure_mode"), FTYPES + ["none"])
    rec = norm_choice(report.get("spend_more_rollouts"), ["yes", "no"])
    return {
        "n_numbers": len(nums),
        "n_unsupported": len(unsupported),
        "unsupported": [{"number": r["number"], "context": r["context"]} for r in unsupported],
        "n_unsupported_prompt_constant": len(unsup_prompt),
        "n_citations": len(cites),
        "n_cited_numbers": sum(1 for r in cites if r.get("number") is not None),
        "n_miscited": sum(1 for r in cites if r.get("number") is not None and r.get("key_exists") and not r.get("cite_match")),
        "n_bad_key": sum(1 for r in cites if not r.get("key_exists")),
        "miscited": [r for r in cites if r.get("number") is not None and not r.get("cite_match")],
        "n_uncited_numbers": sum(1 for r in nums if not r["cited"]),
        "n_failure_mentions": len(fclaims),
        "n_wrong_failure_claims": sum(1 for r in fclaims if r["wrong"]),
        "wrong_failure_claims": [r for r in fclaims if r["wrong"]],
        "n_count_claims": len(cclaims),
        "n_misattributed_counts": sum(1 for r in cclaims if r["wrong"]),
        "misattributed_counts": [r for r in cclaims if r["wrong"]],
        "dominant_reported": dom,
        "dominant_correct": dom in truth["dominant_failure_mode_accepted"],
        "recommend_reported": rec,
        "rule": truth["rule_ckpt"],
        "rule_consistent": (rec == truth["rule_ckpt"]) if truth["rule_ckpt"] is not None else None,
    }


def verify_run_report(report: Dict[str, Any], run_facts: Dict[str, Any], truth_run: Dict[str, Any],
                      ckpt_facts: Dict[str, Dict[str, Any]]):
    flat = flatten(run_facts)
    texts = text_fields(report)
    full = " ".join(t for _, t in texts)
    nrec = check_numbers(full, run_facts, flat, extra_allowed=[])
    nums = [r for k, r in nrec if k == "num"]
    cites = [r for k, r in nrec if k == "cite"]
    unsupported = [r for r in nums if not r["supported"]]
    ids = list(run_facts["checkpoints"].keys())
    scopes = {c: source_counts(ckpt_facts[c]) for c in ids}
    fclaims, cclaims = check_failure_claims(full, scopes, ids)
    pick = str(report.get("evaluate_next") or "").strip()
    pick_valid = pick in ids
    mentioned = set(re.findall(r"\b(?:can|square)_s\d+_e\d{3,4}\b", full + " " + pick + " " +
                               " ".join(map(str, report.get("alternatives") or []))))
    return {
        "n_numbers": len(nums),
        "n_unsupported": len(unsupported),
        "unsupported": [{"number": r["number"], "context": r["context"]} for r in unsupported],
        "n_citations": len(cites),
        "n_cited_numbers": sum(1 for r in cites if r.get("number") is not None),
        "n_miscited": sum(1 for r in cites if r.get("number") is not None and r.get("key_exists") and not r.get("cite_match")),
        "n_bad_key": sum(1 for r in cites if not r.get("key_exists")),
        "miscited": [r for r in cites if r.get("number") is not None and not r.get("cite_match")],
        "n_uncited_numbers": sum(1 for r in nums if not r["cited"]),
        "n_failure_mentions": len(fclaims),
        "n_wrong_failure_claims": sum(1 for r in fclaims if r["wrong"]),
        "wrong_failure_claims": [r for r in fclaims if r["wrong"]],
        "n_count_claims": len(cclaims),
        "n_misattributed_counts": sum(1 for r in cclaims if r["wrong"]),
        "misattributed_counts": [r for r in cclaims if r["wrong"]],
        "evaluate_next": pick,
        "pick_valid_id": pick_valid,
        "invented_checkpoint_ids": sorted(m for m in mentioned if m not in ids),
        "rule_accepted": truth_run["evaluate_next_accepted"],
        "rule_consistent": pick in truth_run["evaluate_next_accepted"],
    }
