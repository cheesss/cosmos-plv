"""Stage E: structured physical-failure diagnostics from a vision-language model.

This module is dependency-free (stdlib only) so it can be imported from both
Environment A (PLV, py3.10) and Environment B (Cosmos, py3.11).

The model is asked for ONE JSON object.  ``parse_diagnostic`` extracts it from
the raw generation (after an optional <think>...</think> block), validates
every field, and never invents missing values: invalid outputs are kept with
``parse_status`` not in ("ok", "ok_repaired") and ``scored=False``.  They are
left out of the parsed-only metrics (``success_acc``, ``balanced_acc``) but
stay in the all-episode metrics (``success_acc_all``, ``balanced_acc_all``),
where each one counts as a wrong prediction.

Model outputs are predictions.  Ground truth comes from failure_annotator.py.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

FAILURE_TYPES = ("none", "no_grasp", "dropped", "not_placed", "other")

SCHEMA: Dict[str, Tuple[type, ...]] = {
    "contact_observed": (bool,),
    "object_lifted": (bool,),
    "object_dropped": (bool,),
    "task_success": (bool,),
    "failure_type": (str,),
    "failure_time_s": (int, float, type(None)),
    "explanation": (str,),
    "uncertainty_note": (str,),
}

SYSTEM_PROMPT = "You are a helpful assistant."

TASK_DESCRIPTIONS = {
    "can": (
        "A single robot arm should pick up the red soda can that starts in the light wooden bin "
        "on the LEFT, carry it, and place it into the FRONT-RIGHT compartment of the dark wooden "
        "bin on the RIGHT (the compartment that already shows a small grey can-shaped marker)."
    ),
    "square": (
        "A single robot arm should pick up the square nut (a flat square ring with a handle) and "
        "slide it down onto the matching square peg so that the peg passes through the nut's hole."
    ),
}

FAILURE_DEFINITIONS = (
    '"none": the task was completed.\n'
    '"no_grasp": the gripper never securely closed on the object (missed, pushed, or only touched it).\n'
    '"dropped": the object was lifted but then fell or slipped out of the gripper before the task was completed.\n'
    '"not_placed": the object was lifted and carried or held, but was not placed at the goal before the video ends.\n'
    '"other": a failure that fits none of the above (for example the object was knocked over or out of the bin).'
)

ANSWER_TEMPLATE = (
    "{\n"
    '  "contact_observed": true or false,\n'
    '  "object_lifted": true or false,\n'
    '  "object_dropped": true or false,\n'
    '  "task_success": true or false,\n'
    '  "failure_type": "none" | "no_grasp" | "dropped" | "not_placed" | "other",\n'
    '  "failure_time_s": number of seconds from the start when the failure first became visible, or null,\n'
    '  "explanation": "one or two sentences describing what physically happened",\n'
    '  "uncertainty_note": "what was hard to see, e.g. occlusion of the gripper"\n'
    "}"
)

REASONING_SUFFIX = (
    "Answer the question using the following format:\n\n"
    "<think>\nYour reasoning.\n</think>\n\n"
    "Write your final answer immediately after the </think> tag."
)


# c1 amendment A1: the simulator's Can success needs the front-right compartment; v2's sentence does not say so
END_STATE_COMPARTMENT = {
    "can": ("Success means that at the END of the attempt the red can rests inside the FRONT-RIGHT compartment of the "
            "dark wooden bin on the right, the compartment with the small grey can-shaped marker. A can resting in any "
            "other compartment of that bin is a failure (not_placed)."),
}
END_STATE = {
    "can": "Success means that at the END of the attempt the red can rests inside the dark wooden bin on the right.",
    "square": "Success means that at the END of the attempt the square nut sits around the square peg, low on the peg.",
}

OBJECT_NAME = {"can": "red can", "square": "square nut"}

PROMPT_VARIANTS = (
    "v1_task_first",
    "v2_evidence_first",
    "v2c_compartment",
    "v3_evidence_lastframe",
    "v4_keyframes",
    "v5_endstate",
    "v6_perception_last",
    "v7_perception_track",
)

# v6: task-free perception of ONE image; success is derived by a fixed rule.
PERCEPTION_LOCATIONS = {
    "can": ("left_bin", "right_bin", "in_gripper", "outside_bins", "not_visible"),
    "square": ("on_peg", "in_gripper", "on_table", "off_table", "not_visible"),
}
PERCEPTION_SCENE = {
    "can": (
        "The image shows a robot arm above two bins: a light wooden bin on the left and a dark wooden bin "
        "with four compartments on the right. There is one red soda can somewhere in the scene "
        "(ignore the small grey can-shaped marker inside the dark bin)."
    ),
    "square": (
        "The image shows a robot arm above a table with two vertical pegs (one square, one round) and a "
        "flat square nut with a handle."
    ),
}
PERCEPTION_HELP = {
    "can": (
        '"left_bin" = the red can rests inside the light wooden bin on the left; '
        '"right_bin" = it rests inside the dark wooden bin on the right; '
        '"in_gripper" = it is held between the gripper fingers, lifted above the bins; '
        '"outside_bins" = it lies outside both bins; "not_visible" = it cannot be seen'
    ),
    "square": (
        '"on_peg" = the nut is around one of the pegs; "in_gripper" = it is held between the gripper '
        'fingers above the table; "on_table" = it lies on the table, not around a peg; '
        '"off_table" = it fell off the table; "not_visible" = it cannot be seen'
    ),
}
SUCCESS_LOCATION = {"can": "right_bin", "square": "on_peg"}
PERCEPTION_SCHEMA: Dict[str, Tuple[type, ...]] = {
    "object_location": (str,),
    "held_by_gripper": (bool,),
    "explanation": (str,),
}

LOCATIONS = ("left_bin", "right_bin", "in_gripper", "table_or_floor", "not_visible")
LOCATION_HELP = {
    "can": (
        '"left_bin" = resting inside the light wooden bin on the left; '
        '"right_bin" = resting inside the dark wooden bin on the right; '
        '"in_gripper" = held between the gripper fingers above the bins; '
        '"table_or_floor" = outside both bins; "not_visible" = cannot be seen'
    ),
    "square": (
        '"left_bin" = lying on the table away from the pegs; "right_bin" = around one of the pegs; '
        '"in_gripper" = held between the gripper fingers; "table_or_floor" = fallen off the table; '
        '"not_visible" = cannot be seen'
    ),
}

ENDSTATE_SCHEMA: Dict[str, Tuple[type, ...]] = {
    "object_location_first": (str,),
    "object_location_last": (str,),
    "object_held_last": (bool,),
    "task_success": (bool,),
    "explanation": (str,),
}

ENDSTATE_TEMPLATE = (
    "{\n"
    '  "object_location_first": one of "left_bin" | "right_bin" | "in_gripper" | "table_or_floor" | "not_visible",\n'
    '  "object_location_last": one of the same values,\n'
    '  "object_held_last": true or false,\n'
    '  "task_success": true or false,\n'
    '  "explanation": "one sentence"\n'
    "}"
)


def build_prompt(
    task: str,
    fps: float,
    duration_s: Optional[float] = None,
    reasoning: bool = True,
    variant: str = "v1_task_first",
) -> str:
    """Prompt variants (all ask for the same JSON schema).

    v1_task_first        task description, then failure taxonomy, then JSON.
    v2_evidence_first    warns that attempts often fail, asks the model to check the
                         object's start/end position and whether it was carried
                         before judging; success is defined by the end state.
    v2c_compartment      v2 with the compartment-explicit Can end state (c1 amendment A1).
    v3_evidence_lastframe  v2 + an extra still image of the final frame is attached
                         (the caller adds the image; the text refers to it).
    """
    if task not in TASK_DESCRIPTIONS:
        raise KeyError(f"no task description for {task!r}")
    if variant not in PROMPT_VARIANTS:
        raise KeyError(f"unknown prompt variant {variant!r}")
    dur = f" The clip lasts about {duration_s:.1f} seconds." if duration_s else ""
    if variant == "v1_task_first":
        text = (
            "The video shows one episode of a simulated robot manipulation task, recorded by a fixed "
            f"third-person camera and sampled at {fps:g} frames per second.{dur}\n"
            f"Task: {TASK_DESCRIPTIONS[task]}\n\n"
            "Watch the whole episode and judge what physically happened to the object. "
            "Use these failure types:\n"
            f"{FAILURE_DEFINITIONS}\n\n"
        )
    else:
        obj = OBJECT_NAME[task]
        still = (
            " After the video, one extra still image shows the FINAL frame of the same episode."
            if variant == "v3_evidence_lastframe"
            else ""
        )
        text = (
            "The video shows one ATTEMPT by a learned robot policy in a simulated manipulation task, "
            f"recorded by a fixed third-person camera and sampled at {fps:g} frames per second.{dur}{still}\n"
            "Many attempts fail. Judge only from what is visible in the frames; do not assume that the "
            "robot did what it was supposed to do.\n"
            f"Intended goal of the attempt: {TASK_DESCRIPTIONS[task]}\n"
            f"{END_STATE_COMPARTMENT.get(task, END_STATE[task]) if variant == 'v2c_compartment' else END_STATE[task]}\n\n"
            f"Before answering, check: (1) where the {obj} is in the first frame; (2) whether the gripper "
            f"fingers ever close around the {obj}; (3) whether the {obj} is ever carried in the air, clearly "
            f"above the bin floor; (4) where the {obj} is in the last frame.\n"
            "Failure types:\n"
            f"{FAILURE_DEFINITIONS}\n\n"
        )
    text += "Return exactly one JSON object with these fields and no other text after it:\n" + ANSWER_TEMPLATE
    if reasoning:
        text = text + "\n\n" + REASONING_SUFFIX
    return text


def build_keyframe_prompt(task: str, times_s: List[float], reasoning: bool = True) -> str:
    """v4: K still images (in time order) instead of a video; same JSON schema."""
    obj = OBJECT_NAME[task]
    stamps = ", ".join(f"image {i + 1} at t={t:.1f}s" for i, t in enumerate(times_s))
    text = (
        f"The {len(times_s)} images are keyframes, in time order, from ONE attempt by a learned robot policy "
        f"in a simulated manipulation task, all from the same fixed camera ({stamps}).\n"
        "Many attempts fail. Judge only from what is visible; do not assume that the robot did what it was "
        "supposed to do.\n"
        f"Intended goal of the attempt: {TASK_DESCRIPTIONS[task]}\n{END_STATE[task]}\n\n"
        f"Before answering, locate the {obj} in every image, note any image where it is held in the air "
        f"by the gripper, and compare its position in the first and the last image.\n"
        "Failure types:\n"
        f"{FAILURE_DEFINITIONS}\n\n"
        "Return exactly one JSON object with these fields and no other text after it:\n" + ANSWER_TEMPLATE
    )
    if reasoning:
        text = text + "\n\n" + REASONING_SUFFIX
    return text


def build_endstate_prompt(task: str, reasoning: bool = True) -> str:
    """v5: two images (first and last frame); locate the object, then judge success."""
    obj = OBJECT_NAME[task]
    text = (
        "Image 1 is the FIRST frame and image 2 is the LAST frame of one attempt by a learned robot policy "
        "in a simulated manipulation task, from the same fixed camera.\n"
        f"Intended goal: {TASK_DESCRIPTIONS[task]}\n{END_STATE[task]}\n"
        "Many attempts fail; judge only from what is visible.\n\n"
        f"Locate the {obj} in each image using exactly one of: {LOCATION_HELP[task]}.\n"
        "Return exactly one JSON object with these fields and no other text after it:\n" + ENDSTATE_TEMPLATE
    )
    if reasoning:
        text = text + "\n\n" + REASONING_SUFFIX
    return text


@dataclass
class ParsedDiagnostic:
    # valid: ok | ok_repaired; invalid: invalid_json | missing_fields | bad_types | bad_enum
    # | no_json | last_frame_unparsed (v7)
    parse_status: str
    fields: Dict[str, Any]
    errors: List[str]
    reasoning: Optional[str]
    answer_text: str

    def to_row(self) -> Dict[str, Any]:
        row = {"parse_status": self.parse_status, "parse_errors": "; ".join(self.errors)}
        for k in SCHEMA:
            row[f"cosmos_{k}"] = self.fields.get(k)
        return row


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)


def split_reasoning(raw: str) -> Tuple[Optional[str], str]:
    m = _THINK_RE.search(raw)
    if m:
        return m.group(1).strip(), raw[m.end() :].strip()
    if "</think>" in raw:  # template may pre-fill the opening tag
        head, tail = raw.split("</think>", 1)
        return head.replace("<think>", "").strip(), tail.strip()
    return None, raw.strip()


def _first_json_object(text: str) -> Optional[str]:
    """Return the first balanced {...} substring (string-aware)."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        start = text.find("{", start + 1)
    return None


def _repair_closing_bracket(answer: str) -> Optional[str]:
    """Single allowed repair: an object opened with '{' but closed with ']'.

    Observed with Cosmos3-Edge (e.g. '... "uncertainty_note": "..."\n]').  Only the
    final closing character is changed; nothing else is edited or inferred.
    """
    start = answer.find("{")
    if start == -1:
        return None
    body = re.sub(r"```\s*$", "", answer[start:].strip()).rstrip()
    if body.endswith("]") and body.count("{") == 1 and body.count("[") == 0:
        return body[:-1] + "}"
    return None


def parse_diagnostic(raw: str, allow_repair: bool = True) -> ParsedDiagnostic:
    """Strict parse; optionally one recorded bracket repair (status 'ok_repaired')."""
    reasoning, answer = split_reasoning(raw or "")
    blob = _first_json_object(answer)
    repaired = False
    if blob is None and allow_repair:
        blob = _repair_closing_bracket(answer)
        repaired = blob is not None
    if blob is None and reasoning is not None:
        # Some generations put the JSON inside the think block only; do not use it
        # as an answer, but record why parsing failed.
        return ParsedDiagnostic("no_json", {}, ["no JSON object after </think>"], reasoning, answer)
    if blob is None:
        return ParsedDiagnostic("no_json", {}, ["no JSON object in output"], reasoning, answer)
    fence = re.sub(r"^```(?:json)?|```$", "", blob.strip())
    try:
        obj = json.loads(fence)
    except json.JSONDecodeError as e:
        return ParsedDiagnostic("invalid_json", {}, [f"json: {e}"], reasoning, answer)
    if not isinstance(obj, dict):
        return ParsedDiagnostic("invalid_json", {}, ["top level is not an object"], reasoning, answer)
    errors: List[str] = ["repaired: closing ']' -> '}'"] if repaired else []
    missing = [k for k in SCHEMA if k not in obj]
    if missing:
        errors.append("missing: " + ",".join(missing))
    for k, types in SCHEMA.items():
        if k in obj and not isinstance(obj[k], types):
            # bool is a subclass of int: reject bools for numeric fields explicitly
            errors.append(f"type {k}={type(obj[k]).__name__}")
        elif k == "failure_time_s" and isinstance(obj.get(k), bool):
            errors.append("type failure_time_s=bool")
    ft = obj.get("failure_type")
    if isinstance(ft, str) and ft not in FAILURE_TYPES:
        errors.append(f"enum failure_type={ft!r}")
    fields = {k: obj.get(k) for k in SCHEMA}
    if missing:
        status = "missing_fields"
    elif any(e.startswith("type") for e in errors):
        status = "bad_types"
    elif any(e.startswith("enum") for e in errors):
        status = "bad_enum"
    else:
        status = "ok_repaired" if repaired else "ok"
    # Logical consistency is reported, not repaired.
    if status in ("ok", "ok_repaired"):
        if fields["task_success"] and fields["failure_type"] != "none":
            errors.append("inconsistent: task_success with failure_type != none")
        if (not fields["task_success"]) and fields["failure_type"] == "none":
            errors.append("inconsistent: failure without failure_type")
    return ParsedDiagnostic(status, fields, errors, reasoning, answer)


def parse_endstate(raw: str) -> ParsedDiagnostic:
    """Parser for the v5 end-state schema (same repair rule as parse_diagnostic)."""
    reasoning, answer = split_reasoning(raw or "")
    blob = _first_json_object(answer)
    repaired = False
    if blob is None:
        blob = _repair_closing_bracket(answer)
        repaired = blob is not None
    if blob is None:
        return ParsedDiagnostic("no_json", {}, ["no JSON object in answer"], reasoning, answer)
    try:
        obj = json.loads(re.sub(r"^```(?:json)?|```$", "", blob.strip()))
    except json.JSONDecodeError as e:
        return ParsedDiagnostic("invalid_json", {}, [f"json: {e}"], reasoning, answer)
    if not isinstance(obj, dict):
        return ParsedDiagnostic("invalid_json", {}, ["top level is not an object"], reasoning, answer)
    errors: List[str] = ["repaired: closing ']' -> '}'"] if repaired else []
    missing = [k for k in ENDSTATE_SCHEMA if k not in obj]
    bad = [k for k, t in ENDSTATE_SCHEMA.items() if k in obj and not isinstance(obj[k], t)]
    enum = [k for k in ("object_location_first", "object_location_last")
            if isinstance(obj.get(k), str) and obj[k] not in LOCATIONS]
    status = ("missing_fields" if missing else "bad_types" if bad else "bad_enum" if enum
              else ("ok_repaired" if repaired else "ok"))
    errors += [f"missing {k}" for k in missing] + [f"type {k}" for k in bad] + [f"enum {k}" for k in enum]
    fields = {k: obj.get(k) for k in ENDSTATE_SCHEMA}
    if status in ("ok", "ok_repaired"):
        implied = fields["object_location_last"] == "right_bin"
        if implied != bool(fields["task_success"]):
            errors.append("inconsistent: task_success vs object_location_last")
    return ParsedDiagnostic(status, fields, errors, reasoning, answer)


def build_perception_prompt(task: str, reasoning: bool = False) -> str:
    """v6: no task goal, no success question; only where the object is."""
    obj = OBJECT_NAME[task]
    locs = " | ".join(f'"{x}"' for x in PERCEPTION_LOCATIONS[task])
    text = (
        f"{PERCEPTION_SCENE[task]}\n"
        f"Describe only what is visible in this single image. Where is the {obj}? "
        f"Use exactly one of: {PERCEPTION_HELP[task]}.\n"
        "Return exactly one JSON object and no other text after it:\n"
        "{\n"
        f'  "object_location": {locs},\n'
        '  "held_by_gripper": true or false,\n'
        '  "explanation": "one sentence about what you see"\n'
        "}"
    )
    if reasoning:
        text = text + "\n\n" + REASONING_SUFFIX
    return text


def parse_perception(raw: str, task: str) -> ParsedDiagnostic:
    """Parser for v6; adds derived ``task_success`` = (location == success location)."""
    reasoning, answer = split_reasoning(raw or "")
    blob = _first_json_object(answer)
    repaired = False
    if blob is None:
        blob = _repair_closing_bracket(answer)
        repaired = blob is not None
    if blob is None:
        return ParsedDiagnostic("no_json", {}, ["no JSON object in answer"], reasoning, answer)
    try:
        obj = json.loads(re.sub(r"^```(?:json)?|```$", "", blob.strip()))
    except json.JSONDecodeError as e:
        return ParsedDiagnostic("invalid_json", {}, [f"json: {e}"], reasoning, answer)
    if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
        obj = obj[0]
    if not isinstance(obj, dict):
        return ParsedDiagnostic("invalid_json", {}, ["top level is not an object"], reasoning, answer)
    errors: List[str] = ["repaired: closing ']' -> '}'"] if repaired else []
    missing = [k for k in PERCEPTION_SCHEMA if k not in obj]
    bad = [k for k, t in PERCEPTION_SCHEMA.items() if k in obj and not isinstance(obj[k], t)]
    loc = obj.get("object_location")
    enum = isinstance(loc, str) and loc not in PERCEPTION_LOCATIONS[task]
    status = ("missing_fields" if missing else "bad_types" if bad else "bad_enum" if enum
              else ("ok_repaired" if repaired else "ok"))
    errors += [f"missing {k}" for k in missing] + [f"type {k}" for k in bad] + ([f"enum {loc!r}"] if enum else [])
    fields = {k: obj.get(k) for k in PERCEPTION_SCHEMA}
    fields["task_success"] = (loc == SUCCESS_LOCATION[task]) if status in ("ok", "ok_repaired") else None
    return ParsedDiagnostic(status, fields, errors, reasoning, answer)


def aggregate_track(task: str, frames: List[ParsedDiagnostic], times_s: List[float]) -> ParsedDiagnostic:
    """v7: derive the main-schema diagnostic from per-keyframe perceptions.

    Rules mirror failure_annotator.py, applied to *visual* states:
      success     last location == success location
      lifted      any frame held_by_gripper or location == in_gripper
      no_grasp    never held
      dropped     held at some frame, not held at the end, not success
      not_placed  still held at the end, not success
    Frames whose perception failed to parse are skipped; if the LAST frame
    failed, the episode is not scored.
    """
    valid = [(t, f) for t, f in zip(times_s, frames) if f.parse_status in ("ok", "ok_repaired")]
    errors = [f"frame {i}: {f.parse_status}" for i, f in enumerate(frames) if f.parse_status not in ("ok", "ok_repaired")]
    if not frames or frames[-1].parse_status not in ("ok", "ok_repaired"):
        return ParsedDiagnostic("last_frame_unparsed", {}, errors or ["no frames"], None, "")
    held = [bool(f.fields["held_by_gripper"]) or f.fields["object_location"] == "in_gripper" for _, f in valid]
    last = valid[-1][1].fields["object_location"]
    success = last == SUCCESS_LOCATION[task]
    ever_held = any(held)
    held_last = held[-1]
    if success:
        ftype = "none"
    elif not ever_held:
        ftype = "no_grasp"
    elif held_last:
        ftype = "not_placed"
    else:
        ftype = "dropped"
    fail_t = None
    if not success and ever_held and not held_last:
        last_held_idx = max(i for i, h in enumerate(held) if h)
        if last_held_idx + 1 < len(valid):
            fail_t = float(valid[last_held_idx + 1][0])
    seq = ", ".join(f"t={t:.1f}s:{f.fields['object_location']}{'(held)' if h else ''}" for (t, f), h in zip(valid, held))
    fields = {
        "contact_observed": ever_held,
        "object_lifted": ever_held,
        "object_dropped": ftype == "dropped",
        "task_success": success,
        "failure_type": ftype,
        "failure_time_s": fail_t,
        "explanation": f"Per-frame object locations: {seq}.",
        "uncertainty_note": "; ".join(errors) if errors else "",
    }
    status = "ok_repaired" if any(f.parse_status == "ok_repaired" for _, f in valid) else "ok"
    return ParsedDiagnostic(status, fields, errors, None, seq)


def agreement_row(parsed: ParsedDiagnostic, truth: Dict[str, Any]) -> Dict[str, Any]:
    """Compare a parsed diagnostic with verified simulator annotations."""
    out: Dict[str, Any] = {}
    if parsed.parse_status not in ("ok", "ok_repaired"):
        return {"scored": False}
    f = parsed.fields
    out["scored"] = True
    out["success_correct"] = bool(f["task_success"]) == bool(truth["success"])
    if "failure_type" not in f:  # v5 end-state schema scores success only
        return out
    out["failure_type_correct"] = f["failure_type"] == truth["failure_type"]
    out["lifted_correct"] = bool(f["object_lifted"]) == bool(truth["lifted"])
    out["contact_correct"] = bool(f["contact_observed"]) == bool(truth["grasped"])
    out["dropped_correct"] = bool(f["object_dropped"]) == bool(truth["dropped"])
    return out
