import json

import pytest

from cosmos_plv import visual_diagnostics as vd

GOOD = {
    "contact_observed": True, "object_lifted": True, "object_dropped": False,
    "task_success": True, "failure_type": "none", "failure_time_s": None,
    "explanation": "lifted and placed", "uncertainty_note": "",
}


def test_prompt_variants_build_for_all_tasks():
    for task in ("can", "square"):
        for v in ("v1_task_first", "v2_evidence_first", "v3_evidence_lastframe"):
            p = vd.build_prompt(task, 4, 10.0, True, v)
            assert "failure_type" in p and "</think>" in p
        assert "object_location" in vd.build_perception_prompt(task)
        assert vd.TASK_DESCRIPTIONS[task] not in vd.build_perception_prompt(task)  # task-free
    with pytest.raises(KeyError):
        vd.build_prompt("can", 4, variant="nope")
    with pytest.raises(KeyError):
        vd.build_prompt("lift", 4)


def test_parse_ok_after_think_block_and_fence():
    raw = "<think>reasoning {not json}</think>\n```json\n" + json.dumps(GOOD) + "\n```"
    p = vd.parse_diagnostic(raw)
    assert p.parse_status == "ok" and p.fields["task_success"] is True
    assert p.reasoning.startswith("reasoning")


def test_json_inside_think_only_is_not_used():
    raw = "<think>" + json.dumps(GOOD) + "</think> I am not sure."
    assert vd.parse_diagnostic(raw).parse_status == "no_json"


def test_bracket_repair_is_recorded_and_limited():
    body = json.dumps(GOOD, indent=1)[:-1] + "]"
    p = vd.parse_diagnostic("```json\n" + body + "\n```")
    assert p.parse_status == "ok_repaired"
    assert any("repaired" in e for e in p.errors)
    assert vd.parse_diagnostic(body, allow_repair=False).parse_status == "no_json"
    # a second '[' anywhere -> no repair
    bad = body.replace('"lifted and placed"', '"at [1, 2]"')
    assert vd.parse_diagnostic(bad).parse_status == "no_json"


@pytest.mark.parametrize("field,value,status", [
    ("failure_type", "slipped", "bad_enum"),
    ("task_success", "yes", "bad_types"),
    ("failure_time_s", True, "bad_types"),
])
def test_invalid_fields(field, value, status):
    obj = dict(GOOD, **{field: value})
    assert vd.parse_diagnostic(json.dumps(obj)).parse_status == status


def test_missing_and_invalid_json():
    obj = dict(GOOD)
    obj.pop("explanation")
    assert vd.parse_diagnostic(json.dumps(obj)).parse_status == "missing_fields"
    assert vd.parse_diagnostic("{not: json}").parse_status == "invalid_json"
    assert vd.parse_diagnostic("").parse_status == "no_json"


def test_inconsistency_is_reported_not_repaired():
    obj = dict(GOOD, failure_type="dropped")
    p = vd.parse_diagnostic(json.dumps(obj))
    assert p.parse_status == "ok" and p.fields["failure_type"] == "dropped"
    assert any("inconsistent" in e for e in p.errors)


def test_agreement_row():
    p = vd.parse_diagnostic(json.dumps(GOOD))
    truth = {"success": False, "failure_type": "dropped", "grasped": True, "lifted": True, "dropped": True}
    a = vd.agreement_row(p, truth)
    assert a["scored"] and not a["success_correct"] and a["lifted_correct"] and not a["dropped_correct"]
    assert vd.agreement_row(vd.parse_diagnostic("x"), truth) == {"scored": False}


def _perc(loc, held):
    return vd.parse_perception(json.dumps({"object_location": loc, "held_by_gripper": held, "explanation": "."}), "can")


def test_perception_parser_and_success_rule():
    assert _perc("right_bin", False).fields["task_success"] is True
    assert _perc("left_bin", False).fields["task_success"] is False
    assert _perc("on_peg", False).parse_status == "bad_enum"  # square label used for can
    p = vd.parse_perception('[{"object_location": "left_bin", "held_by_gripper": false, "explanation": "x"}]', "can")
    assert p.parse_status == "ok"


@pytest.mark.parametrize("seq,expected", [
    ([("left_bin", False)] * 4, "no_grasp"),
    ([("left_bin", False), ("in_gripper", True), ("left_bin", False), ("left_bin", False)], "dropped"),
    ([("left_bin", False), ("in_gripper", True), ("in_gripper", True)], "not_placed"),
    ([("left_bin", False), ("in_gripper", True), ("right_bin", False)], "none"),
])
def test_track_aggregation_rules(seq, expected):
    frames = [_perc(l, h) for l, h in seq]
    out = vd.aggregate_track("can", frames, [float(i) for i in range(len(frames))])
    assert out.parse_status == "ok"
    assert out.fields["failure_type"] == expected
    assert out.fields["task_success"] == (expected == "none")


def test_track_unscored_when_last_frame_unparsed():
    frames = [_perc("left_bin", False), vd.parse_perception("garbage", "can")]
    assert vd.aggregate_track("can", frames, [0.0, 1.0]).parse_status == "last_frame_unparsed"
