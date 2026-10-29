import numpy as np
import pytest

from cosmos_plv.failure_annotator import FAILURE_TYPES, annotate_episode


def trace(heights, grasp, success=None, eef_offset=0.0):
    T = len(heights)
    obj = np.zeros((T, 3))
    obj[:, 2] = 0.8 + np.asarray(heights)
    eef = obj + np.array([0.0, 0.0, eef_offset])
    return {"eef_pos": eef, "object_pos": obj, "grasped": np.asarray(grasp, bool),
            "success": np.zeros(T, bool) if success is None else np.asarray(success, bool)}


def test_success():
    s = [False] * 19 + [True]
    a = annotate_episode(trace([0] * 13 + [0.1] * 7, [0] * 13 + [1] * 7, s))
    assert a["success"] and a["failure_type"] == "none" and a["first_success_step"] == 19 and a["lifted"]


def test_no_grasp_and_other():
    assert annotate_episode(trace([0] * 10, [0] * 10, eef_offset=0.3))["failure_type"] == "no_grasp"
    assert annotate_episode(trace([0] * 12 + [-0.05] * 5, [0] * 17))["failure_type"] == "other"
    # contact but never lifted 5 cm
    assert annotate_episode(trace([0.01] * 20, [1] * 20))["failure_type"] == "no_grasp"


def test_dropped_and_not_placed():
    pre = [0.0] * 11
    a = annotate_episode(trace(pre + [0, 0.1, 0.1, 0.0, 0.0], [0] * 11 + [0, 1, 1, 0, 0]))
    assert a["lifted"] and a["dropped"] and a["failure_type"] == "dropped"
    b = annotate_episode(trace(pre + [0, 0.1, 0.1, 0.1], [0] * 11 + [0, 1, 1, 1]))
    assert b["failure_type"] == "not_placed"


def test_types_are_known_and_empty_rejected():
    assert annotate_episode(trace([0] * 3, [0] * 3))["failure_type"] in FAILURE_TYPES
    with pytest.raises(ValueError):
        annotate_episode(trace([], []))


def test_spawn_settling_is_not_a_fall():
    # object spawns 6 cm above its resting height, settles, is never touched
    h = [0.06, 0.03, 0.0] + [0.0] * 30
    a = annotate_episode(trace(h, [0] * len(h), eef_offset=0.3))
    assert a["failure_type"] == "no_grasp"
    # lift measured from the settled height
    h2 = [0.06, 0.0] + [0.0] * 10 + [0.07] * 10
    b = annotate_episode(trace(h2, [0] * 12 + [1] * 10))
    assert b["lifted"] and b["failure_type"] == "not_placed"
