import numpy as np
import pytest

from cosmos_plv.cheap_metrics import executed_smoothness, replan_alignment, replan_metrics_from_plans


def test_paper_alignment_for_dp_chunking():
    al = replan_alignment(16, 2, 8)
    assert al["old_slice"] == (9, 16) and al["new_slice"] == (1, 8) and al["overlap"] == 7
    assert al["boundary_old_index"] == 8 and al["boundary_new_index"] == 1
    with pytest.raises(ValueError):
        replan_alignment(8, 2, 8)


def test_consistent_plans_have_zero_disagreement():
    # a time-indexed signal: plan made at step t predicts value (t + j - 1)
    T, H, D = 6, 16, 2
    plan_t = np.arange(T) * 8
    ap = np.stack([np.stack([np.full(D, t + j - 1.0) for j in range(H)]) for t in plan_t])
    m = replan_metrics_from_plans(ap, plan_t, 16, 2, 8)
    assert m["sim_replan_overlap_l1"] == pytest.approx(0.0)
    assert m["sim_replan_pairs"] == T - 1
    # boundary jump between consecutive executed steps is exactly 1
    assert m["sim_boundary_jump_l1"] == pytest.approx(1.0)


def test_shifted_replan_and_nonconsecutive_calls():
    ap = np.zeros((3, 16, 1)); ap[1] += 2.0
    m = replan_metrics_from_plans(ap, np.array([0, 8, 20]), 16, 2, 8)
    assert m["sim_replan_pairs"] == 1 and m["sim_replan_overlap_l1"] == pytest.approx(2.0)
    with pytest.raises(ValueError):
        replan_metrics_from_plans(np.zeros((2, 10, 1)), np.array([0, 8]), 16, 2, 8)


def test_executed_smoothness():
    lin = np.zeros((10, 7)); lin[:, 0] = np.arange(10)
    assert executed_smoothness(lin) == pytest.approx(0.0)
    assert np.isnan(executed_smoothness(lin[:2]))
