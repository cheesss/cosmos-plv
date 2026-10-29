from pathlib import Path

import pandas as pd
import pytest

from cosmos_plv.evaluation_pipeline import (checkpoint_report, claims_grasp, failure_type_reliability,
                                            latest_epoch_pick, recommend, rule_regrets)

RES = Path(__file__).resolve().parents[2] / "results" / "hackathon"


def _screen():
    return pd.DataFrame({
        "run_id": ["r"] * 6, "checkpoint_id": [f"c{i}" for i in range(6)],
        "checkpoint_path": [f"p{i}" for i in range(6)], "epoch": range(6),
        "val_action_l1": [0.6, 0.1, 0.2, 0.3, 0.4, 0.5],
        "vlm": [0.0, 0.3, 0.9, 1.0, 1.0, 0.1],
        "success": [0.0, 0.2, 0.9, 1.0, 1.0, 0.0]})


def test_two_stage_recommendation_uses_no_outcomes():
    s = _screen()
    r = recommend(s, "r", q=0.34, vlm_col="vlm")
    # top-3 by Action L1 = c1, c2, c3; then ordered by VLM-judged success
    assert r["k"] == 3 and r["retained"] == ["c3", "c2", "c1"]
    assert r["recommended_checkpoint"] == "c3"
    # changing the hidden outcome column must not change the choice
    s2 = s.copy(); s2["success"] = s2["success"][::-1].values
    assert recommend(s2, "r", q=0.34, vlm_col="vlm")["recommended_checkpoint"] == "c3"


def test_cheap_only_fallback_and_errors():
    r = recommend(_screen(), "r", q=0.34)
    assert r["recommended_checkpoint"] == "c1" and "cheap screen only" in r["basis"]
    with pytest.raises(ValueError):
        recommend(_screen(), "missing")


def test_report_mentions_verified_success():
    row = _screen().iloc[2]
    ep = pd.DataFrame({"env_seed": [1], "truth_failure_type": ["none"], "cosmos_task_success": [True],
                       "cosmos_failure_type": ["none"], "backend": ["local"]})
    txt = checkpoint_report(row, ep)
    assert "c2" in txt and "verified training-time success" in txt


def test_vlm_tie_is_reported_and_broken_by_cheap_cost():
    s = _screen()
    s["vlm"] = [0.0, 1.0, 1.0, 1.0, 0.2, 0.1]
    r = recommend(s, "r", q=0.5, vlm_col="vlm")
    # kept c1, c2, c3 all judged 100%: the lowest Action L1 (c1) wins, and the tie is said out loud
    assert r["recommended_checkpoint"] == "c1" and r["vlm_tie"] == ["c1", "c2", "c3"]
    assert any("tie was broken by Action L1" in x for x in r["reasons"])
    assert recommend(_screen(), "r", q=0.34, vlm_col="vlm")["vlm_tie"] == ["c3"]


def test_predicted_failure_modes_are_labelled_unreliable():
    f = pd.DataFrame({"checkpoint_id": ["c3", "c3", "c3"], "cosmos_failure_type": ["no_grasp", "none", None]})
    r = recommend(_screen(), "r", q=0.34, vlm_col="vlm", failures=f)
    assert r["predicted_failure_modes"] == {"no_grasp": 1}
    assert any(x.startswith("predicted failure modes (unreliable") for x in r["reasons"])


def test_latest_epoch_pick_is_outcome_free():
    s = _screen()
    assert latest_epoch_pick(s, "r")["recommended_checkpoint"] == "c5"
    s["success"] = s["success"][::-1].values
    assert latest_epoch_pick(s, "r") == latest_epoch_pick(_screen(), "r")
    with pytest.raises(ValueError):
        latest_epoch_pick(s, "missing")


def test_rule_regrets_mean_of_per_run_pick_regret():
    rank = pd.DataFrame({"metric": ["epoch", "epoch", "epoch", "two", "two", "two"],
                         "label": ["Epoch"] * 3 + ["Two"] * 3, "run_id": ["a", "a", "b", "a", "a", "b"],
                         "regret@30": [0.0, None, None, 0.0, None, None],
                         "pick_regret@1": [None, 0.0, 0.02, None, 0.02, 0.1]})
    r = rule_regrets(rank)
    assert r.metric.tolist() == ["epoch", "two"]
    assert r.mean_pick_regret.round(3).tolist() == [0.01, 0.06] and r.n_runs.tolist() == [2, 2]
    assert r.loc[1, "b"] == pytest.approx(0.1)
    assert rule_regrets(pd.DataFrame()).empty


def test_failure_type_reliability_majority_baseline():
    exp2 = {"overall": {"nebius": [{"failure_type_acc_on_failures": 0.3}], "local_success": [{}]},
            "failure_type_confusion": {"nebius": {"none": {"none": 5}, "no_grasp": {"no_grasp": 2, "dropped": 4},
                                                  "dropped": {"none": 3, "dropped": 1}}}}
    r = failure_type_reliability(exp2)
    assert list(r) == ["nebius"]
    assert r["nebius"]["majority_type"] == "no_grasp" and r["nebius"]["n_failures"] == 10
    assert r["nebius"]["majority_acc"] == pytest.approx(0.6) and r["nebius"]["acc"] == pytest.approx(0.3)
    assert failure_type_reliability(None) == {}


@pytest.mark.parametrize("text", [
    "The robot picked up the red can from the left bin and carried it over to the right bin, but the video ends.",
    "The robot grasped the red can and lifted it, but the video ends with the can still held in the gripper.",
    "The robot arm is holding a red soda can above the bins.",
    "The square nut is visible on the right peg, being held by the robot arm's gripper.",
    "The robot arm moved around, never closed on the can, but later picked it up.",
])
def test_claims_grasp_positive(text):
    assert claims_grasp(text)


@pytest.mark.parametrize("text", [
    "The robot arm moved around the bins but never closed its gripper around the red can.",
    "The gripper touched the red can but never closed around it or lifted it; the can remained in the bin.",
    "The gripper contacts and pushes the red can but never closes securely around it to lift it off the table.",
    "The square nut is positioned on the square peg, ready to be grasped by the robot arm.",
    "The red can is resting inside the light wooden bin on the left, not in the gripper.",
    "The robot failed to grasp the nut.",
    "", None, float("nan"),
])
def test_claims_grasp_negative(text):
    assert not claims_grasp(text)


@pytest.mark.skipif(not (RES / "hybrid_screening.csv").exists(), reason="results not present")
def test_recommend_reproduces_experiment3_two_stage_pick():
    """At q = 0.3 the dashboard rule picks exactly what analyze_hybrid.py scored as the two-stage rule."""
    s = pd.read_csv(RES / "hybrid_screening.csv")
    for b in ("local", "nebius"):
        for run, g in s.groupby("run_id"):
            r = recommend(s, run, q=0.3, vlm_col=f"vlm_success_{b}")
            assert r["recommended_checkpoint"] == g.loc[g[f"hybrid_twostage_l1_{b}"].idxmax(), "checkpoint_id"]


@pytest.mark.skipif(not (RES / "cosmos_diagnostics.csv").exists(), reason="results not present")
def test_grasp_claim_flags_known_conflict():
    d = pd.read_csv(RES / "cosmos_diagnostics.csv")
    r = d[(d.checkpoint_id == "can_s1_e0020") & (d.env_seed == 200006) & (d.backend == "nebius")].iloc[0]
    assert not r.truth_grasped and claims_grasp(r.cosmos_explanation)
