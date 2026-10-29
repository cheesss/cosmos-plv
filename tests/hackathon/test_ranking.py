import numpy as np
import pandas as pd
import pytest

from cosmos_plv.ranking import evaluate_metric, regret_at_q, spearman


def test_spearman_matches_scipy_with_ties():
    scipy = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(0)
    x = rng.integers(0, 5, 40).astype(float)
    y = rng.normal(size=40)
    assert spearman(x, y) == pytest.approx(scipy.spearmanr(x, y).correlation, abs=1e-12)
    assert np.isnan(spearman([1, 1, 1], [1, 2, 3]))


def test_regret_at_q():
    score = [3, 2, 1, 0]          # higher is better ranked
    succ = [0.5, 0.9, 0.2, 1.0]
    r = regret_at_q(score, succ, 0.5)
    assert r["k"] == 2 and r["regret"] == pytest.approx(0.1)
    assert regret_at_q(score, succ, 1.0)["regret"] == 0.0
    # tie broken by checkpoint path
    r = regret_at_q([1, 1], [0.0, 1.0], 0.5, tiebreak=["b", "a"])
    assert r["regret"] == 0.0


def test_evaluate_metric_orients_costs():
    df = pd.DataFrame({"run_id": ["r"] * 5, "checkpoint_path": list("abcde"),
                       "cost": [5, 4, 3, 2, 1], "success": [0.0, 0.2, 0.4, 0.6, 0.8]})
    rows = evaluate_metric(df, "cost", lower_is_better=True, n_boot=50)
    assert rows[0]["spearman"] == pytest.approx(1.0)
    assert rows[0]["regret@30"] == 0.0
    assert rows[-1]["run_id"] == "mean_over_runs"
