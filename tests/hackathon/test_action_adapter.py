import numpy as np
import pytest

from cosmos_plv import action_adapter as aa


def random_actions(T=20, seed=0):
    rng = np.random.default_rng(seed)
    pos = np.cumsum(rng.normal(scale=0.01, size=(T, 3)), axis=0)
    rot = rng.normal(scale=0.3, size=(T, 3))
    grip = np.where(rng.random((T, 1)) > 0.5, 1.0, -1.0)
    return np.concatenate([pos, rot, grip], axis=1)


def test_rot6d_roundtrip():
    R = aa.axis_angle_to_matrix(np.array([[0.1, -0.2, 0.3], [0.0, 0.0, 0.0]]))
    assert np.allclose(aa.rot6d_to_matrix(aa.matrix_to_rot6d(R)), R, atol=1e-9)
    assert np.allclose(R[1], np.eye(3))


def test_relative_conversion_integrates_back():
    a = random_actions()
    rel = aa.absolute_to_relative_rot6d(a)
    assert rel.shape == (len(a) - 1, aa.COSMOS_ACTION_DIM)
    traj = aa.compose_relative(a[0], rel)
    assert np.allclose(traj[:, :3], a[1:, :3], atol=1e-9)
    assert np.allclose(traj[:, 3:].reshape(-1, 3, 3), aa.axis_angle_to_matrix(a[1:, 3:6]), atol=1e-9)


def test_validation_and_blocked_adapter():
    with pytest.raises(ValueError):
        aa.validate_env_actions(np.zeros((5, 10)))  # policy-space actions are not env actions
    with pytest.raises(ValueError):
        aa.validate_env_actions(np.full((3, 7), np.nan))
    assert aa.ADAPTER_STATUS == "blocked"
    with pytest.raises(NotImplementedError):
        aa.to_cosmos_action(random_actions())
