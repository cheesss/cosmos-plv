"""Dataset checks; skipped when the (git-ignored) robomimic files are absent."""
import json
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("task,obs_dim", [("can", 23), ("square", 23)])
def test_lowdim_abs_dataset(task, obs_dim):
    path = ROOT / f"data/robomimic/datasets/{task}/ph/low_dim_abs.hdf5"
    raw = ROOT / f"data/robomimic/datasets/{task}/ph/low_dim.hdf5"
    if not path.exists():
        pytest.skip("dataset not downloaded")
    keys = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
    with h5py.File(path, "r") as f, h5py.File(raw, "r") as g:
        demos = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[-1]))
        assert len(demos) == 200
        train = {x.decode() for x in f["mask/train"][:]}
        valid = {x.decode() for x in f["mask/valid"][:]}
        assert len(train) == 180 and len(valid) == 20 and not (train & valid)
        d = f["data"][demos[0]]
        assert sum(d["obs"][k].shape[1] for k in keys) == obs_dim
        a = d["actions"][:]
        assert a.shape[1] == 7 and np.all(np.isfinite(a))
        # absolute position targets track the achieved end-effector position (metres)
        eef = d["obs"]["robot0_eef_pos"][:]
        assert np.abs(a[:-1, :3] - eef[1:]).mean() < 0.05
        # the source file holds DELTA actions in [-1, 1]; conversion changed them
        assert np.abs(g["data"][demos[0]]["actions"][:]).max() <= 1.0 + 1e-6
        assert not np.allclose(a, g["data"][demos[0]]["actions"][:])
        env = json.loads(f["data"].attrs["env_args"])
        assert env["env_name"] == {"can": "PickPlaceCan", "square": "NutAssemblySquare"}[task]
