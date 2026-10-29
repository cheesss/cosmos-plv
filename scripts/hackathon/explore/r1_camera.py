#!/usr/bin/env python3
"""r1 helper (Environment A): agentview camera matrix at the recorded video size (448x448), per task.
The camera is static, so one matrix per task serves every episode. Output: results/hackathon/explore/r1_cam_T.json"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from cosmos_plv.rollout_recorder import _make_env  # noqa: E402
from robosuite.utils.camera_utils import get_camera_transform_matrix  # noqa: E402

OBS_KEYS = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
out = {}
for task in ("can", "square"):
    env = _make_env(ROOT / f"data/robomimic/datasets/{task}/ph/low_dim_abs.hdf5", OBS_KEYS, True, False, (448, 448), "agentview")
    env.seed(0)
    env.reset()
    out[task] = np.asarray(get_camera_transform_matrix(env.env.env.sim, "agentview", 448, 448)).tolist()
(ROOT / "results/hackathon/explore/r1_cam_T.json").write_text(json.dumps(out))
print("ok", list(out))
