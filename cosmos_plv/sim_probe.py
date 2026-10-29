"""Read privileged simulator state for ground-truth outcome annotation.

These probes read robosuite internals (object body positions, finger-pad
contacts, the task success check). They are used ONLY to produce verified
labels; they are never fed to the policy or to Cosmos.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


class TaskProbe:
    task = "generic"

    def __init__(self, robosuite_env):
        self.env = robosuite_env

    def object_pos(self) -> np.ndarray:
        raise NotImplementedError

    def grasped(self) -> bool:
        raise NotImplementedError

    def success(self) -> bool:
        return bool(self.env._check_success())

    def sample(self) -> Dict[str, object]:
        robot = self.env.robots[0]
        eef = np.array(self.env.sim.data.site_xpos[robot.eef_site_id["right"]], dtype=np.float32)
        return {
            "eef_pos": eef,
            "object_pos": self.object_pos().astype(np.float32),
            "grasped": bool(self.grasped()),
            "success": self.success(),
        }


class CanProbe(TaskProbe):
    task = "can"

    def __init__(self, robosuite_env):
        super().__init__(robosuite_env)
        self.obj = robosuite_env.objects[robosuite_env.object_id]
        self.body_id = robosuite_env.obj_body_id[self.obj.name]

    def object_pos(self) -> np.ndarray:
        return np.array(self.env.sim.data.body_xpos[self.body_id])

    def grasped(self) -> bool:
        return self.env._check_grasp(gripper=self.env.robots[0].gripper, object_geoms=self.obj.contact_geoms)


class SquareProbe(TaskProbe):
    task = "square"

    def __init__(self, robosuite_env):
        super().__init__(robosuite_env)
        self.nut = robosuite_env.nuts[robosuite_env.nut_id]
        self.body_id = robosuite_env.obj_body_id[self.nut.name]

    def object_pos(self) -> np.ndarray:
        return np.array(self.env.sim.data.body_xpos[self.body_id])

    def grasped(self) -> bool:
        return self.env._check_grasp(gripper=self.env.robots[0].gripper, object_geoms=self.nut.contact_geoms)


PROBES = {"can": CanProbe, "square": SquareProbe}


def make_probe(task: str, robosuite_env) -> TaskProbe:
    if task not in PROBES:
        raise KeyError(f"no simulator probe for task {task!r}; known: {sorted(PROBES)}")
    return PROBES[task](robosuite_env)
