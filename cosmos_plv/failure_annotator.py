"""State-based (verified) outcome and failure-type annotation of a rollout.

Labels come from privileged simulator state recorded by ``sim_probe``.  They
are the ground truth against which Cosmos visual diagnostics are scored.

Failure taxonomy (shared with the Cosmos prompt, see visual_diagnostics.py):
    none          task succeeded
    no_grasp      the gripper never closed on the object
    dropped       the object was lifted, then fell / was released before success
    not_placed    object lifted and still held (or resting high) at timeout,
                  but never placed at the goal
    other         object fell below its starting height (e.g. knocked off the
                  bin) without ever being grasped
"""

from __future__ import annotations

from typing import Dict

import numpy as np

FAILURE_TYPES = ("none", "no_grasp", "dropped", "not_placed", "other")

LIFT_THRESHOLD_M = 0.05  # object raised >= 5 cm above its initial height
REACH_THRESHOLD_M = 0.05  # end effector within 5 cm of the object body
TABLE_RETURN_M = 0.02  # object back within 2 cm of its initial height
FELL_BELOW_M = 0.03  # object ended >= 3 cm below its initial height
SETTLE_STEPS = 10  # objects can spawn above the surface and settle (square nut: ~6 cm)


def annotate_episode(trace: Dict[str, np.ndarray]) -> Dict[str, object]:
    """trace keys: eef_pos (T,3), object_pos (T,3), grasped (T,), success (T,)."""
    eef = np.asarray(trace["eef_pos"], dtype=np.float64)
    obj = np.asarray(trace["object_pos"], dtype=np.float64)
    grasped = np.asarray(trace["grasped"], dtype=bool)
    success = np.asarray(trace["success"], dtype=bool)
    if len(obj) == 0:
        raise ValueError("empty trace")

    # Reference height = settled height over the first SETTLE_STEPS steps (the robot
    # cannot reach the object that early). Using obj[0] mislabelled square nuts,
    # which spawn ~6 cm above the table, as "fell" (fixed 2026-10-29).
    z0 = float(obj[: SETTLE_STEPS + 1, 2].min())
    height = obj[:, 2] - z0
    dist = np.linalg.norm(eef - obj, axis=1)

    succeeded = bool(success.any())
    first_success = int(np.argmax(success)) if succeeded else -1
    upto = first_success + 1 if succeeded else len(obj)

    reached = bool((dist[:upto] < REACH_THRESHOLD_M).any())
    ever_grasped = bool(grasped[:upto].any())
    lifted_mask = height[:upto] >= LIFT_THRESHOLD_M
    lifted_mask[: min(SETTLE_STEPS, len(lifted_mask))] = False  # spawn height is not a lift
    lifted = bool(lifted_mask.any())
    max_height = float(height[min(SETTLE_STEPS, upto - 1):upto].max())
    displacement_xy = float(np.linalg.norm(obj[upto - 1, :2] - obj[0, :2]))

    dropped = False
    if lifted and not succeeded:
        first_lift = int(np.argmax(lifted_mask))
        after = height[first_lift:]
        dropped = bool((after < TABLE_RETURN_M).any()) or not bool(grasped[-1])

    if succeeded:
        failure = "none"
    elif not ever_grasped:
        failure = "other" if float(height[upto - 1]) <= -FELL_BELOW_M else "no_grasp"
    elif dropped:
        failure = "dropped"
    elif lifted:
        failure = "not_placed"
    else:
        # grasp contact but never lifted 5 cm
        failure = "no_grasp"

    return {
        "success": succeeded,
        "first_success_step": first_success,
        "reached": reached,
        "grasped": ever_grasped,
        "lifted": lifted,
        "dropped": dropped,
        "max_object_height_m": round(max_height, 4),
        "object_xy_displacement_m": round(displacement_xy, 4),
        "failure_type": failure,
        "n_steps": int(len(obj)),
    }
