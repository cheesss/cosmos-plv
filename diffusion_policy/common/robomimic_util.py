import numpy as np
import copy

import h5py
import robosuite
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
from scipy.spatial.transform import Rotation

from robomimic.config import config_factory


def make_robosuite_metadata_compatible(env_meta):
    version = getattr(robosuite, "__version__", "0.0.0")
    major_minor = tuple(int(x) for x in version.split(".")[:2])
    if major_minor >= (1, 5):
        return env_meta

    env_kwargs = env_meta.get("env_kwargs", {})
    env_kwargs.pop("lite_physics", None)
    controller_configs = env_kwargs.get("controller_configs")
    if isinstance(controller_configs, dict):
        body_parts = controller_configs.get("body_parts")
        if isinstance(body_parts, dict):
            if "right" in body_parts and len(body_parts) == 1:
                env_kwargs["controller_configs"] = body_parts["right"]
            elif {"right", "left"}.issubset(body_parts):
                env_kwargs["controller_configs"] = [body_parts["right"], body_parts["left"]]
    return env_meta


def set_absolute_controller_config(config):
    if isinstance(config, dict):
        if "control_delta" in config:
            config["control_delta"] = False
        if config.get("type") in {"OSC_POSE", "OSC_POSITION"}:
            config["input_type"] = "absolute"
        for value in config.values():
            set_absolute_controller_config(value)
    elif isinstance(config, list):
        for value in config:
            set_absolute_controller_config(value)


def get_arm_controller(robot, arm_idx=0):
    if hasattr(robot, "controller"):
        return robot.controller
    part_controllers = getattr(robot, "part_controllers", None)
    if part_controllers is not None:
        arm_names = [name for name in part_controllers.keys() if "gripper" not in name]
        if arm_names:
            return part_controllers[arm_names[min(arm_idx, len(arm_names) - 1)]]
    raise AttributeError("Could not find an arm controller on robosuite robot")


class RobomimicAbsoluteActionConverter:
    def __init__(self, dataset_path, algo_name='bc'):
        # default BC config
        config = config_factory(algo_name=algo_name)

        # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
        # must ran before create dataset
        ObsUtils.initialize_obs_utils_with_config(config)

        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
        env_meta = make_robosuite_metadata_compatible(env_meta)
        abs_env_meta = copy.deepcopy(env_meta)
        set_absolute_controller_config(abs_env_meta['env_kwargs']['controller_configs'])

        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            render=False, 
            render_offscreen=False,
            use_image_obs=False, 
        )
        assert len(env.env.robots) in (1, 2)

        abs_env = EnvUtils.create_env_from_metadata(
            env_meta=abs_env_meta,
            render=False, 
            render_offscreen=False,
            use_image_obs=False, 
        )
        abs_controller = get_arm_controller(abs_env.env.robots[0])
        if hasattr(abs_controller, "use_delta"):
            assert not abs_controller.use_delta
        if hasattr(abs_controller, "input_type"):
            assert abs_controller.input_type == "absolute"

        self.env = env
        self.abs_env = abs_env
        self.file = h5py.File(dataset_path, 'r')
        self.demo_keys = sorted(
            self.file['data'].keys(),
            key=lambda name: int(name.split('_')[-1])
        )
    
    def __len__(self):
        return len(self.demo_keys)

    def _get_demo(self, idx):
        if isinstance(idx, str):
            demo_key = idx
        else:
            demo_key = self.demo_keys[idx]
        return self.file[f'data/{demo_key}']

    def convert_actions(self, 
            states: np.ndarray, 
            actions: np.ndarray) -> np.ndarray:
        """
        Given state and delta action sequence
        generate equivalent goal position and orientation for each step
        keep the original gripper action intact.
        """
        # in case of multi robot
        # reshape (N,14) to (N,2,7)
        # or (N,7) to (N,1,7)
        stacked_actions = actions.reshape(*actions.shape[:-1],-1,7)

        env = self.env
        # generate abs actions
        action_goal_pos = np.zeros(
            stacked_actions.shape[:-1]+(3,), 
            dtype=stacked_actions.dtype)
        action_goal_ori = np.zeros(
            stacked_actions.shape[:-1]+(3,), 
            dtype=stacked_actions.dtype)
        action_gripper = stacked_actions[...,[-1]]
        for i in range(len(states)):
            _ = env.reset_to({'states': states[i]})

            # taken from robot_env.py L#454
            for idx, robot in enumerate(env.env.robots):
                # run controller goal generator
                robot.control(stacked_actions[i,idx], policy_step=True)
            
                # read pos and ori from robots
                controller = get_arm_controller(robot, idx)
                action_goal_pos[i,idx] = controller.goal_pos
                action_goal_ori[i,idx] = Rotation.from_matrix(
                    controller.goal_ori).as_rotvec()

        stacked_abs_actions = np.concatenate([
            action_goal_pos,
            action_goal_ori,
            action_gripper
        ], axis=-1)
        abs_actions = stacked_abs_actions.reshape(actions.shape)
        return abs_actions

    def convert_idx(self, idx):
        demo = self._get_demo(idx)
        # input
        states = demo['states'][:]
        actions = demo['actions'][:]

        # generate abs actions
        abs_actions = self.convert_actions(states, actions)
        return abs_actions

    def convert_and_eval_idx(self, idx):
        env = self.env
        abs_env = self.abs_env
        # first step have high error for some reason, not representative
        eval_skip_steps = 1

        demo = self._get_demo(idx)
        # input
        states = demo['states'][:]
        actions = demo['actions'][:]

        # generate abs actions
        abs_actions = self.convert_actions(states, actions)

        # verify
        robot0_eef_pos = demo['obs']['robot0_eef_pos'][:]
        robot0_eef_quat = demo['obs']['robot0_eef_quat'][:]

        delta_error_info = self.evaluate_rollout_error(
            env, states, actions, robot0_eef_pos, robot0_eef_quat, 
            metric_skip_steps=eval_skip_steps)
        abs_error_info = self.evaluate_rollout_error(
            abs_env, states, abs_actions, robot0_eef_pos, robot0_eef_quat,
            metric_skip_steps=eval_skip_steps)

        info = {
            'delta_max_error': delta_error_info,
            'abs_max_error': abs_error_info
        }
        return abs_actions, info

    @staticmethod
    def evaluate_rollout_error(env, 
            states, actions, 
            robot0_eef_pos, 
            robot0_eef_quat, 
            metric_skip_steps=1):
        # first step have high error for some reason, not representative

        # evaluate abs actions
        rollout_next_states = list()
        rollout_next_eef_pos = list()
        rollout_next_eef_quat = list()
        obs = env.reset_to({'states': states[0]})
        for i in range(len(states)):
            obs = env.reset_to({'states': states[i]})
            obs, reward, done, info = env.step(actions[i])
            obs = env.get_observation()
            rollout_next_states.append(env.get_state()['states'])
            rollout_next_eef_pos.append(obs['robot0_eef_pos'])
            rollout_next_eef_quat.append(obs['robot0_eef_quat'])
        rollout_next_states = np.array(rollout_next_states)
        rollout_next_eef_pos = np.array(rollout_next_eef_pos)
        rollout_next_eef_quat = np.array(rollout_next_eef_quat)

        next_state_diff = states[1:] - rollout_next_states[:-1]
        max_next_state_diff = np.max(np.abs(next_state_diff[metric_skip_steps:]))

        next_eef_pos_diff = robot0_eef_pos[1:] - rollout_next_eef_pos[:-1]
        next_eef_pos_dist = np.linalg.norm(next_eef_pos_diff, axis=-1)
        max_next_eef_pos_dist = next_eef_pos_dist[metric_skip_steps:].max()

        next_eef_rot_diff = Rotation.from_quat(robot0_eef_quat[1:]) \
            * Rotation.from_quat(rollout_next_eef_quat[:-1]).inv()
        next_eef_rot_dist = next_eef_rot_diff.magnitude()
        max_next_eef_rot_dist = next_eef_rot_dist[metric_skip_steps:].max()

        info = {
            'state': max_next_state_diff,
            'pos': max_next_eef_pos_dist,
            'rot': max_next_eef_rot_dist
        }
        return info
