import sys
from typing import Optional

import gymnasium as gym
import numpy as np
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl
from sf_examples.mujoco.mujoco_params import mujoco_override_defaults

class CustomAntEnv(gym.Wrapper):
    def __init__(self, env):
        super(CustomAntEnv, self).__init__(env)
        self.center = np.array([0, 0])

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        x_pos = observation[0]  # assuming 0 index is x-coordinate after setting exclude_current_positions_from_observation=False
        y_pos = observation[1]  # assuming 1 index is y-coordinate after setting exclude_current_positions_from_observation=False
        current_pos = np.array([x_pos, y_pos])

        distance_from_center = np.linalg.norm(self.center - current_pos)

        # Modify reward based on distance to the center
        # Here, smaller the distance, higher the reward. You may need to adjust this based on your requirement.
        reward -= distance_from_center

        return observation, reward, terminated, truncated, info
    
def make_gym_env_func(full_env_name, cfg=None, env_config=None, render_mode: Optional[str] = None):
    base_env = gym.make(full_env_name, render_mode=render_mode, exclude_current_positions_from_observation=False)
    return CustomAntEnv(base_env)


def register_custom_components():
    register_env("Ant-v4", make_gym_env_func)


def parse_mujoco_cfg(argv=None, evaluation=False):
    parser, partial_cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    mujoco_override_defaults(partial_cfg.env, parser)
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg


def main():
    """Script entry point."""
    register_custom_components()
    argv=["--env=Ant-v4", "--device=cpu", "--experiment=ant_circular_task"]
    cfg = parse_mujoco_cfg(argv, True)
    status = run_rl(cfg)
    return status


if __name__ == "__main__":
    main()