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

    def reward(self, obs):
        # Extracting relevant observations
        torso_z_position = obs[0]
        torso_orientation = np.array([obs[1], obs[2], obs[3], obs[4]])
        torso_angular_velocity = np.array([obs[16], obs[17], obs[18]])
        front_legs_angles = np.array([obs[5], obs[7]])
        back_legs_angles = np.array([obs[9], obs[11]])
        legs_angular_velocity = np.array([obs[19], obs[20], obs[21], obs[22], obs[23], obs[24], obs[25], obs[26]])

        # Desired conditions
        back_legs_closed = np.abs(back_legs_angles).max() < 0.1
        front_legs_straight = np.abs(front_legs_angles - np.pi).max() < 0.1
        torso_balanced = np.abs(torso_angular_velocity).max() < 0.1
        torso_orientation_upright = np.abs(torso_orientation - [1, 0, 0, 0]).sum() < 0.1
        legs_stable = np.abs(legs_angular_velocity).max() < 0.1

        # Reward for desired conditions
        reward = (
            back_legs_closed * 1.0
            + front_legs_straight * 1.0
            + torso_balanced * 1.0
            + torso_z_position
            + torso_orientation_upright * 1.0
            + legs_stable * 1.0
        )

        # Penalty for not meeting desired conditions
        penalties = (
            (not back_legs_closed) * 0.5
            + (not front_legs_straight) * 0.5
            + (not torso_balanced) * 0.5
            + (not torso_orientation_upright) * 0.5
            + (not legs_stable) * 0.5
        )
        
        # Penalty for falling over
        if torso_z_position < 0.5:
            penalties += 1.0

        reward -= penalties

        return reward

    def step(self, action):
        obs, _, terminated, truncated, info = super().step(action)
        reward = self.reward(obs)
        return obs, reward, terminated, truncated, info
    
def make_gym_env_func(full_env_name, cfg=None, env_config=None, render_mode: Optional[str] = None):
    base_env = gym.make(full_env_name, render_mode=render_mode, exclude_current_positions_from_observation=True)
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
    argv=["--env=Ant-v4", "--device=cpu", "--experiment=ant_custom_task"]
    cfg = parse_mujoco_cfg(argv, True)
    status = run_rl(cfg)
    return status


if __name__ == "__main__":
    main()