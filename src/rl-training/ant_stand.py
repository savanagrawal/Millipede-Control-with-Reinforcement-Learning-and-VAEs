'''
The custom environment is a part of the project, 'Millipede-Control-with-Reinforcement-Learning-and-VAEs'. This script demonstrates 
the implementation of a reinforcement learning algorithm for controlling an ant to stand and wave. It utlises the Gymnasium [1] Mujoco 
Ant-v4 environment [2] and Sample Factory library [3]. 

[1] Gymnasium Github repo: https://github.com/Farama-Foundation/Gymnasium

[2] Schulman, J., Moritz, P., Levine, S., Jordan, M. and Abbeel, P., 2015. High-dimensional continuous control using generalized 
advantage estimation. arXiv preprint arXiv:1506.02438.

[3] Petrenko, A., Huang, Z., Kumar, T., Sukhatme, G. and Koltun, V., 2020, November. Sample factory: Egocentric 3d control from pixels 
at 100000 fps with asynchronous reinforcement learning. In International Conference on Machine Learning (pp. 7652-7662). PMLR. 
http://proceedings.mlr.press/v119/petrenko20a.html. Github repo: https://github.com/alex-petrenko/sample-factory/tree/master


---------------------------------
@author: Savan Agrawal
@file: ant_stand.py
@version: 0.1
---------------------------------
'''

import sys
from typing import Optional
import gymnasium as gym
import numpy as np
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl
from sf_examples.mujoco.mujoco_params import mujoco_override_defaults


class CustomAntEnv(gym.Wrapper):
    """ This class is a wrapper for the Mujoco Ant-v4 environment. """
    def __init__(self, env):
        """ Initialize the wrapper by calling the parent gym.Wrapper class """
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
        '''
        The 'step' function is called at each timestep of the environment. It receives the action chosen by the agent
        and returns the new observation, reward and whether the episode has terminated.

        It is trying to trying to give reward on the various activities in observation space.
        
        Arguments:
        - action : Chosen action

        Returns:
        - observation : State of the environment after action
        - reward : Calculated reward after action
        - terminated : Whether the episode has ended
        - truncated : Whether the episode is truncated
        - info : Additional information, such as velocity in x and y direction
        '''

        obs, _, terminated, truncated, info = super().step(action)
        reward = self.reward(obs)
        return obs, reward, terminated, truncated, info
    
def make_gym_env_func(full_env_name, cfg=None, env_config=None, render_mode: Optional[str] = None):
    '''
    A function to create a new instance of the custom ant environment.
    '''
    base_env = gym.make(full_env_name, render_mode=render_mode, exclude_current_positions_from_observation=True)
    return CustomAntEnv(base_env)


def register_custom_components():
    '''
    Registers the custom ant environment.
    '''
    register_env("Ant-v4", make_gym_env_func)


def parse_mujoco_cfg(argv=None, evaluation=False):
    '''
    Parse the Mujoco configuration.
    '''
    parser, partial_cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    mujoco_override_defaults(partial_cfg.env, parser)
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg

# Main function to run the training
def main():
    '''
    Script entry points.
    '''
    # Register the custom components
    register_custom_components()
    # Define the command line arguments
    argv=["--env=Ant-v4", "--device=cpu", "--experiment=ant_stand", "--train_dir=./src/rl-training/train_dir"]
    # Parse the configuration
    cfg = parse_mujoco_cfg(argv)
    # Run the training and return the status
    status = run_rl(cfg)
    return status


# Run the main function if the script is run directly
if __name__ == "__main__":
    sys.exit(main())