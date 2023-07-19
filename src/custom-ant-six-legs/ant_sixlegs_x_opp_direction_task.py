'''
The custom environment is a part of the project, 'Millipede-Control-with-Reinforcement-Learning-and-VAEs'. This script demonstrates 
the implementation of a reinforcement learning algorithm for controlling a custom six legs ant in the x direction using custom environment
package created [4]. It utlises the Gymnasium [1] Mujoco Ant-v4 environment [2] and Sample Factory library [3]. However, the current 
script utilises custom environment where the ant is made with six legs [4]. With the help of wrapper class, the script rewards the agent for moving in 
the opposite x direction. The step function is overridden to modify the rewards and penalties based on the agent's actions. 
Specifically, the agent is encouraged to move in the opposite x direction (negative x direction). This is done by calculating the 
change in x position between the current and previous steps and subtracting it from the original reward. As a result, if the agent 
moves in the positive x direction, the reward decreases, effectively encouraging the agent to move in the opposite direction.

[1] Gymnasium Github repo: https://github.com/Farama-Foundation/Gymnasium

[2] Schulman, J., Moritz, P., Levine, S., Jordan, M. and Abbeel, P., 2015. High-dimensional continuous control using generalized 
advantage estimation. arXiv preprint arXiv:1506.02438.

[3] Petrenko, A., Huang, Z., Kumar, T., Sukhatme, G. and Koltun, V., 2020, November. Sample factory: Egocentric 3d control from pixels 
at 100000 fps with asynchronous reinforcement learning. In International Conference on Machine Learning (pp. 7652-7662). PMLR. 
http://proceedings.mlr.press/v119/petrenko20a.html. Github repo: https://github.com/alex-petrenko/sample-factory/tree/master

[4] https://github.com/savanagrawal/Gymnasium-MuJoCo-Ant-Six-Legs.git

---------------------------------
@author: Savan Agrawal
@file: ant_sixlegs_x_opp_direction_task.py
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
import gym_mujoco_ant_six_legs

class CustomAntEnv(gym.Wrapper):
    """ This class is a wrapper for the Mujoco Ant-v4 environment. """
    def __init__(self, env):
        """ Initialize the wrapper by calling the parent gym.Wrapper class """
        super(CustomAntEnv, self).__init__(env)

    def step(self, action):
        '''
        The 'step' function is called at each timestep of the environment. It receives the action chosen by the agent
        and returns the new observation, reward and whether the episode has terminated.

        The reward function has been modified to incentivize opposite movement in x direction and penalize movement in x direction.

        Arguments:
        - action : Chosen action

        Returns:
        - observation : State of the environment after action
        - reward : Calculated reward after action
        - terminated : Whether the episode has ended
        - truncated : Whether the episode is truncated
        - info : Additional information, such as velocity in x and y direction
        '''

        observation, reward, terminated, truncated, info = self.env.step(action)
        
        forward_reward = -info["x_velocity"]  
        reward = reward - info["x_velocity"] + forward_reward  
        info["reward_forward"] = info["forward_reward"] = forward_reward

        return observation, reward, terminated, truncated, info

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
    register_env("gym_mujoco_ant_six_legs/AntSixLegs-v0", make_gym_env_func)


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
    argv=["--env=gym_mujoco_ant_six_legs/AntSixLegs-v0", "--device=cpu", "--experiment=ant_sixlegs_x_opp_direction_task", 
          "--train_dir=./src/custom-ant-six-legs/train_dir"]
    # Parse the configuration
    cfg = parse_mujoco_cfg(argv)
    # Run the training and return the status
    status = run_rl(cfg)
    return status


# Run the main function if the script is run directly
if __name__ == "__main__":
    sys.exit(main())