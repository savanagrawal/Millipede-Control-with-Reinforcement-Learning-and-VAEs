'''
The custom environment is a part of the project, 'Millipede-Control-with-Reinforcement-Learning-and-VAEs'. This script demonstrates 
the creation of dataset for VAE training. It utlises the Gymnasium [1] Mujoco Ant-v4 environment [2] and Sample Factory library [3]. 
With the help of 'enjoy' class provided by Sample Factory, some changes are implemented in 'enjoy' class which is used in this script
to take out the observation, actions and rewards dataset.

[1] Gymnasium Github repo: https://github.com/Farama-Foundation/Gymnasium

[2] Schulman, J., Moritz, P., Levine, S., Jordan, M. and Abbeel, P., 2015. High-dimensional continuous control using generalized 
advantage estimation. arXiv preprint arXiv:1506.02438.

[3] Petrenko, A., Huang, Z., Kumar, T., Sukhatme, G. and Koltun, V., 2020, November. Sample factory: Egocentric 3d control from pixels 
at 100000 fps with asynchronous reinforcement learning. In International Conference on Machine Learning (pp. 7652-7662). PMLR. 
http://proceedings.mlr.press/v119/petrenko20a.html. Github repo: https://github.com/alex-petrenko/sample-factory/tree/master


---------------------------------
@author: Savan Agrawal
@file: dataset_creator.py
@version: 0.1
---------------------------------
'''

import sys
import gymnasium as gym
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from typing import Optional
from sf_examples.mujoco.mujoco_params import mujoco_override_defaults
import numpy as np
from enjoy import enjoy

def make_gym_env_func(full_env_name, cfg=None, env_config=None, render_mode: Optional[str] = None):
    '''
    This function creates an environment for the gym.
    '''
    base_env = gym.make(full_env_name, render_mode=render_mode, exclude_current_positions_from_observation=True)
    return base_env

def register_custom_components():
    '''
    This function registers the custom environment components.
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

def create_dataset(experiments):
    '''
    This function calls 'enjoy' class to extract dataset for
    observation, actions and rewards. This function is also saving the dataset in .npy format.
    '''
    observations = []
    actions = []
    rewards = []
    infos = []
    for exp in experiments:
        argv = ["--env=Ant-v4", f"--experiment={exp}", "--train_dir=src/rl-training/train_dir"]
        cfg = parse_mujoco_cfg(argv=argv, evaluation=True)
        observations, actions, rewards = enjoy(cfg, 1000-1, observations, actions, rewards, infos)

    print(np.shape(observations))
    print(np.shape(actions))
    print(np.shape(rewards))
    print(np.shape(infos))

    # Save the data to files
    np.save('./src/dataset-creator/default-ant/assets/observations.npy', observations)
    np.save('./src/dataset-creator/default-ant/assets/actions.npy', actions)
    np.save('./src/dataset-creator/default-ant/assets/rewards.npy', rewards)
    np.save('./src/dataset-creator/default-ant/assets/infos.npy', infos)

def main():  
    '''
    The main function registers the custom components, and calls 'enjoy' class to extract dataset for
    observation, actions and rewards. This function is also saving the dataset in .npy format.
    '''
    register_custom_components()
    experiments = ['ant_x_direction_sync_task', 'ant_x_opp_direction_task', 'ant_y_direction_task', 'ant_y_opp_direction_task']
    create_dataset(experiments)
    return

if __name__ == "__main__": 
    sys.exit(main())

