'''
The custom environment is a part of the project, 'Millipede-Control-with-Reinforcement-Learning-and-VAEs'. This script demonstrates 
the evaluation of a reinforcement learning algorithm for controlling an ant in the several direction. It utlises the Gymnasium [1] Mujoco 
Ant-v4 environment [2] and Sample Factory library [3]. With the help of wrapper class, the experiments are trained and saved. The saved
experiment can be passed in enjoy function provided by Sample Factory library which renders the experiment and show the results. 

[1] Gymnasium Github repo: https://github.com/Farama-Foundation/Gymnasium

[2] Schulman, J., Moritz, P., Levine, S., Jordan, M. and Abbeel, P., 2015. High-dimensional continuous control using generalized 
advantage estimation. arXiv preprint arXiv:1506.02438.

[3] Petrenko, A., Huang, Z., Kumar, T., Sukhatme, G. and Koltun, V., 2020, November. Sample factory: Egocentric 3d control from pixels 
at 100000 fps with asynchronous reinforcement learning. In International Conference on Machine Learning (pp. 7652-7662). PMLR. 
http://proceedings.mlr.press/v119/petrenko20a.html. Github repo: https://github.com/alex-petrenko/sample-factory/tree/master


---------------------------------
@author: Savan Agrawal
@file: evaluate_train.py
@version: 0.1
---------------------------------
'''

import sys
from sample_factory.enjoy import enjoy
import gymnasium as gym
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from typing import Optional
from sf_examples.mujoco.mujoco_params import mujoco_override_defaults


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

def get_experiment():
    '''
    This function asks the user to choose an experiment to run and returns the choice.
    '''
    experiments = ['ant_x_direction_task', 'ant_x_opp_direction_task', 'ant_y_direction_task', 'ant_y_opp_direction_task']
    print("Select an experiment to run:")
    for i, experiment in enumerate(experiments):
        print(f"{i+1}. {experiment}")
    experiment_number = int(input("Enter the number of your experiment choice: "))
    return experiments[experiment_number-1]

    
def main():  
    '''
    The main function registers the custom components, gets the user's experiment choice,
    configures the environment and experiment, and starts the simulation.
    '''
    register_custom_components()
    experiment = get_experiment()
    argv = ["--env=Ant-v4", f"--experiment={experiment}", "--train_dir=./sample-factory/code/train_dir"]
    cfg = parse_mujoco_cfg(argv=argv, evaluation=True)
    status = enjoy(cfg)
    return status

if __name__ == "__main__": 
    sys.exit(main())
