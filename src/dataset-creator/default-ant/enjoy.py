# '''
# This script demonstrates the way to extract observation, action and rewards from Gymnasium [1] trained environment using Sample Factory's [2]
# 'enjoy' class. The 'enjoy' class is almost same as provided in official Git repo [3] of Sample-Factory. Only, some modification are implemented
# to extract the dataset.

# [1] Gymnasium Github repo: https://github.com/Farama-Foundation/Gymnasium

# [2] Schulman, J., Moritz, P., Levine, S., Jordan, M. and Abbeel, P., 2015. High-dimensional continuous control using generalized 
# advantage estimation. arXiv preprint arXiv:1506.02438.

# [3] Github repo: https://github.com/alex-petrenko/sample-factory/tree/master


# ---------------------------------
# @author: Savan Agrawal
# @file: dataset_creator.py
# @version: 0.1
# ---------------------------------
# '''

# import time
# from collections import deque
# from typing import Dict, Tuple

# import gymnasium as gym
# import numpy as np
# import torch
# from torch import Tensor

# from sample_factory.algo.learning.learner import Learner
# from sample_factory.algo.sampling.batched_sampling import preprocess_actions
# from sample_factory.algo.utils.action_distributions import argmax_actions
# from sample_factory.algo.utils.env_info import extract_env_info
# from sample_factory.algo.utils.make_env import make_env_func_batched
# from sample_factory.algo.utils.misc import ExperimentStatus
# from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs
# from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
# from sample_factory.cfg.arguments import load_from_checkpoint
# from sample_factory.huggingface.huggingface_utils import generate_model_card, generate_replay_video, push_to_hf
# from sample_factory.model.actor_critic import create_actor_critic
# from sample_factory.model.model_utils import get_rnn_size
# from sample_factory.utils.attr_dict import AttrDict
# from sample_factory.utils.typing import Config, StatusCode
# from sample_factory.utils.utils import debug_log_every_n, experiment_dir, log


# def render_frame(cfg, env, video_frames, num_episodes, last_render_start) -> float:
#     render_start = time.time()

#     # if not cfg.no_render:
#     target_delay = 1.0 / cfg.fps if cfg.fps > 0 else 0
#     current_delay = render_start - last_render_start
#     time_wait = target_delay - current_delay

#     if time_wait > 0:
#         time.sleep(time_wait)

#     env.render()

#     return render_start


# def enjoy(cfg: Config, steps, observations, actions_data, rewards, info) -> Tuple[StatusCode, float]:
#     cfg = load_from_checkpoint(cfg)

#     cfg.num_envs = 1

#     render_mode = "human"
#     # render_mode = "rgb_array"

#     env = make_env_func_batched(
#         cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0), render_mode=render_mode
#     )


#     env_info = extract_env_info(env, cfg)

#     if hasattr(env.unwrapped, "reset_on_init"):
#         # reset call ruins the demo recording for VizDoom
#         env.unwrapped.reset_on_init = False

#     actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
#     actor_critic.eval()
    
#     device = torch.device("cpu" if cfg.device == "cpu" else "cuda")
#     actor_critic.model_to_device(device)

#     policy_id = cfg.policy_index
#     name_prefix = dict(latest="checkpoint", best="best")[cfg.load_checkpoint_kind]
#     checkpoints = Learner.get_checkpoints(Learner.checkpoint_dir(cfg, policy_id), f"{name_prefix}_*")
#     checkpoint_dict = Learner.load_checkpoint(checkpoints, device)
#     actor_critic.load_state_dict(checkpoint_dict["model"])

#     last_render_start = time.time()

#     obs, _ = env.reset()
#     rnn_states = torch.zeros([env.num_agents, get_rnn_size(cfg)], dtype=torch.float32, device=device)

#     video_frames = []
#     num_episodes = 0

#     count = 0
#     with torch.no_grad():
#         while count <= steps:
#             normalized_obs = prepare_and_normalize_obs(actor_critic, obs)
#             # print(normalized_obs.shape)
#             # print(normalized_obs['obs'].numpy())
#             observations.append(obs['obs'].numpy())
#             policy_outputs = actor_critic(normalized_obs, rnn_states)

#             # sample actions from the distribution by default
#             actions = policy_outputs["actions"]
#             # print(actions)
#             actions = preprocess_actions(env_info, actions)
            
#             last_render_start = render_frame(cfg, env, video_frames, num_episodes, last_render_start)

#             obs, rew, terminated, truncated, infos = env.step(actions)
#             # print(actions)
#             # print(obs['obs'])
#             # print(infos)

#             # observations.append(obs['obs'].numpy())
#             actions_data.append(actions)
#             rewards.append(rew.numpy())
#             info.append(infos)

#             count += 1
#             # if count == 60: break
#             print(count)


#     env.close()

#     return observations, actions_data, rewards
    


'''
This script demonstrates the way to extract observation, action and rewards from Gymnasium [1] trained environment using Sample Factory's [2]
'enjoy' class. The 'enjoy' class is almost same as provided in official Git repo [3] of Sample-Factory. Only, some modification are implemented
to extract the dataset.

[1] Gymnasium Github repo: https://github.com/Farama-Foundation/Gymnasium

[2] Schulman, J., Moritz, P., Levine, S., Jordan, M. and Abbeel, P., 2015. High-dimensional continuous control using generalized 
advantage estimation. arXiv preprint arXiv:1506.02438.

[3] Github repo: https://github.com/alex-petrenko/sample-factory/tree/master


---------------------------------
@author: Savan Agrawal
@file: dataset_creator.py
@version: 0.1
---------------------------------
'''

import time
from collections import deque
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor

from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
from sample_factory.cfg.arguments import load_from_checkpoint
from sample_factory.huggingface.huggingface_utils import generate_model_card, generate_replay_video, push_to_hf
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config, StatusCode
from sample_factory.utils.utils import debug_log_every_n, experiment_dir, log
from PIL import Image


def render_frame(cfg, env, video_frames, num_episodes, last_render_start, timestep) -> float:
    render_start = time.time()
    
    # Get the RGB values of the current frame
    frame = env.render()

    # if the current timestep is the desired timestep, save the frame
    if num_episodes == timestep:
        video_frames.append(frame)

    target_delay = 1.0 / cfg.fps if cfg.fps > 0 else 0
    current_delay = render_start - last_render_start
    time_wait = target_delay - current_delay

    if time_wait > 0:
        time.sleep(time_wait)
    
    return render_start

# def render_frame(cfg, env, video_frames, num_episodes, last_render_start) -> float:
#     render_start = time.time()

#     # if not cfg.no_render:
#     target_delay = 1.0 / cfg.fps if cfg.fps > 0 else 0
#     current_delay = render_start - last_render_start
#     time_wait = target_delay - current_delay

#     if time_wait > 0:
#         time.sleep(time_wait)

#     env.render()

#     return render_start


def enjoy(cfg: Config, steps, observations, actions_data, rewards, info) -> Tuple[StatusCode, float]:
    cfg = load_from_checkpoint(cfg)

    cfg.num_envs = 1

    # render_mode = "human"
    render_mode = "rgb_array"

    env = make_env_func_batched(
        cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0), render_mode=render_mode
    )


    env_info = extract_env_info(env, cfg)

    if hasattr(env.unwrapped, "reset_on_init"):
        # reset call ruins the demo recording for VizDoom
        env.unwrapped.reset_on_init = False

    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
    actor_critic.eval()
    
    device = torch.device("cpu" if cfg.device == "cpu" else "cuda")
    actor_critic.model_to_device(device)

    policy_id = cfg.policy_index
    name_prefix = dict(latest="checkpoint", best="best")[cfg.load_checkpoint_kind]
    checkpoints = Learner.get_checkpoints(Learner.checkpoint_dir(cfg, policy_id), f"{name_prefix}_*")
    checkpoint_dict = Learner.load_checkpoint(checkpoints, device)
    actor_critic.load_state_dict(checkpoint_dict["model"])

    last_render_start = time.time()

    obs, _ = env.reset()
    rnn_states = torch.zeros([env.num_agents, get_rnn_size(cfg)], dtype=torch.float32, device=device)

    video_frames = []
    num_episodes = 0

    count = 0
    with torch.no_grad():
        while count <= steps:
            normalized_obs = prepare_and_normalize_obs(actor_critic, obs)
            # print(normalized_obs.shape)
            # print(normalized_obs['obs'].numpy())
            observations.append(obs['obs'].numpy())
            policy_outputs = actor_critic(normalized_obs, rnn_states)

            # sample actions from the distribution by default
            actions = policy_outputs["actions"]
            # print(actions)
            actions = preprocess_actions(env_info, actions)
            
            # last_render_start = render_frame(cfg, env, video_frames, num_episodes, last_render_start)
            last_render_start = render_frame(cfg, env, video_frames, count, last_render_start, 60)


            obs, rew, terminated, truncated, infos = env.step(actions)
            # print(actions)
            # print(obs['obs'])
            # print(infos)

            # observations.append(obs['obs'].numpy())
            actions_data.append(actions)
            rewards.append(rew.numpy())
            info.append(infos)

            count += 1
            # if count == 60: break
            print(count)
    

    if 60 in range(steps):
        frame = video_frames[0]
        im = Image.fromarray(frame)
        im.save("screenshot.png")


    env.close()

    return observations, actions_data, rewards
    