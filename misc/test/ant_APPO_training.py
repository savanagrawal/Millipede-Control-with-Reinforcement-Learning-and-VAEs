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

    '''
    def step(self, action):
        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        forward_reward = x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward

        costs = ctrl_cost = self.control_cost(action)

        terminated = self.terminated
        observation = self._get_obs()
        info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }
        if self._use_contact_forces:
            contact_cost = self.contact_cost
            costs += contact_cost
            info["reward_ctrl"] = -contact_cost

        reward = rewards - costs

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info
        
    '''

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        return observation, reward, terminated, truncated, info
    
def make_gym_env_func(full_env_name, cfg=None, env_config=None, render_mode: Optional[str] = None):
    base_env = gym.make(full_env_name, render_mode=render_mode)
    return CustomAntEnv(base_env)


def register_custom_components():
    register_env("Ant-v4", make_gym_env_func)


def parse_mujoco_cfg(argv=None, evaluation=False):
    parser, partial_cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    # parser.set_defaults(
    #     batched_sampling=False,
    #     num_workers=8,
    #     num_envs_per_worker=8,
    #     worker_num_splits=2,
    #     train_for_env_steps=10000000,
    #     encoder_mlp_layers=[64, 64],
    #     env_frameskip=1,
    #     nonlinearity="tanh",
    #     batch_size=1024,
    #     kl_loss_coeff=0.1,
    #     use_rnn=False,
    #     adaptive_stddev=False,
    #     policy_initialization="torch_default",
    #     reward_scale=1,
    #     rollout=64,
    #     max_grad_norm=3.5,
    #     num_epochs=2,
    #     num_batches_per_epoch=4,
    #     ppo_clip_ratio=0.2,
    #     value_loss_coeff=1.3,
    #     exploration_loss_coeff=0.0,
    #     learning_rate=0.00295,
    #     lr_schedule="linear_decay",
    #     shuffle_minibatches=False,
    #     gamma=0.99,
    #     gae_lambda=0.95,
    #     with_vtrace=False,
    #     recurrence=1,
    #     normalize_input=True,
    #     normalize_returns=True,
    #     value_bootstrap=True,
    #     experiment_summaries_interval=3,
    #     save_every_sec=15,
    #     serial_mode=False,
    #     async_rl=False,
    # )
    mujoco_override_defaults(partial_cfg.env, parser)
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg


def main():
    """Script entry point."""
    register_custom_components()
    argv=["--env=Ant-v4", "--device=cpu", "--experiment=ant_APPO_tuned_params" "--algo=PPO"]
    cfg = parse_mujoco_cfg(argv, True)
    status = run_rl(cfg)
    return status


if __name__ == "__main__":
    main()