# import gymnasium as gym
# import numpy as np

# from stable_baselines3 import DDPG
# from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

# env = gym.make("Ant-v4", render_mode="rgb_array")

# # The noise objects for DDPG
# n_actions = env.action_space.shape[-1]
# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
# model.learn(total_timesteps=10000, log_interval=10)
# model.save("ddpg_pendulum")
# vec_env = model.get_env()

# # del model # remove to demonstrate saving and loading

# # model = DDPG.load("ddpg_pendulum")

# obs = vec_env.reset()
# for _ in range(500):
# # while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = vec_env.step(action)
    
#     # print(rewards)
    
#     # print(dones)
#     # # print(forward_reward)
#     # print(info)
#     vec_env.render("human")


# env.close()




import gymnasium as gym
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from gymnasium import Wrapper

# Custom wrapper to modify reward
class ForwardAntEnv(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_xpos = None

    def step(self, action):
        a =self.env.step(action)
        # print(len(a))
        obs, reward, done1, done2, info = self.env.step(action)
        forward_reward = info['reward_forward']
        # reward = forward_reward  # modify reward to prioritize forward movement
        # Penalize the agent for moving in the sideways direction
        # if obs[1] > 0 :  # Assuming y position (sideways) is at index 1
        #     reward -= np.abs(obs[1])
        xpos = obs[0]
        print(obs)
        if self.prev_xpos is not None:
            reward += (xpos - self.prev_xpos)
        self.prev_xpos = xpos
        return obs, reward, done1, done2, info

# Instantiate environment and apply custom wrapper
env = gym.make("Ant-v4", render_mode="rgb_array", exclude_current_positions_from_observation=False)
n_actions = env.action_space.shape[-1]
env = ForwardAntEnv(env)
# env = (gym.make("Ant-v4", render_mode="rgb_array"))

# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
model.learn(total_timesteps=100000, log_interval=10)
model.save("ddpg_ant")
vec_env = model.get_env()

# Test the trained agent
obs = vec_env.reset()
for _ in range(500):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")

env.close()