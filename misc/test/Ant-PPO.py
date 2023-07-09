# import gymnasium as gym

# from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env

# class ForwardAntEnv(gym.Wrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         self._max_reward = None

#     def reset(self, **kwargs):
#         self._max_reward = None
#         return self.env.reset(**kwargs)

#     def step(self, action):
#         obs, reward, done, info = self.env.step(action)
#         forward_reward = info['reward_forward']
#         if self._max_reward is None or forward_reward > self._max_reward:
#             self._max_reward = forward_reward
#             reward = forward_reward
#         return obs, reward, done, info


# # Parallel environments
# # vec_env = make_vec_env("Ant-v4", n_envs=4)
# vec_env = gym.make("Ant-v4", render_mode='rgb_array')

# model = PPO("MlpPolicy", vec_env, verbose=1)
# model.learn(total_timesteps=2500)
# model.save("ppo_cartpole")

# del model # remove to demonstrate saving and loading

# model = PPO.load("ppo_cartpole")

# obs = vec_env.reset()[0]
# # print(obs[0])
# for i in range(1000):
#     # print(model.predict(obs))
#     action, _states = model.predict(obs)
#     print(vec_env.step(action))
#     obs, rewards, dones1, dones2, info = vec_env.step(action)
#     vec_env.render("human")







# import gymnasium as gym
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv

# # Initialize environment
# env = gym.make('Ant-v4', render_mode='rgb_array')
# env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

# # Initialize agent
# model = PPO("MlpPolicy", env, verbose=1)

# # Train agent
# model.learn(total_timesteps=2e5, progress_bar=True)

# # Save the agent
# model.save("ppo_ant")

# # Load the trained agent
# model = PPO.load("ppo_ant")

# # Test the trained agent
# obs = env.reset()
# for _ in range(1000):
#     action, _ = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render("human")



import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

class CustomAntEnv(gym.Wrapper):
    def __init__(self, env):
        super(CustomAntEnv, self).__init__(env)

    def step(self, action):
        observation, reward, done1, done2, info = self.env.step(action)

        # ## Check if the ant has fallen
        # z_coordinate = observation[0]
        # if z_coordinate < 0.4:  # This is approximately the height of the ant's body when it's fallen
        #     reward -= 100

        # # Check if the ant is moving in the right direction (positive x direction)
        # x_velocity = observation[13]
        # if x_velocity > 0:
        #     reward += 10  # reward for moving in the right direction
        # elif x_velocity < 0:
        #     reward -= 10  # penalty for moving in the wrong direction

        # # Reward the ant for maintaining a steady speed
        # speed = np.linalg.norm(observation[13:16])
        # if 0.5 < speed < 1.0:  
        #     reward += 10

        # # Penalize excessive rotation
        # rotational_velocity = observation[16:19]
        # excessive_rotation = np.abs(rotational_velocity).max()
        # if excessive_rotation > 0.5:
        #     reward -= excessive_rotation * 10

        # # Reward the ant for keeping its body straight
        # torso_quat = observation[1:5]
        # # The deviation from being upright can be calculated as the absolute difference between the w-component of the quaternion and 1.0
        # torso_deviation = np.abs(torso_quat[0] - 1.0)
        # reward -= torso_deviation * 10

        return observation, reward, done1, done2, info

# Initialize environment
base_env = gym.make('Ant-v4', render_mode="rgb_array")
env = CustomAntEnv(base_env)
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

# Initialize agent
model = PPO("MlpPolicy", env, verbose=1)

# Train agent
model.learn(total_timesteps=2e5, progress_bar=True)

# Save the agent
model.save("ppo_ant")

# Load the trained agent
model = PPO.load("ppo_ant")

# Test the trained agent
obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render("human")
