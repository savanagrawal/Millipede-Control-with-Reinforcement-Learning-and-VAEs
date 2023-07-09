# import gymnasium as gym

# from stable_baselines3 import SAC

# env = gym.make("Ant-v4", render_mode="human")

# # model = SAC("MlpPolicy", env, verbose=1)
# # model.learn(total_timesteps=100000, log_interval=4)
# # model.save("sac_pendulum")

# # del model # remove to demonstrate saving and loading

# model = SAC.load("sac_pendulum")

# obs, info = env.reset()
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, info = env.step(action)
#     if terminated or truncated:
#         obs, info = env.reset()


import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy


# Create environment
env = gym.make("Ant-v4", render_mode="rgb_array")

# Instantiate the agent
model = SAC("MlpPolicy", env, verbose=1)
# Train the agent and display a progress bar
model.learn(total_timesteps=int(2e5), progress_bar=True)
# Save the agent
model.save("sac_ant")
del model  # delete trained model to demonstrate loading

# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
# model = DQN.load("dqn_lunar", env=env, print_system_info=True)
model = SAC.load("sac_ant", env=env)

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")