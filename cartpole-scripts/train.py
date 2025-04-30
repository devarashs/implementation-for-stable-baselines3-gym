import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

log_dir = "cartpole-scripts/logs"

env = Monitor(gym.make("CartPole-v1"))
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
model.learn(total_timesteps=10000)
model.save("models/ppo_cartpole")
