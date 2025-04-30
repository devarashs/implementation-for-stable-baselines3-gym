import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import os

# Directories
log_dir = "cartpole-scripts/logs"
models_dir = "models"
best_model_dir = f"{models_dir}/cartpole/best"
eval_log_dir = f"{log_dir}/eval"

# Create necessary directories
os.makedirs(log_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(best_model_dir, exist_ok=True)
os.makedirs(eval_log_dir, exist_ok=True)

# Create training environment
env = Monitor(gym.make("CartPole-v1"))

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
model.learn(total_timesteps=10000)
model.save("models/ppo_cartpole")
