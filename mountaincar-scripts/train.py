import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
import os

log_dir = "mountaincar-scripts/logs"
os.makedirs(log_dir, exist_ok=True)

# Create environment
env = Monitor(gym.make("MountainCar-v0"), log_dir)

# Create agent
model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

# Train agent
model.learn(total_timesteps=1_000_000)

# Save model
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)
model.save(f"{models_dir}/dqn_mountaincar")

print("✅ Model trained and saved.")
