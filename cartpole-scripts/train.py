import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
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

# Create training and eval environments
env = Monitor(gym.make("CartPole-v1"), log_dir)
eval_env = Monitor(gym.make("CartPole-v1"), eval_log_dir)

# Evaluation callback
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=best_model_dir,
    log_path=eval_log_dir,
    eval_freq=10_000,
    deterministic=True,
    render=False
)

# Create training environment
env = Monitor(gym.make("CartPole-v1"))


# Create PPO agent
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

# Train agent with evaluation
model.learn(total_timesteps=10000, callback=eval_callback)

# Save final model
model.save("models/ppo_cartpole")
