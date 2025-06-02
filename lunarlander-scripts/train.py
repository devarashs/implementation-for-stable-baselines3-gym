import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import os

# Directories
log_dir = "lunarlander-scripts/logs"
models_dir = "models"
best_model_dir = f"{models_dir}/lunarlander/best"
eval_log_dir = f"{log_dir}/eval"

# Create directories if needed
os.makedirs(log_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(best_model_dir, exist_ok=True)
os.makedirs(eval_log_dir, exist_ok=True)

# Create environments
env = Monitor(gym.make("LunarLander-v3"), log_dir)
eval_env = Monitor(gym.make("LunarLander-v3"), eval_log_dir)

# Evaluation callback
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=best_model_dir,
    log_path=eval_log_dir,
    eval_freq=10_000,
    deterministic=True,
    render=False
)

# DQN agent
model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=log_dir,
    learning_rate=1e-4,
    buffer_size=100_000,
    learning_starts=10_000,
    batch_size=64,
    gamma=0.99,
    train_freq=1,
    target_update_interval=500,
    exploration_fraction=0.3,
    exploration_final_eps=0.05
)

# Train
model.learn(total_timesteps=1_000_000, callback=eval_callback)
model.save(f"{models_dir}/dqn_lunarlander")

print("✅ LunarLander training complete.")
