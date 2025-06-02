import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import os

# Directories
log_dir = "acrobot-scripts/logs"
models_dir = "models"
best_model_dir = f"{models_dir}/acrobot/best"
eval_log_dir = f"{log_dir}/eval"

# Create necessary directories
os.makedirs(log_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(best_model_dir, exist_ok=True)
os.makedirs(eval_log_dir, exist_ok=True)

# Create training and eval environments
env = Monitor(gym.make("Acrobot-v1"), log_dir)
eval_env = Monitor(gym.make("Acrobot-v1"), eval_log_dir)

# Evaluation callback
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=best_model_dir,
    log_path=eval_log_dir,
    eval_freq=10_000,
    deterministic=True,
    render=False
)

# Create the DQN agent
model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=log_dir,
    learning_rate=1e-3,
    buffer_size=50_000,
    learning_starts=5_000,
    batch_size=64,
    gamma=0.99,
    train_freq=1,
    target_update_interval=500,
    exploration_fraction=0.3,
    exploration_final_eps=0.05
)

# Train the agent
model.learn(total_timesteps=1_000_000, callback=eval_callback)

# Save final model
model.save(f"{models_dir}/dqn_acrobot")

print("✅ Training complete and model saved.")
