import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import os

# Directories
log_dir = "mountaincar-scripts/logs"
models_dir = "models"
best_model_dir = f"{models_dir}/mountaincar/best"
eval_log_dir = f"{log_dir}/eval"

# Create necessary directories
os.makedirs(log_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(best_model_dir, exist_ok=True)
os.makedirs(eval_log_dir, exist_ok=True)

# Create training environment
env = Monitor(gym.make("MountainCar-v0"), log_dir)

# Create evaluation environment (separate from training env)
eval_env = Monitor(gym.make("MountainCar-v0"), eval_log_dir)

# Create evaluation callback
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=best_model_dir,
    log_path=eval_log_dir,
    eval_freq=10_000,          # Evaluate every 10k steps
    deterministic=True,
    render=False
)

# Create DQN agent
model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=log_dir,
    learning_rate=1e-3,
    buffer_size=50_000,
    learning_starts=10_000,  # don't start learning too early
    batch_size=64,
    tau=1.0,
    gamma=0.99,
    train_freq=1,
    target_update_interval=500,
    exploration_fraction=0.3,   # explore longer
    exploration_final_eps=0.05, # don't go to 0, stay a bit random
    max_grad_norm=10
)

# Train agent with evaluation
model.learn(total_timesteps=1_000_000, callback=eval_callback)

# Save final model
model.save(f"{models_dir}/dqn_mountaincar")

print("✅ Model trained, evaluated, and saved.")
