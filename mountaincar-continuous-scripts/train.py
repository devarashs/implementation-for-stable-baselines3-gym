import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import os

# Directories
log_dir = "mountaincar-continuous-scripts/logs"
models_dir = "models"
best_model_dir = f"{models_dir}/mountaincar-continuous/best"
eval_log_dir = f"{log_dir}/eval"

# Create necessary directories
os.makedirs(log_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(best_model_dir, exist_ok=True)
os.makedirs(eval_log_dir, exist_ok=True)

class ShapedRewardWrapper(gym.RewardWrapper):
    def reward(self, reward):
        # Enhanced reward shaping to overcome local minima
        pos, vel = self.env.unwrapped.state
        position_reward = 0.1 * (pos + 1.2)  # Encourages moving right
        velocity_reward = 15 * abs(vel)      # Increased velocity reward
        return reward + position_reward + velocity_reward

# Create training environment
env = Monitor(ShapedRewardWrapper(gym.make("MountainCarContinuous-v0")), log_dir)

# Create evaluation environment (separate from training env)
eval_env = Monitor(ShapedRewardWrapper(gym.make("MountainCarContinuous-v0")), eval_log_dir)

# Create evaluation callback
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=best_model_dir,
    log_path=eval_log_dir,
    eval_freq=5_000,          # Evaluate every 5k steps
    deterministic=True,
    render=False
)

# Create SAC agent with improved parameters
model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=log_dir,
    learning_rate=5e-4,          # Higher learning rate
    batch_size=256,
    buffer_size=100_000,         # Larger replay buffer
    learning_starts=10_000,      # Wait before learning starts
    ent_coef="auto_0.5",         # Higher initial entropy for exploration
    train_freq=1,
    gradient_steps=2,            # More gradient steps per env step
    gamma=0.98,                  # Slightly lower discount factor
    tau=0.02,                    # Slower target update for stability
    policy_kwargs=dict(
        net_arch=dict(
            pi=[256, 256],       # Larger policy network
            qf=[256, 256]        # Larger Q-function network
        )
    )
)

# Train agent with evaluation
model.learn(total_timesteps=500_000, callback=eval_callback)  # More training time

# Save final model
model.save(f"{models_dir}/sac_mountaincar_continuous")

print("✅ Model trained, evaluated, and saved. Best model saved in", best_model_dir)