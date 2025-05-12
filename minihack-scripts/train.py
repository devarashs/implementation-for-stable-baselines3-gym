import os
import gymnasium as gym
from gymnasium import Wrapper
import minihack
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import NatureCNN
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import CheckpointCallback


class WallPenaltyWrapper(Wrapper):
    def step(self, action):
        prev_pos = self.unwrapped.last_observation
        obs, reward, terminated, truncated, info = self.env.step(action)
        curr_pos = self.unwrapped.last_observation

        # If position didn't change after movement action, assume wall collision
        if prev_pos == curr_pos and action < 4:  # 0-3 are movement actions
            reward -= 0.1  # Small penalty for hitting walls

        return obs, reward, terminated, truncated, info


# Paths
log_dir = "minihack-scripts/logs"
models_dir = "models"
best_model_dir = f"{models_dir}/minihack/best"
eval_log_dir = f"{log_dir}/eval"

# Create directories
os.makedirs(log_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(best_model_dir, exist_ok=True)
os.makedirs(eval_log_dir, exist_ok=True)

# Use wrapper
env = Monitor(WallPenaltyWrapper(gym.make("MiniHack-Room-5x5-v0")), log_dir)
eval_env = Monitor(WallPenaltyWrapper(gym.make("MiniHack-Room-5x5-v0")), eval_log_dir)

policy_kwargs = dict(
    net_arch=[128, 128]  # Keep just the network architecture
)

# Evaluation callback
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=best_model_dir,
    log_path=eval_log_dir,
    eval_freq=10_000,
    deterministic=True,
    render=False
)

# PPO agent
model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    tensorboard_log=log_dir,
    policy_kwargs=policy_kwargs,
    n_steps=4096,         # Increase from 2048
    batch_size=128,       # Increase from 64
    gae_lambda=0.95,
    gamma=0.99,
    n_epochs=5,           # Reduce from 10 to avoid overfitting
    learning_rate=5e-5,   # Lower learning rate for more stable updates
    clip_range=0.1,       # Smaller clip range
    ent_coef=0.01
)

checkpoint_callback = CheckpointCallback(
    save_freq=100_000,
    save_path=f"{models_dir}/minihack/checkpoints/",
    name_prefix="ppo_minihack"
)

# Train
# Use both callbacks
model.learn(total_timesteps=1_000_000,
           callback=[eval_callback, checkpoint_callback])

model.save(f"{models_dir}/ppo_minihack_room")

print("✅ PPO trained on MiniHack-Room-5x5-v0.")
