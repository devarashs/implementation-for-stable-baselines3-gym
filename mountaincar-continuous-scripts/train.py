import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
import os

log_dir = "mountaincar-scripts/logs"
os.makedirs(log_dir, exist_ok=True)

class ShapedRewardWrapper(gym.RewardWrapper):
    def reward(self, reward):
        return reward + self.env.state[1] * 10  # boost velocity

# Create environment
env = Monitor(ShapedRewardWrapper(gym.make("MountainCarContinuous-v0")), log_dir)

# Create agent
model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=log_dir,
    learning_rate=3e-4,
    batch_size=256,
    ent_coef="auto_0.2",  # slightly more entropy early on
    train_freq=1,
    gradient_steps=1,
    gamma=0.99,
)
# Train agent
model.learn(total_timesteps=200_000)

# Save model
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)
model.save(f"{models_dir}/sac_mountaincar")

print("✅ Model trained and saved.")
