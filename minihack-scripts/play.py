import gymnasium as gym
import minihack
import os
import time
from stable_baselines3 import PPO

# Load trained PPO model
model_path = "models/minihack/checkpoints/ppo_minihack_300000_steps"
model = PPO.load(model_path)

env_id = "MiniHack-Room-5x5-v0"
env = gym.make(env_id, render_mode="human")  # Use graphical window

obs, _ = env.reset()
for step in range(200):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()  # Update the window
    time.sleep(0.1)  # Slow down for visibility
    if terminated or truncated:
        break

env.close()
print("✅ Episode finished. Close the window to exit.")