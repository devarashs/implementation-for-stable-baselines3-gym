import gymnasium as gym
from stable_baselines3 import DQN

# Create environment with human rendering
env = gym.make("Acrobot-v1", render_mode="human")

# Load the best trained model
model = DQN.load("../models/dqn_acrobot")

obs, _ = env.reset()

for _ in range(2000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, _ = env.reset()
