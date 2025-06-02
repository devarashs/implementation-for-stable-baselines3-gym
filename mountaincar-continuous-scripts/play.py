import gymnasium as gym
from stable_baselines3 import SAC

# Load environment with render mode
env = gym.make("MountainCarContinuous-v0", render_mode="human")

# Load trained DQN model
model = SAC.load("models/sac_mountaincar")

obs, _ = env.reset()

for _ in range(2000):
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, _ = env.reset()
