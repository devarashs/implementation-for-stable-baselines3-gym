import gymnasium as gym
from stable_baselines3 import DQN

# Load environment with render mode
env = gym.make("MountainCar-v0", render_mode="human")

# Load trained DQN model
model = DQN.load("models/dqn_mountaincar")

obs, _ = env.reset()

for _ in range(2000):
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, _ = env.reset()
