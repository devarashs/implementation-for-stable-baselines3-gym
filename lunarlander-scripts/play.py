import gymnasium as gym
from stable_baselines3 import DQN

# Create the environment with rendering enabled
env = gym.make("LunarLander-v3", render_mode="human")

# Load the trained model (adjust the path if needed)
model = DQN.load("models/lunarlander/best/best_model")

# Reset environment
obs, _ = env.reset()

# Run the agent
for _ in range(2000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, _ = env.reset()
