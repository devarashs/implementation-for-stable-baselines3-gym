import gymnasium as gym
import minihack
import numpy as np
import argparse
import time
from stable_baselines3 import PPO
from gymnasium import Wrapper


class WallPenaltyWrapper(Wrapper):
    def step(self, action):
        prev_pos = self.unwrapped.last_observation
        obs, reward, terminated, truncated, info = self.env.step(action)
        curr_pos = self.unwrapped.last_observation

        if prev_pos == curr_pos and action < 4:
            reward -= 0.1
        return obs, reward, terminated, truncated, info


class GoalRewardWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.goal_reached = False

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        chars = info.get("chars")
        if chars is not None:
            agent_pos = tuple(zip(*np.where(chars == b"@")))[0]
            goal_pos_list = list(zip(*np.where(chars == b">")))
            if goal_pos_list:
                goal_pos = goal_pos_list[0]
                if agent_pos == goal_pos:
                    if not self.goal_reached:
                        reward += 1.0
                        self.goal_reached = True
                    terminated = True
                    info["termination_reason"] = "🎯 Agent reached the goal tile (>)"

        if terminated and "termination_reason" not in info:
            if self.goal_reached:
                info["termination_reason"] = "🎯 Agent reached the goal (detected earlier)"
            else:
                info["termination_reason"] = "❓ Episode terminated but goal not detected."

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.goal_reached = False
        return self.env.reset(**kwargs)




def make_env():
    env = gym.make("MiniHack-Room-5x5-v0", render_mode="human")
    env = GoalRewardWrapper(env)
    env = WallPenaltyWrapper(env)
    return env


def main(model_path: str):
    print(f"🎮 Running model from: {model_path}\n")
    model = PPO.load(model_path)
    env = make_env()

    obs, _ = env.reset()
    total_reward = 0

    for step in range(300):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        time.sleep(0.1)
        if terminated or truncated:
            break

    env.close()

    print("\n⛳ Episode finished.")
    print(f"🏁 Total reward: {total_reward:.2f}")

    # Display reason if known
    if "termination_reason" in info:
        print(f"📌 Reason: {info['termination_reason']}")
    elif terminated:
        print("📌 Reason: Terminated (unknown trigger)")
    elif truncated:
        print("📌 Reason: Truncated (time limit reached)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/ppo_minihack_room",
        help="Path to the saved PPO model"
    )
    args = parser.parse_args()
    main(args.model_path)
