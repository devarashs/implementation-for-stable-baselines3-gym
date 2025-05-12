import os
import gymnasium as gym
import minihack
import numpy as np
from gymnasium import Wrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor


class WallPenaltyWrapper(Wrapper):
    def step(self, action):
        prev_pos = self.unwrapped.last_observation
        obs, reward, terminated, truncated, info = self.env.step(action)
        curr_pos = self.unwrapped.last_observation

        if prev_pos == curr_pos and action < 4:
            reward -= 0.1
        return obs, reward, terminated, truncated, info


class GoalRewardWrapper(Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if info.get("chars", None) is not None:
            chars = info["chars"]
            if b">" in chars:
                goal_pos = tuple(zip(*np.where(chars == b">")))[0]
                agent_pos = tuple(zip(*np.where(chars == b"@")))[0]
                if goal_pos == agent_pos:
                    reward += 1.0
                    terminated = True
        return obs, reward, terminated, truncated, info


def make_env(render_mode=None):
    env = gym.make("MiniHack-Room-5x5-v0", render_mode=render_mode)
    env = GoalRewardWrapper(env)
    env = WallPenaltyWrapper(env)
    return Monitor(env)


if __name__ == "__main__":
    # Setup directories
    log_dir = "minihack-scripts/logs"
    models_dir = "models/minihack"
    checkpoint_dir = f"{models_dir}/checkpoints"
    best_model_dir = f"{models_dir}/best"
    eval_log_dir = f"{log_dir}/eval"

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)
    os.makedirs(eval_log_dir, exist_ok=True)

    # Environments
    env = make_env()
    eval_env = make_env()

    # Model and policy
    policy_kwargs = dict(net_arch=[128, 128])
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        policy_kwargs=policy_kwargs,
        n_steps=4096,
        batch_size=128,
        gae_lambda=0.95,
        gamma=0.99,
        n_epochs=5,
        learning_rate=5e-5,
        clip_range=0.1,
        ent_coef=0.01,
    )

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=eval_log_dir,
        eval_freq=10_000,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=checkpoint_dir,
        name_prefix="ppo_minihack"
    )

    # Training
    model.learn(
        total_timesteps=1_000_000,
        callback=[eval_callback, checkpoint_callback]
    )

    # Save final model
    model.save(f"{models_dir}/ppo_minihack_room")
    print("✅ Training complete and model saved.")
