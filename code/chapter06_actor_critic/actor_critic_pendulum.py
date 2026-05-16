"""
第6章：用 A2C（Advantage Actor-Critic）训练 Pendulum-v1
——展示 Actor-Critic 在连续动作空间中的高斯策略

运行方式：
    python actor_critic_pendulum.py
    python actor_critic_pendulum.py --total-timesteps 20000     # 快速验证
    python actor_critic_pendulum.py --total-timesteps 300000    # 充分训练

Pendulum-v1 的教学意义：
    1. 动作是 1 维连续力矩，范围 [-2, 2]
    2. Actor 输出连续动作分布，而不是离散动作概率
    3. Critic 估计 V(s)，用 advantage 降低策略梯度方差
"""

import argparse
import os
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


os.makedirs("output", exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def parse_args():
    parser = argparse.ArgumentParser(description="A2C 训练 Pendulum-v1")
    parser.add_argument("--total-timesteps", type=int, default=300_000,
                        help="总训练步数（默认 300000）")
    parser.add_argument("--num-envs", type=int, default=8,
                        help="并行环境数量（默认 8）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--eval-episodes", type=int, default=20,
                        help="最终评估回合数")
    return parser.parse_args()


def make_env(seed, rank):
    def _init():
        env = gym.make("Pendulum-v1")
        env = Monitor(env)
        env.reset(seed=seed + rank)
        env.action_space.seed(seed + rank)
        return env

    return _init


class TrainingMonitorCallback(BaseCallback):
    """记录回合奖励、策略熵、价值损失和策略损失。"""

    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.timesteps = []
        self.entropy_losses = []
        self.policy_losses = []
        self.value_losses = []
        self.update_numbers = set()

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])

        logger_values = getattr(self.model.logger, "name_to_value", {})
        entropy_loss = logger_values.get("train/entropy_loss")
        policy_loss = logger_values.get("train/policy_loss")
        value_loss = logger_values.get("train/value_loss")
        n_updates = logger_values.get("train/n_updates")

        if entropy_loss is not None and n_updates not in self.update_numbers:
            self.update_numbers.add(n_updates)
            self.timesteps.append(self.num_timesteps)
            self.entropy_losses.append(float(entropy_loss))
            self.policy_losses.append(float(policy_loss or 0.0))
            self.value_losses.append(float(value_loss or 0.0))

        return True


def moving_average(values, window):
    if not values:
        return np.array([])
    if len(values) < window:
        return np.array(values)
    return np.convolve(values, np.ones(window) / window, mode="valid")


def moving_average_xy(values, window):
    averaged = moving_average(values, window)
    start = 1 if len(values) < window else window
    x_values = np.arange(start, start + len(averaged))
    return x_values, averaged


def save_plots(callback, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    rewards = callback.episode_rewards
    if rewards:
        episodes = np.arange(1, len(rewards) + 1)
        smooth_x, smooth_rewards = moving_average_xy(rewards, 20)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(episodes, rewards, color="#90CAF9", alpha=0.45, linewidth=1.0, label="原始回报")
        ax.plot(smooth_x, smooth_rewards, color="#1565C0", linewidth=2.0,
                label="20回合滑动平均")
        ax.axhline(y=-800, color="green", linestyle="--", alpha=0.6,
                   label="A2C 基线参考线 (-800)")
        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.35)
        ax.set_title("A2C Pendulum-v1 回合奖励", fontsize=14, fontweight="bold")
        ax.set_xlabel("回合")
        ax.set_ylabel("累计奖励")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "actor_critic_pendulum_reward.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

    if callback.entropy_losses:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(callback.timesteps, callback.entropy_losses,
                color="#EF6C00", linewidth=1.5)
        ax.set_title("A2C Pendulum-v1 策略熵损失", fontsize=14, fontweight="bold")
        ax.set_xlabel("时间步")
        ax.set_ylabel("entropy_loss（负熵）")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "actor_critic_pendulum_entropy.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

    if callback.policy_losses and callback.value_losses:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(callback.timesteps, callback.policy_losses,
                color="#00897B", linewidth=1.5, label="Policy loss")
        ax.plot(callback.timesteps, callback.value_losses,
                color="#C62828", linewidth=1.5, label="Value loss")
        ax.set_title("A2C Pendulum-v1 Actor/Critic 损失", fontsize=14, fontweight="bold")
        ax.set_xlabel("时间步")
        ax.set_ylabel("损失")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "actor_critic_pendulum_loss.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

    print("训练曲线已保存到 output/actor_critic_pendulum_*.png")


def main():
    args = parse_args()

    print("=" * 50)
    print("第6章：A2C 训练 Pendulum-v1")
    print("=" * 50)
    print(f"总时间步:   {args.total_timesteps:,}")
    print(f"并行环境:   {args.num_envs}")
    print("动作空间:   连续 1 维力矩 [-2, 2]")

    vec_env = DummyVecEnv([make_env(args.seed, i) for i in range(args.num_envs)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    model = A2C(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=7e-4,
        n_steps=32,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=args.seed,
        verbose=1,
    )

    callback = TrainingMonitorCallback()
    model.learn(total_timesteps=args.total_timesteps, callback=callback, progress_bar=True)

    output_dir = Path("output")
    model.save(output_dir / "actor_critic_pendulum")
    vec_env.save(output_dir / "actor_critic_pendulum_vecnormalize.pkl")
    print("\n模型已保存到 output/actor_critic_pendulum.zip")
    print("归一化统计已保存到 output/actor_critic_pendulum_vecnormalize.pkl")
    save_plots(callback, output_dir)

    eval_env = DummyVecEnv([lambda: Monitor(gym.make("Pendulum-v1"))])
    eval_env = VecNormalize.load(
        output_dir / "actor_critic_pendulum_vecnormalize.pkl", eval_env
    )
    eval_env.training = False
    eval_env.norm_reward = False

    episode_rewards = []
    for episode in range(args.eval_episodes):
        eval_env.seed(args.seed + 10_000 + episode)
        obs = eval_env.reset()
        done = [False]
        total_reward = 0.0
        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = eval_env.step(action)
            total_reward += float(reward[0])
        episode_rewards.append(total_reward)

    eval_env.close()
    print("\n最终确定性策略评估：")
    print(f"  平均奖励: {np.mean(episode_rewards):.1f}")
    print(f"  标准差:   {np.std(episode_rewards):.1f}")
    print(f"  最好一轮: {np.max(episode_rewards):.1f}")
    print(f"  最差一轮: {np.min(episode_rewards):.1f}")


if __name__ == "__main__":
    main()
