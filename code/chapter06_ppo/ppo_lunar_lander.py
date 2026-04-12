"""
第6章：用 Stable-Baselines3 的 PPO 训练 LunarLander-v3
——理解 PPO 的核心超参数与训练监控

运行方式：
    python ppo_lunar_lander.py

PPO（近端策略优化）的核心思想：
    1. 限制每次策略更新的幅度（clip），避免"步子迈太大"
    2. 多轮复用同一批数据（epoch），提高样本效率
    3. 同时优化策略网络和价值网络（Actor-Critic 架构）
"""

import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# 创建输出目录
os.makedirs("output", exist_ok=True)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 第一部分：自定义训练回调 —— 记录关键指标
# ==========================================
class TrainingMonitorCallback(BaseCallback):
    """
    自定义回调：在每次 rollout 结束后记录 PPO 的关键训练指标
    包括：回合奖励、策略熵、裁剪比例、近似 KL 散度
    """

    def __init__(self, check_freq=2048, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        # 记录训练过程中的指标
        self.episode_rewards = []
        self.entropy_list = []
        self.clip_fraction_list = []
        self.approx_kl_list = []
        self.timesteps_list = []

    def _on_step(self):
        # 从信息字典中提取回合奖励（当回合结束时）
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])

        # 每次 rollout 结束后记录策略指标
        if self.num_timesteps % self.check_freq == 0 and self.num_timesteps > 0:
            # 获取 PPO 内部记录的统计信息
            # entropy: 策略的熵，衡量探索程度
            # clip_fraction: 被裁剪的比例，衡量策略更新幅度
            # approx_kl: 近似 KL 散度，衡量新旧策略的差异
            logger = self.model.logger
            if hasattr(logger, "name_to_value"):
                name_to_value = logger.name_to_value

                entropy = name_to_value.get("train/entropy_loss", 0)
                clip_frac = name_to_value.get("train/clip_fraction", 0)
                approx_kl = name_to_value.get("train/approx_kl", 0)

                self.entropy_list.append(entropy)
                self.clip_fraction_list.append(clip_frac)
                self.approx_kl_list.append(approx_kl)
                self.timesteps_list.append(self.num_timesteps)

        return True


# ==========================================
# 第二部分：创建向量化环境
# ==========================================
print("=" * 50)
print("第6章：PPO 训练 LunarLander-v3")
print("=" * 50)

print("\n正在创建向量化环境（4 个并行环境）...")

# 使用 DummyVecEnv 创建 4 个并行环境
# 向量化环境可以同时采集多个环境的数据，提高采样效率
def make_env():
    """环境工厂函数，用于创建多个独立的环境实例"""
    def _init():
        env = gym.make("LunarLander-v3")
        return env
    return _init

num_envs = 4
vec_env = DummyVecEnv([make_env() for _ in range(num_envs)])
print(f"已创建 {num_envs} 个并行环境")


# ==========================================
# 第三部分：配置 PPO 超参数
# ==========================================
print("\n配置 PPO 超参数...")

model = PPO(
    policy="MlpPolicy",       # 使用多层感知机策略
    env=vec_env,              # 向量化环境
    learning_rate=3e-4,       # 学习率：Adam 优化器的步长
    n_steps=2048,             # 每次 rollout 采集的步数（每个环境）
    batch_size=64,            # 小批量大小：每次更新的样本数
    n_epochs=10,              # 每批数据的更新轮数
    clip_range=0.2,           # PPO 裁剪范围：限制策略比率在 [0.8, 1.2] 内
    ent_coef=0.01,            # 熵系数：鼓励探索的正则化项
    vf_coef=0.5,              # 价值函数损失系数
    gamma=0.99,               # 折扣因子
    gae_lambda=0.95,          # GAE lambda：偏差-方差权衡参数
    verbose=1,
    seed=42,
    device="auto",
)

print(f"  学习率:       {model.learning_rate}")
print(f"  Rollout 步数: {model.n_steps}")
print(f"  批量大小:     {model.batch_size}")
print(f"  更新轮数:     {model.n_epochs}")
print(f"  裁剪范围:     [{1 - model.clip_range:.1f}, {1 + model.clip_range:.1f}]")
print(f"  熵系数:       {model.ent_coef}")
print(f"  价值系数:     {model.vf_coef}")


# ==========================================
# 第四部分：训练模型
# ==========================================
print("\n开始训练（200000 时间步）...")
print("-" * 50)

# 创建训练监控回调
callback = TrainingMonitorCallback(check_freq=2048)

# 训练 200,000 个时间步
total_timesteps = 200_000
model.learn(
    total_timesteps=total_timesteps,
    callback=callback,
    progress_bar=True,
)

print("-" * 50)
print("训练完成！")


# ==========================================
# 第五部分：绘制训练曲线
# ==========================================
print("\n正在绘制训练曲线...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("PPO 训练 LunarLander-v3 — 训练指标监控", fontsize=16, fontweight="bold")

# 子图1：回合奖励曲线
ax1 = axes[0, 0]
if callback.episode_rewards:
    # 使用滑动平均平滑曲线
    rewards = callback.episode_rewards
    window = min(20, len(rewards))
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
    ax1.plot(smoothed, color="#2196F3", alpha=0.8, linewidth=1.5)
    ax1.set_title("回合奖励（滑动平均）", fontsize=13)
    ax1.set_xlabel("回合")
    ax1.set_ylabel("累计奖励")
    ax1.grid(True, alpha=0.3)

# 子图2：策略熵
ax2 = axes[0, 1]
if callback.entropy_list:
    ax2.plot(callback.timesteps_list, callback.entropy_list,
             color="#FF9800", alpha=0.8, linewidth=1.5)
    ax2.set_title("策略熵（探索程度）", fontsize=13)
    ax2.set_xlabel("时间步")
    ax2.set_ylabel("熵")
    ax2.grid(True, alpha=0.3)
    # 标注：熵越高 = 探索越多

# 子图3：裁剪比例
ax3 = axes[1, 0]
if callback.clip_fraction_list:
    ax3.plot(callback.timesteps_list, callback.clip_fraction_list,
             color="#F44336", alpha=0.8, linewidth=1.5)
    ax3.axhline(y=0.2, color="gray", linestyle="--", alpha=0.5, label="clip_range=0.2")
    ax3.set_title("裁剪比例（clip fraction）", fontsize=13)
    ax3.set_xlabel("时间步")
    ax3.set_ylabel("被裁剪的比例")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

# 子图4：近似 KL 散度
ax4 = axes[1, 1]
if callback.approx_kl_list:
    ax4.plot(callback.timesteps_list, callback.approx_kl_list,
             color="#4CAF50", alpha=0.8, linewidth=1.5)
    ax4.set_title("近似 KL 散度（新旧策略差异）", fontsize=13)
    ax4.set_xlabel("时间步")
    ax4.set_ylabel("KL 散度")
    ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("output/ppo_lunar_lander_curves.png", dpi=150, bbox_inches="tight")
print("训练曲线已保存至: output/ppo_lunar_lander_curves.png")
plt.show()


# ==========================================
# 第六部分：评估训练好的模型
# ==========================================
print("\n正在评估最终模型（20 个测试回合）...")
print("-" * 50)

# 创建评估用的独立环境
eval_env = gym.make("LunarLander-v3")
mean_reward, std_reward = evaluate_policy(
    model, eval_env, n_eval_episodes=20, deterministic=True
)
print(f"20 回合测试结果：")
print(f"  平均奖励: {mean_reward:.2f}")
print(f"  标准差:   {std_reward:.2f}")

# 逐回合测试，展示详细结果
test_rewards = []
for ep in range(20):
    obs, _ = eval_env.reset()
    done, truncated = False, False
    total_reward = 0.0
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = eval_env.step(action)
        total_reward += reward
    test_rewards.append(total_reward)

print(f"\n逐回合奖励：")
for i, r in enumerate(test_rewards):
    status = "达标" if r >= 200 else "未达标"
    print(f"  回合 {i + 1:2d}: {r:8.2f}  [{status}]")

print(f"\n达标率（>= 200 分）: {sum(1 for r in test_rewards if r >= 200)}/20")
eval_env.close()


# ==========================================
# 第七部分：保存模型
# ==========================================
model.save("output/ppo_lunar_lander")
print(f"\n模型已保存至: output/ppo_lunar_lander.zip")
print("=" * 50)
