"""
第9章：用 SAC（Soft Actor-Critic）训练 HalfCheetah-v4
——理解最大熵强化学习的核心创新

运行方式：
    python sac_halfcheetah.py

SAC 的核心思想：
    1. 熵正则化（Entropy Regularization）：在最大化期望回报的同时，最大化策略的熵
       → 鼓励策略保持随机性，提高探索能力和鲁棒性
    2. 自动温度调节（Automatic Temperature Tuning）：alpha 参数自动调整
       → 不需要手动调节探索-利用的平衡
    3. 双 Q 网络（Twin Critics）：取两个 Q 值中的较小值，缓解过估计
       → 与 TD3 的思想类似，但结合了最大熵框架

SAC 的目标函数：
    J(π) = Σ_t E_{(s,a)~ρ_π}[r(s,a) + α * H(π(·|s))]
    其中 H 是策略的熵，α 是温度参数

与 PPO、TD3 的对比：
    - PPO：同策略（on-policy），简单但样本效率低
    - TD3：异策略（off-policy），确定性策略，双 Q 网络
    - SAC：异策略（off-policy），随机策略，熵正则化，样本效率最高
"""

import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

# 创建输出目录
os.makedirs("output", exist_ok=True)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 第一部分：自定义训练回调 —— 记录 SAC 关键指标
# ==========================================
class SACTrainingCallback(BaseCallback):
    """
    自定义回调：在训练过程中记录 SAC 的关键指标

    SAC 的核心监控指标：
        - episode_reward：回合累计奖励，衡量策略性能
        - entropy/alpha：熵系数（温度参数），衡量探索强度
        - critic_loss：Critic 网络损失，衡量价值估计质量
        - actor_loss：Actor 网络损失，衡量策略优化方向

    与 PPO 回调的区别：
        - SAC 是 off-policy，数据可以复用，没有 clip_fraction
        - SAC 有 alpha（温度参数）自动调节机制
        - SAC 有两个 Critic 网络，关注的是整体 critic_loss
    """

    def __init__(self, check_freq=1000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        # 记录训练过程中的指标
        self.episode_rewards = []
        self.alpha_list = []          # 熵系数（温度参数）
        self.critic_loss_list = []    # Critic 损失
        self.actor_loss_list = []     # Actor 损失
        self.entropy_list = []        # 策略熵
        self.timesteps_list = []      # 对应的时间步

    def _on_step(self):
        # 从信息字典中提取回合奖励（当回合结束时）
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])

        # 每隔 check_freq 步记录一次训练指标
        if self.num_timesteps % self.check_freq == 0 and self.num_timesteps > 0:
            logger = self.model.logger
            if hasattr(logger, "name_to_value"):
                name_to_value = logger.name_to_value

                # alpha：SAC 的温度参数
                # 自动调节模式下，alpha 会根据目标熵自适应调整
                # alpha 越大 → 鼓励更多探索
                # alpha 越小 → 更倾向于利用
                alpha = name_to_value.get("train/entropy_coef", 0)
                # critic_loss：两个 Q 网络的总损失
                # 衡量 Q 值对实际回报的拟合程度
                critic_loss = name_to_value.get("train/critic_loss", 0)
                # actor_loss：策略网络的损失
                # 包含 Q 值项和熵项
                actor_loss = name_to_value.get("train/actor_loss", 0)
                # entropy：当前策略的平均熵
                entropy = name_to_value.get("train/entropy", 0)

                self.alpha_list.append(alpha)
                self.critic_loss_list.append(critic_loss)
                self.actor_loss_list.append(actor_loss)
                self.entropy_list.append(entropy)
                self.timesteps_list.append(self.num_timesteps)

        return True


# ==========================================
# 第二部分：创建连续动作空间环境
# ==========================================
print("=" * 50)
print("第9章：SAC 训练 HalfCheetah-v4（连续控制）")
print("=" * 50)

print("\n正在创建 HalfCheetah-v4 环境...")

# HalfCheetah 是 MuJoCo 中的经典连续控制任务
# 特点：
#   - 状态空间：17 维（关节角度、速度等）
#   - 动作空间：6 维连续向量（各关节的力矩）
#   - 目标：让半猎豹机器人尽可能快地向前跑
#   - 奖励：前进速度 - 控制代价
env = gym.make("HalfCheetah-v4")

state_dim = env.observation_space.shape[0]   # 17
action_dim = env.action_space.shape[0]       # 6
action_low = env.action_space.low            # 动作下界
action_high = env.action_space.high          # 动作上界

print(f"  状态维度:   {state_dim}")
print(f"  动作维度:   {action_dim}")
print(f"  动作范围:   [{action_low[0]:.1f}, {action_high[0]:.1f}] × {action_dim}")
print(f"  动作类型:   连续（Box）")


# ==========================================
# 第三部分：配置 SAC 超参数
# ==========================================
print("\n配置 SAC 超参数...")

# SAC 的关键超参数解析：
#
# learning_rate=3e-4
#   学习率。SAC 通常使用与 PPO 相同的学习率
#   因为有熵正则化的保护，对学习率不太敏感
#
# buffer_size=100000
#   经验回放缓冲区大小
#   SAC 是 off-policy 算法，可以复用旧数据
#   缓冲区越大，数据多样性越好
#
# batch_size=256
#   每次更新使用的小批量大小
#   SAC 通常用较大的 batch_size（256 或 512）
#   比 PPO 的 64 大得多，因为 off-policy 更稳定
#
# tau=0.005
#   目标网络的软更新系数
#   θ_target ← τ * θ + (1 - τ) * θ_target
#   小 tau = 慢更新 = 更稳定，但跟踪延迟
#
# gamma=0.99
#   折扣因子，与 PPO 相同
#   控制对未来回报的重视程度
#
# ent_coef="auto"
#   熵系数自动调节（SAC 的核心创新！）
#   SAC 会自动调整 alpha 值来维持目标熵水平
#   默认目标熵 = -dim(A) = -6（动作维度的负数）

model = SAC(
    policy="MlpPolicy",          # 多层感知机策略
    env=env,                     # 训练环境
    learning_rate=3e-4,          # 学习率
    buffer_size=100_000,         # 经验回放缓冲区大小
    batch_size=256,              # 小批量大小
    tau=0.005,                   # 目标网络软更新系数
    gamma=0.99,                  # 折扣因子
    ent_coef="auto",             # 熵系数：自动调节（SAC 的核心创新）
    target_update_interval=1,    # 目标网络更新频率（每步更新）
    train_freq=1,                # 训练频率（每步训练一次）
    gradient_steps=1,            # 每次训练的梯度步数
    verbose=1,
    seed=42,
    device="auto",
    policy_kwargs=dict(
        net_arch=[256, 256],     # 网络结构：比 PPO 更宽
    ),
)

print(f"  学习率:         {model.learning_rate}")
print(f"  缓冲区大小:     {model.buffer_size}")
print(f"  批量大小:       {model.batch_size}")
print(f"  软更新系数 tau: {model.tau}")
print(f"  折扣因子 gamma: {model.gamma}")
print(f"  熵系数模式:     自动调节（ent_coef='auto'）")
print(f"  目标熵:         {-action_dim}（= -动作维度）")

# 解释 SAC 策略网络结构
# SAC 的 Actor 输出高斯分布的参数：均值 μ 和标准差 σ
# 动作采样：a ~ tanh(N(μ, σ²))
# 使用 tanh 压缩确保动作在有界范围内
print(f"\n  网络结构: {model.policy}")


# ==========================================
# 第四部分：训练模型
# ==========================================
print("\n开始训练（100000 时间步）...")
print("-" * 50)

# 创建训练监控回调
callback = SACTrainingCallback(check_freq=1000)

# 训练 100,000 个时间步（演示用，实际训练通常需要 1M+）
total_timesteps = 100_000
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

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("SAC 训练 HalfCheetah-v4 — 训练指标监控", fontsize=16, fontweight="bold")

# 子图1：回合奖励曲线
ax1 = axes[0, 0]
if callback.episode_rewards:
    rewards = callback.episode_rewards
    window = min(20, len(rewards))
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
    ax1.plot(rewards, alpha=0.3, color="#90CAF9", label="原始奖励")
    ax1.plot(range(window - 1, len(rewards)), smoothed,
             color="#2196F3", linewidth=2, label=f"滑动平均 (窗口={window})")
ax1.set_title("回合奖励", fontsize=13)
ax1.set_xlabel("回合")
ax1.set_ylabel("累计奖励")
ax1.legend()
ax1.grid(True, alpha=0.3)

# 子图2：熵系数（alpha）—— SAC 的核心创新
ax2 = axes[0, 1]
if callback.alpha_list:
    ax2.plot(callback.timesteps_list, callback.alpha_list,
             color="#FF9800", alpha=0.8, linewidth=1.5)
    # 标注：alpha 自动调节的含义
    ax2.annotate(
        "alpha 自适应下降\n→ 策略越来越确定",
        xy=(callback.timesteps_list[-1] * 0.6,
            max(callback.alpha_list) * 0.7),
        fontsize=9, color="gray", style="italic",
    )
ax2.set_title("熵系数 alpha（自动调节）", fontsize=13)
ax2.set_xlabel("时间步")
ax2.set_ylabel("alpha")
ax2.grid(True, alpha=0.3)

# 子图3：策略熵
ax3 = axes[0, 2]
if callback.entropy_list:
    ax3.plot(callback.timesteps_list, callback.entropy_list,
             color="#4CAF50", alpha=0.8, linewidth=1.5)
ax3.set_title("策略熵（探索程度）", fontsize=13)
ax3.set_xlabel("时间步")
ax3.set_ylabel("熵")
ax3.grid(True, alpha=0.3)

# 子图4：Critic 损失
ax4 = axes[1, 0]
if callback.critic_loss_list:
    ax4.plot(callback.timesteps_list, callback.critic_loss_list,
             color="#F44336", alpha=0.8, linewidth=1.5)
ax4.set_title("Critic 损失（双 Q 网络）", fontsize=13)
ax4.set_xlabel("时间步")
ax4.set_ylabel("损失值")
ax4.grid(True, alpha=0.3)

# 子图5：Actor 损失
ax5 = axes[1, 1]
if callback.actor_loss_list:
    ax5.plot(callback.timesteps_list, callback.actor_loss_list,
             color="#9C27B0", alpha=0.8, linewidth=1.5)
ax5.set_title("Actor 损失（策略优化）", fontsize=13)
ax5.set_xlabel("时间步")
ax5.set_ylabel("损失值")
ax5.grid(True, alpha=0.3)

# 子图6：奖励分布直方图
ax6 = axes[1, 2]
if callback.episode_rewards:
    # 将训练分为前半段和后半段，对比奖励分布
    mid = len(callback.episode_rewards) // 2
    first_half = callback.episode_rewards[:mid]
    second_half = callback.episode_rewards[mid:]
    ax6.hist(first_half, bins=20, alpha=0.5, color="#90CAF9", label="前半段")
    ax6.hist(second_half, bins=20, alpha=0.5, color="#2196F3", label="后半段")
    ax6.legend()
ax6.set_title("奖励分布（前半 vs 后半）", fontsize=13)
ax6.set_xlabel("回合奖励")
ax6.set_ylabel("频次")
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("output/sac_halfcheetah_curves.png", dpi=150, bbox_inches="tight")
print("训练曲线已保存至: output/sac_halfcheetah_curves.png")
plt.show()


# ==========================================
# 第六部分：评估训练好的模型
# ==========================================
print("\n正在评估最终模型（10 个测试回合）...")
print("-" * 50)

# 创建评估用的独立环境
eval_env = gym.make("HalfCheetah-v4")
mean_reward, std_reward = evaluate_policy(
    model, eval_env, n_eval_episodes=10, deterministic=True
)
print(f"10 回合测试结果：")
print(f"  平均奖励: {mean_reward:.2f}")
print(f"  标准差:   {std_reward:.2f}")

# 逐回合测试，展示详细结果
test_rewards = []
for ep in range(10):
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
    print(f"  回合 {i + 1:2d}: {r:8.2f}")

print(f"\n最高奖励: {max(test_rewards):.2f}")
print(f"最低奖励: {min(test_rewards):.2f}")
print(f"奖励标准差: {np.std(test_rewards):.2f}")
eval_env.close()


# ==========================================
# 第七部分：保存模型
# ==========================================
model.save("output/sac_halfcheetah")
print(f"\n模型已保存至: output/sac_halfcheetah.zip")

print("\n" + "=" * 50)
print("SAC 核心要点总结：")
print("  1. 熵正则化：在目标函数中加入策略熵，鼓励探索")
print("  2. 自动温度：alpha 参数自适应调节，无需手动调参")
print("  3. 双 Q 网络：取最小 Q 值，缓解过估计问题")
print("  4. 重参数化：使用重参数化技巧降低梯度方差")
print("  5. 随机策略：输出高斯分布，天然支持连续动作空间")
print("=" * 50)
