"""
第9章：连续控制算法大比拼 —— PPO vs TD3 vs SAC
——在相同环境下公平对比三种主流连续控制算法

运行方式：
    python ppo_td3_sac_comparison.py

三种算法的核心差异：

    PPO（Proximal Policy Optimization）—— 简单稳健的"老牌选手"
        类型：同策略（on-policy）
        策略：随机策略（高斯分布）
        核心机制：裁剪目标函数，限制策略更新幅度
        优点：实现简单、超参数鲁棒、训练稳定
        缺点：样本效率低（数据只能用一次）
        适用场景：快速原型验证、对稳定性要求高的场景

    TD3（Twin Delayed DDPG）—— 确定性策略的"精益求精"
        类型：异策略（off-policy）
        策略：确定性策略（直接输出动作）
        核心机制：双 Q 网络 + 延迟策略更新 + 目标策略平滑
        优点：样本效率高、在确定性任务上性能强劲
        缺点：确定性策略探索不足、超参数敏感
        适用场景：动作空间高维、奖励稀疏的精细控制

    SAC（Soft Actor-Critic）—— 最大熵强化学习的"全能选手"
        类型：异策略（off-policy）
        策略：随机策略（高斯分布 + 熵正则化）
        核心机制：熵正则化 + 自动温度调节 + 双 Q 网络
        优点：样本效率高、探索充分、鲁棒性强
        缺点：计算开销略大、理论上更复杂
        适用场景：通用连续控制、需要强探索的场景

公平对比的关键：
    - 相同环境（HalfCheetah-v4）
    - 相同训练预算（50000 时间步）
    - 相同随机种子
    - 相同网络结构规模
"""

import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, TD3, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

# 创建输出目录
os.makedirs("output", exist_ok=True)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 第一部分：通用训练回调 —— 记录回合奖励
# ==========================================
class RewardCallback(BaseCallback):
    """
    通用训练回调：记录每种算法训练过程中的回合奖励

    这个回调对所有 SB3 算法通用，因为回合奖励
    是通过 info["episode"]["r"] 统一获取的。
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_timesteps = []  # 记录每个回合结束时的时间步

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_timesteps.append(self.num_timesteps)
        return True


# ==========================================
# 第二部分：环境配置
# ==========================================
print("=" * 60)
print("第9章：连续控制算法大比拼 — PPO vs TD3 vs SAC")
print("=" * 60)

# 尝试使用 HalfCheetah-v4（需要 MuJoCo）
# 如果 MuJoCo 不可用，回退到 Pendulum-v1
ENV_NAME = "HalfCheetah-v4"
try:
    test_env = gym.make(ENV_NAME)
    test_env.reset()
    test_env.close()
    print(f"\n使用环境: {ENV_NAME}（MuJoCo 连续控制）")
except Exception as e:
    ENV_NAME = "Pendulum-v1"
    print(f"\nMuJoCo 不可用（{e}），回退到: {ENV_NAME}")

# 打印环境信息
env = gym.make(ENV_NAME)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_type = "连续" if isinstance(env.action_space, gym.spaces.Box) else "离散"
print(f"  状态维度:   {state_dim}")
print(f"  动作维度:   {action_dim}")
print(f"  动作类型:   {action_type}")
env.close()

# 统一训练参数
TOTAL_TIMESTEPS = 50_000    # 训练预算（演示用，实际需 1M+）
SEED = 42                   # 统一随机种子
NET_ARCH = [256, 256]       # 统一网络结构


# ==========================================
# 第三部分：训练三种算法
# ==========================================

# ---- 算法 1：PPO ----
print("\n" + "-" * 60)
print("【1/3】训练 PPO（Proximal Policy Optimization）")
print("-" * 60)
print("  特点：同策略、裁剪目标、简单但样本效率低")

# PPO 的超参数要点：
#   - n_steps=2048: 每次采集的步数，PPO 的"批大小"
#   - batch_size=64: 小批量更新大小
#   - n_epochs=10: 同一批数据复用 10 次
#   - clip_range=0.2: 策略比率裁剪范围
#   - ent_coef=0.01: 熵系数（手动设定，不像 SAC 自动调节）
ppo_model = PPO(
    policy="MlpPolicy",
    env=ENV_NAME,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    clip_range=0.2,
    ent_coef=0.01,
    gamma=0.99,
    gae_lambda=0.95,
    verbose=0,
    seed=SEED,
    device="auto",
    policy_kwargs=dict(net_arch=NET_ARCH),
)

ppo_callback = RewardCallback()
ppo_model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=ppo_callback,
    progress_bar=True,
)
print(f"  PPO 训练完成，共 {len(ppo_callback.episode_rewards)} 个回合")


# ---- 算法 2：TD3 ----
print("\n" + "-" * 60)
print("【2/3】训练 TD3（Twin Delayed DDPG）")
print("-" * 60)
print("  特点：异策略、确定性策略、双Q网络、延迟更新")

# TD3 的三大核心改进（在 DDPG 基础上）：
#   1. 双 Q 网络（Clipped Double-Q）：取两个 Q 值的较小值
#      → 缓解 Q 值过估计问题
#   2. 延迟策略更新（Delayed Policy Updates）：
#      → Critic 更新多次后，Actor 才更新一次
#      → policy_delay=2 表示每 2 次 Critic 更新，才做 1 次 Actor 更新
#   3. 目标策略平滑（Target Policy Smoothing）：
#      → 给目标动作加噪声，防止 Q 值在某些动作上出现尖峰
td3_model = TD3(
    policy="MlpPolicy",
    env=ENV_NAME,
    learning_rate=3e-4,
    buffer_size=100_000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    policy_delay=2,           # 延迟策略更新：每 2 次 Critic 更新后才更新 Actor
    action_noise=None,        # 动作噪声（TD3 内部会使用探索噪声）
    verbose=0,
    seed=SEED,
    device="auto",
    policy_kwargs=dict(net_arch=NET_ARCH),
)

td3_callback = RewardCallback()
td3_model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=td3_callback,
    progress_bar=True,
)
print(f"  TD3 训练完成，共 {len(td3_callback.episode_rewards)} 个回合")


# ---- 算法 3：SAC ----
print("\n" + "-" * 60)
print("【3/3】训练 SAC（Soft Actor-Critic）")
print("-" * 60)
print("  特点：异策略、随机策略、熵正则化、自动温度调节")

# SAC 的核心创新 —— 最大熵框架：
#   标准强化学习：max Σ r(s,a)
#   最大熵强化学习：max Σ [r(s,a) + α * H(π(·|s))]
#
# 其中 H 是策略熵，α 是温度参数
# 这使得 SAC 在追求高回报的同时，保持策略的随机性
#
# 自动温度调节原理：
#   alpha 的优化目标是让策略熵接近目标熵
#   目标熵 = -dim(A)（动作维度的负数）
#   当策略过于确定（熵太低）→ alpha 增大 → 鼓励探索
#   当策略过于随机（熵太高）→ alpha 减小 → 鼓励利用
sac_model = SAC(
    policy="MlpPolicy",
    env=ENV_NAME,
    learning_rate=3e-4,
    buffer_size=100_000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    ent_coef="auto",          # 自动温度调节（SAC 的核心创新！）
    train_freq=1,
    gradient_steps=1,
    verbose=0,
    seed=SEED,
    device="auto",
    policy_kwargs=dict(net_arch=NET_ARCH),
)

sac_callback = RewardCallback()
sac_model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=sac_callback,
    progress_bar=True,
)
print(f"  SAC 训练完成，共 {len(sac_callback.episode_rewards)} 个回合")


# ==========================================
# 第四部分：评估所有模型
# ==========================================
print("\n" + "=" * 60)
print("评估阶段：每个算法测试 10 个回合")
print("=" * 60)

eval_env = gym.make(ENV_NAME)
n_eval = 10

results = {}
for name, model in [("PPO", ppo_model), ("TD3", td3_model), ("SAC", sac_model)]:
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=n_eval, deterministic=True
    )
    # 逐回合测试
    test_rewards = []
    for _ in range(n_eval):
        obs, _ = eval_env.reset()
        done, truncated = False, False
        total_r = 0.0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, truncated, _ = eval_env.step(action)
            total_r += r
        test_rewards.append(total_r)

    results[name] = {
        "mean": mean_reward,
        "std": std_reward,
        "rewards": test_rewards,
    }
    print(f"  {name:4s}: 平均奖励 = {mean_reward:8.2f} ± {std_reward:6.2f}")

eval_env.close()


# ==========================================
# 第五部分：绘制对比图
# ==========================================
print("\n正在绘制对比图...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(
    f"连续控制算法对比 — {ENV_NAME}（{TOTAL_TIMESTEPS:,} 时间步）",
    fontsize=16, fontweight="bold",
)

# 颜色方案
colors = {"PPO": "#2196F3", "TD3": "#F44336", "SAC": "#4CAF50"}

# 子图1：训练曲线对比（原始值）
ax1 = axes[0, 0]
for name, cb in [("PPO", ppo_callback), ("TD3", td3_callback), ("SAC", sac_callback)]:
    if cb.episode_rewards:
        ax1.plot(cb.episode_rewards, alpha=0.3, color=colors[name], linewidth=0.8)
ax1.set_title("训练回合奖励（原始值）", fontsize=13)
ax1.set_xlabel("回合")
ax1.set_ylabel("累计奖励")
# 手动添加图例
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color=colors[n], linewidth=2, label=n)
    for n in ["PPO", "TD3", "SAC"]
]
ax1.legend(handles=legend_elements)
ax1.grid(True, alpha=0.3)

# 子图2：训练曲线对比（滑动平均）
ax2 = axes[0, 1]
for name, cb in [("PPO", ppo_callback), ("TD3", td3_callback), ("SAC", sac_callback)]:
    if cb.episode_rewards:
        rewards = cb.episode_rewards
        window = min(20, len(rewards))
        if window > 1:
            smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax2.plot(range(window - 1, len(rewards)), smoothed,
                     color=colors[name], linewidth=2, label=f"{name}")
ax2.set_title("训练回合奖励（滑动平均）", fontsize=13)
ax2.set_xlabel("回合")
ax2.set_ylabel("累计奖励")
ax2.legend()
ax2.grid(True, alpha=0.3)

# 子图3：最终评估对比（柱状图）
ax3 = axes[1, 0]
algo_names = list(results.keys())
means = [results[n]["mean"] for n in algo_names]
stds = [results[n]["std"] for n in algo_names]
bar_colors = [colors[n] for n in algo_names]
bars = ax3.bar(algo_names, means, yerr=stds, color=bar_colors,
               alpha=0.8, capsize=5, edgecolor="white", linewidth=1.5)
ax3.set_title("最终评估对比（10 回合平均）", fontsize=13)
ax3.set_ylabel("平均奖励")
# 在柱子上标注数值
for bar, mean, std in zip(bars, means, stds):
    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 10,
             f"{mean:.0f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax3.grid(True, alpha=0.3, axis="y")

# 子图4：测试回合奖励分布（箱线图）
ax4 = axes[1, 1]
box_data = [results[n]["rewards"] for n in algo_names]
bp = ax4.boxplot(box_data, labels=algo_names, patch_artist=True, widths=0.5)
for patch, color in zip(bp["boxes"], bar_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax4.set_title("测试回合奖励分布", fontsize=13)
ax4.set_ylabel("回合奖励")
ax4.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("output/ppo_td3_sac_comparison.png", dpi=150, bbox_inches="tight")
print("对比图已保存至: output/ppo_td3_sac_comparison.png")
plt.show()


# ==========================================
# 第六部分：打印对比总结表
# ==========================================
print("\n" + "=" * 60)
print("算法对比总结表")
print("=" * 60)

# 表头
print(f"{'指标':<20s} {'PPO':>10s} {'TD3':>10s} {'SAC':>10s}")
print("-" * 60)

# 最终奖励
print(f"{'最终平均奖励':<18s}", end="")
for name in algo_names:
    print(f" {results[name]['mean']:>10.1f}", end="")
print()

# 奖励标准差
print(f"{'奖励标准差':<18s}", end="")
for name in algo_names:
    print(f" {results[name]['std']:>10.1f}", end="")
print()

# 训练回合数
print(f"{'训练回合数':<18s}", end="")
for cb in [ppo_callback, td3_callback, sac_callback]:
    print(f" {len(cb.episode_rewards):>10d}", end="")
print()

# 算法类型
print(f"{'算法类型':<18s} {'同策略':>10s} {'异策略':>10s} {'异策略':>10s}")

# 策略类型
print(f"{'策略类型':<18s} {'随机':>10s} {'确定性':>10s} {'随机+熵':>10s}")

# 样本效率
print(f"{'样本效率':<18s} {'低':>10s} {'高':>10s} {'最高':>10s}")

# 探索机制
print(f"{'探索机制':<18s} {'策略内在':>10s} {'动作噪声':>10s} {'熵正则化':>10s}")

# 超参数敏感度
print(f"{'超参数敏感度':<18s} {'低':>10s} {'中':>10s} {'低':>10s}")

print("-" * 60)

# 确定赢家
winner = max(results.keys(), key=lambda k: results[k]["mean"])
print(f"\n在本实验中，{winner} 获得了最高平均奖励！")
print()
print("注意事项：")
print("  - 50k 时间步仅用于演示，实际对比通常需要 1M+ 时间步")
print("  - 不同环境上的排名可能不同")
print("  - PPO 的同策略特性使其在分布式训练中有独特优势")
print("  - SAC 在大多数 MuJoCo 环境上表现最好（尤其是在长训练后）")
print("  - TD3 在需要精确控制的场景（如机器人操纵）上很有竞争力")
print("=" * 60)
