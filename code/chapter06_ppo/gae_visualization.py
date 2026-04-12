"""
第6章：GAE（广义优势估计）可视化
——直观理解 λ 和 γ 如何控制偏差-方差权衡

GAE 公式：
    δ_t = r_t + γ * V(s_{t+1}) - V(s_t)           # TD 误差
    A_t^GAE(γ,λ) = Σ_{l=0}^{∞} (γλ)^l * δ_{t+l}   # GAE 优势

λ 的含义：
    λ → 0: 高偏差、低方差（只看一步 TD 误差）
    λ → 1: 低偏差、高方差（趋向蒙特卡洛回报）

γ 的含义：
    γ → 0: 短视（只看即时奖励）
    γ → 1: 远见（重视长期累计奖励）

运行方式：
    python gae_visualization.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# 创建输出目录
os.makedirs("output", exist_ok=True)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 第一部分：GAE 计算函数
# ==========================================
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    计算广义优势估计 (GAE)

    参数：
        rewards: 奖励列表
        values:  价值估计列表 V(s)
        dones:   回合结束标志列表
        gamma:   折扣因子
        lam:     GAE lambda

    返回：
        advantages: 优势估计列表
        returns:    目标回报列表
    """
    advantages = []
    gae = 0

    # 在末尾追加一个 V(s_T+1)=0
    values = list(values) + [0.0]

    # 从后往前倒推计算 GAE
    for t in reversed(range(len(rewards))):
        if dones[t]:
            # 回合结束，重置
            gae = 0
            next_value = 0.0
        else:
            next_value = values[t + 1]

        # TD 误差：δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
        delta = rewards[t] + gamma * next_value - values[t]

        # GAE 累加：A_t = δ_t + γλ * A_{t+1}
        gae = delta + gamma * lam * gae

        advantages.insert(0, gae)

    # 目标回报 = 优势 + 价值
    returns = [a + v for a, v in zip(advantages, values[:-1])]

    return advantages, returns


def compute_mc_returns(rewards, gamma=0.99):
    """
    计算蒙特卡洛回报（从后往前累计折扣奖励）
    用于对比参考
    """
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


def compute_td_residuals(rewards, values, gamma=0.99):
    """
    计算单步 TD 误差
    δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
    """
    values = list(values) + [0.0]
    residuals = []
    for t in range(len(rewards)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        residuals.append(delta)
    return residuals


# ==========================================
# 第二部分：创建合成奖励序列
# ==========================================
print("=" * 60)
print("第6章：GAE（广义优势估计）可视化")
print("=" * 60)

# 场景：一个5步的稀疏奖励序列
# 前4步没有奖励，最后一步获得奖励 +1
# 这模拟了真实 RL 中的"延迟奖励"问题
rewards = [0.0, 0.0, 0.0, 0.0, 1.0]
n_steps = len(rewards)

# 假设价值函数的估计（不完美但大致正确）
# V(s) 在接近目标状态时逐渐增大
values = [0.1, 0.2, 0.4, 0.6, 0.9]

# 假设没有提前结束
dones = [False] * n_steps

print(f"\n合成场景设定:")
print(f"  奖励序列:     {rewards}")
print(f"  价值估计:     {values}")
print(f"  特点: 稀疏奖励 — 只有最后一步有奖励")

# 计算蒙特卡洛回报（参考基线）
mc_returns = compute_mc_returns(rewards, gamma=0.99)
print(f"  MC 回报:      {[f'{r:.4f}' for r in mc_returns]}")

# 计算单步 TD 误差
td_residuals = compute_td_residuals(rewards, values, gamma=0.99)
print(f"  TD 误差:      {[f'{r:.4f}' for r in td_residuals]}")


# ==========================================
# 第三部分：不同 λ 值的 GAE 对比
# ==========================================
print("\n" + "=" * 60)
print("不同 λ 值的 GAE 优势估计对比")
print("=" * 60)

lambda_values = [0.0, 0.5, 0.9, 0.95, 1.0]
gamma_fixed = 0.99

# 存储不同 λ 的优势值
advantages_by_lambda = {}
returns_by_lambda = {}

for lam in lambda_values:
    adv, ret = compute_gae(rewards, values, dones, gamma=gamma_fixed, lam=lam)
    advantages_by_lambda[lam] = adv
    returns_by_lambda[lam] = ret

# 打印对比表格
print(f"\n{'λ 值':<8}", end="")
for t in range(n_steps):
    print(f"{'步骤 ' + str(t):>12}", end="")
print()
print("-" * (8 + 12 * n_steps))

for lam in lambda_values:
    label = f"{lam:<8.2f}"
    print(label, end="")
    for t in range(n_steps):
        print(f"{advantages_by_lambda[lam][t]:>12.4f}", end="")
    print()

print(f"\n解释:")
print(f"  λ=0.0: 仅看单步 TD 误差 → 高偏差、低方差")
print(f"  λ=1.0: 等同于蒙特卡洛   → 低偏差、高方差")
print(f"  λ=0.95: PPO 的常用设置  → 折中方案")


# ==========================================
# 第四部分：不同 γ 值的 GAE 对比
# ==========================================
print("\n" + "=" * 60)
print("不同 γ 值的 GAE 优势估计对比（固定 λ=0.95）")
print("=" * 60)

gamma_values = [0.5, 0.9, 0.95, 0.99, 1.0]
lambda_fixed = 0.95

advantages_by_gamma = {}
returns_by_gamma = {}

for gamma in gamma_values:
    adv, ret = compute_gae(rewards, values, dones, gamma=gamma, lam=lambda_fixed)
    advantages_by_gamma[gamma] = adv
    returns_by_gamma[gamma] = ret

# 打印对比表格
print(f"\n{'γ 值':<8}", end="")
for t in range(n_steps):
    print(f"{'步骤 ' + str(t):>12}", end="")
print()
print("-" * (8 + 12 * n_steps))

for gamma in gamma_values:
    label = f"{gamma:<8.2f}"
    print(label, end="")
    for t in range(n_steps):
        print(f"{advantages_by_gamma[gamma][t]:>12.4f}", end="")
    print()

print(f"\n解释:")
print(f"  γ=0.5:  短视 — 只关心近期奖励")
print(f"  γ=0.99: PPO 常用 — 重视长期回报")
print(f"  γ=1.0:  完全远视 — 不折扣未来奖励")


# ==========================================
# 第五部分：绘制可视化图表
# ==========================================
print("\n正在生成可视化图表...")

# 创建图表：2行2列
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("GAE 广义优势估计 — 偏差与方差的权衡", fontsize=18, fontweight="bold")

# 颜色方案
colors_lambda = ["#F44336", "#FF9800", "#4CAF50", "#2196F3", "#9C27B0"]
colors_gamma = ["#E91E63", "#FF5722", "#009688", "#3F51B5", "#000000"]

steps = np.arange(n_steps)
step_labels = [f"步骤 {i}\n(r={rewards[i]})" for i in range(n_steps)]

# ---- 子图1：不同 λ 的优势曲线 ----
ax1 = axes[0, 0]
for i, lam in enumerate(lambda_values):
    adv = advantages_by_lambda[lam]
    ax1.plot(steps, adv, marker="o", linewidth=2.5, markersize=8,
             color=colors_lambda[i], label=f"λ = {lam}")

ax1.set_xticks(steps)
ax1.set_xticklabels(step_labels)
ax1.set_title("不同 λ 值的优势估计", fontsize=14, fontweight="bold")
ax1.set_ylabel("优势值 A(s)", fontsize=12)
ax1.legend(fontsize=11, loc="upper left")
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color="gray", linestyle="-", alpha=0.3)

# 添加注释说明 λ 的含义
ax1.annotate("λ→0: 高偏差、低方差\n（单步 TD）", xy=(0.5, 0.02),
             xycoords="axes fraction", fontsize=10, color="#F44336",
             style="italic", ha="left")
ax1.annotate("λ→1: 低偏差、高方差\n（蒙特卡洛）", xy=(0.5, 0.15),
             xycoords="axes fraction", fontsize=10, color="#9C27B0",
             style="italic", ha="left")

# ---- 子图2：不同 γ 的优势曲线 ----
ax2 = axes[0, 1]
for i, gamma in enumerate(gamma_values):
    adv = advantages_by_gamma[gamma]
    ax2.plot(steps, adv, marker="s", linewidth=2.5, markersize=8,
             color=colors_gamma[i], label=f"γ = {gamma}")

ax2.set_xticks(steps)
ax2.set_xticklabels(step_labels)
ax2.set_title("不同 γ 值的优势估计（λ=0.95）", fontsize=14, fontweight="bold")
ax2.set_ylabel("优势值 A(s)", fontsize=12)
ax2.legend(fontsize=11, loc="upper left")
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color="gray", linestyle="-", alpha=0.3)

# 添加注释说明 γ 的含义
ax2.annotate("γ→0: 短视\n（只看即时奖励）", xy=(0.02, 0.02),
             xycoords="axes fraction", fontsize=10, color="#E91E63",
             style="italic", ha="left")
ax2.annotate("γ→1: 远见\n（重视长期回报）", xy=(0.02, 0.15),
             xycoords="axes fraction", fontsize=10, color="#000000",
             style="italic", ha="left")

# ---- 子图3：不同 λ 的目标回报 ----
ax3 = axes[1, 0]
for i, lam in enumerate(lambda_values):
    ret = returns_by_lambda[lam]
    ax3.plot(steps, ret, marker="o", linewidth=2.5, markersize=8,
             color=colors_lambda[i], label=f"λ = {lam}")

# 同时画出 MC 回报作为参考
ax3.plot(steps, mc_returns, marker="*", linewidth=2, markersize=12,
         color="black", linestyle="--", label="MC 回报 (参考)")

ax3.set_xticks(steps)
ax3.set_xticklabels(step_labels)
ax3.set_title("不同 λ 值的目标回报", fontsize=14, fontweight="bold")
ax3.set_xlabel("时间步", fontsize=12)
ax3.set_ylabel("目标回报 G(s)", fontsize=12)
ax3.legend(fontsize=10, loc="upper left")
ax3.grid(True, alpha=0.3)

# ---- 子图4：偏差-方差权衡示意图 ----
ax4 = axes[1, 1]

# 创建偏差和方差的理论曲线
lams = np.linspace(0, 1, 100)
# 偏差随 λ 增大而减小（示意）
bias = np.exp(-3 * lams) * 1.0
# 方差随 λ 增大而增大（示意）
variance = (np.exp(2 * lams) - 1) / (np.exp(2) - 1) * 1.0
# 总误差 = 偏差² + 方差
total_error = bias ** 2 + variance

ax4.fill_between(lams, 0, bias ** 2, alpha=0.3, color="#2196F3", label="偏差²")
ax4.fill_between(lams, bias ** 2, bias ** 2 + variance, alpha=0.3, color="#F44336", label="方差")
ax4.plot(lams, total_error, color="black", linewidth=2.5, label="总误差")

# 标注最优 λ 的位置
optimal_idx = np.argmin(total_error)
optimal_lam = lams[optimal_idx]
ax4.axvline(x=optimal_lam, color="green", linestyle="--", linewidth=2, alpha=0.8)
ax4.annotate(f"最优 λ ≈ {optimal_lam:.2f}", xy=(optimal_lam, total_error[optimal_idx]),
             xytext=(optimal_lam + 0.15, total_error[optimal_idx] + 0.3),
             fontsize=12, color="green", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="green", lw=2))

# 标注常用范围
ax4.axvspan(0.9, 0.97, alpha=0.15, color="gold", label="PPO 常用范围 (0.9~0.97)")

ax4.set_xlabel("λ 值", fontsize=13)
ax4.set_ylabel("误差", fontsize=13)
ax4.set_title("偏差-方差权衡（示意）", fontsize=14, fontweight="bold")
ax4.legend(fontsize=11, loc="center right")
ax4.set_xlim(0, 1)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("output/gae_visualization.png", dpi=150, bbox_inches="tight")
print("图表已保存至: output/gae_visualization.png")
plt.show()


# ==========================================
# 第六部分：打印完整对比表
# ==========================================
print("\n" + "=" * 60)
print("完整对比表：不同 (γ, λ) 组合的优势值")
print("=" * 60)

# 精选组合
combos = [
    (0.99, 0.0,  "高偏差低方差极端"),
    (0.99, 0.5,  "中等平衡"),
    (0.99, 0.95, "PPO 推荐配置"),
    (0.99, 1.0,  "低偏差高方差极端"),
    (0.5,  0.95, "短视 + GAE"),
    (1.0,  0.95, "不折扣 + GAE"),
]

print(f"\n{'配置':<20} {'γ':>5} {'λ':>5}", end="")
for t in range(n_steps):
    print(f"  {'A(s'+str(t)+')':>8}", end="")
print()
print("-" * (20 + 5 + 5 + 10 * n_steps))

for gamma, lam, desc in combos:
    adv, _ = compute_gae(rewards, values, dones, gamma=gamma, lam=lam)
    print(f"{desc:<20} {gamma:>5.2f} {lam:>5.2f}", end="")
    for t in range(n_steps):
        print(f"  {adv[t]:>8.4f}", end="")
    print()

print("\n" + "=" * 60)
print("关键结论:")
print("  1. λ 控制优势估计的偏差-方差权衡")
print("  2. γ 控制对未来奖励的重视程度")
print("  3. PPO 常用配置: γ=0.99, λ=0.95")
print("  4. λ=0 → 一步 TD，λ=1 → 蒙特卡洛回报")
print("=" * 60)
