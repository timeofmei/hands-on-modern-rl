"""
第8章：GRPO 核心机制演示 —— 群组相对策略优化
==========================================================

本脚本用合成数据逐步演示 GRPO (Group Relative Policy Optimization) 的核心思想：
  1. 生成一组响应（group），获得原始奖励
  2. 计算组内均值和标准差
  3. 通过组内归一化得到优势（advantage）
  4. 与 PPO 的 Critic 基线方法做对比
  5. 可视化归一化前后的奖励分布及优势对比

GRPO 的关键创新：
  - 不需要额外训练 Value Network（Critic）
  - 用同一问题的多个采样响应的组内统计量来替代基线
  - 大幅简化了训练流程，同时保持了优势函数的方差缩减效果

核心公式：
  advantage_i = (reward_i - mean(rewards)) / (std(rewards) + eps)

运行方式：
  python grpo_mechanism.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# 创建输出目录
os.makedirs("output", exist_ok=True)

# 设置中文字体，确保图表标题和标签正常显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 第一部分：GRPO 组内归一化优势计算
# ==========================================
def compute_grpo_advantages(rewards):
    """
    GRPO 的核心操作：对一组奖励进行组内归一化，得到优势值

    公式：
        mean_r = mean(rewards)
        std_r  = std(rewards) + eps   (eps 防止除零)
        advantage_i = (reward_i - mean_r) / std_r

    直觉理解：
        - advantage > 0 表示该响应"比组内平均水平好"，应该被鼓励
        - advantage < 0 表示该响应"比组内平均水平差"，应该被抑制
        - 归一化后优势值围绕 0 波动，方差为 1

    参数：
        rewards: numpy 数组，一组响应的原始奖励值
    返回：
        advantages: numpy 数组，归一化后的优势值
    """
    eps = 1e-8  # 极小常数，防止 std=0 时除零
    mean_r = rewards.mean()
    std_r = rewards.std() + eps
    advantages = (rewards - mean_r) / std_r
    return advantages


def compute_ppo_advantages(rewards, value_predictions):
    """
    PPO 风格的优势计算：使用 Critic 网络的值预测作为基线

    公式：
        advantage_i = reward_i - V(s_i)

    这需要额外训练一个 Value Network 来估计 V(s)，
    增加了训练复杂度，但在理论上更精确。

    参数：
        rewards: numpy 数组，一组响应的原始奖励值
        value_predictions: numpy 数组，Critic 网络对每个状态的值估计
    返回：
        advantages: numpy 数组，Critic 基线优势值
    """
    advantages = rewards - value_predictions
    return advantages


# ==========================================
# 第二部分：生成合成奖励数据
# ==========================================
def generate_synthetic_rewards(group_size=8, seed=42):
    """
    生成一组合成的奖励数据，模拟 GRPO 的采样过程

    场景设定：
        假设给模型一个问题，让它生成 group_size 个不同的回答，
        然后用奖励模型（或规则函数）给每个回答打分。

    奖励范围设定在 [0, 1] 之间，模拟常见的奖励模型输出：
        - 0.0 ~ 0.3：质量较差的回答
        - 0.3 ~ 0.7：中等质量的回答
        - 0.7 ~ 1.0：高质量回答

    参数：
        group_size: 每个问题的采样响应数量（GRPO 论文默认为 8~16）
        seed: 随机种子，确保结果可复现
    返回：
        rewards: numpy 数组，形状为 (group_size,)
    """
    np.random.seed(seed)
    # 模拟一组回答的奖励：大部分中等，少数较好或较差
    rewards = np.array([0.35, 0.52, 0.68, 0.41, 0.89, 0.73, 0.28, 0.61])
    return rewards


def simulate_critic_predictions(rewards, noise_scale=0.08, seed=123):
    """
    模拟 PPO 中 Critic 网络的值预测

    现实中 Critic 是一个独立训练的神经网络，这里用加噪声的方式来模拟：
        V(s) ≈ reward + noise
    Critic 预测通常接近真实奖励但不完全准确。

    参数：
        rewards: 真实奖励数组
        noise_scale: 噪声标准差，模拟 Critic 的预测误差
        seed: 随机种子
    返回：
        value_predictions: numpy 数组，模拟的 Critic 值预测
    """
    np.random.seed(seed)
    noise = np.random.normal(0, noise_scale, size=rewards.shape)
    value_predictions = rewards + noise
    # 确保值预测在合理范围内
    value_predictions = np.clip(value_predictions, 0.0, 1.0)
    return value_predictions


# ==========================================
# 第三部分：逐步展示 GRPO 计算过程
# ==========================================
def demonstrate_grpo_step_by_step():
    """
    完整演示 GRPO 的组内归一化过程，每一步打印详细数值

    步骤：
        Step 1: 展示原始奖励
        Step 2: 计算组内统计量（均值和标准差）
        Step 3: 归一化得到 GRPO 优势
        Step 4: 按优势排序
        Step 5: 与 PPO 的 Critic 基线优势做对比
    """
    group_size = 8

    print("=" * 70)
    print("  GRPO 组内归一化机制 —— 逐步演示")
    print("=" * 70)

    # ---------- Step 1：原始奖励 ----------
    rewards = generate_synthetic_rewards(group_size=group_size)
    print(f"\n【Step 1】原始奖励（group_size = {group_size}）")
    print("-" * 70)
    print("  模拟场景：给模型一个问题，生成 8 个不同回答，用奖励函数打分")
    print()
    for i, r in enumerate(rewards):
        bar = "|" * int(r * 30)  # 简单的文本柱状图
        print(f"  响应 {i+1}: reward = {r:.2f}  {bar}")

    # ---------- Step 2：组内统计量 ----------
    mean_r = rewards.mean()
    std_r = rewards.std()
    print(f"\n【Step 2】组内统计量")
    print("-" * 70)
    print(f"  组内均值 mean(r) = {mean_r:.4f}")
    print(f"  组内标准差 std(r) = {std_r:.4f}")
    print(f"  std(r) + 1e-8 = {std_r + 1e-8:.4f}  （加 epsilon 防止除零）")

    # ---------- Step 3：GRPO 归一化优势 ----------
    grpo_advantages = compute_grpo_advantages(rewards)
    print(f"\n【Step 3】GRPO 归一化优势 = (reward - mean) / (std + eps)")
    print("-" * 70)
    for i in range(group_size):
        adv = grpo_advantages[i]
        sign = "+" if adv >= 0 else ""
        print(f"  响应 {i+1}: ({rewards[i]:.2f} - {mean_r:.4f}) / {std_r + 1e-8:.4f}"
              f" = {sign}{adv:.4f}")

    print()
    print(f"  优势值均值: {grpo_advantages.mean():.6f}  （理论上应接近 0）")
    print(f"  优势值标准差: {grpo_advantages.std():.6f}  （理论上应接近 1）")

    # ---------- Step 4：按优势排序 ----------
    sorted_indices = np.argsort(grpo_advantages)[::-1]  # 降序排列
    print(f"\n【Step 4】按 GRPO 优势排序（从高到低）")
    print("-" * 70)
    print(f"  {'排名':>4s}  {'响应':>4s}  {'原始奖励':>8s}  {'GRPO优势':>10s}  {'解读'}")
    print(f"  {'----':>4s}  {'----':>4s}  {'--------':>8s}  {'----------':>10s}  {'----'}")
    for rank, idx in enumerate(sorted_indices):
        adv = grpo_advantages[idx]
        if adv > 0.5:
            interpretation = "显著优于组内平均，强烈鼓励"
        elif adv > 0:
            interpretation = "略优于组内平均，适度鼓励"
        elif adv > -0.5:
            interpretation = "略低于组内平均，适度抑制"
        else:
            interpretation = "显著低于组内平均，强烈抑制"
        print(f"  {rank+1:>4d}  {idx+1:>4d}  {rewards[idx]:>8.4f}  {adv:>+10.4f}  {interpretation}")

    # ---------- Step 5：与 PPO 对比 ----------
    value_preds = simulate_critic_predictions(rewards)
    ppo_advantages = compute_ppo_advantages(rewards, value_preds)
    print(f"\n【Step 5】GRPO vs PPO 优势对比")
    print("-" * 70)
    print(f"  {'响应':>4s}  {'原始奖励':>8s}  {'Critic预测':>10s}  "
          f"{'PPO优势':>10s}  {'GRPO优势':>10s}  {'差异':>8s}")
    print(f"  {'----':>4s}  {'--------':>8s}  {'----------':>10s}  "
          f"{'----------':>10s}  {'----------':>10s}  {'--------':>8s}")
    for i in range(group_size):
        diff = grpo_advantages[i] - ppo_advantages[i]
        print(f"  {i+1:>4d}  {rewards[i]:>8.4f}  {value_preds[i]:>10.4f}  "
              f"{ppo_advantages[i]:>+10.4f}  {grpo_advantages[i]:>+10.4f}  {diff:>+8.4f}")

    print()
    print(f"  PPO 优势  → 均值: {ppo_advantages.mean():.4f}, 标准差: {ppo_advantages.std():.4f}")
    print(f"  GRPO 优势 → 均值: {grpo_advantages.mean():.6f}, 标准差: {grpo_advantages.std():.4f}")
    print()
    print("  关键区别：")
    print("    PPO  优势 = reward - V(s)，需要额外训练 Critic 网络")
    print("    GRPO 优势 = (reward - mean) / std，仅依赖组内统计量，无需 Critic")

    return rewards, grpo_advantages, ppo_advantages


# ==========================================
# 第四部分：可视化 —— 奖励归一化前后对比
# ==========================================
def plot_reward_normalization(rewards, grpo_advantages):
    """
    绘制奖励归一化前后的柱状图对比

    左图：原始奖励（0~1 之间）
    右图：GRPO 归一化后的优势值（围绕 0 波动）
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    group_size = len(rewards)
    x = np.arange(group_size)

    # 左图：原始奖励
    colors_raw = []
    for r in rewards:
        if r >= 0.7:
            colors_raw.append('#2ecc71')   # 绿色：高质量
        elif r >= 0.4:
            colors_raw.append('#f39c12')   # 橙色：中等质量
        else:
            colors_raw.append('#e74c3c')   # 红色：低质量

    axes[0].bar(x, rewards, color=colors_raw, edgecolor='white', linewidth=1.5)
    axes[0].axhline(y=rewards.mean(), color='black', linestyle='--',
                     linewidth=1.5, label=f'均值 = {rewards.mean():.3f}')
    axes[0].set_xlabel('响应编号', fontsize=12)
    axes[0].set_ylabel('原始奖励', fontsize=12)
    axes[0].set_title('归一化前：原始奖励', fontsize=14)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'#{i+1}' for i in range(group_size)])
    axes[0].legend(fontsize=10)
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(True, alpha=0.3, axis='y')

    # 右图：GRPO 归一化后的优势
    colors_adv = []
    for a in grpo_advantages:
        if a > 0:
            colors_adv.append('#27ae60')   # 绿色：正优势（被鼓励）
        else:
            colors_adv.append('#c0392b')   # 红色：负优势（被抑制）

    axes[1].bar(x, grpo_advantages, color=colors_adv, edgecolor='white', linewidth=1.5)
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1.0)
    axes[1].set_xlabel('响应编号', fontsize=12)
    axes[1].set_ylabel('GRPO 优势值', fontsize=12)
    axes[1].set_title('归一化后：GRPO 优势', fontsize=14)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'#{i+1}' for i in range(group_size)])
    axes[1].grid(True, alpha=0.3, axis='y')

    # 添加注释
    axes[1].annotate('正优势 → 鼓励', xy=(4.5, grpo_advantages[4]),
                      xytext=(5.5, grpo_advantages[4] + 0.3),
                      fontsize=10, color='#27ae60',
                      arrowprops=dict(arrowstyle='->', color='#27ae60'))
    axes[1].annotate('负优势 → 抑制', xy=(6, grpo_advantages[6]),
                      xytext=(0.5, grpo_advantages[6] - 0.4),
                      fontsize=10, color='#c0392b',
                      arrowprops=dict(arrowstyle='->', color='#c0392b'))

    plt.suptitle('GRPO 组内归一化效果', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('output/grpo_reward_normalization.png', dpi=150, bbox_inches='tight')
    print("  奖励归一化对比图已保存为 output/grpo_reward_normalization.png")
    plt.show()


# ==========================================
# 第五部分：可视化 —— GRPO vs PPO 优势分布对比
# ==========================================
def plot_advantage_comparison(grpo_advantages, ppo_advantages):
    """
    绘制 GRPO 和 PPO 优势分布的对比图

    上图：两种方法的优势值柱状图并列对比
    下图：优势值的分布直方图
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    group_size = len(grpo_advantages)
    x = np.arange(group_size)
    bar_width = 0.35

    # 上图：并列柱状图
    bars1 = axes[0].bar(x - bar_width/2, grpo_advantages, bar_width,
                        label='GRPO 优势（组内归一化）', color='#3498db',
                        edgecolor='white', linewidth=1.0)
    bars2 = axes[0].bar(x + bar_width/2, ppo_advantages, bar_width,
                        label='PPO 优势（Critic 基线）', color='#e67e22',
                        edgecolor='white', linewidth=1.0)
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    axes[0].set_xlabel('响应编号', fontsize=12)
    axes[0].set_ylabel('优势值', fontsize=12)
    axes[0].set_title('GRPO vs PPO 优势值对比（逐响应）', fontsize=14)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'#{i+1}' for i in range(group_size)])
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3, axis='y')

    # 下图：分布直方图
    bins = np.linspace(-2, 2, 20)
    axes[1].hist(grpo_advantages, bins=bins, alpha=0.6, color='#3498db',
                 label='GRPO 优势分布', edgecolor='white')
    axes[1].hist(ppo_advantages, bins=bins, alpha=0.6, color='#e67e22',
                 label='PPO 优势分布', edgecolor='white')
    axes[1].axvline(x=0, color='black', linestyle='--', linewidth=1.0)
    axes[1].set_xlabel('优势值', fontsize=12)
    axes[1].set_ylabel('频次', fontsize=12)
    axes[1].set_title('GRPO vs PPO 优势值分布对比', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    # 添加统计信息
    stats_text = (
        f"GRPO: 均值={grpo_advantages.mean():.4f}, 标准差={grpo_advantages.std():.4f}\n"
        f"PPO:  均值={ppo_advantages.mean():.4f}, 标准差={ppo_advantages.std():.4f}"
    )
    axes[1].text(0.02, 0.95, stats_text, transform=axes[1].transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('output/grpo_vs_ppo_advantages.png', dpi=150, bbox_inches='tight')
    print("  优势分布对比图已保存为 output/grpo_vs_ppo_advantages.png")
    plt.show()


# ==========================================
# 第六部分：多组实验 —— 展示 GRPO 的稳定性
# ==========================================
def run_multiple_groups(num_groups=5, group_size=8):
    """
    运行多组实验，展示 GRPO 在不同奖励分布下的表现

    对比 GRPO 和 PPO 优势值的统计特性：
        - 均值是否稳定接近 0
        - 标准差是否稳定接近 1
    """
    print("\n" + "=" * 70)
    print("  多组实验：GRPO 归一化的稳定性")
    print("=" * 70)
    print()
    print(f"  实验设置：{num_groups} 组，每组 {group_size} 个响应")
    print()

    grpo_stats = {"mean": [], "std": []}
    ppo_stats = {"mean": [], "std": []}

    for g in range(num_groups):
        # 每组随机生成不同的奖励分布
        np.random.seed(g * 10 + 7)
        # 模拟不同难度的题目：奖励分布不同
        base = np.random.uniform(0.2, 0.8)
        spread = np.random.uniform(0.1, 0.3)
        rewards = np.clip(np.random.normal(base, spread, size=group_size), 0.0, 1.0)

        # 计算 GRPO 优势
        grpo_adv = compute_grpo_advantages(rewards)
        grpo_stats["mean"].append(grpo_adv.mean())
        grpo_stats["std"].append(grpo_adv.std())

        # 计算 PPO 优势
        value_preds = simulate_critic_predictions(rewards, noise_scale=0.1, seed=g * 5 + 3)
        ppo_adv = compute_ppo_advantages(rewards, value_preds)
        ppo_stats["mean"].append(ppo_adv.mean())
        ppo_stats["std"].append(ppo_adv.std())

        print(f"  第 {g+1} 组：")
        print(f"    原始奖励: mean={rewards.mean():.4f}, std={rewards.std():.4f}")
        print(f"    GRPO 优势: mean={grpo_adv.mean():.6f}, std={grpo_adv.std():.4f}")
        print(f"    PPO  优势: mean={ppo_adv.mean():.4f}, std={ppo_adv.std():.4f}")

    print()
    print("  【汇总统计】")
    print(f"  GRPO 优势均值 → 均值: {np.mean(grpo_stats['mean']):.6f}, "
          f"标准差: {np.std(grpo_stats['mean']):.6f}")
    print(f"  GRPO 优势标准差 → 均值: {np.mean(grpo_stats['std']):.4f}, "
          f"标准差: {np.std(grpo_stats['std']):.4f}")
    print()
    print(f"  PPO  优势均值 → 均值: {np.mean(ppo_stats['mean']):.4f}, "
          f"标准差: {np.std(ppo_stats['mean']):.4f}")
    print(f"  PPO  优势标准差 → 均值: {np.mean(ppo_stats['std']):.4f}, "
          f"标准差: {np.std(ppo_stats['std']):.4f}")
    print()
    print("  结论：GRPO 的优势均值严格为 0（数学保证），标准差严格为 1。")
    print("  而 PPO 的优势统计量取决于 Critic 网络的质量，存在波动。")


# ==========================================
# 程序入口
# ==========================================
if __name__ == "__main__":
    # 逐步演示 GRPO 核心计算
    rewards, grpo_advantages, ppo_advantages = demonstrate_grpo_step_by_step()

    # 可视化：奖励归一化前后对比
    print("\n" + "=" * 70)
    print("  开始生成可视化图表...")
    print("=" * 70)
    plot_reward_normalization(rewards, grpo_advantages)

    # 可视化：GRPO vs PPO 优势分布对比
    plot_advantage_comparison(grpo_advantages, ppo_advantages)

    # 多组实验
    run_multiple_groups()

    # 最终总结
    print("\n" + "=" * 70)
    print("  GRPO 核心机制总结")
    print("=" * 70)
    print("""
  GRPO (Group Relative Policy Optimization) 的核心思想：

  1. 对同一个问题，生成一组（group）多个响应
  2. 用奖励函数（可以是规则或模型）给每个响应打分
  3. 在组内进行归一化：
     advantage = (reward - mean) / (std + eps)
  4. 用归一化后的优势值更新策略

  相比 PPO 的优势：
    - 不需要训练 Critic（Value Network），节省大量计算
    - 组内归一化天然保证优势均值为 0、方差为 1
    - 训练流程更简洁，超参数更少

  相比 DPO 的优势：
    - 不需要成对的偏好数据（chosen/rejected）
    - 只需要可验证的奖励信号（规则/模型均可）
    - 特别适合数学推理等有明确正确答案的任务（RLVR）

  DeepSeek-R1 正是使用 GRPO + RLVR 实现了推理能力的飞跃。
    """)
