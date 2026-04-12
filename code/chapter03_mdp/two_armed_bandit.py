"""
第3章：双臂老虎机实验 —— 探索与利用的经典对比
比较四种策略：随机、贪心、ε-贪心、UCB
通过累计平均奖励和累计遗憾来评估各策略的表现

运行方式：
    python two_armed_bandit.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# 创建输出目录
os.makedirs("output", exist_ok=True)


# ==========================================
# 第一部分：双臂老虎机环境
# ==========================================
class TwoArmedBandit:
    """
    双臂老虎机环境
    - 拉杆 A：获奖概率 0.6（更好的选择）
    - 拉杆 B：获奖概率 0.4

    在强化学习中，这是一个最简化的决策问题：
    状态只有一个（始终相同），动作有两个（A 和 B），
    目标是最大化累计奖励。它虽然是 MDP 的退化形式，
    但完美地展示了"探索 vs 利用"的核心矛盾。
    """

    def __init__(self, prob_a=0.6, prob_b=0.4):
        self.prob_a = prob_a  # 拉杆 A 的获奖概率
        self.prob_b = prob_b  # 拉杆 B 的获奖概率
        # 最优拉杆的概率，用于计算遗憾（regret）
        self.best_prob = max(prob_a, prob_b)

    def pull(self, arm):
        """
        拉动指定的拉杆，返回奖励（0 或 1）

        参数：
            arm: 0 表示拉杆 A，1 表示拉杆 B
        返回：
            reward: 1 表示获奖，0 表示未获奖
        """
        if arm == 0:
            return 1 if np.random.random() < self.prob_a else 0
        else:
            return 1 if np.random.random() < self.prob_b else 0


# ==========================================
# 第二部分：四种策略的实现
# ==========================================

def strategy_random(bandit, n_steps):
    """
    策略一：随机策略
    每一步完全随机选择拉杆 A 或 B，不考虑历史信息。

    这是最基础的基线策略。因为它不做任何学习，
    平均奖励应该接近两个拉杆概率的均值：(0.6 + 0.4) / 2 = 0.5
    """
    rewards = []
    for _ in range(n_steps):
        arm = np.random.choice([0, 1])  # 等概率随机选择
        reward = bandit.pull(arm)
        rewards.append(reward)
    return np.array(rewards)


def strategy_greedy(bandit, n_steps):
    """
    策略二：贪心策略
    始终选择当前估计值最高的拉杆。

    问题：初始估计值相同时，第一步随机选一个拉杆，
    如果碰巧获奖，就永远只拉这个拉杆，不再探索另一个。
    这就是"过早收敛"的典型例子。
    """
    rewards = []
    # Q[a] 表示拉杆 a 的当前估计期望奖励
    Q = np.zeros(2)       # 初始估计值
    counts = np.zeros(2)  # 每个拉杆被拉的次数

    for _ in range(n_steps):
        # 始终选择当前估计值最高的拉杆（利用，不探索）
        arm = np.argmax(Q)
        reward = bandit.pull(arm)
        rewards.append(reward)

        # 更新该拉杆的估计值：增量式平均值
        counts[arm] += 1
        Q[arm] += (reward - Q[arm]) / counts[arm]

    return np.array(rewards)


def strategy_epsilon_greedy(bandit, n_steps, epsilon=0.1):
    """
    策略三：ε-贪心策略 (ε = 0.1)
    以概率 ε 随机探索，以概率 1-ε 选择当前最佳拉杆。

    ε = 0.1 意味着大约 10% 的时间在做随机探索。
    这是解决"探索 vs 利用"矛盾最常用的简单方法。
    ε 太大 → 浪费太多时间在已知不好的选择上；
    ε 太小 → 可能永远找不到最优拉杆。
    """
    rewards = []
    Q = np.zeros(2)
    counts = np.zeros(2)

    for _ in range(n_steps):
        # 以 ε 的概率随机探索，否则贪心选择
        if np.random.random() < epsilon:
            arm = np.random.choice([0, 1])  # 探索
        else:
            arm = np.argmax(Q)  # 利用

        reward = bandit.pull(arm)
        rewards.append(reward)

        counts[arm] += 1
        Q[arm] += (reward - Q[arm]) / counts[arm]

    return np.array(rewards)


def strategy_ucb(bandit, n_steps, c=2.0):
    """
    策略四：上置信界策略 (UCB, Upper Confidence Bound)
    选择 "估计值 + 不确定性上界" 最大的拉杆。

    UCB 公式：Q(a) + c * sqrt(ln(t) / N(a))
    - Q(a)：拉杆 a 的当前估计期望奖励
    - c：控制探索程度的参数（通常 c=2）
    - t：当前总步数
    - N(a)：拉杆 a 被选择的次数

    核心思想：如果一个拉杆很少被选（N(a) 小），
    那么对它的估计不确定，不确定性上界就大，
    所以 UCB 会倾向于去尝试那些"还不确定"的拉杆。
    随着尝试次数增加，不确定性降低，自然转向贪心。
    """
    rewards = []
    Q = np.zeros(2)
    counts = np.zeros(2)

    for t in range(1, n_steps + 1):
        # 前两步各拉一次，确保每个拉杆都被尝试过
        if t <= 2:
            arm = t - 1
        else:
            # 计算 UCB 值
            ucb_values = np.zeros(2)
            for a in range(2):
                # 不确定性上界：拉的次数越少，上界越大
                uncertainty = c * np.sqrt(np.log(t) / counts[a])
                ucb_values[a] = Q[a] + uncertainty
            arm = np.argmax(ucb_values)

        reward = bandit.pull(arm)
        rewards.append(reward)

        counts[arm] += 1
        Q[arm] += (reward - Q[arm]) / counts[arm]

    return np.array(rewards)


# ==========================================
# 第三部分：实验运行与结果对比
# ==========================================
def run_experiment():
    """
    主实验：每种策略运行 n_runs 次，每次 n_steps 步，取平均
    """
    n_steps = 1000  # 每次实验的步数
    n_runs = 200    # 重复实验次数（用于平滑曲线）

    # 用于累计各策略的结果
    all_rewards = {
        '随机策略': np.zeros(n_steps),
        '贪心策略': np.zeros(n_steps),
        'ε-贪心 (ε=0.1)': np.zeros(n_steps),
        'UCB (c=2)': np.zeros(n_steps),
    }

    print("=" * 60)
    print("  双臂老虎机实验：探索与利用策略对比")
    print("=" * 60)
    print(f"  拉杆 A 获奖概率: 0.6（最优）")
    print(f"  拉杆 B 获奖概率: 0.4")
    print(f"  每次实验步数: {n_steps}")
    print(f"  重复实验次数: {n_runs}")
    print("-" * 60)
    print("正在运行实验...")

    for run in range(n_runs):
        bandit = TwoArmedBandit(prob_a=0.6, prob_b=0.4)

        # 运行四种策略
        all_rewards['随机策略'] += strategy_random(bandit, n_steps)
        all_rewards['贪心策略'] += strategy_greedy(bandit, n_steps)
        all_rewards['ε-贪心 (ε=0.1)'] += strategy_epsilon_greedy(bandit, n_steps)
        all_rewards['UCB (c=2)'] += strategy_ucb(bandit, n_steps)

        if (run + 1) % 50 == 0:
            print(f"  已完成 {run + 1}/{n_runs} 轮实验...")

    # 计算平均值
    for key in all_rewards:
        all_rewards[key] /= n_runs

    print("实验完成！")
    print()

    # ==========================================
    # 第四部分：绘制累计平均奖励曲线
    # ==========================================
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 图1：累计平均奖励
    ax1 = axes[0]
    colors = ['#9E9E9E', '#FF9800', '#2196F3', '#4CAF50']
    for (name, rewards), color in zip(all_rewards.items(), colors):
        # 计算累计平均奖励
        cumulative_avg = np.cumsum(rewards) / np.arange(1, n_steps + 1)
        ax1.plot(cumulative_avg, label=name, color=color, alpha=0.85)

    ax1.axhline(y=0.6, color='red', linestyle='--', alpha=0.5, label='最优 (prob=0.6)')
    ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='随机基线 (0.5)')
    ax1.set_xlabel('步数', fontsize=12)
    ax1.set_ylabel('累计平均奖励', fontsize=12)
    ax1.set_title('累计平均奖励对比', fontsize=14)
    ax1.legend(fontsize=9, loc='right')
    ax1.set_ylim(0.35, 0.7)
    ax1.grid(True, alpha=0.3)

    # 图2：累计遗憾（regret）
    ax2 = axes[1]
    best_prob = 0.6
    for (name, rewards), color in zip(all_rewards.items(), colors):
        # 遗憾 = 每步最优奖励 - 实际获得的奖励
        regret = best_prob - rewards
        cumulative_regret = np.cumsum(regret)
        ax2.plot(cumulative_regret, label=name, color=color, alpha=0.85)

    ax2.set_xlabel('步数', fontsize=12)
    ax2.set_ylabel('累计遗憾', fontsize=12)
    ax2.set_title('累计遗憾对比（越低越好）', fontsize=14)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/two_armed_bandit_results.png', dpi=150, bbox_inches='tight')
    print("图表已保存至 output/two_armed_bandit_results.png")
    plt.show()

    # ==========================================
    # 第五部分：打印结果汇总表
    # ==========================================
    print()
    print("=" * 60)
    print("  实验结果汇总")
    print("=" * 60)
    print(f"{'策略':<20s} {'累计平均奖励':<15s} {'最终平均奖励':<15s} {'总遗憾':<10s}")
    print("-" * 60)

    for (name, rewards), color in zip(all_rewards.items(), colors):
        cum_avg = np.cumsum(rewards)
        final_avg = cum_avg[-1] / n_steps
        total_reward = np.sum(rewards)
        total_regret = best_prob * n_steps - total_reward
        print(f"{name:<20s} {final_avg:<15.4f} {rewards[-1]:<15.4f} {total_regret:<10.1f}")

    print("-" * 60)
    print()
    print("分析：")
    print("  - 随机策略：不做学习，奖励稳定在 0.5 附近（两个拉杆概率均值）")
    print("  - 贪心策略：可能过早锁定非最优拉杆，结果不稳定")
    print("  - ε-贪心策略：在探索和利用之间取得平衡，表现良好")
    print("  - UCB 策略：通过不确定性引导探索，通常表现最优")


if __name__ == "__main__":
    run_experiment()
