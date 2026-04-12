"""
第3章：贝尔曼方程的数值验证
用代码验证贝尔曼期望方程和贝尔曼最优方程

本实验展示：
1. 手工计算贝尔曼期望方程 V^π(s)
2. 用价值迭代（Value Iteration）数值求解 V^π(s) 和 V*(s)
3. 对比两种方法的结果，验证一致性
4. 展示价值迭代逐步收敛的过程

运行方式：
    python bellman_equation_verify.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# 创建输出目录
os.makedirs("output", exist_ok=True)


# ==========================================
# 第一部分：定义一个简单的 3 状态 MDP
# ==========================================
# 我们手工构造一个微型 MDP，方便手工计算和代码验证。
#
# 状态集合：S = {s0, s1, s2}
# 动作集合：A = {a0, a1}（每个状态可选两个动作）
#
# 转移概率 P(s'|s,a) 和奖励 R(s,a) 定义如下：
#
# 从 s0 出发：
#   执行 a0 → 转移到 s1（概率 1.0），奖励 = 1
#   执行 a1 → 转移到 s2（概率 1.0），奖励 = 2
#
# 从 s1 出发：
#   执行 a0 → 转移到 s0（概率 0.5），奖励 = -1
#                 转移到 s2（概率 0.5），奖励 = -1
#   执行 a1 → 转移到 s1（概率 0.8），奖励 = 0
#                 转移到 s2（概率 0.2），奖励 = 0
#
# 从 s2 出发：
#   执行 a0 → 转移到 s1（概率 1.0），奖励 = 3
#   执行 a1 → 转移到 s2（概率 1.0），奖励 = 1（自循环）

# 状态和动作数量
N_STATES = 3
N_ACTIONS = 2

# 转移概率：P[s][a] = {下一状态: 概率}
P = {
    0: {  # s0
        0: {1: 1.0},    # a0 → s1 概率 1.0
        1: {2: 1.0},    # a1 → s2 概率 1.0
    },
    1: {  # s1
        0: {0: 0.5, 2: 0.5},  # a0 → s0 概率 0.5, s2 概率 0.5
        1: {1: 0.8, 2: 0.2},  # a1 → s1 概率 0.8, s2 概率 0.2
    },
    2: {  # s2
        0: {1: 1.0},    # a0 → s1 概率 1.0
        1: {2: 1.0},    # a1 → s2 概率 1.0（自循环）
    },
}

# 奖励函数：R[s][a] = 立即奖励
R = {
    0: {0: 1, 1: 2},     # s0: a0 奖励 1, a1 奖励 2
    1: {0: -1, 1: 0},    # s1: a0 奖励 -1, a1 奖励 0
    2: {0: 3, 1: 1},     # s2: a0 奖励 3, a1 奖励 1
}

GAMMA = 0.9  # 折扣因子


# ==========================================
# 第二部分：贝尔曼期望方程 —— 手工计算
# ==========================================
def manual_bellman_expectation():
    """
    手工计算贝尔曼期望方程 V^π(s)

    给定一个固定策略 π，贝尔曼期望方程为：
        V^π(s) = Σ_a π(a|s) * [R(s,a) + γ * Σ_{s'} P(s'|s,a) * V^π(s')]

    我们使用一个简单的均匀随机策略：
        π(a|s) = 0.5（每个动作等概率选择）

    手工推导（令 V^π(s0) = v0, V^π(s1) = v1, V^π(s2) = v2）：

    V^π(s0) = 0.5 * [R(s0,a0) + γ * V^π(s1)] + 0.5 * [R(s0,a1) + γ * V^π(s2)]
            = 0.5 * [1 + 0.9 * v1] + 0.5 * [2 + 0.9 * v2]
            = 0.5 + 0.45*v1 + 1 + 0.45*v2
            = 1.5 + 0.45*v1 + 0.45*v2  ........................ (方程1)

    V^π(s1) = 0.5 * [R(s1,a0) + γ * (0.5*V^π(s0) + 0.5*V^π(s2))]
            + 0.5 * [R(s1,a1) + γ * (0.8*V^π(s1) + 0.2*V^π(s2))]
            = 0.5 * [-1 + 0.9*(0.5*v0 + 0.5*v2)]
            + 0.5 * [0 + 0.9*(0.8*v1 + 0.2*v2)]
            = -0.5 + 0.225*v0 + 0.225*v2 + 0.36*v1 + 0.09*v2
            = -0.5 + 0.225*v0 + 0.36*v1 + 0.315*v2  ............ (方程2)

    V^π(s2) = 0.5 * [R(s2,a0) + γ * V^π(s1)] + 0.5 * [R(s2,a1) + γ * V^π(s2)]
            = 0.5 * [3 + 0.9 * v1] + 0.5 * [1 + 0.9 * v2]
            = 1.5 + 0.45*v1 + 0.5 + 0.45*v2
            = 2.0 + 0.45*v1 + 0.45*v2  ........................ (方程3)
    """
    print("=" * 60)
    print("  贝尔曼期望方程 —— 手工推导")
    print("=" * 60)
    print()
    print("给定策略：均匀随机 π(a|s) = 0.5")
    print(f"折扣因子：γ = {GAMMA}")
    print()
    print("列方程组（v0 = V^π(s0), v1 = V^π(s1), v2 = V^π(s2)）：")
    print("  v0 = 1.5   + 0.45*v1 + 0.45*v2  ...... (方程1)")
    print("  v1 = -0.5  + 0.225*v0 + 0.36*v1 + 0.315*v2  (方程2)")
    print("  v2 = 2.0   + 0.45*v1 + 0.45*v2  ...... (方程3)")
    print()

    # 求解线性方程组 A * v = b
    # 方程1: v0 - 0.45*v1 - 0.45*v2 = 1.5
    # 方程2: -0.225*v0 + (1-0.36)*v1 - 0.315*v2 = -0.5
    # 方程3: -0.45*v1 + (1-0.45)*v2 = 2.0

    A = np.array([
        [1.0,   -0.45,   -0.45],
        [-0.225, 0.64,   -0.315],
        [0.0,   -0.45,    0.55],
    ])
    b = np.array([1.5, -0.5, 2.0])

    manual_V = np.linalg.solve(A, b)

    print("手工求解线性方程组得到：")
    for i in range(N_STATES):
        print(f"  V^π(s{i}) = {manual_V[i]:.6f}")
    print()
    return manual_V


# ==========================================
# 第三部分：策略评估 —— 迭代求解贝尔曼期望方程
# ==========================================
def policy_evaluation(policy, max_iter=1000, tol=1e-8):
    """
    策略评估：通过迭代求解贝尔曼期望方程

    贝尔曼期望方程（迭代形式）：
        V(s) ← Σ_a π(a|s) * [R(s,a) + γ * Σ_{s'} P(s'|s,a) * V(s')]

    反复迭代直到 V(s) 收敛，收敛后的 V(s) 就是 V^π(s)。

    参数：
        policy: 策略 π(a|s)，形状 (N_STATES, N_ACTIONS)
        max_iter: 最大迭代次数
        tol: 收敛阈值
    返回：
        V: 状态价值函数
        history: 每次迭代的 V 值记录（用于可视化收敛过程）
    """
    V = np.zeros(N_STATES)
    history = [V.copy()]

    for iteration in range(max_iter):
        V_new = np.zeros(N_STATES)

        for s in range(N_STATES):
            # 贝尔曼期望方程：对所有动作求加权和
            for a in range(N_ACTIONS):
                # π(a|s) * [R(s,a) + γ * Σ P(s'|s,a) * V(s')]
                action_value = R[s][a]
                for next_s, prob in P[s][a].items():
                    action_value += GAMMA * prob * V[next_s]
                V_new[s] += policy[s][a] * action_value

        # 检查是否收敛
        delta = np.max(np.abs(V_new - V))
        history.append(V_new.copy())
        V = V_new

        if delta < tol:
            break

    return V, history


# ==========================================
# 第四部分：价值迭代 —— 求解贝尔曼最优方程
# ==========================================
def value_iteration(max_iter=1000, tol=1e-8):
    """
    价值迭代：求解贝尔曼最优方程，找到 V*(s)

    贝尔曼最优方程（迭代形式）：
        V(s) ← max_a [R(s,a) + γ * Σ_{s'} P(s'|s,a) * V(s')]

    与贝尔曼期望方程的区别：
    - 期望方程：给定策略 π，求 V^π(s)
    - 最优方程：对所有策略取最优，求 V*(s)

    V*(s) 满足：
        V*(s) = max_a Σ_{s'} P(s'|s,a) [R(s,a) + γ * V*(s')]

    参数：
        max_iter: 最大迭代次数
        tol: 收敛阈值
    返回：
        V_star: 最优状态价值函数
        optimal_policy: 最优策略
        history: 收敛过程
    """
    V = np.zeros(N_STATES)
    history = [V.copy()]

    for iteration in range(max_iter):
        V_new = np.zeros(N_STATES)

        for s in range(N_STATES):
            # 对每个动作计算 Q(s, a)
            q_values = []
            for a in range(N_ACTIONS):
                q = R[s][a]
                for next_s, prob in P[s][a].items():
                    q += GAMMA * prob * V[next_s]
                q_values.append(q)

            # 贝尔曼最优方程：取最大值而不是期望
            V_new[s] = max(q_values)

        delta = np.max(np.abs(V_new - V))
        history.append(V_new.copy())
        V = V_new

        if delta < tol:
            break

    # 从 V* 提取最优策略
    optimal_policy = extract_optimal_policy(V)

    return V, optimal_policy, history


def extract_optimal_policy(V):
    """
    从最优价值函数 V* 提取最优策略 π*

    π*(s) = argmax_a [R(s,a) + γ * Σ_{s'} P(s'|s,a) * V*(s')]
    """
    policy = np.zeros((N_STATES, N_ACTIONS))

    for s in range(N_STATES):
        q_values = []
        for a in range(N_ACTIONS):
            q = R[s][a]
            for next_s, prob in P[s][a].items():
                q += GAMMA * prob * V[next_s]
            q_values.append(q)

        best_action = np.argmax(q_values)
        policy[s][best_action] = 1.0  # 确定性策略

    return policy


# ==========================================
# 第五部分：结果对比与可视化
# ==========================================
def verify_results():
    """验证手工计算与迭代计算的一致性"""

    # 均匀随机策略
    uniform_policy = np.ones((N_STATES, N_ACTIONS)) / N_ACTIONS

    print("=" * 60)
    print("  贝尔曼方程数值验证")
    print("=" * 60)
    print()

    # ------------------------------------------
    # 对比1：手工计算 vs 迭代求解（贝尔曼期望方程）
    # ------------------------------------------
    print("-" * 60)
    print("  对比1：贝尔曼期望方程 V^π(s) 的两种求解方式")
    print("-" * 60)

    # 手工计算
    manual_V = manual_bellman_expectation()

    # 迭代计算
    iter_V, iter_history = policy_evaluation(uniform_policy)
    print("策略评估（迭代求解）结果：")
    for i in range(N_STATES):
        print(f"  V^π(s{i}) = {iter_V[i]:.6f}")
    print()

    # 对比
    print(">>> 对比结果：")
    print(f"  {'状态':<8s} {'手工计算':<15s} {'迭代求解':<15s} {'误差':<15s}")
    for i in range(N_STATES):
        error = abs(manual_V[i] - iter_V[i])
        print(f"  s{i:<6d} {manual_V[i]:<15.8f} {iter_V[i]:<15.8f} {error:<15.2e}")

    all_match = np.allclose(manual_V, iter_V, atol=1e-6)
    print(f"\n  结论：{'完全一致 ✓' if all_match else '存在差异 ✗'}")
    print(f"  （手工求解线性方程组 = 迭代法逐步收敛，殊途同归！）")
    print()

    # ------------------------------------------
    # 对比2：贝尔曼期望方程 vs 贝尔曼最优方程
    # ------------------------------------------
    print("-" * 60)
    print("  对比2：V^π(s) vs V*(s)")
    print("-" * 60)
    print()

    V_star, optimal_policy, vi_history = value_iteration()

    print("贝尔曼期望方程 V^π(s)（均匀随机策略下）：")
    for i in range(N_STATES):
        print(f"  V^π(s{i}) = {iter_V[i]:.6f}")

    print()
    print("贝尔曼最优方程 V*(s)（最优策略下）：")
    for i in range(N_STATES):
        print(f"  V*(s{i}) = {V_star[i]:.6f}")

    print()
    print("最优策略 π*：")
    action_names = ['a0', 'a1']
    for s in range(N_STATES):
        best = np.argmax(optimal_policy[s])
        print(f"  π*(s{s}) = {action_names[best]}")

    print()
    print(f"  {'状态':<8s} {'V^π(s) 随机策略':<20s} {'V*(s) 最优策略':<20s} {'提升':<10s}")
    for i in range(N_STATES):
        improvement = V_star[i] - iter_V[i]
        print(f"  s{i:<6d} {iter_V[i]:<20.6f} {V_star[i]:<20.6f} {improvement:>+10.6f}")

    print()
    print("  分析：V*(s) ≥ V^π(s) 对所有状态成立（最优策略不会更差）")

    # ------------------------------------------
    # 可视化：价值迭代收敛过程
    # ------------------------------------------
    visualize_convergence(iter_history, vi_history)


def visualize_convergence(expectation_history, optimal_history):
    """
    可视化价值迭代的收敛过程

    左图：策略评估（贝尔曼期望方程）的收敛过程
    右图：价值迭代（贝尔曼最优方程）的收敛过程
    """
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ------------------------------------------
    # 左图：贝尔曼期望方程收敛过程
    # ------------------------------------------
    ax1 = axes[0]
    history_arr = np.array(expectation_history)
    colors = ['#e74c3c', '#2ecc71', '#3498db']
    state_labels = ['V^π(s0)', 'V^π(s1)', 'V^π(s2)']

    for s in range(N_STATES):
        ax1.plot(history_arr[:, s], color=colors[s], label=state_labels[s], linewidth=2)

    ax1.set_xlabel('迭代次数', fontsize=11)
    ax1.set_ylabel('状态价值 V(s)', fontsize=11)
    ax1.set_title('策略评估收敛过程\n（贝尔曼期望方程）', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 只显示前 30 次迭代（后面的已经收敛）
    n_show = min(30, len(expectation_history))
    ax1.set_xlim(0, n_show)

    # ------------------------------------------
    # 右图：贝尔曼最优方程收敛过程
    # ------------------------------------------
    ax2 = axes[1]
    history_arr2 = np.array(optimal_history)

    state_labels_star = ['V*(s0)', 'V*(s1)', 'V*(s2)']
    for s in range(N_STATES):
        ax2.plot(history_arr2[:, s], color=colors[s], label=state_labels_star[s], linewidth=2)

    ax2.set_xlabel('迭代次数', fontsize=11)
    ax2.set_ylabel('状态价值 V(s)', fontsize=11)
    ax2.set_title('价值迭代收敛过程\n（贝尔曼最优方程）', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    n_show2 = min(30, len(optimal_history))
    ax2.set_xlim(0, n_show2)

    plt.tight_layout()
    plt.savefig('output/bellman_equation_verify_results.png', dpi=150, bbox_inches='tight')
    print("\n图表已保存至 output/bellman_equation_verify_results.png")
    plt.show()


# ==========================================
# 第六部分：逐步展示价值迭代过程
# ==========================================
def print_value_iteration_steps(n_steps=10):
    """
    打印价值迭代的前 n_steps 步，展示逐步收敛的过程

    这让读者能直观看到：
    - 初始时 V(s) = 0（什么都不知道）
    - 每一步，V(s) 通过贝尔曼方程更新，向真实值逼近
    - 经过足够多步后，V(s) 收敛到最优值
    """
    print()
    print("=" * 60)
    print("  价值迭代逐步收敛过程")
    print("=" * 60)
    print()
    print("  迭代  |  V*(s0)    V*(s1)    V*(s2)   |  最大变化")
    print("  " + "-" * 55)

    V = np.zeros(N_STATES)

    for iteration in range(n_steps):
        V_new = np.zeros(N_STATES)

        for s in range(N_STATES):
            q_values = []
            for a in range(N_ACTIONS):
                q = R[s][a]
                for next_s, prob in P[s][a].items():
                    q += GAMMA * prob * V[next_s]
                q_values.append(q)
            V_new[s] = max(q_values)

        delta = np.max(np.abs(V_new - V))

        print(f"  {iteration + 1:4d}  |"
              f"  {V_new[0]:>8.4f}  {V_new[1]:>8.4f}  {V_new[2]:>8.4f} |"
              f"  {delta:>8.6f}")

        V = V_new

        if delta < 1e-8:
            print(f"\n  在第 {iteration + 1} 步收敛！")
            break

    print("  " + "-" * 55)
    print(f"\n  最终最优状态价值函数：")
    for i in range(N_STATES):
        print(f"    V*(s{i}) = {V[i]:.6f}")

    # 打印最优策略
    print(f"\n  最优策略：")
    action_names = ['a0', 'a1']
    for s in range(N_STATES):
        q_values = []
        for a in range(N_ACTIONS):
            q = R[s][a]
            for next_s, prob in P[s][a].items():
                q += GAMMA * prob * V[next_s]
            q_values.append(q)
        best = np.argmax(q_values)
        print(f"    π*(s{s}) = {action_names[best]}  "
              f"(Q(s{s},a0)={q_values[0]:.4f}, Q(s{s},a1)={q_values[1]:.4f})")


# ==========================================
# 主程序
# ==========================================
def main():
    """主函数：运行所有验证实验"""

    # 1. 验证手工计算与迭代计算的一致性
    verify_results()

    # 2. 展示价值迭代逐步收敛
    print_value_iteration_steps(n_steps=20)


if __name__ == "__main__":
    main()
