"""
第3章：4×4 GridWorld Q-Learning 实验
在网格世界中学习最优路径，直观理解 Q 值和贝尔曼方程

运行方式：
    python gridworld_q_learning.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# 创建输出目录
os.makedirs("output", exist_ok=True)


# ==========================================
# 第一部分：GridWorld 环境
# ==========================================
class GridWorld:
    """
    4×4 网格世界环境

    网格布局（4行4列）：
        0,0  0,1  0,2  0,3
        1,0  1,1  1,2  1,3
        2,0  2,1  2,2  2,3
        3,0  3,1  3,2  3,3

    - 起点：(0, 0)
    - 终点：(3, 3)，到达获得 +10 奖励
    - 障碍物：(1, 1) 和 (2, 2)，撞到获得 -5 惩罚
    - 每走一步：-1 奖励（鼓励尽快到达终点）
    - 撞墙：-5 奖励（位置不变）

    动作空间：
        0 = 上 (↑), 1 = 下 (↓), 2 = 左 (←), 3 = 右 (→)
    """

    def __init__(self):
        self.rows = 4
        self.cols = 4
        self.start = (0, 0)
        self.goal = (3, 3)
        self.obstacles = [(1, 1), (2, 2)]
        self.n_actions = 4  # 上、下、左、右
        self.action_names = ['上(↑)', '下(↓)', '左(←)', '右(→)']
        self.reset()

    def reset(self):
        """重置环境到起点，返回初始状态"""
        self.agent_pos = self.start
        return self.agent_pos

    def step(self, action):
        """
        执行动作，返回 (下一状态, 奖励, 是否结束)

        动作映射：
            0 = 上 → 行 -1
            1 = 下 → 行 +1
            2 = 左 → 列 -1
            3 = 右 → 列 +1
        """
        row, col = self.agent_pos

        # 根据动作计算新位置
        if action == 0:    # 上
            new_pos = (row - 1, col)
        elif action == 1:  # 下
            new_pos = (row + 1, col)
        elif action == 2:  # 左
            new_pos = (row, col - 1)
        elif action == 3:  # 右
            new_pos = (row, col + 1)
        else:
            raise ValueError(f"无效动作: {action}")

        # 检查是否撞墙（出界）
        new_row, new_col = new_pos
        if new_row < 0 or new_row >= self.rows or new_col < 0 or new_col >= self.cols:
            # 撞墙：位置不变，给予惩罚
            return self.agent_pos, -5, False

        # 检查是否撞到障碍物
        if new_pos in self.obstacles:
            # 撞障碍物：位置不变，给予惩罚
            return self.agent_pos, -5, False

        # 合法移动：更新位置
        self.agent_pos = new_pos

        # 检查是否到达终点
        if self.agent_pos == self.goal:
            return self.agent_pos, 10, True  # 到达终点，+10 奖励

        # 普通移动：-1 奖励（鼓励尽快到达）
        return self.agent_pos, -1, False


# ==========================================
# 第二部分：Q-Learning 算法
# ==========================================
def epsilon_greedy(Q, state, epsilon, n_actions):
    """
    ε-贪心动作选择策略

    以 ε 的概率随机探索，以 1-ε 的概率选择当前 Q 值最大的动作。
    这是 Q-Learning 中平衡"探索"与"利用"的标准方法。
    """
    if np.random.random() < epsilon:
        return np.random.randint(n_actions)  # 探索：随机选动作
    else:
        return np.argmax(Q[state])  # 利用：选 Q 值最大的动作


def train_q_learning(env, n_episodes=500, alpha=0.1, gamma=0.95,
                     epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
    """
    Q-Learning 训练

    Q-Learning 的核心更新公式（贝尔曼最优方程的迭代形式）：
        Q(s, a) ← Q(s, a) + α * [r + γ * max_a' Q(s', a') - Q(s, a)]

    其中：
        - s: 当前状态
        - a: 当前动作
        - r: 获得的奖励
        - s': 下一个状态
        - α: 学习率（控制更新步长）
        - γ: 折扣因子（未来奖励的重要程度）
        - max_a' Q(s', a'): 下一个状态的最大 Q 值

    参数：
        n_episodes: 训练回合数
        alpha: 学习率
        gamma: 折扣因子
        epsilon_start: 初始探索率
        epsilon_end: 最低探索率
        epsilon_decay: 探索率衰减因子
    """
    # 初始化 Q 表：所有 Q 值设为 0
    # Q[state][action] = 估计的最优动作价值
    Q = np.zeros((env.rows, env.cols, env.n_actions))

    # 记录训练过程中的数据
    episode_rewards = []  # 每回合的累计奖励
    episode_steps = []    # 每回合的步数
    epsilon = epsilon_start

    print("=" * 60)
    print("  Q-Learning 训练")
    print("=" * 60)
    print(f"  学习率 α = {alpha}")
    print(f"  折扣因子 γ = {gamma}")
    print(f"  初始探索率 ε = {epsilon_start}")
    print(f"  训练回合数 = {n_episodes}")
    print("-" * 60)

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done:
            # 1. 用 ε-贪心策略选择动作
            action = epsilon_greedy(Q, state, epsilon, env.n_actions)

            # 2. 执行动作，观察奖励和下一状态
            next_state, reward, done = env.step(action)

            # 3. Q-Learning 更新（核心公式）
            #    注意：这里用的是 max_a' Q(s', a')，不关心实际采取了什么策略
            #    这就是 Q-Learning "off-policy" 的特点
            best_next_q = np.max(Q[next_state])
            td_target = reward + gamma * best_next_q
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error

            # 4. 转移到下一状态
            state = next_state
            total_reward += reward
            steps += 1

            # 安全阀：防止无限循环
            if steps > 200:
                break

        # 衰减探索率
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        episode_rewards.append(total_reward)
        episode_steps.append(steps)

        # 每 100 回合打印一次进度
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_steps = np.mean(episode_steps[-100:])
            print(f"  回合 {episode + 1:4d} | "
                  f"近100回合平均奖励: {avg_reward:7.2f} | "
                  f"平均步数: {avg_steps:5.1f} | "
                  f"ε: {epsilon:.4f}")

    print("-" * 60)
    return Q, episode_rewards, episode_steps


# ==========================================
# 第三部分：结果可视化
# ==========================================
def print_q_table(Q, env):
    """
    打印格式化的 Q 表

    Q 表展示了每个状态下每个动作的 Q 值（估计的最优动作价值）。
    Q 值越高，说明在该状态下执行该动作的预期累计奖励越大。
    """
    print("\n" + "=" * 60)
    print("  最终 Q 表")
    print("=" * 60)
    print(f"{'状态':<10s}", end="")
    for name in env.action_names:
        print(f"{name:<12s}", end="")
    print(f"{'最优动作':<12s}")
    print("-" * 60)

    for r in range(env.rows):
        for c in range(env.cols):
            state = (r, c)
            if state in env.obstacles:
                print(f"({r},{c}) 障碍  ", end="")
                print("    ---      ---      ---      ---     障碍物")
                continue
            if state == env.goal:
                print(f"({r},{c}) 终点  ", end="")
                print("    ---      ---      ---      ---     终点")
                continue

            print(f"({r},{c})       ", end="")
            for a in range(env.n_actions):
                print(f"{Q[r][c][a]:>8.2f}   ", end="")
            best_action = np.argmax(Q[r][c])
            print(f"  {env.action_names[best_action]}")

    print("-" * 60)


def extract_optimal_path(Q, env):
    """
    从 Q 表中提取最优路径

    在每个状态选择 Q 值最大的动作，根据网格规则计算下一状态。
    不依赖环境的 step() 函数，避免环境状态被意外修改。
    """
    # 动作对应的位移：0=上, 1=下, 2=左, 3=右
    deltas = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    state = env.start
    path = [state]
    visited = set()

    while state != env.goal:
        if state in visited:
            break  # 防止死循环
        visited.add(state)
        action = np.argmax(Q[state])
        dr, dc = deltas[action]
        new_state = (state[0] + dr, state[1] + dc)

        # 检查新位置是否合法（不越界、不是障碍物）
        if (0 <= new_state[0] < env.rows and 0 <= new_state[1] < env.cols
                and new_state not in env.obstacles):
            state = new_state
        # 如果越界或撞障碍物，状态不变（可能导致死循环，由 visited 保护）
        path.append(state)
        if state == env.goal:
            break

    return path


def visualize_results(Q, episode_rewards, env):
    """
    可视化 Q-Learning 的学习结果
    - 图1：每个动作的 Q 值热力图
    - 图2：最优路径在网格上的展示
    - 图3：每回合累计奖励的变化曲线
    """
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure(figsize=(16, 12))

    # ------------------------------------------
    # 图1：四个动作的 Q 值热力图
    # ------------------------------------------
    action_names_short = ['上(↑)', '下(↓)', '左(←)', '右(→)']

    for i in range(4):
        ax = fig.add_subplot(2, 3, i + 1)
        q_values = Q[:, :, i]  # 取出某个动作在所有状态的 Q 值

        im = ax.imshow(q_values, cmap='RdYlGn', aspect='equal')
        # 在每个格子上标注 Q 值
        for r in range(env.rows):
            for c in range(env.cols):
                if (r, c) in env.obstacles:
                    ax.text(c, r, 'X', ha='center', va='center',
                            fontsize=14, fontweight='bold', color='black')
                elif (r, c) == env.goal:
                    ax.text(c, r, 'G', ha='center', va='center',
                            fontsize=14, fontweight='bold', color='blue')
                else:
                    ax.text(c, r, f'{q_values[r, c]:.1f}', ha='center', va='center',
                            fontsize=9)
        ax.set_title(f'Q(s, {action_names_short[i]})', fontsize=12)
        ax.set_xticks(range(env.cols))
        ax.set_yticks(range(env.rows))
        ax.set_xticklabels(range(env.cols))
        ax.set_yticklabels(range(env.rows))
        plt.colorbar(im, ax=ax, shrink=0.8)

    # ------------------------------------------
    # 图2：最优路径可视化
    # ------------------------------------------
    ax_path = fig.add_subplot(2, 3, 5)
    # 绘制网格底色
    grid = np.zeros((env.rows, env.cols))
    for obs in env.obstacles:
        grid[obs] = -1
    grid[env.goal] = 2

    ax_path.imshow(grid, cmap='Set3', aspect='equal', vmin=-2, vmax=3)

    # 提取并绘制最优路径
    path = extract_optimal_path(Q, env)
    path_rows = [p[0] for p in path]
    path_cols = [p[1] for p in path]
    ax_path.plot(path_cols, path_rows, 'b-o', linewidth=2.5, markersize=10)

    # 标注起点、终点、障碍物
    ax_path.text(0, 0, 'S', ha='center', va='center', fontsize=16,
                 fontweight='bold', color='green')
    ax_path.text(3, 3, 'G', ha='center', va='center', fontsize=16,
                 fontweight='bold', color='red')
    for obs in env.obstacles:
        ax_path.text(obs[1], obs[0], 'X', ha='center', va='center',
                     fontsize=16, fontweight='bold', color='black')

    # 在路径上标注步数
    for idx, (r, c) in enumerate(path):
        ax_path.text(c, r, str(idx), ha='center', va='center',
                     fontsize=8, color='white',
                     bbox=dict(boxstyle='round,pad=0.2', fc='blue', alpha=0.5))

    ax_path.set_title('最优路径', fontsize=12)
    ax_path.set_xticks(range(env.cols))
    ax_path.set_yticks(range(env.rows))
    ax_path.set_xticklabels(range(env.cols))
    ax_path.set_yticklabels(range(env.rows))
    ax_path.grid(True, alpha=0.3)

    # ------------------------------------------
    # 图3：训练奖励曲线
    # ------------------------------------------
    ax_reward = fig.add_subplot(2, 3, 6)
    ax_reward.plot(episode_rewards, alpha=0.3, color='lightblue', label='单回合奖励')
    # 计算滑动平均
    window = 20
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards,
                                 np.ones(window) / window, mode='valid')
        ax_reward.plot(range(window - 1, len(episode_rewards)),
                       moving_avg, color='blue', linewidth=2,
                       label=f'{window}回合滑动平均')
    ax_reward.set_xlabel('回合', fontsize=11)
    ax_reward.set_ylabel('累计奖励', fontsize=11)
    ax_reward.set_title('训练奖励曲线', fontsize=12)
    ax_reward.legend(fontsize=9)
    ax_reward.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/gridworld_q_learning_results.png', dpi=150, bbox_inches='tight')
    print("\n图表已保存至 output/gridworld_q_learning_results.png")
    plt.show()


# ==========================================
# 第四部分：主程序
# ==========================================
def main():
    """主函数：创建环境 → 训练 → 打印 Q 表 → 可视化"""

    # 创建 GridWorld 环境
    env = GridWorld()
    print("GridWorld 环境创建完成")
    print(f"  起点: {env.start}")
    print(f"  终点: {env.goal}")
    print(f"  障碍物: {env.obstacles}")
    print(f"  动作空间: {env.action_names}")

    # 训练 Q-Learning
    Q, episode_rewards, episode_steps = train_q_learning(env, n_episodes=500)

    # 打印最终 Q 表
    print_q_table(Q, env)

    # 提取并打印最优路径
    path = extract_optimal_path(Q, env)
    print(f"\n最优路径: {' → '.join([str(p) for p in path])}")
    print(f"路径长度: {len(path) - 1} 步")

    # 计算最优路径的总奖励
    total_r = 0
    for i in range(len(path) - 1):
        if i < len(path) - 2:
            total_r += -1  # 普通步骤：-1
        else:
            total_r += 10  # 到达终点的最后一步：+10
    print(f"最优路径总奖励: {total_r}")

    # 可视化结果
    visualize_results(Q, episode_rewards, env)


if __name__ == "__main__":
    main()
