"""
第13章：多智能体强化学习 (Multi-Agent RL) 实验
——从独立学习到协作学习

本实验构建一个简化的多智能体资源收集场景：
- 3个智能体在网格世界中移动
- 网格中散布资源点，智能体需要收集它们
- 智能体靠近时可以"共享"奖励（模拟协作）
- 对比两种学习范式：
    1. 独立学习 (Independent Q-Learning)：每个智能体独立训练
    2. 共享策略 (Shared Policy)：所有智能体共享 Q 表

运行方式：
    python multi_agent_marl.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# 创建输出目录
os.makedirs("output", exist_ok=True)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 第一部分：多智能体网格世界环境
# ==========================================
class MultiAgentGridWorld:
    """
    多智能体网格世界

    地图布局（8×8）：
        - 0: 空地
        - 1: 资源点（收集后消失，每回合重新生成）
        - 2: 墙壁（不可通过）

    智能体：
        - 3个智能体，初始位置分散在地图边缘
        - 每步可以选择：上、下、左、右、不动
        - 到达资源点自动收集

    协作机制：
        - 两个智能体同时收集相邻资源：额外 +3 协作奖励
        - 鼓励智能体分散收集而非扎堆

    奖励设计：
        - 收集资源: +5
        - 协作收集（相邻）: 额外 +3
        - 每步移动: -0.5（鼓励高效）
        - 撞墙: -1
    """

    # 动作：上、下、左、右、不动
    ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
    ACTION_NAMES = ['上', '下', '左', '右', '不动']
    N_ACTIONS = 5

    def __init__(self, grid_size=8, n_agents=3, n_resources=6):
        self.grid_size = grid_size
        self.n_agents = n_agents
        self.n_resources = n_resources

        # 固定墙壁位置
        self.walls = set()
        # 在地图中央放置一些墙壁，形成需要绕行的结构
        wall_positions = [(3, 3), (3, 4), (4, 3)]
        for wp in wall_positions:
            if 0 <= wp[0] < grid_size and 0 <= wp[1] < grid_size:
                self.walls.add(wp)

        # 智能体初始位置（分散在边缘）
        self.agent_starts = [(0, 0), (0, grid_size - 1), (grid_size - 1, 0)]

        # 资源位置（每回合重新生成）
        self.resource_positions = set()

        # 智能体当前位置
        self.agent_positions = []

    def reset(self):
        """
        重置环境

        重新放置智能体和资源，返回初始观测。
        """
        # 重置智能体位置
        self.agent_positions = [list(pos) for pos in self.agent_starts[:self.n_agents]]

        # 随机生成资源位置
        self.resource_positions = set()
        while len(self.resource_positions) < self.n_resources:
            r = np.random.randint(0, self.grid_size)
            c = np.random.randint(0, self.grid_size)
            pos = (r, c)
            # 资源不能放在墙上或智能体起始位置上
            if pos not in self.walls and pos not in self.agent_starts:
                self.resource_positions.add(pos)

        return self._get_observations()

    def _get_observations(self):
        """
        获取每个智能体的观测

        观测包含：
            - 自身位置
            - 最近资源的方向和距离
            - 其他智能体的相对位置

        为简化 Q-learning，将观测编码为一个离散状态 ID。
        """
        obs = []
        for i in range(self.n_agents):
            r, c = self.agent_positions[i]
            obs.append((r, c))
        return obs

    def step(self, actions):
        """
        执行所有智能体的动作

        参数：
            actions: 长度为 n_agents 的动作列表

        返回：
            observations: 新的观测
            rewards: 每个智能体的奖励
            done: 是否结束
            info: 额外信息
        """
        total_resources_before = len(self.resource_positions)
        rewards = [0.0] * self.n_agents
        collected_positions = []

        # 逐个智能体执行动作
        for i in range(self.n_agents):
            action = actions[i]
            dr, dc = self.ACTIONS[action]

            new_r = self.agent_positions[i][0] + dr
            new_c = self.agent_positions[i][1] + dc

            # 检查合法性
            if (new_r < 0 or new_r >= self.grid_size
                    or new_c < 0 or new_c >= self.grid_size
                    or (new_r, new_c) in self.walls):
                # 撞墙：位置不变，惩罚
                rewards[i] -= 1.0
                continue

            # 更新位置
            self.agent_positions[i] = [new_r, new_c]

            # 移动惩罚
            if action < 4:  # 不是"不动"
                rewards[i] -= 0.5

            # 检查是否收集到资源
            pos = (new_r, new_c)
            if pos in self.resource_positions:
                rewards[i] += 5.0
                collected_positions.append((i, pos))
                self.resource_positions.discard(pos)

        # 协作奖励：检查是否有多个智能体在相邻位置同时收集
        # 以及智能体之间的距离奖励
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                dist = abs(self.agent_positions[i][0] - self.agent_positions[j][0]) + \
                       abs(self.agent_positions[i][1] - self.agent_positions[j][1])
                # 相邻但不重叠：给协作奖励（鼓励分散但保持联络）
                if dist == 1:
                    # 检查两人是否都在资源附近
                    for _, res_pos in collected_positions:
                        dist_i = abs(self.agent_positions[i][0] - res_pos[0]) + \
                                 abs(self.agent_positions[i][1] - res_pos[1])
                        dist_j = abs(self.agent_positions[j][0] - res_pos[0]) + \
                                 abs(self.agent_positions[j][1] - res_pos[1])
                        if dist_i <= 1 and dist_j <= 1:
                            rewards[i] += 1.5  # 协作奖励
                            rewards[j] += 1.5

        # 检查是否所有资源都收集完毕
        done = len(self.resource_positions) == 0

        # 如果资源全收集了，给额外团队奖励
        if done:
            for i in range(self.n_agents):
                rewards[i] += 3.0  # 完成奖励

        info = {
            'resources_remaining': len(self.resource_positions),
            'resources_collected': total_resources_before - len(self.resource_positions),
        }

        return self._get_observations(), rewards, done, info


# ==========================================
# 第二部分：Q-Learning 智能体
# ==========================================
class QLearningAgent:
    """
    Q-Learning 智能体

    使用 ε-贪心策略和标准 Q-learning 更新。
    每个智能体维护自己的 Q 表。
    """

    def __init__(self, agent_id, n_actions=5, lr=0.1, gamma=0.95,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995):
        self.agent_id = agent_id
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Q 表：用字典存储，键为 (state, action)
        self.q_table = defaultdict(lambda: np.zeros(n_actions))

    def get_state_key(self, obs, env):
        """
        将观测转换为 Q 表的键

        状态编码：
            (自身行, 自身列, 最近资源方向, 资源剩余数)

        这里简化为只用自身位置 + 资源剩余数。
        """
        r, c = obs
        resources_left = len(env.resource_positions)
        return (r, c, resources_left)

    def select_action(self, state_key):
        """ε-贪心动作选择"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return int(np.argmax(self.q_table[state_key]))

    def update(self, state_key, action, reward, next_state_key, done):
        """
        Q-Learning 更新

        Q(s, a) ← Q(s, a) + α * [r + γ * max Q(s', a') - Q(s, a)]
        """
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state_key])

        td_error = target - self.q_table[state_key][action]
        self.q_table[state_key][action] += self.lr * td_error

    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


class SharedPolicyAgent:
    """
    共享策略智能体

    所有智能体共享一个 Q 表。
    这模拟了参数共享的多智能体方法（类似 IPPO 中的共享网络）。

    优点：
        - 样本效率更高（3个智能体的经验汇总训练）
        - 自动学到协作行为

    缺点：
        - 无法区分不同角色的策略
    """

    def __init__(self, n_agents=3, n_actions=5, lr=0.1, gamma=0.95,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995):
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # 共享的 Q 表
        self.q_table = defaultdict(lambda: np.zeros(n_actions))

    def get_state_key(self, obs, env, agent_id):
        """
        共享策略的状态编码

        包含 agent_id 以区分不同智能体的策略，
        但 Q 表本身是共享的。
        """
        r, c = obs
        resources_left = len(env.resource_positions)
        return (agent_id, r, c, resources_left)

    def select_action(self, state_key):
        """ε-贪心动作选择"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return int(np.argmax(self.q_table[state_key]))

    def update(self, state_key, action, reward, next_state_key, done):
        """Q-Learning 更新（所有智能体更新同一个 Q 表）"""
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state_key])

        td_error = target - self.q_table[state_key][action]
        self.q_table[state_key][action] += self.lr * td_error

    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


# ==========================================
# 第三部分：训练函数
# ==========================================
def train_independent(env, n_episodes=800, max_steps=100, verbose=True):
    """
    独立学习训练（Independent Q-Learning, IQL）

    每个智能体独立维护自己的 Q 表，
    不了解其他智能体的存在（把它们当作环境的一部分）。

    这是最简单的多智能体方法，但可能无法学到协作行为。
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"  训练模式：独立学习 (Independent Q-Learning)")
        print(f"  智能体数量: {env.n_agents}")
        print(f"  训练回合: {n_episodes}")
        print(f"{'='*60}")

    agents = [QLearningAgent(agent_id=i) for i in range(env.n_agents)]

    # 记录训练数据
    episode_rewards = []       # 团队总奖励
    agent_rewards = [[] for _ in range(env.n_agents)]  # 每个智能体的奖励
    cooperation_counts = []    # 协作事件计数
    completion_rates = []      # 完成率

    for episode in range(n_episodes):
        obs = env.reset()
        total_team_reward = 0
        individual_rewards = [0.0] * env.n_agents
        cooperation_events = 0

        # 记录初始状态键
        state_keys = [agents[i].get_state_key(obs[i], env) for i in range(env.n_agents)]

        for step in range(max_steps):
            # 每个智能体独立选择动作
            actions = [agents[i].select_action(state_keys[i]) for i in range(env.n_agents)]

            # 执行动作
            next_obs, rewards, done, info = env.step(actions)

            # 获取新状态键
            next_state_keys = [agents[i].get_state_key(next_obs[i], env)
                               for i in range(env.n_agents)]

            # 每个智能体独立更新 Q 表
            for i in range(env.n_agents):
                agents[i].update(state_keys[i], actions[i], rewards[i],
                                 next_state_keys[i], done)
                individual_rewards[i] += rewards[i]

            # 统计
            total_team_reward += sum(rewards)
            # 简单的协作检测：两个以上智能体同时获得正奖励
            positive_count = sum(1 for r in rewards if r > 2.0)
            if positive_count >= 2:
                cooperation_events += 1

            state_keys = next_state_keys

            if done:
                break

        # 衰减探索率
        for agent in agents:
            agent.decay_epsilon()

        # 记录本轮数据
        episode_rewards.append(total_team_reward)
        for i in range(env.n_agents):
            agent_rewards[i].append(individual_rewards[i])
        cooperation_counts.append(cooperation_events)
        completion_rates.append(1.0 if done else 0.0)

        # 每 200 回合打印进度
        if verbose and (episode + 1) % 200 == 0:
            avg_reward = np.mean(episode_rewards[-200:])
            avg_coop = np.mean(cooperation_counts[-200:])
            avg_complete = np.mean(completion_rates[-200:])
            print(f"  回合 {episode+1:4d} | "
                  f"团队平均奖励: {avg_reward:7.2f} | "
                  f"协作事件: {avg_coop:4.1f} | "
                  f"完成率: {avg_complete:.0%} | "
                  f"ε: {agents[0].epsilon:.3f}")

    return {
        'episode_rewards': episode_rewards,
        'agent_rewards': agent_rewards,
        'cooperation_counts': cooperation_counts,
        'completion_rates': completion_rates,
        'agents': agents,
    }


def train_shared(env, n_episodes=800, max_steps=100, verbose=True):
    """
    共享策略训练

    所有智能体共享一个 Q 表。
    相当于参数共享（weight sharing），提高样本效率。
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"  训练模式：共享策略 (Shared Policy)")
        print(f"  智能体数量: {env.n_agents}")
        print(f"  训练回合: {n_episodes}")
        print(f"{'='*60}")

    shared_agent = SharedPolicyAgent(n_agents=env.n_agents)

    # 记录训练数据
    episode_rewards = []
    agent_rewards = [[] for _ in range(env.n_agents)]
    cooperation_counts = []
    completion_rates = []

    for episode in range(n_episodes):
        obs = env.reset()
        total_team_reward = 0
        individual_rewards = [0.0] * env.n_agents
        cooperation_events = 0

        state_keys = [shared_agent.get_state_key(obs[i], env, i)
                      for i in range(env.n_agents)]

        for step in range(max_steps):
            # 所有智能体使用同一个 Q 表选择动作
            actions = [shared_agent.select_action(state_keys[i])
                       for i in range(env.n_agents)]

            next_obs, rewards, done, info = env.step(actions)
            next_state_keys = [shared_agent.get_state_key(next_obs[i], env, i)
                               for i in range(env.n_agents)]

            # 共享 Q 表更新（3个智能体的经验都更新同一个表）
            for i in range(env.n_agents):
                shared_agent.update(state_keys[i], actions[i], rewards[i],
                                    next_state_keys[i], done)
                individual_rewards[i] += rewards[i]

            total_team_reward += sum(rewards)
            positive_count = sum(1 for r in rewards if r > 2.0)
            if positive_count >= 2:
                cooperation_events += 1

            state_keys = next_state_keys

            if done:
                break

        shared_agent.decay_epsilon()

        episode_rewards.append(total_team_reward)
        for i in range(env.n_agents):
            agent_rewards[i].append(individual_rewards[i])
        cooperation_counts.append(cooperation_events)
        completion_rates.append(1.0 if done else 0.0)

        if verbose and (episode + 1) % 200 == 0:
            avg_reward = np.mean(episode_rewards[-200:])
            avg_coop = np.mean(cooperation_counts[-200:])
            avg_complete = np.mean(completion_rates[-200:])
            print(f"  回合 {episode+1:4d} | "
                  f"团队平均奖励: {avg_reward:7.2f} | "
                  f"协作事件: {avg_coop:4.1f} | "
                  f"完成率: {avg_complete:.0%} | "
                  f"ε: {shared_agent.epsilon:.3f}")

    return {
        'episode_rewards': episode_rewards,
        'agent_rewards': agent_rewards,
        'cooperation_counts': cooperation_counts,
        'completion_rates': completion_rates,
        'agent': shared_agent,
    }


# ==========================================
# 第四部分：可视化
# ==========================================
def visualize_results(ind_results, shared_results, n_agents=3):
    """
    可视化对比实验结果

    包含4个子图：
        1. 团队总奖励对比曲线
        2. 完成率对比
        3. 协作事件对比
        4. 各智能体个体奖励对比
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('多智能体强化学习 — 独立学习 vs 共享策略',
                 fontsize=16, fontweight='bold')

    window = 50  # 滑动平均窗口
    episodes = range(len(ind_results['episode_rewards']))

    # ---- 子图1：团队总奖励对比 ----
    ax1 = axes[0, 0]

    # 独立学习
    ind_rewards = ind_results['episode_rewards']
    if len(ind_rewards) >= window:
        ind_avg = np.convolve(ind_rewards, np.ones(window) / window, mode='valid')
        ax1.plot(range(window - 1, len(ind_rewards)), ind_avg,
                 color='#F44336', linewidth=2.5, label='独立学习 (IQL)')

    # 共享策略
    shared_rewards = shared_results['episode_rewards']
    if len(shared_rewards) >= window:
        shared_avg = np.convolve(shared_rewards, np.ones(window) / window, mode='valid')
        ax1.plot(range(window - 1, len(shared_rewards)), shared_avg,
                 color='#2196F3', linewidth=2.5, label='共享策略 (Shared)')

    ax1.set_xlabel('训练回合', fontsize=12)
    ax1.set_ylabel('团队总奖励', fontsize=12)
    ax1.set_title('团队总奖励对比', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # ---- 子图2：完成率对比 ----
    ax2 = axes[0, 1]

    ind_completion = ind_results['completion_rates']
    shared_completion = shared_results['completion_rates']

    # 计算滑动平均完成率
    if len(ind_completion) >= window:
        ind_comp_avg = np.convolve(ind_completion, np.ones(window) / window, mode='valid')
        ax2.plot(range(window - 1, len(ind_completion)), ind_comp_avg * 100,
                 color='#F44336', linewidth=2.5, label='独立学习 (IQL)')

    if len(shared_completion) >= window:
        shared_comp_avg = np.convolve(shared_completion, np.ones(window) / window, mode='valid')
        ax2.plot(range(window - 1, len(shared_completion)), shared_comp_avg * 100,
                 color='#2196F3', linewidth=2.5, label='共享策略 (Shared)')

    ax2.set_xlabel('训练回合', fontsize=12)
    ax2.set_ylabel('完成率 (%)', fontsize=12)
    ax2.set_title('任务完成率对比', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)

    # ---- 子图3：协作事件对比 ----
    ax3 = axes[1, 0]

    ind_coop = ind_results['cooperation_counts']
    shared_coop = shared_results['cooperation_counts']

    if len(ind_coop) >= window:
        ind_coop_avg = np.convolve(ind_coop, np.ones(window) / window, mode='valid')
        ax3.plot(range(window - 1, len(ind_coop)), ind_coop_avg,
                 color='#F44336', linewidth=2.5, label='独立学习 (IQL)')

    if len(shared_coop) >= window:
        shared_coop_avg = np.convolve(shared_coop, np.ones(window) / window, mode='valid')
        ax3.plot(range(window - 1, len(shared_coop)), shared_coop_avg,
                 color='#2196F3', linewidth=2.5, label='共享策略 (Shared)')

    ax3.set_xlabel('训练回合', fontsize=12)
    ax3.set_ylabel('协作事件数', fontsize=12)
    ax3.set_title('协作事件数对比', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)

    # ---- 子图4：各智能体个体奖励（共享策略） ----
    ax4 = axes[1, 1]

    agent_colors = ['#4CAF50', '#FF9800', '#9C27B0']
    for i in range(n_agents):
        rewards_i = shared_results['agent_rewards'][i]
        if len(rewards_i) >= window:
            avg_i = np.convolve(rewards_i, np.ones(window) / window, mode='valid')
            ax4.plot(range(window - 1, len(rewards_i)), avg_i,
                     color=agent_colors[i], linewidth=2, label=f'智能体 {i+1}')

    ax4.set_xlabel('训练回合', fontsize=12)
    ax4.set_ylabel('个体累计奖励', fontsize=12)
    ax4.set_title('各智能体奖励（共享策略）', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def visualize_trajectories(env, shared_agent, max_steps=30):
    """
    可视化训练后智能体的运行轨迹

    展示3个智能体如何在网格世界中协同收集资源。
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('多智能体运行轨迹可视化', fontsize=16, fontweight='bold')

    # 运行一回合
    obs = env.reset()
    agent_colors = ['#4CAF50', '#FF9800', '#9C27B0']
    trajectories = [[tuple(obs[i])] for i in range(env.n_agents)]
    initial_resources = set(env.resource_positions)
    collected_steps = []  # 记录资源被收集的步骤

    for step in range(max_steps):
        state_keys = [shared_agent.get_state_key(obs[i], env, i)
                      for i in range(env.n_agents)]
        actions = [shared_agent.select_action(state_keys[i])
                   for i in range(env.n_agents)]
        next_obs, rewards, done, info = env.step(actions)

        for i in range(env.n_agents):
            trajectories[i].append(tuple(next_obs[i]))

        if info['resources_collected'] > 0:
            collected_steps.append(step)

        obs = next_obs
        if done:
            break

    # ---- 左图：网格世界轨迹 ----
    ax1 = axes[0]

    # 绘制网格
    grid_display = np.zeros((env.grid_size, env.grid_size))
    for wr, wc in env.walls:
        grid_display[wr][wc] = -1

    ax1.imshow(grid_display, cmap='Greys', alpha=0.3,
               extent=[-0.5, env.grid_size - 0.5, env.grid_size - 0.5, -0.5])

    # 绘制初始资源位置
    for rr, rc in initial_resources:
        ax1.plot(rc, rr, '*', color='gold', markersize=15,
                 markeredgecolor='orange', markeredgewidth=1.5)

    # 绘制每个智能体的轨迹
    for i in range(env.n_agents):
        traj = trajectories[i]
        rows = [t[0] for t in traj]
        cols = [t[1] for t in traj]

        # 轨迹线
        ax1.plot(cols, rows, '-', color=agent_colors[i], linewidth=2,
                 alpha=0.7, label=f'智能体 {i+1}')
        # 起点
        ax1.plot(cols[0], rows[0], 'o', color=agent_colors[i], markersize=12)
        # 终点
        ax1.plot(cols[-1], rows[-1], 's', color=agent_colors[i], markersize=12)

        # 标注步数
        for step_idx, (r, c) in enumerate(traj):
            if step_idx % 5 == 0 and step_idx > 0:  # 每5步标一次
                ax1.text(c + 0.2, r - 0.2, str(step_idx), fontsize=7,
                         color=agent_colors[i], alpha=0.7)

    # 绘制墙壁
    for wr, wc in env.walls:
        ax1.add_patch(plt.Rectangle((wc - 0.5, wr - 0.5), 1, 1,
                                     facecolor='gray', alpha=0.5))
        ax1.text(wc, wr, '墙', ha='center', va='center', fontsize=10,
                 color='white', fontweight='bold')

    ax1.set_xlim(-0.5, env.grid_size - 0.5)
    ax1.set_ylim(env.grid_size - 0.5, -0.5)
    ax1.set_xticks(range(env.grid_size))
    ax1.set_yticks(range(env.grid_size))
    ax1.set_title('智能体运行轨迹', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # ---- 右图：步数 vs 收集资源数 ----
    ax2 = axes[1]

    total_steps = len(trajectories[0])
    resources_per_step = []
    remaining = len(initial_resources)

    # 重新模拟一遍来跟踪资源变化
    obs2 = env.reset()
    env.resource_positions = set(initial_resources)
    remaining_track = [len(env.resource_positions)]

    for step in range(total_steps - 1):
        state_keys = [shared_agent.get_state_key(obs2[i], env, i)
                      for i in range(env.n_agents)]
        actions = [shared_agent.select_action(state_keys[i])
                   for i in range(env.n_agents)]
        next_obs, rewards, done, info = env.step(actions)
        remaining_track.append(info['resources_remaining'])
        obs2 = next_obs
        if done:
            break

    ax2.fill_between(range(len(remaining_track)), remaining_track,
                     alpha=0.3, color='#4CAF50')
    ax2.plot(range(len(remaining_track)), remaining_track,
             '-o', color='#4CAF50', linewidth=2, markersize=5)
    ax2.set_xlabel('步数', fontsize=12)
    ax2.set_ylabel('剩余资源数', fontsize=12)
    ax2.set_title('资源收集进度', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 标注完成时间
    if remaining_track[-1] == 0:
        complete_step = len(remaining_track) - 1
        ax2.axvline(x=complete_step, color='red', linestyle='--', linewidth=1.5)
        ax2.annotate(f'全部收集完成\n(第{complete_step}步)',
                     xy=(complete_step, 0), xytext=(complete_step + 2, 2),
                     fontsize=10, color='red', fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color='red'))

    plt.tight_layout()
    return fig


def print_cooperation_statistics(ind_results, shared_results, n_agents):
    """
    打印协作统计信息
    """
    print("\n" + "=" * 60)
    print("  协作统计对比")
    print("=" * 60)

    # 取最后200个回合的统计数据
    last_n = 200

    for name, results in [("独立学习", ind_results), ("共享策略", shared_results)]:
        rewards = results['episode_rewards'][-last_n:]
        completion = results['completion_rates'][-last_n:]
        coop = results['cooperation_counts'][-last_n:]

        print(f"\n  [{name}] 最近 {last_n} 回合统计:")
        print(f"    平均团队奖励:   {np.mean(rewards):.2f}")
        print(f"    任务完成率:     {np.mean(completion):.0%}")
        print(f"    平均协作事件:   {np.mean(coop):.2f}")
        print(f"    团队奖励标准差: {np.std(rewards):.2f}")

        # 各智能体的贡献
        print(f"    各智能体平均奖励:", end="")
        for i in range(n_agents):
            avg_r = np.mean(results['agent_rewards'][i][-last_n:])
            print(f"  A{i+1}={avg_r:.1f}", end="")
        print()

    print("\n" + "-" * 60)


# ==========================================
# 第五部分：主程序
# ==========================================
def main():
    """
    主函数：创建环境 → 训练两种策略 → 对比分析 → 可视化

    实验流程：
        1. 创建多智能体网格世界
        2. 训练独立学习 (IQL) 智能体
        3. 训练共享策略智能体
        4. 对比两种方法的团队奖励、完成率、协作程度
        5. 可视化结果和智能体轨迹
    """

    print("=" * 60)
    print("  第13章：多智能体强化学习 (MARL) 实验")
    print("=" * 60)
    print("  场景: 多智能体资源收集")
    print("  智能体: 3个 (独立 Q-Learning vs 共享策略)")
    print("  任务: 在8x8网格世界中协作收集6个资源")
    print("-" * 60)

    # 设置随机种子
    np.random.seed(42)

    # ---- 步骤1：创建环境 ----
    print("\n[步骤1] 创建多智能体网格世界环境")
    env = MultiAgentGridWorld(grid_size=8, n_agents=3, n_resources=6)
    obs = env.reset()
    print(f"  网格大小: {env.grid_size}×{env.grid_size}")
    print(f"  智能体初始位置: {env.agent_starts}")
    print(f"  资源数量: {env.n_resources}")
    print(f"  墙壁位置: {env.walls}")

    # ---- 步骤2：训练独立学习智能体 ----
    print("\n[步骤2] 训练独立学习 (IQL) 智能体...")
    ind_results = train_independent(env, n_episodes=800, max_steps=80)

    # ---- 步骤3：训练共享策略智能体 ----
    print("\n[步骤3] 训练共享策略智能体...")
    shared_results = train_shared(env, n_episodes=800, max_steps=80)

    # ---- 步骤4：对比分析 ----
    print("\n[步骤4] 对比分析")
    print_cooperation_statistics(ind_results, shared_results, env.n_agents)

    # ---- 步骤5：可视化 ----
    print("[步骤5] 生成可视化图表...")

    # 图1：训练曲线对比
    fig1 = visualize_results(ind_results, shared_results, env.n_agents)
    fig1.savefig('output/marl_training_comparison.png', dpi=150, bbox_inches='tight')
    print("  训练曲线对比图已保存至: output/marl_training_comparison.png")

    # 图2：智能体轨迹可视化
    fig2 = visualize_trajectories(env, shared_results['agent'], max_steps=40)
    fig2.savefig('output/marl_trajectories.png', dpi=150, bbox_inches='tight')
    print("  智能体轨迹图已保存至: output/marl_trajectories.png")

    # ---- 关键结论 ----
    print("\n" + "=" * 60)
    print("  关键结论")
    print("=" * 60)
    print("  1. 独立学习 (IQL): 每个智能体独立优化，简单但缺乏协作")
    print("  2. 共享策略: 参数共享提高样本效率，自然涌现协作行为")
    print("  3. 协作奖励设计是多智能体 RL 的核心挑战之一")
    print("  4. 真实多智能体场景需要考虑:")
    print("     - 通信机制 (智能体之间如何传递信息)")
    print("     - 信用分配 (如何将团队奖励分配给个体)")
    print("     - 可扩展性 (智能体数量增多时的计算复杂度)")
    print("  5. 多智能体 RL 的前沿方向:")
    print("     - 基于自博弈 (Self-Play) 的多智能体训练")
    print("     - 大模型驱动的多智能体协作")
    print("     - 开放环境中的多智能体适应性")
    print("=" * 60)

    plt.show()


if __name__ == "__main__":
    main()
