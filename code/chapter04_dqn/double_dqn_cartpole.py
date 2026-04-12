"""
第4章：Double DQN —— 解决 DQN 的过估计问题
在 CartPole-v1 上对比标准 DQN 和 Double DQN

问题背景：
    标准 DQN 在计算目标值时，用同一个网络既选动作又评估 Q 值：
        target = r + γ * max_a Q_target(s', a)
    这会导致 Q 值的系统性过估计 (overestimation)。

解决方案 —— Double DQN (Hasselt et al., 2016)：
    将"选动作"和"评估 Q 值"解耦：
    1. 用 Q 网络选择最优动作：a* = argmax_a Q(s', a)
    2. 用目标网络评估该动作的 Q 值：Q_target(s', a*)
    即：target = r + γ * Q_target(s', argmax_a Q(s', a))

    这样可以有效缓解过估计，提升训练稳定性和最终性能。

运行方式：
    python double_dqn_cartpole.py
"""

import os
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt

# 创建输出目录
os.makedirs("output", exist_ok=True)


# ==========================================
# 第一部分：Q 网络（与标准 DQN 相同）
# ==========================================
class QNetwork(nn.Module):
    """
    Q 网络：将状态映射到每个动作的 Q 值
    结构：state_dim → 128 → 128 → action_dim
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)


# ==========================================
# 第二部分：经验回放缓冲区（与标准 DQN 相同）
# ==========================================
class ReplayBuffer:
    """经验回放缓冲区：存储和采样训练数据"""

    def __init__(self, capacity=10000):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ==========================================
# 第三部分：标准 DQN 智能体
# ==========================================
class DQNAgent:
    """
    标准 DQN 智能体

    目标值计算方式：
        target = r + γ * max_a' Q_target(s', a')

    注意：max 操作既用于"选动作"又用于"评估 Q 值"，
    这就是过估计的根源。
    """

    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.action_dim = action_dim
        self.gamma = gamma

        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(capacity=10000)

    def select_action(self, state, epsilon):
        """ε-贪心动作选择"""
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_net(state_tensor)
            return q_values.argmax(dim=1).item()

    def update(self, batch_size):
        """
        标准 DQN 更新

        目标值：r + γ * Q_target(s').max()
        用目标网络直接取最大 Q 值。
        """
        if len(self.buffer) < batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # 当前 Q 值
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # 标准 DQN 目标值：直接用目标网络取最大值
        with torch.no_grad():
            next_q_max = self.target_net(next_states).max(dim=1)[0]
            targets = rewards + self.gamma * next_q_max * (1 - dones)

        loss = nn.MSELoss()(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10)
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        """硬更新目标网络"""
        self.target_net.load_state_dict(self.q_net.state_dict())


# ==========================================
# 第四部分：Double DQN 智能体
# ==========================================
class DoubleDQNAgent(DQNAgent):
    """
    Double DQN 智能体（继承自标准 DQN）

    唯一的区别在于目标值的计算方式：
        标准 DQN：target = r + γ * Q_target(s').max()
        Double DQN：target = r + γ * Q_target(s')[argmax_a Q(s')]

    直觉理解：
    - Q 网络负责"提名"最优动作（选动作）
    - 目标网络负责"投票"该动作的价值（评估 Q 值）
    - 两个网络各自独立，过估计的概率大大降低
    """

    def update(self, batch_size):
        """
        Double DQN 更新（核心区别在这里！）

        步骤拆解：
        1. 用 q_net 选出下一状态的最优动作：a* = argmax_a q_net(s')
        2. 用 target_net 评估该动作的 Q 值：Q_target(s', a*)
        3. 计算目标值：target = r + γ * Q_target(s', a*)
        """
        if len(self.buffer) < batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # 当前 Q 值（与标准 DQN 相同）
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # ★ Double DQN 核心：解耦动作选择和价值评估 ★
        with torch.no_grad():
            # 第一步：用 Q 网络选择最优动作
            # q_net 输出所有动作的 Q 值，取 argmax 得到最优动作索引
            best_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)

            # 第二步：用目标网络评估这些动作的 Q 值
            # target_net 根据最优动作索引取出对应的 Q 值
            next_q_values = self.target_net(next_states).gather(1, best_actions).squeeze(1)

            # 第三步：计算目标值
            targets = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10)
        self.optimizer.step()

        return loss.item()


# ==========================================
# 第五部分：通用训练函数
# ==========================================
def train_agent(agent, num_episodes=300, batch_size=64,
                epsilon_start=1.0, epsilon_end=0.01,
                epsilon_decay=0.995, target_update_freq=10):
    """
    通用的训练函数，适用于 DQN 和 Double DQN

    参数：
        agent: DQN 或 Double DQN 智能体
        其余参数为训练超参数

    返回：
        reward_history: 每回合的累计奖励列表
    """
    env = gym.make("CartPole-v1")
    reward_history = []
    epsilon = epsilon_start

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0

        while True:
            action = agent.select_action(state, epsilon)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.buffer.push(state, action, reward, next_state, float(done))
            agent.update(batch_size)

            state = next_state
            episode_reward += reward

            if done or truncated:
                break

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        reward_history.append(episode_reward)

        if (episode + 1) % target_update_freq == 0:
            agent.update_target()

    env.close()
    return reward_history


# ==========================================
# 第六部分：对比实验
# ==========================================
def main():
    """运行 DQN 和 Double DQN 的对比实验"""

    # 训练参数
    NUM_EPISODES = 300
    BATCH_SIZE = 64
    LR = 1e-3
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.995
    TARGET_UPDATE_FREQ = 10

    # 创建环境以获取维度信息
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]   # 4
    action_dim = env.action_space.n               # 2
    env.close()

    print("=" * 60)
    print("  DQN vs Double DQN 对比实验 —— CartPole-v1")
    print("=" * 60)
    print(f"  状态空间维度: {state_dim}")
    print(f"  动作空间维度: {action_dim}")
    print(f"  训练回合数: {NUM_EPISODES}")
    print(f"  批次大小: {BATCH_SIZE}")
    print(f"  学习率: {LR}")
    print(f"  折扣因子: {GAMMA}")
    print("=" * 60)

    # ------------------------------------------
    # 训练标准 DQN
    # ------------------------------------------
    print("\n[1/2] 正在训练标准 DQN...")
    print("-" * 60)

    dqn_agent = DQNAgent(state_dim, action_dim, lr=LR, gamma=GAMMA)
    dqn_rewards = train_agent(
        dqn_agent,
        num_episodes=NUM_EPISODES,
        batch_size=BATCH_SIZE,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
        target_update_freq=TARGET_UPDATE_FREQ,
    )

    dqn_avg = np.mean(dqn_rewards[-50:])
    print(f"  DQN 训练完成！最后 50 回合平均奖励: {dqn_avg:.1f}")

    # ------------------------------------------
    # 训练 Double DQN
    # ------------------------------------------
    print("\n[2/2] 正在训练 Double DQN...")
    print("-" * 60)

    double_dqn_agent = DoubleDQNAgent(state_dim, action_dim, lr=LR, gamma=GAMMA)
    double_dqn_rewards = train_agent(
        double_dqn_agent,
        num_episodes=NUM_EPISODES,
        batch_size=BATCH_SIZE,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
        target_update_freq=TARGET_UPDATE_FREQ,
    )

    ddqn_avg = np.mean(double_dqn_rewards[-50:])
    print(f"  Double DQN 训练完成！最后 50 回合平均奖励: {ddqn_avg:.1f}")

    # ==========================================
    # 第七部分：对比结果可视化
    # ==========================================
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- 子图1：原始奖励曲线 ---
    ax1.plot(dqn_rewards, alpha=0.3, color='steelblue')
    ax1.plot(double_dqn_rewards, alpha=0.3, color='coral')

    # 绘制滑动平均
    window = 20
    dqn_ma = [np.mean(dqn_rewards[max(0, i - window): i + 1])
              for i in range(len(dqn_rewards))]
    ddqn_ma = [np.mean(double_dqn_rewards[max(0, i - window): i + 1])
               for i in range(len(double_dqn_rewards))]

    ax1.plot(dqn_ma, color='steelblue', linewidth=2, label='DQN')
    ax1.plot(ddqn_ma, color='coral', linewidth=2, label='Double DQN')
    ax1.set_xlabel('训练回合')
    ax1.set_ylabel('累计奖励')
    ax1.set_title('DQN vs Double DQN 训练曲线')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- 子图2：滑动平均对比（更清晰） ---
    ax2.plot(dqn_ma, color='steelblue', linewidth=2, label='DQN')
    ax2.plot(ddqn_ma, color='coral', linewidth=2, label='Double DQN')
    ax2.fill_between(
        range(len(dqn_ma)), dqn_ma, ddqn_ma,
        where=[d > dd for d, dd in zip(dqn_ma, ddqn_ma)],
        alpha=0.15, color='steelblue', label='DQN 领先区域'
    )
    ax2.fill_between(
        range(len(dqn_ma)), dqn_ma, ddqn_ma,
        where=[dd >= d for d, dd in zip(dqn_ma, ddqn_ma)],
        alpha=0.15, color='coral', label='Double DQN 领先区域'
    )
    ax2.set_xlabel('训练回合')
    ax2.set_ylabel('累计奖励（滑动平均）')
    ax2.set_title(f'{window} 回合滑动平均对比')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("output/dqn_vs_double_dqn.png", dpi=150)
    print("\n对比图已保存到 output/dqn_vs_double_dqn.png")
    plt.show()

    # ==========================================
    # 第八部分：最终结果汇总
    # ==========================================
    print("\n" + "=" * 60)
    print("  最终结果汇总")
    print("=" * 60)

    print(f"\n  标准 DQN:")
    print(f"    最后 50 回合平均奖励: {dqn_avg:.1f}")
    print(f"    最高回合奖励: {max(dqn_rewards):.0f}")

    print(f"\n  Double DQN:")
    print(f"    最后 50 回合平均奖励: {ddqn_avg:.1f}")
    print(f"    最高回合奖励: {max(double_dqn_rewards):.0f}")

    print(f"\n  差异 (Double DQN - DQN): {ddqn_avg - dqn_avg:+.1f}")

    print("\n" + "-" * 60)
    print("  关键区别回顾：")
    print("  标准 DQN：target = r + γ * Q_target(s').max()")
    print("  Double DQN：target = r + γ * Q_target(s')[Q(s').argmax()]")
    print("  ↑ 解耦动作选择与价值评估，减少过估计")
    print("=" * 60)


# ==========================================
# 入口
# ==========================================
if __name__ == "__main__":
    main()
