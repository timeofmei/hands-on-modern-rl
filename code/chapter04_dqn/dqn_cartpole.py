"""
第4章：深度Q网络 (Deep Q-Network, DQN) —— 从零实现
在 CartPole-v1 环境上训练一个完整的 DQN 智能体

核心思想：
    用神经网络近似 Q(s, a)，通过经验回放和目标网络来稳定训练。
    这是 2015 年 DeepMind 发表的里程碑工作，开启了深度强化学习的时代。

关键组件：
    1. Q 网络：用多层感知机 (MLP) 近似动作价值函数
    2. 经验回放缓冲区：打破数据相关性，提高样本效率
    3. 目标网络：延迟更新，提供稳定的训练目标
    4. ε-贪心探索：在探索与利用之间取得平衡

运行方式：
    python dqn_cartpole.py
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
# 第一部分：Q 网络
# ==========================================
class QNetwork(nn.Module):
    """
    Q 网络：将状态映射到每个动作的 Q 值
    结构：state_dim → 128 → 128 → action_dim

    输入是一个状态向量（CartPole 中是 4 维），
    输出是每个动作的 Q 值估计（CartPole 中是 2 维，对应左/右）。
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),   # 第一层：状态 → 128
            nn.ReLU(),
            nn.Linear(128, 128),          # 第二层：128 → 128
            nn.ReLU(),
            nn.Linear(128, action_dim),   # 输出层：128 → 动作数
        )

    def forward(self, x):
        """前向传播：输入状态，输出各动作的 Q 值"""
        return self.net(x)


# ==========================================
# 第二部分：经验回放缓冲区
# ==========================================
class ReplayBuffer:
    """
    经验回放缓冲区：存储和采样训练数据

    为什么需要经验回放？
    - 深度学习要求数据独立同分布 (i.i.d.)
    - 但强化学习中，连续的转移 (s, a, r, s') 是高度相关的
    - 经验回放通过随机打乱采样顺序来打破这种相关性
    - 同时也提高了数据利用率（一条经验可以被多次使用）
    """

    def __init__(self, capacity=10000):
        """用 deque 实现，容量满后自动丢弃旧数据"""
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """存储一条转移经验 (s, a, r, s', done)"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """随机采样一批数据，打破时间相关性"""
        batch = random.sample(self.buffer, batch_size)
        # 解包并转换为 numpy 数组，方便后续转为 tensor
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
# 第三部分：DQN 智能体
# ==========================================
class DQNAgent:
    """
    DQN 智能体：整合 Q 网络、目标网络、经验回放和 ε-贪心策略

    DQN 的三大创新：
    1. 经验回放 (Experience Replay)：打破数据相关性
    2. 目标网络 (Target Network)：提供稳定的训练目标
    3. ε-贪心 (Epsilon-Greedy)：平衡探索与利用
    """

    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.action_dim = action_dim
        self.gamma = gamma  # 折扣因子：未来奖励的衰减系数

        # Q 网络：实时更新的主网络
        self.q_net = QNetwork(state_dim, action_dim)
        # 目标网络：定期从 Q 网络复制权重，用于计算稳定的目标值
        self.target_net = QNetwork(state_dim, action_dim)
        # 初始化时将目标网络与 Q 网络同步
        self.target_net.load_state_dict(self.q_net.state_dict())
        # 目标网络不需要梯度
        self.target_net.eval()

        # 优化器：只优化 Q 网络的参数
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # 经验回放缓冲区
        self.buffer = ReplayBuffer(capacity=10000)

    def select_action(self, state, epsilon):
        """
        ε-贪心动作选择

        以概率 ε 随机选择动作（探索），
        以概率 1-ε 选择 Q 值最大的动作（利用）。
        """
        if random.random() < epsilon:
            # 探索：随机选一个动作
            return random.randint(0, self.action_dim - 1)
        else:
            # 利用：选 Q 值最大的动作
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # 添加 batch 维度
            with torch.no_grad():
                q_values = self.q_net(state_tensor)
            return q_values.argmax(dim=1).item()

    def update(self, batch_size):
        """
        从经验回放中采样并更新 Q 网络

        核心 DQN 更新公式：
            target = r + γ * max_a' Q_target(s', a')
            loss = (target - Q(s, a))²

        注意：目标值用 target_net 计算，Q 值用 q_net 计算，
        这样避免了"自己追自己"的不稳定问题。
        """
        if len(self.buffer) < batch_size:
            return 0.0  # 数据不够时不更新

        # 从经验回放中随机采样
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)

        # 转为 PyTorch 张量
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # 计算 Q(s, a)：当前网络的估计值
        # gather 根据 actions 选出对应动作的 Q 值
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # 计算目标值：r + γ * max_a' Q_target(s', a')
        with torch.no_grad():
            # 用目标网络计算下一状态的最大 Q 值
            next_q_max = self.target_net(next_states).max(dim=1)[0]
            # done = 1 时，没有未来奖励
            targets = rewards + self.gamma * next_q_max * (1 - dones)

        # 均方误差损失
        loss = nn.MSELoss()(q_values, targets)

        # 梯度下降
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪：防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10)
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        """将 Q 网络的权重复制到目标网络（硬更新）"""
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path):
        """保存模型"""
        torch.save(self.q_net.state_dict(), path)
        print(f"模型已保存到 {path}")


# ==========================================
# 第四部分：训练循环
# ==========================================
def train():
    """完整的 DQN 训练流程"""

    # 超参数设置
    NUM_EPISODES = 500       # 训练回合数
    BATCH_SIZE = 64          # 每次更新的批次大小
    EPSILON_START = 1.0      # 初始探索率
    EPSILON_END = 0.01       # 最终探索率
    EPSILON_DECAY = 0.995    # 探索率衰减系数
    TARGET_UPDATE_FREQ = 10  # 目标网络更新频率（每 N 个回合）

    # 创建环境和智能体
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]   # 4
    action_dim = env.action_space.n               # 2
    agent = DQNAgent(state_dim, action_dim, lr=1e-3, gamma=0.99)

    print("=" * 60)
    print("  深度Q网络 (DQN) —— CartPole-v1 训练")
    print("=" * 60)
    print(f"  状态空间维度: {state_dim}")
    print(f"  动作空间维度: {action_dim}")
    print(f"  训练回合数: {NUM_EPISODES}")
    print(f"  批次大小: {BATCH_SIZE}")
    print(f"  初始探索率: {EPSILON_START}")
    print(f"  目标网络更新频率: 每 {TARGET_UPDATE_FREQ} 回合")
    print("=" * 60)

    # 记录训练数据
    reward_history = []
    epsilon = EPSILON_START

    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        episode_reward = 0

        while True:
            # 选择动作（ε-贪心策略）
            action = agent.select_action(state, epsilon)
            # 执行动作
            next_state, reward, done, truncated, _ = env.step(action)
            # 存入经验回放
            agent.buffer.push(state, action, reward, next_state, float(done))
            # 更新 Q 网络
            agent.update(BATCH_SIZE)

            state = next_state
            episode_reward += reward

            if done or truncated:
                break

        # 衰减探索率
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        reward_history.append(episode_reward)

        # 定期更新目标网络
        if (episode + 1) % TARGET_UPDATE_FREQ == 0:
            agent.update_target()

        # 每 50 个回合打印进度
        if (episode + 1) % 50 == 0:
            recent = reward_history[-50:]
            avg_reward = np.mean(recent)
            print(
                f"  回合 {episode + 1:4d}/{NUM_EPISODES} | "
                f"平均奖励(近50): {avg_reward:6.1f} | "
                f"ε: {epsilon:.3f}"
            )

    env.close()

    # ==========================================
    # 第五部分：训练结果可视化
    # ==========================================
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(reward_history, alpha=0.3, color='steelblue', label='每回合奖励')

    # 绘制滑动平均曲线，更清晰地展示趋势
    window = 20
    if len(reward_history) >= window:
        moving_avg = [
            np.mean(reward_history[max(0, i - window): i + 1])
            for i in range(len(reward_history))
        ]
        ax.plot(moving_avg, color='red', linewidth=2,
                label=f'{window} 回合滑动平均')

    ax.set_xlabel('训练回合')
    ax.set_ylabel('累计奖励')
    ax.set_title('DQN 训练曲线 —— CartPole-v1')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/dqn_cartpole_training.png", dpi=150)
    print("\n训练曲线已保存到 output/dqn_cartpole_training.png")
    plt.show()

    # ==========================================
    # 第六部分：测试训练好的智能体
    # ==========================================
    print("\n" + "=" * 60)
    print("  测试阶段：运行 10 个回合")
    print("=" * 60)

    test_env = gym.make("CartPole-v1")
    test_rewards = []

    for ep in range(10):
        state, _ = test_env.reset()
        total_reward = 0

        while True:
            # 测试时不探索，始终选最优动作
            action = agent.select_action(state, epsilon=0.0)
            state, reward, done, truncated, _ = test_env.step(action)
            total_reward += reward

            if done or truncated:
                break

        test_rewards.append(total_reward)
        print(f"  测试回合 {ep + 1:2d}: 得分 = {total_reward:.0f}")

    print("-" * 60)
    print(f"  测试平均得分: {np.mean(test_rewards):.1f} ± {np.std(test_rewards):.1f}")
    print("=" * 60)

    test_env.close()

    # 保存模型
    agent.save("output/dqn_cartpole.pth")

    return reward_history


# ==========================================
# 入口
# ==========================================
if __name__ == "__main__":
    train()
