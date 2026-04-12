"""
第5章：REINFORCE 策略梯度算法 —— CartPole-v1
从零实现最经典的策略梯度方法，理解"好动作多做，坏动作少做"

算法核心思想：
    策略梯度的直观理解 —— 如果一个回合得分很高，
    那么这个回合里的每个动作都应该被"鼓励"（增大概率）；
    反之则应该被"抑制"（降低概率）。

REINFORCE 公式：
    ∇J(θ) ≈ Σ_t [∇log π(a_t|s_t)] * G_t
    其中 G_t 是从时间步 t 开始的折扣累计回报

运行方式：
    python reinforce_cartpole.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque

# 创建输出目录
os.makedirs("output", exist_ok=True)

# 设置中文字体，确保图表标题和标签正常显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 第一部分：策略网络（Policy Network）
# ==========================================
class PolicyNetwork(nn.Module):
    """
    策略网络：将状态映射为动作概率分布

    结构：4 (状态维度) → 128 → 128 → 2 (动作维度)
    输出经过 Softmax 归一化，得到合法的概率分布

    CartPole 的状态空间：[小车位置, 小车速度, 杆子角度, 杆子角速度]
    CartPole 的动作空间：[向左推, 向右推]
    """

    def __init__(self, state_dim=4, action_dim=2, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),   # 输入层 → 第一个隐藏层
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),   # 第一个隐藏层 → 第二个隐藏层
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),   # 第二个隐藏层 → 输出层（logits）
        )

    def forward(self, x):
        """
        前向传播：状态 → 动作概率

        参数：
            x: 状态张量，形状 [batch_size, state_dim]
        返回：
            probs: 动作概率，形状 [batch_size, action_dim]，已经过 Softmax
        """
        logits = self.network(x)
        probs = torch.softmax(logits, dim=-1)
        return probs


# ==========================================
# 第二部分：计算折扣累计回报（Returns）
# ==========================================
def compute_returns(rewards, gamma=0.99):
    """
    从后向前计算每一步的折扣累计回报 G_t

    公式：G_t = r_t + γ * r_{t+1} + γ² * r_{t+2} + ...

    示例（gamma=0.99）：
        rewards = [1, 1, 1, 1, 1]
        G_0 = 1 + 0.99*1 + 0.99²*1 + ... ≈ 4.90
        G_4 = 1

    参数：
        rewards: 每一步的即时奖励列表
        gamma: 折扣因子，越接近1越重视未来奖励
    返回：
        returns: 每一步的折扣累计回报列表
    """
    returns = []
    G = 0  # 累计回报

    # 从后向前遍历：利用 G_t = r_t + gamma * G_{t+1} 的递推关系
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)  # 在列表头部插入，保持时间顺序

    return returns


# ==========================================
# 第三部分：收集完整回合轨迹
# ==========================================
def collect_episode(policy, env):
    """
    让策略网络在环境中完成一个完整回合，收集轨迹数据

    REINFORCE 是 on-policy 算法，必须用当前策略收集数据，
    用完即丢弃，下一轮需要重新收集。

    参数：
        policy: 策略网络
        env: Gymnasium 环境
    返回：
        states: 状态列表
        actions: 动作列表
        rewards: 奖励列表
        episode_reward: 回合总奖励
    """
    state, _ = env.reset()
    states, actions, rewards = [], [], []

    done = False
    truncated = False

    while not (done or truncated):
        # 将状态转为张量
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # 添加 batch 维度

        # 获取动作概率分布
        with torch.no_grad():
            probs = policy(state_tensor)

        # 按概率分布采样动作（探索的关键！不是取 argmax）
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()

        # 执行动作，观察结果
        next_state, reward, done, truncated, _ = env.step(action)

        # 存储转移数据
        states.append(state)
        actions.append(action)
        rewards.append(reward)

        state = next_state

    episode_reward = sum(rewards)
    return states, actions, rewards, episode_reward


# ==========================================
# 第四部分：训练一个回合（REINFORCE 核心更新）
# ==========================================
def train_one_episode(policy, optimizer, states, actions, returns):
    """
    REINFORCE 的核心：用策略梯度公式更新网络参数

    损失函数 = - Σ_t [log π(a_t|s_t) * G_t]

    这个损失函数的梯度恰好等于策略梯度：
    ∇loss = - Σ_t [∇log π(a_t|s_t) * G_t] = -∇J(θ)

    所以 minimize loss = maximize J(θ)（期望回报）

    参数：
        policy: 策略网络
        optimizer: 优化器
        states: 状态列表
        actions: 动作列表
        returns: 折扣累计回报列表
    返回：
        loss_value: 本轮损失值
    """
    # 将数据转为张量
    states_tensor = torch.FloatTensor(np.array(states))
    actions_tensor = torch.LongTensor(actions)
    returns_tensor = torch.FloatTensor(returns)

    # 前向传播：获取每个状态下的动作概率
    probs = policy(states_tensor)

    # 计算所采取动作的对数概率 log π(a_t|s_t)
    # gather(1, actions) 选取每个状态对应动作的概率
    action_probs = probs.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
    log_probs = torch.log(action_probs + 1e-8)  # 加小常数防止 log(0)

    # 策略梯度损失：-log π(a_t|s_t) * G_t
    # 直觉理解：
    #   如果 G_t > 0（好结果），-log_prob * G_t < 0，梯度下降会增大 log_prob → 增大概率
    #   如果 G_t < 0（坏结果），-log_prob * G_t > 0，梯度下降会减小 log_prob → 降低概率
    loss = -(log_probs * returns_tensor).mean()

    # 反向传播 + 参数更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


# ==========================================
# 第五部分：主训练循环
# ==========================================
def train():
    """
    REINFORCE 完整训练流程

    超参数说明：
        - num_episodes = 500：训练 500 个回合
        - gamma = 0.99：折扣因子，重视长期回报
        - learning_rate = 1e-3：学习率
        - hidden_dim = 128：隐藏层宽度
    """
    # ---------- 超参数 ----------
    num_episodes = 500
    gamma = 0.99
    learning_rate = 1e-3
    hidden_dim = 128

    # ---------- 初始化 ----------
    env = gym.make("CartPole-v1")
    policy = PolicyNetwork(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        hidden_dim=hidden_dim,
    )
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    # 记录训练过程
    episode_rewards = []  # 每个回合的总奖励
    episode_losses = []   # 每个回合的损失

    print("=" * 60)
    print("  REINFORCE 策略梯度 —— CartPole-v1 训练")
    print("=" * 60)
    print(f"  超参数:")
    print(f"    回合数: {num_episodes}")
    print(f"    折扣因子 γ: {gamma}")
    print(f"    学习率: {learning_rate}")
    print(f"    隐藏层维度: {hidden_dim}")
    print("=" * 60)

    # ---------- 训练循环 ----------
    for episode in range(num_episodes):
        # 第一步：用当前策略收集一个完整回合的轨迹
        states, actions, rewards, episode_reward = collect_episode(policy, env)

        # 第二步：计算折扣累计回报
        returns = compute_returns(rewards, gamma=gamma)

        # 第三步：执行策略梯度更新
        loss_value = train_one_episode(policy, optimizer, states, actions, returns)

        # 记录数据
        episode_rewards.append(episode_reward)
        episode_losses.append(loss_value)

        # 每 50 个回合打印一次进度
        if (episode + 1) % 50 == 0:
            recent_rewards = episode_rewards[-50:]
            avg_reward = np.mean(recent_rewards)
            print(
                f"  回合 {episode + 1:4d}/{num_episodes} | "
                f"本轮奖励: {episode_reward:6.1f} | "
                f"近 50 回合均值: {avg_reward:6.1f} | "
                f"损失: {loss_value:.4f}"
            )

    env.close()

    # ---------- 训练结果汇总 ----------
    print("=" * 60)
    print("  训练完成！")
    print(f"  最后 50 回合平均奖励: {np.mean(episode_rewards[-50:]):.1f}")
    print(f"  最佳回合奖励: {np.max(episode_rewards):.1f}")
    print("=" * 60)

    # ---------- 绘制训练曲线 ----------
    plot_training_curve(episode_rewards)


# ==========================================
# 第六部分：绘制训练曲线
# ==========================================
def plot_training_curve(episode_rewards):
    """
    绘制奖励曲线和滑动平均线

    滑动平均（window=50）可以更清晰地展示学习趋势，
    过滤掉单回合的随机波动。
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # 原始奖励曲线（浅色，展示波动）
    ax.plot(episode_rewards, alpha=0.3, color='steelblue', label='回合奖励（原始）')

    # 滑动平均曲线（深色，展示趋势）
    window = 50
    if len(episode_rewards) >= window:
        moving_avg = []
        for i in range(len(episode_rewards)):
            start = max(0, i - window + 1)
            moving_avg.append(np.mean(episode_rewards[start:i + 1]))
        ax.plot(moving_avg, color='crimson', linewidth=2.0,
                label=f'滑动平均（窗口={window}）')

    ax.set_xlabel('训练回合', fontsize=12)
    ax.set_ylabel('回合奖励', fontsize=12)
    ax.set_title('REINFORCE 策略梯度 —— CartPole-v1 训练曲线', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/reinforce_cartpole_rewards.png', dpi=150, bbox_inches='tight')
    print("  训练曲线已保存为 output/reinforce_cartpole_rewards.png")
    plt.show()


# ==========================================
# 程序入口
# ==========================================
if __name__ == "__main__":
    train()
