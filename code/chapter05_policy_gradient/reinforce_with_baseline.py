"""
第5章：REINFORCE with Baseline —— 方差缩减对比实验
对比原始 REINFORCE 与加入基线（Value Network）的版本

核心问题：原始 REINFORCE 的梯度方差很大，训练不稳定
解决方案：用优势函数代替原始回报
    优势 = G_t - V(s_t)
    其中 V(s_t) 是一个价值网络对状态的估计值

为什么基线能降低方差？
    - G_t 的绝对值可能很大（比如 200），但不同时间步的差异较小
    - 减去 V(s) 后，优势值围绕 0 波动，幅度小得多
    - 数学上 E[G_t - b] = E[G_t]（只要 b 不依赖动作），期望不变，方差降低

运行方式：
    python reinforce_with_baseline.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# 创建输出目录
os.makedirs("output", exist_ok=True)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 第一部分：网络结构定义
# ==========================================
class PolicyNetwork(nn.Module):
    """
    策略网络（Actor）：状态 → 动作概率

    结构：4 → 128 → 128 → 2（Softmax 输出）
    """

    def __init__(self, state_dim=4, action_dim=2, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        logits = self.network(x)
        probs = torch.softmax(logits, dim=-1)
        return probs


class ValueNetwork(nn.Module):
    """
    价值网络（Baseline/Critic）：状态 → 价值估计

    结构：4 → 128 → 128 → 1（标量输出）
    用于估计 V(s)，即从状态 s 出发的期望累计回报

    这个网络就是"基线"：通过减去 V(s)，我们得到优势函数 A(s)
    """

    def __init__(self, state_dim=4, hidden_dim=128):
        super(ValueNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # 输出单个标量值
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)  # 去掉最后一维，变为 [batch_size]


# ==========================================
# 第二部分：计算折扣累计回报
# ==========================================
def compute_returns(rewards, gamma=0.99):
    """
    计算折扣累计回报 G_t = r_t + γ * r_{t+1} + γ² * r_{t+2} + ...

    参数：
        rewards: 即时奖励列表
        gamma: 折扣因子
    返回：
        returns: 折扣累计回报列表
    """
    returns = []
    G = 0
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)
    return returns


# ==========================================
# 第三部分：收集回合轨迹
# ==========================================
def collect_episode(policy, env):
    """
    用当前策略收集一个完整回合的数据

    参数：
        policy: 策略网络
        env: 环境
    返回：
        states, actions, rewards, episode_reward
    """
    state, _ = env.reset()
    states, actions, rewards = [], [], []
    done, truncated = False, False

    while not (done or truncated):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            probs = policy(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()

        next_state, reward, done, truncated, _ = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state

    episode_reward = sum(rewards)
    return states, actions, rewards, episode_reward


# ==========================================
# 第四部分：原始 REINFORCE 训练
# ==========================================
def train_vanilla_reinforce(num_episodes=500, gamma=0.99, lr=1e-3):
    """
    原始 REINFORCE（无基线）

    损失 = -Σ log π(a_t|s_t) * G_t
    直接用折扣累计回报 G_t 作为权重
    """
    env = gym.make("CartPole-v1")
    policy = PolicyNetwork(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
    )
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    episode_rewards = []
    gradient_estimates = []  # 记录梯度估计值，用于衡量方差

    for episode in range(num_episodes):
        # 收集轨迹
        states, actions, rewards, episode_reward = collect_episode(policy, env)

        # 计算回报
        returns = compute_returns(rewards, gamma)

        # 转为张量
        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.LongTensor(actions)
        returns_t = torch.FloatTensor(returns)

        # 前向传播
        probs = policy(states_t)
        action_probs = probs.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        log_probs = torch.log(action_probs + 1e-8)

        # 策略梯度损失
        loss = -(log_probs * returns_t).mean()

        # 记录梯度估计值（用于后续计算方差）
        with torch.no_grad():
            grad_estimate = (log_probs * returns_t).mean().item()
            gradient_estimates.append(grad_estimate)

        # 更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_rewards.append(episode_reward)

        if (episode + 1) % 100 == 0:
            avg = np.mean(episode_rewards[-50:])
            print(f"  [Vanilla] 回合 {episode+1:4d} | 近50均值: {avg:6.1f}")

    env.close()
    return episode_rewards, gradient_estimates


# ==========================================
# 第五部分：REINFORCE with Baseline 训练
# ==========================================
def train_reinforce_with_baseline(num_episodes=500, gamma=0.99, lr=1e-3):
    """
    REINFORCE + 价值基线

    优势函数：A(s,a) = G_t - V(s_t)
    策略损失：-Σ log π(a_t|s_t) * A(s_t, a_t)
    价值损失：MSE(V(s_t), G_t)

    两个网络同时训练：
        - 策略网络学习"什么动作更好"（相对于基线）
        - 价值网络学习"当前状态平均能拿多少分"（基线）
    """
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 初始化策略网络和价值网络
    policy = PolicyNetwork(state_dim=state_dim, action_dim=action_dim)
    value_net = ValueNetwork(state_dim=state_dim)

    # 两个网络各用独立的优化器
    policy_optimizer = optim.Adam(policy.parameters(), lr=lr)
    value_optimizer = optim.Adam(value_net.parameters(), lr=lr)

    episode_rewards = []
    gradient_estimates = []  # 记录梯度估计值

    for episode in range(num_episodes):
        # 收集轨迹
        states, actions, rewards, episode_reward = collect_episode(policy, env)

        # 计算回报
        returns = compute_returns(rewards, gamma)

        # 转为张量
        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.LongTensor(actions)
        returns_t = torch.FloatTensor(returns)

        # ========== 更新价值网络（Critic） ==========
        # 价值网络的目标：准确预测 V(s) ≈ G_t
        values = value_net(states_t)
        value_loss = nn.MSELoss()(values, returns_t)

        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        # ========== 计算优势函数 ==========
        # 优势 = 实际回报 - 基线预测
        # A > 0 表示"比预期好" → 增大对应动作概率
        # A < 0 表示"比预期差" → 减小对应动作概率
        with torch.no_grad():
            values_pred = value_net(states_t)
        advantages = returns_t - values_pred

        # ========== 更新策略网络（Actor） ==========
        probs = policy(states_t)
        action_probs = probs.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        log_probs = torch.log(action_probs + 1e-8)

        # 策略梯度损失：用优势函数替代原始回报
        policy_loss = -(log_probs * advantages).mean()

        # 记录梯度估计值（用优势替代回报）
        with torch.no_grad():
            grad_estimate = (log_probs * advantages).mean().item()
            gradient_estimates.append(grad_estimate)

        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        episode_rewards.append(episode_reward)

        if (episode + 1) % 100 == 0:
            avg = np.mean(episode_rewards[-50:])
            print(f"  [Baseline] 回合 {episode+1:4d} | 近50均值: {avg:6.1f}")

    env.close()
    return episode_rewards, gradient_estimates


# ==========================================
# 第六部分：对比实验主函数
# ==========================================
def run_comparison():
    """
    运行对比实验：Vanilla REINFORCE vs REINFORCE + Baseline

    对比两个维度：
        1. 学习速度和最终性能（奖励曲线）
        2. 梯度估计的方差（方差越低，训练越稳定）
    """
    num_episodes = 500
    gamma = 0.99
    lr = 1e-3

    print("=" * 60)
    print("  REINFORCE 方差缩减对比实验")
    print("=" * 60)
    print(f"  训练回合数: {num_episodes}")
    print(f"  折扣因子 γ: {gamma}")
    print(f"  学习率: {lr}")
    print("=" * 60)

    # ---------- 实验1：原始 REINFORCE ----------
    print("\n[实验1] 训练 Vanilla REINFORCE（无基线）...")
    vanilla_rewards, vanilla_grads = train_vanilla_reinforce(
        num_episodes=num_episodes, gamma=gamma, lr=lr
    )

    # ---------- 实验2：REINFORCE + Baseline ----------
    print("\n[实验2] 训练 REINFORCE + Baseline（价值基线）...")
    baseline_rewards, baseline_grads = train_reinforce_with_baseline(
        num_episodes=num_episodes, gamma=gamma, lr=lr
    )

    # ---------- 方差统计对比 ----------
    print("\n" + "=" * 60)
    print("  方差对比统计")
    print("=" * 60)

    vanilla_grad_var = np.var(vanilla_grads)
    baseline_grad_var = np.var(baseline_grads)

    print(f"  Vanilla REINFORCE 梯度估计方差: {vanilla_grad_var:.6f}")
    print(f"  REINFORCE+Baseline 梯度估计方差: {baseline_grad_var:.6f}")

    if vanilla_grad_var > 0:
        ratio = vanilla_grad_var / max(baseline_grad_var, 1e-10)
        print(f"  方差比（Vanilla/Baseline）: {ratio:.2f}x")
        print(f"  基线将方差降低至原来的 {1/ratio*100:.1f}%")

    print(f"\n  Vanilla REINFORCE 最后50回合均值: {np.mean(vanilla_rewards[-50:]):.1f}")
    print(f"  REINFORCE+Baseline 最后50回合均值: {np.mean(baseline_rewards[-50:]):.1f}")
    print("=" * 60)

    # ---------- 绘制对比图1：奖励曲线 ----------
    plot_reward_comparison(vanilla_rewards, baseline_rewards, window=50)

    # ---------- 绘制对比图2：方差对比 ----------
    plot_variance_comparison(vanilla_grads, baseline_grads, window=50)


# ==========================================
# 第七部分：绘制奖励对比曲线
# ==========================================
def plot_reward_comparison(vanilla_rewards, baseline_rewards, window=50):
    """
    绘制两组实验的奖励曲线对比

    包含原始曲线和滑动平均曲线，直观展示：
        - Baseline 版本是否收敛更快
        - Baseline 版本是否更稳定（波动更小）
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Vanilla REINFORCE
    ax.plot(vanilla_rewards, alpha=0.2, color='steelblue')
    vanilla_avg = [np.mean(vanilla_rewards[max(0, i-window+1):i+1])
                   for i in range(len(vanilla_rewards))]
    ax.plot(vanilla_avg, color='steelblue', linewidth=2.0,
            label='Vanilla REINFORCE')

    # REINFORCE + Baseline
    ax.plot(baseline_rewards, alpha=0.2, color='crimson')
    baseline_avg = [np.mean(baseline_rewards[max(0, i-window+1):i+1])
                    for i in range(len(baseline_rewards))]
    ax.plot(baseline_avg, color='crimson', linewidth=2.0,
            label='REINFORCE + Baseline')

    ax.set_xlabel('训练回合', fontsize=12)
    ax.set_ylabel('回合奖励', fontsize=12)
    ax.set_title('REINFORCE 奖励曲线对比（Vanilla vs Baseline）', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/reinforce_baseline_reward_comparison.png', dpi=150, bbox_inches='tight')
    print("  奖励对比图已保存为 output/reinforce_baseline_reward_comparison.png")
    plt.show()


# ==========================================
# 第八部分：绘制方差对比图
# ==========================================
def plot_variance_comparison(vanilla_grads, baseline_grads, window=50):
    """
    绘制梯度估计方差的滑动窗口对比

    这张图是本实验的核心：展示基线如何降低策略梯度的方差。
    方差越低，训练过程越稳定，收敛越可靠。
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # 计算滑动窗口方差
    def moving_variance(data, w):
        variances = []
        for i in range(len(data)):
            start = max(0, i - w + 1)
            variances.append(np.var(data[start:i + 1]))
        return variances

    vanilla_var = moving_variance(vanilla_grads, window)
    baseline_var = moving_variance(baseline_grads, window)

    ax.plot(vanilla_var, color='steelblue', linewidth=1.5, alpha=0.8,
            label='Vanilla REINFORCE')
    ax.plot(baseline_var, color='crimson', linewidth=1.5, alpha=0.8,
            label='REINFORCE + Baseline')

    ax.set_xlabel('训练回合', fontsize=12)
    ax.set_ylabel(f'梯度估计方差（窗口={window}）', fontsize=12)
    ax.set_title('策略梯度方差对比 —— 基线的方差缩减效果', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # 添加注释箭头，标明方差差异
    if len(vanilla_var) > 100:
        mid_point = len(vanilla_var) // 2
        ax.annotate(
            '基线显著降低方差',
            xy=(mid_point, baseline_var[mid_point]),
            xytext=(mid_point + 50, max(vanilla_var) * 0.7),
            fontsize=11,
            arrowprops=dict(arrowstyle='->', color='gray'),
            color='gray',
        )

    plt.tight_layout()
    plt.savefig('output/reinforce_baseline_variance_comparison.png', dpi=150, bbox_inches='tight')
    print("  方差对比图已保存为 output/reinforce_baseline_variance_comparison.png")
    plt.show()


# ==========================================
# 程序入口
# ==========================================
if __name__ == "__main__":
    run_comparison()
