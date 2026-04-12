"""
第5章：Actor-Critic 算法 —— CartPole-v1
单步更新的策略梯度方法，比 REINFORCE 更高效

REINFORCE 的问题：
    - 必须等一个完整回合结束才能更新（Monte Carlo 方法）
    - 如果回合很长，数据利用效率低
    - 回合结束后的高方差 G_t 用来更新前面所有步骤

Actor-Critic 的改进：
    - 用 TD(0) 估计替代完整回合回报
    - advantage = r + γ * V(s') - V(s)
    - 每一步都可以立即更新，不需要等回合结束
    - 因为用 V(s') 自举（bootstrap），方差更低

网络结构：
    共享骨干层的 Actor-Critic 网络
    - 共享层：提取状态特征（复用参数，减少计算量）
    - Actor 头：输出动作概率（策略）
    - Critic 头：输出状态价值（价值函数）

运行方式：
    python actor_critic_cartpole.py
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
# 第一部分：Actor-Critic 网络结构
# ==========================================
class ActorCritic(nn.Module):
    """
    Actor-Critic 网络：共享骨干，两个输出头

    架构图：
        输入 state (维度=4)
            │
        ┌───────┐
        │ Linear│ 4 → 128
        │  ReLU │
        └───────┘
            │
        ┌───────┐
        │  Actor │ 128 → 2 → Softmax  （策略：选哪个动作）
        └───────┘
            │
        ┌───────┐
        │ Critic │ 128 → 1            （价值：当前状态值多少）
        └───────┘

    共享骨干的好处：
        - 状态特征只需要计算一次
        - Actor 和 Critic 可以共享底层表示
        - 参数更少，训练更快
    """

    def __init__(self, state_dim=4, action_dim=2, hidden_dim=128):
        super(ActorCritic, self).__init__()

        # 共享骨干层：提取状态的特征表示
        self.shared_backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor 头：输出动作概率分布
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
        )

        # Critic 头：输出状态价值（标量）
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        """
        前向传播，同时输出动作概率和状态价值

        参数：
            x: 状态张量 [batch_size, state_dim]
        返回：
            probs: 动作概率 [batch_size, action_dim]
            value: 状态价值 [batch_size]
        """
        # 共享层提取特征
        features = self.shared_backbone(x)

        # Actor 输出动作概率
        action_logits = self.actor_head(features)
        probs = torch.softmax(action_logits, dim=-1)

        # Critic 输出状态价值
        value = self.critic_head(features).squeeze(-1)

        return probs, value


# ==========================================
# 第二部分：计算 TD 误差和优势
# ==========================================
def compute_advantage(reward, value, next_value, gamma=0.99, done=False):
    """
    计算 TD(0) 优势函数

    TD 优势 = r + γ * V(s') - V(s)

    直觉理解：
        - V(s) 是 Critic 对当前状态的"预测分数"
        - r + γ * V(s') 是"实际获得的奖励 + 对未来的新预测"
        - 两者之差就是"预测误差"：比预期好（正）还是差（负）

    与 REINFORCE 的区别：
        REINFORCE: advantage = G_t（完整回合的累计回报）
        Actor-Critic: advantage = r + γ * V(s') - V(s)（单步 TD 误差）

    参数：
        reward: 即时奖励 r_t
        value: 当前状态价值 V(s_t)
        next_value: 下一状态价值 V(s_{t+1})
        gamma: 折扣因子
        done: 回合是否结束
    返回：
        advantage: TD 优势值
    """
    if done:
        # 回合结束时，没有下一个状态，目标 = r_t
        target = reward
    else:
        # TD 目标：r_t + γ * V(s_{t+1})
        target = reward + gamma * next_value

    advantage = target - value
    return advantage, target


# ==========================================
# 第三部分：主训练循环
# ==========================================
def train():
    """
    Actor-Critic 完整训练流程

    核心区别（与 REINFORCE 对比）：
        REINFORCE：收集完整回合 → 计算所有 G_t → 一次反向传播
        Actor-Critic：每一步都计算 TD 误差 → 立即更新网络

    这意味着 Actor-Critic 可以在线学习（online learning），
    不需要等待回合结束，数据利用效率更高。
    """
    # ---------- 超参数 ----------
    num_episodes = 500
    gamma = 0.99
    learning_rate = 1e-3
    hidden_dim = 128

    # ---------- 初始化 ----------
    env = gym.make("CartPole-v1")
    model = ActorCritic(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        hidden_dim=hidden_dim,
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 记录训练数据
    episode_rewards = []
    episode_actor_losses = []
    episode_critic_losses = []

    print("=" * 60)
    print("  Actor-Critic —— CartPole-v1 训练")
    print("=" * 60)
    print(f"  超参数:")
    print(f"    回合数: {num_episodes}")
    print(f"    折扣因子 γ: {gamma}")
    print(f"    学习率: {learning_rate}")
    print(f"    隐藏层维度: {hidden_dim}")
    print("=" * 60)

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        total_actor_loss = 0
        total_critic_loss = 0
        steps = 0

        done = False
        truncated = False

        while not (done or truncated):
            # ========== 第一步：观察当前状态 ==========
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            # 前向传播：同时获取动作概率和状态价值
            probs, value = model(state_tensor)
            probs = probs.squeeze(0)     # [action_dim]
            value = value.squeeze()       # 标量

            # ========== 第二步：选择动作（按概率采样） ==========
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            # 保存 log π(a|s)，用于后续计算策略梯度
            log_prob = dist.log_prob(action)

            # ========== 第三步：执行动作，观察转移 ==========
            next_state, reward, done, truncated, _ = env.step(action.item())
            episode_reward += reward
            steps += 1

            # ========== 第四步：计算下一状态的价值 ==========
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            with torch.no_grad():
                _, next_value = model(next_state_tensor)
                next_value = next_value.squeeze()

            # ========== 第五步：计算 TD 优势和损失 ==========
            # TD 优势：A(s,a) = r + γ * V(s') - V(s)
            is_done = done or truncated
            advantage, target = compute_advantage(
                reward, value, next_value, gamma, done=is_done
            )

            # Actor 损失：-log π(a|s) * A(s,a)
            # 与 REINFORCE 形式相同，但 advantage 是单步 TD 估计
            actor_loss = -log_prob * advantage

            # Critic 损失：让 V(s) 逼近 TD 目标 r + γ * V(s')
            critic_loss = nn.MSELoss()(value, target.detach())

            # 合并损失（可以加权，这里等权）
            total_loss = actor_loss + critic_loss

            # ========== 第六步：立即更新网络 ==========
            # 注意：REINFORCE 是回合结束后才更新，这里是每一步都更新！
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_losses_save = critic_loss.item()
            total_critic_loss += total_critic_losses_save

            # 移到下一个状态
            state = next_state

        # 记录本回合数据
        episode_rewards.append(episode_reward)
        episode_actor_losses.append(total_actor_loss / max(steps, 1))
        episode_critic_losses.append(total_critic_loss / max(steps, 1))

        # 每 50 回合打印进度
        if (episode + 1) % 50 == 0:
            recent_avg = np.mean(episode_rewards[-50:])
            print(
                f"  回合 {episode + 1:4d}/{num_episodes} | "
                f"本轮奖励: {episode_reward:6.1f} | "
                f"近50均值: {recent_avg:6.1f} | "
                f"步数: {steps:3d}"
            )

    env.close()

    # ---------- 训练结果汇总 ----------
    print("=" * 60)
    print("  训练完成！")
    print(f"  最后 50 回合平均奖励: {np.mean(episode_rewards[-50:]):.1f}")
    print(f"  最佳回合奖励: {np.max(episode_rewards):.1f}")
    print("=" * 60)

    # ---------- 与 REINFORCE 收敛速度对比 ----------
    compare_with_reinforce(episode_rewards)

    # ---------- 绘制训练曲线 ----------
    plot_training_curve(episode_rewards)


# ==========================================
# 第四部分：与 REINFORCE 收敛速度对比
# ==========================================
def compare_with_reinforce(actor_critic_rewards):
    """
    对比 Actor-Critic 与 REINFORCE 的收敛速度

    收敛速度衡量标准：第一次达到目标奖励（如 195）需要多少回合
    CartPole-v1 的"解决"标准：连续 100 回合平均奖励 >= 195
    """
    target_reward = 195

    # 计算 Actor-Critic 的收敛速度
    ac_solve_episode = None
    for i in range(len(actor_critic_rewards) - 99):
        window_avg = np.mean(actor_critic_rewards[i:i + 100])
        if window_avg >= target_reward:
            ac_solve_episode = i + 100
            break

    print("\n" + "-" * 60)
    print("  收敛速度对比")
    print("-" * 60)
    print(f"  CartPole-v1 解决标准: 连续100回合平均奖励 >= {target_reward}")

    if ac_solve_episode:
        print(f"  Actor-Critic 在第 {ac_solve_episode} 回合解决环境")
    else:
        print(f"  Actor-Critic 在 {len(actor_critic_rewards)} 回合内未达到解决标准")

    # 关于 REINFORCE 的说明
    print(f"\n  【参考】一般经验值:")
    print(f"    REINFORCE 通常需要 300-500+ 回合才能解决 CartPole")
    print(f"    Actor-Critic 通常在 200-350 回合内解决")
    print(f"    原因：Actor-Critic 单步更新，数据利用效率更高")
    print(f"           TD(0) 的方差比 Monte Carlo 回报更低")
    print("-" * 60)


# ==========================================
# 第五部分：绘制训练曲线
# ==========================================
def plot_training_curve(episode_rewards):
    """
    绘制 Actor-Critic 的训练奖励曲线

    包含原始奖励和滑动平均线
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # 原始奖励曲线
    ax.plot(episode_rewards, alpha=0.3, color='steelblue', label='回合奖励（原始）')

    # 滑动平均曲线
    window = 50
    moving_avg = [np.mean(episode_rewards[max(0, i - window + 1):i + 1])
                  for i in range(len(episode_rewards))]
    ax.plot(moving_avg, color='crimson', linewidth=2.0,
            label=f'滑动平均（窗口={window}）')

    # 标注解决标准线
    ax.axhline(y=195, color='green', linestyle='--', alpha=0.7,
               label='解决标准（奖励=195）')

    ax.set_xlabel('训练回合', fontsize=12)
    ax.set_ylabel('回合奖励', fontsize=12)
    ax.set_title('Actor-Critic —— CartPole-v1 训练曲线', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/actor_critic_cartpole_rewards.png', dpi=150, bbox_inches='tight')
    print("  训练曲线已保存为 output/actor_critic_cartpole_rewards.png")
    plt.show()


# ==========================================
# 程序入口
# ==========================================
if __name__ == "__main__":
    train()
