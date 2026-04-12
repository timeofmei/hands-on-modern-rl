"""
附录A - 常见坑与解法：训练崩溃（Training Collapse）的诊断与修复

本脚本演示强化学习中另一个常见陷阱：训练崩溃。

什么是训练崩溃？
    训练过程中损失突然爆炸、策略完全退化、奖励断崖式下跌。
    与"奖励作弊"不同，训练崩溃是训练不稳定导致的，
    而非奖励设计问题。

典型表现：
    - 损失值突然变为 NaN 或无穷大
    - 奖励曲线突然掉到零附近且不再恢复
    - 梯度范数持续增大
    - 智能体只选一个动作（策略退化）

本脚本包含三个故障场景和对应的修复方案：
    场景1：学习率过高 → 损失爆炸
    场景2：缺少梯度裁剪 → 梯度爆炸
    场景3：ε 不衰减 → 永远不收敛

然后演示修复后的正确训练，以及4步调试方法论。

运行方式：
    python debug_training_collapse.py
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

# 设置中文字体，确保图表标题和标签正常显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 第一部分：核心组件（Q网络、经验回放）
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
        """前向传播：输入状态，输出各动作的 Q 值"""
        return self.net(x)


class ReplayBuffer:
    """
    经验回放缓冲区：存储和采样训练数据
    """

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
# 第二部分：可配置的 DQN 智能体
# ==========================================
class ConfigurableDQNAgent:
    """
    可配置的 DQN 智能体 —— 支持故意触发各种训练故障

    参数说明：
        lr: 学习率（故意设大可以触发损失爆炸）
        clip_grad: 是否启用梯度裁剪
        clip_max_norm: 梯度裁剪的最大范数
        epsilon_start: 初始探索率
        epsilon_end: 最终探索率
        epsilon_decay: 探索率衰减系数（设为 1.0 则不衰减）
    """

    def __init__(self, state_dim, action_dim,
                 lr=1e-3, gamma=0.99,
                 clip_grad=True, clip_max_norm=10.0,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.action_dim = action_dim
        self.gamma = gamma

        # Q 网络和目标网络
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        # 优化器
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # 经验回放
        self.buffer = ReplayBuffer(capacity=10000)

        # 梯度裁剪配置
        self.clip_grad = clip_grad
        self.clip_max_norm = clip_max_norm

        # 探索率配置
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # 诊断用：记录梯度范数
        self.last_grad_norm = 0.0

    def select_action(self, state):
        """ε-贪心动作选择"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_net(state_tensor)
            return q_values.argmax(dim=1).item()

    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def update(self, batch_size):
        """
        从经验回放中采样并更新 Q 网络
        返回：loss 值（如果发生 NaN 返回特殊标记）
        """
        if len(self.buffer) < batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_max = self.target_net(next_states).max(dim=1)[0]
            targets = rewards + self.gamma * next_q_max * (1 - dones)

        loss = nn.MSELoss()(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()

        # 计算梯度范数（诊断用）
        total_norm = 0.0
        for p in self.q_net.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        self.last_grad_norm = total_norm

        # 是否启用梯度裁剪
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(
                self.q_net.parameters(), max_norm=self.clip_max_norm
            )

        self.optimizer.step()

        return loss.item()

    def update_target(self):
        """将 Q 网络的权重复制到目标网络"""
        self.target_net.load_state_dict(self.q_net.state_dict())


# ==========================================
# 第三部分：统一训练函数
# ==========================================
def train_scenario(agent, env, label, num_episodes=300, batch_size=64,
                   target_update_freq=10, verbose=True):
    """
    在给定配置下训练 DQN 智能体

    返回丰富的诊断数据，用于后续分析和可视化

    参数：
        agent: 可配置的 DQN 智能体
        env: 训练环境
        label: 场景名称
        num_episodes: 训练回合数
        batch_size: 批次大小
        target_update_freq: 目标网络更新频率
        verbose: 是否打印训练日志
    返回：
        results: 字典，包含所有诊断数据
    """
    # 诊断数据收集
    reward_history = []
    loss_history = []
    grad_norm_history = []
    action_counts = {0: [], 1: []}  # 记录每回合左右动作的次数
    nan_detected = False

    if verbose:
        print(f"\n{'─' * 60}")
        print(f"  训练场景：{label}")
        print(f"{'─' * 60}")

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        episode_loss = 0.0
        episode_grad_norm = 0.0
        steps = 0
        actions_taken = {0: 0, 1: 0}

        while True:
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)

            agent.buffer.push(state, action, reward, next_state, float(done))
            loss = agent.update(batch_size)

            # 检测 NaN（训练崩溃的明确信号）
            if np.isnan(loss) or np.isinf(loss):
                nan_detected = True
                if verbose and episode < num_episodes - 1:
                    print(f"  [!] 回合 {episode + 1} 检测到 NaN/Inf！loss={loss}")
                loss = 0.0  # 替换为 0 以便继续绘图

            state = next_state
            episode_reward += reward
            episode_loss += loss
            episode_grad_norm += agent.last_grad_norm
            actions_taken[action] = actions_taken.get(action, 0) + 1
            steps += 1

            if done or truncated:
                break

        # 衰减探索率
        agent.decay_epsilon()

        # 记录诊断数据
        reward_history.append(episode_reward)
        loss_history.append(episode_loss / max(steps, 1))
        grad_norm_history.append(episode_grad_norm / max(steps, 1))
        action_counts[0].append(actions_taken.get(0, 0))
        action_counts[1].append(actions_taken.get(1, 0))

        # 定期更新目标网络
        if (episode + 1) % target_update_freq == 0:
            agent.update_target()

        # 打印进度
        if verbose and (episode + 1) % 100 == 0:
            avg_reward = np.mean(reward_history[-50:])
            avg_loss = np.mean(loss_history[-50:])
            avg_grad = np.mean(grad_norm_history[-50:])
            print(
                f"  回合 {episode + 1:4d}/{num_episodes} | "
                f"奖励: {avg_reward:6.1f} | "
                f"损失: {avg_loss:8.4f} | "
                f"梯度范数: {avg_grad:8.2f} | "
                f"ε: {agent.epsilon:.3f}"
            )

    env.close()

    return {
        'label': label,
        'rewards': reward_history,
        'losses': loss_history,
        'grad_norms': grad_norm_history,
        'action_counts': action_counts,
        'nan_detected': nan_detected,
    }


# ==========================================
# 第四部分：4步调试方法论
# ==========================================
def four_step_diagnosis(results):
    """
    4步调试方法论：系统化诊断训练问题

    步骤1：检查损失曲线（是否爆炸/消失？）
    步骤2：检查奖励曲线（智能体是否在进步？）
    步骤3：检查动作分布（智能体是否在探索？）
    步骤4：检查梯度范数（梯度是否稳定？）
    """
    label = results['label']
    losses = results['losses']
    rewards = results['rewards']
    action_counts = results['action_counts']
    grad_norms = results['grad_norms']

    print(f"\n{'=' * 60}")
    print(f"  4步诊断报告 —— {label}")
    print(f"{'=' * 60}")

    # ── 步骤1：检查损失曲线 ──
    print(f"\n  步骤1：检查损失曲线")
    print(f"  {'─' * 40}")

    # 过滤掉 NaN/Inf 用于统计
    valid_losses = [l for l in losses if not np.isnan(l) and not np.isinf(l)]

    if len(valid_losses) == 0:
        print(f"    所有损失值为 NaN/Inf → 训练已完全崩溃！")
        print(f"    最可能原因：学习率过高")
        print(f"    修复方案：降低学习率到 1e-3 或更小")
    else:
        avg_loss = np.mean(valid_losses)
        max_loss = np.max(valid_losses)
        min_loss = np.min(valid_losses)

        print(f"    损失范围: [{min_loss:.4f}, {max_loss:.4f}]")
        print(f"    平均损失: {avg_loss:.4f}")

        if max_loss > 1000:
            print(f"    ⚠️  损失爆炸！最大损失 {max_loss:.1f} 远超正常范围")
            print(f"    可能原因：学习率过高，或缺少梯度裁剪")
        elif max_loss > 100:
            print(f"    ⚠️  损失偏高，存在不稳定风险")
        else:
            print(f"    ✓  损失在合理范围内")

    # ── 步骤2：检查奖励曲线 ──
    print(f"\n  步骤2：检查奖励曲线")
    print(f"  {'─' * 40}")

    n = len(rewards)
    quarter = max(n // 4, 1)
    early_reward = np.mean(rewards[:quarter])
    late_reward = np.mean(rewards[-quarter:])
    reward_change = late_reward - early_reward

    print(f"    前段平均奖励: {early_reward:.1f}")
    print(f"    后段平均奖励: {late_reward:.1f}")
    print(f"    变化趋势: {'↑' if reward_change > 0 else '↓'} {abs(reward_change):.1f}")

    if reward_change > 10:
        print(f"    ✓  奖励在上升，智能体正在学习")
    elif reward_change > 0:
        print(f"    ~  奖励略有上升，学习缓慢")
    else:
        print(f"    ⚠️  奖励在下降或停滞，智能体没有在学习")
        print(f"    可能原因：探索不足、学习率不当、或策略已退化")

    # ── 步骤3：检查动作分布 ──
    print(f"\n  步骤3：检查动作分布")
    print(f"  {'─' * 40}")

    # 统计最后 50 回合的动作分布
    last_n = min(50, n)
    left_actions = np.sum(action_counts[0][-last_n:])
    right_actions = np.sum(action_counts[1][-last_n:])
    total_actions = left_actions + right_actions

    if total_actions > 0:
        left_ratio = left_actions / total_actions
        right_ratio = right_actions / total_actions
        print(f"    最后 {last_n} 回合动作分布:")
        print(f"      左推: {left_ratio:.1%} ({left_actions} 次)")
        print(f"      右推: {right_ratio:.1%} ({right_actions} 次)")

        if left_ratio > 0.95 or right_ratio > 0.95:
            print(f"    ⚠️  动作分布极度不均！策略已退化（几乎只选一个动作）")
            print(f"    可能原因：ε 衰减过快或根本没有衰减")
        elif left_ratio > 0.8 or right_ratio > 0.8:
            print(f"    ~  动作分布有些不均，但还在可接受范围")
        else:
            print(f"    ✓  动作分布较均匀，探索正常")
    else:
        print(f"    无足够数据")

    # ── 步骤4：检查梯度范数 ──
    print(f"\n  步骤4：检查梯度范数")
    print(f"  {'─' * 40}")

    valid_grads = [g for g in grad_norms if not np.isnan(g) and not np.isinf(g)]

    if len(valid_grads) == 0:
        print(f"    所有梯度范数为 NaN/Inf → 梯度完全爆炸！")
    else:
        avg_grad = np.mean(valid_grads)
        max_grad = np.max(valid_grads)

        print(f"    平均梯度范数: {avg_grad:.2f}")
        print(f"    最大梯度范数: {max_grad:.2f}")

        if max_grad > 1000:
            print(f"    ⚠️  梯度爆炸！最大梯度范数 {max_grad:.1f}")
            print(f"    修复方案：添加梯度裁剪 torch.nn.utils.clip_grad_norm_")
        elif max_grad > 100:
            print(f"    ~  梯度偶尔偏大，建议添加裁剪以防万一")
        else:
            print(f"    ✓  梯度范数在合理范围内")

    print(f"{'=' * 60}")


# ==========================================
# 第五部分：主实验流程
# ==========================================
def run_all_scenarios():
    """
    运行所有场景的训练实验

    场景1：学习率过高（lr=1.0）
    场景2：缺少梯度裁剪
    场景3：ε 不衰减（固定 ε=1.0）
    场景4：修复后的正确训练
    """
    NUM_EPISODES = 300

    print("=" * 60)
    print("  附录A：训练崩溃诊断与修复实验")
    print("=" * 60)
    print(f"  每个场景训练 {NUM_EPISODES} 回合")
    print(f"  前3个场景故意触发故障，第4个场景展示正确配置")
    print("=" * 60)

    # ── 场景1：学习率过高 ──
    print("\n" + "=" * 60)
    print("  场景1：学习率过高 (lr=1.0)")
    print("  正常学习率是 1e-3，这里故意设为 1.0（高1000倍）")
    print("  预期：损失爆炸，可能很快出现 NaN")
    print("=" * 60)

    env1 = gym.make("CartPole-v1")
    state_dim = env1.observation_space.shape[0]
    action_dim = env1.action_space.n

    agent_lr_high = ConfigurableDQNAgent(
        state_dim, action_dim,
        lr=1.0,               # 学习率过高！正常是 1e-3
        clip_grad=True,        # 保留裁剪以便看到纯粹的学习率问题
        clip_max_norm=10.0,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
    )
    results_lr_high = train_scenario(agent_lr_high, env1, "lr=1.0（过高）")

    # ── 场景2：缺少梯度裁剪 ──
    print("\n" + "=" * 60)
    print("  场景2：缺少梯度裁剪")
    print("  关闭梯度裁剪，使用正常学习率")
    print("  预期：偶尔出现梯度爆炸，训练不稳定")
    print("=" * 60)

    env2 = gym.make("CartPole-v1")
    agent_no_clip = ConfigurableDQNAgent(
        state_dim, action_dim,
        lr=1e-3,              # 正常学习率
        clip_grad=False,       # 不做梯度裁剪！
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
    )
    results_no_clip = train_scenario(agent_no_clip, env2, "无梯度裁剪")

    # ── 场景3：ε 不衰减 ──
    print("\n" + "=" * 60)
    print("  场景3：ε 不衰减（固定 ε=1.0）")
    print("  智能体始终完全随机探索，从不利用学到的知识")
    print("  预期：奖励曲线始终在低处波动，永远不收敛")
    print("=" * 60)

    env3 = gym.make("CartPole-v1")
    agent_no_decay = ConfigurableDQNAgent(
        state_dim, action_dim,
        lr=1e-3,
        clip_grad=True,
        clip_max_norm=10.0,
        epsilon_start=1.0,
        epsilon_end=1.0,       # ε 不衰减！始终完全随机
        epsilon_decay=1.0,     # 衰减系数为 1.0 = 不衰减
    )
    results_no_decay = train_scenario(agent_no_decay, env3, "ε=1.0（不衰减）")

    # ── 场景4：修复后的正确训练 ──
    print("\n" + "=" * 60)
    print("  场景4：修复后的正确训练")
    print("  合适的学习率 + 梯度裁剪 + ε 衰减")
    print("  预期：平稳学习，奖励稳步上升")
    print("=" * 60)

    env4 = gym.make("CartPole-v1")
    agent_correct = ConfigurableDQNAgent(
        state_dim, action_dim,
        lr=1e-3,              # 合适的学习率
        clip_grad=True,        # 启用梯度裁剪
        clip_max_norm=10.0,    # 裁剪阈值
        epsilon_start=1.0,     # 初始 ε
        epsilon_end=0.01,      # 最终 ε
        epsilon_decay=0.995,   # 衰减系数
    )
    results_correct = train_scenario(agent_correct, env4, "正确配置")

    return [results_lr_high, results_no_clip, results_no_decay, results_correct]


# ==========================================
# 第六部分：可视化诊断
# ==========================================
def plot_diagnosis(all_results):
    """
    绘制4个场景的诊断对比图

    4行 x 4列的网格：
    - 第1行：损失曲线（是否爆炸？）
    - 第2行：奖励曲线（是否在进步？）
    - 第3行：动作分布（是否在探索？）
    - 第4行：梯度范数（是否稳定？）

    每列是一个场景，直观对比故障和修复效果。
    """
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    fig.suptitle("训练崩溃诊断：4步调试方法论",
                 fontsize=18, fontweight='bold', y=1.01)

    colors = ['#e74c3c', '#e67e22', '#9b59b6', '#27ae60']
    titles = [r['label'] for r in all_results]

    for col, (results, color, title) in enumerate(zip(all_results, colors, titles)):
        losses = results['losses']
        rewards = results['rewards']
        action_counts = results['action_counts']
        grad_norms = results['grad_norms']

        # 处理 NaN/Inf：替换为 None 以便绘图时断开
        safe_losses = [l if not (np.isnan(l) or np.isinf(l)) else None for l in losses]
        safe_grads = [g if not (np.isnan(g) or np.isinf(g)) else None for g in grad_norms]

        # ── 第1行：损失曲线 ──
        ax1 = axes[0, col]
        # 只绘制有效值
        valid_x = [i for i, l in enumerate(safe_losses) if l is not None]
        valid_y = [l for l in safe_losses if l is not None]
        if valid_y:
            ax1.plot(valid_x, valid_y, color=color, alpha=0.6, linewidth=0.8)
            # 对损失取对数，更清楚地看到爆炸趋势
            log_losses = [np.log10(max(l, 1e-8)) for l in valid_y]
            ax1.plot(valid_x, log_losses, color='navy', alpha=0.8, linewidth=1.5,
                     label='log10(loss)')
            ax1.legend(fontsize=8)
        ax1.set_title(f"{title}\n损失曲线", fontsize=11)
        ax1.set_ylabel('损失值 / log10(损失)')
        ax1.grid(True, alpha=0.3)

        # ── 第2行：奖励曲线 ──
        ax2 = axes[1, col]
        ax2.plot(rewards, color=color, alpha=0.3, linewidth=0.8)
        window = 20
        if len(rewards) >= window:
            moving_avg = [
                np.mean(rewards[max(0, i - window): i + 1])
                for i in range(len(rewards))
            ]
            ax2.plot(moving_avg, color='navy', linewidth=2, label='滑动平均')
            ax2.legend(fontsize=8)
        ax2.set_title(f"奖励曲线", fontsize=11)
        ax2.set_ylabel('累计奖励')
        ax2.grid(True, alpha=0.3)

        # ── 第3行：动作分布 ──
        ax3 = axes[2, col]
        left_counts = np.array(action_counts[0], dtype=float)
        right_counts = np.array(action_counts[1], dtype=float)
        total_counts = left_counts + right_counts
        total_counts = np.where(total_counts == 0, 1, total_counts)  # 避免除零

        left_ratio = left_counts / total_counts
        right_ratio = right_counts / total_counts

        ax3.fill_between(range(len(left_ratio)), 0, left_ratio,
                         color='#3498db', alpha=0.6, label='左推')
        ax3.fill_between(range(len(right_ratio)), left_ratio, 1,
                         color='#e74c3c', alpha=0.6, label='右推')
        ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax3.set_title(f"动作分布", fontsize=11)
        ax3.set_ylabel('动作比例')
        ax3.set_ylim(0, 1)
        ax3.legend(fontsize=8, loc='upper right')
        ax3.grid(True, alpha=0.3)

        # ── 第4行：梯度范数 ──
        ax4 = axes[3, col]
        valid_gx = [i for i, g in enumerate(safe_grads) if g is not None]
        valid_gy = [g for g in safe_grads if g is not None]
        if valid_gy:
            ax4.plot(valid_gx, valid_gy, color=color, alpha=0.6, linewidth=0.8)
            # 对梯度范数取对数
            log_grads = [np.log10(max(g, 1e-8)) for g in valid_gy]
            ax4.plot(valid_gx, log_grads, color='navy', alpha=0.8, linewidth=1.5,
                     label='log10(grad_norm)')
            ax4.axhline(y=np.log10(10), color='red', linestyle='--', alpha=0.5,
                        label='裁剪阈值(10)')
            ax4.legend(fontsize=8)
        ax4.set_title(f"梯度范数", fontsize=11)
        ax4.set_xlabel('训练回合')
        ax4.set_ylabel('梯度范数 / log10(梯度范数)')
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("output/training_collapse_diagnosis.png", dpi=150, bbox_inches='tight')
    print("\n诊断图已保存为 output/training_collapse_diagnosis.png")
    plt.show()


# ==========================================
# 第七部分：修复方案总结
# ==========================================
def print_fix_summary():
    """
    打印训练崩溃的常见原因和修复方案
    """
    print("\n" + "=" * 60)
    print("  训练崩溃修复方案速查表")
    print("=" * 60)

    fixes = [
        {
            "故障": "损失爆炸（Loss 突然变为 NaN/Inf）",
            "症状": "loss 曲线突然飙升到天文数字",
            "原因": "学习率过高",
            "修复": "降低学习率（推荐 1e-4 ~ 1e-3），使用学习率调度器",
            "代码": "optimizer = Adam(params, lr=1e-3)  # 而非 1.0",
        },
        {
            "故障": "梯度爆炸（梯度范数持续增大）",
            "症状": "梯度范数 > 1000 且不收敛",
            "原因": "缺少梯度裁剪",
            "修复": "添加梯度裁剪 clip_grad_norm_",
            "代码": "torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)",
        },
        {
            "故障": "永不收敛（奖励始终在低处波动）",
            "症状": "几百个回合后奖励仍无上升趋势",
            "原因": "探索率 ε 不衰减",
            "修复": "使用指数衰减或线性衰减的 ε 调度",
            "代码": "epsilon = max(0.01, epsilon * 0.995)  # 每回合衰减",
        },
        {
            "故障": "策略退化（只选一个动作）",
            "症状": "动作分布极度倾斜（如 99:1）",
            "原因": "ε 衰减太快或初始值太低",
            "修复": "放缓 ε 衰减速度，增大最终 ε 下限",
            "代码": "epsilon = max(0.05, epsilon * 0.999)  # 更慢的衰减",
        },
    ]

    for i, fix in enumerate(fixes, 1):
        print(f"\n  故障 {i}：{fix['故障']}")
        print(f"    症状：{fix['症状']}")
        print(f"    原因：{fix['原因']}")
        print(f"    修复：{fix['修复']}")
        print(f"    代码：{fix['代码']}")

    # 打印调试方法论
    print(f"\n{'=' * 60}")
    print("  4步调试方法论")
    print(f"{'=' * 60}")

    steps = [
        ("步骤1：检查损失曲线",
         "观察 loss 是否在合理范围内波动。",
         "如果 loss > 1000 或出现 NaN → 学习率问题或梯度爆炸。",
         "修复：降低学习率，添加梯度裁剪。"),
        ("步骤2：检查奖励曲线",
         "观察奖励是否有上升趋势。",
         "如果奖励始终不变 → 策略没有在学习。",
         "修复：检查探索率、学习率、奖励函数设计。"),
        ("步骤3：检查动作分布",
         "统计智能体选择各动作的频率。",
         "如果某个动作占比 > 95% → 策略已退化。",
         "修复：调整 ε 调度，确保足够的探索。"),
        ("步骤4：检查梯度范数",
         "监控梯度范数的变化趋势。",
         "如果梯度范数持续 > 100 → 梯度爆炸。",
         "修复：添加或收紧梯度裁剪。"),
    ]

    for title, line1, line2, line3 in steps:
        print(f"\n  {title}")
        print(f"    {line1}")
        print(f"    {line2}")
        print(f"    {line3}")

    # 完整的诊断清单
    print(f"\n{'=' * 60}")
    print("  诊断清单")
    print(f"{'=' * 60}")

    checklist = [
        "□ 损失值是否在合理范围（不是 NaN/Inf/天文数字）？",
        "□ 损失是否有下降趋势（而不是持续上升）？",
        "□ 奖励曲线是否在稳步上升？",
        "□ 动作分布是否合理（不是只选一个动作）？",
        "□ 梯度范数是否稳定（不是持续增大）？",
        "□ ε 是否在按计划衰减（不是固定不变）？",
        "□ 学习率是否在合理范围（1e-4 ~ 1e-3）？",
        "□ 是否启用了梯度裁剪（max_norm=10 左右）？",
    ]

    for item in checklist:
        print(f"  {item}")

    print("=" * 60)


# ==========================================
# 程序入口
# ==========================================
if __name__ == "__main__":
    # 运行全部场景
    all_results = run_all_scenarios()

    # 对每个场景执行4步诊断
    for results in all_results:
        four_step_diagnosis(results)

    # 绘制诊断对比图
    plot_diagnosis(all_results)

    # 打印修复方案总结
    print_fix_summary()
