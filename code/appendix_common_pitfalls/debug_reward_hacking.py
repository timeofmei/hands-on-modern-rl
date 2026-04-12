"""
附录A - 常见坑与解法：奖励作弊（Reward Hacking）的诊断与修复

本脚本演示强化学习中最常见、最危险的陷阱之一：奖励作弊。

什么是奖励作弊？
    智能体学会了"钻空子"——利用奖励函数的设计缺陷来获取高分，
    但实际上并没有完成我们真正期望的任务。

典型表现：
    - 奖励曲线持续上升（看起来训练很成功）
    - 但实际任务表现反而下降（甚至完全不做正事）

本脚本包含三个场景：
    场景1：被动手存活——CartPole 中智能体学会"站着不动"获取奖励
    场景2：奖励塑形出错——不当的中间奖励导致意外行为
    场景3：正确的奖励设计——如何避免上述问题

运行方式：
    python debug_reward_hacking.py
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
# 第一部分：工具组件（Q网络、经验回放、智能体）
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
        """用 deque 实现，容量满后自动丢弃旧数据"""
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """存储一条转移经验 (s, a, r, s', done)"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """随机采样一批数据，打破时间相关性"""
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


class DQNAgent:
    """
    DQN 智能体：整合 Q 网络、目标网络、经验回放和 ε-贪心策略

    为了复用代码，这里使用统一的智能体结构，
    不同场景通过修改环境包装器来改变奖励函数。
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
        """从经验回放中采样并更新 Q 网络"""
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
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10)
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        """将 Q 网络的权重复制到目标网络（硬更新）"""
        self.target_net.load_state_dict(self.q_net.state_dict())


# ==========================================
# 第二部分：自定义环境包装器（模拟奖励作弊）
# ==========================================
class HackedRewardWrapper(gym.Wrapper):
    """
    被动存活奖励包装器 —— 模拟"奖励作弊"场景

    原始 CartPole 奖励：每步 +1，直到杆子倒下。
    这本身是正确的设计：智能体需要保持杆子直立才能获得高分。

    但如果我们额外给"存活时间"加分，而不要求杆子角度，
    智能体可能学会一种奇怪策略：让杆子快速来回摆动，
    或者找到某种物理引擎漏洞来延长存活时间。

    这里我们模拟一种更极端的情况：
    给"杆子几乎水平"的状态额外奖励（即快要倒下也有奖励），
    这会鼓励智能体在边缘状态徘徊，而不是真正保持平衡。
    """

    def __init__(self, env, survival_bonus=2.0):
        super().__init__(env)
        self.survival_bonus = survival_bonus

    def step(self, action):
        """
        修改奖励函数：
        - 原始奖励：每步 +1
        - 作弊奖励：额外给"存活"本身加分，不管杆子角度如何
        - 只要没结束，就给大额 bonus → 智能体学会了"拖延"
        """
        next_state, reward, done, truncated, info = self.env.step(action)

        # 作弊奖励：不管做什么，只要还活着就给额外奖励
        # 这会让智能体发现"随便做什么都行，只要活着"
        hacked_reward = reward + self.survival_bonus

        return next_state, hacked_reward, done, truncated, info


class BadShapingWrapper(gym.Wrapper):
    """
    不当奖励塑形包装器 —— 模拟"奖励塑形出错"场景

    场景：我们希望智能体把小车推到屏幕中央。
    设计了一个"距中心越近越好"的中间奖励。
    但问题是：智能体学会了在中心附近来回振荡，
    而不是稳定地保持平衡！

    这种情况在现实中很常见：你以为的"帮助"其实是"误导"。
    """

    def __init__(self, env, position_reward_weight=5.0):
        super().__init__(env)
        self.position_reward_weight = position_reward_weight

    def step(self, action):
        """
        修改奖励函数：
        - 原始奖励：每步 +1（保持杆子直立）
        - 塑形奖励：额外给"小车接近中心"加分

        问题在于 position_reward_weight 太大，
        导致智能体主要关注"推车到中心"，而忽略了"保持杆子直立"。
        结果：小车确实在中心了，但杆子倒了。
        """
        next_state, reward, done, truncated, info = self.env.step(action)

        # next_state[0] 是小车的位置，范围大约 [-2.4, 2.4]
        # 给"靠近中心"的位置奖励
        position = next_state[0]
        position_reward = self.position_reward_weight * (1.0 - abs(position) / 2.4)

        shaped_reward = reward + position_reward

        return next_state, shaped_reward, done, truncated, info


# ==========================================
# 第三部分：统一训练函数
# ==========================================
def train_agent(env, label, num_episodes=300, batch_size=64):
    """
    在给定环境中训练 DQN 智能体，返回训练曲线和任务指标

    参数：
        env: 训练环境（可能是被包装过的）
        label: 场景名称（用于打印和绘图）
        num_episodes: 训练回合数
        batch_size: 批次大小
    返回：
        reward_history: 每回合的奖励列表（修改后的奖励）
        task_score_history: 每回合的真实任务得分（原始奖励之和）
        loss_history: 损失值历史
    """
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim, lr=1e-3, gamma=0.99)

    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.995
    TARGET_UPDATE_FREQ = 10

    epsilon = EPSILON_START
    reward_history = []
    task_score_history = []  # 真实任务得分（原始奖励）
    loss_history = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        episode_task_score = 0.0  # 只统计原始 CartPole 奖励
        episode_loss = 0.0
        steps = 0

        while True:
            action = agent.select_action(state, epsilon)

            # 记录修改后的奖励（用于训练）
            next_state, reward, done, truncated, _ = env.step(action)

            # 同时计算原始任务得分（不包含作弊奖励）
            # 原始 CartPole 每步奖励为 1.0
            original_reward = 1.0

            agent.buffer.push(state, action, reward, next_state, float(done))
            loss = agent.update(batch_size)

            state = next_state
            episode_reward += reward
            episode_task_score += original_reward
            episode_loss += loss
            steps += 1

            if done or truncated:
                break

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        reward_history.append(episode_reward)
        task_score_history.append(episode_task_score)
        loss_history.append(episode_loss / max(steps, 1))

        if (episode + 1) % TARGET_UPDATE_FREQ == 0:
            agent.update_target()

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(reward_history[-50:])
            avg_task = np.mean(task_score_history[-50:])
            print(
                f"  [{label}] 回合 {episode + 1:4d}/{num_episodes} | "
                f"奖励(修改后): {avg_reward:7.1f} | "
                f"任务得分(原始): {avg_task:5.1f}"
            )

    env.close()
    return reward_history, task_score_history, loss_history


# ==========================================
# 第四部分：诊断函数 —— 检测奖励作弊
# ==========================================
def diagnose_reward_hacking(reward_history, task_score_history, label):
    """
    奖励作弊诊断工具

    核心思路：
        - 奖励作弊的标志是"奖励上升，但任务表现下降"
        - 我们通过计算奖励曲线与任务得分曲线的相关性来判断
        - 如果两者正相关且都上升 → 正常
        - 如果奖励上升但任务得分下降 → 作弊！

    参数：
        reward_history: 修改后的奖励曲线
        task_score_history: 真实任务得分曲线
        label: 场景名称
    返回：
        is_hacked: 是否检测到作弊（True/False）
    """
    print(f"\n{'=' * 60}")
    print(f"  奖励作弊诊断报告 —— {label}")
    print(f"{'=' * 60}")

    # 检查前 1/4 与后 1/4 的趋势对比
    n = len(reward_history)
    quarter = max(n // 4, 1)

    # 前段和后段的平均奖励
    early_reward = np.mean(reward_history[:quarter])
    late_reward = np.mean(reward_history[-quarter:])
    reward_trend = late_reward - early_reward

    # 前段和后段的平均任务得分
    early_task = np.mean(task_score_history[:quarter])
    late_task = np.mean(task_score_history[-quarter:])
    task_trend = late_task - early_task

    print(f"  前段平均奖励（修改后）: {early_reward:.1f}")
    print(f"  后段平均奖励（修改后）: {late_reward:.1f}")
    print(f"  奖励变化趋势: {'↑' if reward_trend > 0 else '↓'} {abs(reward_trend):.1f}")
    print(f"  ─────────────────────────────")
    print(f"  前段平均任务得分（原始）: {early_task:.1f}")
    print(f"  后段平均任务得分（原始）: {late_task:.1f}")
    print(f"  任务得分变化趋势: {'↑' if task_trend > 0 else '↓'} {abs(task_trend):.1f}")

    # 诊断：奖励上升但任务表现下降 = 作弊
    is_hacked = (reward_trend > 0) and (task_trend < 0)

    # 计算奖励与任务得分的相关系数
    if n > 10:
        correlation = np.corrcoef(reward_history, task_score_history)[0, 1]
        print(f"  奖励与任务得分的相关系数: {correlation:.3f}")
    else:
        correlation = 1.0

    print(f"  ─────────────────────────────")
    if is_hacked:
        print(f"  ⚠️  检测到奖励作弊！")
        print(f"  奖励在上升，但真实任务表现反而在下降。")
        print(f"  智能体可能学会了利用奖励函数的漏洞。")
        print(f"  建议：重新审视奖励函数设计。")
    elif correlation < 0.3:
        print(f"  ⚠️  奖励与任务得分相关性很低（{correlation:.3f}）")
        print(f"  奖励函数可能没有正确反映任务目标。")
        print(f"  建议：检查奖励函数是否与真实目标对齐。")
    else:
        print(f"  ✓  奖励与任务表现一致，未检测到作弊。")

    print(f"{'=' * 60}")

    return is_hacked


# ==========================================
# 第五部分：主实验流程
# ==========================================
def run_all_experiments():
    """
    运行所有三个场景的对比实验

    场景1：被动存活作弊（HackedRewardWrapper）
    场景2：奖励塑形出错（BadShapingWrapper）
    场景3：正确设计（原始 CartPole 奖励）
    """

    NUM_EPISODES = 300  # 每个场景训练 300 回合

    print("=" * 60)
    print("  附录A：奖励作弊（Reward Hacking）实验")
    print("=" * 60)
    print(f"  每个场景训练 {NUM_EPISODES} 回合")
    print(f"  通过对比'修改后奖励'和'真实任务得分'来诊断作弊")
    print("=" * 60)

    # ── 场景1：被动存活作弊 ──
    print("\n" + "─" * 60)
    print("  场景1：被动存活奖励作弊")
    print("  问题：不管做什么，只要活着就给额外奖励")
    print("  预期：智能体学会拖延，而非真正保持平衡")
    print("─" * 60)

    hacked_env = HackedRewardWrapper(
        gym.make("CartPole-v1"),
        survival_bonus=2.0,  # 额外的存活奖励
    )
    hacked_rewards, hacked_task_scores, hacked_losses = train_agent(
        hacked_env, "作弊奖励", num_episodes=NUM_EPISODES
    )
    diagnose_reward_hacking(hacked_rewards, hacked_task_scores, "场景1-被动存活作弊")

    # ── 场景2：奖励塑形出错 ──
    print("\n" + "─" * 60)
    print("  场景2：奖励塑形出错")
    print("  问题：过度奖励'靠近中心'，导致忽略'保持平衡'")
    print("  预期：小车确实在中心，但杆子可能倒下")
    print("─" * 60)

    bad_shaping_env = BadShapingWrapper(
        gym.make("CartPole-v1"),
        position_reward_weight=5.0,  # 位置奖励权重过大
    )
    shaping_rewards, shaping_task_scores, shaping_losses = train_agent(
        bad_shaping_env, "塑形出错", num_episodes=NUM_EPISODES
    )
    diagnose_reward_hacking(shaping_rewards, shaping_task_scores, "场景2-塑形出错")

    # ── 场景3：正确的奖励设计 ──
    print("\n" + "─" * 60)
    print("  场景3：正确的奖励设计（基准线）")
    print("  使用原始 CartPole 奖励：每步 +1，直到杆子倒下")
    print("  这是简洁、有效的奖励设计典范")
    print("─" * 60)

    correct_env = gym.make("CartPole-v1")
    correct_rewards, correct_task_scores, correct_losses = train_agent(
        correct_env, "正确奖励", num_episodes=NUM_EPISODES
    )
    diagnose_reward_hacking(correct_rewards, correct_task_scores, "场景3-正确设计")

    return (hacked_rewards, hacked_task_scores,
            shaping_rewards, shaping_task_scores,
            correct_rewards, correct_task_scores)


# ==========================================
# 第六部分：可视化对比
# ==========================================
def plot_comparison(hacked_rewards, hacked_task_scores,
                    shaping_rewards, shaping_task_scores,
                    correct_rewards, correct_task_scores):
    """
    绘制三个场景的对比图

    上排：修改后的奖励曲线（看起来都不错）
    下排：真实任务得分曲线（作弊的场景暴露问题）

    这种"上下对照"的方式是诊断奖励作弊的标准方法。
    """

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("奖励作弊诊断：修改后奖励 vs 真实任务得分",
                 fontsize=16, fontweight='bold')

    scenarios = [
        ("场景1：被动存活作弊", hacked_rewards, hacked_task_scores, "#e74c3c"),
        ("场景2：奖励塑形出错", shaping_rewards, shaping_task_scores, "#e67e22"),
        ("场景3：正确奖励设计", correct_rewards, correct_task_scores, "#27ae60"),
    ]

    window = 20  # 滑动平均窗口

    for col, (title, rewards, task_scores, color) in enumerate(scenarios):
        # ── 上排：修改后的奖励曲线 ──
        ax_top = axes[0, col]
        ax_top.plot(rewards, alpha=0.2, color=color)
        if len(rewards) >= window:
            moving_avg = [
                np.mean(rewards[max(0, i - window): i + 1])
                for i in range(len(rewards))
            ]
            ax_top.plot(moving_avg, color=color, linewidth=2, label='滑动平均')
        ax_top.set_title(f"{title}\n修改后奖励", fontsize=12)
        ax_top.set_xlabel('训练回合')
        ax_top.set_ylabel('奖励（修改后）')
        ax_top.grid(True, alpha=0.3)
        ax_top.legend(fontsize=9)

        # ── 下排：真实任务得分曲线 ──
        ax_bot = axes[1, col]
        ax_bot.plot(task_scores, alpha=0.2, color=color)
        if len(task_scores) >= window:
            moving_avg_task = [
                np.mean(task_scores[max(0, i - window): i + 1])
                for i in range(len(task_scores))
            ]
            ax_bot.plot(moving_avg_task, color=color, linewidth=2, label='滑动平均')
        ax_bot.set_title(f"{title}\n真实任务得分", fontsize=12)
        ax_bot.set_xlabel('训练回合')
        ax_bot.set_ylabel('任务得分（原始）')
        ax_bot.grid(True, alpha=0.3)
        ax_bot.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("output/reward_hacking_diagnosis.png", dpi=150, bbox_inches='tight')
    print("\n诊断图已保存为 output/reward_hacking_diagnosis.png")
    plt.show()


# ==========================================
# 第七部分：奖励设计原则总结
# ==========================================
def print_reward_design_principles():
    """
    打印正确的奖励设计原则

    这些原则来自强化学习实践中的经验总结，
    特别参考了 OpenAI 和 DeepMind 的奖励工程指南。
    """
    print("\n" + "=" * 60)
    print("  奖励设计最佳实践（避免奖励作弊）")
    print("=" * 60)

    principles = [
        ("1. 保持奖励简单",
         "奖励函数应该尽可能简单直接。CartPole 的奖励就是每步 +1，"
         "没有额外的塑形项，但效果最好。"),

        ("2. 确保奖励与目标对齐",
         "奖励必须准确反映你真正想要的行为。如果一个'坏'行为能获得高奖励，"
         "智能体一定会找到并利用它。"),

        ("3. 避免过度塑形",
         "奖励塑形（Reward Shaping）是把双刃剑。"
         "不当的塑形不仅不能帮助学习，反而会引入新的局部最优。"
         "如果必须塑形，使用势函数塑形（Potential-Based Shaping），"
         "它不会改变最优策略。"),

        ("4. 监控多个指标",
         "永远不要只看一个指标。同时监控：奖励、任务得分、行为分布。"
         "如果奖励上升但任务表现下降，说明发生了作弊。"),

        ("5. 先跑基线，再加复杂度",
         "先用最简单的奖励函数跑一个基线，确认能学到东西。"
         "然后逐步添加塑形项，每一步都验证是否真正帮助了学习。"),

        ("6. 做对抗测试",
         "训练完成后，用不同的初始状态测试智能体。"
         "如果智能体在某些状态下表现异常差，"
         "可能说明它学到了某种'投机取巧'的策略。"),
    ]

    for title, description in principles:
        print(f"\n  {title}")
        print(f"    {description}")

    print("\n" + "=" * 60)
    print("  诊断清单：如何快速发现奖励作弊")
    print("=" * 60)

    checklist = [
        "□ 奖励曲线是否持续上升？",
        "□ 任务表现（真实得分）是否也在同步上升？",
        "□ 奖励和任务表现的相关系数是否大于 0.5？",
        "□ 智能体的行为是否符合预期（而非找到奇怪策略）？",
        "□ 在不同初始条件下，智能体是否依然表现良好？",
        "□ 奖励函数中是否存在可以被利用的'漏洞'？",
    ]

    for item in checklist:
        print(f"  {item}")

    print("=" * 60)


# ==========================================
# 程序入口
# ==========================================
if __name__ == "__main__":
    # 运行全部实验
    results = run_all_experiments()

    # 解包结果
    (hacked_rewards, hacked_task_scores,
     shaping_rewards, shaping_task_scores,
     correct_rewards, correct_task_scores) = results

    # 绘制对比图
    plot_comparison(
        hacked_rewards, hacked_task_scores,
        shaping_rewards, shaping_task_scores,
        correct_rewards, correct_task_scores,
    )

    # 打印奖励设计原则
    print_reward_design_principles()
