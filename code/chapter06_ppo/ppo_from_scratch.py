"""
第6章：从零实现 PPO（近端策略优化）
——用纯 PyTorch 在 CartPole-v1 上理解 PPO 的每一步

PPO 的核心公式：
    ratio = exp(new_logprob - old_logprob)
    clipped_ratio = clip(ratio, 1-eps, 1+eps)
    policy_loss = -min(ratio * advantage, clipped_ratio * advantage)

运行方式：
    python ppo_from_scratch.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Categorical

# 创建输出目录
os.makedirs("output", exist_ok=True)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 第一部分：Actor-Critic 网络
# ==========================================
class ActorCritic(nn.Module):
    """
    Actor-Critic 网络：共享主干 + 独立的动作头和价值头

    结构：
        共享层:  state_dim → 64 → 64 (ReLU)
        Actor:   64 → action_dim (输出动作 logits)
        Critic:  64 → 1 (输出状态价值 V(s))

    共享主干的好处：
        - 特征复用，减少参数量
        - Actor 和 Critic 共享底层表示
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()

        # 共享主干网络
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Actor 头：输出动作的 logits
        self.actor_head = nn.Linear(64, action_dim)

        # Critic 头：输出状态价值
        self.critic_head = nn.Linear(64, 1)

    def forward(self, x):
        """前向传播，返回动作概率和价值"""
        shared_features = self.shared_net(x)

        # Actor: 输出动作分布
        action_logits = self.actor_head(shared_features)
        action_probs = F.softmax(action_logits, dim=-1)

        # Critic: 输出状态价值
        value = self.critic_head(shared_features)

        return action_probs, value

    def get_action(self, state):
        """根据当前状态采样动作，返回动作、log概率、价值"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs, value = self.forward(state_tensor)

        # 使用 Categorical 分布采样
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob, value.squeeze()

    def evaluate(self, states, actions):
        """
        评估给定的 (状态, 动作) 对
        返回：log概率、状态价值、分布熵
        """
        action_probs, values = self.forward(states)
        dist = Categorical(action_probs)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, values.squeeze(), entropy


# ==========================================
# 第二部分：GAE（广义优势估计）
# ==========================================
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    计算广义优势估计 (Generalized Advantage Estimation)

    GAE 的核心思想：
        δ_t = r_t + γ * V(s_{t+1}) - V(s_t)    # TD 误差
        A_t = Σ_{l=0}^{∞} (γλ)^l * δ_{t+l}      # GAE 优势

    参数：
        rewards: 每步的奖励
        values:  每步的价值估计 V(s)
        dones:   每步是否结束
        gamma:   折扣因子（控制远期回报的权重）
        lam:     GAE lambda（控制偏差-方差权衡）
            λ=0: 低方差、高偏差（仅看单步 TD 误差）
            λ=1: 高方差、低偏差（蒙特卡洛回报）

    返回：
        advantages: 优势估计
        returns:    目标回报（用于训练 Critic）
    """
    advantages = []
    gae = 0

    # 将列表转为张量方便计算
    values = list(values)
    # 最后一步需要添加一个终止状态的 V(s)=0
    next_value = 0

    # 从后往前倒推计算 GAE
    for t in reversed(range(len(rewards))):
        if dones[t]:
            # 回合结束，下一步价值为 0
            next_value = 0
            gae = 0

        # TD 误差：δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
        delta = rewards[t] + gamma * next_value - values[t]

        # GAE 累加：A_t = δ_t + (γλ) * A_{t+1}
        gae = delta + gamma * lam * gae

        advantages.insert(0, gae)

        # 更新下一步的 V(s)
        next_value = values[t]

    advantages = torch.FloatTensor(advantages)
    returns = advantages + torch.FloatTensor(values)

    # 归一化优势（提高训练稳定性）
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns


# ==========================================
# 第三部分：PPO 裁剪损失
# ==========================================
def ppo_clip_loss(old_logprobs, new_logprobs, advantages, clip_eps=0.2):
    """
    PPO 裁剪目标函数

    核心公式：
        ratio = exp(new_logprob - old_logprob) = π_new(a|s) / π_old(a|s)
        L_CLIP = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)

    当 ratio > 1+ε 或 ratio < 1-ε 时，梯度被截断
    → 防止策略更新步幅过大

    参数：
        old_logprobs: 旧策略的 log 概率
        new_logprobs: 新策略的 log 概率
        advantages:   优势估计
        clip_eps:     裁剪范围 ε（默认 0.2）

    返回：
        policy_loss: 策略损失
        clip_frac:   被裁剪的比例（用于监控训练）
    """
    # 计算重要性采样比率
    ratio = torch.exp(new_logprobs - old_logprobs)

    # 未裁剪的目标
    surr1 = ratio * advantages

    # 裁剪后的目标
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages

    # 取两者中较小值（保守更新）
    policy_loss = -torch.min(surr1, surr2).mean()

    # 计算被裁剪的比例（监控指标）
    with torch.no_grad():
        clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean().item()

    return policy_loss, clip_frac


# ==========================================
# 第四部分：收集轨迹数据
# ==========================================
def collect_trajectories(model, env, n_steps=2048):
    """
    使用当前策略在环境中收集 n_steps 步的轨迹数据

    收集内容：
        - states:  状态
        - actions: 动作
        - logprobs: 旧策略的 log 概率（用于后续 PPO 更新）
        - rewards: 奖励
        - dones:   回合结束标志
        - values:  价值估计

    返回：
        batch 字典 + 累计回合奖励列表
    """
    states = []
    actions = []
    old_logprobs = []
    rewards = []
    dones = []
    values = []

    obs, _ = env.reset()
    episode_rewards = []
    current_ep_reward = 0

    for step in range(n_steps):
        state_tensor = torch.FloatTensor(obs)

        # 用当前策略采样动作
        with torch.no_grad():
            action_probs, value = model(state_tensor)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        # 存储数据
        states.append(obs.copy())
        actions.append(action.item())
        old_logprobs.append(log_prob.item())
        values.append(value.item())

        # 执行动作
        next_obs, reward, done, truncated, _ = env.step(action.item())
        rewards.append(reward)
        dones.append(done or truncated)

        current_ep_reward += reward

        if done or truncated:
            episode_rewards.append(current_ep_reward)
            current_ep_reward = 0
            next_obs, _ = env.reset()

        obs = next_obs

    # 转为张量
    batch = {
        "states": torch.FloatTensor(np.array(states)),
        "actions": torch.LongTensor(actions),
        "old_logprobs": torch.FloatTensor(old_logprobs),
        "rewards": rewards,
        "dones": dones,
        "values": values,
    }

    return batch, episode_rewards


# ==========================================
# 第五部分：PPO 更新
# ==========================================
def ppo_update(model, optimizer, batch, n_epochs=10, batch_size=64,
               clip_eps=0.2, vf_coef=0.5, ent_coef=0.01):
    """
    用收集的数据进行多轮 PPO 更新

    每轮更新：
        1. 用新策略重新评估旧数据 → 得到新的 log_probs
        2. 计算 PPO 裁剪损失（策略损失）
        3. 计算价值函数损失（Critic）
        4. 计算熵奖励（鼓励探索）
        5. 总损失 = 策略损失 + 价值损失 - 熵奖励

    返回：
        训练指标字典（用于监控）
    """
    # 先计算 GAE 优势和目标回报
    advantages, returns = compute_gae(
        batch["rewards"], batch["values"], batch["dones"],
        gamma=0.99, lam=0.95
    )

    # 将数据移到 CPU（保持简单）
    states = batch["states"]
    actions = batch["actions"]
    old_logprobs = batch["old_logprobs"]

    dataset_size = states.shape[0]
    total_policy_loss = 0
    total_value_loss = 0
    total_entropy = 0
    total_clip_frac = 0
    update_count = 0

    for epoch in range(n_epochs):
        # 随机打乱数据
        indices = torch.randperm(dataset_size)

        for start in range(0, dataset_size, batch_size):
            end = start + batch_size
            mb_indices = indices[start:end]

            mb_states = states[mb_indices]
            mb_actions = actions[mb_indices]
            mb_old_logprobs = old_logprobs[mb_indices]
            mb_advantages = advantages[mb_indices]
            mb_returns = returns[mb_indices]

            # 用新策略评估旧数据
            new_logprobs, new_values, entropy = model.evaluate(mb_states, mb_actions)

            # ---- 策略损失（PPO-Clip）----
            policy_loss, clip_frac = ppo_clip_loss(
                mb_old_logprobs, new_logprobs, mb_advantages, clip_eps
            )

            # ---- 价值函数损失 ----
            value_loss = F.mse_loss(new_values, mb_returns)

            # ---- 熵奖励 ----
            entropy_bonus = entropy.mean()

            # ---- 总损失 ----
            # 总损失 = 策略损失 + vf_coef * 价值损失 - ent_coef * 熵
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_bonus

            # 梯度更新
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪（防止梯度爆炸）
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            # 累计统计量
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy_bonus.item()
            total_clip_frac += clip_frac
            update_count += 1

    # 返回平均指标
    metrics = {
        "policy_loss": total_policy_loss / update_count,
        "value_loss": total_value_loss / update_count,
        "entropy": total_entropy / update_count,
        "clip_fraction": total_clip_frac / update_count,
    }

    return metrics


# ==========================================
# 第六部分：主训练循环
# ==========================================
def train():
    """PPO 主训练函数"""
    print("=" * 50)
    print("第6章：从零实现 PPO — CartPole-v1")
    print("=" * 50)

    # 创建环境
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]   # 4
    action_dim = env.action_space.n               # 2

    # 创建模型和优化器
    model = ActorCritic(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    print(f"\n网络结构:")
    print(model)
    print(f"\n状态维度: {state_dim}, 动作维度: {action_dim}")

    # 训练参数
    n_steps = 2048        # 每次收集的步数
    n_epochs = 10         # 每批数据的更新轮数
    batch_size = 64       # 小批量大小
    clip_eps = 0.2        # PPO 裁剪范围
    total_episodes = 1000 # 总训练回合数

    # 记录训练指标
    all_rewards = []
    all_policy_losses = []
    all_value_losses = []
    all_entropies = []
    all_clip_fracs = []

    print(f"\n开始训练（目标: {total_episodes} 回合）...")
    print("-" * 50)

    episode_count = 0
    iteration = 0

    while episode_count < total_episodes:
        iteration += 1

        # 第一步：收集轨迹
        batch, ep_rewards = collect_trajectories(model, env, n_steps=n_steps)
        episode_count += len(ep_rewards)
        all_rewards.extend(ep_rewards)

        # 第二步：PPO 更新
        metrics = ppo_update(
            model, optimizer, batch,
            n_epochs=n_epochs,
            batch_size=batch_size,
            clip_eps=clip_eps,
        )

        all_policy_losses.append(metrics["policy_loss"])
        all_value_losses.append(metrics["value_loss"])
        all_entropies.append(metrics["entropy"])
        all_clip_fracs.append(metrics["clip_fraction"])

        # 定期打印训练信息
        if iteration % 5 == 0 or len(ep_rewards) > 0:
            recent_rewards = all_rewards[-20:] if len(all_rewards) >= 20 else all_rewards
            avg_reward = np.mean(recent_rewards)
            print(
                f"  迭代 {iteration:3d} | "
                f"回合: {episode_count:4d} | "
                f"平均奖励: {avg_reward:6.1f} | "
                f"策略损失: {metrics['policy_loss']:.4f} | "
                f"价值损失: {metrics['value_loss']:.4f} | "
                f"熵: {metrics['entropy']:.3f} | "
                f"裁剪比例: {metrics['clip_fraction']:.3f}"
            )

    print("-" * 50)
    print(f"训练完成！共训练 {episode_count} 回合，{iteration} 次迭代")

    # 最终评估
    test_rewards = []
    for _ in range(20):
        obs, _ = env.reset()
        done, truncated = False, False
        total_reward = 0
        while not (done or truncated):
            state_tensor = torch.FloatTensor(obs)
            with torch.no_grad():
                action_probs, _ = model(state_tensor)
            action = torch.argmax(action_probs).item()
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
        test_rewards.append(total_reward)

    mean_reward = np.mean(test_rewards)
    std_reward = np.std(test_rewards)
    print(f"\n20 回合测试结果: 平均奖励 = {mean_reward:.1f} ± {std_reward:.1f}")

    env.close()

    # ==========================================
    # 第七部分：绘制训练曲线
    # ==========================================
    print("\n正在绘制训练曲线...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("PPO 从零实现 — CartPole-v1 训练曲线", fontsize=16, fontweight="bold")

    # 子图1：回合奖励
    ax1 = axes[0, 0]
    window = min(20, len(all_rewards))
    if window > 0:
        smoothed = np.convolve(all_rewards, np.ones(window) / window, mode="valid")
        ax1.plot(range(len(all_rewards)), all_rewards, alpha=0.3, color="#90CAF9", label="原始奖励")
        ax1.plot(range(window - 1, len(all_rewards)), smoothed, color="#2196F3",
                 linewidth=2, label=f"滑动平均 (窗口={window})")
        ax1.axhline(y=475, color="green", linestyle="--", alpha=0.5, label="目标线 (475)")
    ax1.set_title("回合奖励", fontsize=13)
    ax1.set_xlabel("回合")
    ax1.set_ylabel("累计奖励")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 子图2：策略损失 & 价值损失
    ax2 = axes[0, 1]
    if all_policy_losses:
        ax2.plot(all_policy_losses, color="#F44336", alpha=0.8, linewidth=1.2, label="策略损失")
        ax2.plot(all_value_losses, color="#2196F3", alpha=0.8, linewidth=1.2, label="价值损失")
    ax2.set_title("损失曲线", fontsize=13)
    ax2.set_xlabel("迭代")
    ax2.set_ylabel("损失值")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 子图3：策略熵
    ax3 = axes[1, 0]
    if all_entropies:
        ax3.plot(all_entropies, color="#FF9800", alpha=0.8, linewidth=1.5)
        ax3.set_title("策略熵（探索程度）", fontsize=13)
        ax3.set_xlabel("迭代")
        ax3.set_ylabel("熵")
        ax3.annotate("熵下降 = 策略更确定", xy=(len(all_entropies) * 0.6, max(all_entropies) * 0.8),
                     fontsize=10, color="gray", style="italic")
    ax3.grid(True, alpha=0.3)

    # 子图4：裁剪比例
    ax4 = axes[1, 1]
    if all_clip_fracs:
        ax4.plot(all_clip_fracs, color="#9C27B0", alpha=0.8, linewidth=1.5)
        ax4.axhline(y=0.2, color="gray", linestyle="--", alpha=0.5, label="clip_range = 0.2")
        ax4.set_title("裁剪比例", fontsize=13)
        ax4.set_xlabel("迭代")
        ax4.set_ylabel("被裁剪的比例")
        ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("output/ppo_from_scratch_curves.png", dpi=150, bbox_inches="tight")
    print("训练曲线已保存至: output/ppo_from_scratch_curves.png")
    plt.show()


# ==========================================
# 入口
# ==========================================
if __name__ == "__main__":
    train()
