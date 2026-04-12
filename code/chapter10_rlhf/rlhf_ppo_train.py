"""
第10章：RLHF PPO 对齐训练
==========================

本脚本演示 RLHF 三阶段流水线的第三阶段 —— PPO 对齐训练。
内容包括：
  1. 加载 SFT 模型作为策略模型（Actor）
  2. 使用奖励模型对生成回复进行评分
  3. PPO 训练循环：生成 → 评分 → 计算优势 → 裁剪更新
  4. 跟踪奖励、KL 散度、回复长度等指标
  5. 对比对齐前后的回复质量

注意：这是一个简化/模拟版本。完整的 RLHF-PPO 训练通常需要：
  - 大规模集群（数十到数百 GPU）
  - 数十万条偏好数据
  - 复杂的分布式训练框架
  本脚本旨在帮助理解 PPO 在 RLHF 中的核心算法流程。
"""

import os
import json
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 创建输出目录
os.makedirs("output", exist_ok=True)

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ==========================================
# 1. 辅助函数
# ==========================================

def generate_response(model, tokenizer, prompt, max_new_tokens=80, temperature=0.7):
    """
    使用模型生成回复。
    """
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )
    input_length = inputs["input_ids"].shape[-1]
    return response, input_length, outputs[0]


def compute_log_probs(model, input_ids, attention_mask):
    """
    计算模型在给定序列上的对数概率。
    用于 PPO 中的重要性采样比率计算。
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    # 取每个位置预测下一个 token 的对数概率
    # logits[:, :-1, :] 对应位置 t 的预测，target 是 input_ids[:, 1:]
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    # 计算对数 softmax
    log_probs = F.log_softmax(shift_logits, dim=-1)

    # 提取实际 token 的对数概率
    token_log_probs = log_probs.gather(
        2, shift_labels.unsqueeze(-1)
    ).squeeze(-1)

    # 用 attention mask 排除 padding 部分（对齐到 shift 后的位置）
    shift_mask = attention_mask[:, 1:]
    token_log_probs = token_log_probs * shift_mask

    # 返回序列的平均对数概率
    return token_log_probs.sum(dim=-1) / shift_mask.sum(dim=-1)


# ==========================================
# 2. 简化奖励模型
# ==========================================

class SimpleRewardModel:
    """
    简化的奖励模型。

    在实际 RLHF 中，奖励模型是通过偏好对训练的深度神经网络。
    这里我们使用基于规则的评分函数来模拟奖励模型的行为，
    以便在单机上快速演示 PPO 对齐流程。

    评分规则：
      - 回复长度适中（50-200字）：加分
      - 包含有用的结构化信息（编号、代码块等）：加分
      - 态度友好、有礼貌：加分
      - 回复过短或拒绝回答：扣分
    """

    def __init__(self, tokenizer, backbone_model=None):
        self.tokenizer = tokenizer
        self.backbone_model = backbone_model

        # 如果提供了骨干模型，尝试加载训练好的价值头
        self.value_head = None
        if backbone_model is not None:
            hidden_size = backbone_model.config.hidden_size
            self.value_head = nn.Linear(hidden_size, 1)

            # 尝试加载已训练的价值头参数
            value_head_path = "./output/rm_results/value_head.pt"
            if os.path.exists(value_head_path):
                self.value_head.load_state_dict(
                    torch.load(value_head_path, map_location="cpu")
                )
                print(f"  已加载训练好的价值头参数：{value_head_path}")

    def score(self, prompt, response):
        """
        对 (prompt, response) 对进行评分。
        返回一个标量奖励值。

        如果有训练好的神经网络奖励模型，优先使用；
        否则使用基于规则的评分。
        """
        # 尝试使用神经网络奖励模型
        if self.backbone_model is not None and self.value_head is not None:
            return self._neural_score(prompt, response)
        else:
            return self._rule_based_score(prompt, response)

    def _neural_score(self, prompt, response):
        """使用神经网络奖励模型评分"""
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        enc = self.tokenizer(
            text, truncation=True, max_length=256,
            padding=True, return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self.backbone_model(
                **enc, output_hidden_states=True
            )
            last_hidden = outputs.hidden_states[-1]
            seq_len = enc["attention_mask"].sum(dim=1) - 1
            last_token_hidden = last_hidden[0, seq_len[0]]
            reward = self.value_head(last_token_hidden).item()

        # 将神经网络输出与规则评分结合，提高鲁棒性
        rule_score = self._rule_based_score(prompt, response)
        return 0.5 * reward + 0.5 * rule_score

    def _rule_based_score(self, prompt, response):
        """基于规则的评分函数（模拟奖励模型）"""
        score = 0.0

        # ---- 长度评分 ----
        length = len(response)
        if length < 10:
            score -= 2.0  # 回复太短
        elif length < 30:
            score -= 0.5  # 回复偏短
        elif 50 <= length <= 300:
            score += 1.5  # 长度适中
        elif length > 500:
            score -= 0.5  # 过长可能有冗余

        # ---- 结构化内容评分 ----
        if any(marker in response for marker in ["1.", "2.", "3.", "（1）", "（2）"]):
            score += 1.0  # 有编号列表，结构化好
        if "```" in response:
            score += 1.0  # 包含代码块
        if any(marker in response for marker in ["：\n", "：\r\n", "步骤", "方法"]):
            score += 0.5  # 有结构化说明

        # ---- 态度评分 ----
        positive_words = ["请", "建议", "可以帮助", "以下", "当然", "好的"]
        negative_words = ["不关我事", "自己搜", "随便", "不想", "懒得"]
        for word in positive_words:
            if word in response:
                score += 0.3
        for word in negative_words:
            if word in response:
                score -= 1.0

        # ---- 内容相关性评分 ----
        # 检查回复是否与提问相关（简单关键词匹配）
        prompt_keywords = set(prompt.replace("？", "").replace("？", "").replace("，", "").split())
        response_words = set(response.replace("，", "").replace("。", "").split())
        overlap = len(prompt_keywords & response_words)
        if overlap > 0:
            score += min(overlap * 0.2, 1.0)

        return score


# ==========================================
# 3. PPO 训练器
# ==========================================

class PPOTrainer:
    """
    简化的 PPO 训练器。

    PPO（Proximal Policy Optimization）在 RLHF 中的核心流程：
      1. 策略模型（Actor）生成回复
      2. 奖励模型对回复进行评分
      3. 计算优势函数（Advantage）
      4. 使用裁剪目标函数更新策略模型
      5. 添加 KL 散度惩罚，防止策略偏离参考模型太远

    PPO 裁剪目标：
      L_CLIP = min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)

    其中 r(θ) = π_θ(a|s) / π_ref(a|s) 是新旧策略的概率比。

    总损失 = -L_CLIP + β * KL(π_θ || π_ref)
    """

    def __init__(
        self,
        policy_model,
        reference_model,
        reward_model,
        tokenizer,
        kl_coef=0.1,
        clip_range=0.2,
        learning_rate=1e-6,
    ):
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.kl_coef = kl_coef        # KL 散度惩罚系数
        self.clip_range = clip_range  # PPO 裁剪范围

        self.optimizer = torch.optim.AdamW(
            policy_model.parameters(), lr=learning_rate
        )

        # 训练统计
        self.stats = {
            "rewards": [],
            "kl_divergences": [],
            "policy_losses": [],
            "total_losses": [],
            "response_lengths": [],
        }

    def compute_kl_divergence(self, input_ids, attention_mask):
        """
        计算策略模型与参考模型之间的 KL 散度。
        KL(π_θ || π_ref) = Σ π_θ * log(π_θ / π_ref)

        这里使用近似计算：对每个 token 位置的 KL 散度求平均。
        """
        with torch.no_grad():
            # 策略模型的 logits
            policy_outputs = self.policy_model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            policy_logits = policy_outputs.logits[:, :-1, :]
            policy_log_probs = F.log_softmax(policy_logits, dim=-1)
            policy_probs = torch.softmax(policy_logits, dim=-1)

            # 参考模型的 logits
            ref_outputs = self.reference_model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            ref_logits = ref_outputs.logits[:, :-1, :]
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)

            # 逐 token 计算 KL 散度
            kl_per_token = (
                policy_probs * (policy_log_probs - ref_log_probs)
            ).sum(dim=-1)

            # 排除 padding token
            shift_mask = attention_mask[:, 1:]
            kl_div = (kl_per_token * shift_mask).sum() / shift_mask.sum()

        return kl_div.item()

    def train_step(self, prompts):
        """
        执行一步 PPO 训练。

        步骤：
          1. 用策略模型为每个 prompt 生成回复
          2. 用奖励模型对回复评分
          3. 计算优势函数
          4. 计算 PPO 裁剪损失 + KL 惩罚
          5. 反向传播更新策略模型
        """
        self.policy_model.train()

        batch_rewards = []
        batch_kl = []
        batch_lengths = []
        all_input_ids = []
        all_attention_masks = []
        all_old_log_probs = []

        for prompt in prompts:
            # ---- 步骤1：生成回复 ----
            response, input_len, full_ids = generate_response(
                self.policy_model, self.tokenizer, prompt,
                max_new_tokens=60, temperature=0.8,
            )

            # 准备编码
            input_ids = full_ids.unsqueeze(0)
            attention_mask = torch.ones_like(input_ids)

            # ---- 步骤2：奖励模型评分 ----
            reward = self.reward_model.score(prompt, response)
            batch_rewards.append(reward)
            batch_lengths.append(len(response))

            # ---- 步骤3：计算 KL 散度 ----
            kl_div = self.compute_kl_divergence(input_ids, attention_mask)
            batch_kl.append(kl_div)

            # ---- 步骤4：记录旧策略的对数概率 ----
            with torch.no_grad():
                old_log_prob = compute_log_probs(
                    self.policy_model, input_ids, attention_mask
                )
            all_old_log_probs.append(old_log_prob)
            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)

        # ---- 步骤5：计算优势函数 ----
        # 简化版：使用奖励值本身作为优势（不做 GAE 估计）
        rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float32)
        advantages = rewards_tensor - rewards_tensor.mean()
        advantages = advantages / (advantages.std() + 1e-8)

        # ---- 步骤6：PPO 裁剪更新 ----
        total_policy_loss = 0.0
        for i, (input_ids, att_mask, old_log_p) in enumerate(
            zip(all_input_ids, all_attention_masks, all_old_log_probs)
        ):
            # 新策略的对数概率
            new_log_prob = compute_log_probs(
                self.policy_model, input_ids, att_mask
            )

            # 重要性采样比率
            ratio = torch.exp(new_log_prob - old_log_p)

            # PPO 裁剪目标
            advantage = advantages[i]
            surr1 = ratio * advantage
            surr2 = torch.clamp(
                ratio, 1.0 - self.clip_range, 1.0 + self.clip_range
            ) * advantage

            # 取较小值（保守更新）
            policy_loss = -torch.min(surr1, surr2)
            total_policy_loss += policy_loss

        # 平均策略损失
        avg_policy_loss = total_policy_loss / len(prompts)

        # KL 惩罚项
        avg_kl = sum(batch_kl) / len(batch_kl)
        kl_penalty = self.kl_coef * avg_kl

        # 总损失 = 策略损失 + KL 惩罚
        total_loss = avg_policy_loss + kl_penalty

        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
        self.optimizer.step()

        # 记录统计信息
        self.stats["rewards"].append(sum(batch_rewards) / len(batch_rewards))
        self.stats["kl_divergences"].append(avg_kl)
        self.stats["policy_losses"].append(avg_policy_loss.item())
        self.stats["total_losses"].append(total_loss.item())
        self.stats["response_lengths"].append(
            sum(batch_lengths) / len(batch_lengths)
        )

        return {
            "avg_reward": self.stats["rewards"][-1],
            "avg_kl": avg_kl,
            "policy_loss": avg_policy_loss.item(),
            "total_loss": total_loss.item(),
        }


# ==========================================
# 4. 训练指标可视化
# ==========================================

def plot_training_stats(stats, save_path="output/ppo_training_stats.png"):
    """
    绘制 PPO 训练过程中的各项指标变化。
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # ---- 平均奖励 ----
    ax = axes[0, 0]
    ax.plot(stats["rewards"], "g-o", markersize=4)
    ax.set_title("平均奖励 (Average Reward)")
    ax.set_xlabel("训练步数")
    ax.set_ylabel("奖励值")
    ax.grid(True, alpha=0.3)

    # ---- KL 散度 ----
    ax = axes[0, 1]
    ax.plot(stats["kl_divergences"], "r-o", markersize=4)
    ax.set_title("KL 散度 (KL Divergence)")
    ax.set_xlabel("训练步数")
    ax.set_ylabel("KL 散度")
    ax.grid(True, alpha=0.3)

    # ---- 策略损失 ----
    ax = axes[1, 0]
    ax.plot(stats["policy_losses"], "b-o", markersize=4)
    ax.set_title("策略损失 (Policy Loss)")
    ax.set_xlabel("训练步数")
    ax.set_ylabel("损失值")
    ax.grid(True, alpha=0.3)

    # ---- 回复长度 ----
    ax = axes[1, 1]
    ax.plot(stats["response_lengths"], "m-o", markersize=4)
    ax.set_title("平均回复长度 (Response Length)")
    ax.set_xlabel("训练步数")
    ax.set_ylabel("字符数")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  训练指标图已保存至：{save_path}")


# ==========================================
# 5. 主流程
# ==========================================

def main():
    print("=" * 60)
    print("第10章：RLHF PPO 对齐训练")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  使用设备：{device}")
    print("  注意：这是简化/模拟版本，用于演示 PPO 对齐的核心算法流程。")

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    # ---- 5.1 加载 SFT 模型（策略模型） ----
    print("\n[步骤1] 加载策略模型（SFT 模型）...")

    sft_path = "./output/sft_results/sft_model"
    if os.path.exists(sft_path):
        print(f"  发现已保存的 SFT 模型：{sft_path}")
        policy_model = AutoModelForCausalLM.from_pretrained(
            sft_path, torch_dtype=torch.float32,
        )
        tokenizer = AutoTokenizer.from_pretrained(sft_path)
        print("  已加载 SFT 模型作为策略模型。")
    else:
        print(f"  未找到 SFT 模型，直接加载基础模型 {model_name}")
        print("  （建议先运行 sft_pipeline.py 进行 SFT 训练）")
        policy_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- 5.2 创建参考模型（冻结，不参与训练） ----
    print("\n[步骤2] 创建参考模型（冻结的 SFT 模型副本）...")
    # 参考模型是策略模型的初始副本，用于计算 KL 散度
    reference_model = copy.deepcopy(policy_model)
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False
    print("  参考模型已创建并冻结参数。")

    # ---- 5.3 初始化奖励模型 ----
    print("\n[步骤3] 初始化奖励模型...")

    # 尝试加载之前训练好的奖励模型骨干
    rm_backbone = None
    rm_backbone_path = "./output/rm_results"
    if os.path.exists(os.path.join(rm_backbone_path, "value_head.pt")):
        print(f"  发现训练好的奖励模型参数：{rm_backbone_path}")
        rm_backbone = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32,
        )
    else:
        print("  未找到训练好的奖励模型，将使用基于规则的评分函数。")
        print("  （建议先运行 reward_model_training.py 训练奖励模型）")

    reward_model = SimpleRewardModel(tokenizer, backbone_model=rm_backbone)
    print("  奖励模型初始化完成。")

    # ---- 5.4 对齐前测试 ----
    print("\n[步骤4] PPO 对齐前的模型输出测试...")
    test_prompts = [
        "用 Python 写一个求列表最大值的函数。",
        "解释什么是机器学习。",
        "给我讲一个有趣的故事。",
        "如何提高英语水平？",
    ]

    print("  --- PPO 对齐前的输出 ---")
    before_responses = []
    before_rewards = []
    for prompt in test_prompts:
        response, _, _ = generate_response(
            policy_model, tokenizer, prompt,
            max_new_tokens=80, temperature=0.7,
        )
        reward = reward_model.score(prompt, response)
        before_responses.append(response)
        before_rewards.append(reward)
        print(f"  Q: {prompt}")
        print(f"  A: {response[:80]}...")
        print(f"  奖励分数: {reward:.3f}")
        print()

    # ---- 5.5 配置 PPO 训练 ----
    print("[步骤5] 配置 PPO 训练...")
    print("  超参数：")
    print("    - learning_rate = 1e-6")
    print("    - KL 惩罚系数 (β) = 0.1")
    print("    - PPO 裁剪范围 (ε) = 0.2")
    print("    - 训练步数 = 10")

    ppo_trainer = PPOTrainer(
        policy_model=policy_model,
        reference_model=reference_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        kl_coef=0.1,
        clip_range=0.2,
        learning_rate=1e-6,
    )

    # 用于训练的 prompt 池
    train_prompts_pool = [
        "请解释什么是深度学习。",
        "用 Python 写一个冒泡排序。",
        "如何学习编程？",
        "什么是人工智能？",
        "请推荐几本技术书籍。",
        "如何准备技术面试？",
        "解释什么是 RESTful API。",
        "写一段鼓励正在学习的人的话。",
    ]

    # ---- 5.6 执行 PPO 训练循环 ----
    print("\n[步骤6] 开始 PPO 对齐训练...")
    print("  " + "-" * 60)
    print(f"  {'步数':>4} | {'平均奖励':>8} | {'KL散度':>8} | {'策略损失':>8} | {'总损失':>8}")
    print("  " + "-" * 60)

    num_steps = 10
    for step in range(num_steps):
        # 每步随机选择一批 prompt
        step_prompts = random_sample(train_prompts_pool, k=4)

        # 执行一步 PPO 训练
        step_stats = ppo_trainer.train_step(step_prompts)

        print(f"  {step + 1:>4} | "
              f"{step_stats['avg_reward']:>8.3f} | "
              f"{step_stats['avg_kl']:>8.4f} | "
              f"{step_stats['policy_loss']:>8.4f} | "
              f"{step_stats['total_loss']:>8.4f}")

    print("  " + "-" * 60)

    # ---- 5.7 打印训练指标汇总 ----
    stats = ppo_trainer.stats

    print("\n[步骤7] 训练指标汇总：")
    print(f"  奖励变化：{stats['rewards'][0]:.3f} → {stats['rewards'][-1]:.3f} "
          f"(变化: {stats['rewards'][-1] - stats['rewards'][0]:+.3f})")
    print(f"  KL 散度变化：{stats['kl_divergences'][0]:.4f} → {stats['kl_divergences'][-1]:.4f}")
    print(f"  回复长度变化：{stats['response_lengths'][0]:.1f} → {stats['response_lengths'][-1]:.1f}")

    # ---- 5.8 可视化训练过程 ----
    print("\n[步骤8] 可视化训练过程...")
    plot_training_stats(stats, save_path="output/ppo_training_stats.png")

    # ---- 5.9 对齐后测试 ----
    print("\n[步骤9] PPO 对齐后的模型输出测试...")
    print("  --- PPO 对齐后的输出 ---")

    policy_model.eval()
    after_responses = []
    after_rewards = []
    for prompt in test_prompts:
        response, _, _ = generate_response(
            policy_model, tokenizer, prompt,
            max_new_tokens=80, temperature=0.7,
        )
        reward = reward_model.score(prompt, response)
        after_responses.append(response)
        after_rewards.append(reward)
        print(f"  Q: {prompt}")
        print(f"  A: {response[:80]}...")
        print(f"  奖励分数: {reward:.3f}")
        print()

    # ---- 5.10 前后对比总结 ----
    print("=" * 60)
    print("PPO 对齐前后对比总结：")
    print("=" * 60)

    for i, prompt in enumerate(test_prompts):
        print(f"\n  提示：{prompt}")
        print(f"  对齐前 [{before_rewards[i]:.3f}]：{before_responses[i][:60]}...")
        print(f"  对齐后 [{after_rewards[i]:.3f}]：{after_responses[i][:60]}...")
        print(f"  奖励变化：{after_rewards[i] - before_rewards[i]:+.3f}")

    avg_before = sum(before_rewards) / len(before_rewards)
    avg_after = sum(after_rewards) / len(after_rewards)
    print(f"\n  平均奖励：对齐前 {avg_before:.3f} → 对齐后 {avg_after:.3f} "
          f"({avg_after - avg_before:+.3f})")

    # ---- 5.11 保存对齐后的模型 ----
    print("\n[步骤10] 保存 PPO 对齐后的模型...")
    save_dir = "./output/ppo_results"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "aligned_model")
    policy_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"  对齐后的模型已保存至：{save_path}")

    # 保存训练统计
    stats_path = os.path.join(save_dir, "ppo_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"  训练统计已保存至：{stats_path}")

    # ---- 总结 ----
    print("\n" + "=" * 60)
    print("RLHF 三阶段流水线全部完成！")
    print("=" * 60)
    print("\n  阶段回顾：")
    print("  [1] SFT（监督微调）    → output/sft_results/sft_model")
    print("  [2] RM（奖励模型训练）  → output/rm_results/value_head.pt")
    print("  [3] PPO（对齐训练）     → output/ppo_results/aligned_model")
    print("\n  核心概念总结：")
    print("  - SFT：用指令数据让模型学会基本的指令跟随能力")
    print("  - RM：学习人类偏好，给好回复高分、差回复低分")
    print("  - PPO：用奖励模型的反馈优化策略，同时用 KL 惩罚保持稳定")
    print("=" * 60)


def random_sample(lst, k):
    """从列表中随机采样 k 个元素"""
    import random
    return random.sample(lst, min(k, len(lst)))


if __name__ == "__main__":
    main()
