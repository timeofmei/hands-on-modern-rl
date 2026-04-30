import re
import torch


# [A] 组采样：每个 prompt 生成 group_size 个回答
def sample_groups(model, tokenizer, prompts, group_size=8, max_new_tokens=256):
    expanded_prompts = [
        prompt
        for prompt in prompts
        for _ in range(group_size)
    ]
    inputs = tokenizer(expanded_prompts, padding=True, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.8,
            max_new_tokens=max_new_tokens,
        )

    responses = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    group_ids = torch.arange(len(prompts), device=model.device).repeat_interleave(group_size)
    return responses, group_ids


# [B] 规则奖励：数学答案正确、格式规范就给分
def rule_reward(response, ground_truth):
    reward = 0.0
    boxed = re.search(r"\\boxed\{([^}]+)\}", response)

    if boxed:
        reward += 0.5
        if boxed.group(1).strip() == str(ground_truth).strip():
            reward += 1.0

    return reward


def score_responses(responses, ground_truths, group_size=8, device="cpu"):
    rewards = []
    for i, response in enumerate(responses):
        prompt_id = i // group_size
        rewards.append(rule_reward(response, ground_truths[prompt_id]))
    return torch.tensor(rewards, dtype=torch.float32, device=device)


# [C] 组内优势：用同题目的回答均值替代 Critic 基线
def group_advantages(rewards, group_size=8, eps=1e-8):
    grouped_rewards = rewards.view(-1, group_size)
    group_mean = grouped_rewards.mean(dim=1, keepdim=True)
    group_std = grouped_rewards.std(dim=1, keepdim=True)

    advantages = (grouped_rewards - group_mean) / (group_std + eps)
    advantages = torch.where(
        group_std < eps,
        torch.zeros_like(advantages),
        advantages,
    )
    return advantages.reshape(-1)


# [D] 计算回答序列 log probability
def sequence_logprob(model, input_ids, attention_mask, labels):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    target_ids = input_ids[:, 1:]
    target_mask = labels[:, 1:].ne(-100)

    token_logprobs = logits.log_softmax(dim=-1)
    picked_logprobs = token_logprobs.gather(
        dim=-1,
        index=target_ids.unsqueeze(-1),
    ).squeeze(-1)

    response_logprobs = picked_logprobs * target_mask
    return response_logprobs.sum(dim=-1)


# [E] GRPO 更新：PPO-style ratio + clip，但优势来自组内比较
def grpo_loss(policy_model, ref_model, batch, old_logprobs, advantages,
              clip_eps=0.2, kl_coef=0.04):
    new_logprobs = sequence_logprob(
        policy_model,
        batch["input_ids"],
        batch["attention_mask"],
        batch["labels"],
    )

    with torch.no_grad():
        ref_logprobs = sequence_logprob(
            ref_model,
            batch["input_ids"],
            batch["attention_mask"],
            batch["labels"],
        )

    ratio = torch.exp(new_logprobs - old_logprobs)
    surr1 = ratio * advantages
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    surr2 = clipped_ratio * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # [F] KL 惩罚：防止 Policy 离 Reference 太远
    log_ratio_ref = ref_logprobs - new_logprobs
    approx_kl = (torch.exp(log_ratio_ref) - log_ratio_ref - 1.0).mean()
    loss = policy_loss + kl_coef * approx_kl

    metrics = {
        "loss": loss.detach(),
        "policy_loss": policy_loss.detach(),
        "approx_kl": approx_kl.detach(),
    }
    return loss, metrics


# [G] 训练步骤：采样、打分、组内归一化、再反向传播
def train_step(policy_model, ref_model, optimizer, tokenizer, prompts, ground_truths,
               group_size=8):
    responses, _ = sample_groups(policy_model, tokenizer, prompts, group_size)
    rewards = score_responses(responses, ground_truths, group_size, policy_model.device)
    advantages = group_advantages(rewards, group_size)

    batch = tokenizer(responses, padding=True, return_tensors="pt")
    batch = {key: value.to(policy_model.device) for key, value in batch.items()}
    batch["labels"] = batch["input_ids"].clone()

    with torch.no_grad():
        old_logprobs = sequence_logprob(
            policy_model,
            batch["input_ids"],
            batch["attention_mask"],
            batch["labels"],
        )

    loss, metrics = grpo_loss(policy_model, ref_model, batch, old_logprobs, advantages)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return metrics


# [H] GRPO 训练循环：每轮都在线生成新回答
def train_grpo(policy_model, ref_model, optimizer, tokenizer, dataloader):
    ref_model.eval()
    for prompts, ground_truths in dataloader:
        metrics = train_step(
            policy_model,
            ref_model,
            optimizer,
            tokenizer,
            prompts,
            ground_truths,
        )
        print("loss=", float(metrics["loss"]), "kl=", float(metrics["approx_kl"]))
