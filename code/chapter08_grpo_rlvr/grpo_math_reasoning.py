"""
第8章：GRPO 数学推理训练 —— GSM8K 风格
==========================================================

本脚本使用 GRPO 算法训练一个小型语言模型（Qwen2.5-0.5B-Instruct）
完成数学推理任务，展示 RLVR (Reinforcement Learning with Verifiable Rewards) 的完整流程。

核心流程：
  1. 构造 GSM8K 风格的算术应用题数据集（20 题）
  2. 加载 Qwen2.5-0.5B-Instruct 模型
  3. 对每道题生成 group_size=4 个回复
  4. 用规则奖励函数计算每个回复的奖励
  5. GRPO 组内归一化得到优势值
  6. 用优势值更新策略（策略梯度）
  7. 训练 3 个 epoch，追踪准确率、平均奖励、回复长度

运行方式：
  pip install -r requirements.txt
  python grpo_math_reasoning.py

注意：
  - 本脚本可在 CPU 上运行，但速度较慢
  - 推荐使用 GPU（约需要 2~4 GB 显存）
  - 如果没有 GPU，可以使用更小的模型或减少数据量
"""

import re
import copy
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


# ==========================================
# 第一部分：GSM8K 风格数学数据集
# ==========================================
# 20 道小学算术应用题，覆盖加减乘除
# 每道题包含 question（题目）和 answer（标准答案）

math_dataset = [
    {
        "question": "小明有 15 个苹果，小红给了他 27 个，小明现在有多少个苹果？",
        "answer": "42",
    },
    {
        "question": "一个班级有 36 个学生，平均分成 6 组，每组有多少个学生？",
        "answer": "6",
    },
    {
        "question": "张三每天跑步 5 公里，他一周（7天）一共跑了多少公里？",
        "answer": "35",
    },
    {
        "question": "一本书有 240 页，李四已经读了 85 页，还剩多少页没读？",
        "answer": "155",
    },
    {
        "question": "水果店有苹果 48 个，橘子 36 个，香蕉 25 个，一共有多少个水果？",
        "answer": "109",
    },
    {
        "question": "一箱饮料有 24 瓶，8 箱饮料一共有多少瓶？",
        "answer": "192",
    },
    {
        "question": "小红有 100 元钱，买了一本书花了 32 元，买了一支笔花了 8 元，还剩多少钱？",
        "answer": "60",
    },
    {
        "question": "一个长方形的长是 12 厘米，宽是 8 厘米，它的面积是多少平方厘米？",
        "answer": "96",
    },
    {
        "question": "学校买了 5 盒铅笔，每盒 12 支，一共有多少支铅笔？",
        "answer": "60",
    },
    {
        "question": "王师傅每小时加工 15 个零件，工作了 8 小时，一共加工了多少个零件？",
        "answer": "120",
    },
    {
        "question": "一列火车每小时行驶 120 公里，行驶了 3 小时，一共行驶了多少公里？",
        "answer": "360",
    },
    {
        "question": "农场里有鸡 45 只，鸭比鸡多 18 只，鸭有多少只？",
        "answer": "63",
    },
    {
        "question": "一个三角形的底边长 10 厘米，高 6 厘米，面积是多少平方厘米？",
        "answer": "30",
    },
    {
        "question": "妈妈买了 3 千克苹果，每千克 8 元，又买了 2 千克橘子，每千克 5 元，一共花了多少钱？",
        "answer": "34",
    },
    {
        "question": "学校图书馆有故事书 280 本，科技书比故事书少 95 本，科技书有多少本？",
        "answer": "185",
    },
    {
        "question": "小明从家到学校走了 15 分钟，每分钟走 60 米，家到学校有多远？",
        "answer": "900",
    },
    {
        "question": "一个游泳池长 50 米，宽 25 米，它的周长是多少米？",
        "answer": "150",
    },
    {
        "question": "有 4 个小朋友分 72 颗糖，平均每人分到多少颗？",
        "answer": "18",
    },
    {
        "question": "工厂生产了 500 个零件，合格的有 468 个，不合格的有多少个？",
        "answer": "32",
    },
    {
        "question": "一桶油重 18 千克，用掉了三分之一，还剩多少千克？",
        "answer": "12",
    },
]


# ==========================================
# 第二部分：规则奖励函数
# ==========================================
def extract_answer_from_response(response):
    """
    从模型回复中提取最终答案

    优先级：
      1. \\boxed{...} 格式
      2. "答案是/为：" 后面的数字
      3. 回复中最后一个数字（兜底）

    参数：
        response: 模型生成的文本
    返回：
        str 或 None: 提取到的答案字符串
    """
    # 方法1：\boxed{} 格式
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', response)
    if boxed_match:
        return boxed_match.group(1).strip()

    # 方法2：中文答案标记
    cn_match = re.search(r'答案[是为：:]\s*([+-]?\d+\.?\d*)', response)
    if cn_match:
        return cn_match.group(1).strip()

    # 方法3：英文答案标记
    en_match = re.search(r'[Tt]he answer is\s*([+-]?\d+\.?\d*)', response)
    if en_match:
        return en_match.group(1).strip()

    # 方法4：最后一个数字（兜底）
    numbers = re.findall(r'([+-]?\d+\.?\d*)', response)
    if numbers:
        return numbers[-1]

    return None


def compute_reward(response, ground_truth):
    """
    计算回复的奖励值

    这是一个简化的奖励函数，主要关注答案正确性：
      - 答案正确：奖励 = 1.0
      - 答案错误：奖励 = 0.0
      - 格式加分：如果使用了步骤式推理格式，额外 +0.1
      - 回复格式规范（有步骤标记），最多 +0.1

    最终奖励上限为 1.0

    参数：
        response: 模型生成的文本
        ground_truth: 标准答案字符串
    返回：
        float: 奖励值 [0.0, 1.0]
    """
    reward = 0.0

    # 提取答案并判断正确性
    extracted = extract_answer_from_response(response)
    if extracted is not None:
        try:
            if abs(float(extracted) - float(ground_truth)) < 1e-6:
                reward = 1.0
        except (ValueError, TypeError):
            if str(extracted).strip() == str(ground_truth).strip():
                reward = 1.0

    # 格式加分（仅当答案正确时才考虑，避免奖励错误但有格式的回复）
    if reward > 0:
        has_steps = bool(re.search(r'步骤|第\d+步|Step|首先|然后|接着|最后', response))
        if has_steps:
            reward = min(1.0, reward + 0.1)

    return reward


# ==========================================
# 第三部分：GRPO 组内归一化
# ==========================================
def compute_grpo_advantages(rewards):
    """
    GRPO 核心：对一组奖励进行组内归一化

    公式：
        advantage_i = (reward_i - mean) / (std + eps)

    参数：
        rewards: list[float]，一组回复的奖励值
    返回：
        numpy 数组：归一化后的优势值
    """
    rewards_arr = np.array(rewards, dtype=np.float64)
    mean_r = rewards_arr.mean()
    std_r = rewards_arr.std() + 1e-8
    advantages = (rewards_arr - mean_r) / std_r
    return advantages


# ==========================================
# 第四部分：模型生成回复
# ==========================================
def generate_responses(model, tokenizer, prompt, num_responses=4, max_new_tokens=200):
    """
    让模型为同一个问题生成多个不同的回复

    使用 do_sample=True + temperature > 0 来产生多样性
    每个回复使用不同的随机种子，确保多样性

    参数：
        model: 语言模型
        tokenizer: 分词器
        prompt: 问题文本
        num_responses: 生成的回复数量（group_size）
        max_new_tokens: 最大生成 token 数
    返回：
        list[str]: 生成的回复列表
    """
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    responses = []
    for _ in range(num_responses):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        # 解码：只取生成部分（跳过输入 prompt）
        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[-1]:],
            skip_special_tokens=True,
        )
        responses.append(response)

    return responses


# ==========================================
# 第五部分：GRPO 策略梯度更新
# ==========================================
def grpo_policy_update(model, tokenizer, optimizer, prompt, responses, advantages):
    """
    使用 GRPO 策略梯度更新模型参数

    核心公式：
        loss = -mean(advantage_i * log_prob_i)

    其中 log_prob_i 是模型生成第 i 个回复的对数概率。
    为简化实现，这里使用 token 级别的平均对数概率作为近似。

    参数：
        model: 语言模型
        tokenizer: 分词器
        optimizer: 优化器
        prompt: 原始问题
        responses: 生成的回复列表
        advantages: GRPO 归一化后的优势值
    返回：
        float: 本轮损失值
    """
    model.train()

    # 构建 prompt 的 token ids
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(model.device)
    prompt_len = prompt_ids.shape[-1]

    total_loss = 0.0
    num_valid = 0

    for response, advantage in zip(responses, advantages):
        if advantage <= 0:
            continue  # 跳过负优势的回复，减少计算量

        # 将 prompt + response 拼接
        response_ids = tokenizer.encode(response, return_tensors="pt").to(model.device)
        # 去掉 response_ids 中可能的 BOS token
        if response_ids[0, 0] == tokenizer.bos_token_id:
            response_ids = response_ids[:, 1:]

        full_ids = torch.cat([prompt_ids, response_ids], dim=-1)

        # 前向传播获取 logits
        with torch.cuda.amp.autocast(enabled=False):
            outputs = model(full_ids)
            logits = outputs.logits

        # 计算生成部分的 log_prob（只取 response 部分的 token）
        # logits[:, :-1, :] 对应位置 t 的预测，full_ids[:, 1:] 对应位置 t+1 的真实 token
        response_logits = logits[0, prompt_len - 1:-1, :]  # response 部分的 logits
        response_tokens = full_ids[0, prompt_len:]          # response 部分的 token ids

        if response_tokens.shape[0] == 0:
            continue

        # 计算每个 token 的对数概率
        log_probs = torch.nn.functional.log_softmax(response_logits, dim=-1)
        token_log_probs = log_probs.gather(1, response_tokens.unsqueeze(1)).squeeze(1)

        # 平均对数概率（按 token 数归一化，避免长回复的偏差）
        avg_log_prob = token_log_probs.mean()

        # 策略梯度损失：-advantage * log_prob
        loss = -advantage * avg_log_prob
        total_loss += loss
        num_valid += 1

    if num_valid == 0:
        return 0.0

    # 平均损失
    avg_loss = total_loss / num_valid

    # 反向传播和参数更新
    optimizer.zero_grad()
    avg_loss.backward()
    # 梯度裁剪，防止梯度爆炸
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return avg_loss.item()


# ==========================================
# 第六部分：完整训练流程
# ==========================================
def train():
    """
    GRPO 数学推理训练的完整流程

    超参数：
        - model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        - group_size = 4：每个问题生成 4 个回复
        - num_epochs = 3：训练轮次
        - learning_rate = 1e-5：学习率
        - max_new_tokens = 200：最大生成 token 数
    """
    # ---------- 超参数 ----------
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    group_size = 4       # 每个问题的采样数（GRPO 的 group）
    num_epochs = 3       # 训练轮次
    learning_rate = 1e-5 # 学习率
    max_new_tokens = 200 # 最大生成长度

    print("=" * 70)
    print("  GRPO 数学推理训练（GSM8K 风格）")
    print("=" * 70)
    print(f"  模型: {model_name}")
    print(f"  训练数据: {len(math_dataset)} 道数学题")
    print(f"  采样数 (group_size): {group_size}")
    print(f"  训练轮次: {num_epochs}")
    print(f"  学习率: {learning_rate}")
    print(f"  最大生成长度: {max_new_tokens}")
    print("=" * 70)

    # ---------- 加载模型 ----------
    print("\n正在加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # 判断设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  使用设备: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=device,
    )
    model.eval()  # 初始设为评估模式

    # 创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # ---------- 训练前测试 ----------
    print("\n" + "=" * 70)
    print("  训练前模型表现（随机抽取 3 道题测试）")
    print("=" * 70)
    test_indices = [0, 7, 14]  # 抽取第 1、8、15 题测试
    pre_train_results = evaluate_on_samples(model, tokenizer, math_dataset, test_indices)

    # ---------- 训练循环 ----------
    print("\n" + "=" * 70)
    print("  开始 GRPO 训练")
    print("=" * 70)

    # 记录训练指标
    epoch_metrics = []

    for epoch in range(num_epochs):
        print(f"\n{'#' * 70}")
        print(f"  Epoch {epoch + 1}/{num_epochs}")
        print(f"{'#' * 70}")

        epoch_total_rewards = []
        epoch_accuracies = []
        epoch_lengths = []
        epoch_losses = []

        for idx, problem in enumerate(math_dataset):
            question = problem["question"]
            ground_truth = problem["answer"]

            # Step 1：为当前问题生成一组回复
            responses = generate_responses(
                model, tokenizer, question,
                num_responses=group_size,
                max_new_tokens=max_new_tokens,
            )

            # Step 2：计算每个回复的奖励
            rewards = []
            for resp in responses:
                r = compute_reward(resp, ground_truth)
                rewards.append(r)

            # Step 3：GRPO 组内归一化
            advantages = compute_grpo_advantages(rewards)

            # Step 4：策略梯度更新
            loss = grpo_policy_update(
                model, tokenizer, optimizer,
                question, responses, advantages,
            )

            # 记录指标
            avg_reward = np.mean(rewards)
            accuracy = np.mean([1.0 if r >= 1.0 else 0.0 for r in rewards])
            avg_length = np.mean([len(r) for r in responses])

            epoch_total_rewards.append(avg_reward)
            epoch_accuracies.append(accuracy)
            epoch_lengths.append(avg_length)
            epoch_losses.append(loss)

            # 每 5 题打印一次进度
            if (idx + 1) % 5 == 0:
                recent_acc = np.mean(epoch_accuracies[-5:])
                recent_reward = np.mean(epoch_total_rewards[-5:])
                print(
                    f"  题目 {idx + 1:2d}/{len(math_dataset)} | "
                    f"近5题准确率: {recent_acc:.1%} | "
                    f"平均奖励: {recent_reward:.3f} | "
                    f"损失: {loss:.4f}"
                )

        # Epoch 结束，汇总指标
        epoch_metrics.append({
            "epoch": epoch + 1,
            "avg_reward": np.mean(epoch_total_rewards),
            "accuracy": np.mean(epoch_accuracies),
            "avg_length": np.mean(epoch_lengths),
            "avg_loss": np.mean([l for l in epoch_losses if l != 0.0]),
        })

        print(f"\n  Epoch {epoch + 1} 汇总:")
        print(f"    平均奖励:   {epoch_metrics[-1]['avg_reward']:.4f}")
        print(f"    平均准确率: {epoch_metrics[-1]['accuracy']:.2%}")
        print(f"    平均回复长度: {epoch_metrics[-1]['avg_length']:.0f} 字符")
        print(f"    平均损失:   {epoch_metrics[-1]['avg_loss']:.4f}")

    # ---------- 训练后测试 ----------
    print("\n" + "=" * 70)
    print("  训练后模型表现（同样 3 道题测试）")
    print("=" * 70)
    post_train_results = evaluate_on_samples(model, tokenizer, math_dataset, test_indices)

    # ---------- 训练前后对比 ----------
    print("\n" + "=" * 70)
    print("  训练前后对比")
    print("=" * 70)
    for i, test_idx in enumerate(test_indices):
        problem = math_dataset[test_idx]
        pre = pre_train_results[i]
        post = post_train_results[i]
        print(f"\n  题目: {problem['question'][:40]}...")
        print(f"  标准答案: {problem['answer']}")
        print(f"  训练前回答: {pre['response'][:60]}... → 奖励: {pre['reward']:.2f}")
        print(f"  训练后回答: {post['response'][:60]}... → 奖励: {post['reward']:.2f}")

    # ---------- 训练指标汇总 ----------
    print("\n" + "=" * 70)
    print("  训练指标汇总")
    print("=" * 70)
    print()
    print(f"  {'Epoch':>5s}  {'平均奖励':>10s}  {'准确率':>10s}  {'回复长度':>10s}  {'损失':>10s}")
    print(f"  {'─────':>5s}  {'──────────':>10s}  {'──────────':>10s}  {'──────────':>10s}  {'──────────':>10s}")
    for m in epoch_metrics:
        print(f"  {m['epoch']:>5d}  {m['avg_reward']:>10.4f}  "
              f"{m['accuracy']:>10.2%}  {m['avg_length']:>10.0f}  "
              f"{m['avg_loss']:>10.4f}")

    # ---------- 最终总结 ----------
    print("\n" + "=" * 70)
    print("  GRPO 数学推理训练总结")
    print("=" * 70)
    print(f"""
  本实验展示了 GRPO + RLVR 在数学推理任务上的完整流程：

  1. 数据准备：
     - {len(math_dataset)} 道 GSM8K 风格的算术应用题
     - 每道题有明确的标准答案，可用于规则验证

  2. GRPO 训练：
     - 每道题采样 {group_size} 个回复
     - 用规则奖励函数验证答案正确性
     - 组内归一化计算优势值，无需 Critic 网络
     - 策略梯度更新模型参数

  3. 关键观察：
     - 训练前：模型可能随机输出，准确率低
     - 训练后：模型逐渐学会推理格式和正确计算
     - 平均奖励和准确率应随训练逐步提升

  4. 与 DPO 的对比：
     - DPO 需要成对偏好数据（chosen/rejected）
     - GRPO 只需要可验证的奖励信号
     - 对于有标准答案的任务（数学、编程），GRPO + RLVR 更加自然

  5. 扩展方向：
     - 增加训练数据量（完整 GSM8K 数据集）
     - 使用更大的模型（Qwen2.5-1.5B / 7B）
     - 增加 group_size（8~16）获得更稳定的优势估计
     - 使用 DeepSeek-R1 风格的多阶段训练
    """)


# ==========================================
# 第七部分：评估函数
# ==========================================
def evaluate_on_samples(model, tokenizer, dataset, indices):
    """
    在指定样本上评估模型表现（贪心解码）

    参数：
        model: 语言模型
        tokenizer: 分词器
        dataset: 数据集列表
        indices: 要评估的样本索引
    返回：
        list[dict]: 每个样本的评估结果
    """
    model.eval()
    results = []

    for idx in indices:
        problem = dataset[idx]
        # 使用贪心解码生成回复（确定性输出）
        messages = [{"role": "user", "content": problem["question"]}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
            )

        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[-1]:],
            skip_special_tokens=True,
        )

        reward = compute_reward(response, problem["answer"])
        extracted = extract_answer_from_response(response)

        result = {
            "question": problem["question"],
            "ground_truth": problem["answer"],
            "response": response,
            "extracted_answer": extracted,
            "reward": reward,
            "correct": str(extracted) == str(problem["answer"]) if extracted else False,
        }
        results.append(result)

        # 打印详细信息
        print(f"\n  题目: {problem['question'][:50]}...")
        print(f"  标准答案: {problem['answer']}")
        print(f"  模型回复: {response[:80]}{'...' if len(response) > 80 else ''}")
        print(f"  提取答案: {extracted}")
        print(f"  是否正确: {'是' if result['correct'] else '否'} | 奖励: {reward:.2f}")

    return results


# ==========================================
# 程序入口
# ==========================================
if __name__ == "__main__":
    train()
