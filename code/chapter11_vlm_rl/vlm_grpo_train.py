"""
第11章：VLM GRPO 训练模拟演示
==========================================================

本脚本模拟 VLM（视觉语言模型）上的 GRPO 训练过程：
  1. 构造几何图形计数任务的训练数据
  2. 模拟 GRPO 训练循环（组采样 → 奖励计算 → 归一化 → 策略更新）
  3. 展示 VLM GRPO 与纯文本 GRPO 的关键区别
  4. 跟踪训练指标：准确率、平均奖励、响应质量
  5. 训练前后对比

重要说明：
  本脚本是**简化演示版**，使用模拟数据而非真实 VLM 模型。
  完整的 VLM GRPO 训练需要：
    - GPU 显存 >= 40GB（如 A100）
    - transformers 库中的 VLM 模型（如 Qwen2-VL、LLaVA）
    - 图像编码器 + 视觉 token 处理
    - 分布式训练框架（如 DeepSpeed）

  本脚本的目的是帮助理解 VLM GRPO 的训练流程和关键概念。

运行方式：
  python vlm_grpo_train.py
"""

import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt

# 创建输出目录
os.makedirs("output", exist_ok=True)

# 设置中文字体，确保图表标题和标签正常显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 第一部分：VLM GRPO 与纯文本 GRPO 的区别
# ==========================================
# VLM GRPO 的独特之处在于：
#   1. 输入包含图像 → 需要图像编码器（Vision Encoder）
#   2. 图像被编码为视觉 token → 与文本 token 拼接后送入 LLM
#   3. 奖励函数需要考虑视觉定位能力
#   4. 训练时需要同时处理图像和文本数据

def print_vlm_grpo_overview():
    """
    打印 VLM GRPO 与纯文本 GRPO 的架构对比
    """
    print("=" * 70)
    print("  VLM GRPO vs 纯文本 GRPO — 架构对比")
    print("=" * 70)
    print()
    print("  纯文本 GRPO 流程：")
    print("  ┌──────────┐    ┌──────────┐    ┌──────────┐")
    print("  │ 文本提示  │ → │ LLM 采样  │ → │ 文本奖励  │")
    print("  │ (prompt) │    │ (group)  │    │ (score)  │")
    print("  └──────────┘    └──────────┘    └──────────┘")
    print()
    print("  VLM GRPO 流程：")
    print("  ┌──────┐ ┌──────────┐    ┌──────────────┐    ┌──────────────┐")
    print("  │ 图像 │→│ 视觉编码  │→│ VLM 采样      │→│ 多模态奖励    │")
    print("  │(img) │ │(Encoder) │    │(vision+text) │    │(correct+vis) │")
    print("  └──────┘ └──────────┘    └──────────────┘    └──────────────┘")
    print("  ┌──────────┐     ↑")
    print("  │ 文本提示  │ ────┘")
    print("  │ (prompt) │")
    print("  └──────────┘")
    print()
    print("  关键区别：")
    print("    1. 输入：纯文本 prompt → (图像, 文本 prompt) 对")
    print("    2. 模型：LLM → VLM（LLM + Vision Encoder + 投影层）")
    print("    3. Token：仅文本 token → 文本 token + 视觉 token")
    print("    4. 奖励：仅文本质量 → 文本质量 + 视觉定位能力")
    print("    5. 显存：较小 → 显著增加（图像编码 + 更长序列）")
    print()


# ==========================================
# 第二部分：模拟数据与响应生成
# ==========================================
# 由于完整 VLM 推理需要大量 GPU 资源，
# 这里使用预设模板模拟不同质量的模型响应

# 几何图形计数任务的标准提示词
STANDARD_PROMPT = "请数一下图片中有多少个三角形、圆形和正方形"

# 模拟不同质量水平的响应模板
# 每个模板包含一个函数，根据 ground_truth 生成响应
def generate_correct_response(gt):
    """生成完全正确的响应（含推理过程）"""
    shapes_desc = []
    for shape, count in gt.items():
        if count > 0:
            shapes_desc.append(f"{shape}{count}个")
        else:
            shapes_desc.append(f"{shape}0个（没有{shape}）")
    return (
        f"让我仔细观察图片中的形状。\n"
        f"首先，分析三角形：我数到了{gt['三角形']}个三角形。\n"
        f"然后，分析圆形：图片中有{gt['圆形']}个圆形。\n"
        f"接着，分析正方形：有{gt['正方形']}个正方形。\n"
        f"所以，{shapes_desc[0]}，{shapes_desc[1]}，{shapes_desc[2]}。\n"
        f"答：三角形{gt['三角形']}个，圆形{gt['圆形']}个，正方形{gt['正方形']}个。"
    )


def generate_short_correct_response(gt):
    """生成正确但简短的响应"""
    return f"三角形{gt['三角形']}个，圆形{gt['圆形']}个，正方形{gt['正方形']}个。"


def generate_wrong_response(gt):
    """生成有错误的响应（随机修改一个数字）"""
    wrong_gt = dict(gt)
    shape_to_modify = random.choice(list(gt.keys()))
    # 随机增加或减少 1~2
    delta = random.choice([-2, -1, 1, 2])
    wrong_gt[shape_to_modify] = max(0, wrong_gt[shape_to_modify] + delta)
    return (
        f"我来看一下图片。\n"
        f"三角形{wrong_gt['三角形']}个，圆形{wrong_gt['圆形']}个，正方形{wrong_gt['正方形']}个。\n"
        f"总共{sum(wrong_gt.values())}个形状。"
    )


def generate_partially_correct_response(gt):
    """生成部分正确的响应（只有一部分形状数对了）"""
    shapes = list(gt.keys())
    # 选择一个形状保持正确，其他随机修改
    correct_shape = random.choice(shapes)
    wrong_gt = {}
    for s in shapes:
        if s == correct_shape:
            wrong_gt[s] = gt[s]
        else:
            wrong_gt[s] = max(0, gt[s] + random.choice([-1, 1]))
    return f"分析图片后，三角形{wrong_gt['三角形']}个，圆形{wrong_gt['圆形']}个，正方形{wrong_gt['正方形']}个。"


def generate_low_quality_response(gt):
    """生成低质量响应（无格式、无推理）"""
    shapes = list(gt.keys())
    random.shuffle(shapes)
    nums = [max(0, gt[s] + random.choice([-2, -1, 0, 1, 2])) for s in shapes]
    return f"大概有{shapes[0]}{nums[0]}个{shapes[1]}{nums[1]}个{shapes[2]}{nums[2]}个吧"


# 响应生成器列表，按质量从高到低排列
RESPONSE_GENERATORS = [
    generate_correct_response,           # 质量：高
    generate_short_correct_response,     # 质量：中高（正确但无推理）
    generate_wrong_response,             # 质量：中低（有推理但错误）
    generate_partially_correct_response, # 质量：低（部分正确）
    generate_low_quality_response,       # 质量：很低
]


# ==========================================
# 第三部分：奖励函数（简化版，复用 multi_modal_reward.py 的逻辑）
# ==========================================
def extract_numbers(response, shape_names):
    """从响应中提取每种形状对应的数字"""
    import re
    extracted = {}
    for shape_name in shape_names:
        patterns = [
            rf'{shape_name}[^0-9]*?(\d+)',
            rf'(\d+)\s*个\s*{shape_name}',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, response)
            if matches:
                extracted[shape_name] = int(matches[-1])
                break
    return extracted


def simple_reward(response, ground_truth):
    """
    简化版奖励函数：计算响应的综合得分

    组成部分：
      - 正确性（0.0 或 1.0）：答案完全正确
      - 推理加分（0.0~0.3）：包含推理步骤
      - 格式加分（0.0~0.2）：格式规范

    总分范围：0.0 ~ 1.5
    """
    import re

    score = 0.0

    # 正确性
    extracted = extract_numbers(response, ['三角形', '圆形', '正方形'])
    all_correct = True
    for shape, expected in ground_truth.items():
        if extracted.get(shape, -1) != expected:
            all_correct = False
            break
    if all_correct and len(extracted) == len(ground_truth):
        score += 1.0

    # 推理加分
    step_keywords = ['首先', '然后', '接着', '最后', '分析', '观察', '因为', '所以']
    step_count = sum(1 for kw in step_keywords if kw in response)
    score += min(step_count * 0.06, 0.3)

    # 格式加分
    if all(s in response for s in ['三角形', '圆形', '正方形']):
        score += 0.1
    if re.search(r'答[：:]', response):
        score += 0.1

    return score


# ==========================================
# 第四部分：GRPO 训练数据准备
# ==========================================
def create_training_samples(num_samples=20, seed=42):
    """
    创建模拟的训练样本

    每个样本包含：
      - 图像标识（模拟）
      - 文本提示
      - ground_truth（每种形状的正确数量）

    参数：
        num_samples: 样本数量
        seed: 随机种子
    返回：
        list[dict]: 训练样本列表
    """
    random.seed(seed)
    samples = []

    for i in range(num_samples):
        # 随机生成每种形状的数量（0~5）
        gt = {
            '三角形': random.randint(0, 5),
            '圆形': random.randint(0, 5),
            '正方形': random.randint(0, 5),
        }

        samples.append({
            'sample_id': i,
            'prompt': STANDARD_PROMPT,
            'ground_truth': gt,
        })

    return samples


# ==========================================
# 第五部分：模拟 GRPO 训练循环
# ==========================================
def simulate_grpo_training(samples, group_size=4, num_epochs=5,
                           initial_quality=0.3, seed=42):
    """
    模拟 VLM GRPO 的训练过程

    GRPO 训练循环（每个 epoch）：
      1. 对每个训练样本，生成 group_size 个响应
      2. 用奖励函数给每个响应打分
      3. 计算组内归一化优势（advantage）
      4. 用优势值更新策略（模拟）
      5. 随着训练进行，模型响应质量逐步提升

    参数：
        samples: 训练样本列表
        group_size: 每个问题的采样数（GRPO 论文推荐 4~16）
        num_epochs: 训练轮数
        initial_quality: 初始响应质量（0~1，越高表示初始模型越好）
        seed: 随机种子
    返回：
        dict: 训练历史记录
    """
    random.seed(seed)
    np.random.seed(seed)

    print("=" * 70)
    print("  VLM GRPO 训练模拟")
    print("=" * 70)
    print()
    print(f"训练配置：")
    print(f"  训练样本数: {len(samples)}")
    print(f"  组大小 (group_size): {group_size}")
    print(f"  训练轮数: {num_epochs}")
    print(f"  初始响应质量: {initial_quality:.1f}")
    print()
    print("注意：这是模拟训练，完整的 VLM GRPO 需要：")
    print("  - VLM 模型（如 Qwen2-VL、LLaVA）")
    print("  - 图像编码器和视觉 token 处理")
    print("  - GPU 显存 >= 40GB")
    print("  - DeepSpeed 或 FSDP 分布式训练框架")
    print()

    # 记录训练历史
    history = {
        'epoch': [],
        'accuracy': [],         # 完全正确的比例
        'avg_reward': [],       # 平均奖励
        'best_reward': [],      # 最佳响应的平均奖励
        'avg_advantage_std': [],# 平均优势标准差（衡量区分度）
    }

    # 模拟训练过程：随着 epoch 增加，模型质量逐步提升
    # quality_factor 从 initial_quality 线性增长到接近 1.0
    for epoch in range(num_epochs):
        quality_factor = initial_quality + (1.0 - initial_quality) * (epoch / max(num_epochs - 1, 1))

        epoch_correct = 0
        epoch_rewards = []
        epoch_best_rewards = []
        epoch_adv_stds = []

        for sample in samples:
            gt = sample['ground_truth']

            # 模拟生成 group_size 个响应
            group_rewards = []
            for g in range(group_size):
                # 根据质量因子决定是否生成正确响应
                # 质量因子越高，正确响应的概率越大
                if random.random() < quality_factor:
                    # 生成正确或接近正确的响应
                    if random.random() < 0.7:
                        response = generate_correct_response(gt)
                    else:
                        response = generate_short_correct_response(gt)
                else:
                    # 生成有缺陷的响应
                    choice = random.random()
                    if choice < 0.4:
                        response = generate_wrong_response(gt)
                    elif choice < 0.7:
                        response = generate_partially_correct_response(gt)
                    else:
                        response = generate_low_quality_response(gt)

                # 计算奖励
                reward = simple_reward(response, gt)
                group_rewards.append(reward)

            # GRPO 核心：组内归一化
            rewards_arr = np.array(group_rewards)
            mean_r = rewards_arr.mean()
            std_r = rewards_arr.std() + 1e-8
            advantages = (rewards_arr - mean_r) / std_r

            # 记录统计量
            epoch_rewards.extend(group_rewards)
            epoch_best_rewards.append(max(group_rewards))
            epoch_adv_stds.append(std_r)

            # 检查最佳响应是否正确
            best_idx = np.argmax(rewards_arr)
            if rewards_arr[best_idx] >= 1.0:
                epoch_correct += 1

        # 计算本 epoch 的指标
        accuracy = epoch_correct / len(samples)
        avg_reward = np.mean(epoch_rewards)
        best_reward = np.mean(epoch_best_rewards)
        avg_adv_std = np.mean(epoch_adv_stds)

        history['epoch'].append(epoch + 1)
        history['accuracy'].append(accuracy)
        history['avg_reward'].append(avg_reward)
        history['best_reward'].append(best_reward)
        history['avg_advantage_std'].append(avg_adv_std)

        # 打印本 epoch 的训练日志
        print(f"  Epoch {epoch+1}/{num_epochs} | "
              f"准确率: {accuracy:.3f} | "
              f"平均奖励: {avg_reward:.3f} | "
              f"最佳响应奖励: {best_reward:.3f} | "
              f"优势std: {avg_adv_std:.3f}")

    return history


# ==========================================
# 第六部分：训练前后对比
# ==========================================
def print_before_after_comparison(samples, history, seed=42):
    """
    打印训练前后的对比结果

    展示：
      1. 训练前后在相同样本上的表现差异
      2. 几个具体样本的响应对比
    """
    random.seed(seed)

    print("\n" + "=" * 70)
    print("  训练前后对比")
    print("=" * 70)

    # 选取 5 个样本展示对比
    display_samples = samples[:5]

    print("\n--- 训练前（低质量模型）的响应示例 ---")
    for i, sample in enumerate(display_samples):
        gt = sample['ground_truth']
        # 训练前：低质量响应
        response = generate_wrong_response(gt)
        reward = simple_reward(response, gt)
        extracted = extract_numbers(response, ['三角形', '圆形', '正方形'])
        is_correct = all(extracted.get(s, -1) == gt[s] for s in gt)
        print(f"\n  样本 {i+1} (GT: 三角形={gt['三角形']}, 圆形={gt['圆形']}, "
              f"正方形={gt['正方形']})")
        print(f"    回答: {response[:60]}...")
        print(f"    奖励: {reward:.2f} | {'正确' if is_correct else '错误'}")

    print("\n\n--- 训练后（高质量模型）的响应示例 ---")
    for i, sample in enumerate(display_samples):
        gt = sample['ground_truth']
        # 训练后：高质量响应
        response = generate_correct_response(gt)
        reward = simple_reward(response, gt)
        extracted = extract_numbers(response, ['三角形', '圆形', '正方形'])
        is_correct = all(extracted.get(s, -1) == gt[s] for s in gt)
        print(f"\n  样本 {i+1} (GT: 三角形={gt['三角形']}, 圆形={gt['圆形']}, "
              f"正方形={gt['正方形']})")
        print(f"    回答: {response[:60]}...")
        print(f"    奖励: {reward:.2f} | {'正确' if is_correct else '错误'}")

    # 汇总
    print("\n" + "-" * 70)
    print("  汇总对比：")
    print(f"  {'指标':>12s}  {'训练前':>10s}  {'训练后':>10s}  {'变化':>10s}")
    print(f"  {'----':>12s}  {'------':>10s}  {'------':>10s}  {'----':>10s}")

    before_acc = history['accuracy'][0]
    after_acc = history['accuracy'][-1]
    before_reward = history['avg_reward'][0]
    after_reward = history['avg_reward'][-1]
    before_best = history['best_reward'][0]
    after_best = history['best_reward'][-1]

    print(f"  {'准确率':>12s}  {before_acc:>10.3f}  {after_acc:>10.3f}  {after_acc - before_acc:>+10.3f}")
    print(f"  {'平均奖励':>12s}  {before_reward:>10.3f}  {after_reward:>10.3f}  {after_reward - before_reward:>+10.3f}")
    print(f"  {'最佳奖励':>12s}  {before_best:>10.3f}  {after_best:>10.3f}  {after_best - before_best:>+10.3f}")


# ==========================================
# 第七部分：绘制训练曲线
# ==========================================
def plot_training_curves(history):
    """
    绘制 VLM GRPO 训练曲线

    包含 4 个子图：
      1. 准确率随训练轮次的变化
      2. 平均奖励随训练轮次的变化
      3. 最佳响应奖励的变化
      4. 优势标准差的变化（衡量组内区分度）
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("VLM GRPO 训练曲线（模拟）", fontsize=16, fontweight='bold')

    epochs = history['epoch']

    # 子图1：准确率
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['accuracy'], 'o-', color='#2196F3', linewidth=2,
             markersize=8, label='准确率')
    ax1.fill_between(epochs, 0, history['accuracy'], alpha=0.1, color='#2196F3')
    ax1.set_title('准确率', fontsize=13)
    ax1.set_xlabel('训练轮次 (Epoch)')
    ax1.set_ylabel('准确率')
    ax1.set_ylim(0, 1.05)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 子图2：平均奖励
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['avg_reward'], 's-', color='#FF9800', linewidth=2,
             markersize=8, label='平均奖励')
    ax2.fill_between(epochs, 0, history['avg_reward'], alpha=0.1, color='#FF9800')
    ax2.set_title('平均奖励', fontsize=13)
    ax2.set_xlabel('训练轮次 (Epoch)')
    ax2.set_ylabel('平均奖励')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 子图3：最佳响应奖励
    ax3 = axes[1, 0]
    ax3.plot(epochs, history['best_reward'], 'D-', color='#4CAF50', linewidth=2,
             markersize=8, label='最佳响应奖励')
    ax3.set_title('每组最佳响应的平均奖励', fontsize=13)
    ax3.set_xlabel('训练轮次 (Epoch)')
    ax3.set_ylabel('奖励分数')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 子图4：优势标准差
    ax4 = axes[1, 1]
    ax4.plot(epochs, history['avg_advantage_std'], '^-', color='#9C27B0', linewidth=2,
             markersize=8, label='平均优势标准差')
    ax4.set_title('组内优势标准差（区分度）', fontsize=13)
    ax4.set_xlabel('训练轮次 (Epoch)')
    ax4.set_ylabel('标准差')
    ax4.annotate('std 下降 = 组内响应质量趋同\n（模型变得更稳定）',
                 xy=(epochs[-1] * 0.5, max(history['avg_advantage_std']) * 0.8),
                 fontsize=9, color='gray', style='italic')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/vlm_grpo_training_curves.png', dpi=150, bbox_inches='tight')
    print("\n  训练曲线已保存为 output/vlm_grpo_training_curves.png")
    plt.show()


# ==========================================
# 第八部分：GRPO 核心计算演示
# ==========================================
def demonstrate_grpo_normalization():
    """
    用一个具体例子演示 GRPO 的组内归一化过程

    展示：对同一个图像，VLM 生成 4 个不同质量的响应，
    GRPO 如何通过组内比较来决定每个响应的"好坏"
    """
    print("\n" + "=" * 70)
    print("  GRPO 组内归一化演示（VLM 场景）")
    print("=" * 70)

    # 模拟一个样本的 ground truth
    gt = {'三角形': 3, '圆形': 1, '正方形': 2}

    print(f"\n输入图像包含：三角形={gt['三角形']}，圆形={gt['圆形']}，正方形={gt['正方形']}")
    print(f"提示词：{STANDARD_PROMPT}")
    print()

    # 模拟 4 个响应及其奖励
    responses = [
        ("正确+推理完整", generate_correct_response(gt)),
        ("正确但简短", generate_short_correct_response(gt)),
        ("有推理但错误", generate_wrong_response(gt)),
        ("低质量", generate_low_quality_response(gt)),
    ]

    rewards = []
    print("生成的响应及奖励：")
    print("-" * 70)
    for i, (desc, resp) in enumerate(responses):
        r = simple_reward(resp, gt)
        rewards.append(r)
        print(f"  响应 {i+1} ({desc})")
        print(f"    内容: {resp[:70]}...")
        print(f"    奖励: {r:.3f}")
        print()

    # GRPO 归一化
    rewards_arr = np.array(rewards)
    mean_r = rewards_arr.mean()
    std_r = rewards_arr.std() + 1e-8
    advantages = (rewards_arr - mean_r) / std_r

    print("-" * 70)
    print("GRPO 归一化过程：")
    print(f"  组内均值: {mean_r:.4f}")
    print(f"  组内标准差: {std_r:.4f}")
    print()
    print(f"  {'响应':>4s}  {'原始奖励':>8s}  {'GRPO优势':>10s}  {'含义'}")
    print(f"  {'----':>4s}  {'--------':>8s}  {'----------':>10s}  {'----'}")
    for i in range(len(responses)):
        adv = advantages[i]
        if adv > 0.5:
            meaning = "显著优于组内平均，强烈鼓励"
        elif adv > 0:
            meaning = "略优于组内平均，适度鼓励"
        elif adv > -0.5:
            meaning = "略低于组内平均，适度抑制"
        else:
            meaning = "显著低于组内平均，强烈抑制"
        print(f"  {i+1:>4d}  {rewards[i]:>8.4f}  {adv:>+10.4f}  {meaning}")

    print()
    print("  关键：GRPO 不需要绝对奖励值，只需要组内的相对排序！")
    print("  这就是为什么 GRPO 不需要训练 Critic 网络来估计基线。")


# ==========================================
# 程序入口
# ==========================================
if __name__ == "__main__":
    # 第一步：打印 VLM GRPO 概述
    print_vlm_grpo_overview()

    # 第二步：GRPO 核心计算演示
    demonstrate_grpo_normalization()

    # 第三步：创建训练数据
    print("\n" + "=" * 70)
    print("  创建模拟训练数据")
    print("=" * 70)
    samples = create_training_samples(num_samples=20, seed=42)
    print(f"  已创建 {len(samples)} 个训练样本")

    # 打印几个样本的 ground truth
    print("\n  样本示例：")
    for i, s in enumerate(samples[:5]):
        gt = s['ground_truth']
        print(f"    样本 {i+1}: 三角形={gt['三角形']}, 圆形={gt['圆形']}, 正方形={gt['正方形']}, "
              f"总计={sum(gt.values())}")

    # 第四步：模拟 GRPO 训练
    print()
    history = simulate_grpo_training(
        samples,
        group_size=4,
        num_epochs=5,
        initial_quality=0.3,
        seed=42,
    )

    # 第五步：训练前后对比
    print_before_after_comparison(samples, history, seed=42)

    # 第六步：绘制训练曲线
    print("\n" + "=" * 70)
    print("  开始生成可视化图表...")
    print("=" * 70)
    plot_training_curves(history)

    # 最终总结
    print("\n" + "=" * 70)
    print("  VLM GRPO 训练模拟总结")
    print("=" * 70)
    print("""
  本脚本模拟了 VLM GRPO 的训练流程，展示了以下关键概念：

  1. VLM GRPO 与纯文本 GRPO 的区别：
     - 输入从纯文本变为 (图像, 文本) 对
     - 需要视觉编码器将图像转换为视觉 token
     - 视觉 token 与文本 token 拼接后送入 LLM
     - 奖励函数需要考虑视觉定位能力

  2. GRPO 核心机制：
     - 对同一图像生成多个响应（group）
     - 用奖励函数打分，组内归一化得到优势
     - 正优势的响应被鼓励，负优势的被抑制
     - 不需要额外训练 Critic 网络

  3. 完整 VLM GRPO 的实现要点：
     - 使用 transformers 库中的 VLM 模型
       例: Qwen2VLForConditionalGeneration
     - 图像通过 Vision Transformer 编码
     - 视觉特征通过投影层映射到 LLM 的嵌入空间
     - 训练时需要处理更长的序列（视觉 token 占用额外长度）
     - 建议使用 DeepSpeed ZeRO-3 或 FSDP 进行分布式训练

  4. VLM GRPO 的应用场景：
     - 视觉问答（VQA）的推理能力提升
     - 图像描述（Image Captioning）的质量优化
     - 多模态数学推理（结合图像和文本的推理）
     - 具身智能（机器人视觉-语言-动作联合训练）

  5. 关键超参数：
     - group_size: 每个问题的采样数（推荐 4~16）
     - clip_ratio: PPO 风格的裁剪范围（推荐 0.2）
     - learning_rate: 学习率（推荐 1e-6 ~ 5e-6）
     -KL_coefficient: KL 散度惩罚系数（推荐 0.01~0.1）
    """)
