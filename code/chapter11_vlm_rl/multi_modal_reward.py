"""
第11章：多模态奖励函数设计
==========================================================

本脚本演示为 VLM（视觉语言模型）强化学习设计多模态奖励函数：
  1. reward_correctness:       答案正确性奖励（0.0 或 1.0）
  2. reward_reasoning_quality: 推理质量奖励（0.0~0.5）
  3. reward_format:            格式规范奖励（0.0~0.2）
  4. reward_visual_grounding:  视觉定位奖励（0.0~0.3）
  5. compute_total_reward:     加权总分

奖励设计原则：
  - 正确性是核心，权重最高
  - 推理质量鼓励"展示思考过程"
  - 格式规范确保输出可解析
  - 视觉定位鼓励模型真正"看懂"图片

运行方式：
  python multi_modal_reward.py
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt

# 创建输出目录
os.makedirs("output", exist_ok=True)

# 设置中文字体，确保图表标题和标签正常显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 第一部分：奖励函数定义
# ==========================================

def reward_correctness(response, ground_truth):
    """
    正确性奖励：检查回答中的数字是否与标注一致

    逻辑：
      1. 从回答中提取每种形状对应的数量
      2. 与 ground_truth 逐个比较
      3. 全部正确得 1.0，否则得 0.0

    参数：
        response: 模型的文本回答
        ground_truth: dict，{'三角形': int, '圆形': int, '正方形': int}
    返回：
        float: 0.0（错误）或 1.0（正确）
    """
    # 尝试从回答中提取每种形状对应的数字
    extracted = {}

    for shape_name in ['三角形', '圆形', '正方形']:
        # 模式1：形状名 + 数字，如 "三角形3个" 或 "三角形：3" 或 "三角形有3个"
        patterns = [
            rf'{shape_name}[^0-9]*?(\d+)',
            rf'(\d+)\s*个\s*{shape_name}',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, response)
            if matches:
                # 取最后一个匹配（通常是最终结论）
                extracted[shape_name] = int(matches[-1])
                break

    # 如果未能提取到所有形状的数量，直接返回 0
    for shape_name in ground_truth:
        if shape_name not in extracted:
            return 0.0

    # 逐个比较
    for shape_name, expected_count in ground_truth.items():
        if extracted.get(shape_name, -1) != expected_count:
            return 0.0

    return 1.0


def reward_reasoning_quality(response):
    """
    推理质量奖励：检查回答是否包含逐步推理过程

    评分标准（0.0~0.5）：
      - 0.0: 没有任何推理步骤，直接给答案
      - 0.1~0.2: 简单提及过程
      - 0.3~0.4: 包含明确的分步骤推理
      - 0.5: 完整的逐步推理，有分析过程

    判断依据：
      - 是否包含序号标记（"第一步"、"Step 1"、"1." 等）
      - 是否包含推理关键词（"因为"、"所以"、"因此"、"分析"、"计算"）
      - 回答长度是否足以容纳推理过程

    参数：
        response: 模型的文本回答
    返回：
        float: 0.0~0.5
    """
    score = 0.0

    # 检查是否有分步骤标记（0.2 分）
    step_markers = [
        r'第[一二三四五六七八九十\d]+步',
        r'[Ss]tep\s*\d+',
        r'首先',
        r'然后',
        r'接着',
        r'最后',
    ]
    step_count = sum(1 for m in step_markers if re.search(m, response))
    if step_count >= 3:
        score += 0.2
    elif step_count >= 1:
        score += 0.1

    # 检查是否有推理关键词（0.2 分）
    reasoning_keywords = [
        '因为', '所以', '因此', '分析', '计算',
        '可以', '发现', '观察', '总共', '合计',
    ]
    keyword_count = sum(1 for kw in reasoning_keywords if kw in response)
    if keyword_count >= 3:
        score += 0.2
    elif keyword_count >= 1:
        score += 0.1

    # 检查回答长度（0.1 分）——足够长才可能包含推理
    if len(response) >= 50:
        score += 0.1

    return min(score, 0.5)


def reward_format(response):
    """
    格式规范奖励：检查回答是否遵循预期的输出格式

    评分标准（0.0~0.2）：
      - 0.0: 完全没有格式
      - 0.05: 有部分格式（如列出了一些形状名）
      - 0.1: 格式基本正确（提到了三种形状）
      - 0.15: 格式良好（三种形状都有对应的数量）
      - 0.2: 格式完美（有总结性结论 + 三种形状数量齐全）

    期望格式示例：
      "图片中有 X 个三角形、Y 个圆形和 Z 个正方形，总共 N 个形状。"

    参数：
        response: 模型的文本回答
    返回：
        float: 0.0~0.2
    """
    score = 0.0

    # 检查是否提到了所有三种形状（0.1 分）
    shapes_mentioned = sum(1 for s in ['三角形', '圆形', '正方形'] if s in response)
    if shapes_mentioned == 3:
        score += 0.1
    elif shapes_mentioned >= 2:
        score += 0.05

    # 检查是否有总结性表述（0.05 分）
    summary_patterns = [
        r'总共\s*\d+',
        r'合计\s*\d+',
        r'一共\s*\d+',
        r'总计\s*\d+',
        r'答[：:]',
    ]
    if any(re.search(p, response) for p in summary_patterns):
        score += 0.05

    # 检查是否有数字和形状的对应关系（0.05 分）
    has_number_shape_pair = bool(re.search(r'\d+\s*个', response))
    if has_number_shape_pair:
        score += 0.05

    return min(score, 0.2)


def reward_visual_grounding(response, image_info):
    """
    视觉定位奖励：检查回答是否正确引用了图片中的视觉特征

    评分标准（0.0~0.3）：
      - 检查回答中是否提及了图片中实际存在的形状
      - 对于提及的形状，是否给出了视觉描述（颜色、位置等）
      - 模型是否表现出"真正在看图"的迹象

    image_info 示例：
      {
        'present_shapes': ['三角形', '圆形'],  # 图片中实际存在的形状
        'absent_shapes': ['正方形'],             # 图片中不存在的形状
      }

    参数：
        response: 模型的文本回答
        image_info: dict，包含图片中的形状信息
    返回：
        float: 0.0~0.3
    """
    score = 0.0

    present_shapes = image_info.get('present_shapes', [])
    absent_shapes = image_info.get('absent_shapes', [])

    # 检查是否提到了存在的形状（0.15 分）
    mentioned_present = sum(1 for s in present_shapes if s in response)
    if len(present_shapes) > 0:
        ratio = mentioned_present / len(present_shapes)
        score += 0.15 * ratio

    # 检查是否正确说明不存在的形状（0.1 分）
    # 如果某种形状不存在，模型说"0 个"或"没有"，这是正确的视觉定位
    for shape in absent_shapes:
        if shape in response:
            # 检查是否正确标注为 0
            zero_patterns = [
                rf'{shape}[^0-9]*?0',
                rf'0\s*个\s*{shape}',
                rf'没有\s*{shape}',
            ]
            if any(re.search(p, response) for p in zero_patterns):
                score += 0.05
                break  # 最多加 0.05

    # 检查是否有视觉描述性语言（0.05 分）
    visual_keywords = [
        '颜色', '红色', '蓝色', '绿色', '橙色', '紫色',
        '位置', '左边', '右边', '上方', '下方',
        '大小', '大', '小',
        '可以看到', '图中', '图片中', '画布上',
    ]
    visual_count = sum(1 for kw in visual_keywords if kw in response)
    if visual_count >= 2:
        score += 0.05

    return min(score, 0.3)


def compute_total_reward(response, ground_truth, image_info):
    """
    计算加权总奖励

    权重分配：
      - 正确性（correctness）:       × 1.0  → 满分 1.0
      - 推理质量（reasoning）:        × 1.0  → 满分 0.5
      - 格式规范（format）:           × 1.0  → 满分 0.2
      - 视觉定位（visual_grounding）: × 1.0  → 满分 0.3
      ---------------------------------------------
      理论最高总分: 2.0

    参数：
        response: 模型的文本回答
        ground_truth: dict，标注数据
        image_info: dict，图片信息
    返回：
        dict，包含各分项奖励和总奖励
    """
    r_correct = reward_correctness(response, ground_truth)
    r_reasoning = reward_reasoning_quality(response)
    r_format = reward_format(response)
    r_visual = reward_visual_grounding(response, image_info)

    total = r_correct + r_reasoning + r_format + r_visual

    return {
        'correctness': r_correct,
        'reasoning': r_reasoning,
        'format': r_format,
        'visual_grounding': r_visual,
        'total': total,
    }


# ==========================================
# 第二部分：测试用例定义
# ==========================================
# 8 个测试用例，覆盖不同质量水平的回答：
#   完美回答、正确但简短、错误但有推理、错误且混乱、
#   部分正确、格式好但内容错、有视觉描述、无格式无推理

test_cases = [
    {
        'name': '完美回答（正确 + 推理 + 格式 + 视觉定位）',
        'response': (
            '我仔细观察了图片。\n'
            '首先，分析图中的三角形：我数到了 3 个三角形，分别在左上方和右侧。\n'
            '然后，分析圆形：图片中有 1 个圆形，颜色是蓝色，位于中间偏左。\n'
            '接着，分析正方形：图片中有 2 个正方形，一个红色一个绿色。\n'
            '所以，图片中总共有 3 个三角形、1 个圆形和 2 个正方形，合计 6 个形状。\n'
            '答：三角形3个，圆形1个，正方形2个，总共6个。'
        ),
        'ground_truth': {'三角形': 3, '圆形': 1, '正方形': 2},
        'image_info': {
            'present_shapes': ['三角形', '圆形', '正方形'],
            'absent_shapes': [],
        },
    },
    {
        'name': '正确但简短（无推理过程）',
        'response': '三角形3个，圆形1个，正方形2个。',
        'ground_truth': {'三角形': 3, '圆形': 1, '正方形': 2},
        'image_info': {
            'present_shapes': ['三角形', '圆形', '正方形'],
            'absent_shapes': [],
        },
    },
    {
        'name': '错误但有推理过程',
        'response': (
            '让我一步步来分析。\n'
            '首先，观察图片中的三角形。可以看到有 2 个三角形。\n'
            '然后，观察圆形。图片中有 1 个圆形。\n'
            '最后，观察正方形。有 2 个正方形。\n'
            '所以，总共是 2+1+2=5 个形状。\n'
            '答：三角形2个，圆形1个，正方形2个。'
        ),
        'ground_truth': {'三角形': 3, '圆形': 1, '正方形': 2},
        'image_info': {
            'present_shapes': ['三角形', '圆形', '正方形'],
            'absent_shapes': [],
        },
    },
    {
        'name': '错误且混乱',
        'response': '图里有好多形状，大概是三角形2个圆形3个正方形5个吧。',
        'ground_truth': {'三角形': 3, '圆形': 1, '正方形': 2},
        'image_info': {
            'present_shapes': ['三角形', '圆形', '正方形'],
            'absent_shapes': [],
        },
    },
    {
        'name': '部分正确（只数对了一种）',
        'response': (
            '我来看一下图片中的形状。\n'
            '三角形有 3 个，这个我数对了。\n'
            '圆形有 2 个，正方形有 1 个。\n'
            '总共 6 个形状。'
        ),
        'ground_truth': {'三角形': 3, '圆形': 1, '正方形': 2},
        'image_info': {
            'present_shapes': ['三角形', '圆形', '正方形'],
            'absent_shapes': [],
        },
    },
    {
        'name': '格式好但内容错',
        'response': (
            '答：经过分析，图片中包含以下形状：\n'
            '三角形0个，圆形5个，正方形5个，总共10个。'
        ),
        'ground_truth': {'三角形': 0, '圆形': 2, '正方形': 3},
        'image_info': {
            'present_shapes': ['圆形', '正方形'],
            'absent_shapes': ['三角形'],
        },
    },
    {
        'name': '有视觉描述但答案有误',
        'response': (
            '观察图片，可以看到图片中左侧有一个蓝色的圆形，'
            '右边有一个红色的三角形，下方有两个绿色的正方形。\n'
            '三角形1个，圆形2个，正方形2个，合计5个形状。'
        ),
        'ground_truth': {'三角形': 2, '圆形': 1, '正方形': 3},
        'image_info': {
            'present_shapes': ['三角形', '圆形', '正方形'],
            'absent_shapes': [],
        },
    },
    {
        'name': '某些形状不存在的情况（正确处理）',
        'response': (
            '首先，分析图中的三角形：有 2 个三角形。\n'
            '然后，分析圆形：有 3 个圆形。\n'
            '接着，分析正方形：图片中没有正方形，0个。\n'
            '所以，三角形2个，圆形3个，正方形0个，总共5个形状。\n'
            '答：三角形2个，圆形3个，正方形0个。'
        ),
        'ground_truth': {'三角形': 2, '圆形': 3, '正方形': 0},
        'image_info': {
            'present_shapes': ['三角形', '圆形'],
            'absent_shapes': ['正方形'],
        },
    },
]


# ==========================================
# 第三部分：测试奖励函数并打印详细结果
# ==========================================
def run_reward_tests():
    """
    在所有测试用例上运行奖励函数，打印详细的奖励分解表
    """
    print("=" * 80)
    print("  多模态奖励函数测试 — 详细分解表")
    print("=" * 80)
    print()
    print("奖励组成：")
    print("  正确性（correctness）:       0.0 ~ 1.0   答案是否完全正确")
    print("  推理质量（reasoning）:        0.0 ~ 0.5   是否有逐步推理过程")
    print("  格式规范（format）:           0.0 ~ 0.2   输出格式是否规范")
    print("  视觉定位（visual_grounding）: 0.0 ~ 0.3   是否引用了正确的视觉特征")
    print("  ----------------------------------------------")
    print("  总计:                        0.0 ~ 2.0")
    print()

    # 表头
    header = (
        f"{'编号':>4s}  "
        f"{'正确性':>6s}  "
        f"{'推理':>4s}  "
        f"{'格式':>4s}  "
        f"{'视觉':>4s}  "
        f"{'总分':>5s}  "
        f"{'说明'}"
    )
    separator = "-" * 80
    print(header)
    print(separator)

    all_rewards = []

    for i, tc in enumerate(test_cases):
        rewards = compute_total_reward(
            tc['response'],
            tc['ground_truth'],
            tc['image_info'],
        )
        all_rewards.append(rewards)

        print(
            f"  {i+1:>2d}  "
            f"{rewards['correctness']:>6.2f}  "
            f"{rewards['reasoning']:>4.2f}  "
            f"{rewards['format']:>4.2f}  "
            f"{rewards['visual_grounding']:>4.2f}  "
            f"{rewards['total']:>5.2f}  "
            f"{tc['name']}"
        )

    print(separator)
    print()

    # 统计摘要
    totals = [r['total'] for r in all_rewards]
    print(f"奖励统计：")
    print(f"  最高总分: {max(totals):.2f}")
    print(f"  最低总分: {min(totals):.2f}")
    print(f"  平均总分: {np.mean(totals):.2f}")
    print()

    # 打印每个测试用例的详细分析
    print("=" * 80)
    print("  逐条详细分析")
    print("=" * 80)
    for i, (tc, rewards) in enumerate(zip(test_cases, all_rewards)):
        gt = tc['ground_truth']
        print(f"\n--- 测试用例 {i+1}: {tc['name']} ---")
        print(f"  标注: 三角形={gt['三角形']}, 圆形={gt['圆形']}, 正方形={gt['正方形']}")
        print(f"  回答: {tc['response'][:80]}...")
        print(f"  奖励分解:")
        print(f"    正确性:   {rewards['correctness']:.2f}  {'[满分]' if rewards['correctness'] == 1.0 else '[未得分]'}")
        print(f"    推理质量: {rewards['reasoning']:.2f}  {'[满分]' if rewards['reasoning'] == 0.5 else ''}")
        print(f"    格式规范: {rewards['format']:.2f}  {'[满分]' if rewards['format'] == 0.2 else ''}")
        print(f"    视觉定位: {rewards['visual_grounding']:.2f}  {'[满分]' if rewards['visual_grounding'] == 0.3 else ''}")
        print(f"    加权总分: {rewards['total']:.2f}")

    return all_rewards


# ==========================================
# 第四部分：可视化奖励权重分布
# ==========================================
def plot_reward_weights(all_rewards):
    """
    绘制奖励分量权重的饼图和各测试用例的堆叠柱状图

    左图：奖励理论满分占比（饼图）
    右图：各测试用例的奖励分量堆叠柱状图
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ---------- 左图：理论满分占比饼图 ----------
    ax1 = axes[0]

    # 各分量的理论满分
    max_scores = [1.0, 0.5, 0.2, 0.3]  # correctness, reasoning, format, visual
    labels = ['正确性\n(满分1.0)', '推理质量\n(满分0.5)', '格式规范\n(满分0.2)', '视觉定位\n(满分0.3)']
    colors = ['#e74c3c', '#3498db', '#f39c12', '#2ecc71']
    explode = (0.05, 0.02, 0.02, 0.02)

    wedges, texts, autotexts = ax1.pie(
        max_scores,
        labels=labels,
        colors=colors,
        explode=explode,
        autopct=lambda pct: f'{pct:.1f}%',
        startangle=90,
        textprops={'fontsize': 10},
    )
    for autotext in autotexts:
        autotext.set_fontsize(10)
    ax1.set_title('奖励分量理论满分占比', fontsize=14, fontweight='bold')

    # ---------- 右图：堆叠柱状图 ----------
    ax2 = axes[1]

    n_cases = len(all_rewards)
    x = np.arange(n_cases)

    correctness_vals = [r['correctness'] for r in all_rewards]
    reasoning_vals = [r['reasoning'] for r in all_rewards]
    format_vals = [r['format'] for r in all_rewards]
    visual_vals = [r['visual_grounding'] for r in all_rewards]

    bar_width = 0.6
    ax2.bar(x, correctness_vals, bar_width, label='正确性', color='#e74c3c')
    ax2.bar(x, reasoning_vals, bar_width, bottom=correctness_vals,
            label='推理质量', color='#3498db')
    bottom2 = [c + r for c, r in zip(correctness_vals, reasoning_vals)]
    ax2.bar(x, format_vals, bar_width, bottom=bottom2,
            label='格式规范', color='#f39c12')
    bottom3 = [b + f for b, f in zip(bottom2, format_vals)]
    ax2.bar(x, visual_vals, bar_width, bottom=bottom3,
            label='视觉定位', color='#2ecc71')

    # 在柱顶标注总分
    for i, r in enumerate(all_rewards):
        ax2.text(i, r['total'] + 0.05, f"{r['total']:.2f}",
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2.set_xlabel('测试用例编号', fontsize=12)
    ax2.set_ylabel('奖励分数', fontsize=12)
    ax2.set_title('各测试用例的奖励分量分解', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'#{i+1}' for i in range(n_cases)])
    ax2.legend(fontsize=10, loc='upper right')
    ax2.set_ylim(0, 2.3)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('output/multi_modal_reward_breakdown.png', dpi=150, bbox_inches='tight')
    print("\n  奖励分解图已保存为 output/multi_modal_reward_breakdown.png")
    plt.show()


# ==========================================
# 程序入口
# ==========================================
if __name__ == "__main__":
    # 第一步：运行奖励函数测试
    all_rewards = run_reward_tests()

    # 第二步：可视化奖励权重
    print("\n" + "=" * 80)
    print("  开始生成可视化图表...")
    print("=" * 80)
    plot_reward_weights(all_rewards)

    # 最终总结
    print("\n" + "=" * 80)
    print("  多模态奖励函数设计总结")
    print("=" * 80)
    print("""
  奖励函数设计要点：

    1. 正确性（权重最高，满分 1.0）
       - 只有全部正确才得满分，任何错误都得 0 分
       - 这种"全对或全错"的设计鼓励模型追求完美
       - 在 GRPO 训练中，组内正确响应会获得正优势

    2. 推理质量（满分 0.5）
       - 鼓励模型展示思考过程，而非直接给出答案
       - 通过检测步骤标记和推理关键词来评分
       - 有助于 Chain-of-Thought 推理能力的培养

    3. 格式规范（满分 0.2）
       - 确保输出可被自动解析
       - 要求包含所有三种形状和总结性表述
       - 降低后续评估的复杂度

    4. 视觉定位（满分 0.3）
       - 鼓励模型真正"看懂"图片内容
       - 检查是否正确引用存在的形状和说明不存在的形状
       - 对于 VLM 训练尤为重要，避免模型"猜答案"

  与纯文本 RL 的区别：
    - 纯文本 RL 的奖励只关注文本质量
    - VLM RL 的奖励需要额外考虑视觉定位能力
    - 模型不仅要回答正确，还要"看着图片"回答

  实际应用中的扩展：
    - 可加入 OCR 奖励（检查模型是否正确读取图中文本）
    - 可加入空间关系奖励（检查对物体位置关系的理解）
    - 可使用 LLM-as-Judge 代替规则奖励，获得更细腻的评分
    """)
