"""
第8章：可验证奖励函数 —— RLVR 的核心组件
==========================================================

本脚本实现了 RLVR (Reinforcement Learning with Verifiable Rewards) 中
常用的规则奖励函数，用于自动评估模型生成的推理过程和最终答案。

奖励函数组件：
  1. check_answer_correctness  —— 检查最终答案是否正确
  2. check_format              —— 检查回复格式是否规范
  3. check_reasoning_quality   —— 评估推理过程的质量
  4. compute_total_reward      —— 加权组合以上各项，计算总奖励

运行方式：
  python rule_based_reward.py
"""

import re


# ==========================================
# 第一部分：答案正确性检查
# ==========================================
def check_answer_correctness(response, ground_truth):
    """
    从模型回复中提取最终答案并与标准答案对比

    提取规则：
      1. 优先从 \\boxed{...} 中提取（LaTeX 格式）
      2. 如果没有 boxed，尝试匹配"答案是..."、"最终答案为..."等模式
      3. 支持整数、小数、分数、百分数、负数等格式

    参数：
        response: 模型生成的回复文本
        ground_truth: 标准答案（字符串或数字）
    返回：
        dict: {
            "score": float (0.0 或 1.0),
            "extracted": str (提取到的答案),
            "method": str (提取方法),
            "correct": bool (是否正确),
        }
    """
    extracted = None
    method = "未提取到答案"

    # 方法1：从 \boxed{...} 中提取
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', response)
    if boxed_match:
        extracted = boxed_match.group(1).strip()
        method = "\\boxed{} 提取"

    # 方法2：匹配"答案是/为/："等中文模式
    if extracted is None:
        cn_patterns = [
            r'答案[是为：:]\s*([+-]?\d+\.?\d*)',       # "答案是 42"
            r'最终答案[是为：:]\s*([+-]?\d+\.?\d*)',    # "最终答案为 42"
            r'结果[是为：:]\s*([+-]?\d+\.?\d*)',         # "结果是 42"
            r'所以[，,]\s*(?:答案[是为])?\s*([+-]?\d+\.?\d*)',  # "所以答案是 42"
        ]
        for pattern in cn_patterns:
            match = re.search(pattern, response)
            if match:
                extracted = match.group(1).strip()
                method = "中文模式匹配"
                break

    # 方法3：匹配 "The answer is ..." 等英文模式
    if extracted is None:
        en_patterns = [
            r'[Tt]he answer is\s*([+-]?\d+\.?\d*)',
            r'[Tt]herefore[,.]?\s*(?:the answer is\s*)?([+-]?\d+\.?\d*)',
            r'[Ss]o the answer is\s*([+-]?\d+\.?\d*)',
        ]
        for pattern in en_patterns:
            match = re.search(pattern, response)
            if match:
                extracted = match.group(1).strip()
                method = "英文模式匹配"
                break

    # 方法4：提取最后一个独立数字（最后的手段）
    if extracted is None:
        all_numbers = re.findall(r'([+-]?\d+\.?\d*)', response)
        if all_numbers:
            extracted = all_numbers[-1]
            method = "最后一个数字（兜底）"

    # 对比答案
    correct = False
    if extracted is not None:
        try:
            # 统一转为浮点数比较，支持容差
            extracted_num = float(extracted)
            truth_num = float(ground_truth)
            correct = abs(extracted_num - truth_num) < 1e-6
        except (ValueError, TypeError):
            # 如果无法转为数字，做字符串精确匹配
            correct = str(extracted).strip() == str(ground_truth).strip()

    score = 1.0 if correct else 0.0

    return {
        "score": score,
        "extracted": extracted if extracted else "（未提取到）",
        "method": method,
        "correct": correct,
    }


# ==========================================
# 第二部分：格式规范性检查
# ==========================================
def check_format(response):
    """
    检查回复是否具有规范的推理格式

    检查项目：
      1. 是否包含推理步骤（步骤标记，如"步骤"、"第X步"、"Step"等）
      2. 是否包含最终答案标记（如"答案"、"\\boxed{}"等）
      3. 回复长度是否合理（不能太短也不能太长）
      4. 是否包含数学表达式

    参数：
        response: 模型生成的回复文本
    返回：
        dict: {
            "score": float (0.0 ~ 1.0),
            "details": dict (各检查项的详细得分),
        }
    """
    details = {}

    # 检查1：推理步骤标记（0.25 分）
    step_patterns = [
        r'步骤\s*\d',         # "步骤1"
        r'第\s*\d+\s*步',     # "第1步"
        r'[Ss]tep\s*\d',      # "Step 1"
        r'\d+\)\s',           # "1) "
        r'首先|然后|接着|最后',  # 中文连接词
    ]
    has_steps = any(re.search(p, response) for p in step_patterns)
    details["有步骤标记"] = 0.25 if has_steps else 0.0

    # 检查2：最终答案标记（0.25 分）
    answer_patterns = [
        r'\\boxed\{',         # LaTeX 格式
        r'答案[是为：:]',       # 中文标记
        r'[Tt]he answer is',  # 英文标记
        r'最终结果',           # 中文标记
    ]
    has_answer = any(re.search(p, response) for p in answer_patterns)
    details["有答案标记"] = 0.25 if has_answer else 0.0

    # 检查3：回复长度合理（0.25 分）
    length = len(response)
    if 20 <= length <= 2000:
        details["长度合理"] = 0.25
    elif 10 <= length < 20 or 2000 < length <= 5000:
        details["长度合理"] = 0.10
    else:
        details["长度合理"] = 0.0

    # 检查4：包含数学表达式（0.25 分）
    math_patterns = [
        r'\d+\s*[+\-*/×÷]\s*\d+',   # 算术运算：3 + 5
        r'\d+\s*[=＝]\s*\d+',        # 等式：x = 10
        r'[（(][^)]*[)）]',          # 括号表达式
        r'\\frac|\\sqrt|\\times',     # LaTeX 数学命令
    ]
    has_math = any(re.search(p, response) for p in math_patterns)
    details["有数学表达式"] = 0.25 if has_math else 0.0

    total_score = sum(details.values())
    return {
        "score": total_score,
        "details": details,
    }


# ==========================================
# 第三部分：推理质量评估
# ==========================================
def check_reasoning_quality(response):
    """
    基于启发式规则评估推理过程的基本质量

    评估维度：
      1. 推理步骤数量 —— 步骤越多（在一定范围内），推理越详细
      2. 数值计算的连贯性 —— 是否有中间计算结果
      3. 逻辑连接词的使用 —— 是否有因果/递进关系
      4. 是否存在明显错误 —— 如除以零、负数开方等

    参数：
        response: 模型生成的回复文本
    返回：
        dict: {
            "score": float (0.0 ~ 1.0),
            "details": dict (各维度的详细得分),
        }
    """
    details = {}

    # 维度1：推理步骤数量（0 ~ 0.3 分）
    # 计算回复中包含的句子/行数作为推理步骤的代理指标
    sentences = re.split(r'[。.！!？?\n]', response)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
    num_steps = len(sentences)
    if num_steps >= 5:
        details["步骤数量"] = 0.30
    elif num_steps >= 3:
        details["步骤数量"] = 0.20
    elif num_steps >= 1:
        details["步骤数量"] = 0.10
    else:
        details["步骤数量"] = 0.0

    # 维度2：中间计算结果（0 ~ 0.3 分）
    # 检查是否包含 "=" 或 "得" 等计算标记
    calc_patterns = [
        r'\d+\s*[+－\-]\s*\d+\s*[=＝]\s*\d+',   # 加减法
        r'\d+\s*[*×]\s*\d+\s*[=＝]\s*\d+',       # 乘法
        r'\d+\s*[/÷]\s*\d+\s*[=＝]\s*[\d.]+',   # 除法
        r'=+\s*\d+',                              # 等号后跟数字
    ]
    calc_count = sum(len(re.findall(p, response)) for p in calc_patterns)
    if calc_count >= 3:
        details["中间计算"] = 0.30
    elif calc_count >= 1:
        details["中间计算"] = 0.15
    else:
        details["中间计算"] = 0.0

    # 维度3：逻辑连接词（0 ~ 0.2 分）
    logic_words = [
        '因此', '所以', '因为', '由于', '于是',
        '那么', '从而', '也就是说', '换句话说',
        '根据', '根据题意', '由题意可知',
        '首先', '然后', '接着', '最后',
    ]
    logic_count = sum(1 for word in logic_words if word in response)
    if logic_count >= 3:
        details["逻辑连接"] = 0.20
    elif logic_count >= 1:
        details["逻辑连接"] = 0.10
    else:
        details["逻辑连接"] = 0.0

    # 维度4：无明显错误（0 ~ 0.2 分）
    # 检查常见的推理错误
    error_patterns = [
        r'除以\s*0',           # 除以零
        r'÷\s*0',             # 除以零（符号形式）
        r'/\s*0(?!\d)',        # 除以零（斜杠形式）
        r'负数.*开方',         # 负数开方
        r'负数.*开根号',       # 负数开根号
    ]
    has_errors = any(re.search(p, response) for p in error_patterns)
    details["无错误"] = 0.0 if has_errors else 0.20

    total_score = sum(details.values())
    return {
        "score": min(total_score, 1.0),
        "details": details,
    }


# ==========================================
# 第四部分：加权总奖励计算
# ==========================================
def compute_total_reward(response, ground_truth, weights=None):
    """
    加权组合各项奖励，计算最终总奖励

    总奖励 = w1 * 答案正确性 + w2 * 格式规范性 + w3 * 推理质量

    默认权重：
        - 答案正确性（w1 = 0.6）：最重要，答案对不对
        - 格式规范性（w2 = 0.15）：格式是否规范
        - 推理质量  （w3 = 0.25）：推理过程是否合理

    参数：
        response: 模型生成的回复文本
        ground_truth: 标准答案
        weights: 权重字典 {"correctness": w1, "format": w2, "reasoning": w3}
    返回：
        dict: {
            "total_reward": float (0.0 ~ 1.0),
            "answer_check": dict,
            "format_check": dict,
            "reasoning_check": dict,
        }
    """
    if weights is None:
        weights = {
            "correctness": 0.6,
            "format": 0.15,
            "reasoning": 0.25,
        }

    # 分别计算各项得分
    answer_result = check_answer_correctness(response, ground_truth)
    format_result = check_format(response)
    reasoning_result = check_reasoning_quality(response)

    # 加权求和
    total_reward = (
        weights["correctness"] * answer_result["score"]
        + weights["format"] * format_result["score"]
        + weights["reasoning"] * reasoning_result["score"]
    )

    return {
        "total_reward": total_reward,
        "answer_check": answer_result,
        "format_check": format_result,
        "reasoning_check": reasoning_result,
        "weights": weights,
    }


# ==========================================
# 第五部分：打印奖励分解
# ==========================================
def print_reward_breakdown(result, response_label=""):
    """
    以清晰的格式打印奖励分解

    参数：
        result: compute_total_reward 的返回值
        response_label: 回复的标签（用于区分不同测试用例）
    """
    print(f"  【{response_label}】总奖励: {result['total_reward']:.4f}")
    print(f"  ├── 答案正确性 ({result['weights']['correctness']:.0%} 权重): "
          f"{result['answer_check']['score']:.2f}")
    print(f"  │   ├── 提取到的答案: {result['answer_check']['extracted']}")
    print(f"  │   ├── 提取方法: {result['answer_check']['method']}")
    print(f"  │   └── 是否正确: {'是' if result['answer_check']['correct'] else '否'}")
    print(f"  ├── 格式规范性 ({result['weights']['format']:.0%} 权重): "
          f"{result['format_check']['score']:.2f}")
    for item, score in result['format_check']['details'].items():
        icon = "+" if score > 0 else "-"
        print(f"  │   ├── [{icon}] {item}: {score:.2f}")
    print(f"  └── 推理质量 ({result['weights']['reasoning']:.0%} 权重): "
          f"{result['reasoning_check']['score']:.2f}")
    for item, score in result['reasoning_check']['details'].items():
        icon = "+" if score > 0 else "-"
        print(f"      ├── [{icon}] {item}: {score:.2f}")
    print()


# ==========================================
# 第六部分：测试用例
# ==========================================
def run_tests():
    """
    在多个测试用例上验证奖励函数的行为

    测试场景覆盖：
      1. 完美回答 —— 答案正确、格式规范、推理详尽
      2. 答案正确但格式差 —— 没有步骤标记和答案标记
      3. 答案错误但推理详细 —— 格式好但最终答案算错
      4. 完全不合规的回答 —— 太短，无推理，无格式
      5. LaTeX 格式的回答 —— 用 \\boxed{} 标注答案
    """
    print("=" * 70)
    print("  可验证奖励函数测试")
    print("=" * 70)

    # 定义测试用例
    test_cases = [
        {
            "label": "完美回答",
            "ground_truth": "42",
            "response": (
                "我们来一步步解决这个问题。\n"
                "步骤1：根据题意，小明有 15 个苹果，小红给了他 27 个。\n"
                "步骤2：所以总数 = 15 + 27 = 42。\n"
                "因此，小明现在一共有 42 个苹果。\n"
                "答案是：42"
            ),
        },
        {
            "label": "答案正确但格式差",
            "ground_truth": "42",
            "response": "42",
        },
        {
            "label": "答案错误但推理详细",
            "ground_truth": "42",
            "response": (
                "首先，我们需要计算苹果的总数。\n"
                "根据题意，小明有 15 个苹果，小红给了他 27 个。\n"
                "然后，我们计算 15 + 27 = 35。\n"
                "所以，总数 = 35 个苹果。\n"
                "由于这个加法比较简单，我们直接得出结果。\n"
                "答案是：35"
            ),
        },
        {
            "label": "完全不合规",
            "ground_truth": "42",
            "response": "不知道",
        },
        {
            "label": "LaTeX 格式回答",
            "ground_truth": "36",
            "response": (
                "首先，根据题意我们需要计算 4 × 9。\n"
                "第1步：4 × 9 = 36\n"
                "所以答案是 \\boxed{36}"
            ),
        },
    ]

    # 运行测试并打印结果
    results = []
    for tc in test_cases:
        print(f"\n{'─' * 70}")
        print(f"  测试用例: {tc['label']}")
        print(f"  标准答案: {tc['ground_truth']}")
        print(f"  模型回复: {tc['response'][:80]}{'...' if len(tc['response']) > 80 else ''}")
        print(f"{'─' * 70}")

        result = compute_total_reward(tc["response"], tc["ground_truth"])
        print_reward_breakdown(result, tc["label"])
        results.append((tc["label"], result))

    # ==========================================
    # 第七部分：汇总对比
    # ==========================================
    print("=" * 70)
    print("  奖励汇总对比")
    print("=" * 70)
    print()
    print(f"  {'测试用例':<20s}  {'总奖励':>8s}  {'正确性':>8s}  {'格式':>8s}  {'推理':>8s}")
    print(f"  {'─' * 20}  {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 8}")
    for label, result in results:
        total = result["total_reward"]
        correct = result["answer_check"]["score"]
        fmt = result["format_check"]["score"]
        reasoning = result["reasoning_check"]["score"]
        print(f"  {label:<20s}  {total:>8.4f}  {correct:>8.2f}  {fmt:>8.2f}  {reasoning:>8.2f}")

    print()
    print("=" * 70)
    print("  奖励函数设计总结")
    print("=" * 70)
    print("""
  1. 答案正确性（权重 60%）
     - 最终判断标准：答案对了就是对的
     - 支持多种答案提取方式，提高鲁棒性
     - 这是 RLVR 中最重要的信号

  2. 格式规范性（权重 15%）
     - 鼓励模型输出结构化的推理过程
     - 包含步骤标记、答案标记、数学表达式
     - 不强制要求特定格式，但给予适当奖励

  3. 推理质量（权重 25%）
     - 鼓励详细的中间计算步骤
     - 检查逻辑连接词的使用
     - 检测明显的数学错误

  权重设计理念：
     - 答案正确性是最重要的（60%），但不是唯一的
     - 推理过程的质量也很重要（25%），防止模型"蒙对"
     - 格式规范给予少量奖励（15%），引导模型养成好习惯
    """)


# ==========================================
# 程序入口
# ==========================================
if __name__ == "__main__":
    run_tests()
