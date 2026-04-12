"""
第13章：思维树 (Tree of Thought) 推理实验
——探索 LLM 测试时推理的搜索策略

Tree of Thought (ToT) 是一种在推理时进行显式搜索的方法：
- 在每一步生成多个候选思路（广度优先）
- 用评估函数对每个思路打分
- 保留得分最高的 top-k 思路继续搜索
- 最终找到最优推理路径

本实验以"24点游戏"为任务，对比三种策略：
1. Chain-of-Thought (CoT): 单路径贪心推理
2. Tree of Thought (ToT): 多分支搜索 + 评分剪枝
3. Random: 随机尝试

运行方式：
    python tree_of_thought.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from copy import deepcopy

# 创建输出目录
os.makedirs("output", exist_ok=True)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 第一部分：24点游戏环境
# ==========================================
class TwentyFourGame:
    """
    24点游戏环境

    规则：
        - 给定4个数字（1~13）
        - 使用 +, -, *, / 四则运算
        - 每个数字恰好使用一次
        - 运算结果等于24

    本实现将问题建模为"逐步构建表达式"的过程：
        - 每一步选择两个剩余数字和一个运算符
        - 计算中间结果，作为新的"虚拟数字"
        - 重复直到只剩一个数字
    """

    # 四种运算
    OPERATIONS = ['+', '-', '*', '/']

    @staticmethod
    def apply_op(a, b, op):
        """对两个数执行运算，返回结果；除以零返回 None"""
        if op == '+':
            return a + b
        elif op == '-':
            return a - b
        elif op == '*':
            return a * b
        elif op == '/':
            if abs(b) < 1e-10:
                return None  # 除零保护
            return a / b

    @staticmethod
    def is_close_to_target(value, target=24.0, tol=1e-6):
        """判断是否接近目标值"""
        return abs(value - target) < tol


# ==========================================
# 第二部分：搜索节点与评分函数
# ==========================================
class ThoughtNode:
    """
    思维树节点

    每个节点代表一个"中间状态"：
        - numbers: 当前剩余的数字列表
        - history: 已执行的运算步骤
        - score: 该节点的评分（越高越好）
        - parent: 父节点
        - children: 子节点列表
    """

    def __init__(self, numbers, history=None, parent=None):
        self.numbers = list(numbers)  # 当前剩余数字
        self.history = history or []  # 运算历史
        self.parent = parent
        self.children = []
        self.score = 0.0  # 节点评分

    def is_terminal(self):
        """是否到达终点（只剩一个数字）"""
        return len(self.numbers) <= 1

    def get_expression(self):
        """返回完整的运算表达式"""
        return ' → '.join(self.history) if self.history else '初始状态'

    def __repr__(self):
        return f"Node(nums={self.numbers}, score={self.score:.2f})"


def evaluate_node(node, target=24.0):
    """
    评估节点的质量

    评分策略：
        - 终态（只剩1个数字）：结果越接近24分越高
        - 非终态（剩余多个数字）：基于剩余数字能否组合出目标的"潜力"

    这个评估函数模拟了 LLM 在 ToT 中的"自我评估"能力。
    实际系统中用 LLM 打分，这里用启发式规则替代。
    """
    if node.is_terminal():
        # 终态：直接看结果是否接近24
        diff = abs(node.numbers[0] - target)
        return max(0.0, 1.0 - diff / 50.0)  # 差距越小分数越高

    # 非终态：评估剩余数字组合出24的"潜力"
    # 启发式：看剩余数字中能否两两组合产生更接近目标的中间值
    nums = node.numbers
    best_potential = 0.0

    for i in range(len(nums)):
        for j in range(len(nums)):
            if i == j:
                continue
            for op in TwentyFourGame.OPERATIONS:
                result = TwentyFourGame.apply_op(nums[i], nums[j], op)
                if result is None:
                    continue
                # 构造新的剩余数字
                new_nums = [nums[k] for k in range(len(nums)) if k != i and k != j]
                new_nums.append(result)

                if len(new_nums) == 1:
                    # 只剩一个数，看是否接近24
                    potential = max(0.0, 1.0 - abs(new_nums[0] - target) / 50.0)
                else:
                    # 递归一步：看下一层的最优潜力
                    potential = 0.5 * max(0.0, 1.0 - abs(result - target) / 50.0)
                best_potential = max(best_potential, potential)

    return best_potential


def generate_children(node):
    """
    为节点生成所有可能的下一步

    从当前剩余数字中任选两个和一个运算符，
    产生一个新的中间状态。
    """
    children = []
    nums = node.numbers
    n = len(nums)

    if n < 2:
        return children

    for i in range(n):
        for j in range(n):
            if i == j:
                continue  # 不能选同一个数字
            for op in TwentyFourGame.OPERATIONS:
                result = TwentyFourGame.apply_op(nums[i], nums[j], op)
                if result is None:
                    continue  # 除零跳过

                # 构造新的剩余数字
                new_nums = [nums[k] for k in range(n) if k != i and k != j]
                new_nums.append(result)

                # 记录运算步骤
                step = f"{nums[i]} {op} {nums[j]} = {result:.2f}"
                new_history = node.history + [step]

                child = ThoughtNode(new_nums, new_history, parent=node)
                children.append(child)

    return children


# ==========================================
# 第三部分：三种搜索策略
# ==========================================
def search_tree_of_thought(numbers, breadth=3, max_depth=4, target=24.0, verbose=True):
    """
    Tree of Thought 搜索

    核心思想：
        1. 在每一层生成所有候选节点（分支）
        2. 用评估函数对每个节点打分
        3. 只保留得分最高的 breadth 个节点
        4. 继续展开下一层

    参数：
        numbers: 初始数字列表
        breadth: 每层保留的候选数量（束宽）
        max_depth: 最大搜索深度
        target: 目标值
        verbose: 是否打印详细过程

    返回：
        best_node: 找到的最优节点
        tree_data: 搜索树数据（用于可视化）
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Tree of Thought 搜索 (breadth={breadth})")
        print(f"  初始数字: {numbers}, 目标: {target}")
        print(f"{'='*60}")

    root = ThoughtNode(numbers)
    current_beam = [root]  # 当前层的候选节点
    tree_data = {'nodes_per_level': [], 'scores_per_level': []}
    total_evaluated = 0

    for depth in range(max_depth):
        if verbose:
            print(f"\n--- 第 {depth + 1} 层展开 ---")

        # 第一步：展开当前层的所有节点，生成子节点
        all_children = []
        for node in current_beam:
            children = generate_children(node)
            # 为每个子节点评分
            for child in children:
                child.score = evaluate_node(child, target)
                total_evaluated += 1
            all_children.extend(children)

        if not all_children:
            if verbose:
                print("  没有更多可展开的节点")
            break

        # 第二步：按得分排序，保留 top-k
        all_children.sort(key=lambda x: x.score, reverse=True)
        current_beam = all_children[:breadth]

        # 记录搜索树数据
        level_nodes = [c.get_expression() for c in current_beam]
        level_scores = [c.score for c in current_beam]
        tree_data['nodes_per_level'].append(level_nodes)
        tree_data['scores_per_level'].append(level_scores)

        if verbose:
            print(f"  生成了 {len(all_children)} 个候选节点（共评估 {total_evaluated} 次）")
            print(f"  保留 top-{breadth}:")
            for idx, node in enumerate(current_beam):
                nums_str = ', '.join([f"{n:.1f}" for n in node.numbers])
                print(f"    [{idx+1}] 分数={node.score:.3f} | 剩余=[{nums_str}] | {node.get_expression()}")

        # 检查是否已找到精确解
        for node in current_beam:
            if (node.is_terminal()
                    and TwentyFourGame.is_close_to_target(node.numbers[0], target)):
                if verbose:
                    print(f"\n  *** 找到精确解！***")
                    print(f"  结果 = {node.numbers[0]:.4f}")
                    print(f"  推理路径: {node.get_expression()}")
                return node, tree_data

    # 没找到精确解，返回最接近的
    best_node = max(current_beam, key=lambda x: x.score)
    if verbose:
        if best_node.is_terminal():
            print(f"\n  未找到精确解，最接近的结果:")
            print(f"  结果 = {best_node.numbers[0]:.4f}")
            print(f"  推理路径: {best_node.get_expression()}")
        else:
            print(f"\n  搜索未完成（达到最大深度）")
            print(f"  当前最优节点: {best_node}")

    return best_node, tree_data


def search_chain_of_thought(numbers, target=24.0, verbose=True):
    """
    Chain-of-Thought (CoT) 搜索

    核心思想：
        在每一步只保留一个最优节点（贪心），
        等价于 breadth=1 的 ToT。

    这模拟了标准 CoT 的"单链推理"模式。
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Chain-of-Thought 推理 (贪心单路径)")
        print(f"  初始数字: {numbers}, 目标: {target}")
        print(f"{'='*60}")

    # CoT 等价于 breadth=1 的 ToT
    result, _ = search_tree_of_thought(
        numbers, breadth=1, max_depth=4, target=target, verbose=verbose
    )
    return result


def search_random(numbers, n_trials=50, target=24.0, verbose=True):
    """
    随机搜索（基线对比）

    核心思想：
        随机选择数字对和运算符，
        多次尝试，记录最佳结果。
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"  随机搜索 (尝试 {n_trials} 次)")
        print(f"  初始数字: {numbers}, 目标: {target}")
        print(f"{'='*60}")

    best_result = None
    best_diff = float('inf')
    best_history = []
    successes = 0

    for trial in range(n_trials):
        remaining = list(numbers)
        history = []

        for step in range(3):  # 4个数字需要3步运算
            if len(remaining) < 2:
                break

            # 随机选择两个不同位置的数字
            indices = np.random.choice(len(remaining), 2, replace=False)
            i, j = indices
            op = np.random.choice(TwentyFourGame.OPERATIONS)

            result = TwentyFourGame.apply_op(remaining[i], remaining[j], op)
            if result is None:
                break

            step_str = f"{remaining[i]} {op} {remaining[j]} = {result:.2f}"
            history.append(step_str)

            # 更新剩余数字
            new_remaining = [remaining[k] for k in range(len(remaining))
                             if k != i and k != j]
            new_remaining.append(result)
            remaining = new_remaining

        if len(remaining) == 1:
            diff = abs(remaining[0] - target)
            if diff < best_diff:
                best_diff = diff
                best_result = remaining[0]
                best_history = history
            if TwentyFourGame.is_close_to_target(remaining[0], target):
                successes += 1

    if verbose:
        if best_result is not None:
            print(f"  随机搜索最佳结果: {best_result:.4f} (误差={best_diff:.4f})")
            print(f"  成功找到精确解: {successes}/{n_trials} 次")
        else:
            print(f"  随机搜索未找到有效结果")

    return best_result, best_diff, successes


# ==========================================
# 第四部分：可视化函数
# ==========================================
def visualize_search_tree(tree_data, title="Tree of Thought 搜索树"):
    """
    可视化搜索树

    每一层显示保留的节点及其评分，
    节点大小和颜色反映评分高低。
    """
    n_levels = len(tree_data['nodes_per_level'])
    if n_levels == 0:
        print("没有搜索树数据可可视化")
        return

    fig, ax = plt.subplots(figsize=(16, 8))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # 层间距离
    y_spacing = 1.0
    max_score = 0.0

    for level_scores in tree_data['scores_per_level']:
        for s in level_scores:
            max_score = max(max_score, s)

    for level in range(n_levels):
        nodes = tree_data['nodes_per_level'][level]
        scores = tree_data['scores_per_level'][level]
        n_nodes = len(nodes)

        # 当前层的 y 坐标
        y = (n_levels - 1 - level) * y_spacing

        # 水平均匀分布
        x_positions = np.linspace(0.5, n_nodes - 0.5, n_nodes)

        for i, (node_expr, score) in enumerate(zip(nodes, scores)):
            x = x_positions[i] if n_nodes > 1 else 0.5

            # 节点大小和颜色由评分决定
            size = 200 + score * 800
            color_val = score / max(max_score, 0.01)
            color = plt.cm.RdYlGn(color_val)

            # 绘制节点
            ax.scatter(x, y, s=size, c=[color], edgecolors='black',
                       linewidths=1.5, zorder=5)

            # 标注评分
            ax.text(x, y + 0.15, f'{score:.2f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

            # 标注节点信息（简化显示）
            expr_short = node_expr.split(' → ')[-1] if ' → ' in node_expr else node_expr
            ax.text(x, y - 0.15, expr_short, ha='center', va='top',
                    fontsize=7, color='#333333',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow',
                              alpha=0.7))

        # 绘制与上一层的连线
        if level > 0:
            prev_nodes = tree_data['nodes_per_level'][level - 1]
            prev_scores = tree_data['scores_per_level'][level - 1]
            prev_n = len(prev_nodes)
            prev_x = np.linspace(0.5, prev_n - 0.5, prev_n) if prev_n > 1 else [0.5]
            prev_y = (n_levels - level) * y_spacing

            for pi in range(prev_n):
                for ci in range(n_nodes):
                    ax.plot([prev_x[pi], x_positions[ci]],
                            [prev_y, y],
                            color='gray', alpha=0.2, linewidth=0.8, zorder=1)

    # 设置坐标轴
    ax.set_ylabel('搜索深度', fontsize=12)
    ax.set_xticks([])
    y_ticks = [(n_levels - 1 - l) * y_spacing for l in range(n_levels)]
    y_labels = [f'第{l+1}层' for l in range(n_levels)]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    return fig


def visualize_comparison(results, problem_labels):
    """
    可视化三种策略的对比结果

    参数：
        results: 字典，包含三种策略在不同问题上的表现
        problem_labels: 问题标签列表
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('思维树 vs 思维链 vs 随机搜索 — 对比实验', fontsize=16, fontweight='bold')

    colors = ['#4CAF50', '#2196F3', '#FF9800']
    strategies = ['Tree of Thought', 'Chain of Thought', 'Random']
    keys = ['tot', 'cot', 'random']

    # ---- 子图1：成功率 ----
    ax1 = axes[0]
    success_rates = [results[k]['success_rate'] for k in keys]
    bars = ax1.bar(strategies, success_rates, color=colors, edgecolor='black', linewidth=0.8)
    ax1.set_ylabel('成功率', fontsize=12)
    ax1.set_title('求解成功率', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 1.1)
    for bar, rate in zip(bars, success_rates):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{rate:.0%}', ha='center', fontsize=11, fontweight='bold')
    ax1.grid(True, axis='y', alpha=0.3)

    # ---- 子图2：平均误差 ----
    ax2 = axes[1]
    avg_errors = [results[k]['avg_error'] for k in keys]
    bars = ax2.bar(strategies, avg_errors, color=colors, edgecolor='black', linewidth=0.8)
    ax2.set_ylabel('平均误差', fontsize=12)
    ax2.set_title('与目标(24)的平均误差', fontsize=13, fontweight='bold')
    for bar, err in zip(bars, avg_errors):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f'{err:.2f}', ha='center', fontsize=11, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)

    # ---- 子图3：搜索宽度对成功率的影响 ----
    ax3 = axes[2]
    breadths = [1, 2, 3, 5, 8]
    # 重新跑不同宽度的实验
    success_by_breadth = results['tot']['breadth_success_rates']
    ax3.plot(breadths[:len(success_by_breadth)], success_by_breadth,
             'o-', color='#4CAF50', linewidth=2.5, markersize=10, label='ToT 成功率')
    ax3.axhline(y=results['cot']['success_rate'], color='#2196F3',
                linestyle='--', linewidth=2, label='CoT (breadth=1)')
    ax3.axhline(y=results['random']['success_rate'], color='#FF9800',
                linestyle='--', linewidth=2, label='Random')
    ax3.set_xlabel('搜索宽度 (breadth)', fontsize=12)
    ax3.set_ylabel('成功率', fontsize=12)
    ax3.set_title('搜索宽度对 ToT 性能的影响', fontsize=13, fontweight='bold')
    ax3.set_ylim(0, 1.1)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(breadths[:len(success_by_breadth)])

    plt.tight_layout()
    return fig


# ==========================================
# 第五部分：实验主流程
# ==========================================
def run_experiment():
    """
    运行完整的对比实验

    实验流程：
        1. 准备多个24点问题
        2. 分别用 ToT、CoT、Random 三种策略求解
        3. 统计成功率、平均误差
        4. 测试不同搜索宽度对 ToT 的影响
        5. 可视化对比
    """

    # 精心挑选的24点题目（都确保有解）
    problems = [
        [1, 2, 3, 4],
        [2, 3, 4, 6],
        [1, 5, 5, 5],
        [3, 3, 8, 8],
        [4, 4, 10, 10],
        [1, 4, 5, 6],
        [2, 6, 7, 7],
        [3, 6, 6, 8],
        [2, 3, 5, 7],
        [1, 3, 4, 6],
    ]

    target = 24.0

    print("=" * 60)
    print("  第13章：思维树 (Tree of Thought) 推理实验")
    print("=" * 60)
    print(f"  任务: 24点游戏 — 用4个数字凑出 {target}")
    print(f"  题目数量: {len(problems)}")
    print(f"  对比策略: Tree of Thought, Chain of Thought, Random")
    print("-" * 60)

    # ---- 策略1: Tree of Thought (breadth=3) ----
    print("\n" + "=" * 60)
    print("  策略1: Tree of Thought (breadth=3)")
    print("=" * 60)

    tot_successes = 0
    tot_errors = []
    # 记录第一个问题的搜索树用于详细展示
    first_tree_data = None

    for idx, nums in enumerate(problems):
        result, tree_data = search_tree_of_thought(
            nums, breadth=3, max_depth=4, target=target, verbose=(idx == 0)
        )
        if idx == 0:
            first_tree_data = tree_data

        if result.is_terminal():
            error = abs(result.numbers[0] - target)
            tot_errors.append(error)
            if TwentyFourGame.is_close_to_target(result.numbers[0], target):
                tot_successes += 1
                print(f"  题目 {idx+1} {nums}: 成功! 结果={result.numbers[0]:.2f}")
            else:
                print(f"  题目 {idx+1} {nums}: 未达目标, 结果={result.numbers[0]:.2f} (误差={error:.2f})")
        else:
            tot_errors.append(abs(target))  # 未完成的视为最大误差
            print(f"  题目 {idx+1} {nums}: 搜索未完成")

    tot_success_rate = tot_successes / len(problems)
    tot_avg_error = np.mean(tot_errors)

    # ---- 策略2: Chain of Thought (breadth=1) ----
    print("\n" + "=" * 60)
    print("  策略2: Chain of Thought (贪心单路径)")
    print("=" * 60)

    cot_successes = 0
    cot_errors = []

    for idx, nums in enumerate(problems):
        result = search_chain_of_thought(nums, target=target, verbose=(idx == 0))
        if result.is_terminal():
            error = abs(result.numbers[0] - target)
            cot_errors.append(error)
            if TwentyFourGame.is_close_to_target(result.numbers[0], target):
                cot_successes += 1
                print(f"  题目 {idx+1} {nums}: 成功! 结果={result.numbers[0]:.2f}")
            else:
                print(f"  题目 {idx+1} {nums}: 未达目标, 结果={result.numbers[0]:.2f} (误差={error:.2f})")
        else:
            cot_errors.append(abs(target))
            print(f"  题目 {idx+1} {nums}: 搜索未完成")

    cot_success_rate = cot_successes / len(problems)
    cot_avg_error = np.mean(cot_errors)

    # ---- 策略3: Random ----
    print("\n" + "=" * 60)
    print("  策略3: 随机搜索基线")
    print("=" * 60)

    random_successes = 0
    random_errors = []

    for idx, nums in enumerate(problems):
        best_result, best_diff, successes = search_random(
            nums, n_trials=50, target=target, verbose=(idx == 0)
        )
        if best_result is not None:
            random_errors.append(best_diff)
            if TwentyFourGame.is_close_to_target(best_result, target):
                random_successes += 1
                print(f"  题目 {idx+1} {nums}: 成功! 结果={best_result:.2f}")
            else:
                print(f"  题目 {idx+1} {nums}: 最佳={best_result:.2f} (误差={best_diff:.2f})")
        else:
            random_errors.append(abs(target))
            print(f"  题目 {idx+1} {nums}: 未找到有效结果")

    random_success_rate = random_successes / len(problems)
    random_avg_error = np.mean(random_errors)

    # ---- 不同搜索宽度的 ToT 实验 ----
    print("\n" + "=" * 60)
    print("  不同搜索宽度的 ToT 成功率")
    print("=" * 60)

    breadth_values = [1, 2, 3, 5, 8]
    breadth_success_rates = []

    for b in breadth_values:
        successes = 0
        for nums in problems:
            result, _ = search_tree_of_thought(
                nums, breadth=b, max_depth=4, target=target, verbose=False
            )
            if (result.is_terminal()
                    and TwentyFourGame.is_close_to_target(result.numbers[0], target)):
                successes += 1
        rate = successes / len(problems)
        breadth_success_rates.append(rate)
        print(f"  breadth={b}: 成功率 = {rate:.0%} ({successes}/{len(problems)})")

    # ---- 汇总结果 ----
    print("\n" + "=" * 60)
    print("  实验结果汇总")
    print("=" * 60)
    print(f"  {'策略':<25} {'成功率':>10} {'平均误差':>12}")
    print(f"  {'-'*47}")
    print(f"  {'Tree of Thought (b=3)':<25} {tot_success_rate:>9.0%} {tot_avg_error:>12.2f}")
    print(f"  {'Chain of Thought (b=1)':<25} {cot_success_rate:>9.0%} {cot_avg_error:>12.2f}")
    print(f"  {'Random (50次)':<25} {random_success_rate:>9.0%} {random_avg_error:>12.2f}")
    print("-" * 60)

    # 打印关键结论
    print("\n关键结论:")
    print("  1. ToT 通过多分支搜索 + 评分剪枝，显著提高求解成功率")
    print("  2. CoT (breadth=1) 是贪心策略，容易陷入局部最优")
    print("  3. 增大搜索宽度能提高成功率，但计算量也增大")
    print("  4. 这展示了测试时搜索(Test-Time Search)的力量：")
    print("     在推理阶段投入更多计算可以获得更好的结果")
    print("  5. ToT 的核心思想可迁移到 LLM 的推理优化中")

    # ---- 可视化 ----
    results = {
        'tot': {
            'success_rate': tot_success_rate,
            'avg_error': tot_avg_error,
            'breadth_success_rates': breadth_success_rates,
        },
        'cot': {
            'success_rate': cot_success_rate,
            'avg_error': cot_avg_error,
        },
        'random': {
            'success_rate': random_success_rate,
            'avg_error': random_avg_error,
        },
    }

    # 图1: 搜索树可视化
    if first_tree_data:
        fig1 = visualize_search_tree(first_tree_data,
                                     title="Tree of Thought 搜索过程（第1题）")
        fig1.savefig('output/tot_search_tree.png', dpi=150, bbox_inches='tight')
        print("\n搜索树可视化已保存至: output/tot_search_tree.png")

    # 图2: 对比实验
    problem_labels = [str(p) for p in problems]
    fig2 = visualize_comparison(results, problem_labels)
    fig2.savefig('output/tot_comparison.png', dpi=150, bbox_inches='tight')
    print("对比实验图已保存至: output/tot_comparison.png")

    plt.show()


# ==========================================
# 第六部分：主程序入口
# ==========================================
if __name__ == "__main__":
    # 设置随机种子，确保可复现
    np.random.seed(42)

    # 运行实验
    run_experiment()
