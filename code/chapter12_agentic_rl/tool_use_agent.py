"""
第12章：工具调用 Agent 的强化学习训练
==========================================================

本脚本模拟一个工具调用 Agent，通过 REINFORCE 算法学习
在不同用户查询下选择正确的工具。

场景设定：
  - 用户提出各种问题
  - Agent 需要从 3 个工具中选择合适的一个：
      1. search(query)      —— 知识搜索
      2. calculate(expr)    —— 数学计算
      3. run_code(code)     —— 代码执行
  - 选择正确的工具 → 正奖励 (+1)
  - 选择错误的工具 → 负奖励 (-0.1)

训练方法：
  - 策略：简单的概率分布（softmax 参数化）
  - 算法：REINFORCE（蒙特卡洛策略梯度）
  - 训练 50 个 episode，观察策略的进化过程

运行方式：
    python tool_use_agent.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# 创建输出目录
os.makedirs("output", exist_ok=True)

# 设置中文字体，确保图表标题和标签正常显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 第一部分：定义工具和查询数据集
# ==========================================

# 三个可用工具
TOOL_NAMES = ["search", "calculate", "run_code"]
TOOL_DESCRIPTIONS = {
    "search": "搜索引擎，适合回答知识性问题",
    "calculate": "数学计算器，适合数值运算",
    "run_code": "代码执行器，适合编程相关任务",
}

# 训练数据集：每条包含问题文本和正确工具
# 设计思路：问题类型分布均匀（各约 1/3），增加多样性
TRAINING_QUERIES = [
    # ---- search 类型 ----
    {"query": "中国的首都是哪里？",                     "correct_tool": "search"},
    {"query": "Python 是什么时候发明的？",               "correct_tool": "search"},
    {"query": "光速是多少？",                           "correct_tool": "search"},
    {"query": "法国的人口有多少？",                      "correct_tool": "search"},
    {"query": "什么是量子计算？",                        "correct_tool": "search"},
    {"query": "第二次世界大战什么时候结束？",             "correct_tool": "search"},
    {"query": "地球到月球的距离是多少？",                "correct_tool": "search"},
    {"query": "谁发明了电话？",                          "correct_tool": "search"},
    # ---- calculate 类型 ----
    {"query": "请计算 123 + 456",                       "correct_tool": "calculate"},
    {"query": "25 乘以 37 等于多少？",                   "correct_tool": "calculate"},
    {"query": "1024 除以 8 是多少？",                   "correct_tool": "calculate"},
    {"query": "求 17 的平方根",                          "correct_tool": "calculate"},
    {"query": "计算圆的面积，半径为 5",                  "correct_tool": "calculate"},
    {"query": "3 的 10 次方是多少？",                    "correct_tool": "calculate"},
    {"query": "99 乘法表中 7×8 是多少？",               "correct_tool": "calculate"},
    {"query": "把 1024 转换成二进制",                    "correct_tool": "calculate"},
    # ---- run_code 类型 ----
    {"query": "帮我写一个冒泡排序",                      "correct_tool": "run_code"},
    {"query": "写一个 Python 函数判断回文",              "correct_tool": "run_code"},
    {"query": "执行这段排序代码并输出结果",              "correct_tool": "run_code"},
    {"query": "帮我调试这段代码的语法错误",              "correct_tool": "run_code"},
    {"query": "写一个爬虫抓取网页标题",                  "correct_tool": "run_code"},
    {"query": "实现一个简单的 REST API",                 "correct_tool": "run_code"},
    {"query": "运行这段数据分析脚本",                    "correct_tool": "run_code"},
    {"query": "帮我写一个单元测试",                      "correct_tool": "run_code"},
]


def simulate_tool_result(tool_name, query, correct_tool):
    """
    模拟工具执行结果

    参数：
        tool_name:     实际调用的工具名
        query:         用户查询文本
        correct_tool:  正确的工具名
    返回：
        result_dict:   包含结果文本和正确性标志
    """
    if tool_name == correct_tool:
        # 选对了工具
        return {"success": True, "message": f"使用 {tool_name} 成功处理查询"}
    else:
        # 选错了工具
        return {"success": False, "message": f"工具 {tool_name} 不适合处理此查询"}


# ==========================================
# 第二部分：策略参数化
# ==========================================

class ToolPolicy:
    """
    简单的策略模型：为每个查询类型维护工具选择概率

    策略参数化方式：
      - 使用 logits（未归一化分数）表示对每个工具的偏好
      - 通过 softmax 将 logits 转换为概率分布
      - 训练过程就是调整 logits 的值

    直觉理解：
      - 如果一个工具经常获得正奖励，它的 logit 会增大
      - 如果一个工具经常获得负奖励，它的 logit 会减小
      - softmax 保证概率之和始终为 1

    参数：
        n_tools: 可用工具数量
        learning_rate: 学习率
    """

    def __init__(self, n_tools=3, learning_rate=0.05):
        self.n_tools = n_tools
        self.learning_rate = learning_rate

        # 初始化 logits：全为零 → 等概率选择（1/3, 1/3, 1/3）
        self.query_type_logits = {
            "search":    np.zeros(n_tools),
            "calculate": np.zeros(n_tools),
            "run_code":  np.zeros(n_tools),
        }

    def get_query_type(self, query):
        """
        根据查询内容判断类型（简化版分类器）

        在实际系统中，这会是一个 NLP 分类器。
        这里用关键词匹配模拟：
          - 包含数学关键词 → calculate
          - 包含编程关键词 → run_code
          - 其他 → search
        """
        calc_keywords = ["计算", "乘", "除", "平方", "面积", "次方", "等于多少",
                         "加", "减", "根", "乘法", "进制"]
        code_keywords = ["写", "执行", "代码", "函数", "调试", "爬虫", "API",
                         "排序", "运行", "测试", "编程", "脚本"]

        for kw in calc_keywords:
            if kw in query:
                return "calculate"
        for kw in code_keywords:
            if kw in query:
                return "run_code"
        return "search"

    def get_probabilities(self, query_type):
        """
        获取指定查询类型下的工具选择概率

        使用 softmax 将 logits 转为概率：
          π(a|s) = exp(logit_a) / Σ exp(logit_i)
        """
        logits = self.query_type_logits[query_type]
        # 数值稳定的 softmax
        logits_shifted = logits - np.max(logits)
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / np.sum(exp_logits)
        return probs

    def sample_action(self, query_type):
        """
        按照当前策略采样一个动作（工具选择）

        不是取 argmax！采样是探索的关键。
        即使某个工具概率最高，其他工具也有机会被选中。
        """
        probs = self.get_probabilities(query_type)
        action = np.random.choice(self.n_tools, p=probs)
        return action, probs[action]

    def update(self, query_type, action, reward):
        """
        REINFORCE 策略梯度更新

        核心公式：
          θ ← θ + α * ∇log π(a|s) * G

        对于 softmax 策略，梯度为：
          ∇log π(a_k|s) = e_k - π(s)
          其中 e_k 是 one-hot 向量，π(s) 是概率向量

        直觉理解：
          - 如果 reward > 0：增大被选中工具的概率
          - 如果 reward < 0：减小被选中工具的概率
        """
        probs = self.get_probabilities(query_type)

        # 计算对数概率的梯度
        # ∇log π(a_k|s) = e_k - π(s)
        grad = -probs.copy()       # -π(s)
        grad[action] += 1.0        # +e_k

        # 策略梯度更新：θ ← θ + α * grad * reward
        self.query_type_logits[query_type] += self.learning_rate * grad * reward


# ==========================================
# 第三部分：训练循环
# ==========================================

def train(policy, n_episodes=50, queries_per_episode=8):
    """
    REINFORCE 训练主循环

    每个 episode：
      1. 随机采样一批查询
      2. Agent 为每个查询选择工具
      3. 获得奖励（选对 +1.0，选错 -0.1）
      4. 立即更新策略参数（在线学习）

    参数：
        policy:             策略对象
        n_episodes:         训练回合数
        queries_per_episode: 每回合采样的查询数
    返回：
        history: 训练历史记录
    """
    # 记录训练过程
    history = {
        "episode_rewards": [],      # 每回合平均奖励
        "episode_accuracy": [],     # 每回合准确率
        "tool_probs_history": {     # 每回合的工具选择概率
            "search": [],
            "calculate": [],
            "run_code": [],
        },
    }

    for episode in range(n_episodes):
        episode_reward = 0.0
        correct_count = 0

        # 随机采样查询
        indices = np.random.choice(len(TRAINING_QUERIES),
                                   size=min(queries_per_episode, len(TRAINING_QUERIES)),
                                   replace=False)

        for idx in indices:
            query_data = TRAINING_QUERIES[idx]
            query = query_data["query"]
            correct_tool = query_data["correct_tool"]

            # 第一步：判断查询类型
            query_type = policy.get_query_type(query)

            # 第二步：按策略采样工具
            action, prob = policy.sample_action(query_type)
            chosen_tool = TOOL_NAMES[action]

            # 第三步：执行工具，获得奖励
            result = simulate_tool_result(chosen_tool, query, correct_tool)

            if result["success"]:
                reward = 1.0    # 选对了：正奖励
                correct_count += 1
            else:
                reward = -0.1   # 选错了：负奖励（小惩罚）

            # 第四步：更新策略
            policy.update(query_type, action, reward)
            episode_reward += reward

        # 记录本轮统计
        avg_reward = episode_reward / queries_per_episode
        accuracy = correct_count / queries_per_episode
        history["episode_rewards"].append(avg_reward)
        history["episode_accuracy"].append(accuracy)

        # 记录当前每种查询类型的工具选择概率
        for qt in ["search", "calculate", "run_code"]:
            probs = policy.get_probabilities(qt)
            history["tool_probs_history"][qt].append(probs.copy())

        # 每 10 回合打印进度
        if (episode + 1) % 10 == 0:
            print(f"  回合 {episode+1:3d}/{n_episodes} | "
                  f"平均奖励: {avg_reward:+.3f} | "
                  f"准确率: {accuracy:.1%}")

    return history


# ==========================================
# 第四部分：运行训练
# ==========================================
print("=" * 70)
print("  第12章：工具调用 Agent 的强化学习训练")
print("=" * 70)

np.random.seed(42)  # 固定随机种子

# 初始化策略
policy = ToolPolicy(n_tools=3, learning_rate=0.05)

# ---- 训练前测试 ----
print("\n【训练前】工具选择概率（随机初始化，均匀分布）:")
print(f"  {'查询类型':<12} {'search':<12} {'calculate':<12} {'run_code':<12}")
print(f"  {'─' * 48}")
for qt in ["search", "calculate", "run_code"]:
    probs = policy.get_probabilities(qt)
    print(f"  {qt:<12} {probs[0]:<12.4f} {probs[1]:<12.4f} {probs[2]:<12.4f}")

# 快速测试训练前准确率
correct_before = 0
total_before = len(TRAINING_QUERIES)
for qd in TRAINING_QUERIES:
    qt = policy.get_query_type(qd["query"])
    action, _ = policy.sample_action(qt)
    if TOOL_NAMES[action] == qd["correct_tool"]:
        correct_before += 1
accuracy_before = correct_before / total_before
print(f"\n  训练前准确率: {correct_before}/{total_before} = {accuracy_before:.1%}")

# ---- 开始训练 ----
print("\n" + "─" * 70)
print("  开始 REINFORCE 训练（50 回合）")
print("─" * 70)

history = train(policy, n_episodes=50, queries_per_episode=8)

# ---- 训练后测试 ----
print("\n" + "─" * 70)
print("  训练完成！")
print("─" * 70)

print("\n【训练后】工具选择概率（应该趋近正确分配）:")
print(f"  {'查询类型':<12} {'search':<12} {'calculate':<12} {'run_code':<12}  {'最优工具'}")
print(f"  {'─' * 64}")
for qt, optimal in [("search", "search"), ("calculate", "calculate"), ("run_code", "run_code")]:
    probs = policy.get_probabilities(qt)
    print(f"  {qt:<12} {probs[0]:<12.4f} {probs[1]:<12.4f} {probs[2]:<12.4f}  {optimal}")

# 测试训练后准确率（用确定性策略：选概率最大的工具）
correct_after = 0
for qd in TRAINING_QUERIES:
    qt = policy.get_query_type(qd["query"])
    probs = policy.get_probabilities(qt)
    best_action = np.argmax(probs)
    if TOOL_NAMES[best_action] == qd["correct_tool"]:
        correct_after += 1
accuracy_after = correct_after / total_before
print(f"\n  训练后准确率（确定性策略）: {correct_after}/{total_before} = {accuracy_after:.1%}")
print(f"  准确率提升: {accuracy_after - accuracy_before:+.1%}")

# ---- 逐条展示训练后预测 ----
print("\n【逐条预测展示】:")
print(f"  {'问题':<30} {'正确工具':<10} {'预测工具':<10} {'结果'}")
print(f"  {'─' * 65}")
for qd in TRAINING_QUERIES:
    qt = policy.get_query_type(qd["query"])
    probs = policy.get_probabilities(qt)
    best_action = np.argmax(probs)
    predicted_tool = TOOL_NAMES[best_action]
    correct = predicted_tool == qd["correct_tool"]
    mark = "正确" if correct else "错误"
    query_short = qd["query"][:28] + ".." if len(qd["query"]) > 28 else qd["query"]
    print(f"  {query_short:<30} {qd['correct_tool']:<10} {predicted_tool:<10} {mark}")


# ==========================================
# 第五部分：可视化
# ==========================================
print("\n正在生成可视化图表...")

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("工具调用 Agent —— REINFORCE 训练过程", fontsize=18, fontweight="bold")

# ---- 子图1：工具选择概率进化 ----
ax1 = axes[0]

episodes = np.arange(1, len(history["episode_accuracy"]) + 1)

# 提取每种查询类型的工具选择概率变化
for qt_idx, (qt, color, marker) in enumerate([
    ("search",    "#2196F3", "o"),
    ("calculate", "#FF9800", "s"),
    ("run_code",  "#4CAF50", "^"),
]):
    probs_history = np.array(history["tool_probs_history"][qt])
    # 画出该查询类型下，正确工具被选中的概率
    ax1.plot(episodes, probs_history[:, qt_idx],
             marker=marker, linewidth=2.5, markersize=6,
             color=color, label=f"{qt} 类查询 → 选 {TOOL_NAMES[qt_idx]}")

ax1.axhline(y=1/3, color="gray", linestyle="--", alpha=0.5, label="随机基线 (1/3)")
ax1.set_title("工具选择概率进化过程", fontsize=14, fontweight="bold")
ax1.set_xlabel("训练回合", fontsize=12)
ax1.set_ylabel("正确工具被选中的概率", fontsize=12)
ax1.legend(fontsize=10, loc="center right")
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1.05)

# 添加注释
ax1.annotate("训练前：均匀分布 (~33%)",
             xy=(1, 1/3), xytext=(8, 0.15),
             fontsize=10, color="gray",
             arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
ax1.annotate("训练后：收敛到正确分配",
             xy=(50, 0.9), xytext=(30, 0.95),
             fontsize=10, color="green", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="green", lw=1.5))

# ---- 子图2：准确率曲线 ----
ax2 = axes[1]

ax2.plot(episodes, history["episode_accuracy"],
         linewidth=1.5, alpha=0.4, color="steelblue", label="回合准确率（原始）")

# 滑动平均
window = 5
if len(history["episode_accuracy"]) >= window:
    moving_avg = []
    for i in range(len(history["episode_accuracy"])):
        start = max(0, i - window + 1)
        moving_avg.append(np.mean(history["episode_accuracy"][start:i+1]))
    ax2.plot(episodes, moving_avg, color="crimson", linewidth=2.5,
             label=f"滑动平均（窗口={window}）")

ax2.axhline(y=1/3, color="gray", linestyle="--", alpha=0.5, label="随机基线 (33.3%)")
ax2.axhline(y=accuracy_after, color="green", linestyle=":", alpha=0.7,
            label=f"最终准确率 ({accuracy_after:.1%})")

ax2.set_title("工具选择准确率变化", fontsize=14, fontweight="bold")
ax2.set_xlabel("训练回合", fontsize=12)
ax2.set_ylabel("准确率", fontsize=12)
ax2.legend(fontsize=10, loc="lower right")
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig("output/tool_use_agent_training.png", dpi=150, bbox_inches="tight")
print("图表已保存至: output/tool_use_agent_training.png")
plt.show()


# ==========================================
# 第六部分：总结
# ==========================================
print("\n" + "=" * 70)
print("  关键结论")
print("=" * 70)
print(f"""
  1. 工具调用是 Agentic RL 的核心能力
     - Agent 必须学会根据用户意图选择正确的工具
     - 这是从"被动回答"到"主动行动"的关键跨越

  2. REINFORCE 算法能有效学习工具选择策略
     - 训练前：均匀分布，准确率约 {accuracy_before:.1%}
     - 训练后：收敛到正确分配，准确率提升至 {accuracy_after:.1%}
     - 仅需 50 个 episode 即可学到合理策略

  3. 实际系统中的扩展方向：
     - 更复杂的策略网络（如 Transformer）
     - 多步工具链（串联多个工具完成复杂任务）
     - 过程奖励模型（PRM）引导中间步骤
     - 工具参数的学习（不仅选工具，还要构造参数）

  4. 从单工具到多工具链：
     - 本实验：每个查询只选一个工具
     - 进阶：Agent 需要规划多步工具调用序列
     - 这就是 multi_turn_rl.py 中讨论的多轮信用分配问题
""")
print("=" * 70)
