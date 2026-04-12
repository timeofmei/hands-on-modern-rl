"""
第12章：多轮对话 RL —— ORM 与 PRM 的信用分配对比
==========================================================

本脚本模拟一个多轮工具调用 Agent（每回合 3~5 轮对话），
对比两种奖励分配策略：

  1. ORM（Outcome Reward Model）：
     只有最终结果获得奖励（1.0 或 0.0），中间步骤无信号

  2. PRM（Process Reward Model）：
     每一步都获得部分奖励（0.0~1.0），及时提供学习信号

核心概念：
  - 信用分配问题（Credit Assignment）：如何将最终奖励归因到每一步？
  - 折扣回报：G_t = r_t + γ * G_{t+1}，γ 控制信用传播范围
  - γ 越大 → 远期信用传播越远；γ 越小 → 只关注近期步骤

运行方式：
    python multi_turn_rl.py
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
# 第一部分：定义模拟工具
# ==========================================
# Agent 可以调用三个工具来完成用户任务：
#   - calculator: 数学计算器
#   - search:      知识搜索
#   - code_executor: 代码执行器
# 每个工具返回一个模拟结果和正确性分数

def tool_calculator(query):
    """
    模拟计算器工具
    正确处理数学表达式，偶尔模拟计算错误（增加真实性）
    """
    # 简单模拟：对常见数学问题返回结果
    if "123 + 456" in query:
        return {"result": "579", "correct": True}
    elif "25 * 4" in query:
        return {"result": "100", "correct": True}
    elif "sqrt(144)" in query or "144" in query:
        return {"result": "12", "correct": True}
    else:
        # 未知计算，模拟 70% 正确率
        correct = np.random.random() < 0.7
        return {"result": "模拟结果", "correct": correct}


def tool_search(query):
    """
    模拟搜索工具
    返回知识检索结果
    """
    # 模拟搜索质量
    if "Python" in query or "python" in query:
        return {"result": "Python 是一种广泛使用的高级编程语言...", "correct": True}
    elif "RL" in query or "强化学习" in query:
        return {"result": "强化学习是机器学习的一个分支...", "correct": True}
    else:
        correct = np.random.random() < 0.7
        return {"result": "搜索到相关内容...", "correct": correct}


def tool_code_executor(code):
    """
    模拟代码执行器
    执行代码并返回运行结果
    """
    # 模拟代码执行成功率
    if "print" in code or "def " in code:
        return {"result": "代码执行成功", "correct": True}
    elif "import" in code:
        return {"result": "模块导入成功", "correct": True}
    else:
        correct = np.random.random() < 0.6
        return {"result": "模拟执行结果", "correct": correct}


# 工具注册表：工具名 → (调用函数, 工具描述)
TOOLS = {
    "calculator": (tool_calculator, "数学计算器，用于数值运算"),
    "search": (tool_search, "知识搜索引擎，用于事实查询"),
    "code_executor": (tool_code_executor, "代码执行器，用于运行代码片段"),
}


# ==========================================
# 第二部分：模拟多轮对话场景
# ==========================================
# 预定义若干多轮任务，每个任务包含 3~5 轮工具调用

SCENARIOS = [
    {
        "task": "计算 123 + 456 的结果，然后搜索 Python 相关知识，最后写代码打印结果",
        "turns": [
            {"tool": "calculator",  "query": "123 + 456",   "description": "第一步：调用计算器进行加法运算"},
            {"tool": "search",      "query": "Python 编程",  "description": "第二步：搜索 Python 相关知识"},
            {"tool": "code_executor","query": "print(579)",  "description": "第三步：执行代码打印结果"},
        ],
    },
    {
        "task": "搜索强化学习资料，计算 25*4，然后执行一个简单程序",
        "turns": [
            {"tool": "search",       "query": "强化学习入门",       "description": "第一步：搜索强化学习资料"},
            {"tool": "calculator",   "query": "25 * 4",            "description": "第二步：计算乘法"},
            {"tool": "code_executor","query": "def hello(): pass",  "description": "第三步：执行简单程序"},
        ],
    },
    {
        "task": "计算平方根，搜索算法资料，执行代码，再搜索深度学习",
        "turns": [
            {"tool": "calculator",    "query": "sqrt(144)",         "description": "第一步：计算平方根"},
            {"tool": "search",        "query": "排序算法比较",       "description": "第二步：搜索算法资料"},
            {"tool": "code_executor", "query": "import numpy",       "description": "第三步：导入模块"},
            {"tool": "search",        "query": "深度学习框架",       "description": "第四步：搜索深度学习资料"},
        ],
    },
    {
        "task": "搜索数学公式，执行计算脚本，验证结果",
        "turns": [
            {"tool": "search",        "query": "欧拉公式推导",       "description": "第一步：搜索数学公式"},
            {"tool": "code_executor", "query": "import math",        "description": "第二步：执行计算脚本"},
            {"tool": "calculator",    "query": "圆周率计算",         "description": "第三步：数值计算"},
        ],
    },
    {
        "task": "搜索 RL 策略梯度，执行训练代码，计算奖励，搜索 PPO",
        "turns": [
            {"tool": "search",        "query": "RL 策略梯度",        "description": "第一步：搜索策略梯度"},
            {"tool": "code_executor", "query": "def train(): pass",   "description": "第二步：执行训练代码"},
            {"tool": "calculator",    "query": "计算累积奖励",        "description": "第三步：计算奖励"},
            {"tool": "search",        "query": "PPO 算法详解",       "description": "第四步：搜索 PPO"},
            {"tool": "code_executor", "query": "print('done')",       "description": "第五步：执行收尾代码"},
        ],
    },
]


# ==========================================
# 第三部分：ORM 和 PRM 奖励计算
# ==========================================

def compute_orm_rewards(turns):
    """
    ORM（Outcome Reward Model）：只有最终结果获得奖励

    如果任务最终成功，所有步骤共享奖励 1.0；
    如果任务最终失败，所有步骤获得 0.0。

    这就像考试只看最终答案——过程不给分。
    """
    n = len(turns)
    rewards = [0.0] * n  # 初始化：所有步骤奖励为 0

    # 模拟最终结果：所有步骤都正确则任务成功
    all_correct = all(turn.get("correct", False) for turn in turns)
    final_reward = 1.0 if all_correct else 0.0

    # 只有最后一步获得奖励
    rewards[-1] = final_reward

    return rewards


def compute_prm_rewards(turns):
    """
    PRM（Process Reward Model）：每一步都获得部分奖励

    每一步独立评估正确性，获得 0.0~1.0 的奖励。
    这就像考试每道题单独打分——过程也给分。

    奖励策略：
    - 工具调用正确：获得基础奖励 + 工具选择正确加成
    - 工具调用错误：获得较低奖励
    """
    rewards = []
    for turn in turns:
        correct = turn.get("correct", False)

        if correct:
            # 正确的步骤：获得较高奖励
            # 额外考虑工具选择是否合理
            base_reward = 0.7 + np.random.uniform(0, 0.3)  # 0.7 ~ 1.0
        else:
            # 错误的步骤：仍然有少量奖励（鼓励探索）
            base_reward = np.random.uniform(0.0, 0.3)  # 0.0 ~ 0.3

        rewards.append(round(base_reward, 3))

    return rewards


def compute_discounted_returns(rewards, gamma=0.99):
    """
    计算折扣回报：G_t = r_t + γ * G_{t+1}

    从后往前递推计算每一步的折扣累计回报。
    γ（折扣因子）控制信用的传播范围：
      - γ 接近 1.0：信用传播得更远（远视）
      - γ 接近 0.0：信用只影响当前步（短视）

    参数：
        rewards: 每一步的即时奖励列表
        gamma:   折扣因子
    返回：
        returns: 每一步的折扣累计回报列表
    """
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


# ==========================================
# 第四部分：运行模拟实验
# ==========================================
print("=" * 70)
print("  第12章：多轮对话 RL —— ORM vs PRM 信用分配对比")
print("=" * 70)

np.random.seed(42)  # 固定随机种子，确保结果可复现

# 存储所有场景的结果
all_orm_rewards = []   # ORM 每步奖励
all_prm_rewards = []   # PRM 每步奖励
all_orm_returns = {}   # ORM 折扣回报（不同 gamma）
all_prm_returns = {}   # PRM 折扣回报（不同 gamma）
gamma_values = [0.5, 0.9, 0.99]  # 测试不同折扣因子

for gamma in gamma_values:
    all_orm_returns[gamma] = []
    all_prm_returns[gamma] = []

# 遍历每个场景进行模拟
for idx, scenario in enumerate(SCENARIOS):
    print(f"\n{'─' * 70}")
    print(f"  场景 {idx + 1}：{scenario['task']}")
    print(f"  总共 {len(scenario['turns'])} 轮工具调用")
    print(f"{'─' * 70}")

    # 模拟每一步的工具调用结果
    turns = scenario["turns"]
    for t, turn in enumerate(turns):
        tool_name = turn["tool"]
        query = turn["query"]

        # 调用对应工具
        tool_func, _ = TOOLS[tool_name]
        result = tool_func(query)
        turn["correct"] = result["correct"]

        status = "正确" if result["correct"] else "错误"
        print(f"  第 {t+1} 轮：{turn['description']}")
        print(f"         调用 {tool_name}({query}) → {status}")

    # ---- ORM 奖励计算 ----
    orm_rewards = compute_orm_rewards(turns)
    all_orm_rewards.append(orm_rewards)

    print(f"\n  [ORM 奖励] 只有最终结果有信号:")
    for t, r in enumerate(orm_rewards):
        bar = "█" * int(r * 20)
        print(f"    第 {t+1} 轮奖励: {r:.1f}  {bar}")

    # ---- PRM 奖励计算 ----
    prm_rewards = compute_prm_rewards(turns)
    all_prm_rewards.append(prm_rewards)

    print(f"\n  [PRM 奖励] 每一步都有学习信号:")
    for t, r in enumerate(prm_rewards):
        bar = "█" * int(r * 20)
        print(f"    第 {t+1} 轮奖励: {r:.3f}  {bar}")

    # ---- 折扣回报对比（多个 gamma 值）----
    for gamma in gamma_values:
        orm_returns = compute_discounted_returns(orm_rewards, gamma=gamma)
        prm_returns = compute_discounted_returns(prm_rewards, gamma=gamma)
        all_orm_returns[gamma].append(orm_returns)
        all_prm_returns[gamma].append(prm_returns)

    # 详细展示 gamma=0.99 的折扣回报计算过程
    gamma_demo = 0.99
    orm_ret_demo = compute_discounted_returns(orm_rewards, gamma=gamma_demo)
    prm_ret_demo = compute_discounted_returns(prm_rewards, gamma=gamma_demo)

    print(f"\n  折扣回报计算过程（γ = {gamma_demo}）:")
    print(f"    {'轮次':<6} {'即时奖励':<12} {'折扣回报 G_t':<16} {'计算过程'}")
    print(f"    {'─' * 60}")

    # ORM 折扣回报逐步展示
    print(f"    [ORM 模式]")
    G = 0.0
    for t in reversed(range(len(orm_rewards))):
        old_G = G
        G = orm_rewards[t] + gamma_demo * old_G
        formula = f"G_{t} = {orm_rewards[t]:.1f} + {gamma_demo} * {old_G:.4f} = {G:.4f}"
        print(f"    第 {t+1} 轮  r={orm_rewards[t]:<8.1f}  G={G:<12.4f}  {formula}")

    # PRM 折扣回报逐步展示
    print(f"    [PRM 模式]")
    G = 0.0
    for t in reversed(range(len(prm_rewards))):
        old_G = G
        G = prm_rewards[t] + gamma_demo * old_G
        formula = f"G_{t} = {prm_rewards[t]:.3f} + {gamma_demo} * {old_G:.4f} = {G:.4f}"
        print(f"    第 {t+1} 轮  r={prm_rewards[t]:<8.3f}  G={G:<12.4f}  {formula}")


# ==========================================
# 第五部分：ORM vs PRM 综合对比分析
# ==========================================
print("\n" + "=" * 70)
print("  ORM vs PRM 综合对比分析")
print("=" * 70)

print("\n  【奖励信号密度对比】")
for idx in range(len(SCENARIOS)):
    n_turns = len(SCENARIOS[idx]["turns"])
    orm_nonzero = sum(1 for r in all_orm_rewards[idx] if r > 0)
    prm_nonzero = sum(1 for r in all_prm_rewards[idx] if r > 0)
    print(f"    场景 {idx+1}（{n_turns} 轮）:"
          f" ORM 有信号步数 = {orm_nonzero}/{n_turns},"
          f" PRM 有信号步数 = {prm_nonzero}/{n_turns}")

print(f"\n  关键结论:")
print(f"    - ORM 信号稀疏：只有最后一步有奖励，中间步骤缺乏信号")
print(f"    - PRM 信号密集：每一步都有反馈，学习效率更高")
print(f"    - 对于多轮 Agent，PRM 能显著加速策略学习")

print("\n  【折扣因子 γ 对信用传播的影响】")
for gamma in gamma_values:
    print(f"\n    γ = {gamma}:")
    for idx in range(len(SCENARIOS)):
        orm_ret = all_orm_returns[gamma][idx]
        prm_ret = all_prm_returns[gamma][idx]
        n = len(orm_ret)
        print(f"      场景 {idx+1}（{n} 轮）:")
        print(f"        ORM 折扣回报: {[f'{v:.4f}' for v in orm_ret]}")
        print(f"        PRM 折扣回报: {[f'{v:.4f}' for v in prm_ret]}")

print(f"\n  γ 的影响总结:")
print(f"    - γ=0.5: 信用衰减很快，只有最后几步能感受到最终奖励")
print(f"    - γ=0.9: 信用传播适中，平衡近期和远期信号")
print(f"    - γ=0.99: 信用传播很远，早期步骤也能获得可观的回报信号")


# ==========================================
# 第六部分：可视化图表
# ==========================================
print("\n正在生成可视化图表...")

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle("多轮对话 RL —— ORM vs PRM 信用分配对比", fontsize=18, fontweight="bold")

# ---- 子图1：Turn 级奖励热力图 ----
ax1 = axes[0, 0]

# 构造热力图数据矩阵
max_turns = max(len(r) for r in all_orm_rewards)
n_scenarios = len(SCENARIOS)

heatmap_orm = np.zeros((n_scenarios, max_turns))
heatmap_prm = np.zeros((n_scenarios, max_turns))

for i in range(n_scenarios):
    for j in range(len(all_orm_rewards[i])):
        heatmap_orm[i, j] = all_orm_rewards[i][j]
        heatmap_prm[i, j] = all_prm_rewards[i][j]

# 绘制 PRM 热力图（更有教育意义）
im = ax1.imshow(heatmap_prm, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
ax1.set_xticks(range(max_turns))
ax1.set_xticklabels([f"第 {i+1} 轮" for i in range(max_turns)])
ax1.set_yticks(range(n_scenarios))
ax1.set_yticklabels([f"场景 {i+1}" for i in range(n_scenarios)])
ax1.set_title("PRM 每步奖励热力图", fontsize=14, fontweight="bold")
ax1.set_xlabel("对话轮次", fontsize=12)
ax1.set_ylabel("场景", fontsize=12)

# 在热力图上标注数值
for i in range(n_scenarios):
    for j in range(len(all_prm_rewards[i])):
        ax1.text(j, i, f"{heatmap_prm[i, j]:.2f}",
                 ha="center", va="center", fontsize=10, fontweight="bold")

fig.colorbar(im, ax=ax1, label="奖励值")

# ---- 子图2：ORM vs PRM 折扣回报对比（gamma=0.99）----
ax2 = axes[0, 1]

gamma_plot = 0.99
colors_orm = plt.cm.Blues(np.linspace(0.4, 0.9, n_scenarios))
colors_prm = plt.cm.Reds(np.linspace(0.4, 0.9, n_scenarios))

for i in range(n_scenarios):
    n = len(all_orm_returns[gamma_plot][i])
    x = np.arange(n)

    # ORM 用虚线，PRM 用实线
    ax2.plot(x, all_orm_returns[gamma_plot][i],
             marker="o", linestyle="--", linewidth=2, markersize=6,
             color=colors_orm[i],
             label=f"场景{i+1} ORM" if i == 0 else None)
    ax2.plot(x, all_prm_returns[gamma_plot][i],
             marker="s", linestyle="-", linewidth=2, markersize=6,
             color=colors_prm[i],
             label=f"场景{i+1} PRM" if i == 0 else None)

# 只画两根示意线（避免图例太拥挤）
ax2.plot([], [], marker="o", linestyle="--", color="steelblue", linewidth=2, label="ORM 折扣回报")
ax2.plot([], [], marker="s", linestyle="-", color="crimson", linewidth=2, label="PRM 折扣回报")

ax2.set_title(f"ORM vs PRM 折扣回报（γ={gamma_plot}）", fontsize=14, fontweight="bold")
ax2.set_xlabel("对话轮次", fontsize=12)
ax2.set_ylabel("折扣回报 G_t", fontsize=12)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# 添加注释
ax2.annotate("ORM: 只有最后一步有信号\n→ 中间步骤梯度 ≈ 0",
             xy=(0.02, 0.95), xycoords="axes fraction",
             fontsize=10, color="steelblue", va="top",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.3))
ax2.annotate("PRM: 每步都有信号\n→ 梯度信号密集",
             xy=(0.02, 0.75), xycoords="axes fraction",
             fontsize=10, color="crimson", va="top",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.3))

# ---- 子图3：不同 gamma 对信用传播的影响（场景5，PRM）----
ax3 = axes[1, 0]

# 选择最长的场景（场景5有5轮）
demo_idx = 4  # 场景5（索引4）
colors_gamma = ["#E91E63", "#FF9800", "#4CAF50"]

for gi, gamma in enumerate(gamma_values):
    ret = all_prm_returns[gamma][demo_idx]
    x = np.arange(len(ret))
    ax3.plot(x, ret, marker="o", linewidth=2.5, markersize=8,
             color=colors_gamma[gi], label=f"γ = {gamma}")

ax3.set_title(f"折扣因子 γ 对信用传播的影响（场景5，PRM）", fontsize=14, fontweight="bold")
ax3.set_xlabel("对话轮次", fontsize=12)
ax3.set_ylabel("折扣回报 G_t", fontsize=12)
ax3.legend(fontsize=12)
ax3.grid(True, alpha=0.3)
ax3.set_xticks(range(len(all_prm_returns[0.99][demo_idx])))
ax3.set_xticklabels([f"第{i+1}轮" for i in range(len(all_prm_returns[0.99][demo_idx]))])

# ---- 子图4：ORM vs PRM 奖励信号密度对比柱状图 ----
ax4 = axes[1, 1]

x_pos = np.arange(n_scenarios)
bar_width = 0.35

# 计算每个场景的非零奖励步数占比
orm_density = []
prm_density = []
for i in range(n_scenarios):
    n = len(all_orm_rewards[i])
    orm_density.append(sum(1 for r in all_orm_rewards[i] if r > 0) / n * 100)
    prm_density.append(sum(1 for r in all_prm_rewards[i] if r > 0) / n * 100)

bars1 = ax4.bar(x_pos - bar_width/2, orm_density, bar_width,
                label='ORM（结果奖励）', color='steelblue', alpha=0.8)
bars2 = ax4.bar(x_pos + bar_width/2, prm_density, bar_width,
                label='PRM（过程奖励）', color='crimson', alpha=0.8)

ax4.set_title("奖励信号密度对比（非零奖励步数占比）", fontsize=14, fontweight="bold")
ax4.set_xlabel("场景", fontsize=12)
ax4.set_ylabel("有信号的步骤占比 (%)", fontsize=12)
ax4.set_xticks(x_pos)
ax4.set_xticklabels([f"场景{i+1}" for i in range(n_scenarios)])
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3, axis='y')
ax4.set_ylim(0, 110)

# 在柱状图上标注百分比
for bar in bars1:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.0f}%', ha='center', va='bottom', fontsize=9, fontweight="bold")
for bar in bars2:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.0f}%', ha='center', va='bottom', fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig("output/multi_turn_orm_vs_prm.png", dpi=150, bbox_inches="tight")
print("图表已保存至: output/multi_turn_orm_vs_prm.png")
plt.show()


# ==========================================
# 第七部分：总结
# ==========================================
print("\n" + "=" * 70)
print("  关键结论")
print("=" * 70)
print("""
  1. 信用分配是多轮 Agent RL 的核心挑战
     - Agent 需要经历多轮交互才能完成任务
     - 如何将最终成功/失败归因到每一步？

  2. ORM 的优缺点：
     ✓ 实现简单，只需标注最终结果
     ✗ 信号稀疏，中间步骤"盲飞"
     ✗ 信用传播依赖折扣因子，可能衰减过快

  3. PRM 的优缺点：
     ✓ 信号密集，每步都有学习信号
     ✓ 能区分"好的中间步骤"和"坏的中间步骤"
     ✗ 标注成本高，需要为每步打分
     ✗ 奖励模型可能引入噪声

  4. 折扣因子 γ 的作用：
     - γ 越大，信用传播越远（远期奖励也被考虑）
     - γ 越小，信用衰减越快（只关注近期步骤）
     - 多轮对话中建议 γ ≥ 0.9

  5. 实际应用建议：
     - 简单任务：ORM 足够（如单轮问答）
     - 复杂多步推理：PRM 更优（如数学证明、代码生成）
     - 混合方案：PRM 过程奖励 + ORM 结果验证
""")
print("=" * 70)
