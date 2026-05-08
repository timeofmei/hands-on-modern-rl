# 3.7 数据来源：On-policy、Off-policy、Online 与 Offline

## 本节导读

**核心内容**

- 行为策略 $\mu$ 与目标策略 $\pi_\theta$：区分“谁产生数据”和“谁正在被学习”。
- On-policy / Off-policy：判断训练数据和目标策略是否来自同一个策略分布。
- Online / Offline：判断训练过程中数据集还能不能继续增长。

前面几节讨论的是价值和策略怎样更新。本节换一个维度：这些更新用的数据从哪里来。很多算法的差别，不只在损失函数，也在数据是当前策略采的、旧策略留下的，还是训练前已经固定好的。

本节要回答的问题是：**训练样本和当前策略之间是什么关系？训练时还能不能继续产生新样本？**

::: info 核心概念
On-policy / Off-policy 是策略分布的关系，Online / Offline 是数据集是否继续增长的关系。前者问“样本来自谁”，后者问“训练时还能不能继续采样”。两条轴彼此独立。
:::

先从几个常见训练场景看这个差别。

先看一个熟悉的 CartPole 场景。DQN 智能体一边玩游戏，一边把转移样本

$$
(s_t,a_t,r_{t+1},s_{t+1})
$$

存进 replay buffer。过了一会儿，网络参数已经更新了很多次，但 buffer 里还躺着旧策略采出来的经验。训练时我们会从这些旧经验里随机抽 batch，继续更新当前的 Q 网络。

这时就有一个自然疑问：用旧策略产生的数据，更新现在的策略或价值函数，为什么可以？

再换一个场景。PPO 训练语言模型时，通常先让当前模型对一批 prompt 生成回答，得到一批 rollout，然后用奖励模型或规则给分。接下来 PPO 会在这批 rollout 上做几轮更新。可是更新一开始，当前模型就已经不再等于当初采样的模型了。那这些 rollout 还能不能继续用？能用多久？

最后看 DPO。DPO 常常使用一份已经固定的人类偏好数据，例如同一个 prompt 下回答 $y_w$ 比回答 $y_l$ 更受偏好。训练过程中，模型不一定再去环境里生成新回答，也不一定每轮重新找人标注偏好。它只是反复读取这份固定数据来移动策略分布。

这三个例子的差别，不在于它们的损失函数写得像不像，而在于两件更基础的事：

1. 这批数据是谁产生的？
2. 训练时还能不能继续产生新数据？

这就是本节要建立的分类方式。On-policy / Off-policy 讨论第一件事：行为策略和目标策略是什么关系。Online / Offline 讨论第二件事：训练数据集是否还能继续增长。这两条轴彼此独立，不能混在一起。

## 两个策略角色

先把“谁产生数据”和“谁正在被学习”分开。

产生数据的策略叫**行为策略**（behavior policy），记作

$$
\mu(a\mid s).
$$

这个符号回答的问题是：训练集中为什么会出现状态 $s$ 下的动作 $a$？在经典控制任务里，$\mu$ 可能是带 $\epsilon$-greedy 探索的策略，也可能是 replay buffer 中某个旧版本的策略。在机器人或自动驾驶日志里，$\mu$ 可能是人类驾驶员、专家控制器或许多历史策略的混合。在大模型训练里，$\mu$ 可能是当前模型、旧 checkpoint、teacher model，或者人类写好的回答和偏好记录。

我们真正想评估、改进或最终部署的策略叫**目标策略**（target policy），记作

$$
\pi_\theta(a\mid s).
$$

它回答的不是“数据当初怎么来的”，而是“我们希望学好的策略是什么”。参数 $\theta$ 变化时，目标策略也在变化。

这两个角色有时重合，有时不重合。REINFORCE 用当前策略采样一整条 episode，再用这条 episode 的回报更新同一个策略，行为策略和目标策略基本一致。Q-Learning 则不同：智能体实际探索时可以用 $\epsilon$-greedy，但更新目标里使用的是下一状态的最大动作价值，学习方向指向贪心策略。DQN 又进一步把许多旧策略产生的样本放进 replay buffer，拿来更新当前网络。

一旦分清 $\mu$ 和 $\pi_\theta$，On-policy 和 Off-policy 就不再只是算法标签，而是在描述数据分布和学习目标之间的关系。

## 第一条轴：数据由谁产生

On-policy / Off-policy 判断的是一次更新中，训练样本和目标策略是否来自同一个策略分布。

更具体地说，我们不是在问“算法名字是什么”，而是在问：这次拿来计算梯度或 TD 目标的数据，是不是当前目标策略自己采出来的，或者至少来自一个离它很近的旧版本？

## On-policy：用自己的近邻数据更新自己

如果训练数据来自当前正在学习的策略，或者来自一个足够接近当前策略的旧快照，就称为 **on-policy**。

可以粗略写成

$$
\mu(a\mid s)\approx \pi_{\theta_{\text{collect}}}(a\mid s).
$$

这里的 $\pi_{\theta_{\text{collect}}}$ 表示采样那一刻的策略。公式里用的是“约等于”，因为实际训练很少做到采样和更新完全同时发生。通常是先用旧参数采一批数据，再用这批数据更新出新参数。只要新策略没有离采样策略太远，这批样本仍然可以按 on-policy 的方式理解。

PPO 正是这个例子。它先用旧策略 $\pi_{\theta_{\text{old}}}$ 采样 rollout，再更新当前策略 $\pi_\theta$。为了判断新旧策略是否偏离太多，PPO 使用比率

$$
r_t(\theta)=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}.
$$

这个比率的含义很直接：同一个状态动作 $(s_t,a_t)$，新策略认为它有多可能，旧采样策略当初认为它有多可能。如果 $r_t(\theta)$ 接近 1，说明新旧策略看待这一步差不多；如果它离 1 很远，说明当前策略已经明显不同于采样策略。

所以 PPO 可以在同一批 rollout 上训练多个 epoch，但不能无限复用它们。真正的问题不是“样本是否只用了一次”，而是“当前策略是否已经离产生这些样本的策略太远”。PPO 的 clipping 或 KL 约束，就是为了把这种偏离控制在可接受范围内。[^ppo2017]

Sarsa 也是 on-policy 的经典例子。它的更新目标是

$$
Q(s,a)\leftarrow Q(s,a)+\alpha
\left[
r+\gamma Q(s',a')-Q(s,a)
\right].
$$

这里的 $a'$ 是智能体在下一状态 $s'$ 中**实际会执行的动作**。如果行为策略带有探索，$a'$ 也包含这种探索。因此 Sarsa 学到的是“继续按照当前行为策略行动时”的价值，而不是直接假设未来每一步都贪心。

REINFORCE 也属于这个思路。它用当前策略生成完整轨迹，再根据轨迹回报调整同一个策略。采样分布和优化对象一致，因此梯度的解释最清楚，但代价是样本效率通常较低。

## Off-policy：用别人的数据学习目标策略

如果行为策略 $\mu$ 和目标策略 $\pi_\theta$ 可以不同，就称为 **off-policy**。

可以写成

$$
\mu(a\mid s)\neq \pi_\theta(a\mid s).
$$

这句话的含义是：数据可以来自探索策略、旧策略、人类日志、专家策略或 teacher model，但学习目标仍然可以指向另一个策略。

Q-Learning 是最重要的例子。它的更新为

$$
Q(s,a)\leftarrow Q(s,a)+\alpha
\left[
r+\gamma\max_{a'}Q(s',a')-Q(s,a)
\right].
$$

这个式子和 Sarsa 只差一个地方：Sarsa 使用实际采到的 $a'$，Q-Learning 使用 $\max_{a'}Q(s',a')$。这一步的含义是：虽然当前数据可能来自带探索的行为策略，但学习目标假设未来会选择当前估计下最好的动作。换句话说，采样可以探索，学习却朝贪心最优策略前进。Watkins 和 Dayan 证明，在表格设定、访问充分和学习率合适等条件下，Q-Learning 可以收敛到最优动作价值。[^watkins1992]

DQN 把这个想法搬到深度网络里。CartPole 或 Atari 里的智能体不断把经验放进 replay buffer，而训练时反复抽取旧经验更新当前 Q 网络。buffer 里的样本来自许多历史版本的策略，所以 DQN 通常是 off-policy。经验回放提高了样本利用率，也打散了相邻样本之间的强相关性，这是 DQN 能稳定训练的重要工程设计。[^mnih2015]

不过，Off-policy 不是“任何旧数据都能随便用”。它至少需要一个覆盖条件：行为策略采出来的数据，要覆盖目标策略想学习的状态和动作。

如果存在某个状态动作对 $(s,a)$，目标策略想选择它：

$$
\pi_\theta(a\mid s)>0,
$$

但行为策略从来没有采过它：

$$
\mu(a\mid s)=0,
$$

那么算法就没有可靠证据判断这个动作的后果。用重要性采样来修正分布差异时，还会出现比率

$$
\rho_t=
\frac{\pi_\theta(a_t\mid s_t)}
{\mu(a_t\mid s_t)}.
$$

分母很小会让估计方差变得很大；分母为 0 时，问题甚至没有定义。换个角度说，如果 replay buffer 或历史日志从未展示过某些行为，函数近似器只能在这些区域外推，而外推出来的高价值很可能是假的。Sutton 和 Barto 在讨论 off-policy prediction 和 control 时反复强调了这一点。[^sutton-barto]

这就是 Off-policy 的双刃剑：它能复用旧数据，所以样本效率高；但它必须处理行为策略和目标策略之间的分布差异。

## 第二条轴：训练时还能不能采样

现在看另一条轴。Online / Offline 不关心行为策略和目标策略是不是同一个，它只问：训练过程中，数据集还能不能继续增长？

这条轴和 On-policy / Off-policy 是独立的。DQN 是 off-policy，但它通常仍然继续和环境交互，所以不是 offline。DPO 常常在固定偏好数据上训练，训练阶段不再重新采样和标注，所以更接近 offline 偏好优化。二者讨论的根本不是同一个问题。

## Online RL：训练中继续产生数据

如果训练过程中仍然可以和环境交互，并持续把新样本加入数据集，就称为 **online RL**。

可以写成

$$
\mathcal{D}_{k+1}
=
\mathcal{D}_k\cup\{\tau_k\},
$$

其中 $\tau_k$ 是第 $k$ 轮新采到的轨迹或样本。这个公式只是说：随着训练进行，数据集还在变大。

在 CartPole 里，智能体继续玩新的 episode，新的转移样本继续进入 replay buffer，这就是 online。DQN 即使大量复用旧数据，也仍然是 online，因为训练并没有停止采样。

在 PPO 或 GRPO 训练大模型时，环境交互不一定是物理世界或游戏模拟器。给模型一个 prompt，让它生成回答，再由奖励模型、规则评测、人类反馈或另一个模型给分，也是一种交互。只要每轮仍然用当前模型生成新的回答或轨迹，数据就在继续增长。因此 PPO/GRPO 通常被看作 online + on-policy。

这里容易误解的一点是：prompt 集合可以是固定的。即使 prompt 池不变，只要回答是每轮由当前策略重新生成，动作数据仍然是新的。从策略学习的角度看，模型正在继续采样。

## Offline RL：只能使用固定数据

如果训练开始前数据集已经固定，训练过程中不能再与环境交互，也不能让当前策略继续产生新样本，就称为 **offline RL**。

可以写成

$$
\mathcal{D}=\mathcal{D}_{\text{fixed}}.
$$

这一步的含义是：算法只能从已有数据中学习，不能通过新的探索来弥补数据盲区。Levine 等人的 offline RL tutorial 就把这个设定描述为使用预先收集的数据、没有额外 online data collection 的强化学习。[^levine2020]

这种设定在很多高风险任务里很自然。自动驾驶不能让一个还没训练好的策略随便上路试错；医疗推荐不能为了探索而给病人尝试不确定方案；机器人也可能因为真实试错成本太高，只能先使用历史日志或专家演示。

大模型对齐里也有类似场景。DPO 通常使用固定偏好数据：给定 prompt、被偏好的回答 $y_w$ 和不被偏好的回答 $y_l$，训练时根据这些偏好对调整策略。DPO 的形式看起来像二分类，但它不是单纯预测哪个回答更好，而是在参考模型约束下移动当前策略分布。DPO 论文的关键贡献之一，就是把 RLHF 中的策略优化目标改写成可以在偏好数据上直接训练的损失。[^dpo2023]

因此，DPO 是否 offline，取决于训练阶段数据是否固定，而不是取决于 loss 是否像监督学习。如果系统在训练过程中不断让当前模型生成新回答，再重新收集偏好标签，那它就不再是固定数据意义上的 offline 设定。

Offline RL 的困难也来自这个固定数据约束。假设自动驾驶日志里大多是保守驾驶：慢速、少变道、很少接近复杂路口。如果目标策略想学得更激进，数据里却几乎没有这些行为的后果，那么价值函数可能在数据覆盖不到的区域给出虚假的高分。训练阶段不能让策略出去试一试，所以错误很难被及时纠正。

CQL 正是为这个问题设计的。它认为标准 off-policy RL 直接用于离线数据时，容易对数据分布外的动作产生 Q 值过估计，于是引入保守项，让模型更不愿意给未充分观察过的动作高价值。[^cql2020]

## 四个象限

把两条轴放在一起，可以得到一张更清楚的地图。

| 数据形态                 | 含义                                                       | 典型算法或场景                                  | 主要风险                           |
| ------------------------ | ---------------------------------------------------------- | ----------------------------------------------- | ---------------------------------- |
| **Online + On-policy**   | 训练时继续采样，并主要用当前策略或近邻旧快照的数据更新自己 | REINFORCE、Sarsa、PPO、GRPO                     | 样本效率低，旧 rollout 很快过期    |
| **Online + Off-policy**  | 训练时继续采样，但更新时允许复用旧策略或探索策略的数据     | Q-Learning、DQN、SAC、TD3                       | 要处理行为策略和目标策略的分布差异 |
| **Offline + Off-policy** | 不再采样，只能从固定历史数据中学习目标策略                 | CQL、IQL、离线 Q 学习、固定偏好数据上的 DPO/IPO | 覆盖不足时容易外推和高估           |
| **Offline + On-policy**  | 数据固定，同时又希望数据代表当前目标策略                   | 固定策略评估、很小步长的模仿式更新              | 策略一更新，数据就不再代表新策略   |

这张表里最值得停一下的是左下和右上。

DQN 是 **Online + Off-policy**。它使用 replay buffer 中的旧经验，所以是 off-policy；但它训练时仍然继续玩游戏、继续收集新经验，所以是 online。Off-policy 不等于 offline。

DPO 在常见设定下更接近 **Offline + Off-policy** 的偏好优化。偏好数据来自人类、旧模型或某些历史采样过程，而当前训练的语言模型是目标策略；如果训练阶段不再生成新回答并重新标注偏好，那么数据集就是固定的。Offline 不等于“没有 RL 味道的监督学习”，因为策略分布仍然在移动，分布覆盖和外推风险仍然存在。

右下角的 **Offline + On-policy** 不太常见。严格的 on-policy 控制通常需要当前策略自己采样；一旦数据固定，而策略还在更新，数据很快就会变成旧策略的数据。它更像边界情形，例如只评估一套固定策略，或者在很小范围内做接近模仿学习的更新。

## 算法名字只是结果

现在可以重新理解几个常见算法。

REINFORCE 用当前策略采样完整 episode，再用回报更新当前策略。它是典型 on-policy 方法；如果训练中持续采样新 episode，就是 online + on-policy。

Sarsa 使用实际执行的下一个动作 $a'$ 来构造 TD 目标，因此学习的是当前行为策略本身的价值。带探索的 Sarsa 仍然是在评估和改进这个带探索策略，所以通常归为 on-policy。

Q-Learning 的采样策略可以带探索，但更新目标使用 $\max_{a'}Q(s',a')$，学习方向指向贪心目标策略。因此它是 off-policy。若训练中继续采样，它就是 online + off-policy。

DQN 使用 Q-Learning 目标和 replay buffer。CartPole replay buffer 中的样本可能来自几千步以前的旧网络，但当前网络仍然拿它们学习，所以 DQN 是 off-policy；同时它通常继续和环境交互，所以仍然是 online。

PPO 和 GRPO 每轮通常先用当前模型或近邻旧快照生成 rollout，再限制新策略不要离采样策略太远。因此它们常按 online + on-policy 理解。它们可以短期复用 rollout，但不能长期把很旧的数据当作当前策略数据。

CQL 面向固定数据集，重点处理 offline RL 中的分布外高估问题。它通常属于 offline + off-policy，因为数据来自某个行为策略，而学习目标是从这份数据中得到更好的目标策略。

DPO 常用固定偏好数据训练语言模型。它的损失形式类似分类，但目标是根据偏好对移动策略分布，并用参考模型约束移动幅度。若训练阶段不再重新采样和标注，它更接近 offline 偏好优化；如果系统持续生成新回答并收集新偏好，那数据轴就会变成 online。

## 常见误解

**On-policy 不等于样本只能用一次。** PPO 的 rollout 可以训练多个 epoch。关键是新策略不能离采样策略太远，否则这些样本就不再像当前策略数据。

**Off-policy 不等于旧数据随便用。** 旧数据必须覆盖目标策略想学习的区域。覆盖不足时，重要性采样方差会变大，Q 函数也可能在数据看不到的动作上胡乱外推。

**Off-policy 不等于 Offline。** DQN 是最好的反例。它能复用旧经验，所以是 off-policy；它训练时继续采新经验，所以是 online。

**Offline RL 不一定更简单。** 它避免了在线试错的风险，却失去了用新探索纠正错误的机会。固定数据越窄，策略可靠改进的范围越窄。

**DPO 不只是普通分类。** 它可以用分类式损失实现，但训练效果仍然取决于偏好数据是谁产生的、覆盖了哪些回答类型，以及当前策略相对参考模型移动了多远。

## 小结

本节讨论的是 RL 算法的数据来源，而不是新的更新公式。

1. 行为策略 $\mu(a\mid s)$ 负责产生数据，目标策略 $\pi_\theta(a\mid s)$ 是正在被评估或改进的策略。
2. On-policy / Off-policy 判断的是数据由谁产生：数据来自当前策略或近邻旧快照，就是 on-policy；允许来自其他策略，就是 off-policy。
3. Online / Offline 判断的是训练时还能不能继续采样：数据集持续增长，就是 online；只能使用固定数据集，就是 offline。
4. 这两条轴彼此独立。DQN 通常是 online + off-policy；PPO/GRPO 通常是 online + on-policy；CQL 和固定偏好数据上的 DPO 更接近 offline + off-policy。
5. Off-policy 的关键风险是覆盖不足，Offline RL 的关键风险是固定数据导致的分布外外推和价值高估。

到这里，我们已经知道算法可以沿着 $Q(s,a)$ 或 $J(\theta)$ 两条路线优化，也知道训练数据可以来自当前策略、旧策略、日志数据或固定数据集。下一节继续追问一个更根本的问题：所有这些算法都在最大化奖励，但**奖励本身是谁写的？写歪了会怎样？**

下一节：[奖励函数设计](./reward-design)

## 参考文献

[^sutton-barto]: Sutton, R. S., & Barto, A. G. (2018). _Reinforcement Learning: An Introduction_ (2nd ed.). MIT Press. 参见第 5.5、5.7、6.4、6.5 章关于 off-policy prediction、off-policy control、Sarsa 和 Q-learning 的讨论。MIT Press 页面：<https://mitpress.mit.edu/9780262039246/reinforcement-learning/>

[^watkins1992]: Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. _Machine Learning_, 8, 279-292. <https://www.gatsby.ucl.ac.uk/~dayan/papers/wd92.html>

[^mnih2015]: Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). Human-level control through deep reinforcement learning. _Nature_, 518, 529-533. <https://doi.org/10.1038/nature14236>

[^ppo2017]: Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. _arXiv:1707.06347_. <https://arxiv.org/abs/1707.06347>

[^levine2020]: Levine, S., Kumar, A., Tucker, G., & Fu, J. (2020). Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems. _arXiv:2005.01643_. <https://arxiv.org/abs/2005.01643>

[^cql2020]: Kumar, A., Zhou, A., Tucker, G., & Levine, S. (2020). Conservative Q-Learning for Offline Reinforcement Learning. _NeurIPS 2020_. <https://papers.nips.cc/paper_files/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html>

[^dpo2023]: Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. _arXiv:2305.18290_. <https://arxiv.org/abs/2305.18290>
