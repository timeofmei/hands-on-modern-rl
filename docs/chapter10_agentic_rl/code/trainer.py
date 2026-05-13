# trainer.py

from rollout_worker import RolloutWorker


class GRPOAgentTrainer:
    """
    编排 Agentic RL 训练循环。
    rollout -> reward -> train -> repeat
    """

    def __init__(self, policy, env, reward_fn, group_size=4, max_turns=5):
        self.policy = policy
        self.env = env
        self.reward_fn = reward_fn
        self.group_size = group_size
        self.worker = RolloutWorker(policy, env, max_turns=max_turns)
        self.history = []

    def fit(self, prompts: list, n_steps: int = 50):
        """
        主训练循环。
        prompts: 训练用的 prompt 列表
        n_steps: 训练步数
        """
        for step in range(n_steps):
            # ---- Rollout 阶段 ----
            batch_trajectories = []
            for prompt in prompts:
                group = []
                for _ in range(self.group_size):
                    traj = self.worker.rollout(prompt, self.reward_fn)
                    group.append(traj)
                batch_trajectories.append(group)

            # ---- Reward 归一化（GRPO 的组内比较）----
            all_rewards = []
            for group in batch_trajectories:
                group_rewards = [t["reward"] for t in group]
                mean_r = sum(group_rewards) / len(group_rewards)
                std_r = (sum((r - mean_r) ** 2 for r in group_rewards) / len(group_rewards)) ** 0.5 + 1e-8
                for t, r in zip(group, group_rewards):
                    t["advantage"] = (r - mean_r) / std_r
                all_rewards.extend(group_rewards)

            # ---- Train 阶段 ----
            train_data = []
            for group in batch_trajectories:
                for traj in group:
                    full_response = self._serialize_trajectory(traj)
                    train_data.append((
                        traj["prompt"],
                        full_response,
                        traj["advantage"],
                    ))

            loss = self.policy.train_step_with_advantage(train_data)

            # ---- 记录指标 ----
            mean_reward = sum(all_rewards) / len(all_rewards)
            self.history.append({
                "step": step,
                "loss": loss,
                "mean_reward": mean_reward,
                "max_reward": max(all_rewards),
            })
            if step % 5 == 0:
                print(f"Step {step:3d} | loss={loss:.4f} | "
                      f"reward_mean={mean_reward:.3f} | "
                      f"reward_max={max(all_rewards):.3f}")

        return self.history

    def _serialize_trajectory(self, traj: dict) -> str:
        """把多轮轨迹序列化为一段文本，用于 train_step。"""
        parts = []
        for interaction in traj["interactions"]:
            parts.append(f"Assistant: {interaction['response']}")
            if interaction["observation"]:
                parts.append(f"Observation: {interaction['observation']}")
        return "\n".join(parts)
