"""
第1章：CartPole 训练 + TensorBoard 可视化
在训练过程中记录指标，方便用 TensorBoard 观察训练曲线

运行方式：
    python hello_rl_tensorboard.py
    # 然后在另一个终端运行：
    tensorboard --logdir ./ppo_cartpole_tensorboard/
"""

import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# 创建输出目录
os.makedirs("output", exist_ok=True)

# 创建环境
env = gym.make("CartPole-v1")

# 初始化 PPO，启用 TensorBoard 日志
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./output/ppo_cartpole_tensorboard/",
)

print("开始训练（带 TensorBoard 日志）...")
model.learn(total_timesteps=20000)

# 评估
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"训练完成！平均奖励: {mean_reward} +/- {std_reward}")

model.save("output/ppo_cartpole_tb")
env.close()

print("\n训练完成！运行以下命令查看训练曲线：")
print("  tensorboard --logdir ./output/ppo_cartpole_tensorboard/")
print("  然后在浏览器打开 http://localhost:6006")
