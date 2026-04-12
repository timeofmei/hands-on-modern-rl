import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# 创建输出目录
os.makedirs("output", exist_ok=True)

# ==========================================
# 第一阶段：训练智能体
# ==========================================
print("正在创建 CartPole 环境...")
# 创建倒立摆环境
env = gym.make("CartPole-v1")

# 初始化 PPO (近端策略优化) 算法模型，使用多层感知机 (MlpPolicy)
model = PPO("MlpPolicy", env, verbose=1)

print("开始训练，请稍候 (通常只需几秒钟)...")
# 训练 20000 个时间步
model.learn(total_timesteps=20000)

# 评估训练好的模型
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"训练完成！平均奖励: {mean_reward} +/- {std_reward}")

# 保存模型
model.save("output/ppo_cartpole")
env.close()

# ==========================================
# 第二阶段：可视化展示学习成果
# ==========================================
print("正在展示智能体的学习成果...")
# 重新创建一个带有渲染画面的环境
env = gym.make("CartPole-v1", render_mode="human")
model = PPO.load("output/ppo_cartpole")

# 运行 5 个回合的视觉演示
for episode in range(5):
    obs, info = env.reset()
    done = False
    truncated = False
    score = 0
    
    while not (done or truncated):
        # 智能体根据当前观察 (obs) 决定动作
        action, _states = model.predict(obs, deterministic=True)
        # 环境执行动作，返回新的状态和奖励
        obs, reward, done, truncated, info = env.step(action)
        score += reward
        
    print(f"回合 {episode + 1} 得分: {score}")

env.close()
