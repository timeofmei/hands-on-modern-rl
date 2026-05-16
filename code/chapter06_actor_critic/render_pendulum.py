"""
第6章：渲染 A2C Pendulum-v1 回放

运行方式：
    python render_pendulum.py --model output/actor_critic_pendulum.zip
"""

import argparse
from pathlib import Path

import gymnasium as gym
import imageio
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def main():
    parser = argparse.ArgumentParser(description="渲染 Pendulum A2C 回放")
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--vecnormalize", type=str,
                        default="output/actor_critic_pendulum_vecnormalize.pkl",
                        help="VecNormalize 统计文件")
    parser.add_argument("--output", type=str, default="output/pendulum_actor_critic.gif",
                        help="GIF 输出路径")
    parser.add_argument("--seed", type=int, default=0, help="环境 seed")
    parser.add_argument("--max-steps", type=int, default=200, help="最大步数")
    parser.add_argument("--fps", type=int, default=30, help="GIF 帧率")
    args = parser.parse_args()

    model = A2C.load(args.model)
    render_env = gym.make("Pendulum-v1", render_mode="rgb_array")
    vec_env = DummyVecEnv([lambda: render_env])
    vec_env = VecNormalize.load(args.vecnormalize, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    vec_env.seed(args.seed)
    obs = vec_env.reset()
    frames = []
    total_reward = 0.0

    for _ in range(args.max_steps):
        frames.append(render_env.render())
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = vec_env.step(action)
        total_reward += float(reward[0])
        if done[0]:
            break

    vec_env.close()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, duration=1000 / args.fps, loop=0)
    print(f"回报: {total_reward:.1f}")
    print(f"GIF 已保存到 {output_path}")


if __name__ == "__main__":
    main()
