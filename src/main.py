import os
import argparse
from typing import List, Optional

import numpy as np
import torch

from game.gui_pygame import run_gui
from AI.env_2048 import Game2048
from AI.dqn import DQNAgent, DQNConfig


def train_model(
    model_path: str,
    size: int = 4,
    seed: int = 42,
    device: str = "cpu",
    episodes: int = 200,
    max_steps: int = 2000,
    batch_size: int = 256,
    lr: float = 1e-3,
    target_update: int = 1000,
    eps_decay_steps: int = 100_000,
    replay_capacity: int = 200_000,
):
    assert size == 4, "当前示例仅支持 4x4 棋盘（DQN 输入维度需对应修改）"
    env = Game2048(size=size, seed=seed)
    agent = DQNAgent(
        state_dim=size * size,
        num_actions=4,
        device=device,
        config=DQNConfig(
            gamma=0.99,
            lr=lr,
            batch_size=batch_size,
            target_update=target_update,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay_steps=eps_decay_steps,
            replay_capacity=replay_capacity,
        ),
    )

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)

    global_step = 0
    for ep in range(episodes):
        state = env.reset(seed=seed + ep)
        ep_steps = 0
        done = False
        while not done and ep_steps < max_steps:
            # 基于合法动作修正 DQN 输出，避免无效操作
            legal = env.legal_actions()
            action = agent.act(state, eval_mode=False)
            if action not in legal:
                # 退化为随机合法动作（亦可在 DQN 内部实现基于 mask 的选取）
                action = int(np.random.choice(legal))

            next_state, reward, done, _ = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            agent.learn()

            state = next_state
            ep_steps += 1
            global_step += 1

        # 周期性保存
        if (ep + 1) % max(1, min(50, max(1, episodes // 10))) == 0:
            agent.save(model_path)

    agent.save(model_path)
    print(f"模型已保存: {model_path}")


def build_gui_agent(model_path: str, size: int = 4, device: str = "cpu"):
    assert size == 4, "当前示例仅支持 4x4 棋盘（DQN 输入维度需对应修改）"

    agent = DQNAgent(state_dim=size * size, num_actions=4, device=device)
    if os.path.exists(model_path):
        agent.load(model_path, map_location=device)
        print(f"已加载模型: {model_path}")
    else:
        print(f"未找到模型: {model_path}，GUI 将使用未训练权重（建议先训练）。")

    # DQN 行为空间: 0=up, 1=down, 2=left, 3=right
    # GUI/核心行为空间: 0=up, 1=right, 2=down, 3=left
    dqn_to_gui = [0, 2, 3, 1]

    def board_to_obs(state: List[List[int]]) -> np.ndarray:
        arr = np.array(state, dtype=np.int64).reshape(size, size)
        obs = np.zeros_like(arr, dtype=np.float32)
        nz = arr > 0
        if np.any(nz):
            obs[nz] = np.log2(arr[nz]).astype(np.float32) / 16.0
        return obs.flatten()

    def agent_fn(state: List[List[int]], legal_actions_gui: List[int]) -> Optional[int]:
        """
        GUI 调用的 Agent：确保返回值在 legal_actions_gui 内。
        """
        obs = board_to_obs(state)
        a_dqn = agent.act(obs, eval_mode=True)
        a_gui = dqn_to_gui[a_dqn]
        if a_gui not in legal_actions_gui:
            # 若 DQN 输出非法，回退到第一个合法动作
            return legal_actions_gui[0] if len(legal_actions_gui) > 0 else None
        return a_gui

    return agent_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="models/dqn_2048.pt")
    parser.add_argument("--train-episodes", type=int, default=0, help="训练轮数；>0 时先训练再进入 GUI")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--target-update", type=int, default=1000)
    parser.add_argument("--eps-decay-steps", type=int, default=100_000)
    parser.add_argument("--replay-capacity", type=int, default=200_000)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    # 若需训练或模型不存在则先训练
    need_train = args.train_episodes > 0 or (not os.path.exists(args.model))
    if need_train:
        episodes = args.train_episodes if args.train_episodes > 0 else 200
        print(f"开始训练（episodes={episodes}）...")
        train_model(
            model_path=args.model,
            size=args.size,
            seed=args.seed,
            device=device,
            episodes=episodes,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            lr=args.lr,
            target_update=args.target_update,
            eps_decay_steps=args.eps_decay_steps,
            replay_capacity=args.replay_capacity,
        )

    gui_agent = build_gui_agent(args.model, size=args.size, device=device)
    # 传入 agent 后，GUI 默认自动模式已开启，可用 A 键开关
    run_gui(agent=gui_agent, size=args.size)


if __name__ == "__main__":
    main()
