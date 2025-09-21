import os
import argparse
from tqdm import trange

import numpy as np
import torch

from env_2048 import Game2048
from dqn import DQNAgent, DQNConfig


def _print_device_info(device_str: str):
    dev = torch.device(device_str)
    print(f"[Device] Using device: {dev}")
    print(f"[PyTorch] torch={torch.__version__}, cuda_build={torch.version.cuda}, cuda_available={torch.cuda.is_available()}")
    if dev.type == "cuda" and torch.cuda.is_available():
        idx = dev.index or 0
        print(f"[CUDA] device_count={torch.cuda.device_count()}, selected_index={idx}, name={torch.cuda.get_device_name(idx)}")


def _select_device(args) -> str:
    if args.cpu:
        _print_device_info("cpu")
        return "cpu"
    if args.device == "cuda":
        # 显式要求用 CUDA
        if not torch.cuda.is_available():
            print("[Warning] --device=cuda 但当前 CUDA 不可用，改用 CPU。请按说明安装带 CUDA 的 PyTorch 或更新驱动。")
            _print_device_info("cpu")
            return "cpu"
        idx = args.gpu_index if args.gpu_index < torch.cuda.device_count() else 0
        dev = f"cuda:{idx}"
        torch.cuda.set_device(idx)
        _print_device_info(dev)
        return dev
    # auto: 自动选择
    if torch.cuda.is_available():
        idx = args.gpu_index if args.gpu_index < torch.cuda.device_count() else 0
        dev = f"cuda:{idx}"
        torch.cuda.set_device(idx)
        _print_device_info(dev)
        return dev
    else:
        print("[Info] 未检测到可用 CUDA，使用 CPU。")
        _print_device_info("cpu")
        return "cpu"


def train(args):
    device = _select_device(args)
    env = Game2048(seed=args.seed)
    agent = DQNAgent(
        state_dim=16,
        num_actions=4,
        device=device,
        config=DQNConfig(
            gamma=0.99,
            lr=args.lr,
            batch_size=args.batch_size,
            target_update=args.target_update,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay_steps=args.eps_decay_steps,
            replay_capacity=args.replay_capacity,
        ),
    )

    os.makedirs("models", exist_ok=True)

    global_step = 0
    best_score = 0

    for ep in trange(args.episodes, desc="Training"):
        state = env.reset()
        ep_reward = 0.0
        ep_steps = 0
        done = False
        info = {}

        while not done and ep_steps < args.max_steps:
            action = agent.act(state, eval_mode=False)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            loss = agent.learn()

            state = next_state
            ep_reward += reward
            ep_steps += 1
            global_step += 1

        best_score = max(best_score, info.get("score", 0))
        if (ep + 1) % max(1, args.checkpoint_every) == 0:
            save_path = os.path.join("models", "dqn_2048.pt")
            agent.save(save_path)

    final_path = os.path.join("models", "dqn_2048.pt")
    agent.save(final_path)
    print(f"Saved model to: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--target-update", type=int, default=1000)
    parser.add_argument("--eps-decay-steps", type=int, default=100_000)
    parser.add_argument("--replay-capacity", type=int, default=200_000)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="auto(默认)/cuda/cpu")
    parser.add_argument("--gpu-index", type=int, default=0, help="当有多块 GPU 时选择索引")
    args = parser.parse_args()
    train(args)