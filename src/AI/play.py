import argparse
import time

import numpy as np
import torch

from env_2048 import Game2048
from dqn import DQNAgent


def main(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    env = Game2048(seed=args.seed)
    agent = DQNAgent(device=device)
    agent.load(args.model, map_location=device)

    state = env.reset()
    done = False
    steps = 0
    total_reward = 0.0
    info = {}

    while not done and steps < args.max_steps:
        env.render()
        action = agent.act(state, eval_mode=True)
        state, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        time.sleep(args.delay)

    env.render()
    print(f"Final score: {info.get('score', 0)}, total reward: {total_reward:.1f}, steps: {steps}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/dqn_2048.pt")
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--delay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    main(args)