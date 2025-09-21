import math
import random
from typing import Tuple, Dict, Any, Optional

import numpy as np



class Game2048:
    def __init__(self, size: int = 4, seed: Optional[int] = None):
        self.size = size
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.board = np.zeros((size, size), dtype=np.int64)
        self.score = 0
        self.done = False

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng = random.Random(seed)
            self.np_rng = np.random.default_rng(seed)
        self.board[:] = 0
        self.score = 0
        self.done = False
        self._add_random_tile()
        self._add_random_tile()
        return self._get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        # actions: 0=up, 1=down, 2=left, 3=right
        if self.done:
            return self._get_observation(), 0.0, True, {"info": "already done"}

        changed, gained = self._move(action)
        reward = float(gained)

        if not changed:
            # 惩罚无效动作（不生成新块）
            reward -= 1.0
        else:
            self.score += gained
            self._add_random_tile()
            if not self._can_move():
                self.done = True

        return self._get_observation(), reward, self.done, {"score": self.score}

    def render(self) -> None:
        print("-" * (self.size * 7 + 1))
        for r in range(self.size):
            row = "|"
            for c in range(self.size):
                v = self.board[r, c]
                row += f"{v:^6d}|"
            print(row)
            print("-" * (self.size * 7 + 1))
        print(f"Score: {self.score}")

    # 观测：将每个格子转为 log2(value)/16 的浮点数（空为0）
    def _get_observation(self) -> np.ndarray:
        obs = np.zeros((self.size, self.size), dtype=np.float32)
        nonzero = self.board > 0
        if np.any(nonzero):
            exponents = np.zeros_like(self.board, dtype=np.float32)
            exponents[nonzero] = np.log2(self.board[nonzero])
            obs = (exponents / 16.0).astype(np.float32)
        return obs.flatten()

    def _add_random_tile(self) -> None:
        empties = np.argwhere(self.board == 0)
        if len(empties) == 0:
            return
        idx = self.np_rng.integers(0, len(empties))
        r, c = empties[idx]
        value = 4 if self.np_rng.random() < 0.1 else 2
        self.board[r, c] = value

    def _can_move(self) -> bool:
        if np.any(self.board == 0):
            return True
        # Check merges horizontally
        for r in range(self.size):
            for c in range(self.size - 1):
                if self.board[r, c] == self.board[r, c + 1]:
                    return True
        # Check merges vertically
        for c in range(self.size):
            for r in range(self.size - 1):
                if self.board[r, c] == self.board[r + 1, c]:
                    return True
        return False

    def _move(self, action: int) -> Tuple[bool, int]:
        original = self.board.copy()
        reward = 0

        def compress_and_merge(row: np.ndarray) -> Tuple[np.ndarray, int]:
            non_zero = row[row != 0]
            merged = []
            gained_local = 0
            i = 0
            while i < len(non_zero):
                if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                    new_val = non_zero[i] * 2
                    merged.append(new_val)
                    gained_local += int(new_val)
                    i += 2
                else:
                    merged.append(int(non_zero[i]))
                    i += 1
            merged_arr = np.array(merged, dtype=np.int64)
            if len(merged_arr) < self.size:
                merged_arr = np.pad(merged_arr, (0, self.size - len(merged_arr)), constant_values=0)
            return merged_arr, gained_local

        if action == 2:  # left
            for r in range(self.size):
                new_row, g = compress_and_merge(self.board[r, :])
                self.board[r, :] = new_row
                reward += g
        elif action == 3:  # right
            for r in range(self.size):
                reversed_row = self.board[r, ::-1]
                new_row, g = compress_and_merge(reversed_row)
                self.board[r, ::-1] = new_row
                reward += g
        elif action == 0:  # up
            for c in range(self.size):
                col = self.board[:, c]
                new_col, g = compress_and_merge(col)
                self.board[:, c] = new_col
                reward += g
        elif action == 1:  # down
            for c in range(self.size):
                col = self.board[::-1, c]
                new_col, g = compress_and_merge(col)
                self.board[::-1, c] = new_col
                reward += g
        else:
            raise ValueError("Invalid action")

        changed = not np.array_equal(original, self.board)
        return changed, reward

    def legal_actions(self) -> list[int]:
        """
        返回当前棋盘下的合法动作列表（0=up, 1=down, 2=left, 3=right）。
        通过模拟一步动作并回滚判断是否会产生变化。
        """
        legal: list[int] = []
        board_backup = self.board.copy()
        score_backup = self.score
        for a in range(4):
            changed, _ = self._move(a)
            # 回滚
            self.board[...] = board_backup
            self.score = score_backup
            if changed:
                legal.append(a)
        return legal