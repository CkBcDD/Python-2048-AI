from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import random


# 行为定义：0=上, 1=右, 2=下, 3=左
ACTIONS: Tuple[int, int, int, int] = (0, 1, 2, 3)
ACTION_NAMES: Dict[int, str] = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}


class Game2048Core:
    def __init__(self, size: int = 4, seed: Optional[int] = None):
        self.size = size
        self.rng = random.Random(seed)
        self.board: List[List[int]] = [[0] * size for _ in range(size)]
        self.score: int = 0
        self.done: bool = False

    def reset(self, seed: Optional[int] = None) -> List[List[int]]:
        if seed is not None:
            self.rng.seed(seed)
        self.board = [[0] * self.size for _ in range(self.size)]
        self.score = 0
        self.done = False
        self._spawn_tile()
        self._spawn_tile()
        return self.get_state()

    def get_state(self) -> List[List[int]]:
        return [row[:] for row in self.board]

    def is_over(self) -> bool:
        return self.done

    def step(self, action: int) -> Tuple[List[List[int]], int, bool, Dict]:
        if self.done:
            return self.get_state(), 0, True, {}

        moved, reward = self._move(action)
        if moved:
            self._spawn_tile()
        self.score += reward
        self.done = not self._can_move()
        return self.get_state(), reward, self.done, {"score": self.score}

    def _spawn_tile(self) -> None:
        empties = [(r, c) for r in range(self.size) for c in range(self.size) if self.board[r][c] == 0]
        if not empties:
            return
        r, c = self.rng.choice(empties)
        self.board[r][c] = 4 if self.rng.random() < 0.1 else 2

    def _can_move(self) -> bool:
        # 任一空格
        if any(0 in row for row in self.board):
            return True
        # 任一相邻可合并
        for r in range(self.size):
            for c in range(self.size):
                v = self.board[r][c]
                if r + 1 < self.size and self.board[r + 1][c] == v:
                    return True
                if c + 1 < self.size and self.board[r][c + 1] == v:
                    return True
        return False

    def _move(self, action: int) -> Tuple[bool, int]:
        moved = False
        total_reward = 0

        def move_line(line: List[int]) -> Tuple[List[int], int, bool]:
            filtered = [x for x in line if x != 0]
            reward = 0
            i = 0
            out: List[int] = []
            while i < len(filtered):
                if i + 1 < len(filtered) and filtered[i] == filtered[i + 1]:
                    merged = filtered[i] * 2
                    out.append(merged)
                    reward += merged
                    i += 2
                else:
                    out.append(filtered[i])
                    i += 1
            out += [0] * (self.size - len(out))
            return out, reward, out != line

        if action == 3:  # LEFT
            for r in range(self.size):
                new_line, reward, line_moved = move_line(self.board[r])
                if line_moved:
                    moved = True
                self.board[r] = new_line
                total_reward += reward

        elif action == 1:  # RIGHT
            for r in range(self.size):
                reversed_row = list(reversed(self.board[r]))
                new_line, reward, line_moved = move_line(reversed_row)
                new_line = list(reversed(new_line))
                if line_moved:
                    moved = True
                self.board[r] = new_line
                total_reward += reward

        elif action == 0:  # UP
            for c in range(self.size):
                col = [self.board[r][c] for r in range(self.size)]
                new_col, reward, col_moved = move_line(col)
                if col_moved:
                    moved = True
                for r in range(self.size):
                    self.board[r][c] = new_col[r]
                total_reward += reward

        elif action == 2:  # DOWN
            for c in range(self.size):
                col = [self.board[r][c] for r in range(self.size)]
                col_rev = list(reversed(col))
                new_col, reward, col_moved = move_line(col_rev)
                new_col = list(reversed(new_col))
                if col_moved:
                    moved = True
                for r in range(self.size):
                    self.board[r][c] = new_col[r]
                total_reward += reward

        else:
            # 非法动作不移动
            return False, 0

        return moved, total_reward

    def max_tile(self) -> int:
        return max(max(row) for row in self.board)

    def legal_actions(self) -> List[int]:
        # 返回会产生变化的动作集合
        legal = []
        for a in ACTIONS:
            snapshot = self.get_state()
            score_snapshot = self.score
            moved, _ = self._move(a)
            # 回滚
            self.board = snapshot
            self.score = score_snapshot
            if moved:
                legal.append(a)
        return legal
