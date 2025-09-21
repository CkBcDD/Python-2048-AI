from __future__ import annotations
from typing import Tuple, Dict, List, Optional
from game_core import Game2048Core, ACTIONS, ACTION_NAMES


class Game2048Env:
    """
    极简 Env 接口：
    - reset(seed: Optional[int]) -> state
    - step(action: int) -> (state, reward, done, info)
    - get_state() -> state
    - legal_actions() -> List[int]
    - score 属性
    """
    def __init__(self, size: int = 4, seed: Optional[int] = None):
        self.core = Game2048Core(size=size, seed=seed)

    @property
    def score(self) -> int:
        return self.core.score

    def reset(self, seed: Optional[int] = None) -> List[List[int]]:
        return self.core.reset(seed=seed)

    def get_state(self) -> List[List[int]]:
        return self.core.get_state()

    def step(self, action: int) -> Tuple[List[List[int]], int, bool, Dict]:
        return self.core.step(action)

    def legal_actions(self) -> List[int]:
        return self.core.legal_actions()

    def is_over(self) -> bool:
        return self.core.is_over()


# 便于外部引用
ACTIONS = ACTIONS
ACTION_NAMES = ACTION_NAMES