from __future__ import annotations
import pygame
import sys
from typing import Callable, Optional, List
from api import Game2048Env, ACTIONS, ACTION_NAMES


# 颜色配置（简化版）
BG_COLOR = (250, 248, 239)
GRID_COLOR = (187, 173, 160)
EMPTY_COLOR = (205, 193, 180)
TILE_COLORS = {
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
}
TILE_TEXT_COLOR_DARK = (119, 110, 101)
TILE_TEXT_COLOR_LIGHT = (249, 246, 242)


def run_gui(agent: Optional[Callable[[List[List[int]]], int]] = None, size: int = 4):
    pygame.init()
    pygame.display.set_caption("2048 - Minimal PyGame")

    cell_size = 100
    margin = 15
    header_h = 90
    board_pixels = margin + size * (cell_size + margin)
    width = board_pixels
    height = header_h + board_pixels

    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()

    # 字体
    font_big = pygame.font.SysFont("arial", 48, bold=True)
    font_mid = pygame.font.SysFont("arial", 28, bold=True)
    font_small = pygame.font.SysFont("arial", 20)

    env = Game2048Env(size=size)
    env.reset()

    auto_mode = agent is not None
    step_interval_ms = 120
    last_step_time = 0

    def draw():
        screen.fill(BG_COLOR)

        # 顶部信息栏
        score_text = font_mid.render(f"Score: {env.score}", True, TILE_TEXT_COLOR_DARK)
        hint_text = font_small.render("Arrows: Move | R: Reset | A: Toggle Auto | Esc: Quit", True, TILE_TEXT_COLOR_DARK)
        screen.blit(score_text, (margin, (header_h - score_text.get_height()) // 2))
        screen.blit(hint_text, (margin, header_h - hint_text.get_height() - 8))

        # 棋盘背景
        board_top = header_h
        pygame.draw.rect(screen, GRID_COLOR, pygame.Rect(0, board_top, width, height - board_top))
        # 网格 + 方块
        for r in range(size):
            for c in range(size):
                x = margin + c * (cell_size + margin)
                y = board_top + margin + r * (cell_size + margin)
                val = env.get_state()[r][c]
                color = TILE_COLORS.get(val, EMPTY_COLOR if val == 0 else (60, 58, 50))
                pygame.draw.rect(screen, color, pygame.Rect(x, y, cell_size, cell_size), border_radius=6)
                if val:
                    text_color = TILE_TEXT_COLOR_DARK if val <= 4 else TILE_TEXT_COLOR_LIGHT
                    # 自适应字号
                    if val < 100:
                        f = font_big
                    elif val < 1000:
                        f = font_mid
                    else:
                        f = font_small
                    text = f.render(str(val), True, text_color)
                    screen.blit(text, (x + (cell_size - text.get_width()) // 2, y + (cell_size - text.get_height()) // 2))

        # 结束遮罩
        if env.is_over():
            overlay = pygame.Surface((width, height), pygame.SRCALPHA)
            overlay.fill((255, 255, 255, 180))
            screen.blit(overlay, (0, 0))
            go_text = font_big.render("Game Over", True, (80, 70, 60))
            screen.blit(go_text, (width // 2 - go_text.get_width() // 2, height // 2 - go_text.get_height() // 2))

        pygame.display.flip()

    def key_to_action(key: int) -> Optional[int]:
        if key == pygame.K_UP:
            return 0
        if key == pygame.K_RIGHT:
            return 1
        if key == pygame.K_DOWN:
            return 2
        if key == pygame.K_LEFT:
            return 3
        return None

    def random_valid_action(state: List[List[int]]) -> Optional[int]:
        # 演示用的简单自动动作：尝试优先顺序，上右下左中第一个合法
        for a in ACTIONS:
            if a in env.legal_actions():
                return a
        return None

    if agent is None:
        agent = random_valid_action  # 没提供 agent 时，A 开关使用内置随机策略

    running = True
    while running:
        dt = clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    break
                if event.key == pygame.K_r:
                    env.reset()
                if event.key == pygame.K_a:
                    auto_mode = not auto_mode
                action = key_to_action(event.key)
                if action is not None and not env.is_over():
                    env.step(action)

        if auto_mode and not env.is_over():
            now = pygame.time.get_ticks()
            if now - last_step_time >= step_interval_ms:
                st = env.get_state()
                act = agent(st)
                if act is not None:
                    env.step(act)
                last_step_time = now

        draw()

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    run_gui()