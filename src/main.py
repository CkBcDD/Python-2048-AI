from gui_pygame import run_gui

if __name__ == "__main__":
    # 可传入自定义 agent: Callable[[state], action:int]
    run_gui(agent=None, size=4)
