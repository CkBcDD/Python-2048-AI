import importlib
import pkgutil
from typing import List

# AI package init
__all__ = [m.name for m in pkgutil.iter_modules(__path__) if not m.name.startswith("_")]  # type: ignore[reportUnsupportedDunderAll]

def __getattr__(name: str):
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"{__name__} has no attribute {name!r}")

def __dir__() -> List[str]:
    return sorted(list(globals().keys()) + __all__)

def available_modules() -> List[str]:
    return list(__all__)

def load_module(name: str):
    if name not in __all__:
        raise ImportError(f"Unknown submodule: {name}")
    return importlib.import_module(f"{__name__}.{name}")