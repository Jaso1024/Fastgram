__all__ = ["GramEngine"]


def __getattr__(name: str):
    if name == "GramEngine":
        from .engine import GramEngine

        return GramEngine
    raise AttributeError(name)
