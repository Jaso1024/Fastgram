__all__ = ["gram", "GramEngine"]


def __getattr__(name: str):
    if name == "gram" or name == "GramEngine":
        from .engine import GramEngine

        return GramEngine
    raise AttributeError(name)
