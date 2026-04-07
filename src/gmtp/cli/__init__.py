from __future__ import annotations

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    from .main import main as _main

    return _main(argv)
