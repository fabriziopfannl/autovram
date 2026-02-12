from __future__ import annotations

from collections.abc import Callable

from .base import Engine
from .heuristic import get_engine as get_heuristic
from .vllm_engine import get_engine as get_vllm


def available_engines() -> list[str]:
    names = ["heuristic", "torch", "vllm"]

    # Optional: llm-autobatch
    try:
        import llm_autobatch  # noqa: F401

        names.append("llm-autobatch")
    except Exception:
        pass

    return names


def resolve_engine(name: str) -> Engine:
    normalized = name.strip().lower()

    factories: dict[str, Callable[[], Engine]] = {
        "heuristic": get_heuristic,
        "torch": get_heuristic,
        "vllm": get_vllm,
    }

    if normalized == "llm-autobatch":
        try:
            from .llm_autobatch_engine import get_engine as get_llm_autobatch

            return get_llm_autobatch()
        except RuntimeError as e:
            raise RuntimeError(str(e)) from e

    if normalized not in factories:
        raise ValueError(f"Unknown engine '{name}'. Available: {', '.join(available_engines())}")

    return factories[normalized]()
