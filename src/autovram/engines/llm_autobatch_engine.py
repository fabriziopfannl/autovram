from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

from autovram.core.types import AutoVRAMConfig, RunResult

from .base import Engine, EngineContext


class LlmAutobatchEngine:
    """Optional integration with `llm-autobatch`.

    This is intentionally minimal in v1: we only use it to propose candidate
    batch sizes / microbatch sizes when available.
    """

    name = "llm-autobatch"

    def __init__(self) -> None:
        try:
            import llm_autobatch  # type: ignore

            self._lib = llm_autobatch
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Optional dependency not installed. Install: pip install 'autovram[llm-autobatch]'"
            ) from e

    def propose_configs(self, context: EngineContext) -> Iterable[AutoVRAMConfig]:
        # The upstream project API may change; keep this best-effort and safe.
        base = AutoVRAMConfig(batch_size=1, micro_batch=1)

        proposer = getattr(self._lib, "propose", None)
        if proposer is None:
            # Fall back to a small conservative set.
            for bs in (1, 2, 4, 8):
                yield replace(base, batch_size=bs, micro_batch=bs)
            return

        try:
            # Best-effort: allow propose(context=...) style.
            candidates = proposer()  # type: ignore[misc]
        except TypeError:
            candidates = proposer(context=None)  # type: ignore[misc]
        except Exception:
            candidates = []

        for c in candidates or []:
            bs = int(getattr(c, "batch_size", 1)) if hasattr(c, "batch_size") else int(c)
            yield replace(base, batch_size=bs, micro_batch=bs)

    def score_result(self, result: RunResult) -> float:
        return float(result.metric_value or 0.0)


def get_engine() -> Engine:
    return LlmAutobatchEngine()
