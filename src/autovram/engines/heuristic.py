from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

from autovram.core.types import AutoVRAMConfig, RunResult

from .base import Engine, EngineContext


class HeuristicEngine:
    """Simple, dependency-free engine.

    This engine proposes a small set of precisions and expects the core
    to run a batch-size search.
    """

    name = "heuristic"

    def __init__(self, *, precisions: tuple[str, ...] = ("fp16", "bf16", "fp32")):
        self._precisions = precisions

    def propose_configs(self, context: EngineContext) -> Iterable[AutoVRAMConfig]:
        base = AutoVRAMConfig(batch_size=1, micro_batch=1)
        for p in self._precisions:
            yield replace(base, precision=p)  # type: ignore[arg-type]

    def score_result(self, result: RunResult) -> float:
        return float(result.metric_value or 0.0)


def get_engine() -> Engine:
    return HeuristicEngine()
