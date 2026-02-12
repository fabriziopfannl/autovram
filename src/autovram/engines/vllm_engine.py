from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

from autovram.core.types import AutoVRAMConfig, RunResult

from .base import Engine, EngineContext


class VLLMEngine:
    """vLLM-oriented engine (best-effort).

    The MVP keeps this conservative: propose a few dtypes and let the core
    tune a primary knob like max_num_seqs.

    This engine does not hard-depend on vLLM.
    """

    name = "vllm"

    def propose_configs(self, context: EngineContext) -> Iterable[AutoVRAMConfig]:
        base = AutoVRAMConfig(batch_size=1, micro_batch=1)
        for dtype in ("float16", "bfloat16", "float32"):
            yield replace(base, dtype=dtype)

    def score_result(self, result: RunResult) -> float:
        return float(result.metric_value or 0.0)


def get_engine() -> Engine:
    return VLLMEngine()
