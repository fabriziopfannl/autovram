from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from autovram.core.types import AutoVRAMConfig, RunContext, RunResult


@dataclass(frozen=True)
class EngineContext:
    run: RunContext


class Engine(Protocol):
    """Engine interface.

    Engines propose configurations and optionally provide scoring.
    The core tuner owns the search loop and stability/OOM handling.
    """

    name: str

    def propose_configs(self, context: EngineContext) -> Iterable[AutoVRAMConfig]: ...

    def score_result(self, result: RunResult) -> float: ...
