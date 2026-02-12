from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Precision = Literal["fp32", "fp16", "bf16"]


@dataclass(frozen=True)
class SystemInfo:
    os: str
    arch: str
    python: str
    torch_version: str | None
    compute: str
    gpu_name: str | None
    vram_total_mib: int | None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AutoVRAMConfig:
    """Common tuning knobs for script mode and vLLM mode.

    Not all fields are used in all modes.
    """

    # Script mode
    batch_size: int = 1
    micro_batch: int | None = None
    grad_accum: int = 1
    precision: Precision = "fp16"
    seq_len: int | None = None

    # vLLM mode
    dtype: str | None = None
    gpu_memory_utilization: float | None = None
    max_num_batched_tokens: int | None = None
    max_num_seqs: int | None = None
    quantization: str | None = None


@dataclass(frozen=True)
class RunContext:
    mode: Literal["script", "vllm"]
    system: SystemInfo
    metric_name: str
    timeout_s: float
    work_dir: Path
    exec_cwd: Path
    engine_name: str


@dataclass(frozen=True)
class RunResult:
    config: AutoVRAMConfig
    ok: bool
    exit_code: int | None
    timed_out: bool
    oom: bool
    metric_value: float | None
    stdout: str
    stderr: str
    duration_s: float
    artifacts_dir: Path
    notes: list[str] = field(default_factory=list)
