from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, cast

Precision = Literal["fp32", "fp16", "bf16"]


@dataclass(frozen=True)
class RuntimeConfig:
    batch_size: int
    micro_batch: int
    grad_accum: int
    precision: Precision
    seq_len: int | None


def _get_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def get_runtime_config() -> RuntimeConfig:
    """Read tuning knobs from environment variables.

    Environment variables:

    - AUTOVRAM_BATCH_SIZE
    - AUTOVRAM_MICRO_BATCH
    - AUTOVRAM_GRAD_ACCUM
    - AUTOVRAM_PRECISION (fp32|fp16|bf16)
    - AUTOVRAM_SEQ_LEN
    """

    bs = _get_int("AUTOVRAM_BATCH_SIZE", 1)
    mb = _get_int("AUTOVRAM_MICRO_BATCH", bs)
    ga = _get_int("AUTOVRAM_GRAD_ACCUM", 1)

    precision_raw = os.environ.get("AUTOVRAM_PRECISION", "fp16").strip().lower()
    precision = cast(
        Precision,
        precision_raw if precision_raw in ("fp32", "fp16", "bf16") else "fp16",
    )

    seq_raw = os.environ.get("AUTOVRAM_SEQ_LEN")
    seq_len: int | None = None
    if seq_raw is not None:
        try:
            seq_len = int(seq_raw)
        except ValueError:
            seq_len = None

    return RuntimeConfig(
        batch_size=bs, micro_batch=mb, grad_accum=ga, precision=precision, seq_len=seq_len
    )


def print_metric(**metrics: float) -> None:
    """Print metrics in a format autovram can parse.

    Example:
        print_metric(it_per_s=12.3)

    Output:
        AUTOVRAM_METRIC it_per_s=12.3
    """

    for k, v in metrics.items():
        print(f"AUTOVRAM_METRIC {k}={float(v)}")
