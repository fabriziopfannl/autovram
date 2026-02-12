from __future__ import annotations

import platform
import re
import sys
from contextlib import suppress
from dataclasses import asdict
from typing import Any

from .types import SystemInfo


def _try_import_torch() -> Any | None:
    try:
        import torch  # type: ignore

        return torch
    except Exception:
        return None


def detect_system() -> SystemInfo:
    """Detect system information.

    This function is intentionally dependency-light. It uses PyTorch if available
    to detect CUDA/MPS/ROCm presence and VRAM.
    """

    os_name = platform.system()
    arch = platform.machine()
    py = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    torch = _try_import_torch()
    torch_version: str | None = None
    compute = "CPU"
    gpu_name: str | None = None
    vram_total_mib: int | None = None
    extra: dict[str, Any] = {}

    if torch is not None:
        torch_version = getattr(torch, "__version__", None)

        # CUDA
        cuda_available = bool(getattr(torch.cuda, "is_available", lambda: False)())
        if cuda_available:
            compute = "CUDA"
            try:
                idx = int(torch.cuda.current_device())
                props = torch.cuda.get_device_properties(idx)
                gpu_name = str(getattr(props, "name", None))
                total = int(getattr(props, "total_memory", 0))
                vram_total_mib = int(total / (1024 * 1024)) if total else None
            except Exception as e:
                extra["cuda_error"] = repr(e)

            # Best-effort versions
            with suppress(Exception):
                extra["cuda_runtime"] = str(getattr(torch.version, "cuda", None))

        # Apple MPS
        mps_backend = getattr(torch.backends, "mps", None)
        if compute == "CPU" and mps_backend is not None:
            try:
                if bool(getattr(mps_backend, "is_available", lambda: False)()):
                    compute = "MPS"
            except Exception:
                pass

        # ROCm (best-effort heuristic)
        if compute == "CPU" and torch_version is not None and re.search(r"\+rocm", torch_version):
            compute = "ROCm"

    return SystemInfo(
        os=os_name,
        arch=arch,
        python=py,
        torch_version=torch_version,
        compute=compute,
        gpu_name=gpu_name,
        vram_total_mib=vram_total_mib,
        extra=extra,
    )


def system_summary_lines(system: SystemInfo) -> list[str]:
    lines = [
        "System summary",
        "──────────────",
        f"OS: {system.os} ({system.arch})",
        f"Python: {system.python}",
        f"PyTorch: {system.torch_version or 'not installed'}",
        f"Compute: {system.compute}",
    ]

    if system.gpu_name:
        lines.append(f"GPU: {system.gpu_name}")
    if system.vram_total_mib is not None:
        lines.append(f"VRAM: {system.vram_total_mib} MiB")

    # Keep extra compact and stable.
    if system.extra:
        for k in sorted(system.extra.keys()):
            v = system.extra[k]
            lines.append(f"{k}: {v}")

    return lines


def system_summary_dict(system: SystemInfo) -> dict[str, Any]:
    return asdict(system)
