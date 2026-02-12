from __future__ import annotations

import re

_OOM_PATTERNS = [
    r"CUDA out of memory",
    r"CUBLAS_STATUS_ALLOC_FAILED",
    r"cuBLAS.*alloc",
    r"out of memory",
    r"hipErrorOutOfMemory",
    r"MPS.*out of memory",
]

_OOM_RE = re.compile("|".join(f"(?:{p})" for p in _OOM_PATTERNS), re.IGNORECASE)


def looks_like_oom(stderr: str) -> bool:
    """Best-effort OOM detection from stderr.

    This is intentionally broad to catch different frameworks.
    """

    return bool(_OOM_RE.search(stderr))
