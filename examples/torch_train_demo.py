"""Tiny demo script for autovram script mode.

This script is designed to run even without a GPU. If PyTorch is installed and
CUDA is available, it will use GPU. Otherwise it uses CPU and prints metrics
anyway (useful for validating the control loop).

The goal is to demonstrate how to:
- read autovram env vars
- apply batch size and precision
- print an AUTOVRAM_METRIC line

This is not a realistic training loop.
"""

from __future__ import annotations

import time

from autovram.runtime import get_runtime_config, print_metric


def main() -> None:
    cfg = get_runtime_config()

    try:
        import torch

        has_torch = True
    except Exception:
        torch = None
        has_torch = False

    # Simulate a tiny workload that scales with batch size.
    steps = 50
    t0 = time.time()

    if has_torch:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else (
                "mps"
                if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                else "cpu"
            )
        )
        dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }.get(cfg.precision, torch.float16)

        x = torch.randn((cfg.micro_batch, 1024), device=device)
        w = torch.randn((1024, 1024), device=device, dtype=dtype)
        for _ in range(steps):
            y = x.to(dtype) @ w
            if device == "cuda":
                torch.cuda.synchronize()
            _ = y.sum().item()
    else:
        for _ in range(steps):
            _ = sum(i * i for i in range(20000 * max(1, cfg.micro_batch // 2)))

    dt = max(time.time() - t0, 1e-9)
    it_per_s = steps / dt

    print_metric(it_per_s=it_per_s)


if __name__ == "__main__":
    main()
