"""Runtime helpers for integrating autovram with user scripts.

In script mode, autovram communicates via environment variables.
This module provides a small, stable API to read them.
"""

from .runtime import RuntimeConfig, get_runtime_config, print_metric

__all__ = ["RuntimeConfig", "get_runtime_config", "print_metric"]
