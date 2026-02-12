from __future__ import annotations

import pytest

from autovram.engines.registry import resolve_engine


def test_resolve_unknown_engine() -> None:
    with pytest.raises(ValueError):
        resolve_engine("does-not-exist")


def test_llm_autobatch_missing_dependency_message() -> None:
    # Force selection. In most test envs llm_autobatch won't be installed.
    with pytest.raises(RuntimeError) as ei:
        resolve_engine("llm-autobatch")
    assert "Install" in str(ei.value)
