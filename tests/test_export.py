from __future__ import annotations

import json
from pathlib import Path

from autovram.core.types import AutoVRAMConfig
from autovram.io.export import export_config


def test_export_json(tmp_path: Path) -> None:
    cfg = AutoVRAMConfig(batch_size=3, micro_batch=3, grad_accum=2, precision="bf16")
    out = tmp_path / "cfg.json"
    export_config(cfg, fmt="json", out_path=out)

    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["batch_size"] == 3
    assert loaded["precision"] == "bf16"


def test_export_dotenv(tmp_path: Path) -> None:
    cfg = AutoVRAMConfig(batch_size=2, micro_batch=2, grad_accum=1, precision="fp16")
    out = tmp_path / ".env"
    export_config(cfg, fmt="dotenv", out_path=out)

    text = out.read_text(encoding="utf-8")
    assert "AUTOVRAM_BATCH_SIZE=2" in text
    assert "AUTOVRAM_PRECISION=fp16" in text
