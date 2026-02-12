from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from autovram.core.types import AutoVRAMConfig

ExportFormat = Literal["json", "yaml", "dotenv"]


def export_config(config: AutoVRAMConfig, *, fmt: ExportFormat, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "json":
        out_path.write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")
        return

    if fmt == "dotenv":
        lines = []
        d = asdict(config)
        for k, v in d.items():
            if v is None:
                continue
            env_key = f"AUTOVRAM_{k.upper()}"
            lines.append(f"{env_key}={v}")
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    if fmt == "yaml":
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "YAML export requires optional dependency. Install: pip install 'autovram[yaml]'"
            ) from e
        out_path.write_text(yaml.safe_dump(asdict(config), sort_keys=False), encoding="utf-8")
        return

    raise ValueError(f"Unknown export format: {fmt}")
