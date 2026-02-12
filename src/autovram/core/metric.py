from __future__ import annotations

import re

_METRIC_RE = re.compile(r"^AUTOVRAM_METRIC\s+(?P<key>[A-Za-z0-9_]+)=(?P<val>[-+0-9.eE]+)\s*$")


def parse_metric(stdout: str, metric_name: str) -> float | None:
    """Parse metric lines from stdout.

    Expected format:
        AUTOVRAM_METRIC it_per_s=12.3

    Returns the last seen value for the requested metric name.
    """

    last: float | None = None
    for line in stdout.splitlines():
        m = _METRIC_RE.match(line.strip())
        if not m:
            continue
        if m.group("key") != metric_name:
            continue
        try:
            last = float(m.group("val"))
        except ValueError:
            continue
    return last
