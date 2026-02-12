from __future__ import annotations

import os
import shlex
import signal
import subprocess
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from .oom import looks_like_oom


@dataclass(frozen=True)
class SubprocessOutcome:
    ok: bool
    exit_code: int | None
    timed_out: bool
    oom: bool
    stdout: str
    stderr: str
    duration_s: float


class SubprocessRunner:
    """Run commands with timeouts and robust termination."""

    def run(self, cmd: str, env: dict[str, str], timeout_s: float, cwd: Path) -> SubprocessOutcome:
        start = time.time()

        merged_env = os.environ.copy()
        merged_env.update(env)

        # Use shell=False to avoid quoting issues; accept cmd string and split.
        args = shlex.split(cmd)

        proc = subprocess.Popen(
            args,
            cwd=str(cwd),
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        timed_out = False
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            stdout, stderr = self._terminate_and_collect(proc, kill_after_s=2.0)
            exit_code = proc.returncode

        duration_s = time.time() - start
        oom = looks_like_oom(stderr)
        ok = (not timed_out) and (exit_code == 0) and (not oom)

        return SubprocessOutcome(
            ok=ok,
            exit_code=exit_code,
            timed_out=timed_out,
            oom=oom,
            stdout=stdout,
            stderr=stderr,
            duration_s=duration_s,
        )

    def _terminate_and_collect(
        self, proc: subprocess.Popen[str], kill_after_s: float
    ) -> tuple[str, str]:
        """Terminate a process group, then kill if needed."""

        with suppress(Exception):
            proc.send_signal(signal.SIGTERM)

        try:
            return proc.communicate(timeout=kill_after_s)
        except subprocess.TimeoutExpired:
            with suppress(Exception):
                proc.kill()
            try:
                return proc.communicate(timeout=kill_after_s)
            except Exception:
                return ("", "")
