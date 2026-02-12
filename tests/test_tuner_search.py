from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from autovram.core.detect import detect_system
from autovram.core.runner import SubprocessOutcome, SubprocessRunner
from autovram.core.tuner import autotune_batch_size
from autovram.core.types import AutoVRAMConfig, RunContext


class FakeRunner(SubprocessRunner):
    def __init__(self, *, oom_at: int) -> None:
        self.oom_at = oom_at

    def run(self, cmd: str, env: dict[str, str], timeout_s: float, cwd: Path) -> SubprocessOutcome:  # type: ignore[override]
        bs = int(env["AUTOVRAM_BATCH_SIZE"])
        if bs >= self.oom_at:
            return SubprocessOutcome(
                ok=False,
                exit_code=1,
                timed_out=False,
                oom=True,
                stdout="",
                stderr="CUDA out of memory",
                duration_s=0.01,
            )
        # Metric scales with batch size.
        metric = 10.0 * bs
        stdout = f"AUTOVRAM_METRIC it_per_s={metric}\n"
        return SubprocessOutcome(
            ok=True,
            exit_code=0,
            timed_out=False,
            oom=False,
            stdout=stdout,
            stderr="",
            duration_s=0.01,
        )


def test_autotune_batch_size_binary_search(tmp_path: Path) -> None:
    system = detect_system()
    ctx = RunContext(
        mode="script",
        system=system,
        metric_name="it_per_s",
        timeout_s=1.0,
        work_dir=tmp_path,
        exec_cwd=tmp_path,
        engine_name="heuristic",
    )

    runner = FakeRunner(oom_at=4)
    base = AutoVRAMConfig(batch_size=1, micro_batch=1, precision="fp16")

    best, results = autotune_batch_size(
        context=ctx,
        runner=runner,
        cmd="python -c 'print(1)'",
        base_config=base,
        max_trials=10,
    )

    assert best is not None
    assert best.batch_size == 3
    assert any(r.oom for r in results)


def test_autotune_returns_none_if_all_bad(tmp_path: Path) -> None:
    system = detect_system()
    ctx = replace(
        RunContext(
            mode="script",
            system=system,
            metric_name="it_per_s",
            timeout_s=1.0,
            work_dir=tmp_path,
            exec_cwd=tmp_path,
            engine_name="heuristic",
        )
    )

    runner = FakeRunner(oom_at=1)
    base = AutoVRAMConfig(batch_size=1, micro_batch=1, precision="fp16")

    best, _ = autotune_batch_size(
        context=ctx, runner=runner, cmd="x", base_config=base, max_trials=5
    )
    assert best is None
