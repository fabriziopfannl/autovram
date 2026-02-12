from __future__ import annotations

import json
import time
from dataclasses import asdict, replace
from pathlib import Path

from .metric import parse_metric
from .runner import SubprocessRunner
from .types import AutoVRAMConfig, RunContext, RunResult


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S")


def ensure_run_dir(base: Path) -> Path:
    run_dir = base / "runs" / _timestamp()
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_trial_artifacts(trial_dir: Path, result: RunResult) -> None:
    trial_dir.mkdir(parents=True, exist_ok=True)

    (trial_dir / "stdout.txt").write_text(result.stdout, encoding="utf-8")
    (trial_dir / "stderr.txt").write_text(result.stderr, encoding="utf-8")

    payload = {
        **asdict(result),
        # Make Paths JSON-friendly
        "artifacts_dir": str(result.artifacts_dir),
    }
    (trial_dir / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_script_trial(
    *,
    context: RunContext,
    runner: SubprocessRunner,
    cmd: str,
    config: AutoVRAMConfig,
    trial_dir: Path,
) -> RunResult:
    env = {
        "AUTOVRAM_BATCH_SIZE": str(config.batch_size),
        "AUTOVRAM_MICRO_BATCH": str(config.micro_batch or config.batch_size),
        "AUTOVRAM_GRAD_ACCUM": str(config.grad_accum),
        "AUTOVRAM_PRECISION": str(config.precision),
    }

    if config.seq_len is not None:
        env["AUTOVRAM_SEQ_LEN"] = str(config.seq_len)

    outcome = runner.run(cmd=cmd, env=env, timeout_s=context.timeout_s, cwd=context.exec_cwd)
    metric_val = parse_metric(outcome.stdout, context.metric_name)

    result = RunResult(
        config=config,
        ok=outcome.ok and (metric_val is not None),
        exit_code=outcome.exit_code,
        timed_out=outcome.timed_out,
        oom=outcome.oom,
        metric_value=metric_val,
        stdout=outcome.stdout,
        stderr=outcome.stderr,
        duration_s=outcome.duration_s,
        artifacts_dir=trial_dir,
        notes=[] if metric_val is not None else ["Metric not found in stdout"],
    )

    write_trial_artifacts(trial_dir, result)
    return result


def autotune_batch_size(
    *,
    context: RunContext,
    runner: SubprocessRunner,
    cmd: str,
    base_config: AutoVRAMConfig,
    min_bs: int = 1,
    max_trials: int = 25,
) -> tuple[AutoVRAMConfig | None, list[RunResult]]:
    """Tune batch size using exponential growth + binary search.

    The tuning objective is to maximize the metric value among stable runs.
    """

    results: list[RunResult] = []

    best: RunResult | None = None

    def trial(cfg: AutoVRAMConfig, idx: int) -> RunResult:
        trial_dir = context.work_dir / "trials" / f"trial_{idx:03d}"
        rr = run_script_trial(
            context=context,
            runner=runner,
            cmd=cmd,
            config=cfg,
            trial_dir=trial_dir,
        )
        results.append(rr)
        nonlocal best
        if (
            rr.ok
            and rr.metric_value is not None
            and (best is None or rr.metric_value > (best.metric_value or float("-inf")))
        ):
            best = rr
        return rr

    # Phase 1: exponential growth until first bad
    last_good_bs: int | None = None
    first_bad_bs: int | None = None

    bs = max(min_bs, 1)
    i = 1
    while i <= max_trials:
        cfg = replace(base_config, batch_size=bs, micro_batch=bs)
        rr = trial(cfg, i)
        if rr.ok:
            last_good_bs = bs
            bs *= 2
            i += 1
            continue
        first_bad_bs = bs
        break

    # If we never found a good config, return None.
    if last_good_bs is None:
        return None, results

    # If we never found a bad config, we already tried max_trials.
    if first_bad_bs is None:
        return best.config if best else None, results

    # Phase 2: binary search between last_good and first_bad
    lo = last_good_bs
    hi = first_bad_bs
    while i <= max_trials and (hi - lo) > 1:
        mid = (lo + hi) // 2
        cfg = replace(base_config, batch_size=mid, micro_batch=mid)
        rr = trial(cfg, i)
        if rr.ok:
            lo = mid
        else:
            hi = mid
        i += 1

    return best.config if best else None, results
