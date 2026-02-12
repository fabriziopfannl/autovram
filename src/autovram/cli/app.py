from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import typer
from rich.console import Console
from rich.text import Text

from autovram.core.detect import detect_system, system_summary_lines
from autovram.core.runner import SubprocessRunner
from autovram.core.tuner import autotune_batch_size, ensure_run_dir
from autovram.core.types import AutoVRAMConfig, RunContext
from autovram.engines.base import EngineContext
from autovram.engines.registry import available_engines, resolve_engine
from autovram.io.export import export_config

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


def _autovram_dir() -> Path:
    return Path.cwd() / ".autovram"


@app.command()
def info(json_output: bool = typer.Option(False, "--json", help="Print JSON output")) -> None:
    """Print a compact system summary."""

    system = detect_system()
    if json_output:
        console.print(json.dumps(asdict(system), indent=2))
        raise typer.Exit(0)

    for line in system_summary_lines(system):
        console.print(line)


@app.command()
def doctor() -> None:
    """Check prerequisites and provide actionable guidance."""

    system = detect_system()
    for line in system_summary_lines(system):
        console.print(line)

    console.print("")

    issues: list[str] = []

    if system.torch_version is None:
        issues.append(
            "PyTorch is not installed. Script mode usually requires PyTorch (or your ML stack)."
        )

    if system.compute == "CPU":
        issues.append(
            "No CUDA/MPS/ROCm GPU detected via PyTorch. Tuning will still run, but it may be slow and not GPU-representative."
        )

    if system.compute == "MPS":
        issues.append(
            "Apple MPS detected. Support is best-effort; OOM patterns and VRAM reporting may be limited."
        )

    if system.compute == "ROCm":
        issues.append(
            "ROCm detected (best-effort). autovram is primarily tested on CUDA; please report issues."
        )

    if not issues:
        console.print("[green]OK[/green] No obvious issues detected.")
        raise typer.Exit(0)

    console.print("[yellow]Notes[/yellow]")
    for it in issues:
        console.print(f"- {it}")


@app.command()
def tune(
    cmd: str = typer.Option(..., "--cmd", help="Command to run (quoted)"),
    metric: str = typer.Option("it_per_s", "--metric", help="Metric key to parse from stdout"),
    timeout: float = typer.Option(60.0, "--timeout", help="Timeout per trial (seconds)"),
    max_trials: int = typer.Option(25, "--max-trials", help="Maximum number of trials"),
    engine: str = typer.Option(
        "heuristic", "--engine", help=f"Engine: {', '.join(available_engines())}"
    ),
) -> None:
    """Tune an ML script by running it multiple times with different env vars."""

    system = detect_system()
    for line in system_summary_lines(system):
        console.print(line)

    console.print("")

    try:
        eng = resolve_engine(engine)
    except Exception as e:
        console.print(f"[red]Engine error[/red]: {e}")
        raise typer.Exit(2) from None

    av_dir = _autovram_dir()
    run_dir = ensure_run_dir(av_dir)

    console.print(Text(f"Tuning (engine={eng.name}, metric={metric})", style="bold"))
    console.print("────────────────────────────────────────")

    runner = SubprocessRunner()
    base_ctx = RunContext(
        mode="script",
        system=system,
        metric_name=metric,
        timeout_s=timeout,
        work_dir=run_dir,
        exec_cwd=Path.cwd(),
        engine_name=eng.name,
    )

    # MVP: we tune batch size per precision candidate proposed by the engine.
    best_config: AutoVRAMConfig | None = None
    best_score: float = float("-inf")

    configs = list(eng.propose_configs(EngineContext(run=base_ctx)))
    if not configs:
        console.print("[red]Engine produced no candidate configs.[/red]")
        raise typer.Exit(2)

    for cfg0 in configs:
        base_config = cfg0
        cfg, results = autotune_batch_size(
            context=base_ctx,
            runner=runner,
            cmd=cmd,
            base_config=base_config,
            max_trials=max_trials,
        )

        # Print trial lines compactly
        for idx, rr in enumerate(results, start=1):
            status = "OK" if rr.ok else ("OOM" if rr.oom else "BAD")
            metric_str = f"{rr.metric_value:.3f}" if rr.metric_value is not None else "-"
            console.print(
                f"Trial {idx:<2d} cfg=batch_size={rr.config.batch_size} precision={rr.config.precision} → {status}  {metric}={metric_str}"
            )

        if cfg is None:
            continue

        # Pick the best from this precision based on last known best trial score.
        # (The tuner already tracked best per run; we re-evaluate by rerunning score on cached results.)
        for rr in results:
            if rr.ok:
                s = eng.score_result(rr)
                if s > best_score:
                    best_score = s
                    best_config = rr.config

    if best_config is None:
        console.print("[red]No stable configuration found.[/red]")
        console.print(f"See run directory: {run_dir}")
        raise typer.Exit(1)

    # Write default export
    cfg_path = av_dir / "config.json"
    export_config(best_config, fmt="json", out_path=cfg_path)

    console.print("")
    console.print(Text("Best stable config", style="bold"))
    console.print("──────────────────")
    console.print(json.dumps(asdict(best_config), indent=2))
    console.print("")
    console.print(f"Exported: {cfg_path}")
    console.print(f"Run directory: {run_dir}")


@app.command("tune-vllm")
def tune_vllm(
    model: str = typer.Option(..., "--model", help="Model name/path"),
    max_model_len: int = typer.Option(4096, "--max-model-len", help="Max model length"),
    timeout: float = typer.Option(120.0, "--timeout", help="Timeout per trial (seconds)"),
    engine: str = typer.Option("vllm", "--engine", help="Engine to use (default: vllm)"),
) -> None:
    """Tune common vLLM flags (best-effort MVP).

    This command is intentionally conservative in v1. If vLLM is not installed,
    it prints guidance and exits.
    """

    system = detect_system()
    for line in system_summary_lines(system):
        console.print(line)

    try:
        import vllm  # noqa: F401
    except Exception:
        console.print("")
        console.print(
            "[yellow]vLLM is not installed.[/yellow] Install optional dependency: pip install 'autovram[vllm]'"
        )
        raise typer.Exit(2) from None

    console.print("")
    console.print(
        "This MVP does not yet run a full vLLM benchmark loop offline. It will export a conservative starter config."
    )

    # Conservative starter config export.
    cfg = AutoVRAMConfig(
        dtype="float16",
        gpu_memory_utilization=0.90,
        max_num_seqs=16,
        max_num_batched_tokens=max_model_len * 16,
    )

    av_dir = _autovram_dir()
    cfg_path = av_dir / "config.json"
    export_config(cfg, fmt="json", out_path=cfg_path)

    console.print("")
    console.print(Text("Exported vLLM starter config", style="bold"))
    console.print(json.dumps(asdict(cfg), indent=2))
    console.print(f"Exported: {cfg_path}")


@app.command()
def export(
    fmt: str = typer.Option("json", "--format", help="json|yaml|dotenv"),
    out: str = typer.Option(".autovram/config.json", "--out", help="Output path"),
) -> None:
    """Export the last config from .autovram/config.json to another format."""

    av_dir = _autovram_dir()
    cfg_in = av_dir / "config.json"
    if not cfg_in.exists():
        console.print("[red]No config found.[/red] Run `autovram tune ...` first.")
        raise typer.Exit(1)

    raw = json.loads(cfg_in.read_text(encoding="utf-8"))
    cfg = AutoVRAMConfig(**raw)

    out_path = Path(out)

    fmt_norm = fmt.strip().lower()
    if fmt_norm not in ("json", "yaml", "dotenv"):
        console.print("[red]Invalid format[/red]. Use: json|yaml|dotenv")
        raise typer.Exit(2)

    try:
        export_config(cfg, fmt=fmt_norm, out_path=out_path)  # type: ignore[arg-type]
    except Exception as e:
        console.print(f"[red]Export failed[/red]: {e}")
        raise typer.Exit(2) from None

    console.print(f"Exported: {out_path}")
