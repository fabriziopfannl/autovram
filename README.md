# autovram — GPU Auto-Optimizer

Stop guessing batch size and precision. **autovram** automatically searches for the **fastest stable** configuration (batch size, microbatch, precision, accumulation, vLLM knobs) for your machine and exports it as a reusable config.

- Works **offline** (no telemetry, no paid APIs)
- macOS/Linux first, Windows best-effort
- NVIDIA CUDA supported; Apple MPS / AMD ROCm best-effort detection + helpful messaging
- CLI + library API

> Status: MVP (alpha). Contributions welcome.

---

## Quickstart

### Install

Recommended (isolated):

```bash
pipx install autovram
```

Editable (for development):

```bash
git clone https://github.com/your-org/autovram
cd autovram
pipx install -e .
```

### 1) Inspect your system

```bash
autovram info
```

Expected output (example):

```text
System summary
──────────────
OS: Linux (x86_64)
Python: 3.11.7
PyTorch: 2.4.0
Compute: CUDA
GPU: NVIDIA RTX 4090
VRAM: 24564 MiB
```

### 2) Tune an ML script (script mode)

Run your script multiple times while autovram changes environment variables.

```bash
autovram tune --cmd "python examples/torch_train_demo.py" --timeout 60
```

Example output (GIF-friendly):

```text
System summary
──────────────
OS: Linux (x86_64)
Python: 3.11.7
PyTorch: 2.4.0
Compute: CUDA
GPU: NVIDIA RTX 4090
VRAM: 24564 MiB

Tuning (engine=heuristic, metric=it_per_s)
────────────────────────────────────────
Trial 1  cfg=batch_size=1  precision=fp16  → OK  it_per_s=21.4
Trial 2  cfg=batch_size=2  precision=fp16  → OK  it_per_s=39.7
Trial 3  cfg=batch_size=4  precision=fp16  → OOM
Binary search between 2 and 4...
Trial 4  cfg=batch_size=3  precision=fp16  → OK  it_per_s=55.2

Best stable config
──────────────────
{
  "batch_size": 3,
  "micro_batch": 3,
  "grad_accum": 1,
  "precision": "fp16"
}

Exported: .autovram/config.json
Run directory: .autovram/runs/2026-02-12_15-56-02
```

### 3) Export the config

```bash
autovram export --format dotenv --out .env.autovram
```

---

## Why this exists

VRAM tuning is still mostly folklore:

- “Try batch size 8… now 16… oops OOM.”
- “Maybe bf16 is faster… unless it isn’t.”
- “vLLM knobs are confusing and GPU-specific.”

**autovram** makes it mechanical: run a short benchmark loop, push to the edge of OOM safely, and keep the best throughput.

---

## How it works (high-level)

For a primary knob (e.g. `batch_size`):

1. Start conservative.
2. Exponentially increase until instability/OOM.
3. Binary search between last-good and first-bad.
4. For each candidate: run a short benchmark window and parse a metric.
5. Save all trials to a local run directory; export the best configuration.

OOM/stability detection is based on:

- non-zero exit code
- timeout/hang
- stderr OOM patterns (CUDA OOM, cuBLAS alloc failures, etc.)

---

## Integrating with your script

In **script mode**, autovram communicates through environment variables.

### Minimal integration

```python
from autovram.runtime import get_runtime_config, print_metric

cfg = get_runtime_config()  # reads env vars with safe defaults

# Use cfg.batch_size, cfg.precision, etc.
# ... run a tiny benchmark window ...

print_metric(it_per_s=12.34)
```

Your script should print metric lines like:

```text
AUTOVRAM_METRIC it_per_s=12.34
```

See: `examples/torch_train_demo.py`.

---

## Engines (plugins)

autovram keeps the core minimal and supports optional engines.

Built-in:

- `heuristic` (default): conservative heuristics + safe search loop
- `vllm`: vLLM-oriented knobs (requires `vllm` installed to actually run)

Optional:

- `llm-autobatch`: integrates <https://github.com/fabriziopfannl/llm-autobatch>

Install the optional engine:

```bash
pip install "autovram[llm-autobatch]"
```

---

## Roadmap

- [x] Script mode autotuning (batch size + precision)
- [x] Offline, no telemetry
- [x] Config export (json / dotenv, yaml optional)
- [x] Engine abstraction + optional llm-autobatch
- [ ] Multi-GPU support (data parallel)
- [ ] Better vLLM benchmarking (client-driven QPS)
- [ ] Torch compile / cudagraphs tuning
- [ ] More robust VRAM headroom estimation

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

Apache-2.0. See [LICENSE](LICENSE).
