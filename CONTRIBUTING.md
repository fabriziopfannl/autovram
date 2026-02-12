# Contributing to autovram

Thanks for helping!

## Development setup

```bash
git clone https://github.com/your-org/autovram
cd autovram
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -U pytest ruff pre-commit mypy
pre-commit install
```

## Running checks

```bash
ruff check .
ruff format --check .
pytest
mypy src/autovram
```

## Pull requests

- Keep changes focused and small.
- Add or update tests.
- Avoid adding hard dependencies.
- Prefer clear error messages and docstrings.

## Reporting issues

Please include:

- OS + Python version
- GPU model + VRAM
- CUDA driver/runtime versions (if available)
- The autovram command you ran
- The run directory contents (`.autovram/runs/...`)
