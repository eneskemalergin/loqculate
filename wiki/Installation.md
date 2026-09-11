# Installation

loqculate v0.4.0 supports Python 3.10-3.12. Local setup is [uv](https://docs.astral.sh/uv/) on 3.12 (`.python-version`). Not on PyPI.

```bash
git clone https://github.com/eneskemalergin/loqculate
cd loqculate
uv sync
uv run loqculate --version
```

That installs the package into `.venv` plus pytest and ruff. Runtime libraries: numpy, scipy, pandas, matplotlib, tqdm.

Need to run `old/calculate-loq.py`? Add lmfit with `uv sync --extra legacy`.

```bash
uv run loqculate fit --help
```

Use `uv run loqculate ...` so you hit this checkout, not some other Python on `PATH`. CI still tests 3.10 and 3.11. `uv sync --reinstall` if the environment looks stale.
