"""Shared path constants and module loaders for the benchmark suite.

Repository layout assumed:
    loqculate/
    ├── benchmarks/       ← this directory
    ├── data/
    │   ├── demo/
    │   │   ├── one_protein.csv
    │   │   ├── filename2samplegroup_map.csv
    │   │   └── multiplier_file.csv
    │   └── full/
    │       ├── dataSearchedTogether1ng-PinoFormat.csv
    │       └── metadata_dataSearchedTogether1ng-PinoFormat.csv
    ├── old/              ← original Pino implementation (reference)
    │   ├── calculate-loq.py
    │   └── loq_by_cv.py
    ├── loqculate/        ← main package source code
    └── tests/            ← package unit tests
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Root-relative paths
# ---------------------------------------------------------------------------

ROOT    = Path(__file__).parent.parent          # loqculate/
DATA    = ROOT / "data"
DEMO    = DATA / "demo"
FULL    = DATA / "full"

DEMO_DATA = DEMO / "one_protein.csv"
DEMO_MAP  = DEMO / "filename2samplegroup_map.csv"
DEMO_MULT = DEMO / "multiplier_file.csv"

FULL_DATA = FULL / "dataSearchedTogether1ng-PinoFormat.csv"
FULL_MAP  = FULL / "metadata_dataSearchedTogether1ng-PinoFormat.csv"

OLD_DIR = ROOT / "old"


def _ensure_package_on_path() -> None:
    """Insert the repo root into sys.path so that `import loqculate` resolves correctly."""
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def load_original_calc() -> Any:
    """Import old/calculate-loq.py as a module without altering the file.

    Returns the module object; call its functions directly, e.g.::

        orig = load_original_calc()
        df = orig.read_input(str(DEMO_DATA), str(DEMO_MAP))
        row = orig.process_peptide(100, 0.2, '/tmp', pep, 'n', 2, 2, 1, subset, 'n', 'piecewise')
    """
    path = OLD_DIR / "calculate-loq.py"
    spec = importlib.util.spec_from_file_location("original_calc", path)
    mod  = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)                   # type: ignore[union-attr]
    return mod


def load_original_cv() -> Any:
    """Import old/loq_by_cv.py as a module."""
    path = OLD_DIR / "loq_by_cv.py"
    spec = importlib.util.spec_from_file_location("original_cv", path)
    mod  = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)                   # type: ignore[union-attr]
    return mod


# Make loqculate importable as soon as this helper is imported
_ensure_package_on_path()


# ---------------------------------------------------------------------------
# JSON serialisation helper (shared by all benchmark scripts)
# ---------------------------------------------------------------------------

def _json_safe(obj):
    """Recursively convert *obj* to a JSON-serialisable form.

    * numpy integers / floats are cast to Python int / float.
    * Non-finite floats (inf, nan) — both Python and numpy — become ``None``.
    * numpy arrays are converted via ``tolist()``.
    * dicts and lists are processed element-wise.
    """
    import math
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    try:
        import numpy as _np
        if isinstance(obj, _np.integer):
            return int(obj)
        if isinstance(obj, _np.floating):
            return None if not _np.isfinite(obj) else float(obj)
        if isinstance(obj, _np.ndarray):
            return _json_safe(obj.tolist())
    except ImportError:
        pass
    return obj


# ---------------------------------------------------------------------------
# Timing statistics helpers (shared by bench_scale, bench_real_data, …)
# ---------------------------------------------------------------------------

def _ci95(runs) -> float:
    """95% CI half-width: 2 × SEM.  Returns 0.0 for a single observation."""
    import numpy as _np
    arr = list(runs)
    if len(arr) < 2:
        return 0.0
    return 2.0 * float(_np.std(arr, ddof=1)) / (len(arr) ** 0.5)


def _timing_stats(runs) -> dict:
    """Return a dict of wall-time statistics from a list of floats (seconds).

    Keys: ``runs``, ``mean``, ``std``, ``ci95``.
    """
    import numpy as _np
    arr = list(runs)
    mean = float(_np.mean(arr))
    std  = float(_np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return {'runs': arr, 'mean': mean, 'std': std, 'ci95': _ci95(arr)}


# ---------------------------------------------------------------------------
# Resident-set memory helper (Linux /proc, falls back to None on other OS)
# ---------------------------------------------------------------------------

def _rss_mb() -> float:
    """Current process RSS in MB from /proc/self/status (Linux).

    Unlike tracemalloc, this includes C-extension allocations (NumPy, scipy).
    Returns float('nan') if /proc is unavailable (macOS, Windows).
    """
    try:
        with open('/proc/self/status') as _f:
            for _line in _f:
                if _line.startswith('VmRSS:'):
                    return int(_line.split()[1]) / 1024.0   # kB → MB
    except OSError:
        pass
    return float('nan')


# ---------------------------------------------------------------------------
# Shared LOQ-rule registry used by bench_simulation and bench_window_rules
# ---------------------------------------------------------------------------

# Each entry: display_name → window_size fed to find_loq_threshold
RULES: dict = {
    'window=1 (liberal)': 1,
    'window=3 (default)': 3,
    'window=5':           5,
}

# Default non-blank concentration grid for simulation benchmarks
# (same grid used in bench_simulation and bench_window_rules)
SIM_CONCS: list = [
    0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0,
    100.0, 200.0, 500.0, 1000.0, 2000.0,
]
