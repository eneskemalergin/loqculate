"""Shared path constants and module loaders for the cross-version benchmark suite.

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
    ├── v1/
    │   ├── calculate-loq.py
    │   └── loq_by_cv.py
    └── v2/
        └── loqculate/   ← the v2 package
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

V1_DIR = ROOT / "v1"
V2_DIR = ROOT / "v2"


def _ensure_v2_on_path() -> None:
    """Insert v2/ into sys.path so that `import loqculate` resolves to v2."""
    v2 = str(V2_DIR)
    if v2 not in sys.path:
        sys.path.insert(0, v2)


def load_v1_calc() -> Any:
    """Import v1/calculate-loq.py as a module without altering the file.

    Returns the module object; call its functions directly, e.g.::

        v1 = load_v1_calc()
        df = v1.read_input(str(DEMO_DATA), str(DEMO_MAP))
        row = v1.process_peptide(100, 0.2, '/tmp', pep, 'n', 2, 2, 1, subset, 'n', 'piecewise')
    """
    path = V1_DIR / "calculate-loq.py"
    spec = importlib.util.spec_from_file_location("v1_calc", path)
    mod  = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)                   # type: ignore[union-attr]
    return mod


def load_v1_cv() -> Any:
    """Import v1/loq_by_cv.py as a module."""
    path = V1_DIR / "loq_by_cv.py"
    spec = importlib.util.spec_from_file_location("v1_cv", path)
    mod  = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)                   # type: ignore[union-attr]
    return mod


# Make v2 importable as soon as this helper is imported
_ensure_v2_on_path()
