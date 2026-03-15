# Changelog

All notable changes to `loqculate` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.2.2] — 2026-03-15

### Added

- **`loqculate.compat` subpackage** — verbatim API ports of the original Pino 2020 scripts:
    - `OriginalWLS` (`loqculate/compat/wls.py`): ports `old/calculate-loq.py`
    `process_peptide(model='piecewise')` exactly, using legacy initialisation
    (slope from top-two concentration points), scipy TRF solver with the same
    bounds as lmfit, the original LOD formula (flat noise floor), and single-point
    bootstrap LOQ rule (no sliding window).  Bootstrap seeding uses
    `SeedSequence(i)` for `i in range(n_boot)` — identical to the original script.
    - `OriginalCV` (`loqculate/compat/cv.py`): ports `old/loq_by_cv.py`
    `calculate_LOQ_byCV()` exactly, using per-concentration Bessel-corrected CV and
    the non-strict `≤` threshold (original uses `<= 0.2`, not `< 0.2`).
    - Both classes implement the `CalibrationModel` ABC and are registered in
    `MODEL_REGISTRY` as `'original_wls'` and `'original_cv'`.
- **`loqculate.utils.cv`** — extracted `vectorized_cv_stats()` helper shared between
  `EmpiricalCV` and `OriginalCV`.
- **`tests/test_compat.py`** — 37 tests covering interface compliance, MODEL_REGISTRY
  integration, synthetic edge cases, and regression against `bench_real_data.json`
  (LOD within 20%, matching benchmark's own tolerance) and `bench_empirical.json`
  (OriginalCV LOQ exact match).

### Changed

- `loqculate.models.cv_empirical` now imports `vectorized_cv_stats` from
  `loqculate.utils.cv` instead of defining it inline.
- `__version__` bumped from `0.2.0` → `0.2.2`.

### Known Limitations

- `OriginalWLS` uses scipy TRF instead of lmfit Levenberg-Marquardt.  For 2/27
  peptides in the demo dataset, the solvers find different local optima (up to
  29% LOD difference), consistent with `bench_real_data.json` `diverge` entries.
- LOQ from `OriginalWLS` is seeding-exact but solver-approximate: for peptides
  with unstable CV profiles (bouncy LOQ), the scipy TRF bootstrap may produce
  different finite/inf outcomes than lmfit.  LOD regression is robust; LOQ is
  tested at the category level only.

---

## [0.2.0] — 2026-03-12

### Added

- **EmpiricalCV model**: model-free LOQ estimation from raw replicate CVs
- **Sliding-window LOQ rule** (`window` parameter, default=3): requires `window` consecutive
  concentration points below the CV threshold, reducing FDR from ~100% to <5%
- **Bootstrap guard**: prevents infinite loops when all replicates in a bootstrap resample are zero
- **6 input formats**: DIA-NN report, DIA-NN pr_matrix, Spectronaut, Skyline, EncyclopeDIA, generic CSV
- **Parallel processing** via `ProcessPoolExecutor` for whole-proteome runs
- **`loqculate compare`** CLI sub-command for side-by-side model comparison
- **Multiplier file support** for applying dilution/concentration correction factors
- Simulation framework (`loqculate.testing`) for benchmark and unit-test generation
- Full benchmark suite (`benchmarks/`) with frozen JSON results
- `comparison_report.ipynb`: reproducible 12-section validation report
- MIT License
- GitHub Actions CI (Python 3.10 / 3.11 / 3.12)

### Changed

- Optimizer changed from lmfit Levenberg-Marquardt → scipy TRF for PiecewiseWLS
  (same numeric results, no lmfit dependency at runtime for the new path)
- LOQ window=1 now matches original tools exactly (regression-tested)

### Fixed

- Bootstrap infinite-loop on all-zero peptide data

---

## [0.1.0] — 2024 (original scripts)

Original `calculate-loq.py` and `loq_by_cv.py` from Pino 2020.
