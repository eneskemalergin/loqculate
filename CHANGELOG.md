# Changelog

All notable changes to `loqculate` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

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
