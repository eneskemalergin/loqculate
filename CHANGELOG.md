# Changelog

<!-- markdownlint-disable MD024 -->

All notable changes to `loqculate` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.4.0] - 2026-07-28

Opt-in delta-method LOQ for `PiecewiseCF`: an analytical prediction-variance CV path beside the existing bootstrap default. Use `loq(method="delta")` / `loq_delta()` in Python, or `loqculate fit --fast` on the CLI (delta first; bootstrap once if delta LOQ is infinite). Default `loq()` and `summary()["loq"]` stay bootstrap. Delta and bootstrap are different estimators and can disagree; `--fast` is CF-only; LOD/knot uncertainty is not propagated into the analytical point estimate.

On the demo 27-peptide set (`PiecewiseCF`, $B=200$, seed 42), measured LOQ-only medians were about $32\,\mathrm{ms}$ (bootstrap) vs $0.16\,\mathrm{ms}$ (delta), roughly $200\times$ on that workload. Full `fit`+LOQ medians were about $34\,\mathrm{ms}$ vs $1.0\,\mathrm{ms}$ (about $35\times$). Those gains come from a vectorized CV grid and a vectorized sliding-window threshold search, not from closed-form LOQ confidence intervals.

### Added

- **Delta-method analytical LOQ** (`loqculate/models/delta_method.py`): prediction-variance CV on the linear segment ($x > \kappa$), kink band around the piecewise join $x^{*}=(c-b)/a$, and LOQ via `find_loq_threshold` (non-strict $\le$).
- **`PiecewiseCF.loq_delta(...)`** and **`PiecewiseCF.loq(..., method="bootstrap"|"delta")`**. Pure delta never falls back to bootstrap.
- **CLI `loqculate fit --fast`** for `piecewise_cf` only; refused for other models.
- **Config** defaults `DEFAULT_DELTA_GRID_POINTS = 1000` and `DEFAULT_KINK_GUARD_FACTOR = 2`.
- **`benchmarks/bench_delta_vs_boot.py`** for wall time and agreement on the demo set.
- **Tests** for delta variance, kink band, small-$n_L$ warnings, cache isolation, CLI `--fast` vs pure API, and threshold edges.
- CI on Linux, macOS, and Windows (x64 and arm64). Full Python 3.10–3.12 on Linux x64; Python 3.12 on the other five runners. Dependabot weekly updates for pip and GitHub Actions.

### Changed

- `__version__` bumped from `0.3.0` to `0.4.0`.
- LOQ cache keys include method as well as CV threshold, so bootstrap and delta cannot overwrite each other.
- Missing weights after fit raise on the delta path; missing covariance or undefined MSE yield infinite analytical LOQ.
- Delta path emits `UserWarning` when $3 \le n_L < 5$; it still computes.
- Some regression float tolerances widened slightly so the same checks hold across OS and CPU architectures.

### Removed

- Unused runtime dependency `numba` (not imported anywhere; no Windows arm64 wheel on PyPI).

### Fixed

- **`PiecewiseCF.covariance()`** returns `None` when $n_{L} < 3$, avoiding division by zero in the residual MSE ($n_{L}-2$).
- **`find_loq_threshold`** returns infinite LOQ for `window < 1` instead of raising inside the window scan.
- Project URLs in `pyproject.toml` point at the current GitHub repository.
- **`PiecewiseWLS.fit()`** treats constant-area curves (including all zeros) as flat: slope 0 and infinite LOD. Avoids a platform-dependent tiny TRF slope that produced a finite LOD on arm64.

---

## [0.3.0] - 2026-04-01

### Added

- **`PiecewiseCF` model** (`loqculate/models/piecewise_cf.py`): closed-form piecewise WLS solver using a discrete knot search. Replaces `scipy.optimize.curve_fit` (TRF) for all default-path fitting. Same statistical model as `PiecewiseWLS` ($y = \max(c, ax + b)$, weights $1/\sqrt{x}$), different solver. Zero convergence failures on the 27-peptide reference dataset; finds the globally optimal partition on 13/18 cases where it differs from TRF.
- **`utils/normal_equations.py`**: scalar `solve_2x2_wls`, batch `solve_2x2_wls_batch`, and `weighted_mean`. Pure NumPy, no state.
- **`utils/knot_search.py`**: `find_knot` (scalar) and `find_knot_batch` (vectorized). Searches all interior unique-concentration candidates, enforces constraints analytically ($\hat{a} \geq 0$; $\hat{c} \geq \hat{b}$), returns `KnotResult` namedtuple.
- **`PiecewiseCF.covariance()`**: returns the 2x2 parameter covariance matrix for the linear segment, computed from the Gram matrix inverse stored during `fit()`. Storage cost is negligible; retained to support delta-method CI in v0.4.0.
- **Vectorized bootstrap path** in `PiecewiseCF`: builds the full resample matrix upfront via `find_knot_batch`. Default path for all fits. Falls back to loop bootstrap with a `ResourceWarning` when the matrix exceeds 100 MB. Agreement with loop path: worst cv_diff = 2.0e-15 across 26 finite-LOD peptides.
- **`config.DEFAULT_MODEL = "piecewise_cf"`**: central constant used by CLI and API.
- **`benchmarks/bench_knot_vs_curvefit.py`** and **`benchmarks/bench_vectorized_boot.py`**: timing and correctness benchmarks for the new solver and bootstrap paths.
- **numba** added as a runtime dependency (incompatible with `@njit` on the closed-form path as-is without refactoring; retained for future use).

### Changed

- **`PiecewiseCF` is the new default model.** `--model piecewise_cf` is the CLI default. All previous outputs used `PiecewiseWLS` (TRF). LOD/LOQ values will differ on the same dataset whenever the two solvers select different partition boundaries (13/27 reference peptides). The CF result is better-fitting (lower RSS) in 13 of those 18 cases. Max relative LOD deviation on same-partition peptides: 3.1e-6 (numerical only).
- `PiecewiseWLS` is preserved. Pass `--model piecewise_wls` to restore the TRF solver path.
- CLI `--models` compare default changed from `piecewise_wls,cv_empirical` to `piecewise_cf,piecewise_wls`.
- `__version__` bumped from `0.2.2` to `0.3.0`.

### Performance

Full-pipeline speedup (vectorized CF vs WLS, n_boot=200): **15.8x**. Single-fit speedup: **2.7x median** (range 1.3-6.3x across 27 peptides). Vectorized bootstrap vs loop bootstrap: **6.8x** at 500 replicates.

### Known Limitations

- Numba `@njit` is not compatible with `solve_2x2_wls` without refactoring (untyped global name; TypingError in nopython mode on Numba 0.64.0). The v0.8.0 Numba path requires a rewrite.
- LOD/LOQ values differ from v0.2.x `PiecewiseWLS` wherever the two solvers disagree on the partition boundary. This is expected behavior, not a regression. Use `--model piecewise_wls` to reproduce v0.2.x results exactly.

---

## [0.2.2] - 2026-03-15

### Added

- **`loqculate.compat` subpackage**: verbatim API ports of the original Pino 2020 scripts:
    - `OriginalWLS` (`loqculate/compat/wls.py`): ports `old/calculate-loq.py` `process_peptide(model='piecewise')` exactly, using legacy initialisation (slope from top-two concentration points), scipy TRF solver with the same bounds as lmfit, the original LOD formula (flat noise floor), and single-point bootstrap LOQ rule (no sliding window).  Bootstrap seeding uses `SeedSequence(i)` for `i in range(n_boot)` identical to the original script.
    - `OriginalCV` (`loqculate/compat/cv.py`): ports `old/loq_by_cv.py` `calculate_LOQ_byCV()` exactly, using per-concentration Bessel-corrected CV and the non-strict `≤` threshold (original uses `<= 0.2`, not `< 0.2`).
    - Both classes implement the `CalibrationModel` ABC and are registered in `MODEL_REGISTRY` as `'original_wls'` and `'original_cv'`.
- **`loqculate.utils.cv`**: extracted `vectorized_cv_stats()` helper shared between `EmpiricalCV` and `OriginalCV`.
- **`tests/test_compat.py`**: 37 tests covering interface compliance, MODEL_REGISTRY integration, synthetic edge cases, and regression against `bench_real_data.json` (LOD within 20%, matching benchmark's own tolerance) and `bench_empirical.json` (OriginalCV LOQ exact match).

### Changed

- `loqculate.models.cv_empirical` now imports `vectorized_cv_stats` from `loqculate.utils.cv` instead of defining it inline.
- `__version__` bumped from `0.2.0` to `0.2.2`.

### Known Limitations

- `OriginalWLS` uses scipy TRF instead of lmfit Levenberg-Marquardt. For 2/27 peptides in the demo dataset, the solvers find different local optima (up to 29% LOD difference), consistent with `bench_real_data.json` `diverge` entries.
- LOQ from `OriginalWLS` is seeding-exact but solver-approximate: for peptides with unstable CV profiles (bouncy LOQ), the scipy TRF bootstrap may produce different finite/inf outcomes than lmfit.  LOD regression is robust; LOQ is tested at the category level only.

---

## [0.2.0] - 2026-03-12

### Added

- **EmpiricalCV model**: model-free LOQ estimation from raw replicate CVs
- **Sliding-window LOQ rule** (`window` parameter, default=3): requires `window` consecutive concentration points below the CV threshold, reducing FDR from ~100% to <5%
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

- Optimizer changed from lmfit Levenberg-Marquardt to scipy TRF for PiecewiseWLS (same numeric results, no lmfit dependency at runtime for the new path)
- LOQ window=1 now matches original tools exactly (regression-tested)

### Fixed

- Bootstrap infinite-loop on all-zero peptide data

---

## [0.1.0] - 2024 (original scripts)

Original `calculate-loq.py` and `loq_by_cv.py` from Pino 2020.
