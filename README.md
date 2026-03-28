<!-- markdownlint-disable MD033 MD041 -->
<div align="center">
  <img src="assets/loqculate-readme-header.svg" alt="loqculate logo" width="420" />
  <br />
  LOD &amp; LOQ calculator for mass-spectrometry calibration curves.
  <br />
  Current version: <strong>v0.3.0</strong>. New default model: PiecewiseCF (closed-form knot search, with speedup over PiecewiseWLS).
  <br />
  <br />
  <a href="https://github.com/eneskemalergin/loqculate/actions/workflows/ci.yml"><img src="https://img.shields.io/badge/CI-passing-22c55e?style=for-the-badge" alt="CI" /></a>
  <a href="#installation"><img src="https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-3776AB?style=for-the-badge" alt="Python versions" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-f59e0b?style=for-the-badge" alt="License: MIT" /></a>
  <a href="https://codecov.io/gh/eneskemalergin/loqculate"><img src="https://img.shields.io/badge/coverage-tracked-6366f1?style=for-the-badge" alt="codecov" /></a>
</div>
<!-- markdownlint-enable MD041 -->

---

`loqculate` v0.3.0 reimplements and extends the Pino 2020 LOD/LOQ scripts:

- **Closed-form knot search** (`PiecewiseCF`): fits $y = \max(c, ax + b)$ by exhaustively searching all interior concentration values as candidate partition boundaries. No optimizer, no convergence failures, no initial-guess sensitivity. 15.8× full-pipeline speedup over `PiecewiseWLS`. Default since v0.3.0.
- A **sliding-window LOQ rule** that reduces FDR from ~100% to <5%: LOQ is declared at $C_i$ only when $\mathrm{CV}(C_i)$, $\mathrm{CV}(C_{i+1})$, and $\mathrm{CV}(C_{i+2})$ all fall below `cv_thresh`. Three consecutive passing concentrations, not one.
- **11.4× lower memory** per peptide vs the original scripts (VmRSS, 1 000 peptide batch). Batch throughput is 3.3× faster at 10 000 peptides due to reduced GC pressure (measured in `comparison_report.ipynb` against `OriginalWLS`)
- **EmpiricalCV** as a model-free alternative to PiecewiseWLS. Fast, but FDR is higher at low replicate counts (n < 5).
- **`loqculate.compat`**: drop-in `OriginalWLS` and `OriginalCV` wrappers that reproduce paper results without the legacy scripts. Registered in `MODEL_REGISTRY` and usable via `--model original_wls`.
- **Bootstrap guard** preventing infinite loops on zero-signal replicates
- Support for **6 MS input formats**: DIA-NN report, DIA-NN pr_matrix, Spectronaut, Skyline, EncyclopeDIA, and generic CSV
- Parallel processing via `ProcessPoolExecutor` for whole-proteome throughput

---

## Installation

Install from source (not yet on PyPI):

```bash
git clone https://github.com/eneskemalergin/loqculate
cd loqculate
pip install -e ".[dev]"
```

Requires Python ≥ 3.10.

---

## Quick start

### CLI

```bash
# Fit with default PiecewiseCF model (default since v0.3.0)
loqculate fit data.tsv conc_map.csv

# Fit with PiecewiseWLS model
loqculate fit data.tsv conc_map.csv --model piecewise_wls

# Fit with EmpiricalCV model
loqculate fit data.tsv conc_map.csv --model cv_empirical

# Fit with original Pino 2020 WLS (for paper reproducibility)
loqculate fit data.tsv conc_map.csv --model original_wls

# Fit with original Pino 2020 CV method
loqculate fit data.tsv conc_map.csv --model original_cv

# Compare PiecewiseCF and PiecewiseWLS side-by-side
loqculate compare data.tsv conc_map.csv --models piecewise_cf,piecewise_wls
```

`conc_map.csv` maps each raw filename to its calibration concentration:

```csv
filename,concentration
sample_1ng_rep1.raw,1.0
sample_1ng_rep2.raw,1.0
sample_5ng_rep1.raw,5.0
```

### Python API

```python
from loqculate import read_calibration_data, PiecewiseCF, PiecewiseWLS, EmpiricalCV
from loqculate.compat import OriginalWLS, OriginalCV

data = read_calibration_data("data.tsv", "conc_map.csv")

for peptide, x, y in data.iter_peptides():
    result = PiecewiseCF().fit(x, y)  # default since v0.3.0
    print(peptide, result.loq(), result.lod())

# TRF optimizer path
for peptide, x, y in data.iter_peptides():
    result = PiecewiseWLS().fit(x, y)
    print(peptide, result.loq(), result.lod())

# Reproduce original Pino 2020 results exactly
for peptide, x, y in data.iter_peptides():
    m = OriginalWLS().fit(x, y)
    print(peptide, m.lod(), m.loq())

# Or via MODEL_REGISTRY
from loqculate import MODEL_REGISTRY
ModelClass = MODEL_REGISTRY['original_wls']
```

---

## Supported input formats

| Format           | Auto-detected by                                   |           Needs `conc_map`?           |
| ---------------- | -------------------------------------------------- | :-----------------------------------: |
| DIA-NN report    | `File.Name` + `Precursor.Id` columns               |                  Yes                  |
| DIA-NN pr_matrix | `.pr_matrix.tsv` extension                         |                  Yes                  |
| Spectronaut      | `R.FileName` + `PG.ProteinGroups` columns          |                  Yes                  |
| Skyline          | `Peptide Sequence` + `Total Area Fragment` columns |                  Yes                  |
| EncyclopeDIA     | `numFragments` column                              |                  Yes                  |
| Generic CSV      | `peptide`, `concentration`, `area` columns         | No (concentration column is embedded) |

**`conc_map.csv` rules:**

- Two columns: `filename` (basename, no path) and `concentration` (numeric, same units as your calibration range).
- Filenames that appear in the data but are missing from the map are dropped with a warning. Check your logs if peptide counts appear low.
- All 6 formats except Generic CSV require a map; passing `None` raises a `ValueError` at load time.

---

## Models

### PiecewiseCF (default since v0.3.0)

Fits a piecewise linear model (noise floor + linear signal) using closed-form weighted least squares with a discrete knot search. Enumerates all interior unique concentration values as candidate partition boundaries and picks the one that minimises total weighted RSS under the constraint set. No optimizer, no initial-guess sensitivity, no convergence tolerance. Constraints (slope >= 0, noise intercept <= linear intercept) are enforced analytically by clamping after solving the normal equations.

$$y = \max\left(c,\ a x + b\right)$$

$a$: slope. $b$: linear-segment intercept. $c$: noise plateau. $\kappa$: knot (the selected partition boundary). Weights: $w_i = 1/\sqrt{x_i}$; precision weight $W_i = w_i^2 = 1/x_i$. LOQ via bootstrap CV profile with sliding-window rule.

### PiecewiseWLS

Same statistical model as PiecewiseCF. Differs in the solver: uses scipy TRF (`curve_fit`) with explicit parameter bounds and a legacy initial-guess heuristic (slope from top-two concentration points). Retained for comparison and compatibility. For new analyses, PiecewiseCF is faster (15.8× full pipeline speedup) and finds the globally optimal partition.

### EmpiricalCV

Computes CV directly from replicate measurements at each concentration level. LOQ is the lowest concentration where `window` consecutive CVs fall below `cv_thresh`. No parametric assumptions. Faster than WLS, but FDR is higher at low replicate counts (n < 5).

$$\mathrm{LOQ} = \min\{ C_i : \mathrm{CV}(C_i) < \tau,\ \mathrm{CV}(C_{i+1}) < \tau,\ \mathrm{CV}(C_{i+2}) < \tau \}$$

$\tau$ is `cv_thresh` (default 0.20). Requiring three consecutive passing concentrations reduces single-point false discoveries from ~100% to <5%.

### OriginalWLS (`loqculate.compat`)

Verbatim port of `old/calculate-loq.py` `process_peptide(model='piecewise')` from Pino 2020. Reproduces the exact original logic: legacy parameter initialisation (slope from top-two concentration points), scipy TRF solver with the same bounds as the original lmfit call, original LOD formula (flat noise floor + $n \cdot \sigma$), and **single-point bootstrap LOQ rule** (no sliding window). Bootstrap seeds are `SeedSequence(i)` for `i in range(n_boot)`, identical to the original script.

Use when you need byte-for-byte reproducibility of published results. For new analyses, prefer `PiecewiseCF` (15.8× faster full pipeline, globally optimal partitioning).

### OriginalCV (`loqculate.compat`)

Verbatim port of `old/loq_by_cv.py` `calculate_LOQ_byCV()`. LOQ is the lowest positive concentration where CV ≤ `cv_thresh` (non-strict `≤`, matching the original `<= 0.2` check). No LOD, no sliding window, no parametric model. Deterministic and near-instant. Matches EmpiricalCV's speed but FDR is higher due to the single-point rule (64-99% on null curves).

---

## Configuration & defaults

| Parameter   | Default        | Description                                                                               |
| ----------- | -------------- | ----------------------------------------------------------------------------------------- |
| `cv_thresh` | `0.20`         | CV threshold; LOQ requires CV < this value                                                |
| `window`    | `3`            | Consecutive passing concentration points required                                         |
| `n_boot`    | `100`          | Bootstrap resamples for LOD/LOQ CI estimation                                             |
| `std_mult`  | `2`            | Noise-floor multiplier for LOD ($\mathrm{LOD} = \alpha + 2\sigma$)                        |
| `model`     | `piecewise_cf` | Active model; also accepts `piecewise_wls`, `cv_empirical`, `original_wls`, `original_cv` |

Pass these as CLI flags (`--cv_thresh 0.15`) or keyword arguments to `.fit()`.

---

## Benchmark summary

Machine: AMD Ryzen 9 3950X, 32 threads. FDR measured on 300 simulated null curves across 4-14 concentration levels (window=3). Throughput measured on real DIA-NN data.

| Metric                                      | PiecewiseCF (v0.3.0) | PiecewiseWLS | EmpiricalCV | OriginalWLS | OriginalCV |
| ------------------------------------------- | :------------------: | :----------: | :---------: | :---------: | :--------: |
| Per-fit, single-threaded                    |       ~0.73 ms       |   ~1.96 ms   |  ~0.02 ms   |   ~1.8 ms   |  ~0.03 ms  |
| Full pipeline vs WLS (n_boot=200)           |   **15.8× faster**   |   baseline   |     N/A     |     N/A     |    N/A     |
| Null FDR, window=3 (range over n_conc 4-14) |         0-2%         |     0-2%     |    2-18%    |    1-2%     |   64-99%   |
| Memory per peptide (VmRSS)                  |       0.013 MB       |   0.013 MB   |     ~0      |  0.151 MB   |     ~0     |

PiecewiseCF replaces the TRF optimizer with a discrete knot search (exhaustive enumeration of interior candidate boundaries). On 27-peptide reference data: 0 convergence failures, globally optimal partition on 13/18 cases where models disagree, and one additional finite-LOD rescue vs PiecewiseWLS. OriginalCV FDR reaches 64-99% due to the single-point rule. Raw benchmark results are in `tmp/results/` (JSON); reproduction commands are in `benchmarks/`.

---

## Design decisions & validation

[`comparison_report.ipynb`](comparison_report.ipynb) is a frozen 12-section validation notebook comparing the **original Pino 2020 scripts** (`OriginalWLS`, `OriginalCV`) to the v0.2.x `loqculate` models (`PiecewiseWLS`, `EmpiricalCV`). It covers FDR and TPR across concentration-level density (4-14 levels) and replicate count (2-20), LOQ accuracy and precision benchmarks, and per-fit speed and throughput scaling up to 10,000 peptides. It does not cover `PiecewiseCF` or any v0.3.0 changes. A systematic comparison of PiecewiseCF against PiecewiseWLS is tracked in `benchmarks/bench_knot_vs_curvefit.py` and `benchmarks/bench_vectorized_boot.py`.

---

## Development

```bash
pip install -e ".[dev]"
pre-commit install      # install git hooks (runs ruff on every commit)
pytest                 # 165 passed, 42 skipped
ruff check loqculate/  # lint
```

Run benchmarks (requires ~5 min, tested with 24 threads):

```bash
python benchmarks/bench_simulation.py --save tmp/results/bench_simulation.json
python benchmarks/bench_n_concentrations.py --n_workers 24 --save tmp/results/bench_n_concentrations.json
python benchmarks/bench_n_replicates.py --n_workers 24 --save tmp/results/bench_n_replicates.json
python benchmarks/bench_knot_vs_curvefit.py --save tmp/results/bench_knot_vs_curvefit.json
python benchmarks/bench_vectorized_boot.py --save tmp/results/bench_vectorized_boot.json
```

---

## Roadmap

A guiding constraint: **loqculate is a practical proteomics tool, not a statistics library**. Each addition must justify its complexity against the realistic needs of a DIA/PRM lab.

### Phase 1: Solidify core (v0.2.x - v0.5.0)

Same piecewise/CV framework, no new model classes. Ends with a pip release after a bug-finding period on real datasets.

- [x] **v0.2.2** ✓: `loqculate.compat` (`OriginalWLS`, `OriginalCV`) wrappers; paper results reproducible without legacy scripts
- [x] **v0.3.0** ✓: closed-form piecewise solver (`PiecewiseCF`). Discrete knot search, no optimizer, 15.8× full-pipeline speedup vs PiecewiseWLS; formalizes $w_i$ vs $W_i$ weight convention; `PiecewiseCF` is new default
- [ ] **v0.4.0**: delta-method uncertainty (`--fast` flag, double-dash). Analytical LOQ CIs in $O(1)$; smooth models only. Falls back to bootstrap near the piecewise kink (non-differentiable $\max$)
- [ ] **v0.5.0**: segmented regression (`SegmentedWLS`). Continuous piecewise linear with estimated breakpoint $\psi$, AIC/BIC vs flat-noise piecewise
- [ ] **pip release**: after real-data bug-finding marathon. Semantic versioning locked, public API stable for Phase 2

### Phase 2: New models + UX (v0.6.0 - v1.0.0)

New curve families, model selection, performance, and publication-ready output.

- [ ] **v0.6.0**: 4PL/5PL model (`FourPL`): $y = d + (a-d)/[1+(x/c)^b]$; LLOQ + ULOQ; FDA bioanalytical guidance compliant; fitted via `scipy.optimize.curve_fit` (non-linear, unlike the current piecewise linear approach); auto-downgrades to piecewise when upper asymptote is unidentifiable
- [ ] **v0.7.0**: model selection (`loqculate.selection`): AIC/BIC + approximate LOO-CV (PSIS-LOO, fast) per peptide; **auto-recommend** with explicit graceful degradation when models fail to converge or AIC penalty blows up on sparse curves
- [ ] **v0.8.0**: performance: Numba `@njit` on closed-form solver + bootstrap (~3-10×); chunked parquet I/O (`pyarrow`); shared-memory multiprocessing for 100k+ peptide datasets; adaptive bootstrap early-stopping co-designed with Numba cost model
- [ ] **v1.0.0**: stable public API contract. Expanded plotting API (publication-ready diagnostics, waterfall, model-agreement scatter, yield curve); exact LOO-CV and WAIC (requires Bayesian models, deferred to v1.x)

### Phase 3: Optional extensions (v1.x+)

Opt-in, not bundled in core.

- [ ] **Bayesian piecewise** (opt-in extra): MCMC / variational inference via `numpyro` or `pymc`; LOQ from posterior predictive. *Scope-limited to targeted panels (≤100 peptides). HMC alone is insufficient for the discrete knot parameter; requires marginalization.*
- [ ] **`loqculate-rs`** (separate package, if needed): Rust bindings for batch computation on very large datasets (500k+ peptides); only warranted if Numba saturation is demonstrated

---

## Citation

If you use `loqculate` in your work, please cite both the original method paper and this software:

**Original method (Pino 2020):**

> Pino, L.K. et al. (2020). *Matrix-matched calibration curves for assay characterization in data-independent acquisition proteomics.* Analytical Chemistry. <https://doi.org/10.1021/acs.analchem.9b04826>

**This software:**

```bibtex
@software{ergin_loqculate_2026,
  author    = {Ergin, Enes Kemal},
  title     = {loqculate: Limit of Detection and Quantitation calculator
               for mass-spectrometry calibration curves},
  year      = {2026},
  version   = {0.3.0},
  url       = {https://github.com/eneskemalergin/loqculate},
  license   = {MIT},
}
```

---

## License

MIT. See [LICENSE](LICENSE).

---

<div align="center">

*Three points hold the line,*  
*Through the sliding window's gaze,*  
*Noise becomes the truth.*

</div>
