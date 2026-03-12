<div align="center">

<!-- Drop your logo file at docs/logo.png to activate -->
<img src="docs/logo.png" alt="loqculate logo" width="120" />

# loqculate

**Limit of Detection and Quantitation calculator for mass-spectrometry calibration curves.**

<table>
<tr>
  <td><a href="https://github.com/MatrixMatched-Project/loqculate/actions/workflows/ci.yml"><img src="https://github.com/MatrixMatched-Project/loqculate/actions/workflows/ci.yml/badge.svg" alt="CI"></a></td>
  <td><img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue" alt="Python versions"></td>
  <td><a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a></td>
  <td><a href="https://codecov.io/gh/MatrixMatched-Project/loqculate"><img src="https://codecov.io/gh/MatrixMatched-Project/loqculate/branch/main/graph/badge.svg" alt="codecov"></a></td>
</tr>
</table>

</div>

---

`loqculate` re-implements and extends the original Pino 2020 LOD/LOQ tools with:

- A **sliding-window LOQ rule** (window=3 consecutive points, replacing single-point) that reduces FDR from ~100% to <5%
- A second model — **EmpiricalCV** — as a model-free alternative to PiecewiseWLS, very fast but higher FDR at low replicate counts (n < 5), similar to the original `loq_by_cv.py` script.
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
# Fit with default PiecewiseWLS model
loqculate fit data.tsv conc_map.csv

# Fit with EmpiricalCV model
loqculate fit data.tsv conc_map.csv --model cv_empirical

# Compare both models side-by-side
loqculate compare data.tsv conc_map.csv --models piecewise_wls,cv_empirical
```

### Python API

```python
from loqculate import read_calibration_data, PiecewiseWLS, EmpiricalCV

data = read_calibration_data("data.tsv", "conc_map.csv")

for peptide, x, y in data.iter_peptides():
    result = PiecewiseWLS().fit(x, y)
    print(peptide, result.loq, result.lod)
```

---

## Supported input formats

| Format           | Auto-detected by                                   |
| ---------------- | -------------------------------------------------- |
| DIA-NN report    | `File.Name` + `Precursor.Id` columns               |
| DIA-NN pr_matrix | `.pr_matrix.tsv` extension                         |
| Spectronaut      | `R.FileName` + `PG.ProteinGroups` columns          |
| Skyline          | `Peptide Sequence` + `Total Area Fragment` columns |
| EncyclopeDIA     | `numFragments` column                              |
| Generic CSV      | `peptide`, `concentration`, `area` columns         |

---

## Models

### PiecewiseWLS (default)

Fits a piecewise linear model (noise floor + linear signal) using weighted least squares. LOD and LOQ are derived from the fitted parameters. The sliding-window rule requires `window` consecutive concentration points below the CV threshold before declaring a LOQ.

### EmpiricalCV

Computes CV directly from replicate measurements at each concentration level. LOQ is the lowest concentration where `window` consecutive CVs fall below `cv_thresh`. No parametric assumptions — faster than WLS, but FDR is higher at low replicate counts (n < 5).

---

## Design decisions & validation

See [`comparison_report.ipynb`](comparison_report.ipynb) for a full frozen, 12-section comparison of `loqculate` vs the original Pino 2020 scripts, including:

- FDR and TPR across concentration-level density (4–14 levels) and replicate count (2–20)
- LOQ accuracy and precision benchmarks
- Per-fit speed and whole-proteome throughput scaling (up to 10,000 peptides)

---

## Development

```bash
pip install -e ".[dev]"
pytest                 # 64 tests
ruff check loqculate/  # lint
```

Run benchmarks (requires ~5 min, tested with 24 threads):

```bash
python benchmarks/bench_simulation.py --save tmp/results/bench_simulation.json
python benchmarks/bench_n_concentrations.py --n_workers 24 --save tmp/results/bench_n_concentrations.json
python benchmarks/bench_n_replicates.py --n_workers 24 --save tmp/results/bench_n_replicates.json
```

---

## Roadmap

A guiding constraint: **loqculate is a practical proteomics tool, not a statistics library**. Each addition must justify its complexity against the realistic needs of a DIA/PRM lab.

#### Phase 1 — Solidify core (v0.2.x – v0.5.0)

Same piecewise/CV framework, no new model classes. Ends with a pip release after a bug-finding period on real datasets.

- [ ] **v0.2.x** — `loqculate.compat`: `OriginalWLS` / `OriginalCV` API wrappers; paper results reproducible without legacy scripts
- [ ] **v0.3.0** — closed-form piecewise solver (`PiecewiseCF`): linear-algebra knot search, no optimizer, ~5–20× faster, identical results; formalizes $w_i$ (per-point) vs $W_i$ (per-concentration) weight convention as a hard invariant across all WLS models
- [ ] **v0.4.0** — delta-method uncertainty (`--fast` flag): analytical LOQ CIs in $O(1)$; smooth models only — auto-falls back to bootstrap near the piecewise kink (non-differentiable $\max$)
- [ ] **v0.5.0** — segmented regression (`SegmentedWLS`): continuous piecewise linear with estimated breakpoint $\psi$, AIC/BIC vs flat-noise piecewise
- [ ] **pip release** — after real-data bug-finding marathon; semantic versioning locked, public API stable for Phase 2

#### Phase 2 — New models + UX (v0.6.0 – v1.0.0)

New curve families, model selection, performance, and publication-ready output.

- [ ] **v0.6.0** — 4PL/5PL model (`FourPL`): $y = d + (a-d)/[1+(x/c)^b]$; LLOQ + ULOQ; FDA bioanalytical guidance compliant; auto-downgrades to piecewise when upper asymptote is unidentifiable
- [ ] **v0.7.0** — model selection (`loqculate.selection`): AIC/BIC + approximate LOO-CV (PSIS-LOO, fast) per peptide; **auto-recommend** with explicit graceful degradation when models fail to converge or AIC penalty blows up on sparse curves
- [ ] **v0.8.0** — performance: Numba `@njit` on closed-form solver + bootstrap (~3–10×); chunked parquet I/O (`pyarrow`); shared-memory multiprocessing for 100k+ peptide datasets; adaptive bootstrap early-stopping co-designed with Numba cost model
- [ ] **v1.0.0** — stable public API contract; expanded plotting API (publication-ready diagnostics, waterfall, model-agreement scatter, yield curve — see [`plan/plotting_designs.md`](plan/plotting_designs.md)); exact LOO-CV and WAIC (requires Bayesian models, deferred to v1.x)

#### Phase 3 — Optional extensions (v1.x+)

Opt-in, not bundled in core.

- [ ] **Bayesian piecewise** (opt-in extra): MCMC / variational inference via `numpyro` or `pymc`; LOQ from posterior predictive. *Scope-limited to targeted panels (≤100 peptides) — HMC alone insufficient for discrete knot parameter; requires marginalization*
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
  version   = {0.2.0},
  url       = {https://github.com/MatrixMatched-Project/loqculate},
  license   = {MIT},
}
```

---

## License

MIT — see [LICENSE](LICENSE).

---

<div align="center">

*Three points hold the line,*  
*Through the sliding window's gaze,*  
*Noise becomes the truth.*

</div>
