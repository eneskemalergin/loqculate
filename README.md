<div align="center">

<!-- Drop your logo file at docs/logo.png to activate -->
<img src="docs/logo.png" alt="loqculate logo" width="120" />

# loqculate

**Limit of Detection and Quantitation calculator for mass-spectrometry calibration curves.**

<table>
<tr>
  <td><a href="https://github.com/eneskemalergin/loqculate/actions/workflows/ci.yml"><img src="https://github.com/eneskemalergin/loqculate/actions/workflows/ci.yml/badge.svg" alt="CI"></a></td>
  <td><img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue" alt="Python versions"></td>
  <td><a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a></td>
  <td><a href="https://codecov.io/gh/eneskemalergin/loqculate"><img src="https://codecov.io/gh/eneskemalergin/loqculate/branch/main/graph/badge.svg" alt="codecov"></a></td>
</tr>
</table>

</div>

---

`loqculate` (v0.2.x) is a ground-up rewrite of the original Pino 2020 LOD/LOQ scripts with a validated, production-ready engine:

- A **sliding-window LOQ rule** that reduces FDR from ~100% to <5%: LOQ is declared at $C_i$ only when $\mathrm{CV}(C_i)$, $\mathrm{CV}(C_{i+1})$, and $\mathrm{CV}(C_{i+2})$ all fall below `cv_thresh` — three consecutive passing concentrations, not one
- **11.4× lower memory** per peptide vs the original scripts (VmRSS, 1 000 peptide batch); throughput advantage scales with dataset size — batch is 3.3× faster at 10 000 peptides due to reduced GC pressure (see [`comparison_report.ipynb`](comparison_report.ipynb))
- **TRF solver** (`scipy.optimize.least_squares`, method='trf') with guaranteed convergence bounds, replacing unconstrained curve_fit calls
- A second model — **EmpiricalCV** — as a model-free alternative to PiecewiseWLS, very fast but higher FDR at low replicate counts (n < 5)
- **`loqculate.compat`** — drop-in `OriginalWLS` / `OriginalCV` wrappers reproducing paper results without the legacy scripts; registered in `MODEL_REGISTRY` and usable via the CLI (`--model original_wls`)
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

# Fit with original Pino 2020 WLS (for paper reproducibility)
loqculate fit data.tsv conc_map.csv --model original_wls

# Fit with original Pino 2020 CV method
loqculate fit data.tsv conc_map.csv --model original_cv

# Compare both modern models side-by-side
loqculate compare data.tsv conc_map.csv --models piecewise_wls,cv_empirical
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
from loqculate import read_calibration_data, PiecewiseWLS, EmpiricalCV
from loqculate.compat import OriginalWLS, OriginalCV

data = read_calibration_data("data.tsv", "conc_map.csv")

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
| Generic CSV      | `peptide`, `concentration`, `area` columns         | No — concentration column is embedded |

**`conc_map.csv` rules:**

- Two columns: `filename` (basename, no path) and `concentration` (numeric, same units as your calibration range).
- Filenames that appear in the data but are missing from the map are **silently dropped** with a warning — check your logs if peptide counts look low.
- All 6 formats except Generic CSV require a map; passing `None` raises a `ValueError` at load time.

---

## Models

### PiecewiseWLS (default)

Fits a piecewise linear model (noise floor + linear signal) using weighted least squares. LOD and LOQ are derived from the fitted parameters. The sliding-window rule requires `window` consecutive concentration points below the CV threshold before declaring a LOQ.

$$y = \begin{cases} \alpha & x \le \kappa \\ \alpha + \beta\,(x - \kappa) & x > \kappa \end{cases}$$

$\alpha$ — noise floor, $\kappa$ — knot (signal onset), $\beta$ — slope in the quantifiable range.
Weights: $w_i = 1/\sqrt{x_i}$ (user-facing, per-point); precision weight $W_i = w_i^2 = 1/x_i$. LOD and LOQ are derived from $\alpha$, $\beta$, and $\kappa$ under a CV threshold.

### EmpiricalCV

Computes CV directly from replicate measurements at each concentration level. LOQ is the lowest concentration where `window` consecutive CVs fall below `cv_thresh`. No parametric assumptions — faster than WLS, but FDR is higher at low replicate counts (n < 5).

$$\mathrm{LOQ} = \min\{ C_i : \mathrm{CV}(C_i) < \tau,\ \mathrm{CV}(C_{i+1}) < \tau,\ \mathrm{CV}(C_{i+2}) < \tau \}$$

$\tau$ is `cv_thresh` (default 0.20). Requiring three consecutive passing concentrations reduces single-point false discoveries from ~100% to <5%.

### OriginalWLS (`loqculate.compat`)

Verbatim port of `old/calculate-loq.py` `process_peptide(model='piecewise')` from Pino 2020. Reproduces the exact original logic: legacy parameter initialisation (slope from top-two concentration points), scipy TRF solver with the same bounds as the original lmfit call, original LOD formula (flat noise floor + $n \cdot \sigma$), and **single-point bootstrap LOQ rule** (no sliding window). Bootstrap seeds are `SeedSequence(i)` for `i in range(n_boot)` — identical to the original script.

Use when you need byte-for-byte reproducibility of published results. For new analyses, prefer `PiecewiseWLS` (3.3× faster in batch, identical math, sliding-window LOQ).

### OriginalCV (`loqculate.compat`)

Verbatim port of `old/loq_by_cv.py` `calculate_LOQ_byCV()`. LOQ is the lowest positive concentration where CV ≤ `cv_thresh` (non-strict `≤`, matching the original `<= 0.2` check). No LOD, no sliding window, no parametric model. Deterministic and near-instant — same speed as EmpiricalCV but with higher FDR due to the single-point rule (64–99% on null curves).

---

## Configuration & defaults

| Parameter   | Default         | Description                                                              |
| ----------- | --------------- | ------------------------------------------------------------------------ |
| `cv_thresh` | `0.20`          | CV threshold; LOQ requires CV < this value                               |
| `window`    | `3`             | Consecutive passing concentration points required                        |
| `n_boot`    | `100`           | Bootstrap resamples for LOD/LOQ CI estimation                            |
| `std_mult`  | `2`             | Noise-floor multiplier for LOD ($\mathrm{LOD} = \alpha + 2\sigma$)       |
| `model`     | `piecewise_wls` | Active model; also accepts `cv_empirical`, `original_wls`, `original_cv` |

Pass these as CLI flags (`--cv_thresh 0.15`) or keyword arguments to `.fit()`.

---

## Benchmark summary

Machine: AMD Ryzen 9 3950X, 32 threads. FDR measured on 300 simulated null curves across 4–14 concentration levels (window=3). Throughput measured on real DIA-NN data.

| Metric                                      |   PiecewiseWLS   | EmpiricalCV |   OriginalWLS    | OriginalCV |
| ------------------------------------------- | :--------------: | :---------: | :--------------: | :--------: |
| Per-fit, single-threaded                    |     ~1.9 ms      |  ~0.02 ms   |     ~1.8 ms      |  ~0.03 ms  |
| Null FDR, window=3 (range over n_conc 4–14) |       0–2%       |    2–18%    |       1–2%       |   64–99%   |
| Memory per peptide (VmRSS)                  |     0.013 MB     |     ~0      |     0.151 MB     |     ~0     |
| Throughput, 10 000 pep, 32 threads          | 41 s (241 pep/s) |   0.19 s    | 136 s (74 pep/s) |   0.65 s   |

Key takeaway: WLS methods (PiecewiseWLS and OriginalWLS) have near-identical per-fit cost, but PiecewiseWLS is **3.3× faster in batch** due to 11.4× lower memory enabling better parallel utilisation. OriginalCV has catastrophically high FDR (64–99%) — the single-point rule is the problem, not the CV model itself. Full benchmark data and reproduction commands in [`comparison_report.ipynb`](comparison_report.ipynb).

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
pre-commit install      # install git hooks (runs ruff on every commit)
pytest                 # 101 tests
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

- [x] **v0.2.2** ✓ — `loqculate.compat`: `OriginalWLS` / `OriginalCV` API wrappers; paper results reproducible without legacy scripts
- [ ] **v0.3.0** — closed-form piecewise solver (`PiecewiseCF`): linear-algebra knot search, no optimizer, ~5–20× faster, identical results; formalizes $w_i$ (per-point) vs $W_i$ (per-concentration) weight convention as a hard invariant across all WLS models
- [ ] **v0.4.0** — delta-method uncertainty (`--fast` flag, double-dash): analytical LOQ CIs in $O(1)$; smooth models only — auto-falls back to bootstrap near the piecewise kink (non-differentiable $\max$)
- [ ] **v0.5.0** — segmented regression (`SegmentedWLS`): continuous piecewise linear with estimated breakpoint $\psi$, AIC/BIC vs flat-noise piecewise
- [ ] **pip release** — after real-data bug-finding marathon; semantic versioning locked, public API stable for Phase 2

#### Phase 2 — New models + UX (v0.6.0 – v1.0.0)

New curve families, model selection, performance, and publication-ready output.

- [ ] **v0.6.0** — 4PL/5PL model (`FourPL`): $y = d + (a-d)/[1+(x/c)^b]$; LLOQ + ULOQ; FDA bioanalytical guidance compliant; fitted via `scipy.optimize.curve_fit` (non-linear, unlike the current piecewise linear approach); auto-downgrades to piecewise when upper asymptote is unidentifiable
- [ ] **v0.7.0** — model selection (`loqculate.selection`): AIC/BIC + approximate LOO-CV (PSIS-LOO, fast) per peptide; **auto-recommend** with explicit graceful degradation when models fail to converge or AIC penalty blows up on sparse curves
- [ ] **v0.8.0** — performance: Numba `@njit` on closed-form solver + bootstrap (~3–10×); chunked parquet I/O (`pyarrow`); shared-memory multiprocessing for 100k+ peptide datasets; adaptive bootstrap early-stopping co-designed with Numba cost model
- [ ] **v1.0.0** — stable public API contract; expanded plotting API (publication-ready diagnostics, waterfall, model-agreement scatter, yield curve); exact LOO-CV and WAIC (requires Bayesian models, deferred to v1.x)

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
  version   = {0.2.2},
  url       = {https://github.com/eneskemalergin/loqculate},
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
