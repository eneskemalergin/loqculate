# EmpiricalCV

CLI: `--model cv_empirical`. Class: `loqculate.models.cv_empirical.EmpiricalCV`.

No regression. LOD is always $\infty$ (`supports_lod()` is false). Weights are ignored.

## Fit

Observations are grouped by unique concentration. Per group, CV is Bessel-corrected std / mean (`vectorized_cv_stats`). Groups with fewer than 2 observations get `nan` CV. If every group has fewer than 2 observations, `fit()` raises `ValueError`.

`min_replicates` default 3: concentrations with fewer replicates emit `UserWarning` and are still used.

`params_` is `{}`.

## LOQ

Positive concentrations only (blanks at $x=0$ are dropped).

`find_loq_threshold` on the sorted unique positive *calibration levels*, not on an interpolated grid. Default `sliding_window=3`, `cv_thresh=0.20`, non-strict `<=`. The window is consecutive **levels**.

## Predict

`predict(x_new)` linearly interpolates group means. Outside the calibration range it holds the nearest edge mean (`np.interp`).

## CLI

`--sliding_window` is not forwarded to this class. A CLI `cv_empirical` run uses constructor `sliding_window=3`. `--cv_thresh` is used, because `loq(cv_thresh)` reads the argument.

`compute_loqs_bulk` exists on the class for a vectorized many-peptide pass. The CLI `fit` path does not call it; it constructs one `EmpiricalCV` per peptide.

## What the notebook actually showed

`comparison_report.ipynb` compared EmpiricalCV and PiecewiseWLS (not PiecewiseCF). Findings recorded there, not a universal ranking:

- EmpiricalCV FDR on null curves depended strongly on replicate count. At `cv_thresh=0.20`, window 3, FDR was high at $n=2$ (the notebook prints 49% in that panel) and reached 0% at $n \ge 10$ in that experiment.
- PiecewiseWLS with window 3 already had FDR $< 1\%$ at $n=2$ in the same notebook.
- Original CV (window 1) was described there as 64-99% FDR across concentration-grid densities.

I treat EmpiricalCV as a model-free screen. I do not treat those FDR figures as a property of PiecewiseCF; CF was not in that notebook.
