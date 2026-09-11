# OriginalCV

CLI: `--model original_cv`. Class: `loqculate.compat.cv.OriginalCV`. Registered on `MODEL_REGISTRY` when you `import loqculate`.

Port of `old/loq_by_cv.py` `calculate_LOQ_byCV()`. No regression, no LOD, no sliding window.

## Fit

Groups by unique concentration. CV is Bessel-corrected std / mean (`vectorized_cv_stats`), `nan` when count $< 2$ or mean is 0.

LOQ is computed during `fit()`:

$$\mathrm{LOQ} = \min\{ C > 0 : \mathrm{CV}(C) \le \tau \}$$

$\tau$ is the constructor `cv_thresh` (default 0.20). Non-strict `<=`, matching the original `<= 0.2` filter. If no level qualifies, $\infty$.

`loq(cv_thresh)` after fit ignores the argument and returns the stored value. CLI `--cv_thresh` is therefore unused for this model. `--sliding_window` and `--bootreps` are unused.

`lod()` is always $\infty`. `supports_lod()` is false.

## Predict

`predict()` raises `NotImplementedError`. There is no curve.

## Compared with EmpiricalCV

Same CV estimator. Different LOQ rule: OriginalCV is single-point `<=`; EmpiricalCV requires `sliding_window` consecutive positive levels. `comparison_report.ipynb` treats OriginalCV FDR as the window=1 case (64-99% across the concentration-density experiment in that notebook).
