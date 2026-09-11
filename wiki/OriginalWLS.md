# OriginalWLS

CLI: `--model original_wls`. Class: `loqculate.compat.wls.OriginalWLS`. Registered on `MODEL_REGISTRY` when you `import loqculate`.

Port of `old/calculate-loq.py` `process_peptide(model='piecewise')`. I keep it so a rewrite can still be compared to the paper script. It is not a drop-in for PiecewiseCF: the LOQ rule is different, and LOD guards are different.

The original script used lmfit. This class uses `scipy.optimize.least_squares(..., method="trf")` with the same $(a, b, c-b)$ bounds. CHANGELOG 0.2.2 already records that 2 of 27 demo peptides can land in a different local optimum than lmfit.

## Fit

- At least 3 observations and at least 2 unique $x$, else `ValueError`.
- Weights: `min(1/(sqrt(x)+eps), 1000)`, same cap as `inverse_sqrt_weights`.
- Init: `_initialize_params_legacy` (top-two concentrations).
- Solver: TRF, `max_nfev=5000`. Unlike PiecewiseWLS, a solver failure is not caught; `result.x` is used as returned.
- LOD and bootstrap LOQ are computed **inside** `fit()`, not lazily.

`lod(std_mult)` and `loq(cv_thresh)` after fit ignore those arguments and return the values stored at fit time. Constructor `std_mult` and `cv_thresh` (defaults 2 and 0.20) are the ones that matter. The CLI passes `--std_mult` / `--cv_thresh` into `lod()` / `loq()` after fit, so **those CLI flags do not change OriginalWLS output**. `--bootreps` is also not forwarded (name does not start with `piecewise`).

## LOD

`_calculate_lod_legacy`:

1. $a \le 0$ $\to$ $\infty$.
2. $x_{\mathrm{int}} = (c-b)/a$.
3. $\sigma$ from all $y$ with $x < x_{\mathrm{int}}$, `ddof=1`, if at least one such point; otherwise $\sigma=\infty$.
4. $\mathrm{LOD} = (c + \mathrm{std\_mult}\cdot\sigma - b)/a$.
5. If $\mathrm{LOD} > \max(x)$, $\infty$.
6. Let `inner_points` be unique $x$ with the global min and max removed. If that set is empty, or $\mathrm{LOD} < \min(\mathrm{inner\_points})$, $\infty$.

`min_noise_points` and `min_linear_points` are accepted on the constructor and passed into this function. They are not read in the function body.

## LOQ

Bootstrap predictions on `np.linspace(LOD, max(x), 100)`. Per-replicate seed `SeedSequence(i)` for `i in range(n_boot)`, matching the original `_bootstrap_once`. CV is `std/mean` with `ddof=1` on the bootstrap matrix.

Single-point rule, **strict** inequality:

$$\mathrm{LOQ} = \min\{ x_{\mathrm{grid}} : x_{\mathrm{grid}} > \mathrm{LOD},\ \mathrm{CV}(x_{\mathrm{grid}}) < \tau \}$$

No sliding window. If none, or the min is $\le 0$ or $\ge \max(x_{\mathrm{grid}})$, $\infty$.

This is not `find_loq_threshold`, and it is not `<=`.
