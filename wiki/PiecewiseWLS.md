# PiecewiseWLS

CLI: `--model piecewise_wls`. Class: `loqculate.models.piecewise_wls.PiecewiseWLS`.

Same mean function as [PiecewiseCF](PiecewiseCF): $y = \max(c, ax+b)$. Same default weights. Different solver.

I keep this class so I can compare TRF against the closed-form search on the same peptides. It is not the CLI default.

## Solver

`scipy.optimize.curve_fit` with `method` defaulting to TRF, `maxfev=5000`, bounds $a \ge 0$, $c-b \ge 0$, $b$ unbounded. The curve is parameterized as $(a, b, c-b)$ so the noise-above-linear constraint is a bound.

`init_method` default is `"legacy"`: slope from the mean signals at the two highest unique concentrations, noise intercept from the lowest, linear intercept from the top point. `"auto"` is a WLS line above the two lowest concentrations. The CLI does not expose `init_method`; a CLI `piecewise_wls` fit is legacy init.

If TRF raises `RuntimeError` or `ValueError`, `fit()` stores the clamped initial guess and does not raise.

If `np.ptp(y) == 0` (constant area, including all zeros), TRF is skipped. Slope is 0 and both intercepts equal that constant. That path exists because a tiny positive TRF slope on a flat curve produced a finite LOD on arm64.

`params_` has `slope`, `intercept_linear`, `intercept_noise`. There is no `knot_x`.

## LOD and LOQ

LOD uses the same arithmetic as PiecewiseCF: intersection $(c-b)/a$, noise observations $x < x_{\mathrm{int}}$, `ddof=1` std, $\mathrm{LOD}=(c+\mathrm{std\_mult}\cdot\sigma-b)/a$, then the `min_noise_points` / `min_linear_points` / `max(x)` guards.

LOQ is bootstrap only. `loq()` has no `method` argument. Grid, window, and `<=` threshold match PiecewiseCF (`grid_points=100`, `sliding_window=3`, `cv_thresh=0.20`). The window is consecutive **grid points**.

There is no `loq_delta` and no CLI `--fast` on this model.

## Constructor defaults

`init_method="legacy"`, `n_boot_reps=100`, `seed=42`, `min_noise_points=2`, `min_linear_points=1`, `sliding_window=3`, `grid_points=100`. `fit()` requires at least 3 observations.
