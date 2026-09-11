# PiecewiseCF

CLI: `--model piecewise_cf`. Default since v0.3.0 (`DEFAULT_MODEL` in `loqculate/config.py`). Class: `loqculate.models.piecewise_cf.PiecewiseCF`.

I use this as the default because the mean function is the same piecewise line as PiecewiseWLS, and the knot is chosen by a search I can state, not by where TRF happened to stop.

## Mean function

$$y = \max(c,\ ax + b)$$

`params_` after a successful `fit()`:

- `slope` ($a$)
- `intercept_linear` ($b$)
- `intercept_noise` ($c$)
- `knot_x` (partition used for the linear/noise split)

`predict(x)` returns `max(c, a*x + b)`.

## Weights

If you do not pass `weights`, `fit()` calls `inverse_sqrt_weights`:

$$w_i = \min\left(\frac{1}{\sqrt{x_i}+\varepsilon},\ 1000\right)$$

$\varepsilon$ is `np.finfo(float).eps`. Cap is `DEFAULT_WEIGHT_CAP = 1000`. Precision weights used in the knot search are $W_i = w_i^2$. Stored `weights_` is $w_i$, not $W_i$.

This is not exactly $W_i = 1/x_i$. Near $x=0$ the cap binds.

## Knot search

`fit()` requires at least 3 observations. `find_knot` then requires at least 3 unique $x$ values or it raises `ValueError`.

First pass. Candidates are `unique(x)[1:-1]` (interior unique concentrations). At each candidate $k$, observations with $x \le k$ are the noise segment (weighted mean for $c$); $x > k$ are the linear WLS segment for $(a,b)$. Constraints after each candidate:

- if $a < 0$, set $a=0$ and replace the linear fit with a weighted mean
- if $c < b$, set $c = b$ and recompute noise RSS

The candidate with the lowest total weighted RSS wins. Ties keep the lower $x$ (strict `<` on RSS).

Second pass. If $a > 0$, compute $x_{\mathrm{join}} = (c-b)/a$. If that value lies strictly inside $(\min x, \max x)$, both segments are refit with the split at $x_{\mathrm{join}}$, and `knot_x` becomes $x_{\mathrm{join}}$. That refit is applied even if the new RSS is larger. Otherwise `knot_x` stays the discrete winner.

The search is exhaustive over those candidates. It is not a continuous breakpoint estimator, and it is not "globally optimal" over all possible real-valued splits.

## LOD

`lod(std_mult=2)`:

1. If $a \le 0$, return $\infty$.
2. Intersection $x_{\mathrm{int}} = (c-b)/a$.
3. Noise observations: $y$ with $x < x_{\mathrm{int}}$. If there are fewer than `min_noise_points` (default 2) such *observations*, return $\infty$.
4. $\sigma$ is `np.std(noise_y, ddof=1)`.
5. $\mathrm{LOD} = (c + \mathrm{std\_mult}\cdot\sigma - b) / a$.
6. If LOD is non-finite or `LOD > max(x)`, return $\infty$.
7. If the number of *unique* $x$ values strictly above LOD is less than `min_linear_points` (default 1), return $\infty$.

`min_noise_points` counts observations. `min_linear_points` counts unique concentrations.

## Bootstrap LOQ (default)

`loq()` and `loq(method="bootstrap")` (default `method`):

1. If LOD is infinite, LOQ is infinite.
2. Build a grid `np.linspace(LOD, max(x), grid_points)` with `grid_points=100`.
3. Nonparametric bootstrap of observation triples $(x,y,W)$, `n_boot_reps=100`, `seed=42`. Each replicate refits the CF knot search. CV at a grid point is `nanstd(..., ddof=1) / nanmean` of the bootstrap predictions.
4. `find_loq_threshold` on that *grid*: first positive grid $x$ where `cv <= cv_thresh` for `sliding_window` consecutive grid points (`cv_thresh=0.20`, `window=3`). Window is capped at the number of positive-grid points. `window < 1` yields $\infty$.
5. If that value is non-finite, $\le 0$, or $\ge \max(x)$, return $\infty$.

The three-point window is consecutive **grid points**, not consecutive calibration levels.

If the float64 replicate matrix would exceed 100 MB, the vectorized bootstrap emits `ResourceWarning` and falls back to a per-replicate loop. Both paths draw indices from `SeedSequence(seed).spawn(n_reps)`.

`summary()["loq"]` calls `loq()` and is therefore bootstrap.

## Delta-method LOQ

`loq(method="delta")` and `loq_delta()` share one path. They never call bootstrap.

Linear-segment residuals use $x > \texttt{knot\_x}$. MSE is $\mathrm{WRSS}/(n_L-2)$ and is undefined for $n_L < 3$ (`covariance()` is then `None`). Prediction variance at $x_0$ is

$$\mathrm{Var}(\hat y(x_0)) = \frac{\mathrm{MSE}}{W(x_0)} + [x_0,\ 1]\,\mathrm{Cov}(\hat a,\hat b)\,[x_0,\ 1]^\top$$

CV is $\sqrt{\mathrm{Var}} / \hat y$, on a 1000-point grid from LOD to $\max(x)$. Inside a kink band of half-width `2 * min_spacing` around $x^* = (c-b)/a$, CV is set to $\infty$ so the linear-branch formula is not used at the join. Missing weights raise. Missing covariance or undefined MSE yields empty profile and LOQ $\infty$. When $3 \le n_L < 5$, a `UserWarning` is emitted and the value is still computed.

Same `find_loq_threshold` rule as bootstrap (`<=`, window from `sliding_window`).

I leave bootstrap as the default. Delta is a different estimator. I will not pretend a faster number is the same number.

## CLI `--fast`

`loqculate fit ... --fast` is CF-only. It calls `loq(method="delta")` first; if that LOQ is infinite, it calls `loq(method="bootstrap")` once. Other models exit with an error. The Python methods above do not do that fallback.

## Constructor defaults

`n_boot_reps=100`, `seed=42`, `min_noise_points=2`, `min_linear_points=1`, `sliding_window=3`, `grid_points=100`.
