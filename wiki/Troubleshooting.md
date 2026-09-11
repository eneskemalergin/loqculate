# Troubleshooting

## `loqculate: command not found`

The environment that ran the install is not the one on `PATH`. Use `uv run loqculate` from the checkout, or see [Installation](Installation).

## `--fast requires --model piecewise_cf`

`--fast` is [PiecewiseCF](PiecewiseCF) only. The Python `loq(method="delta")` path does not fall back to bootstrap; the CLI flag does, once, when delta LOQ is infinite.

## Infinite LOD or LOQ

A stored result. Causes depend on the model:

- [PiecewiseCF](PiecewiseCF) / [PiecewiseWLS](PiecewiseWLS): $a\le 0$, too few noise observations, LOD above $\max(x)$, or the CV grid never holds `<= cv_thresh` for `sliding_window` consecutive **grid** points.
- Delta: $n_L<3$, no covariance, or the kink band covers the crossing.
- [OriginalWLS](OriginalWLS): different LOD inner-point guard; LOQ needs a grid point with CV **strictly** `<` threshold.
- [EmpiricalCV](EmpiricalCV) / [OriginalCV](OriginalCV): no LOD. LOQ infinite if no qualifying concentration level.

Look at `--plot y` before lowering `cv_thresh`.

## `--cv_thresh` or `--bootreps` seemed to do nothing

[CLI reference](CLI-Reference): several flags are parsed always and forwarded only to some models. OriginalWLS and OriginalCV compute LOQ during `fit()` from constructor defaults.

## Peptide count looks low

Unmapped filenames are dropped. Join rules: [Input formats](Input-Formats).

## `WARNING: Use DIA-NN diann_report.tsv instead of pr_matrix`

Header had `Stripped.Sequence` without `Precursor.Quantity`. Detection chose `diann_matrix`.

## Generic CSV still wants a map path

The second positional is always required. Generic layout does not join on it.

## Skyline or Spectronaut not detected

Header substrings, not extensions. Skyline needs `Total Area Fragment`, `Peptide Sequence`, and `File Name`. Spectronaut needs `PEP.StrippedSequence`. Pass `--format` if the header was rewritten.

## No `figuresofmerit.csv` from `compare`

`compare` only writes overlay plots.

## `ERROR processing ...` on one peptide

Logged to stderr. Row written with infinite LOD/LOQ. The run continues.
