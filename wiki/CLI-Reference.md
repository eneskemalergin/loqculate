# CLI reference

loqculate v0.4.0. Subcommands: `fit`, `compare`. Both take `curve_data` then `filename_concentration_map`.

```bash
loqculate --version
loqculate fit --help
loqculate compare --help
```

## Shared flags (parsed for both commands)

| Flag | Default | Who actually uses it |
| --- | --- | --- |
| `--std_mult` | `2` | Passed to `lod(std_mult)` after fit. [PiecewiseCF](PiecewiseCF) and [PiecewiseWLS](PiecewiseWLS) honor it. [OriginalWLS](OriginalWLS) computes LOD inside `fit()` and ignores the later argument. |
| `--cv_thresh` | `0.20` | Passed to `loq(cv_thresh)` after fit. Honored by PiecewiseCF, PiecewiseWLS, [EmpiricalCV](EmpiricalCV). OriginalWLS and [OriginalCV](OriginalCV) ignore it after fit. Threshold sense is `<=` for `find_loq_threshold`, strict `<` for OriginalWLS. |
| `--bootreps` | `100` | Forwarded only when `--model` starts with `piecewise`. OriginalWLS keeps constructor `n_boot=100`. |
| `--min_noise_points` | `2` | Piecewise constructors only. OriginalWLS stores the value and does not use it in LOD. |
| `--min_linear_points` | `1` | Piecewise constructors only. Unused in OriginalWLS LOD. |
| `--sliding_window` | `3` | Piecewise constructors only. EmpiricalCV keeps constructor `3`. |
| `--multiplier_file` | unset | All models. CSV `peptide,multiplier`. Inner join. |
| `--output_path` | cwd | Created if missing. |
| `--plot` | `y` | `y` or `n`. |
| `--format` | `auto` | See [Input formats](Input-Formats). |
| `--chunk_size` | `100` | Peptides per worker batch (`fit`). |
| `--n_threads` | `cpu_count - 2` | Worker processes. `-1` means all CPUs. |

The LOD formula is not `noise + std_mult * sigma`. For the piecewise models it is $(c + \mathrm{std\_mult}\cdot\sigma - b)/a$. See [PiecewiseCF](PiecewiseCF).

## `fit`

```bash
loqculate fit DATA MAP [--model piecewise_cf] [--fast]
```

`--model`: `piecewise_cf`, `piecewise_wls`, `cv_empirical`, `original_wls`, `original_cv`.

`--fast`: [PiecewiseCF](PiecewiseCF) only. Delta LOQ first; bootstrap once if that value is infinite. Other models `sys.exit` with an error.

A peptide that raises during fit is logged to stderr and still written with infinite LOD/LOQ. The run continues.

## `compare`

```bash
loqculate compare DATA MAP [--models piecewise_cf,piecewise_wls]
```

Unknown names exit and print `MODEL_REGISTRY` keys. Failed models on one peptide are logged and omitted from that peptide's overlay. No CSV of LOD/LOQ.
