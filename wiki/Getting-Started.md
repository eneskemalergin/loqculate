# Getting started

After [Installation](Installation), from the repository root:

```bash
uv run loqculate fit data/demo/one_protein.csv data/demo/filename2samplegroup_map.csv
```

The demo export is EncyclopeDIA-style (`numFragments` in the header). The command writes `figuresofmerit.csv` in the current directory. `--plot y` is the default, so calibration figures go there too.

## Concentration map

Second positional argument. Two columns: `filename` (basename, no directory) and `concentration` (numeric).

```csv
filename,concentration
sample_1ng_rep1.raw,1.0
sample_1ng_rep2.raw,1.0
sample_5ng_rep1.raw,5.0
```

How each reader joins on that file is on [Input formats](Input-Formats). Generic CSV still requires the path even though it does not join.

## Output

`fit` writes `peptide`, `LOD`, `LOQ`, `slope`, `intercept_linear`, `intercept_noise`. Infinite LOD or LOQ is a stored result, not a crash. CV-only models leave the slope and intercept cells empty.

`compare` writes overlay plots only. It does not write `figuresofmerit.csv`.

```bash
uv run loqculate compare data/demo/one_protein.csv data/demo/filename2samplegroup_map.csv --plot y
```

Default `--models` is `piecewise_cf,piecewise_wls`.

## Next

Flags: [CLI reference](CLI-Reference). Models: [Models](Models). Python: [Python API](Python-API).
