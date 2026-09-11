# loqculate

This wiki documents loqculate v0.4.1. Run `loqculate --version` before following version-specific behavior.

loqculate reads a calibration export, joins run filenames to concentrations, fits one model per peptide, and writes LOD, LOQ, and optional plots. The default model is [PiecewiseCF](PiecewiseCF). Default LOQ is bootstrap.

## Start

1. [Install from source](Installation) (not on PyPI).
2. Run the [demo fit](Getting-Started).
3. Open the [CLI reference](CLI-Reference) when a flag does not do what you expected. Several flags are parsed for every model and only forwarded to some of them.

## Models

Each model has its own page. [Models](Models) is only the comparison.

- [PiecewiseCF](PiecewiseCF): default. Closed-form knot search. Bootstrap LOQ, optional delta.
- [PiecewiseWLS](PiecewiseWLS): same piecewise mean function, scipy TRF solver.
- [EmpiricalCV](EmpiricalCV): replicate CVs at calibration levels. No LOD.
- [OriginalWLS](OriginalWLS): Pino 2020 WLS path (`old/calculate-loq.py`).
- [OriginalCV](OriginalCV): Pino 2020 CV path (`old/loq_by_cv.py`).

## Elsewhere

- [Python API](Python-API)
- [Input formats](Input-Formats)
- [Benchmarks](Benchmarks)
- [Troubleshooting](Troubleshooting)
- [Development](Development)
