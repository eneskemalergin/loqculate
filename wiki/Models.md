# Models

Five classes implement `CalibrationModel`. CLI `--model` keys are in backticks.

| Class | CLI key | Mean function | LOD | LOQ rule | Solver |
| --- | --- | --- | --- | --- | --- |
| [PiecewiseCF](PiecewiseCF) | `piecewise_cf` (default) | $y=\max(c, ax+b)$ | yes | sliding window on a CV *grid* (`<=`) | discrete knot search |
| [PiecewiseWLS](PiecewiseWLS) | `piecewise_wls` | $y=\max(c, ax+b)$ | yes | sliding window on a CV *grid* (`<=`) | scipy TRF `curve_fit` |
| [EmpiricalCV](EmpiricalCV) | `cv_empirical` | none | no (`inf`) | sliding window on calibration *levels* (`<=`) | group CV |
| [OriginalWLS](OriginalWLS) | `original_wls` | $y=\max(c, ax+b)$ | yes | single grid point with CV *strictly* `<` threshold | scipy TRF `least_squares` |
| [OriginalCV](OriginalCV) | `original_cv` | none | no (`inf`) | lowest level with CV `<=` threshold | group CV |

I use PiecewiseCF for new curves. I use OriginalWLS or OriginalCV when I need the published script's rule, including its LOQ definition. EmpiricalCV is the model-free check. PiecewiseWLS is the TRF sibling of CF, kept so the two solvers can be compared on the same data.

> [!IMPORTANT]
> "Three consecutive passing points" is not the same object in every class. Piecewise bootstrap and delta evaluate a linspace grid from LOD to $\max(x)$. EmpiricalCV evaluates unique positive concentrations. OriginalWLS does not use a window at all.

These numbers are not an agency LLOQ. ICH M10 / FDA bioanalytical text is about QC accuracy and precision at LLOQ, not this estimator.
