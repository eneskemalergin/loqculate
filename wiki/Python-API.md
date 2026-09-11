# Python API

Public root imports after `import loqculate`:

```python
from loqculate import (
    CalibrationData,
    EmpiricalCV,
    MODEL_REGISTRY,
    PiecewiseCF,
    PiecewiseWLS,
    read_calibration_data,
    __version__,
)
from loqculate.compat import OriginalCV, OriginalWLS
```

`MODEL_REGISTRY` keys: `piecewise_cf`, `piecewise_wls`, `cv_empirical`, `original_wls`, `original_cv`. Compat classes are registered at package import, not inside `loqculate.models`.

## Arrays, not an iterator

```python
import numpy as np
from loqculate import PiecewiseCF, read_calibration_data

data = read_calibration_data("data.tsv", "conc_map.csv")
# data.peptide, data.concentration, data.area
# data.metadata["format"], ["n_peptides"], ["n_measurements"]

for peptide in np.unique(data.peptide):
    mask = data.peptide == peptide
    m = PiecewiseCF().fit(data.concentration[mask], data.area[mask])
    print(peptide, m.lod(), m.loq())
```

There is no `iter_peptides()`. `fmt=` defaults to `"auto"`. The map path is required even for generic CSV.

`apply_multiplier(data, path)` is in `loqculate.io`. Columns `peptide`, `multiplier`. Inner join.

## Per-class methods

The math is on the per-model pages. [PiecewiseCF](PiecewiseCF) documents `loq(method=...)`, `loq_delta()`, `covariance()`, and `--fast`. [PiecewiseWLS](PiecewiseWLS) `loq()` has no `method`. [OriginalWLS](OriginalWLS) and [OriginalCV](OriginalCV) compute LOQ during `fit()` and ignore later `loq(cv_thresh)` arguments.
