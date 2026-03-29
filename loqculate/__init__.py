# loqculate package
__version__ = "0.3.0"

from loqculate.compat import OriginalCV, OriginalWLS
from loqculate.io import CalibrationData, read_calibration_data
from loqculate.models import MODEL_REGISTRY, EmpiricalCV, PiecewiseCF, PiecewiseWLS

# Register compat models here — after both loqculate.models and loqculate.compat
# are fully initialised — to avoid the circular import that arises when
# models/__init__.py tries to import from loqculate.compat at module load time.
MODEL_REGISTRY["original_wls"] = OriginalWLS
MODEL_REGISTRY["original_cv"] = OriginalCV

__all__ = [
    "__version__",
    "PiecewiseWLS",
    "PiecewiseCF",
    "EmpiricalCV",
    "OriginalWLS",
    "OriginalCV",
    "MODEL_REGISTRY",
    "CalibrationData",
    "read_calibration_data",
]
