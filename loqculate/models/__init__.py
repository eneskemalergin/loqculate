# Models subpackage
from loqculate.compat import OriginalCV, OriginalWLS
from loqculate.models.cv_empirical import EmpiricalCV
from loqculate.models.piecewise_wls import PiecewiseWLS

MODEL_REGISTRY: dict = {
    "piecewise_wls": PiecewiseWLS,
    "cv_empirical": EmpiricalCV,
    "original_wls": OriginalWLS,
    "original_cv": OriginalCV,
}

__all__ = ["PiecewiseWLS", "EmpiricalCV", "OriginalWLS", "OriginalCV", "MODEL_REGISTRY"]
