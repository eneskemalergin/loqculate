# Models subpackage
from loqculate.models.cv_empirical import EmpiricalCV
from loqculate.models.piecewise_wls import PiecewiseWLS

MODEL_REGISTRY: dict = {
    'piecewise_wls': PiecewiseWLS,
    'cv_empirical': EmpiricalCV,
}

__all__ = ['PiecewiseWLS', 'EmpiricalCV', 'MODEL_REGISTRY']

