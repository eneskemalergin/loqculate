# utils subpackage
from loqculate.utils.normal_equations import (
    solve_2x2_wls,
    solve_2x2_wls_batch,
    weighted_mean,
)

__all__ = [
    "weighted_mean",
    "solve_2x2_wls",
    "solve_2x2_wls_batch",
]
