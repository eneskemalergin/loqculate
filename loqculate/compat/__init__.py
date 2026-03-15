"""loqculate.compat — verbatim API wrappers for the original Pino 2020 scripts.

These classes reproduce the exact numerical logic of:
  - ``old/calculate-loq.py`` → :class:`OriginalWLS`
  - ``old/loq_by_cv.py``     → :class:`OriginalCV`

They satisfy the :class:`~loqculate.models.base.CalibrationModel` interface so
they can be used anywhere PiecewiseWLS or EmpiricalCV are used, including the
CLI (``loqculate fit --model original_wls``).

The intent is reproducibility: users who published results with the legacy
scripts can verify agreement with loqculate's engine, and new users can
compare both approaches side-by-side.
"""

from loqculate.compat.cv import OriginalCV
from loqculate.compat.wls import OriginalWLS

__all__ = ["OriginalWLS", "OriginalCV"]
