from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np

from loqculate.config import DEFAULT_CV_THRESH, DEFAULT_STD_MULT


class CalibrationModel(ABC):
    """Abstract contract that every calibration model must satisfy.

    Design notes
    ------------
    * Follows scikit-learn-style ``fit()`` → ``predict()`` convention.
    * ``fit()`` stores raw training data as numpy arrays and returns *self*
      so calls can be chained: ``model.fit(x, y).lod()``.
    * All inputs are expected to be numpy arrays.  Convert pandas Series in
      the caller, not inside the model.
    """

    def __init__(self) -> None:
        self.params_: dict = {}
        self.is_fitted_: bool = False
        self.x_: Optional[np.ndarray] = None
        self.y_: Optional[np.ndarray] = None
        self.weights_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None) -> "CalibrationModel":
        """Fit the model.  Return *self* for chaining."""

    @abstractmethod
    def predict(self, x_new: np.ndarray) -> np.ndarray:
        """Return predicted signal at *x_new* concentrations."""

    @abstractmethod
    def lod(self, std_mult: float = DEFAULT_STD_MULT) -> float:
        """Limit of detection.  Return ``np.inf`` if undetermined."""

    @abstractmethod
    def loq(self, cv_thresh: float = DEFAULT_CV_THRESH) -> float:
        """Limit of quantitation.  Return ``np.inf`` if undetermined."""

    @abstractmethod
    def summary(self) -> dict:
        """Return all parameters, LOD, LOQ and fit statistics as a flat dict."""

    # ------------------------------------------------------------------
    # Optional methods with sensible defaults
    # ------------------------------------------------------------------

    def uloq(self) -> float:
        """Upper limit of quantitation.  Not supported by default."""
        return np.inf

    def prediction_interval(
        self, x_new: np.ndarray, alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Bootstrap prediction interval.  Subclasses override this."""
        raise NotImplementedError(f"{type(self).__name__} does not implement prediction_interval()")

    def supports_lod(self) -> bool:
        """Whether this model can compute an LOD."""
        return True

    def supports_uloq(self) -> bool:
        """Whether this model can compute an upper LOQ."""
        return False

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _check_is_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError(f"{type(self).__name__}.fit() must be called before predict().")
