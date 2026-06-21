"""Shared interface contracts."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt


@runtime_checkable
class Classifier(Protocol):
    """Minimal sklearn-compatible classifier interface.

    Any object exposing ``fit`` and ``predict`` satisfies this Protocol, so the
    simulation layer never depends on a concrete sklearn class.
    """

    def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.int_]) -> "Classifier": ...

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.int_]: ...
