"""Small NumPy losses for training summaries."""

import numpy as np


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    """Return mean squared error."""
    return float(np.mean((np.asarray(pred) - np.asarray(target)) ** 2))
