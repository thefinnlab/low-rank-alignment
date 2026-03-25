"""Project-wide constants."""

import numpy as np

# Default ridge regression alpha grid for RidgeCV.
# Covers light regularisation (1e-2) through heavy regularisation (1e10).
RIDGE_ALPHAS = np.logspace(-2, 10, 13)
