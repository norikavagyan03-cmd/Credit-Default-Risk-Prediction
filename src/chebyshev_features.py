"""Chebyshev trend features for short behavioural time series.

A series is ordered oldest -> newest (e.g. April -> September).
Time is mapped to [-1, 1] and a Chebyshev polynomial of a small degree
is fitted by least squares.  The coefficients describe the trend; the
residual (what the polynomial cannot explain) measures irregularity.
"""
import numpy as np
from numpy.polynomial import chebyshev as C


def normalize_time(n):
    """n equally spaced points in [-1, 1]."""
    return np.linspace(-1.0, 1.0, n)


def chebyshev_fit(values, degree=3):
    """Return (coefficients, fitted values, residuals).

    With n points the degree must be < n.  For n = 6 use degree <= 3:
    a degree n-1 polynomial passes through every point, so the
    residual is always 0 and carries no information.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if degree >= n:
        raise ValueError(f"degree ({degree}) must be smaller than the series length ({n})")
    x = normalize_time(n)
    coef = C.chebfit(x, values, deg=degree)
    fitted = C.chebval(x, coef)
    return coef, fitted, values - fitted


def chebyshev_features(values, degree=3):
    """[a_0, ..., a_degree, rmse]"""
    coef, _, residual = chebyshev_fit(values, degree)
    rmse = np.sqrt(np.mean(residual ** 2))
    return np.concatenate([coef, [rmse]])


def chebyshev_feature_names(degree=3):
    return [f"cheb_a{k}" for k in range(degree + 1)] + ["cheb_rmse"]
