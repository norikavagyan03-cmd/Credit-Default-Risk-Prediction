"""Legendre trend features (same idea as chebyshev_features.py).

Used to compare the two polynomial systems on the same series.
"""
import numpy as np
from numpy.polynomial import legendre as L


def normalize_time(n):
    """n equally spaced points in [-1, 1]."""
    return np.linspace(-1.0, 1.0, n)


def legendre_fit(values, degree=3):
    """Return (coefficients, fitted values, residuals)."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if degree >= n:
        raise ValueError(f"degree ({degree}) must be smaller than the series length ({n})")
    x = normalize_time(n)
    coef = L.legfit(x, values, deg=degree)
    fitted = L.legval(x, coef)
    return coef, fitted, values - fitted


def legendre_features(values, degree=3):
    """[b_0, ..., b_degree, rmse]"""
    coef, _, residual = legendre_fit(values, degree)
    rmse = np.sqrt(np.mean(residual ** 2))
    return np.concatenate([coef, [rmse]])


def legendre_feature_names(degree=3):
    return [f"leg_b{k}" for k in range(degree + 1)] + ["leg_rmse"]
