"""Walsh-Hadamard features.

Key points
----------
* The transform needs a length that is a power of two (4, 8, 16, ...).
  Our series have 6 points, so `to_power_of_two` pads / resamples to 8.
* scipy's `hadamard` and a plain FWHT give coefficients in *natural*
  order, where the index is NOT a frequency.  Walsh coefficients are
  reordered by *sequency* (number of sign changes), so index k = k sign
  changes: k = 0 is the mean, small k = slow changes, large k = rapid
  alternation.  Only then do "low / mid / high" bands make sense.
"""
from functools import lru_cache

import numpy as np
from scipy.linalg import hadamard

EPS = 1e-12


def is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0


def to_power_of_two(values, method="edge"):
    """Bring a series to the next power-of-two length.

    method="edge"   repeat the last value (default)
    method="zero"   append zeros
    method="interp" resample by linear interpolation
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if is_power_of_two(n):
        return values
    m = 1 << int(np.ceil(np.log2(n)))
    if method == "edge":
        return np.pad(values, (0, m - n), mode="edge")
    if method == "zero":
        return np.pad(values, (0, m - n))
    if method == "interp":
        return np.interp(np.linspace(0, n - 1, m), np.arange(n), values)
    raise ValueError("method must be 'edge', 'zero' or 'interp'")


def fwht(x):
    """Fast Walsh-Hadamard transform, natural order, unnormalised: H @ x.
    Complexity O(n log n)."""
    a = np.array(x, dtype=float)
    n = len(a)
    if not is_power_of_two(n):
        raise ValueError("length must be a power of two")
    h = 1
    while h < n:
        for i in range(0, n, 2 * h):
            lo = a[i:i + h].copy()
            hi = a[i + h:i + 2 * h].copy()
            a[i:i + h] = lo + hi
            a[i + h:i + 2 * h] = lo - hi
        h *= 2
    return a


@lru_cache(maxsize=None)
def sequency_order(n):
    """Permutation that sorts natural-order rows by number of sign changes."""
    H = hadamard(n)
    sequency = (np.diff(H, axis=1) != 0).sum(axis=1)
    return np.argsort(sequency)


def walsh_coefficients(values):
    """Orthonormal Walsh coefficients in sequency order: c = W x / sqrt(n)."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    c = fwht(values) / np.sqrt(n)
    return c[sequency_order(n)]


def band_edges(n):
    """Sequency bands (DC excluded): low [1, e1), mid [e1, e2), high [e2, n).
    For n = 8: low = 1..2, mid = 3..4, high = 5..7."""
    e1 = max(2, int(round(n * 3 / 8)))
    e2 = max(e1 + 1, int(round(n * 5 / 8)))
    return e1, e2


def walsh_features(values, pad="edge", normalize=True):
    """[low, mid, high relative energy, spectral entropy]

    The series is padded to a power of two, optionally z-scored, and
    transformed.  Energies are shares of the non-DC energy (they sum
    to 1); a rising 'high' share means rapid, irregular changes.
    """
    x = to_power_of_two(values, pad)
    n = len(x)
    if n < 4:
        raise ValueError("need at least 4 points")
    if normalize:
        x = (x - x.mean()) / (x.std() + EPS)
    power = walsh_coefficients(x) ** 2
    e1, e2 = band_edges(n)
    total = power[1:].sum()
    low, mid, high = (power[1:e1].sum(), power[e1:e2].sum(), power[e2:].sum())
    p = power[1:] / (total + EPS)
    entropy = -np.sum(p * np.log(p + EPS))
    return np.array([low / (total + EPS), mid / (total + EPS), high / (total + EPS), entropy])


def walsh_feature_names():
    return ["walsh_low", "walsh_mid", "walsh_high", "walsh_entropy"]
