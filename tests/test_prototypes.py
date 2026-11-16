"""Checks that the Chebyshev, Legendre and Walsh prototypes are correct.

Run from the project root:  python tests/test_prototypes.py
"""
import pathlib
import sys

import numpy as np
from scipy.linalg import hadamard

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.chebyshev_features import chebyshev_features
from src.legendre_features import legendre_features
from src.walsh_features import (fwht, sequency_order, walsh_coefficients,
                                walsh_features, to_power_of_two)

x = np.linspace(-1, 1, 6)


def test_chebyshev_recovers_known_polynomial():
    # 2 + 3*T1(x) + 0.5*T2(x), where T1 = x and T2 = 2x^2 - 1
    y = 2 + 3 * x + 0.5 * (2 * x ** 2 - 1)
    f = chebyshev_features(y, degree=3)
    assert np.allclose(f[:4], [2, 3, 0.5, 0], atol=1e-9)
    assert f[-1] < 1e-9


def test_legendre_recovers_known_polynomial():
    # 1 + 2*P1(x) + P2(x), where P2 = (3x^2 - 1)/2
    y = 1 + 2 * x + (3 * x ** 2 - 1) / 2
    f = legendre_features(y, degree=3)
    assert np.allclose(f[:4], [1, 2, 1, 0], atol=1e-9)
    assert f[-1] < 1e-9


def test_fwht_matches_hadamard_matrix():
    v = np.random.default_rng(0).normal(size=8)
    assert np.allclose(fwht(v), hadamard(8) @ v)


def test_walsh_is_orthonormal():
    v = np.random.default_rng(1).normal(size=16)
    c = walsh_coefficients(v)
    assert np.isclose(np.sum(c ** 2), np.sum(v ** 2))  # energy is preserved


def test_sequency_order_counts_sign_changes():
    n = 8
    W = hadamard(n)[sequency_order(n)]
    changes = (np.diff(W, axis=1) != 0).sum(axis=1)
    assert list(changes) == list(range(n))


def test_walsh_detects_step_and_alternation():
    step = np.array([0, 0, 0, 0, 1, 1, 1, 1.0])      # one slow jump
    zigzag = np.array([1, -1, 1, -1, 1, -1, 1, -1.0])  # fastest change
    assert np.argmax(np.abs(walsh_coefficients(step)[1:])) + 1 == 1
    assert np.argmax(np.abs(walsh_coefficients(zigzag)[1:])) + 1 == 7


def test_padding_gives_length_8():
    s = np.arange(6.0)
    for m in ("edge", "zero", "interp"):
        assert len(to_power_of_two(s, m)) == 8


def test_features_are_finite_on_constant_series():
    flat = np.zeros(6)
    for f in (chebyshev_features(flat), legendre_features(flat), walsh_features(flat)):
        assert np.all(np.isfinite(f))


def test_walsh_high_share_grows_for_irregular_series():
    smooth = np.array([0, 1, 2, 3, 4, 5.0])
    jumpy = np.array([0, 5, 0, 5, 0, 5.0])
    assert walsh_features(jumpy)[2] > walsh_features(smooth)[2]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("OK  ", t.__name__)
    print(f"\nAll {len(tests)} checks passed.")
