"""Track A: synthetic utility proxy (Eq. 14) and calibrated coupling."""

import math

from analysis.accounting import rdp_epsilon
from analysis.utility import calibrated_utility, utility


def test_a_utility_clipped_and_monotonic_in_sigma():
    u_lo = utility(c=2.0, sigma=0.5)
    u_hi = utility(c=2.0, sigma=8.0)
    assert 0.0 <= u_hi <= u_lo <= 1.0


def test_a_utility_formula_point():
    # U(c,σ)=clip(1 − 0.10/√c − 0.12·log(1+σ²), 0, 1)
    expected = 1.0 - 0.10 / math.sqrt(4.0) - 0.12 * math.log(1.0 + 4.0)
    assert abs(utility(4.0, 2.0) - expected) < 1e-9


def test_a_calibrated_utility_matches_rdp_target():
    eps = 2.0
    # calibrated_utility must use a sigma that realizes the target ε under RDP.
    u = calibrated_utility(c=2.0, eps=eps, T=1)
    # Invert: find sigma by the same bisection contract and re-check utility.
    lo, hi = 1e-3, 1e3
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if rdp_epsilon(mid, 1) > eps:
            lo = mid
        else:
            hi = mid
    sigma = 0.5 * (lo + hi)
    assert abs(rdp_epsilon(sigma, 1) - eps) < 1e-3
    assert abs(u - utility(2.0, sigma)) < 1e-3
