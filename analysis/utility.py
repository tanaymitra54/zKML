"""Synthetic utility proxy and RDP-calibrated coupling (paper Eq. 14)."""

from __future__ import annotations

import math

from analysis.accounting import rdp_epsilon


def utility(c: float, sigma: float) -> float:
    """Bounded synthetic utility score U(c, σ) (Eq. 14)."""
    if c <= 0:
        raise ValueError("c must be positive")
    raw = 1.0 - 0.10 / math.sqrt(c) - 0.12 * math.log(1.0 + sigma * sigma)
    return min(1.0, max(0.0, raw))


def _sigma_for_epsilon(
    eps: float, T: int = 1, delta: float = 1e-5, tol: float = 1e-8
) -> float:
    """Bisection: smallest-noise σ such that rdp_epsilon(σ, T) ≈ eps."""
    if eps <= 0:
        raise ValueError("eps must be positive")
    # Larger σ → smaller ε. Bracket: lo has ε>eps, hi has ε<eps.
    lo, hi = 1e-6, 1e6
    while rdp_epsilon(hi, T, delta=delta) > eps:
        hi *= 2.0
    while rdp_epsilon(lo, T, delta=delta) < eps:
        lo *= 0.5
        if lo < 1e-12:
            break
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if rdp_epsilon(mid, T, delta=delta) > eps:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def calibrated_utility(
    c: float, eps: float, T: int = 1, delta: float = 1e-5
) -> float:
    """Utility at the σ that realizes target ε under RDP (Eq. 4 + 14)."""
    sigma = _sigma_for_epsilon(eps, T=T, delta=delta)
    return utility(c, sigma)
