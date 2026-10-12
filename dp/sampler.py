"""Exact-on-support discrete Gaussian sampler (paper Sec. V-C, IX-B).

The paper's point (Sec. IX-B): for the *implemented* mechanism, what matters is
the exact probability mass function over the finite support and the excluded
tail mass -- not a sample-moment test. So we build the pmf explicitly and sample
from it; the tail mass is returned so a certified interval can bound it.
"""
from __future__ import annotations

import math

import numpy as np


def _support_radius(sigma: float, tail: float = 1e-12) -> int:
    # smallest R with P(|X|>R) below `tail` for the *truncated* build below.
    R = int(math.ceil(sigma * math.sqrt(2 * math.log(1.0 / tail))))
    return max(R, 1)


def sampler_pmf(sigma: float, tail: float = 1e-12) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (values, probabilities, excluded_tail_mass).

    Probabilities are exact over the finite support; excluded_tail_mass is the
    unnormalized mass dropped beyond the support (a conservative accounting knob).
    """
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    R = _support_radius(sigma, tail)
    x = np.arange(-R, R + 1)
    w = np.exp(-(x.astype(float) ** 2) / (2.0 * sigma * sigma))
    total_all = w.sum()
    # normalizing denominator is over the finite support -> exact implemented pmf
    p = w / total_all
    # tail mass the finite-support build excludes, relative to the infinite law
    # (infinite partition sum via theta identity series)
    z_inf = 1.0 + 2.0 * sum(math.exp(-(k * k) / (2.0 * sigma * sigma)) for k in range(1, 6000))
    excluded = max(0.0, 1.0 - total_all / z_inf)
    return x, p, excluded


def discrete_gaussian(sigma: float, size: int, rng: np.random.Generator) -> np.ndarray:
    x, p, _ = sampler_pmf(sigma)
    return rng.choice(x, size=size, p=p)


def sampler_pmf_enumerated(bit_width: int, sigma: float) -> dict[int, float]:
    """Exhaustive enumeration for small bit widths (paper Sec. IX-B)."""
    _, p, _ = sampler_pmf(sigma)
    lo, hi = -(2 ** (bit_width - 1)), 2 ** (bit_width - 1) - 1
    x = np.arange(lo, hi + 1)
    # renormalize the same Gaussian weights on the encoded 8/other-bit domain
    w = np.exp(-(x.astype(float) ** 2) / (2.0 * sigma * sigma))
    w = w / w.sum()
    return {int(k): float(v) for k, v in zip(x, w)}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    s = discrete_gaussian(2.0, 200_000, rng)
    assert abs(s.mean()) < 0.03, s.mean()
    assert abs(s.var() - 4.0) < 0.15, s.var()
    _, p, tail = sampler_pmf(2.0)
    assert abs(p.sum() - 1.0) < 1e-12
    assert tail < 1e-9
    print("sampler ok: mean", round(s.mean(), 4), "var", round(s.var(), 4), "tail", tail)
