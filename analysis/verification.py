"""Analytical log-normal verification-work model (paper Eq. 15)."""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import lognorm


def log_iv_params(s: float) -> tuple[float, float]:
    """Parameters of log(I_V) ~ N(log(1+0.003 s), 0.18²)."""
    mu = math.log(1.0 + 0.003 * s)
    return mu, 0.18


def verification_density(
    s: float, n: int = 512, n_sigma: float = 3.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Analytical density of I_V on [exp(μ−nσ), exp(μ+nσ)].

    Returns (iv_grid, density) for synthetic panels (Fig. 8).
    """
    mu, sigma = log_iv_params(s)
    # scipy lognorm: shape=σ, scale=exp(μ) for Y=exp(X), X~N(μ,σ²)
    dist = lognorm(s=sigma, scale=math.exp(mu))
    lo = math.exp(mu - n_sigma * sigma)
    hi = math.exp(mu + n_sigma * sigma)
    xs = np.linspace(lo, hi, n)
    dens = dist.pdf(xs)
    # renormalize on the truncated window so trapz ≈ 1
    mass = float(np.trapezoid(dens, xs)) if hasattr(np, "trapezoid") else float(np.trapz(dens, xs))
    if mass > 0:
        dens = dens / mass
    return xs, dens


def verification_median(s: float) -> float:
    """Model median of I_V (white dots in Fig. 8)."""
    mu, _ = log_iv_params(s)
    return math.exp(mu)
