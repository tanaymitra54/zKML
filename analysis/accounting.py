"""RDP and zCDP privacy accounting (paper Eqs. 4–5)."""

from __future__ import annotations

import math
from typing import Sequence

DEFAULT_ORDERS: list[int] = list(range(2, 65))


def rdp_epsilon(
    sigma: float,
    T: int,
    delta: float = 1e-5,
    orders: Sequence[int] | None = None,
    delta_f: float = 1.0,
) -> float:
    """Grid RDP→(ε,δ) bound for T unsampled Gaussian releases (Eq. 4)."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if T < 0:
        raise ValueError("T must be non-negative")
    if orders is None:
        orders = DEFAULT_ORDERS
    best = float("inf")
    log_inv_delta = math.log(1.0 / delta)
    for alpha in orders:
        rho = T * (alpha * delta_f * delta_f) / (2.0 * sigma * sigma)
        eps = rho + log_inv_delta / (alpha - 1)
        if eps < best:
            best = eps
    return best


def max_releases(
    sigma: float, eps_cap: float, delta: float = 1e-5
) -> int:
    """Largest T with rdp_epsilon(sigma, T) <= eps_cap."""
    t = 0
    while rdp_epsilon(sigma, t + 1, delta=delta) <= eps_cap:
        t += 1
    return t


def zcdp_epsilon(
    sigma: float,
    T: int,
    delta: float = 1e-5,
    delta_f: float = 1.0,
) -> float:
    """zCDP sufficient (ε,δ) parameter for T Gaussian releases (Eq. 5)."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    rho_tot = T * (delta_f * delta_f) / (2.0 * sigma * sigma)
    return rho_tot + math.sqrt(2.0 * rho_tot * math.log(1.0 / delta))
