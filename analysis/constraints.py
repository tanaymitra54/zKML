"""Declared constraint and proving-work proxies (paper Eqs. 12–13)."""

from __future__ import annotations

import math
from typing import Sequence

from analysis.accounting import DEFAULT_ORDERS


def constraint_proxy(
    d: int,
    w: int,
    m: int = 64,
    b: int = 8,
    bu: int = 32,
    sigma: float = 2.0,
    orders: Sequence[int] | None = None,
    bind: int = 8192,
) -> dict:
    """Assumed dense-network constraint breakdown (Eq. 12)."""
    if orders is None:
        orders = DEFAULT_ORDERS
    k = math.ceil(6.0 * sigma)
    r_ml = d * w * w
    r_range = 2 * d * w * b
    r_sample = m * (2 * bu + math.ceil(math.log2(2 * k + 1)))
    r_account = 4 * len(orders)
    r_bind = bind
    r = r_ml + r_range + r_sample + r_account + r_bind
    overhead_pct = 100.0 * (r - r_ml) / r_ml if r_ml else 0.0
    return {
        "RML": r_ml,
        "Rrange": r_range,
        "Rsample": r_sample,
        "Raccount": r_account,
        "Rbind": r_bind,
        "R": r,
        "overhead_pct": overhead_pct,
    }


def proving_work_index(R: int, R0: int = 131072) -> float:
    """FFT-dominated proving-work index IP(R) (Eq. 13)."""
    if R <= 1 or R0 <= 1:
        raise ValueError("R and R0 must exceed 1")
    return (R * math.log2(R)) / (R0 * math.log2(R0))
