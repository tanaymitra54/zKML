"""Conservative RDP / zCDP accounting (paper Eq. 4, 5).

Interval-valued: `epsilon()` returns (point_estimate, certified_upper_bound).
Budget enforcement must use the upper bound (paper Sec. IV-B A3, V-E).
"""
from __future__ import annotations

import math

DEFAULT_ORDERS = list(range(2, 65))  # A = {2,...,64}, paper Table VI


def rdp_epsilon(sigma: float, T: int, delta: float = 1e-5,
                orders=DEFAULT_ORDERS, delta_f: float = 1.0) -> float:
    """Eq. (4): min over orders of (sum_t r_t(a) + log(1/delta)/(a-1)).

    r_t(a) = a * delta_f^2 / (2 sigma^2)  (unsampled Gaussian, Eq. after (4)).
    """
    if sigma <= 0 or T < 0:
        raise ValueError("sigma>0 and T>=0 required")
    best = math.inf
    per = delta_f * delta_f / (2.0 * sigma * sigma)
    for a in orders:
        r = T * a * per + math.log(1.0 / delta) / (a - 1)
        if r < best:
            best = r
    return best


def zcdp_epsilon(sigma: float, T: int, delta: float = 1e-5, delta_f: float = 1.0) -> float:
    """Eq. (5): rho_tot = sum_t delta_f^2/(2 sigma^2); eps = rho + sqrt(2 rho log(1/delta))."""
    if sigma <= 0 or T < 0:
        raise ValueError("sigma>0 and T>=0 required")
    rho = T * delta_f * delta_f / (2.0 * sigma * sigma)
    return rho + math.sqrt(2.0 * rho * math.log(1.0 / delta))


def max_releases(sigma: float, eps_cap: float, delta: float = 1e-5,
                 orders=DEFAULT_ORDERS, delta_f: float = 1.0) -> int:
    """Largest T with rdp_epsilon(sigma,T) <= eps_cap (linear in T -> closed form)."""
    per = delta_f * delta_f / (2.0 * sigma * sigma)
    best_T = 0
    for a in orders:
        # solve T*a*per + log(1/delta)/(a-1) <= eps_cap  => T <= ...
        rem = eps_cap - math.log(1.0 / delta) / (a - 1)
        if rem < 0:
            continue
        T_a = int(rem / (a * per))
        best_T = max(best_T, T_a)
    # if the min-over-orders bound is what matters, scan down to be exact
    T = best_T + 2
    while T > 0 and rdp_epsilon(sigma, T, delta, orders, delta_f) > eps_cap:
        T -= 1
    return T


class Accountant:
    """Interval-valued ledger of privacy expenditure."""

    def __init__(self, delta: float = 1e-5, orders=DEFAULT_ORDERS,
                 sampler_allowance: float = 0.05):
        self.delta = delta
        self.orders = orders
        self.sampler_allowance = sampler_allowance  # covers finite-sampler error
        self._charges: list[tuple[float, float]] = []  # (sigma, delta_f)

    def charge(self, sigma: float, delta_f: float = 1.0) -> None:
        if sigma <= 0:
            raise ValueError("sigma>0 required")
        self._charges.append((sigma, delta_f))

    def epsilon(self, delta: float | None = None) -> tuple[float, float]:
        d = self.delta if delta is None else delta
        T = len(self._charges)
        if T == 0:
            return (0.0, 0.0)
        # homogeneous charges use the fast path; heterogeneous sum per-order terms
        if len({(s, f) for s, f in self._charges}) == 1:
            sigma, df = self._charges[0]
            point = min(zcdp_epsilon(sigma, T, d, df), rdp_epsilon(sigma, T, d, self.orders, df))
            upper = rdp_epsilon(sigma, T, d, self.orders, df) + self.sampler_allowance
            return (point, upper)
        point, upper = 0.0, self.sampler_allowance
        for order in self.orders:
            acc = sum(a * f * f / (2.0 * s * s) for s, f in self._charges)
            acc += math.log(1.0 / d) / (order - 1)
            upper = min(upper, acc) if order == self.orders[0] else upper
        # conservative: use certificate from the tightest zCDP aggregation
        rho = sum(f * f / (2.0 * s * s) for s, f in self._charges)
        point = rho + math.sqrt(2.0 * rho * math.log(1.0 / d))
        upper = max(point, 0.0) + self.sampler_allowance
        return (point, upper)


if __name__ == "__main__":
    assert abs(rdp_epsilon(4, 10) - 4.106) < 5e-3, rdp_epsilon(4, 10)
    assert abs(rdp_epsilon(4, 40) - 8.838) < 5e-3, rdp_epsilon(4, 40)
    assert max_releases(4, 8) == 33, max_releases(4, 8)
    ac = Accountant()
    for _ in range(10):
        ac.charge(4.0)
    lo, hi = ac.epsilon()
    # both are valid upper bounds; hi is the conservative one used for enforcement
    assert lo <= hi <= 4.2, (lo, hi)
    assert abs(hi - 4.106 - 0.05) < 1e-2, hi
    print("accountant ok:", rdp_epsilon(4, 10), rdp_epsilon(4, 40), max_releases(4, 8), (lo, hi))
