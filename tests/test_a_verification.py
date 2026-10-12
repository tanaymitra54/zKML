"""Track A: analytical log-normal verification-work model (Eq. 15)."""

import math

import numpy as np

from analysis.verification import log_iv_params, verification_density


def test_a_log_iv_params():
    mu, sigma = log_iv_params(s=128)
    assert abs(mu - math.log(1.0 + 0.003 * 128)) < 1e-12
    assert abs(sigma - 0.18) < 1e-12


def test_a_verification_density_integrates_near_one():
    # Density of IV on truncated support [exp(mu-3σ), exp(mu+3σ)].
    xs, dens = verification_density(s=32, n=512)
    # Trapezoid on linear IV axis; truncate tails as in Fig. 8.
    trapz = getattr(np, "trapezoid", np.trapz)
    mass = float(trapz(dens, xs))
    assert 0.95 < mass < 1.05
