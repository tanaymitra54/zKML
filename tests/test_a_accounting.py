"""Track A acceptance: RDP / zCDP accounting (Eq. 4, 5)."""

from analysis.accounting import max_releases, rdp_epsilon, zcdp_epsilon


def test_a_rdp_epsilon_acceptance():
    assert abs(rdp_epsilon(4, 10) - 4.106) < 5e-4
    assert abs(rdp_epsilon(4, 40) - 8.838) < 5e-4


def test_a_max_releases_acceptance():
    assert max_releases(4, 8) == 33


def test_a_zcdp_epsilon_positive_and_below_rdp_grid():
    # zCDP (Eq. 5) is a compact reference; grid RDP (Eq. 4) is often looser.
    eps_z = zcdp_epsilon(4, 10)
    eps_r = rdp_epsilon(4, 10)
    assert eps_z > 0
    assert eps_z < eps_r
