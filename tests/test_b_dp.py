import numpy as np
import pytest

from dp.accountant import Accountant, max_releases, rdp_epsilon, zcdp_epsilon
from dp.ledger import GENESIS, ReleaseLedger
from dp.sampler import discrete_gaussian, sampler_pmf


def test_paper_accounting_numbers():
    assert rdp_epsilon(4, 10) == pytest.approx(4.106, abs=5e-3)
    assert rdp_epsilon(4, 40) == pytest.approx(8.838, abs=5e-3)
    assert max_releases(4, 8) == 33


def test_monotone_and_conservative():
    assert rdp_epsilon(2, 10) > rdp_epsilon(8, 10)   # less noise -> more loss
    assert rdp_epsilon(4, 20) > rdp_epsilon(4, 10)   # more releases -> more loss
    assert zcdp_epsilon(4, 10) <= rdp_epsilon(4, 10)


def test_accountant_interval():
    ac = Accountant()
    assert ac.epsilon() == (0.0, 0.0)
    for _ in range(10):
        ac.charge(4.0)
    lo, hi = ac.epsilon()
    assert lo <= hi <= 4.2
    assert hi == pytest.approx(4.106 + ac.sampler_allowance, abs=1e-2)


def test_sampler_exact_pmf_and_moments():
    x, p, tail = sampler_pmf(2.0)
    assert p.sum() == pytest.approx(1.0)
    assert tail < 1e-9
    s = discrete_gaussian(2.0, 200_000, np.random.default_rng(0))
    assert s.mean() == pytest.approx(0.0, abs=0.05)
    assert s.var() == pytest.approx(4.0, abs=0.2)


def test_ledger_single_successor(tmp_path):
    lg = ReleaseLedger(str(tmp_path / "l.db"))
    assert lg.root() == GENESIS
    lg.propose("A", GENESIS, {"eps": 0.5})
    lg.propose("B", GENESIS, {"eps": 0.6})
    lg.commit("A")
    with pytest.raises(ValueError):
        lg.commit("B")                       # state already consumed
    with pytest.raises(ValueError):
        lg.propose("A", lg.root(), {})       # duplicate id
    with pytest.raises(ValueError):
        lg.propose("C", GENESIS, {})         # stale predecessor
