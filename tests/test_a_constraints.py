"""Track A acceptance: constraint proxy and proving-work index (Eq. 12, 13)."""

from analysis.constraints import constraint_proxy, proving_work_index


def test_a_constraint_proxy_default():
    out = constraint_proxy(d=8, w=128)
    assert out["RML"] == 131072
    assert out["Rrange"] == 16384
    assert out["Rsample"] == 4416
    assert out["Raccount"] == 252
    assert out["Rbind"] == 8192
    assert out["R"] == 160316
    assert abs(out["overhead_pct"] - 22.3) < 5e-2


def test_a_proving_work_index_default():
    assert abs(proving_work_index(160316) - 1.244) < 5e-4
