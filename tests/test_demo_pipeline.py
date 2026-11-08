"""Demo-layer regression tests.

These lock the *invariants the video depends on*: one release chains onto the
next, the budget interval only grows, and every attack in paper Table VII is
rejected. If a refactor breaks a beat of the demo, this fails loudly instead of
failing on camera.
"""
from __future__ import annotations

import numpy as np
import pytest

from demo.pipeline import (
    EPS_CAP,
    DIM,
    ReleaseSession,
    verify_report,
)


@pytest.fixture()
def session() -> ReleaseSession:
    return ReleaseSession(sigma=4.0, c_q=8.0, dim=DIM, eps_cap=EPS_CAP)


def test_demo_release_is_well_formed(session: ReleaseSession) -> None:
    rel = session.release(np.random.default_rng(0).normal(0, 1, size=(64, DIM)))
    assert rel.y.shape == (DIM,)
    assert np.array_equal(rel.y, rel.z + rel.eta)
    assert rel.tail_mass <= 1e-12
    assert 0 < rel.eps_upper <= rel.eps_cap
    assert rel.prev_root == "0" * 64


def test_demo_releases_chain_and_budget_is_monotone(session: ReleaseSession) -> None:
    first = session.release()
    second = session.release()
    assert second.prev_root == first.root
    assert second.eps_upper > first.eps_upper
    assert first.releases_used == 1 and second.releases_used == 2


def test_demo_verify_has_no_failures(session: ReleaseSession) -> None:
    rel = session.release()
    ok, bad, undecided = verify_report(session.verify(rel))
    assert bad == 0
    # exactly two checks are honest gaps: the hidden seed witness and the
    # unimplemented proof backend. Anything else would be overclaiming.
    assert undecided == 2
    assert ok >= 8


def test_demo_tampered_artifact_fails(session: ReleaseSession) -> None:
    rel = session.release()
    y = rel.y.copy()
    y[0] += 1
    _, bad, _ = verify_report(session.verify(rel, artifact_y=y))
    assert bad == 1


@pytest.mark.parametrize(
    ("attack", "needle"),
    [
        ("attack_replay", "duplicate release id"),
        ("attack_stale", "stale predecessor"),
        ("attack_second_child", "already consumed"),
    ],
)
def test_demo_attacks_are_rejected(session: ReleaseSession, attack: str, needle: str) -> None:
    rel = session.release()
    msg = getattr(session, attack)(rel) if attack != "attack_second_child" else session.attack_second_child()
    assert needle in msg


def test_demo_budget_exhaustion_fails_closed(session: ReleaseSession) -> None:
    rel = session.release()
    # spend the whole allowance, then one more release must refuse
    for _ in range(rel.releases_allowed - 1):
        session.release()
    with pytest.raises(ValueError, match="budget exhausted"):
        session.release()
