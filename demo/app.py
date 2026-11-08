"""Streamlit console for the DP-zkML release protocol.

    streamlit run demo/app.py

Four tabs mirror the four protocol roles. Every button calls the shared
`demo.pipeline` core, so the UI cannot drift from the CLI or the tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st

# `streamlit run` puts demo/ on sys.path, not the repo root -- make the core importable.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from quant.reference_query import synthetic_records  # noqa: E402
from demo.pipeline import (  # noqa: E402
    C_Q,
    DIM,
    EPS_CAP,
    N_RECORDS,
    SIGMA,
    ReleaseSession,
    verify_report,
)

st.set_page_config(page_title="DP-zkML Release Console", page_icon="🔒", layout="wide")


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------
def new_session(sigma: float, c_q: float, eps_cap: float) -> ReleaseSession:
    s = ReleaseSession(sigma=sigma, c_q=c_q, dim=DIM, eps_cap=eps_cap)
    s.release(synthetic_records(N_RECORDS, DIM, seed=0, spread=0.5))
    return s


if "session" not in st.session_state:
    st.session_state.session = new_session(SIGMA, C_Q, EPS_CAP)
if "attacks" not in st.session_state:
    st.session_state.attacks = {}


@st.cache_data(show_spinner=False)
def make_records(n: int, seed: int = 0) -> np.ndarray:
    return synthetic_records(n, DIM, seed=seed, spread=0.5)


def _flip(y: np.ndarray) -> np.ndarray:
    out = np.array(y, dtype=np.int64, copy=True)
    out[0] += 1
    return out


session: ReleaseSession = st.session_state.session

# ---------------------------------------------------------------------------
# header
# ---------------------------------------------------------------------------
st.title("DP–zkML · Release Console")
st.caption("quantize → clip → aggregate → sample noise → publish → charge budget → "
           "advance ledger. One release, end to end.")

with st.sidebar:
    st.header("Parameters")
    sigma = st.slider("Noise scale σ", 0.5, 8.0, float(session.sigma), 0.5)
    c_q = st.slider("Clip radius C_Q", 1.0, 32.0, float(session.c_q), 1.0)
    eps_cap = st.slider("Budget cap ε", 1.0, 20.0, float(session.eps_cap), 0.5)
    n_rec = st.slider("Records", 16, 1024, N_RECORDS, 16)

    if st.button("↺ Reset session"):
        st.session_state.session = new_session(sigma, c_q, eps_cap)
        st.session_state.attacks = {}
        st.rerun()

    if st.button("＋ Run next release", type="primary"):
        try:
            st.session_state.session.release(records=make_records(n_rec))
            st.session_state.attacks = {}
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

    st.divider()
    st.caption("Proof backend: **not bound** (Track D). Everything else on this "
               "screen is a real re-derivation of the released artifact.")

rel = session.releases[-1]
budget = session.budget()

# ---------------------------------------------------------------------------
# metric strip
# ---------------------------------------------------------------------------
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("ε (enforced)", f"{rel.eps_upper:.3f}", help="conservative upper end of the interval")
m2.metric("ε remaining", f"{budget['eps_remaining']:.3f}", f"cap {budget['eps_cap']:g}")
m3.metric("Releases", f"{budget['releases_used']}/{budget['releases_allowed']}")
m4.metric("Release id", rel.release_id)
m5.metric("Ledger root", rel.root[:8] + "…", help=rel.root)

tab_in, tab_compute, tab_release, tab_verify = st.tabs(
    ["① Input", "② Compute", "③ Release", "④ Verify"]
)

# ---------------------------------------------------------------------------
# 1 input
# ---------------------------------------------------------------------------
with tab_in:
    st.subheader("Data custodian commits the record set")
    left, right = st.columns([3, 2])
    with left:
        st.caption(f"D ∈ R^({rel.records.shape[0]} × {DIM}) — synthetic, seed 0")
        st.dataframe(rel.records[:12], height=280)
    with right:
        st.markdown("**Public commitments**")
        st.code(
            "\n".join([
                f"C_D     {rel.commitments['C_D'][:40]}…",
                f"C_θ     {rel.commitments['C_theta'][:40]}…",
                f"C_ρ     {rel.commitments['C_rho'][:40]}…",
                "        (witness — hidden from the verifier)",
            ]),
            language="text",
        )
        st.info("Sensitivity certificate is derived from C_Q, not from D — so it "
                "can be approved before any record is seen (assumption A1).",
                icon="ℹ️")

# ---------------------------------------------------------------------------
# 2 compute
# ---------------------------------------------------------------------------
with tab_compute:
    st.subheader("Quantize → clip → aggregate, then add noise")
    a, b = st.columns(2)

    with a:
        st.markdown("**Aggregate z = g_θ(D)**  *(never published)*")
        st.bar_chart({"z": rel.z.tolist()}, height=240, color="#4C78A8")
        st.caption(f"|z|₁ = {int(np.abs(rel.z).sum()):,} · "
                   f"post-clip max ‖v‖₂ = {rel.max_clipped_norm:.2f} ≤ C_Q = {rel.c_q:g} · "
                   f"{rel.clipped}/{rel.records.shape[0]} records clipped")
        st.metric("Replacement sensitivity", f"2·C_Q = {rel.sensitivity:g}")
        st.metric("Sufficient field bound", f"{rel.field_bound:,}")

    with b:
        st.markdown("**Noise η from the discrete Gaussian**")
        hist = {}
        for v in rel.eta.tolist():
            hist[v] = hist.get(v, 0) + 1
        st.bar_chart({"draws": dict(sorted(hist.items()))}, height=240, color="#F58518")
        st.caption(f"σ = {rel.sigma:g} · support "
                   f"[−{int(np.abs(rel.pmf_support).max())}, "
                   f"{int(np.abs(rel.pmf_support).max())}] · "
                   f"pmf sums to {rel.pmf_probs.sum():.12f}")
        st.metric("Excluded tail mass", f"{rel.tail_mass:.2e}", help="bound 1e-12")
        st.info("The tail mass — not a sample-moment test — is what a certified "
                "interval can bound, so it is what the protocol publishes.",
                icon="ℹ️")

    st.markdown("**Published artifact y = Π(z + η)**")
    st.code(str(rel.y[:16].tolist()) + " …", language="python")

# ---------------------------------------------------------------------------
# 3 release
# ---------------------------------------------------------------------------
with tab_release:
    st.subheader("Budget advanced, ledger transitioned")
    a, b = st.columns([3, 2])

    with a:
        rows = [
            {"seq": e[0], "release_id": e[1], "prev_root": e[2][:10] + "…",
             "root": e[3][:10] + "…", "status": e[4]}
            for e in session.ledger.entries()
        ]
        st.dataframe(rows, height=260)
        st.caption("Append-only, single-successor: exactly one transition may "
                   "consume a predecessor state.")

    with b:
        st.markdown("**Privacy expenditure**")
        st.progress(min(1.0, rel.eps_upper / rel.eps_cap),
                    text=f"ε {rel.eps_upper:.3f} of {rel.eps_cap:g}")
        st.caption(f"tighter estimate {rel.eps_point:.3f} · enforced "
                   f"{rel.eps_upper:.3f} · sampler allowance "
                   f"+{rel.eps_upper - rel.eps_point:.3f}")
        st.metric("Releases remaining", budget["releases_remaining"])
        st.code(f"{rel.prev_root[:24]}…\n        ↓\n{rel.root[:24]}…", language="text")
        st.warning("Enforce on the upper end. A crash after publication keeps its "
                   "charge — claiming the answer was never received cannot refund "
                   "privacy budget (paper Table VII).", icon="⚠️")

# ---------------------------------------------------------------------------
# 4 verify
# ---------------------------------------------------------------------------
with tab_verify:
    st.subheader("Re-derive every public check")
    tamper = st.checkbox("Tamper with the artifact before verifying (flip one integer)")

    y = rel.y.copy()
    if tamper:
        y[0] += 1

    checks = session.verify(rel, artifact_y=y)
    n_ok, n_bad, n_open = verify_report(checks)

    c1, c2, c3 = st.columns(3)
    c1.metric("Passed", n_ok)
    c2.metric("Failed", n_bad, delta=None if n_bad == 0 else -n_bad, delta_color="inverse")
    c3.metric("Undecided", n_open)

    icons = {True: "✅", False: "❌", None: "⚠️"}
    st.dataframe(
        [{"": icons[c.ok], "check": c.name, "detail": c.detail} for c in checks],
        height=380, hide_index=True,
    )

    st.divider()
    st.markdown("**Negative tests** — each of these must be rejected")
    a, b, c, d = st.columns(4)
    if a.button("Replay release id"):
        st.session_state.attacks["replay"] = session.attack_replay(rel)
    if b.button("Stale predecessor"):
        st.session_state.attacks["stale"] = session.attack_stale(rel)
    if c.button("Second successor"):
        st.session_state.attacks["branch"] = session.attack_second_child()
    if d.button("Tampered artifact"):
        bad = [ch.name for ch in session.verify(rel, artifact_y=_flip(rel.y)) if ch.ok is False]
        st.session_state.attacks["tamper"] = ", ".join(bad) + " failed"

    for key, msg in st.session_state.attacks.items():
        st.error(f"**{key}** → rejected: {msg}")

st.divider()
st.caption("Metrics shown are analytical/structural, never benchmark timings — "
           "paper Sec. VII-A. Educational demo; not audited cryptographic software.")
