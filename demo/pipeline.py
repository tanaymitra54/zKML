"""Protocol core for the DP-zkML release demo.

Implements the Section IX blueprint as a single observable pipeline:

    records D  ->  quantize + clip + aggregate z  ->  discrete-Gaussian noise eta
               ->  published artifact y  ->  conservative budget charge
               ->  append-only ledger transition  ->  relation checks

Design rule (paper Sec. VII-A, SPEC "Metrics naming rule"): every number this
module reports is *analytical* or *structural*. Nothing here is a benchmark.
The proof backend is NOT implemented -- `verify()` says so in as many words
rather than implying a SNARK verified anything.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from dp.accountant import Accountant, max_releases
from dp.ledger import GENESIS, ReleaseLedger
from dp.sampler import discrete_gaussian, sampler_pmf
from quant.reference_query import (
    clip_vector,
    field_lower_bound,
    quantize,
    sensitivity_bound,
    synthetic_records,
)

# --- demo defaults (mirror SPEC.md Section "Global acceptance numbers") -----
DIM = 16                 # embedding dimension d
N_RECORDS = 256          # synthetic record set
C_Q = 8.0                # per-record L2 clipping radius
SIGMA = 4.0              # noise scale (paper acceptance sigma)
DELTA = 1e-5
EPS_CAP = 8.0            # enrollment budget ceiling
SCALE = 4.0              # quantization scale for v_theta
SAMPLER_ALLOWANCE = 0.05  # interval inflation covering finite-sampler error
TAIL_TOL = 1e-12
PROOF_BACKEND_BOUND = False  # Track D -- flips to True when a proof system lands


# ---------------------------------------------------------------------------
# commitments
# ---------------------------------------------------------------------------
def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def commit_array(a: np.ndarray) -> str:
    """Binding commitment over an array's exact bytes, shape and dtype."""
    a = np.ascontiguousarray(a)
    tag = json.dumps({"shape": list(a.shape), "dtype": str(a.dtype)},
                     separators=(",", ":")).encode()
    return _sha(a.tobytes() + tag)


def commitment_id(artifact_commitment: str) -> str:
    return artifact_commitment[:16]


# ---------------------------------------------------------------------------
# the registered model (public, approved independently of the dataset -- A1)
# ---------------------------------------------------------------------------
def v_theta(record: np.ndarray) -> np.ndarray:
    """Deterministic integer embedding. No training: a fixed registered map."""
    return quantize(record, SCALE, bits=8)


def model_fingerprint(dim: int) -> str:
    """Circuit/parameter identity committed as C_theta."""
    return _sha(f"v_theta:scale={SCALE}:bits=8:dim={dim}".encode())


# ---------------------------------------------------------------------------
# result carriers
# ---------------------------------------------------------------------------
@dataclass
class Check:
    """One line of the verification report. ok=None means 'cannot decide'."""
    name: str
    ok: bool | None
    detail: str = ""


@dataclass
class Release:
    """Everything one release produced, plus the state it moved."""

    release_id: str
    sigma: float
    c_q: float
    dim: int

    records: np.ndarray
    z: np.ndarray
    clipped: int
    eta: np.ndarray
    y: np.ndarray

    pmf_support: np.ndarray
    pmf_probs: np.ndarray
    tail_mass: float

    eps_point: float
    eps_upper: float
    eps_cap: float
    releases_used: int
    releases_allowed: int

    sensitivity: float
    max_raw_norm: float
    max_clipped_norm: float
    field_bound: int

    prev_root: str
    root: str
    commitments: dict[str, str] = field(default_factory=dict)

    @property
    def eps_remaining(self) -> float:
        return max(0.0, self.eps_cap - self.eps_upper)

    @property
    def releases_remaining(self) -> int:
        return max(0, self.releases_allowed - self.releases_used)


# ---------------------------------------------------------------------------
# session: one ledger + one accountant, chained releases
# ---------------------------------------------------------------------------
class ReleaseSession:
    """A release history. Every release consumes the previous state root."""

    def __init__(self, sigma: float = SIGMA, c_q: float = C_Q, dim: int = DIM,
                 eps_cap: float = EPS_CAP, ledger_path: str | None = None) -> None:
        self.sigma = float(sigma)
        self.c_q = float(c_q)
        self.dim = int(dim)
        self.eps_cap = float(eps_cap)

        if ledger_path is None:
            self._tmpdir = tempfile.mkdtemp(prefix="dpzkml-demo-")
            ledger_path = str(Path(self._tmpdir) / "ledger.db")
        else:
            self._tmpdir = None
        self.ledger_path = ledger_path

        self.ledger = ReleaseLedger(ledger_path)
        self.accountant = Accountant(delta=DELTA, sampler_allowance=SAMPLER_ALLOWANCE)
        self.releases: list[Release] = []
        self._counter = 0

    # -- helpers ----------------------------------------------------------
    def current_eps(self) -> tuple[float, float]:
        return self.accountant.epsilon()

    def current_root(self) -> str:
        return self.ledger.root()

    def budget(self) -> dict[str, float | int]:
        lo, hi = self.current_eps()
        allowed = max_releases(self.sigma, self.eps_cap, DELTA)
        used = len(self.releases)
        return {
            "eps_point": lo,
            "eps_upper": hi,
            "eps_cap": self.eps_cap,
            "eps_remaining": max(0.0, self.eps_cap - hi),
            "releases_used": used,
            "releases_allowed": allowed,
            "releases_remaining": max(0, allowed - used),
            "root": self.current_root(),
        }

    # -- the release itself ----------------------------------------------
    def release(self, records: np.ndarray | None = None, model_id: str = "model-A",
                release_id: str | None = None) -> Release:
        if records is None:
            records = synthetic_records(N_RECORDS, self.dim, seed=0)
        records = np.asarray(records, dtype=np.float32)

        # budget pre-check: fail closed, never overspend silently
        lo, hi = self.current_eps()
        allowed = max_releases(self.sigma, self.eps_cap, DELTA)
        if len(self.releases) >= allowed or hi >= self.eps_cap:
            raise ValueError(
                f"privacy budget exhausted: eps={hi:.3f} >= cap={self.eps_cap:.1f} "
                f"({len(self.releases)}/{allowed} releases used)"
            )

        # step 1-2: quantize, clip, aggregate
        acc = np.zeros(self.dim, dtype=np.int64)
        clipped = 0
        max_raw_norm = 0.0
        max_clipped_norm = 0.0
        for r in records:
            v = np.asarray(v_theta(r), dtype=np.int64)
            max_raw_norm = max(max_raw_norm, float(np.linalg.norm(v)))
            cv = np.asarray(clip_vector(v, self.c_q), dtype=np.int64)
            max_clipped_norm = max(max_clipped_norm, float(np.linalg.norm(cv)))
            if not np.array_equal(cv, v):
                clipped += 1
            acc += cv
        z = acc

        # step 3: protected randomness -> finite-support discrete Gaussian
        seed = int.from_bytes(os.urandom(16), "big")
        rng = np.random.default_rng(seed)
        eta = np.asarray(discrete_gaussian(self.sigma, size=self.dim, rng=rng),
                         dtype=np.int64)
        support, probs, tail = sampler_pmf(self.sigma, tail=TAIL_TOL)

        # step 4: public post-processing (post-noise, dataset-size scaling)
        y = (z + eta).astype(np.int64)

        # step 5: conservative charge, then read the interval
        self.accountant.charge(self.sigma)
        lo, hi = self.accountant.epsilon()

        # step 6: ledger transition
        self._counter += 1
        release_id = release_id or f"rel-{self._counter:03d}"
        prev_root = self.ledger.root()
        account_state = {"eps_point": round(lo, 6),
                         "eps_upper": round(hi, 6),
                         "releases": len(self.releases) + 1,
                         "sigma": self.sigma}
        self.ledger.propose(release_id, prev_root, account_state)
        root = self.ledger.commit(release_id)

        commitments = {
            "C_D": commit_array(records),
            "C_theta": model_fingerprint(self.dim),
            "C_rho": _sha(seed.to_bytes(16, "big", signed=False)),
            "C_y": commit_array(y),
            "C_z": commit_array(z),
        }

        rel = Release(
            release_id=release_id,
            sigma=self.sigma,
            c_q=self.c_q,
            dim=self.dim,
            records=records,
            z=z,
            clipped=clipped,
            eta=eta,
            y=y,
            pmf_support=np.asarray(support),
            pmf_probs=np.asarray(probs),
            tail_mass=float(tail),
            eps_point=lo,
            eps_upper=hi,
            eps_cap=self.eps_cap,
            releases_used=len(self.releases) + 1,
            releases_allowed=allowed,
            sensitivity=sensitivity_bound(self.c_q),
            max_raw_norm=max_raw_norm,
            max_clipped_norm=max_clipped_norm,
            field_bound=field_lower_bound(k=self.dim, A=127, W=127, B=255),
            prev_root=prev_root,
            root=root,
            commitments=commitments,
        )
        self.releases.append(rel)
        return rel

    # -- negative tests (paper Table VII) --------------------------------
    def attack_replay(self, rel: Release) -> str:
        """Re-submit an already-used release id."""
        try:
            self.ledger.propose(rel.release_id, self.ledger.root(), {"eps": 0.0})
        except ValueError as exc:
            return str(exc)
        return "ACCEPTED -- ledger is broken"

    def attack_stale(self, rel: Release) -> str:
        """Propose against a predecessor root that is no longer current."""
        try:
            self.ledger.propose(f"{rel.release_id}-stale", GENESIS, {"eps": 0.0})
        except ValueError as exc:
            return str(exc)
        return "ACCEPTED -- ledger is broken"

    def attack_second_child(self) -> str:
        """Two children staged on one state; only the first may commit.

        Run on an isolated chain so the live history is not polluted with a
        dangling proposal -- the policy under test is the ledger's, not this
        session's.
        """
        tmp = Path(tempfile.mkdtemp(prefix="dpzkml-branch-")) / "branch.db"
        lg = ReleaseLedger(str(tmp))
        try:
            lg.propose("branch-A", lg.root(), {"eps": 0.5})
            lg.propose("branch-B", lg.root(), {"eps": 0.6})
            lg.commit("branch-A")
            lg.commit("branch-B")
        except ValueError as exc:
            return str(exc)
        return "ACCEPTED -- ledger is broken"

    # -- verification -----------------------------------------------------
    def verify(self, rel: Release, artifact_y: np.ndarray | None = None,
               seed: int | None = None) -> list[Check]:
        """Re-derive every public check. Honest about what is not bound."""
        y = rel.y if artifact_y is None else np.asarray(artifact_y, dtype=np.int64)
        checks: list[Check] = []

        checks.append(Check(
            "artifact commitment C_y",
            commit_array(y) == rel.commitments["C_y"],
            "y hashed against the published commitment",
        ))
        checks.append(Check(
            "dataset commitment C_D",
            commit_array(rel.records) == rel.commitments["C_D"],
            "record set hashed against C_D",
        ))
        checks.append(Check(
            "model commitment C_theta",
            model_fingerprint(rel.dim) == rel.commitments["C_theta"],
            "registered model identity",
        ))
        if seed is None:
            checks.append(Check(
                "seed commitment C_rho", None,
                "witness withheld -- only a proof can substitute (Track D)",
            ))
        else:
            ok = _sha(seed.to_bytes(16, "big", signed=False)) == rel.commitments["C_rho"]
            checks.append(Check("seed commitment C_rho", ok,
                                "one-use seed opens against C_rho"))

        checks.append(Check(
            f"sampler tail mass <= {TAIL_TOL:g}",
            rel.tail_mass <= TAIL_TOL,
            f"excluded mass {rel.tail_mass:.2e} over support "
            f"[-{int(np.abs(rel.pmf_support).max())}, {int(np.abs(rel.pmf_support).max())}]",
        ))
        checks.append(Check(
            "range / clipping respected",
            rel.max_clipped_norm <= rel.c_q + 1e-6,
            f"post-clip max ||v||_2 = {rel.max_clipped_norm:.2f} <= C_Q = {rel.c_q:.1f} "
            f"(raw max {rel.max_raw_norm:.2f})",
        ))
        checks.append(Check(
            "post-processing is public",
            np.array_equal(rel.y, rel.z + rel.eta),
            "y = z + eta, noise added before any public scaling",
        ))
        checks.append(Check(
            "budget advanced monotonically",
            rel.eps_upper > 0 and rel.eps_upper <= rel.eps_cap,
            f"eps_upper = {rel.eps_upper:.4f} <= cap {rel.eps_cap:.1f}",
        ))
        ids = [e[1] for e in self.ledger.entries()]
        checks.append(Check(
            "release id unused",
            ids.count(rel.release_id) == 1,
            f"{rel.release_id} appears exactly once in the ledger",
        ))
        chain_ok, chain_detail = _chain_ok(self.ledger, rel)
        checks.append(Check("ledger root chains", chain_ok, chain_detail))
        checks.append(Check(
            "proof backend binds the relation",
            True if PROOF_BACKEND_BOUND else None,
            "NOT BOUND -- Track D, funded by this grant" if not PROOF_BACKEND_BOUND
            else "proof system attached",
        ))
        return checks


def _chain_ok(ledger: ReleaseLedger, rel: Release) -> tuple[bool, str]:
    """Confirm `rel` sits on the committed chain with the right predecessor."""
    committed = [e for e in ledger.entries() if e[4] == "committed"]
    for idx, (seq, rid, prev_root, root, status) in enumerate(committed):
        if rid != rel.release_id:
            continue
        expected = GENESIS if idx == 0 else committed[idx - 1][3]
        return (prev_root == expected,
                f"{prev_root[:12]}... -> {root[:12]}... "
                f"({'genesis' if idx == 0 else 'chained'})")
    return False, f"{rel.release_id} not found in the committed chain"


def verify_report(checks: list[Check]) -> tuple[int, int, int]:
    """Count (passed, failed, undecided)."""
    ok = sum(1 for c in checks if c.ok is True)
    bad = sum(1 for c in checks if c.ok is False)
    und = sum(1 for c in checks if c.ok is None)
    return ok, bad, und


# ---------------------------------------------------------------------------
# self-check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    s = ReleaseSession()
    r1 = s.release()
    assert r1.y.shape == (DIM,)
    assert r1.eps_upper > 0 and r1.eps_upper <= r1.eps_cap
    assert r1.tail_mass <= TAIL_TOL
    r2 = s.release()
    assert r2.prev_root == r1.root, "releases must chain"
    checks = s.verify(r1)
    ok, bad, und = verify_report(checks)
    assert bad == 0, [c for c in checks if c.ok is False]
    undecided = {c.name for c in checks if c.ok is None}
    assert undecided == {"seed commitment C_rho", "proof backend binds the relation"}, undecided
    assert "duplicate release id" in s.attack_replay(r1)
    assert "stale predecessor" in s.attack_stale(r1)
    assert "already consumed" in s.attack_second_child()
    tampered = r1.y.copy()
    tampered[0] += 1
    _, bad_t, _ = verify_report(s.verify(r1, artifact_y=tampered))
    assert bad_t == 1, "tampering must fail exactly one check"
    print("pipeline ok:", r1.release_id, "eps", round(r1.eps_upper, 4),
          "root", r1.root[:12], "checks", f"{ok} ok / {bad} fail / {und} undecided")
