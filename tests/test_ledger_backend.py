"""Track B: ledger backend CAS semantics and RFC 6962 inclusion proofs."""

import threading

import pytest

from dp.ledger_backend import (
    GENESIS,
    DuplicateReleaseError,
    FileLedgerBackend,
    SQLiteLedgerBackend,
    StaleParentError,
    inclusion_proof,
    record_root,
    verify_inclusion_proof,
)


@pytest.fixture(params=["sqlite", "file"])
def backend(request, tmp_path):
    if request.param == "sqlite":
        return SQLiteLedgerBackend(str(tmp_path / "ledger.db"))
    return FileLedgerBackend(str(tmp_path / "ledger.log"))


def test_empty_head_is_genesis(backend):
    assert backend.head() == GENESIS
    assert backend.entries() == []


def test_commit_advances_head_and_links_parent(backend):
    root1 = backend.commit(GENESIS, {"release_id": "A", "eps": 0.5})
    assert backend.head() == root1
    assert root1 == record_root(1, GENESIS, {"release_id": "A", "eps": 0.5})

    root2 = backend.commit(root1, {"release_id": "B", "eps": 0.6})
    entries = backend.entries()
    assert [e["release_id"] for e in entries] == ["A", "B"]
    assert entries[1]["parent"] == root1
    assert backend.head() == root2


def test_stale_parent_rejected(backend):
    root1 = backend.commit(GENESIS, {"release_id": "A", "eps": 0.5})
    with pytest.raises(StaleParentError):
        backend.commit(GENESIS, {"release_id": "B", "eps": 0.6})
    # A rejected transition must not move the head.
    assert backend.head() == root1


def test_duplicate_release_rejected(backend):
    root1 = backend.commit(GENESIS, {"release_id": "A", "eps": 0.5})
    with pytest.raises(DuplicateReleaseError):
        backend.commit(root1, {"release_id": "A", "eps": 0.9})


def test_two_writers_racing_same_parent_exactly_one_wins(backend):
    """The concurrency contract: one CAS succeeds, the other is rejected."""
    barrier = threading.Barrier(2)
    results: list[tuple[str, object]] = []
    lock = threading.Lock()

    def writer(name: str) -> None:
        barrier.wait()
        try:
            root = backend.commit(GENESIS, {"release_id": name, "eps": 1.0})
            outcome: object = ("ok", root)
        except StaleParentError as exc:
            outcome = ("stale", exc)
        with lock:
            results.append((name, outcome))

    threads = [threading.Thread(target=writer, args=(n,)) for n in ("A", "B")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert len(results) == 2
    winners = [r for _, r in results if r[0] == "ok"]
    losers = [r for _, r in results if r[0] == "stale"]
    assert len(winners) == 1
    assert len(losers) == 1
    assert backend.head() == winners[0][1]
    assert len(backend.entries()) == 1


def test_merkle_inclusion_proof_roundtrip(backend):
    roots = []
    for i in range(7):
        parent = backend.head()
        roots.append(backend.commit(parent, {"release_id": f"R{i}", "eps": i}))
    for seq in range(1, 8):
        proof = inclusion_proof(backend, seq)
        assert proof["size"] == 7
        assert verify_inclusion_proof(
            proof["leaf"], proof["index"], proof["size"], proof["proof"], proof["root"]
        )
        # Tampering with any element must invalidate the proof.
        assert not verify_inclusion_proof(
            "00" * 32, proof["index"], proof["size"], proof["proof"], proof["root"]
        )


def test_merkle_single_leaf_and_bad_index(backend):
    backend.commit(GENESIS, {"release_id": "only", "eps": 1.0})
    proof = inclusion_proof(backend, 1)
    assert proof["proof"] == []
    assert verify_inclusion_proof(
        proof["leaf"], 0, 1, proof["proof"], proof["root"]
    )
    assert not verify_inclusion_proof(proof["leaf"], 1, 1, proof["proof"], proof["root"])


def test_merkle_proof_matches_other_backend(tmp_path):
    """Both backends hash the same records to the same tree."""
    a = SQLiteLedgerBackend(str(tmp_path / "a.db"))
    b = FileLedgerBackend(str(tmp_path / "b.log"))
    for i in range(5):
        rec = {"release_id": f"R{i}", "eps": i, "tags": ["x", "y"]}
        ra = a.commit(a.head(), rec)
        rb = b.commit(b.head(), rec)
        assert ra == rb
    assert inclusion_proof(a, 3) == inclusion_proof(b, 3)
