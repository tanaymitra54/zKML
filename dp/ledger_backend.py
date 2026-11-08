"""Release-ledger backends with compare-and-swap transitions (paper Sec. V-E).

The analytical ledger in :mod:`dp.ledger` stores a *proposal* and *commit* as two
separate steps.  This module exposes the smaller, storage-agnostic contract the
paper actually relies on:

    head()                                  -> current root
    commit(expected_parent, record)         -> new root, or raise

``commit`` is a single atomic compare-and-swap.  A transition whose
``expected_parent`` is not the current head is rejected, so two writers that
race for the same predecessor cannot both succeed.  Two reference backends are
provided:

* :class:`SQLiteLedgerBackend` -- durable, uses ``BEGIN IMMEDIATE`` so the
  compare and the insert happen under one write lock.
* :class:`FileLedgerBackend` -- append-only JSON lines guarded by ``flock``.

Both expose the same RFC 6962-style Merkle tree over committed roots, via
:func:`inclusion_proof` and :func:`verify_inclusion_proof`.

This file is additive: it does not change :class:`dp.ledger.ReleaseLedger`, so
the existing single-successor tests remain valid.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from typing import Any, Iterator, Protocol, runtime_checkable

try:  # POSIX advisory locking; the reference environment is macOS/Linux.
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None  # type: ignore[assignment]


GENESIS = "0" * 64


class LedgerError(RuntimeError):
    """Base class for rejected ledger transitions."""


class StaleParentError(LedgerError):
    """``expected_parent`` does not match the current head."""


class DuplicateReleaseError(LedgerError):
    """The release id has already been committed."""


def _canonical(record: dict[str, Any]) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":")).encode()


def record_root(seq: int, parent: str, record: dict[str, Any]) -> str:
    """Content hash linking a record to its predecessor."""
    payload = _canonical({"seq": seq, "parent": parent, "record": record})
    return hashlib.sha256(payload).hexdigest()


def _release_id(record: dict[str, Any]) -> str:
    try:
        rid = record["release_id"]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError("record must contain 'release_id'") from exc
    if not isinstance(rid, str) or not rid:
        raise ValueError("release_id must be a non-empty string")
    return rid


@runtime_checkable
class LedgerBackend(Protocol):
    """Storage contract for a single-successor release ledger."""

    def head(self) -> str:
        """Return the root of the last committed record (``GENESIS`` if empty)."""

    def commit(self, expected_parent: str, record: dict[str, Any]) -> str:
        """Atomically append ``record`` iff ``expected_parent`` is the head."""

    def entries(self) -> list[dict[str, Any]]:
        """Return committed records in sequence order."""


# ---------------------------------------------------------------------------
# SQLite backend
# ---------------------------------------------------------------------------


class SQLiteLedgerBackend:
    """SQLite-backed ledger; CAS is one ``BEGIN IMMEDIATE`` transaction."""

    def __init__(self, path: str):
        self.path = str(path)
        with self._connect() as db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS ledger (
                       seq INTEGER PRIMARY KEY AUTOINCREMENT,
                       release_id TEXT UNIQUE NOT NULL,
                       parent TEXT NOT NULL,
                       root TEXT NOT NULL,
                       record TEXT NOT NULL
                   )"""
            )
            db.execute("PRAGMA journal_mode=WAL")

    def _connect(self) -> sqlite3.Connection:
        # A fresh connection per call keeps use safe across threads; SQLite
        # serialises writers for us once BEGIN IMMEDIATE is taken.
        db = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        db.execute("PRAGMA busy_timeout=30000")
        return db

    def head(self) -> str:
        with self._connect() as db:
            row = db.execute(
                "SELECT root FROM ledger ORDER BY seq DESC LIMIT 1"
            ).fetchone()
        return row[0] if row else GENESIS

    def commit(self, expected_parent: str, record: dict[str, Any]) -> str:
        rid = _release_id(record)
        db = self._connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT root FROM ledger ORDER BY seq DESC LIMIT 1"
            ).fetchone()
            current = row[0] if row else GENESIS
            if current != expected_parent:
                db.execute("ROLLBACK")
                raise StaleParentError(
                    f"expected parent {expected_parent[:12]} but head is {current[:12]}"
                )
            if db.execute(
                "SELECT 1 FROM ledger WHERE release_id=?", (rid,)
            ).fetchone():
                db.execute("ROLLBACK")
                raise DuplicateReleaseError(f"duplicate release id: {rid}")
            seq = db.execute("SELECT COALESCE(MAX(seq),0)+1 FROM ledger").fetchone()[0]
            root = record_root(seq, expected_parent, record)
            db.execute(
                "INSERT INTO ledger (seq, release_id, parent, root, record)"
                " VALUES (?,?,?,?,?)",
                (seq, rid, expected_parent, root, _canonical(record).decode()),
            )
            db.execute("COMMIT")
            return root
        except LedgerError:
            raise
        except Exception:
            if db.in_transaction:
                db.execute("ROLLBACK")
            raise
        finally:
            db.close()

    def entries(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT seq, release_id, parent, root, record FROM ledger ORDER BY seq"
            ).fetchall()
        return [
            {
                "seq": seq,
                "release_id": rid,
                "parent": parent,
                "root": root,
                "record": json.loads(payload),
            }
            for seq, rid, parent, root, payload in rows
        ]


# ---------------------------------------------------------------------------
# Append-only file backend
# ---------------------------------------------------------------------------


class FileLedgerBackend:
    """Append-only JSON-lines ledger guarded by ``flock`` on a sidecar lock."""

    def __init__(self, path: str):
        self.path = str(path)
        parent = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(parent, exist_ok=True)
        self._lock_path = self.path + ".lock"
        open(self.path, "a", encoding="utf-8").close()
        open(self._lock_path, "a", encoding="utf-8").close()

    @contextmanager
    def _locked(self, mode: str) -> Iterator[Any]:
        if fcntl is None:  # pragma: no cover - Windows fallback
            raise RuntimeError("FileLedgerBackend requires POSIX fcntl")
        lock = open(self._lock_path, "a+", encoding="utf-8")
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            yield open(self.path, mode, encoding="utf-8")
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()

    def _read_entries(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    # A torn final write is not a committed record.  The
                    # append-only format never rewrites earlier lines, so this
                    # can only be the last one.
                    break
        return entries

    def head(self) -> str:
        entries = self._read_entries()
        return entries[-1]["root"] if entries else GENESIS

    def commit(self, expected_parent: str, record: dict[str, Any]) -> str:
        rid = _release_id(record)
        with self._locked("a") as fh:
            entries = self._read_entries()
            current = entries[-1]["root"] if entries else GENESIS
            if current != expected_parent:
                raise StaleParentError(
                    f"expected parent {expected_parent[:12]} but head is {current[:12]}"
                )
            if any(e["release_id"] == rid for e in entries):
                raise DuplicateReleaseError(f"duplicate release id: {rid}")
            seq = (entries[-1]["seq"] + 1) if entries else 1
            root = record_root(seq, expected_parent, record)
            fh.write(
                json.dumps(
                    {
                        "seq": seq,
                        "release_id": rid,
                        "parent": expected_parent,
                        "root": root,
                        "record": record,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            fh.flush()
            os.fsync(fh.fileno())
            return root

    def entries(self) -> list[dict[str, Any]]:
        return self._read_entries()


# ---------------------------------------------------------------------------
# RFC 6962 Merkle inclusion proofs over committed roots
# ---------------------------------------------------------------------------


def _leaf_hash(data: bytes) -> bytes:
    return hashlib.sha256(b"\x00" + data).digest()


def _node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(b"\x01" + left + right).digest()


def _split(n: int) -> int:
    """Largest power of two strictly smaller than ``n`` (RFC 6962)."""
    k = 1
    while k * 2 < n:
        k *= 2
    return k


def merkle_tree_hash(leaves: list[bytes]) -> bytes:
    if not leaves:
        return hashlib.sha256(b"").digest()
    if len(leaves) == 1:
        return _leaf_hash(leaves[0])
    k = _split(len(leaves))
    return _node_hash(merkle_tree_hash(leaves[:k]), merkle_tree_hash(leaves[k:]))


def _audit_path(index: int, leaves: list[bytes]) -> list[bytes]:
    if len(leaves) <= 1:
        return []
    k = _split(len(leaves))
    if index < k:
        return _audit_path(index, leaves[:k]) + [merkle_tree_hash(leaves[k:])]
    return _audit_path(index - k, leaves[k:]) + [merkle_tree_hash(leaves[:k])]


def inclusion_proof(backend: LedgerBackend, seq: int) -> dict[str, Any]:
    """Return an RFC 6962 inclusion proof for the record at ``seq``."""
    entries = backend.entries()
    leaves = [bytes.fromhex(e["root"]) for e in entries]
    for i, entry in enumerate(entries):
        if entry["seq"] == seq:
            return {
                "seq": seq,
                "index": i,
                "size": len(leaves),
                "leaf": entry["root"],
                "proof": [p.hex() for p in _audit_path(i, leaves)],
                "root": merkle_tree_hash(leaves).hex(),
            }
    raise KeyError(f"unknown seq: {seq}")


def verify_inclusion_proof(
    leaf_hex: str,
    index: int,
    size: int,
    proof: list[str],
    root_hex: str,
) -> bool:
    """Verify an RFC 6962 audit path without trusting the backend."""
    if index < 0 or index >= size:
        return False
    node = _leaf_hash(bytes.fromhex(leaf_hex))
    fn, sn = index, size - 1
    for sibling_hex in proof:
        sibling = bytes.fromhex(sibling_hex)
        if fn % 2 == 1 or fn == sn:
            node = _node_hash(sibling, node)
            while fn % 2 == 0 and fn != 0:
                fn //= 2
                sn //= 2
        else:
            node = _node_hash(node, sibling)
        fn //= 2
        sn //= 2
    return sn == 0 and node.hex() == root_hex


__all__ = [
    "GENESIS",
    "DuplicateReleaseError",
    "FileLedgerBackend",
    "LedgerBackend",
    "LedgerError",
    "SQLiteLedgerBackend",
    "StaleParentError",
    "inclusion_proof",
    "merkle_tree_hash",
    "record_root",
    "verify_inclusion_proof",
]
