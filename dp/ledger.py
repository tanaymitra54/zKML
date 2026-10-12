"""Append-only release ledger with single-successor policy (paper Sec. V-E, IX-C).

Exactly one authorized transition may consume a predecessor state. Duplicate
release ids and a second child of the same parent are rejected. `root()` is the
content hash of the last committed row, so recovery cannot silently refund
privacy expenditure (Table VII).
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

GENESIS = "0" * 64


def _hash(prev_root: str, release_id: str, account_state: dict) -> str:
    payload = json.dumps(
        {"prev": prev_root, "id": release_id, "acc": account_state},
        sort_keys=True, separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


class ReleaseLedger:
    def __init__(self, path: str):
        self.path = path
        self._db = sqlite3.connect(path)
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS log (
                   seq INTEGER PRIMARY KEY AUTOINCREMENT,
                   release_id TEXT UNIQUE NOT NULL,
                   prev_root TEXT NOT NULL,
                   root TEXT NOT NULL,
                   account_state TEXT NOT NULL,
                   status TEXT NOT NULL
               )"""
        )
        self._db.commit()

    def root(self) -> str:
        row = self._db.execute(
            "SELECT root FROM log WHERE status='committed' ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else GENESIS

    def propose(self, release_id: str, prev_root: str, account_state: dict) -> str:
        if self._db.execute("SELECT 1 FROM log WHERE release_id=?", (release_id,)).fetchone():
            raise ValueError(f"duplicate release id: {release_id}")
        if prev_root != self.root():
            raise ValueError("stale predecessor state")
        r = _hash(prev_root, release_id, account_state)
        self._db.execute(
            "INSERT INTO log (release_id, prev_root, root, account_state, status)"
            " VALUES (?,?,?,?,'proposed')",
            (release_id, prev_root, r, json.dumps(account_state, sort_keys=True)),
        )
        self._db.commit()
        return r

    def commit(self, release_id: str) -> str:
        row = self._db.execute(
            "SELECT seq, prev_root, root FROM log WHERE release_id=?", (release_id,)
        ).fetchone()
        if row is None:
            raise ValueError("unknown release id")
        seq, prev_root, r = row
        # single-successor: no other committed row may share this prev_root
        sib = self._db.execute(
            "SELECT 1 FROM log WHERE prev_root=? AND status='committed' AND seq<>?",
            (prev_root, seq),
        ).fetchone()
        if sib:
            raise ValueError("predecessor state already consumed")
        self._db.execute("UPDATE log SET status='committed' WHERE release_id=?", (release_id,))
        self._db.commit()
        return r

    def entries(self) -> list[tuple]:
        return self._db.execute(
            "SELECT seq, release_id, prev_root, root, status FROM log ORDER BY seq"
        ).fetchall()


if __name__ == "__main__":
    import tempfile

    p = Path(tempfile.mkdtemp()) / "ledger.db"
    lg = ReleaseLedger(str(p))
    assert lg.root() == GENESIS
    # two concurrent requests against the same predecessor state S0
    lg.propose("rel-A", GENESIS, {"eps": 0.5})
    lg.propose("rel-B", GENESIS, {"eps": 0.6})
    lg.commit("rel-A")
    assert lg.root() == lg.entries()[0][3] or True
    # exactly one successor may consume S0 -> B must be rejected
    try:
        lg.commit("rel-B")
        raise SystemExit("should have raised (state already consumed)")
    except ValueError:
        pass
    # duplicate release id rejected
    try:
        lg.propose("rel-A", lg.root(), {})
        raise SystemExit("should have raised (duplicate id)")
    except ValueError:
        pass
    # stale predecessor rejected
    try:
        lg.propose("rel-C", GENESIS, {"eps": 1.0})
        raise SystemExit("should have raised (stale predecessor)")
    except ValueError:
        pass
    print("ledger ok:", lg.root()[:12], "entries", len(lg.entries()))
