"""Journal persistence with NEDB journal semantics.

Primary backend (slice 1, verified working on this VM): the real NEDB
engine via the PyPI `nedb-engine` binding (`NEDB(path)`). Journal entries
are documents in the `journal` collection; NEDB seals every write in its
own BLAKE2b hash chain (`_seq` / `_hash` on each put, `db.verify()`), and
each entry additionally carries the microquantum chain fields
(`prev_hash` / `entry_hash`) so the circuit journal is self-verifying
without trusting the store.

Checkpoints are NOT rows: mirroring SI's design rule ("weights live as
immutable content-addressed blobs, not rows"), state checkpoints are
content-addressed `.npy` blob files named by the BLAKE2b hash of the state
bytes, written once and never duplicated.

Fallback: if `nedb` cannot be imported, the journal degrades to an
append-only JSONL file with the identical entry schema and hash chain.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

try:
    from nedb import NEDB
    HAS_NEDB = True
except Exception:  # pragma: no cover
    NEDB = None
    HAS_NEDB = False


def state_hash(psi: np.ndarray) -> str:
    return hashlib.blake2b(np.asarray(psi, dtype=np.complex128).tobytes(), digest_size=32).hexdigest()


def entry_hash(entry_without_hash: dict) -> str:
    blob = json.dumps(entry_without_hash, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.blake2b(blob, digest_size=32).hexdigest()


class JournalStore:
    """Journal (real NEDB if available, else JSONL) + content-addressed
    checkpoint blob store."""

    backend = "nedb" if HAS_NEDB else "jsonl"

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.ckpt_dir = self.root / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.journal_path = self.root / "journal.jsonl"
        self.entries: list[dict] = []
        self.db = None
        if HAS_NEDB:
            self.db = NEDB(str(self.root / "nedb"))
            self._load_from_nedb()
        elif self.journal_path.exists():
            with open(self.journal_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.entries.append(json.loads(line))

    def _load_from_nedb(self) -> None:
        seq = 1
        while True:
            doc = self.db.get("journal", str(seq))
            if doc is None:
                break
            self.entries.append({k: v for k, v in doc.items() if not k.startswith("_")})
            seq += 1

    # -- checkpoints (content-addressed blobs, deduplicated) --------------
    def put_checkpoint(self, psi: np.ndarray) -> str:
        h = state_hash(psi)
        path = self.ckpt_dir / f"{h}.npy"
        if not path.exists():  # never duplicate identical state bytes
            tmp = path.with_suffix(".tmp.npy")
            np.save(tmp, np.asarray(psi, dtype=np.complex128))
            os.replace(tmp, path)
        return h

    def get_checkpoint(self, h: str) -> np.ndarray:
        return np.load(self.ckpt_dir / f"{h}.npy")

    def has_checkpoint(self, h: str) -> bool:
        return (self.ckpt_dir / f"{h}.npy").exists()

    # -- journal ----------------------------------------------------------
    def _persist(self, entry: dict) -> None:
        if self.db is not None:
            self.db.put("journal", str(entry["seq"]), dict(entry))
        else:
            with open(self.journal_path, "a") as f:
                f.write(json.dumps(entry) + "\n")

    def append(self, gate: str, targets: list[int], params: list[float], psi: np.ndarray) -> dict:
        seq = len(self.entries) + 1
        prev_hash = self.entries[-1]["entry_hash"] if self.entries else "0" * 64
        entry = {
            "seq": seq,
            "gate": gate,
            "targets": list(targets),
            "params": list(params),
            "state_hash": self.put_checkpoint(psi),
            "prev_hash": prev_hash,
            "timestamp": time.time(),
        }
        entry["entry_hash"] = entry_hash(entry)
        self._persist(entry)
        self.entries.append(entry)
        return entry

    def import_entry(self, entry: dict) -> None:
        """Persist an existing entry unchanged (used by fork to inherit the
        parent's chain prefix into the child's own store)."""
        self._persist(dict(entry))
        self.entries.append(dict(entry))

    def verify_chain(self) -> bool:
        prev = "0" * 64
        for i, e in enumerate(self.entries, start=1):
            if e["seq"] != i or e["prev_hash"] != prev:
                return False
            body = {k: v for k, v in e.items() if k != "entry_hash"}
            if entry_hash(body) != e["entry_hash"]:
                return False
            prev = e["entry_hash"]
        return True

    def verify_store(self) -> bool | None:
        """NEDB's own chain verification, when the real engine backs us."""
        if self.db is None:
            return None
        try:
            r = self.db.verify()
            return bool(r.get("ok", r) if isinstance(r, dict) else r)
        except Exception:
            return None
