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

    @staticmethod
    def _verify_entries(entries: list[dict]) -> bool:
        """Verify the microquantum hash chain over an ordered entry list
        (a whole journal or any prefix of one)."""
        prev = "0" * 64
        for i, e in enumerate(entries, start=1):
            if e["seq"] != i or e["prev_hash"] != prev:
                return False
            body = {k: v for k, v in e.items() if k != "entry_hash"}
            if entry_hash(body) != e["entry_hash"]:
                return False
            prev = e["entry_hash"]
        return True

    def verify_chain(self) -> bool:
        return self._verify_entries(self.entries)

    # -- provenance reads (AS OF / TRACE, in journal seq) ------------------
    # Seq distinction, stated plainly: every `seq` below is the
    # *microquantum journal seq* (1-based position in this circuit's
    # journal). It is NOT NEDB's global op seq: NEDB's native
    # ``get(coll, id, as_of=N)`` interprets N against NEDB's own
    # store-wide sequence, where writes to other collections interleave,
    # so passing a journal seq there would read the wrong point in time.
    # Journal entries are keyed ``str(journal_seq)`` and each key is
    # written exactly once, so a plain per-key get IS the as-of read for
    # that journal seq, and a prefix of per-key gets plus microquantum
    # chain verification gives correct journal-seq AS OF / TRACE
    # semantics without ever conflating the two sequences.
    def _stored_entry(self, seq: int) -> dict | None:
        """One journal entry sourced from the backing store: a per-key
        NEDB get (``_``-prefixed NEDB fields stripped, as in
        `_load_from_nedb`) when NEDB-backed, else the in-memory/JSONL
        entries. Returns a copy, or None if the key is absent."""
        if self.db is not None:
            doc = self.db.get("journal", str(seq))
            if doc is None:
                return None
            return {k: v for k, v in doc.items() if not k.startswith("_")}
        if 1 <= seq <= len(self.entries):
            return dict(self.entries[seq - 1])
        return None

    @staticmethod
    def _check_journal_seq(seq, head: int) -> None:
        if not isinstance(seq, int) or isinstance(seq, bool):
            raise ValueError(f"journal seq must be an int, got {seq!r}")
        if seq < 0 or seq > head:
            raise ValueError(f"journal seq {seq} out of range 0..{head}")

    def entry_as_of(self, seq: int) -> dict | None:
        """The journal entry as of journal `seq` (see the seq-distinction
        note above: journal seq, not NEDB's global op seq).

        `seq == 0` is genesis — there is no entry yet — and returns
        None. Any other out-of-range seq raises ValueError."""
        self._check_journal_seq(seq, len(self.entries))
        if seq == 0:
            return None
        entry = self._stored_entry(seq)
        if entry is None:  # in range per the mirror, but missing in store
            raise ValueError(f"journal entry {seq} missing from store")
        return entry

    def trace(self, seq: int | None = None) -> dict:
        """TRACE: the ordered chain prefix leading to journal `seq`
        (default: the head), with the prefix's hash chain verified.

        Returns ``{seq, entries, verified, head_hash, state_hash}``.
        `seq == 0` is genesis: an empty prefix, vacuously verified,
        `head_hash` the genesis sentinel ``"0" * 64`` and
        `state_hash` None (no gate has run). Out-of-range raises
        ValueError."""
        if seq is None:
            seq = len(self.entries)
        self._check_journal_seq(seq, len(self.entries))
        entries = []
        for i in range(1, seq + 1):
            entry = self._stored_entry(i)
            if entry is None:
                raise ValueError(f"journal entry {i} missing from store")
            entries.append(entry)
        return {
            "seq": seq,
            "entries": entries,
            "verified": self._verify_entries(entries),
            "head_hash": entries[-1]["entry_hash"] if entries else "0" * 64,
            "state_hash": entries[-1]["state_hash"] if entries else None,
        }

    def verify_store(self) -> bool | None:
        """NEDB's own chain verification, when the real engine backs us."""
        if self.db is None:
            return None
        try:
            r = self.db.verify()
            return bool(r.get("ok", r) if isinstance(r, dict) else r)
        except Exception:
            return None
