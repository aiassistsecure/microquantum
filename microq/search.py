"""S4: Grover search over records actually stored in NEDB.

The records live in a real NEDB database (separate from the circuit
journal store). The Grover oracle is *built from stored data*: the
record matching the query is read back out of NEDB, its index is
converted to the marked bitstring, and that derived string — never a
literal in the search code — drives ``microq.demos.oracle``.

Honest framing: this is classical simulation of quantum mathematics.
There is no quantum hardware and no speedup — reading every record
from NEDB directly (as ``find_by_tag`` does to locate the match) is
faster than the simulated search. The point is that the oracle is
data-driven from NEDB, and the whole run is journaled, hash-chained
and replay-verifiable like every other microquantum circuit.
"""
from __future__ import annotations

from pathlib import Path

from .circuit import Circuit
from .demos import diffuser, oracle
from .journal import HAS_NEDB, JournalStore

if HAS_NEDB:
    from nedb import NEDB
else:  # pragma: no cover
    NEDB = None

COLLECTION = "records"
N_RECORDS = 16

_NAMES = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
    "hotel", "india", "juliet", "kilo", "lima", "mike", "november",
    "oscar", "papa",
]
_TAGS = [
    "red", "orange", "yellow", "green", "blue", "indigo", "violet",
    "silver", "gold", "copper", "iron", "nickel", "zinc", "tin",
    "lead", "carbon",
]

#: The deterministic record set seeded into NEDB (index 0..15).
RECORDS: list[dict] = [
    {"index": i, "name": _NAMES[i], "tag": _TAGS[i]} for i in range(N_RECORDS)
]


def _clean(doc: dict | None) -> dict | None:
    """Strip NEDB's bookkeeping fields (`_id`/`_seq`/`_hash`/`_coll`)."""
    if doc is None:
        return None
    return {k: v for k, v in doc.items() if not k.startswith("_")}


class RecordStore:
    """A tiny record store on a real NEDB instance.

    Records are documents in the ``records`` collection, keyed by
    ``str(index)``. Reads use ``get()`` over the known sequential keys
    (NEDB exposes no scan/keys method).
    """

    def __init__(self, db_path: str | Path):
        if not HAS_NEDB:
            raise RuntimeError("nedb binding unavailable; RecordStore needs real NEDB")
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = NEDB(str(self.path))

    def seed(self, records: list[dict] | None = None) -> int:
        records = RECORDS if records is None else records
        for rec in records:
            self.db.put(COLLECTION, str(rec["index"]), dict(rec))
        return len(records)

    def get_record(self, index: int) -> dict | None:
        return _clean(self.db.get(COLLECTION, str(index)))

    def all_records(self) -> list[dict]:
        out, i = [], 0
        while (rec := self.get_record(i)) is not None:
            out.append(rec)
            i += 1
        return out

    def find_by(self, field: str, value) -> dict | None:
        """First record read from NEDB whose ``field`` equals ``value``."""
        for rec in self.all_records():
            if rec.get(field) == value:
                return rec
        return None

    def find_by_tag(self, tag: str) -> dict | None:
        return self.find_by("tag", tag)

    def verify(self) -> bool:
        r = self.db.verify()
        return bool(r.get("ok", r) if isinstance(r, dict) else r)


def _as_store(root_or_store) -> RecordStore:
    if isinstance(root_or_store, RecordStore):
        return root_or_store
    return RecordStore(Path(root_or_store) / "records-nedb")


def seed_records(root_or_store, records: list[dict] | None = None) -> int:
    """PUT each record into NEDB under ``str(index)``; returns the count.

    ``root_or_store`` is a RecordStore or a root path (the NEDB database
    then lives at ``<root>/records-nedb``).
    """
    return _as_store(root_or_store).seed(records)


def read_record(root_or_store, index: int) -> dict | None:
    return _as_store(root_or_store).get_record(index)


def read_all_records(root_or_store) -> list[dict]:
    return _as_store(root_or_store).all_records()


def find_by_tag(root_or_store, tag: str) -> dict | None:
    """Find a record by tag by reading records back out of NEDB."""
    return _as_store(root_or_store).find_by_tag(tag)


def _p_search(c: Circuit, marked: str) -> float:
    """Marginal probability of the marked state over search qubits 0..3."""
    return sum(v for k, v in c.probabilities().items() if k[:4] == marked)


def grover_search_nedb(root, target_tag: str, iterations: int = 3) -> dict:
    """Grover-search the NEDB record store for the record tagged
    ``target_tag``.

    Seeds the record DB (idempotent puts), reads the matching record
    back from NEDB, derives ``marked`` from that record's stored index,
    and amplifies it with the shared Grover oracle/diffuser on a
    journaled Circuit (6 qubits: 4 search + 2 ancillas, as in S1).
    """
    root = Path(root)
    store = RecordStore(root / "records-nedb")
    seeded = store.seed()
    record = store.find_by_tag(target_tag)
    if record is None:
        raise ValueError(f"no record in NEDB with tag {target_tag!r}")
    marked = format(int(record["index"]), "04b")

    journal_store = JournalStore(root / "journal")
    c = Circuit(6, store=journal_store)
    for q in range(4):
        c.h(q)
    p_before = _p_search(c, marked)
    for _ in range(iterations):
        oracle(c, marked)
        diffuser(c)
    p_after = _p_search(c, marked)

    shots = c.sample(1000, seed=3)
    top_sample = max(shots.items(), key=lambda kv: kv[1])[0]
    marginals: dict[str, float] = {}
    for k, v in c.probabilities().items():
        marginals[k[:4]] = marginals.get(k[:4], 0.0) + v

    return {
        "record": record,
        "records_seeded": seeded,
        "marked": marked,
        "iterations": iterations,
        "p_marked_before": p_before,
        "p_marked_after": p_after,
        "p_marked": p_after,
        "probabilities": marginals,
        "top_sample": top_sample,
        "journal_length": len(c.journal),
        "replay": c.replay(),
        "record_db_verify": store.verify(),
        "journal_chain_verify": journal_store.verify_chain(),
        "journal_store_verify": journal_store.verify_store(),
        "circuit": c,
    }
