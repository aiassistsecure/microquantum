"""P1 provenance reads: journal-seq AS OF / TRACE on JournalStore and
Circuit. The seq used here is the microquantum journal seq, never
NEDB's global op seq (see the note in microq/journal.py)."""
import pytest

from microq import Circuit, JournalStore


@pytest.fixture()
def store(tmp_path):
    return JournalStore(tmp_path / "data")


def _bell_plus(store):
    """Bell pair, snapshot, then two more gates (4 journal entries)."""
    c = Circuit(2, store=store)
    c.h(0).cnot(0, 1)
    mid_hash = c.current_hash()
    c.x(1).rz(0, 0.3)
    return c, mid_hash


def test_entry_as_of_returns_the_entry_at_each_seq(store):
    c, mid_hash = _bell_plus(store)
    e1 = store.entry_as_of(1)
    assert e1["gate"] == "H" and e1["targets"] == [0] and e1["seq"] == 1
    e2 = store.entry_as_of(2)
    assert e2["gate"] == "CNOT"
    # AS OF semantics: seq 2 still reports the state as it was then,
    # even though the circuit has since moved on.
    assert e2["state_hash"] == mid_hash
    assert store.entry_as_of(4)["state_hash"] == c.current_hash()
    # Circuit-level read agrees with the store-level read.
    assert c.as_of(2) == e2
    assert c.as_of(4)["gate"] == "RZ"


def test_as_of_on_a_longer_grover_style_circuit(store):
    c = Circuit(4, store=store)
    for q in range(4):
        c.h(q)
    snap = {e["seq"]: e["state_hash"] for e in c.journal}
    c.x(0).cnot(0, 3).h(2)
    for seq, h in snap.items():
        assert store.entry_as_of(seq)["state_hash"] == h
        assert c.as_of(seq)["state_hash"] == h


def test_genesis_seq_zero(store):
    c, _ = _bell_plus(store)
    assert store.entry_as_of(0) is None
    assert c.as_of(0) is None
    for tr in (store.trace(0), c.trace(0)):
        assert tr["seq"] == 0
        assert tr["entries"] == []
        assert tr["verified"] is True
        assert tr["head_hash"] == "0" * 64
        assert tr["state_hash"] is None


def test_out_of_range_raises(store):
    c, _ = _bell_plus(store)
    for bad in (-1, 5, 100):
        with pytest.raises(ValueError):
            store.entry_as_of(bad)
        with pytest.raises(ValueError):
            store.trace(bad)
        with pytest.raises(ValueError):
            c.as_of(bad)
        with pytest.raises(ValueError):
            c.trace(bad)


def test_trace_returns_full_ordered_verified_prefix(store):
    c, _ = _bell_plus(store)
    tr = c.trace()
    assert tr["seq"] == 4
    assert [e["seq"] for e in tr["entries"]] == [1, 2, 3, 4]
    assert [e["gate"] for e in tr["entries"]] == ["H", "CNOT", "X", "RZ"]
    assert tr["verified"] is True
    assert tr["head_hash"] == c.journal[-1]["entry_hash"]
    assert tr["state_hash"] == c.current_hash()
    # A prefix trace stops where asked and still verifies.
    tr2 = store.trace(2)
    assert [e["gate"] for e in tr2["entries"]] == ["H", "CNOT"]
    assert tr2["verified"] is True
    assert tr2["head_hash"] == c.journal[1]["entry_hash"]


def test_trace_detects_a_tampered_entry(store):
    c, _ = _bell_plus(store)
    store.entries[1]["gate"] = "Z"  # in-memory mirror tamper
    assert store.verify_chain() is False
    if store.db is not None:
        # NEDB-backed: trace() re-reads per-key from the store, so the
        # untampered persisted chain still verifies — the persisted
        # entry was never rewritten (keys are write-once).
        assert store.trace()["verified"] is True
        assert store.entry_as_of(2)["gate"] == "CNOT"
    else:
        # JSONL fallback: the in-memory entries ARE the source, so the
        # tamper shows up in the traced prefix's verification.
        assert store.trace()["verified"] is False


def test_trace_after_fork(store):
    c, _ = _bell_plus(store)
    f = c.fork(2)
    f.x(0)
    tr = f.trace()
    assert [e["gate"] for e in tr["entries"]] == ["H", "CNOT", "X"]
    assert tr["verified"] is True
    # The fork's prefix is exactly the parent's chain prefix.
    assert tr["entries"][:2] == c.trace(2)["entries"]
    # The parent is untouched by the fork's extra gate.
    assert c.trace()["seq"] == 4
    assert f.as_of(2)["state_hash"] == c.as_of(2)["state_hash"]


def test_provenance_survives_reopening_the_store(tmp_path):
    root = tmp_path / "data"
    store = JournalStore(root)
    c, mid_hash = _bell_plus(store)
    expected_entries = [dict(e) for e in c.journal]
    expected_trace = store.trace()

    reopened = JournalStore(root)  # fresh store view of the same root
    assert reopened.entry_as_of(2)["gate"] == "CNOT"
    assert reopened.entry_as_of(2)["state_hash"] == mid_hash
    assert reopened.trace() == expected_trace
    assert reopened.trace()["entries"] == expected_entries
    assert reopened.trace()["verified"] is True


def test_provenance_on_storeless_circuit_is_honest():
    c = Circuit(2)  # no JournalStore: entries carry no hash chain
    c.h(0).cnot(0, 1)
    assert c.as_of(1)["gate"] == "H"
    assert c.as_of(0) is None
    tr = c.trace()
    assert [e["gate"] for e in tr["entries"]] == ["H", "CNOT"]
    assert tr["verified"] is None  # no chain recorded -> no claim made
    with pytest.raises(ValueError):
        c.as_of(3)
