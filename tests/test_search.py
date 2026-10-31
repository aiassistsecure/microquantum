"""S4 tests: Grover search over records actually stored in real NEDB."""
import pytest

from microq.journal import HAS_NEDB
from microq.search import (
    RECORDS,
    RecordStore,
    find_by_tag,
    grover_search_nedb,
    read_all_records,
    read_record,
    seed_records,
)

pytestmark = pytest.mark.skipif(not HAS_NEDB, reason="nedb binding unavailable")


@pytest.fixture()
def store(tmp_path):
    s = RecordStore(tmp_path / "records-nedb")
    seed_records(s)
    return s


def test_records_roundtrip_real_nedb(store, tmp_path):
    assert len(read_all_records(store)) == 16
    for rec in RECORDS:
        assert read_record(store, rec["index"]) == rec
    assert store.verify() is True
    # Re-open the same NEDB path: records persist on disk, not in memory.
    reopened = RecordStore(tmp_path / "records-nedb")
    assert read_all_records(reopened) == RECORDS


def test_find_by_tag_returns_stored_record(store):
    for tag, idx in [("gold", 8), ("red", 0), ("carbon", 15)]:
        rec = find_by_tag(store, tag)
        assert rec == RECORDS[idx]
        assert read_record(store, rec["index"])["tag"] == tag
    assert find_by_tag(store, "no-such-tag") is None


def test_grover_over_nedb_amplifies_derived_marked(tmp_path):
    r = grover_search_nedb(tmp_path / "run", "gold", iterations=3)
    assert r["record"] == RECORDS[8]
    assert r["marked"] == format(8, "04b")
    assert r["p_marked_before"] == pytest.approx(1 / 16, abs=1e-9)
    assert r["p_marked_after"] > 0.9
    assert r["probabilities"][r["marked"]] == pytest.approx(
        r["p_marked_after"], abs=1e-12)


def test_oracle_is_data_driven_two_tags(tmp_path):
    a = grover_search_nedb(tmp_path / "a", "gold", iterations=3)
    b = grover_search_nedb(tmp_path / "b", "violet", iterations=3)
    assert a["marked"] != b["marked"]
    assert b["record"] == RECORDS[6]
    assert b["marked"] == format(6, "04b")
    assert b["p_marked_after"] > 0.9
    # The other query's state is NOT amplified by this oracle.
    assert b["probabilities"][a["marked"]] < 0.1


def test_search_journal_replays_and_verifies(tmp_path):
    r = grover_search_nedb(tmp_path / "run", "blue", iterations=3)
    assert r["journal_length"] > 0
    assert r["replay"] is True
    assert r["journal_chain_verify"] is True
    assert r["record_db_verify"] is True
    assert r["journal_store_verify"] is True
