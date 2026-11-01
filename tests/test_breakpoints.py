"""D3 debugger: breakpoints / watchpoints (run_with_watchpoints)."""
import numpy as np
import pytest

from microq import (Circuit, EntropyWatch, FidelityWatch, JournalStore,
                    ProbabilityWatch, run_with_watchpoints)

BELL_STEPS = [("H", 0), ("CNOT", (0, 1)), ("X", 0)]


@pytest.fixture()
def store(tmp_path):
    return JournalStore(tmp_path / "data")


def test_entropy_watch_trips_exactly_at_cnot(store):
    # After H (seq 1) entropy is 0; only the CNOT (seq 2) entangles to
    # 1.0 bit — trip there, not seq 1; trailing X never applied.
    c = Circuit(2, store=store)
    r = run_with_watchpoints(c, BELL_STEPS, [EntropyWatch(0, 0.5)])
    assert r["tripped"] is True
    assert r["seq"] == 2 and r["final_seq"] == 2
    assert r["condition"]["kind"] == "entropy"
    assert r["condition"]["qubit"] == 0
    assert r["observed_value"] == pytest.approx(1.0, abs=1e-9)
    assert "entropy" in r["condition"]["description"]
    assert len(c.journal) == 2  # halt: journal length == tripped seq
    assert r["steps_applied"] == 2
    rep = r["inspect"]
    assert rep["seq"] == 2
    assert rep["probabilities"]["00"] == pytest.approx(0.5, abs=1e-9)
    assert rep["probabilities"]["11"] == pytest.approx(0.5, abs=1e-9)
    assert r["report"] is rep


def test_probability_watch_trips_at_exact_seq(store):
    # P(|11>): 0 after H, 0.5 after CNOT — threshold 0.5 trips at seq 2.
    c = Circuit(2, store=store)
    r = run_with_watchpoints(c, BELL_STEPS[:2], [ProbabilityWatch("11", 0.5)])
    assert r["tripped"] is True and r["seq"] == 2
    assert r["condition"]["kind"] == "probability"
    assert r["condition"]["bitstring"] == "11"
    assert r["observed_value"] == pytest.approx(0.5, abs=1e-9)


def test_fidelity_watch_trips_and_halts(store):
    # Fidelity vs |00> after H is 0.5 < 0.9: trips seq 1, CNOT unapplied.
    c = Circuit(2, store=store)
    target = np.array([1, 0, 0, 0], dtype=complex)
    r = run_with_watchpoints(c, BELL_STEPS[:2], [FidelityWatch(target, 0.9)])
    assert r["tripped"] is True and r["seq"] == 1
    assert r["condition"]["kind"] == "fidelity"
    assert r["observed_value"] == pytest.approx(0.5, abs=1e-9)
    assert len(c.journal) == 1


def test_no_trip_control_runs_all_steps(store):
    # Entropy reaches exactly 1.0 (never above); P(|11>) 0.5 < 0.75.
    c = Circuit(2, store=store)
    conds = [EntropyWatch(0, 1.0), ProbabilityWatch("11", 0.75)]
    r = run_with_watchpoints(c, BELL_STEPS[:2], conds)
    assert r["tripped"] is False and r["seq"] is None
    assert r["condition"] is None and r["event"] is None
    assert len(c.journal) == 2 and r["steps_applied"] == 2
    assert r["final_seq"] == 2
    final = {d["kind"]: d["observed_value"] for d in r["final_observed"]}
    assert final["entropy"] == pytest.approx(1.0, abs=1e-9)
    assert final["probability"] == pytest.approx(0.5, abs=1e-9)
    assert store.get_debug_events() == []  # no false trip recorded


def test_trip_records_event_and_journal_unaffected(store):
    c = Circuit(2, store=store)
    r = run_with_watchpoints(c, BELL_STEPS, [EntropyWatch(0, 0.5)])
    events = store.get_debug_events()
    assert len(events) == 1 and r["debug_events"] == events
    ev = events[0]
    assert ev["event_seq"] == 1 and ev["event_kind"] == "breakpoint"
    assert ev["journal_seq"] == 2 and ev["seq"] == 2
    assert ev["condition_kind"] == "entropy"
    assert ev["observed_value"] == pytest.approx(1.0, abs=1e-9)
    assert ev["state_hash"] == c.journal[-1]["state_hash"]
    doc = store.db.get("debug_events", "1")  # own NEDB collection
    assert doc is not None and doc["condition_kind"] == "entropy"
    # Gate journal unaffected: length, chain, replay, trace.
    assert len(store.entries) == 2 and len(c.journal) == 2
    assert store.verify_chain() is True
    assert c.replay() is True
    assert c.trace()["verified"] is True
    assert c.as_of(2)["gate"] == "CNOT"


def test_storeless_circuit_events_in_memory():
    c = Circuit(2)  # no store: no crash; events live on the circuit
    r = run_with_watchpoints(c, BELL_STEPS, [EntropyWatch(0, 0.5)])
    assert r["tripped"] is True and r["seq"] == 2
    assert len(c.journal) == 2
    assert len(c.debug_events) == 1
    assert c.debug_events[0]["condition_kind"] == "entropy"
    assert r["event"] == c.debug_events[0]
    assert r["inspect"]["probabilities"]["11"] == pytest.approx(0.5, abs=1e-9)  # live-state report


def test_dict_and_canonical_step_shapes(store):
    c = Circuit(2, store=store)
    steps = [{"gate": "H", "targets": [0]},
             ("RY", (1,), (0.0,)),
             {"gate": "CNOT", "targets": [0, 1]}]
    r = run_with_watchpoints(c, steps, [EntropyWatch(1, 0.5)])
    assert r["tripped"] is True and r["seq"] == 3
    assert [e["gate"] for e in c.journal] == ["H", "RY", "CNOT"]


def test_invalid_conditions_raise():
    with pytest.raises(ValueError):
        ProbabilityWatch("2", 0.5)  # not a bitstring
    with pytest.raises(ValueError):
        ProbabilityWatch("", 0.5)
    with pytest.raises(ValueError):
        ProbabilityWatch("11", 1.5)  # threshold out of [0, 1]
    with pytest.raises(ValueError):
        ProbabilityWatch("11", -0.1)
    with pytest.raises(ValueError):
        EntropyWatch(-1, 0.5)
    with pytest.raises(ValueError):
        EntropyWatch(0, 2.0)
    with pytest.raises(ValueError):
        FidelityWatch(np.array([1, 0, 0, 0], dtype=complex), 1.5)
    with pytest.raises(ValueError):
        FidelityWatch(np.array([1, 1, 0, 0], dtype=complex), 0.5)  # not normalised
    with pytest.raises(ValueError):
        FidelityWatch(np.ones(3, dtype=complex) / np.sqrt(3), 0.5)  # not 2^k


def test_mismatched_conditions_raise_before_any_gate(store):
    c = Circuit(2, store=store)
    bad = [ProbabilityWatch("111", 0.5), EntropyWatch(5, 0.5),
           FidelityWatch(np.eye(1, 8, 0, dtype=complex)[0], 0.5)]
    for cond in bad:
        with pytest.raises(ValueError):
            run_with_watchpoints(c, BELL_STEPS, [cond])
    with pytest.raises(ValueError):
        run_with_watchpoints(c, BELL_STEPS, [object()])
    with pytest.raises(ValueError):
        run_with_watchpoints(c, [("H",)], [EntropyWatch(0, 0.5)])
    assert len(c.journal) == 0  # validation happens before gate 1
