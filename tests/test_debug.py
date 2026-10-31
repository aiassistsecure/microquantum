"""D1 debugger inspection core: inspect_at / diff_states."""
import numpy as np
import pytest

from microq import Circuit, JournalStore
from microq.debug import diff_states, inspect_at
from microq.state import StateVector


@pytest.fixture()
def store(tmp_path):
    return JournalStore(tmp_path / "data")


def _bell(store):
    c = Circuit(2, store=store)
    c.h(0).cnot(0, 1)
    return c


def test_bell_final_inspection(store):
    rep = inspect_at(_bell(store), 2)
    assert rep["seq"] == 2
    assert rep["probabilities"]["00"] == pytest.approx(0.5, abs=1e-9)
    assert rep["probabilities"]["11"] == pytest.approx(0.5, abs=1e-9)
    assert rep["probabilities"]["01"] == pytest.approx(0.0, abs=1e-9)
    for b in rep["bloch_vectors"]:  # maximally mixed marginals
        assert b == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)
    for s in rep["entanglement_entropies"]:
        assert s == pytest.approx(1.0, abs=1e-9)
    for rho in rep["reduced_states"]:
        assert rho == pytest.approx(np.eye(2) / 2)
    top = rep["top_amplitudes"]
    assert {t["bitstring"] for t in top[:2]} == {"00", "11"}
    for t in top[:2]:
        assert abs(t["amplitude"]) == pytest.approx(1 / np.sqrt(2), abs=1e-9)
        assert t["prob"] == pytest.approx(0.5, abs=1e-9)


def test_product_state_zero_entropy_and_bloch_order(store):
    # H on qubit 1 only (qubit 0 = MSB): |00>+|01> over sqrt(2).
    c = Circuit(2, store=store)
    c.h(1)
    rep = inspect_at(c, 1)
    assert rep["entanglement_entropies"] == pytest.approx([0.0, 0.0], abs=1e-9)
    assert rep["bloch_vectors"][0] == pytest.approx((0.0, 0.0, 1.0), abs=1e-9)
    assert rep["bloch_vectors"][1] == pytest.approx((1.0, 0.0, 0.0), abs=1e-9)
    assert rep["probabilities"]["01"] == pytest.approx(0.5, abs=1e-9)
    # Mirror: H on qubit 0 -> |00>+|10>; Bloch roles swap. Pins MSB order.
    c2 = Circuit(2, store=JournalStore(store.root / "b"))
    c2.h(0)
    rep2 = inspect_at(c2, 1)
    assert rep2["bloch_vectors"][0] == pytest.approx((1.0, 0.0, 0.0), abs=1e-9)
    assert rep2["bloch_vectors"][1] == pytest.approx((0.0, 0.0, 1.0), abs=1e-9)
    assert rep2["probabilities"]["10"] == pytest.approx(0.5, abs=1e-9)


def test_genesis_seq0(store):
    rep = inspect_at(_bell(store), 0)
    assert rep["probabilities"]["00"] == pytest.approx(1.0, abs=1e-12)
    assert rep["top_amplitudes"][0]["bitstring"] == "00"
    assert rep["bloch_vectors"][0] == pytest.approx((0.0, 0.0, 1.0), abs=1e-9)
    assert rep["entanglement_entropies"] == pytest.approx([0.0, 0.0], abs=1e-9)


def test_intermediate_matches_manually_stepped(store):
    c = Circuit(3, store=store)
    c.h(0).cnot(0, 1).ry(2, 0.7).cnot(1, 2)
    manual = StateVector(3)
    from microq.state import gate_matrix
    manual.apply_1q(gate_matrix("H"), 0)
    manual.cnot(0, 1)  # state after seq 2
    rep = inspect_at(c, 2)
    assert rep["state_hash"] == c.journal[1]["state_hash"]
    assert rep["probabilities"] == pytest.approx(manual.prob_dict(), abs=1e-12)
    for bs, amp in rep["amplitudes"].items():
        assert amp == pytest.approx(manual.to_numpy()[int(bs, 2)], abs=1e-12)


def test_inspect_and_diff_do_not_mutate(store):
    c = _bell(store)
    head_hash, head_len = c.current_hash(), len(c.journal)
    psi_before = c.state.to_numpy().copy()
    inspect_at(c, 1)
    diff_states(c, 0, 2)
    assert c.current_hash() == head_hash and len(c.journal) == head_len
    assert np.array_equal(c.state.to_numpy(), psi_before)


def test_out_of_range_and_storeless_raise(store):
    c = _bell(store)
    for bad in (-1, 3, 99):
        with pytest.raises(ValueError):
            inspect_at(c, bad)
        with pytest.raises(ValueError):
            diff_states(c, 0, bad)
    bare = Circuit(2)  # no store -> no checkpoints to restore from
    bare.h(0)
    with pytest.raises(ValueError):
        inspect_at(bare, 1)


def test_diff_identical_seqs(store):
    d = diff_states(_bell(store), 2, 2)
    assert d["fidelity"] == pytest.approx(1.0, abs=1e-12)
    assert all(v == pytest.approx(0.0) for v in d["prob_deltas"].values())
    assert all(abs(v) == pytest.approx(0.0) for v in d["amplitude_deltas"].values())
    assert all(m["delta_prob"] == pytest.approx(0.0) for m in d["largest_movers"])


def test_diff_pre_post_cnot_matches_manual(store):
    c = _bell(store)  # seq 1: (|00>+|10>)/sqrt2, seq 2: Bell
    d = diff_states(c, 1, 2)
    psi_a = np.array([1, 0, 1, 0], dtype=complex) / np.sqrt(2)
    psi_b = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    assert d["fidelity"] == pytest.approx(abs(np.vdot(psi_a, psi_b)) ** 2, abs=1e-12)
    assert d["fidelity"] == pytest.approx(0.25, abs=1e-12)
    assert d["largest_movers"][0]["bitstring"] in ("10", "11")
    assert abs(d["largest_movers"][0]["delta_prob"]) == pytest.approx(0.5, abs=1e-9)
    assert d["prob_deltas"]["11"] == pytest.approx(0.5, abs=1e-9)
    assert d["prob_deltas"]["10"] == pytest.approx(-0.5, abs=1e-9)


def test_diff_exposes_wrong_gate(store):
    # Correct Bell at seq 2; a stray X on qubit 0 at seq 3 is the bug:
    # state becomes (|10>+|01>)/sqrt2, orthogonal to the Bell state.
    c = _bell(store)
    c.x(0)
    d = diff_states(c, 2, 3)
    assert d["fidelity"] == pytest.approx(0.0, abs=1e-12)
    assert {m["bitstring"] for m in d["largest_movers"][:4]} == {"00", "01", "10", "11"}
    rep = inspect_at(c, 3)
    assert rep["probabilities"]["01"] == pytest.approx(0.5, abs=1e-9)
    assert rep["probabilities"]["00"] == pytest.approx(0.0, abs=1e-9)


def test_ghz_entropies(store):
    c = Circuit(3, store=store)
    c.h(0).cnot(0, 1).cnot(0, 2)
    rep = inspect_at(c, 3)
    assert rep["entanglement_entropies"] == pytest.approx([1.0, 1.0, 1.0], abs=1e-9)
    assert rep["probabilities"]["000"] == pytest.approx(0.5, abs=1e-9)
    assert rep["probabilities"]["111"] == pytest.approx(0.5, abs=1e-9)
