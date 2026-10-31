import numpy as np
import pytest

from microq import Circuit, JournalStore
from microq.demos import grover, oracle, diffuser, toffoli
from microq.state import StateVector


@pytest.fixture()
def store(tmp_path):
    return JournalStore(tmp_path / "data")


def test_bell_probabilities(store):
    c = Circuit(2, store=store)
    c.h(0).cnot(0, 1)
    p = c.probabilities()
    assert p["00"] == pytest.approx(0.5, abs=1e-9)
    assert p["11"] == pytest.approx(0.5, abs=1e-9)
    assert p["01"] == pytest.approx(0.0, abs=1e-9)
    assert p["10"] == pytest.approx(0.0, abs=1e-9)


def test_ghz_probabilities(store):
    c = Circuit(5, store=store)
    c.h(0)
    for q in range(1, 5):
        c.cnot(0, q)
    p = c.probabilities()
    assert p["00000"] == pytest.approx(0.5, abs=1e-9)
    assert p["11111"] == pytest.approx(0.5, abs=1e-9)
    assert sum(v for k, v in p.items() if k not in ("00000", "11111")) < 1e-9


def test_toffoli_truth_table(store):
    for bits in range(8):
        sv = StateVector(3)
        src = [(bits >> (2 - i)) & 1 for i in range(3)]
        for i, b in enumerate(src):
            if b:
                from microq.state import gate_matrix
                sv.apply_1q(gate_matrix("X"), i)
        c = Circuit(3, store=JournalStore(store.root / f"tf{bits}"))
        c.state = sv
        toffoli(c, 0, 1, 2)
        expected = src.copy()
        if src[0] and src[1]:
            expected[2] ^= 1
        idx = int("".join(map(str, expected)), 2)
        assert abs(c.state.psi[idx]) == pytest.approx(1.0, abs=1e-9)


def test_grover_amplification(store):
    c = Circuit(6, store=store)
    marked = "1011"
    for q in range(4):
        c.h(q)
    for _ in range(3):
        oracle(c, marked)
        diffuser(c)
    total = sum(v for k, v in c.probabilities().items() if k[:4] == marked)
    assert total > 0.9


def test_rewind_correctness(store):
    c = Circuit(3, store=store)
    c.h(0).cnot(0, 1)
    snap_hash = c.current_hash()
    snap_psi = c.state.psi.copy()
    c.x(2).cnot(1, 2).h(2)
    assert len(c.journal) == 5
    c.rewind(2)
    assert c.current_hash() == snap_hash
    assert np.allclose(c.state.psi, snap_psi)
    assert len(c.journal) == 2


def test_rewind_to_genesis(store):
    c = Circuit(2, store=store)
    c.h(0).cnot(0, 1)
    c.rewind(0)
    assert c.state.psi[0] == 1.0
    assert len(c.journal) == 0


def test_fork_independence(store):
    c = Circuit(2, store=store)
    c.h(0).cnot(0, 1)
    parent_hash = c.current_hash()
    f = c.fork()
    f.x(0)
    assert c.current_hash() == parent_hash
    assert f.current_hash() != parent_hash
    # parent continues independently
    c.z(1)
    assert f.journal[-1]["gate"] == "X"


def test_replay_hash_equality(store):
    c = Circuit(3, store=store)
    c.h(0).cnot(0, 2).ry(1, 0.7).cnot(1, 2).rz(0, -0.3)
    assert c.replay() is True


def test_journal_hash_chain(store):
    c = Circuit(2, store=store)
    c.h(0).cnot(0, 1).x(1)
    assert store.verify_chain() is True
    # tamper: flip a recorded gate name -> chain must fail
    store.entries[1]["gate"] = "Z"
    assert store.verify_chain() is False


def test_checkpoint_dedup(store, tmp_path):
    c = Circuit(2, store=store)
    c.h(0)
    n1 = len(list(store.ckpt_dir.glob("*.npy")))
    c2 = Circuit(2, store=store)
    c2.h(0)  # identical state bytes -> no new checkpoint file
    n2 = len(list(store.ckpt_dir.glob("*.npy")))
    assert n1 == n2


def test_fidelity_and_norm(store):
    c = Circuit(2, store=store)
    c.h(0).cnot(0, 1)
    assert c.state.norm() == pytest.approx(1.0, abs=1e-12)
    assert c.state.fidelity(c.state) == pytest.approx(1.0, abs=1e-12)
