"""D2 debugger: find_divergence bisection."""
import numpy as np
import pytest

import microq.debug as debug
from microq import Circuit, JournalStore, find_divergence


def _pair(tmp_path, build_a, build_b, n=2):
    a = Circuit(n, store=JournalStore(tmp_path / "a"))
    b = Circuit(n, store=JournalStore(tmp_path / "b"))
    build_a(a)
    build_b(b)
    return a, b


def _prefix(c):  # returns None so `lambda c: _prefix(c) or c.x(0)` chains
    c.h(0).cnot(0, 1)


def test_planted_bug_exact_gate(tmp_path):
    # Shared prefix seq 1-2, bug at seq 3 (RY angle), gates after it.
    def good(c):
        _prefix(c); c.ry(1, 0.5).rz(0, 0.3).x(1)

    def bad(c):
        _prefix(c); c.ry(1, 1.2).rz(0, 0.3).x(1)

    a, b = _pair(tmp_path, good, bad)
    r = find_divergence(a, b)
    assert r["diverged"] is True
    assert r["seq"] == 3
    assert r["suspect_gate"]["gate"] == "RY"
    assert r["suspect_gate"]["targets"] == [1]
    assert r["suspect_gate"]["params"] == pytest.approx([1.2])
    assert r["gate_a"]["params"] == pytest.approx([0.5])
    assert r["gate_b"] == r["suspect_gate"]
    assert r["fidelity"] < 1 - 1e-9
    assert r["state_hash_a"] != r["state_hash_b"]
    assert (r["length_a"], r["length_b"], r["lengths"]) == (5, 5, (5, 5))


def test_no_bug_identical_circuits(tmp_path):
    a, b = _pair(tmp_path, lambda c: _prefix(c) or c.ry(1, 0.5),
                 lambda c: _prefix(c) or c.ry(1, 0.5))
    r = find_divergence(a, b)
    assert r["diverged"] is False
    assert r["seq"] is None and r["suspect_gate"] is None
    assert r["fidelity"] == pytest.approx(1.0, abs=1e-12)
    assert r["state_hash_a"] == r["state_hash_b"]
    assert "no divergence" in r["reason"]


def test_differing_lengths_extra_gate(tmp_path):
    a, b = _pair(tmp_path, _prefix, lambda c: _prefix(c) or c.x(0))
    r = find_divergence(a, b)
    assert r["diverged"] is True
    assert r["seq"] == 3  # min_len (2) + 1
    assert r["fidelity"] is None  # no counterpart state; not fabricated
    assert "length mismatch" in r["reason"]
    assert r["suspect_gate"]["gate"] == "X"
    assert r["suspect_gate"]["targets"] == [0]
    assert r["gate_a"] is None and r["state_hash_a"] is None
    assert r["state_hash_b"] is not None


def test_differing_lengths_longer_a_suspect_falls_back(tmp_path):
    a, b = _pair(tmp_path, lambda c: _prefix(c) or c.z(1), _prefix)
    r = find_divergence(a, b)
    assert r["seq"] == 3 and r["fidelity"] is None
    assert r["suspect_gate"]["gate"] == "Z"  # from a: b has no seq 3


def test_tolerance_tiny_angle_no_divergence(tmp_path):
    a, b = _pair(tmp_path, lambda c: c.ry(0, 0.5), lambda c: c.ry(0, 0.5 + 1e-6), n=1)
    r = find_divergence(a, b)  # fidelity ~1 - 2.5e-13, within 1e-9
    assert r["diverged"] is False and r["seq"] is None
    assert r["fidelity"] > 1 - 1e-9
    assert r["state_hash_a"] != r["state_hash_b"]  # hash differs, fidelity decides


def test_first_gate_divergence(tmp_path):
    a, b = _pair(tmp_path, lambda c: c.ry(0, 0.5).h(1),
                 lambda c: c.ry(0, 1.5).h(1))
    r = find_divergence(a, b)
    assert r["diverged"] is True and r["seq"] == 1
    assert r["suspect_gate"]["gate"] == "RY"


def test_genesis_seq0_divergence_via_fork(tmp_path):
    parent = Circuit(2, store=JournalStore(tmp_path / "p"))
    parent.h(0).cnot(0, 1)
    child = parent.fork(2)  # child's genesis is the Bell state, not |00>
    r = find_divergence(parent, child)
    assert r["diverged"] is True and r["seq"] == 0
    assert r["suspect_gate"] is None and r["gate_a"] is None
    assert r["fidelity"] == pytest.approx(0.5, abs=1e-9)


def test_binary_search_is_log_step(tmp_path, monkeypatch):
    calls = 0
    orig = debug._state_at

    def counting(circuit, seq):
        nonlocal calls
        calls += 1
        return orig(circuit, seq)

    monkeypatch.setattr(debug, "_state_at", counting)

    def build(angle13):
        def f(c):
            for i in range(16):
                c.ry(0, angle13 if i == 12 else 0.2)
        return f

    a, b = _pair(tmp_path, build(0.2), build(1.0), n=1)
    calls = 0  # ignore construction-time checkpoint writes (none use _state_at)
    r = find_divergence(a, b)
    assert r["seq"] == 13 and r["suspect_gate"]["gate"] == "RY"
    # 2 calls per probe; log search needs ~6 probes, a linear scan to
    # seq 13 would need 13 probes (26 calls).
    assert calls <= 16


def test_different_n_and_storeless_raise(tmp_path):
    a, _ = _pair(tmp_path, _prefix, _prefix)
    other_n = Circuit(3, store=JournalStore(tmp_path / "c"))
    with pytest.raises(ValueError):
        find_divergence(a, other_n)
    bare_a, bare_b = Circuit(2), Circuit(2)
    bare_a.h(0); bare_b.h(0)
    with pytest.raises(ValueError):
        find_divergence(bare_a, bare_b)
    with pytest.raises(ValueError):
        find_divergence(a, bare_b)


def test_find_divergence_does_not_mutate(tmp_path):
    a, b = _pair(tmp_path, lambda c: _prefix(c) or c.ry(1, 0.5),
                 lambda c: _prefix(c) or c.ry(1, 1.2))
    before = [(c.current_hash(), len(c.journal), c.state.to_numpy().copy())
              for c in (a, b)]
    find_divergence(a, b)
    for c, (h, n, psi) in zip((a, b), before):
        assert c.current_hash() == h and len(c.journal) == n
        assert np.array_equal(c.state.to_numpy(), psi)
