"""P2 checkpoint striding: checkpoint_every=k persists blobs only at
seqs divisible by k (genesis always); every other seq resolves via
nearest-checkpoint restore + replay-forward (state_at)."""
import numpy as np
import pytest

from microq import Circuit, JournalStore
from microq.cli import inspect_journal
from microq.debug import find_divergence, inspect_at
from microq.journal import state_hash

# 11 distinct-state gates: cumulative rotations never repeat a state.
_STEPS = [("RY", (0,), (0.31,)), ("RX", (1,), (0.47,)), ("H", (0,), ()),
          ("CNOT", (0, 1), ()), ("RY", (1,), (0.53,)), ("RZ", (0,), (0.29,)),
          ("RY", (0,), (0.41,)), ("CP", (0, 1), (0.37,)), ("RX", (0,), (0.43,)),
          ("CNOT", (1, 0), ()), ("RY", (1,), (0.59,))]


def _build(root, every, steps=_STEPS):
    store = JournalStore(root)
    c = Circuit(2, store=store, checkpoint_every=every)
    for gate, targets, params in steps:
        c.apply(gate, *targets, params=params)
    return c, store


def test_stride_blobs_only_at_stride_seqs(tmp_path):
    c, store = _build(tmp_path / "s4", 4)
    assert len(c.journal) == 11 and c.checkpoint_every == 4
    hashes = [e["state_hash"] for e in c.journal]
    assert len(set(hashes)) == 11  # distinct states, no dedup masking
    assert store.has_checkpoint(c._genesis_hash)  # genesis always kept
    assert store.has_checkpoint(hashes[3])  # seq 4
    assert store.has_checkpoint(hashes[7])  # seq 8
    assert not store.has_checkpoint(hashes[4])  # seq 5: no blob
    assert not store.has_checkpoint(hashes[10])  # seq 11: no blob
    blobs = list(store.ckpt_dir.glob("*.npy"))
    assert len(blobs) == 3  # genesis + seqs 4 and 8 only
    assert len(blobs) < len(c.journal)
    assert store.verify_chain() is True  # schema / hash chain unchanged


def test_state_at_every_seq_matches_stride1_reference(tmp_path):
    strided, _ = _build(tmp_path / "s", 4)
    ref, _ = _build(tmp_path / "r", 1)
    assert [e["state_hash"] for e in strided.journal] == \
        [e["state_hash"] for e in ref.journal]  # hashes unchanged
    genesis = np.zeros(4, dtype=np.complex128); genesis[0] = 1.0
    assert state_hash(strided.state_at(0)) == state_hash(genesis)
    np.testing.assert_array_equal(strided.state_at(0), ref.state_at(0))
    for seq in range(1, len(ref.journal) + 1):
        np.testing.assert_array_equal(strided.state_at(seq), ref.state_at(seq))
        assert state_hash(strided.state_at(seq)) == ref.journal[seq - 1]["state_hash"]


def test_state_at_errors(tmp_path):
    c, _ = _build(tmp_path / "s", 4)
    for bad in (-1, len(c.journal) + 1, 99):
        with pytest.raises(ValueError):
            c.state_at(bad)
    bare = Circuit(2)
    bare.h(0)
    with pytest.raises(ValueError):
        bare.state_at(1)


def test_rewind_and_fork_from_noncheckpoint_seq(tmp_path):
    c, _ = _build(tmp_path / "s", 4)
    ref, _ = _build(tmp_path / "r", 1)
    expected = ref.journal[5]["state_hash"]  # seq 6, not checkpointed (k=4)
    f = c.fork(6)
    assert f.checkpoint_every == 4
    assert len(f.journal) == 6 and f.current_hash() == expected
    c.rewind(6)
    assert len(c.journal) == 6 and c.current_hash() == expected
    np.testing.assert_array_equal(c.state.to_numpy(), ref.state_at(6))


def _div_pair(tmp_path, angle_b, every=4):
    steps = lambda ang: [("H", (0,), ()), ("CNOT", (0, 1), ()),
                         ("RY", (1,), (ang,)), ("RZ", (0,), (0.3,)),
                         ("RY", (0,), (0.2,)), ("X", (1,), ())]
    a, _ = _build(tmp_path / "a", every, steps(0.5))
    b, _ = _build(tmp_path / "b", every, steps(angle_b))
    return a, b


def test_inspect_and_divergence_on_strided(tmp_path):
    a, b = _div_pair(tmp_path, 1.2)  # planted bug at seq 3 (non-stride)
    rep = inspect_at(a, 5)  # seq 5 resolves via seq-4 checkpoint + 1 gate
    assert rep["state_hash"] == a.journal[4]["state_hash"]
    r = find_divergence(a, b)
    assert r["diverged"] is True and r["seq"] == 3
    assert r["suspect_gate"]["gate"] == "RY"
    assert r["suspect_gate"]["params"] == pytest.approx([1.2])
    a2, b2 = _div_pair(tmp_path / "ctl", 0.5)  # identical control
    r2 = find_divergence(a2, b2)
    assert r2["diverged"] is False and r2["seq"] is None


def test_append_checkpoint_false_keeps_hash_skips_blob(tmp_path):
    store = JournalStore(tmp_path / "raw")
    psi = np.zeros(4, dtype=np.complex128); psi[0] = 1.0
    e = store.append("H", [0], [], psi, checkpoint=False)
    assert e["state_hash"] == state_hash(psi)  # hash still recorded
    assert not store.has_checkpoint(e["state_hash"])  # but no blob
    e2 = store.append("X", [0], [], psi)  # default True: blob written
    assert store.has_checkpoint(e2["state_hash"])
    assert store.verify_chain() is True


@pytest.mark.parametrize("bad", [0, -1, -4, 1.5, 2.0, "4", None, True, False])
def test_invalid_checkpoint_every_raises(bad):
    with pytest.raises(ValueError):
        Circuit(2, checkpoint_every=bad)


def test_cli_inspect_strided_head_and_mid(tmp_path):
    root = tmp_path / "s"
    c, _ = _build(root, 4)  # head seq 11 carries no checkpoint blob
    assert not c.store.has_checkpoint(c.journal[-1]["state_hash"])
    head = inspect_journal(root)
    assert head["seq"] == 11
    assert head["state_hash"] == c.journal[-1]["state_hash"]
    mid = inspect_journal(root, seq=6)
    assert mid["seq"] == 6 and mid["state_hash"] == c.journal[5]["state_hash"]
