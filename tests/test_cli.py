"""D4 debugger proof demo + CLI: debug-demo and inspect."""
import pytest

from microq import Circuit, JournalStore
from microq.cli import debug_demo, inspect_journal, main


def test_debug_demo_finds_planted_bug_at_exact_gate(tmp_path, capsys):
    r = debug_demo(root=tmp_path / "demo")
    assert r["diverged"] is True
    assert r["seq"] == 2
    assert r["suspect_gate"]["gate"] == "RY"
    assert r["suspect_gate"]["targets"] == [1]
    assert r["suspect_gate"]["params"] == pytest.approx([1.1])
    assert r["gate_a"]["params"] == pytest.approx([0.5])
    assert r["expected"]["params"] == pytest.approx([0.5])
    assert r["fidelity"] < 1 - 1e-9
    out = capsys.readouterr().out
    assert "BUG FOUND" in out and "seq 2" in out


def test_debug_demo_control_no_divergence(tmp_path):
    r = debug_demo(root=tmp_path / "demo")
    assert r["control_diverged"] is False
    assert r["control"]["diverged"] is False
    assert r["control"]["seq"] is None


def test_debug_demo_cli_exit_zero(tmp_path, monkeypatch, capsys):
    # debug-demo uses temp stores; just check the CLI verdict/exit code.
    assert main(["debug-demo"]) == 0
    out = capsys.readouterr().out
    assert "BUG FOUND" in out
    assert "no divergence" in out


def _bell_journal(tmp_path):
    root = tmp_path / "bell"
    c = Circuit(2, store=JournalStore(root))
    c.h(0).cnot(0, 1)
    return root, c


def test_inspect_cli_bell_journal(tmp_path, capsys):
    root, c = _bell_journal(tmp_path)
    assert main(["inspect", str(root)]) == 0
    out = capsys.readouterr().out
    assert "seq: 2" in out
    assert c.journal[-1]["state_hash"] in out
    # Bell probabilities 0.5 for 00 and 11 appear in the printed report.
    assert "P(|00>) = 0.5" in out
    assert "P(|11>) = 0.5" in out


def test_inspect_journal_report_values(tmp_path):
    root, _ = _bell_journal(tmp_path)
    rep = inspect_journal(root)
    assert rep["seq"] == 2 and rep["n_qubits"] == 2
    assert rep["probabilities"]["00"] == pytest.approx(0.5, abs=1e-9)
    assert rep["probabilities"]["11"] == pytest.approx(0.5, abs=1e-9)


def test_inspect_cli_seq1(tmp_path, capsys):
    root, _ = _bell_journal(tmp_path)
    assert main(["inspect", str(root), "--seq", "1"]) == 0
    out = capsys.readouterr().out
    assert "seq: 1" in out


def test_inspect_missing_and_empty_return_nonzero(tmp_path, capsys):
    assert main(["inspect", str(tmp_path / "nope")]) == 2
    assert "error" in capsys.readouterr().err
    empty = tmp_path / "empty"
    JournalStore(empty)  # creates an empty store (no entries)
    assert main(["inspect", str(empty)]) == 2
    assert "error" in capsys.readouterr().err
    with pytest.raises(ValueError):
        inspect_journal(tmp_path / "nope")
    with pytest.raises(ValueError):
        inspect_journal(empty)


def test_inspect_out_of_range_seq_nonzero(tmp_path, capsys):
    root, _ = _bell_journal(tmp_path)
    assert main(["inspect", str(root), "--seq", "99"]) == 2
    assert "error" in capsys.readouterr().err
