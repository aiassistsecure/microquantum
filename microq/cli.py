"""D4: proof demo + CLI for the debugger.

Classical state-vector simulation only — no quantum hardware.

- ``debug_demo`` builds a known-good circuit and a deliberately
  mutated twin, then uses :func:`microq.debug.find_divergence` to
  bisect to the exact mutated gate. An identical-twin control pair
  must report no divergence.
- ``inspect`` / :func:`inspect_journal` print the D1
  :func:`microq.debug.inspect_at` report for a stored journal.
"""
from __future__ import annotations

import argparse
import math
import sys
import tempfile
from pathlib import Path

from .circuit import Circuit
from .debug import find_divergence, inspect_at
from .journal import JournalStore

_GOOD_ANGLE = 0.5
_BAD_ANGLE = 1.1


def _build_demo_circuit(root: Path, ry_angle: float) -> Circuit:
    """The D4 demo circuit in its own fresh store.

    seq 1: H(0), seq 2: RY(1, ry_angle), seq 3: CNOT(0, 1),
    seq 4: RY(0, 0.25). Gates after seq 2 make bisection meaningful.
    """
    store = JournalStore(root)
    c = Circuit(2, store=store)
    c.h(0)
    c.ry(1, ry_angle)
    c.cnot(0, 1)
    c.ry(0, 0.25)
    return c


def debug_demo(root=None) -> dict:
    """Run the planted-bug proof demo; print a human-readable verdict.

    Builds the good circuit (RY angle 0.5 at seq 2) and a mutated twin
    (only that angle changed to 1.1) in separate fresh stores, plus an
    identical copy of the good circuit as a no-bug control. Returns a
    dict with the divergence result and the control result.
    """
    if root is None:
        base = Path(tempfile.mkdtemp(prefix="microq-debug-demo-"))
    else:
        base = Path(root)
        base.mkdir(parents=True, exist_ok=True)
    good = _build_demo_circuit(base / "good", _GOOD_ANGLE)
    mutated = _build_demo_circuit(base / "mutated", _BAD_ANGLE)
    control = _build_demo_circuit(base / "control", _GOOD_ANGLE)

    result = find_divergence(good, mutated)
    control_result = find_divergence(good, control)

    out = {
        "diverged": bool(result["diverged"]),
        "seq": result["seq"],
        "suspect_gate": result["suspect_gate"],
        "gate_a": result["gate_a"],
        "gate_b": result["gate_b"],
        "expected": result["gate_a"],
        "actual": result["gate_b"],
        "fidelity": result["fidelity"],
        "state_hash_a": result["state_hash_a"],
        "state_hash_b": result["state_hash_b"],
        "reason": result["reason"],
        "control": control_result,
        "control_diverged": bool(control_result["diverged"]),
        "control_fidelity": control_result["fidelity"],
    }

    ok = (out["diverged"] and out["seq"] == 2
          and out["suspect_gate"] is not None
          and out["suspect_gate"].get("gate") == "RY"
          and out["suspect_gate"].get("targets") == [1]
          and len(out["suspect_gate"].get("params") or []) == 1
          and abs(out["suspect_gate"]["params"][0] - _BAD_ANGLE) < 1e-12
          and not out["control_diverged"])
    out["ok"] = bool(ok)

    if out["diverged"] and out["seq"] == 2 and out["suspect_gate"]:
        sg = out["suspect_gate"]
        exp = out["gate_a"] or {}
        print(f"BUG FOUND at seq {out['seq']}: gate {sg.get('gate')} "
              f"targets {sg.get('targets')} params {sg.get('params')}")
        print(f"  expected (good): gate {exp.get('gate')} "
              f"targets {exp.get('targets')} params {exp.get('params')}")
        print(f"  actual (mutated): gate {sg.get('gate')} "
              f"targets {sg.get('targets')} params {sg.get('params')}")
        print(f"  fidelity at seq {out['seq']}: {out['fidelity']}")
    else:
        print(f"BUG NOT FOUND at expected seq 2: diverged={out['diverged']} "
              f"seq={out['seq']} suspect_gate={out['suspect_gate']} "
              f"reason={out['reason']}")
    if out["control_diverged"]:
        print(f"CONTROL FAILED: identical twin falsely diverged "
              f"at seq {control_result['seq']} "
              f"(fidelity {control_result['fidelity']})")
    else:
        print(f"Control: no divergence for identical twin "
              f"(fidelity {control_result['fidelity']}) — as expected.")
    if not ok:
        print("DEMO FAILED: planted bug was not isolated cleanly "
              "(see lines above).")
    return out


def load_circuit(journal_path) -> Circuit:
    """Reopen a stored journal as a Circuit usable by inspect_at.

    Note: a fresh Circuit over a reopened store starts with an empty
    journal mirror — this loader restores the mirror from the store
    (``circuit.journal`` = copies of ``store.entries``) so
    ``inspect_at`` works on the reopened journal. Seq 0 assumes the
    standard |0...0> genesis, matching how circuits are created.
    """
    path = Path(journal_path)
    if not path.exists():
        raise ValueError(f"journal path does not exist: {journal_path}")
    store = JournalStore(path)
    if not store.entries:
        raise ValueError(f"journal is empty: {journal_path}")
    # Under checkpoint striding the head entry may have no blob: derive
    # n from the nearest checkpointed entry, scanning backwards.
    psi = None
    for entry in reversed(store.entries):
        if store.has_checkpoint(entry["state_hash"]):
            psi = store.get_checkpoint(entry["state_hash"])
            break
    if psi is None:
        raise ValueError(
            f"no checkpointed entry found for journal: {journal_path}")
    dim = int(psi.shape[0]) if psi.ndim == 1 else 0
    if dim < 2 or (dim & (dim - 1)) != 0:
        raise ValueError(
            f"checkpoint dimension {dim} is not a power of two "
            f"(journal: {journal_path})")
    n = int(math.log2(dim))
    circuit = Circuit(n, store=store)
    circuit.journal = [dict(e) for e in store.entries]
    return circuit


def inspect_journal(journal_path, seq=None) -> dict:
    """inspect_at report for a stored journal (default: head seq)."""
    circuit = load_circuit(journal_path)
    if seq is None:
        seq = len(circuit.journal)
    return inspect_at(circuit, seq)


def _fmt(value: float) -> str:
    return f"{float(value):.10g}"


def _print_inspect(report: dict) -> None:
    print(f"seq: {report['seq']}")
    print(f"n_qubits: {report['n_qubits']}")
    print(f"state_hash: {report['state_hash']}")
    print("top amplitudes:")
    for t in report["top_amplitudes"]:
        print(f"  |{t['bitstring']}> amplitude={t['amplitude']} "
              f"prob={_fmt(t['prob'])}")
    print("probabilities:")
    for bs in sorted(report["probabilities"]):
        print(f"  P(|{bs}>) = {_fmt(report['probabilities'][bs])}")
    print("per-qubit:")
    for q in report["qubits"]:
        b = q["bloch_vector"]
        print(f"  qubit {q['qubit']}: "
              f"bloch=({_fmt(b[0])}, {_fmt(b[1])}, {_fmt(b[2])}) "
              f"entanglement_entropy={_fmt(q['entanglement_entropy'])}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="microq",
        description="microquantum debugger CLI — classical "
                    "state-vector simulation, no quantum hardware.")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("debug-demo", help="planted-bug bisection proof demo")
    p_inspect = sub.add_parser(
        "inspect", help="print the inspect_at report for a stored journal")
    p_inspect.add_argument("journal", help="journal store root directory")
    p_inspect.add_argument("--seq", type=int, default=None,
                           help="journal seq to inspect (default: head)")
    args = parser.parse_args(argv)

    if args.command == "debug-demo":
        result = debug_demo()
        return 0 if result["ok"] else 1
    if args.command == "inspect":
        try:
            report = inspect_journal(args.journal, seq=args.seq)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        _print_inspect(report)
        return 0
    parser.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
