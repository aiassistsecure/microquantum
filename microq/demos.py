"""Runnable demos: Bell, GHZ-5, Grover-4. Run: python -m microq.demos

All demos are classical state-vector simulation — not quantum hardware.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from .circuit import Circuit
from .journal import JournalStore

DATA = Path(__file__).resolve().parent.parent / "data"


def _fresh_store(name: str) -> JournalStore:
    root = DATA / name
    if root.exists():
        shutil.rmtree(root)
    return JournalStore(root)


def _report(c: Circuit) -> None:
    length = len(c.journal)
    head = c.current_hash()
    ok_replay = c.replay()
    mid = max(1, length // 2)
    expected = c.journal[mid - 1]["state_hash"]
    c.rewind(mid)
    print(f"  journal length: {length} entries; head state hash {head[:16]}...")
    print(f"  replay() from |0...0> reproduces final state: {ok_replay}")
    print(f"  rewind(seq={mid}) -> state == checkpoint at seq {mid}: {c.current_hash() == expected}")


def bell() -> Circuit:
    print("=== Demo A: Bell state (2 qubits) ===")
    c = Circuit(2, store=_fresh_store("bell"))
    c.h(0).cnot(0, 1)
    probs = c.probabilities()
    for k in sorted(probs):
        print(f"  P(|{k}>) = {probs[k]:.4f}")
    print(f"  1000 shots: {c.sample(1000, seed=42)}")
    _report(c)
    return c


def ghz() -> Circuit:
    print("=== Demo B: GHZ state (5 qubits) ===")
    c = Circuit(5, store=_fresh_store("ghz"))
    c.h(0)
    for q in range(1, 5):
        c.cnot(0, q)
    for k, v in sorted(c.probabilities().items()):
        if v > 1e-9:
            print(f"  P(|{k}>) = {v:.4f}")
    print(f"  1000 shots: {c.sample(1000, seed=7)}")
    _report(c)
    return c


# --- Grover building blocks ---------------------------------------------
def toffoli(c: Circuit, c1: int, c2: int, t: int) -> None:
    """Exact CCX from H/CNOT/T/TDG (Nielsen & Chuang decomposition)."""
    c.h(t)
    c.cnot(c2, t); c.apply("TDG", t); c.cnot(c1, t); c.t(t)
    c.cnot(c2, t); c.apply("TDG", t); c.cnot(c1, t)
    c.t(c2); c.t(t); c.cnot(c1, c2); c.h(t); c.t(c1); c.apply("TDG", c2); c.cnot(c1, c2)


def mcz_search(c: Circuit) -> None:
    """Phase-flip |1111> on search qubits 0..3, exact, using ancillas
    4,5 as a Toffoli ladder with compute/uncompute (ancillas end |0>)."""
    toffoli(c, 0, 1, 4)
    toffoli(c, 4, 2, 5)
    c.cz(5, 3)
    toffoli(c, 4, 2, 5)
    toffoli(c, 0, 1, 4)


def oracle(c: Circuit, marked: str) -> None:
    zeros = [i for i, b in enumerate(marked) if b == "0"]
    for q in zeros:
        c.x(q)
    mcz_search(c)
    for q in zeros:
        c.x(q)


def diffuser(c: Circuit) -> None:
    for q in range(4):
        c.h(q); c.x(q)
    mcz_search(c)  # flips |1111> after wraps == reflects about |0000> up to sign
    for q in range(4):
        c.x(q); c.h(q)


def grover(marked: str = "1011", iterations: int = 3) -> Circuit:
    print("=== Demo C: Grover search (4 search qubits + 2 ancillas) ===")
    print("Searching an unsorted space of 16 'records' for marked |1011>.")
    print("Classical simulation: amplitude amplification, no quantum speedup.")
    c = Circuit(6, store=_fresh_store("grover"))

    def p_marked() -> float:
        # marginal probability over the 4 search qubits (ancillas traced out)
        total = 0.0
        for k, v in c.probabilities().items():
            if k[:4] == marked:
                total += v
        return total

    for q in range(4):
        c.h(q)
    print(f"  initial P(marked) = {p_marked():.4f}")
    for it in range(1, iterations + 1):
        oracle(c, marked)
        diffuser(c)
        print(f"  after iteration {it}: P(marked) = {p_marked():.4f}")
    shots = c.sample(1000, seed=3)
    top = sorted(shots.items(), key=lambda kv: -kv[1])[:3]
    print(f"  1000 shots, top outcomes (6-bit: search+ancillas): {top}")
    _report(c)
    return c


if __name__ == "__main__":
    bell()
    print()
    ghz()
    print()
    grover()
