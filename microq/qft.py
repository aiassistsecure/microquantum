"""S5: Quantum Fourier transform + quantum phase estimation (QPE).

Both are built entirely from journaled Circuit gates (H, CP, SWAP, X),
so a QFT or QPE run replays and hash-chain-verifies like every other
microquantum circuit.

Honest framing: this is classical simulation of quantum mathematics
on a state vector. There is no quantum hardware, no speedup, and no
advantage — the simulator pays 2^n amplitudes to represent the state.
The QPE here estimates the phase of the 1-qubit unitary
``U = diag(1, exp(2*pi*i*phi))`` acting on its eigenstate ``|1>``.

Qubit-ordering note (verified empirically in tests, not assumed):
this engine's qubit 0 is the most-significant index bit. With the
QFT circuit below (H + controlled phases, then SWAPs reversing the
register) the unitary equals the textbook DFT matrix
``F[y, x] = exp(2*pi*i*x*y / N) / sqrt(N)`` in that MSB-first order.
Consequently, in QPE the counting qubit at position ``i`` (counting
register listed MSB-first) must drive ``U**(2**(n_count-1-i))`` for
the peak counting bitstring, read MSB-first as ``int / 2**n_count``,
to equal phi. The tests pin both facts down.
"""
from __future__ import annotations

import math
from pathlib import Path

from .circuit import Circuit
from .journal import JournalStore


def _check_qubits(qubits) -> list[int]:
    qs = list(qubits)
    if not qs or len(set(qs)) != len(qs):
        raise ValueError("qubits must be a non-empty list of distinct indices")
    return qs


def qft(c: Circuit, qubits) -> Circuit:
    """Apply the QFT to ``qubits`` (in register order, first = MSB).

    For each qubit i: H, then CP from each later qubit j with angle
    pi / 2**(j-i); finally SWAPs reverse the register order.
    """
    qs = _check_qubits(qubits)
    n = len(qs)
    for i in range(n):
        c.h(qs[i])
        for j in range(i + 1, n):
            c.cp(qs[j], qs[i], math.pi / 2 ** (j - i))
    for i in range(n // 2):
        c.swap(qs[i], qs[n - 1 - i])
    return c


def inverse_qft(c: Circuit, qubits) -> Circuit:
    """Exact inverse of :func:`qft`: SWAPs first, then the H/CP ladder
    in reverse order with negated controlled-phase angles."""
    qs = _check_qubits(qubits)
    n = len(qs)
    for i in range(n // 2):
        c.swap(qs[i], qs[n - 1 - i])
    for i in reversed(range(n)):
        for j in reversed(range(i + 1, n)):
            c.cp(qs[j], qs[i], -math.pi / 2 ** (j - i))
        c.h(qs[i])
    return c


def _as_journal_store(root_or_store) -> JournalStore | None:
    if root_or_store is None or isinstance(root_or_store, JournalStore):
        return root_or_store
    return JournalStore(Path(root_or_store) / "journal")


def phase_estimation(root_or_store, phi: float, n_count: int = 3) -> dict:
    """Estimate ``phi`` for U = diag(1, exp(2*pi*i*phi)), eigenstate |1>.

    ``root_or_store`` is a JournalStore, a root path (the journal then
    lives at ``<root>/journal``), or None for an in-memory circuit.
    Layout: counting qubits 0..n_count-1 (MSB-first), target qubit
    ``n_count`` prepared in |1>. Controlled-U**(2**k) is a CP gate
    counting->target with angle 2*pi*phi*2**k; per the verified
    MSB-first convention, counting position i drives power
    2**(n_count-1-i). Then the inverse QFT acts on the counting
    register and its marginal probabilities are read out.
    """
    if not 0.0 <= phi < 1.0:
        raise ValueError("phi must be in [0, 1)")
    if n_count < 1:
        raise ValueError("n_count must be >= 1")
    store = _as_journal_store(root_or_store)
    counting = list(range(n_count))
    target = n_count
    c = Circuit(n_count + 1, store=store)
    c.x(target)  # eigenstate |1> of U
    for q in counting:
        c.h(q)
    for i, q in enumerate(counting):
        power = 2 ** (n_count - 1 - i)
        c.cp(q, target, 2 * math.pi * phi * power)
    inverse_qft(c, counting)

    # Marginal probabilities over the counting register (target traced
    # out); counting bits are the first n_count bits, MSB-first.
    probs: dict[str, float] = {}
    for bits, p in c.probabilities().items():
        key = bits[:n_count]
        probs[key] = probs.get(key, 0.0) + p
    peak = max(probs, key=probs.get)
    estimated = int(peak, 2) / 2 ** n_count
    return {
        "true_phi": phi,
        "phi": phi,
        "n_count": n_count,
        "estimated_phi": estimated,
        "peak_bitstring": peak,
        "peak_probability": probs[peak],
        "peak_prob": probs[peak],
        "probabilities": probs,
        "journal_length": len(c.journal),
        "replay": c.replay(),
        "journal_chain_verify": store.verify_chain() if store is not None else None,
        "journal_store_verify": store.verify_store() if store is not None else None,
        "circuit": c,
    }
