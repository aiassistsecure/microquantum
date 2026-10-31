"""Debugger inspection core (D1): inspect / diff journaled states.

Classical state-vector simulation only — no quantum hardware.

Design: the state at any journal ``seq`` is obtained by *checkpoint
restore* — ``store.get_checkpoint(state_hash)`` using the hash recorded
in that journal entry (the genesis checkpoint for ``seq == 0``) — never
by re-running gates from scratch when a checkpoint exists. Inspection
never mutates the passed circuit: its state and journal are only read.

Storeless circuits: there is no checkpoint store to restore from, so
:func:`inspect_at` / :func:`diff_states` raise ``ValueError`` rather
than silently falling back to replaying gates. Qubit 0 is the MSB,
matching ``microq.state`` / ``microq.circuit``.
"""
from __future__ import annotations

import numpy as np

from .journal import JournalStore, state_hash

_TOP_K = 5


def _state_at(circuit, seq: int) -> tuple[np.ndarray, str]:
    """(psi, state_hash) at journal `seq` via checkpoint restore."""
    JournalStore._check_journal_seq(seq, len(circuit.journal))
    if circuit.store is None:
        raise ValueError(
            "inspect/diff needs a JournalStore-backed circuit: states are "
            "restored from checkpoints, and a storeless circuit has none"
        )
    if seq == 0:
        h = getattr(circuit, "_genesis_hash", None)
        if h is None:  # genesis is |0...0> by construction
            genesis = np.zeros(2 ** circuit.n, dtype=np.complex128)
            genesis[0] = 1.0
            h = state_hash(genesis)
    else:
        h = circuit.journal[seq - 1]["state_hash"]
    try:
        psi = np.array(circuit.store.get_checkpoint(h), dtype=np.complex128)
    except FileNotFoundError as exc:
        raise ValueError(f"no checkpoint stored for state at seq {seq}") from exc
    return psi, h


def _reduced_state(psi: np.ndarray, n: int, qubit: int) -> np.ndarray:
    """2x2 reduced density matrix for `qubit` (qubit 0 = MSB)."""
    tensor = psi.reshape([2] * n)
    moved = np.moveaxis(tensor, qubit, 0).reshape(2, -1)
    return moved @ moved.conj().T


def _bloch(rho: np.ndarray) -> tuple[float, float, float]:
    x = float(2.0 * rho[0, 1].real)
    y = float(-2.0 * rho[0, 1].imag)
    z = float((rho[0, 0] - rho[1, 1]).real)
    return (x, y, z)


def _entropy_bits(rho: np.ndarray) -> float:
    eig = np.clip(np.linalg.eigvalsh(rho).real, 0.0, 1.0)
    eig = eig[eig > 0.0]
    return float(-np.sum(eig * np.log2(eig))) if eig.size else 0.0


def inspect_at(circuit, seq: int) -> dict:
    """Full inspection report for the state at journal `seq`.

    Returns a dict with ``seq``, ``n_qubits``, ``state_hash``,
    ``probabilities`` (bitstring -> prob, qubit 0 first/MSB),
    ``amplitudes`` (bitstring -> complex), ``top_amplitudes`` (top 5
    by |amplitude|: bitstring/amplitude/prob), and per-qubit data both
    as parallel lists (``reduced_states``, ``bloch_vectors``,
    ``entanglement_entropies``) and as ``qubits`` dicts. ``seq == 0``
    is genesis |0...0>; out-of-range raises ``ValueError``.
    """
    psi, h = _state_at(circuit, seq)
    n = circuit.n
    fmt = f"0{n}b"
    probs = np.abs(psi) ** 2
    probabilities = {format(i, fmt): float(p) for i, p in enumerate(probs)}
    amplitudes = {format(i, fmt): complex(a) for i, a in enumerate(psi)}
    order = np.argsort(-np.abs(psi), kind="stable")[:_TOP_K]
    top = [{"bitstring": format(int(i), fmt), "amplitude": complex(psi[i]),
            "prob": float(probs[i])} for i in order]
    reduced, bloch, entropies, qubits = [], [], [], []
    for q in range(n):
        rho = _reduced_state(psi, n, q)
        b = _bloch(rho)
        s = _entropy_bits(rho)
        reduced.append(rho)
        bloch.append(b)
        entropies.append(s)
        qubits.append({"qubit": q, "reduced_state": rho, "bloch_vector": b,
                       "bloch": {"x": b[0], "y": b[1], "z": b[2]},
                       "entanglement_entropy": s})
    return {"seq": seq, "n_qubits": n, "state_hash": h,
            "probabilities": probabilities, "amplitudes": amplitudes,
            "top_amplitudes": top, "reduced_states": reduced,
            "bloch_vectors": bloch, "entanglement_entropies": entropies,
            "qubits": qubits}


def diff_states(circuit, seq_a: int, seq_b: int) -> dict:
    """Compare the states at journal `seq_a` and `seq_b` (same circuit).

    Both states come from checkpoint restore (see :func:`inspect_at`);
    the circuit is not mutated. Returns ``fidelity``
    (|⟨psi_a|psi_b⟩|^2), per-bitstring ``per_amplitude`` details
    (amplitudes, probs, and their deltas, b − a), the flat
    ``amplitude_deltas`` / ``prob_deltas`` maps, and ``largest_movers``
    (top 5 by |delta prob|, ties broken by |delta amplitude|).
    """
    psi_a, hash_a = _state_at(circuit, seq_a)
    psi_b, hash_b = _state_at(circuit, seq_b)
    n = circuit.n
    fmt = f"0{n}b"
    fidelity = float(abs(np.vdot(psi_a, psi_b)) ** 2)
    per_amplitude, amp_deltas, prob_deltas, movers = {}, {}, {}, []
    for i in range(2 ** n):
        bs = format(i, fmt)
        da = complex(psi_b[i] - psi_a[i])
        dp = float(abs(psi_b[i]) ** 2 - abs(psi_a[i]) ** 2)
        per_amplitude[bs] = {"amplitude_a": complex(psi_a[i]),
                             "amplitude_b": complex(psi_b[i]),
                             "delta_amplitude": da,
                             "prob_a": float(abs(psi_a[i]) ** 2),
                             "prob_b": float(abs(psi_b[i]) ** 2),
                             "delta_prob": dp}
        amp_deltas[bs] = da
        prob_deltas[bs] = dp
        movers.append({"bitstring": bs, "delta_prob": dp,
                       "delta_amplitude": da,
                       "prob_a": per_amplitude[bs]["prob_a"],
                       "prob_b": per_amplitude[bs]["prob_b"],
                       "amplitude_a": complex(psi_a[i]),
                       "amplitude_b": complex(psi_b[i])})
    movers.sort(key=lambda m: (-abs(m["delta_prob"]),
                               -abs(m["delta_amplitude"]), m["bitstring"]))
    return {"seq_a": seq_a, "seq_b": seq_b, "state_hash_a": hash_a,
            "state_hash_b": hash_b, "fidelity": fidelity,
            "per_amplitude": per_amplitude,
            "amplitude_deltas": amp_deltas, "prob_deltas": prob_deltas,
            "largest_movers": movers[:_TOP_K]}
