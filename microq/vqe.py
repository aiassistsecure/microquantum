"""S6: VQE-style variational loop with journaled parameter checkpoints
and rollback to best-so-far via time travel.

A fixed 2-qubit Hamiltonian, a small RY/CNOT ansatz built only from
journaled Circuit gates, exact expectation values from the state
vector (no sampling), and a deterministic Nelder-Mead optimizer
implemented in pure NumPy (no SciPy, no randomness). Every optimizer
evaluation is one iteration: the ansatz is built on a Circuit backed
by a JournalStore (real NEDB), and the iteration's params + energy +
final state hash are persisted as a parameter checkpoint in a NEDB
collection on the same database (``vqe_checkpoints``, keyed by
iteration), sealed by NEDB's own hash chain just like the journal.
After the loop, the best-so-far iteration is restored via the
existing time-travel machinery (``Circuit.rewind`` to that
iteration's checkpoint, states fetched via
``JournalStore.get_checkpoint``) and the restoration is proved by
hash, params, and re-evaluated energy.

Honest framing: classical simulation of quantum mathematics on a
state vector. No quantum hardware, no speedup, no sampling noise --
expectations are exact linear algebra, and the "optimizer" is a
textbook classical simplex search. V100 numbers are UNMEASURED.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .circuit import Circuit
from .journal import JournalStore, state_hash
from .state import I2, X, Y, Z

_PAULI = {"I": I2, "X": X, "Y": Y, "Z": Z}

#: Fixed 2-qubit Hamiltonian as (coefficient, Pauli string) terms.
#: The first character acts on qubit 0 (the MSB, as everywhere here).
#: H couples the |01>/|10> and |00>/|11> blocks, so its ground state
#: is entangled and its ground energy (-2.220656...) is not any
#: basis state's energy -- non-trivial for a 4-parameter ansatz.
HAMILTONIAN_TERMS: list[tuple[float, str]] = [
    (1.0, "ZZ"),
    (0.6, "XX"),
    (0.4, "YY"),
    (-0.5, "ZI"),
    (0.2, "IZ"),
]

N_QUBITS = 2
N_PARAMS = 4
#: Deterministic optimizer start point (fixed, no randomness).
START_PARAMS = (0.3, -0.2, 0.1, 0.4)
#: Ansatz gates per iteration (RY, RY, CNOT, RY, RY) -- the journal
#: entries of iteration ``i`` are entries ``i*GATES_PER_ITERATION``
#: through ``(i+1)*GATES_PER_ITERATION - 1`` of the run's store.
GATES_PER_ITERATION = 5

CHECKPOINT_COLLECTION = "vqe_checkpoints"


def hamiltonian_matrix(terms=HAMILTONIAN_TERMS) -> np.ndarray:
    """The 4x4 Hamiltonian matrix, built by Kronecker products."""
    H = np.zeros((4, 4), dtype=np.complex128)
    for coeff, paulis in terms:
        H = H + coeff * np.kron(_PAULI[paulis[0]], _PAULI[paulis[1]])
    return H


def exact_ground_energy(terms=HAMILTONIAN_TERMS) -> float:
    return float(np.linalg.eigvalsh(hamiltonian_matrix(terms))[0])


def _psi_of(source) -> np.ndarray:
    if isinstance(source, Circuit):
        return source.state.to_numpy()
    if hasattr(source, "to_numpy"):  # StateVector
        return source.to_numpy()
    return np.asarray(source, dtype=np.complex128)


def expectation(source, terms=HAMILTONIAN_TERMS) -> float:
    """Exact <H> of a Circuit, StateVector, or amplitude array.

    Computed as psi-dagger H psi from the full state vector -- no
    sampling, so the value is exact up to float rounding.
    """
    psi = _psi_of(source)
    return float(np.real(np.vdot(psi, hamiltonian_matrix(terms) @ psi)))


def build_ansatz(circuit: Circuit, params) -> Circuit:
    """RY layer, CNOT entanglement, RY layer (4 params, 5 gates)."""
    p = [float(x) for x in params]
    if len(p) != N_PARAMS:
        raise ValueError(f"ansatz needs {N_PARAMS} params")
    circuit.ry(0, p[0])
    circuit.ry(1, p[1])
    circuit.cnot(0, 1)
    circuit.ry(0, p[2])
    circuit.ry(1, p[3])
    return circuit


def ansatz_state(params) -> np.ndarray:
    """The (unjournaled) state vector of the ansatz at ``params``."""
    c = Circuit(N_QUBITS)
    build_ansatz(c, params)
    return c.state.to_numpy()


# -- parameter checkpoints (NEDB-backed, on the journal's database) -----

def _record_checkpoint(store: JournalStore, iteration: int, params,
                       energy: float, state_hash_: str, journal_seq: int) -> dict:
    if store.db is None:
        raise RuntimeError("VQE checkpoints need the real NEDB-backed JournalStore")
    rec = {
        "iteration": int(iteration),
        "params": [float(x) for x in params],
        "energy": float(energy),
        "state_hash": state_hash_,
        "journal_seq": int(journal_seq),
    }
    store.db.put(CHECKPOINT_COLLECTION, str(iteration), dict(rec))
    return rec


def _as_journal_store(root_or_store) -> JournalStore:
    if isinstance(root_or_store, JournalStore):
        return root_or_store
    return JournalStore(Path(root_or_store) / "journal")


def load_checkpoints(root_or_store) -> list[dict]:
    """Read every persisted parameter checkpoint back from NEDB, in
    iteration order (a fresh store view when given a root path)."""
    store = _as_journal_store(root_or_store)
    if store.db is None:
        raise RuntimeError("VQE checkpoints need the real NEDB-backed JournalStore")
    out, i = [], 0
    while (doc := store.db.get(CHECKPOINT_COLLECTION, str(i))) is not None:
        out.append({k: v for k, v in doc.items() if not k.startswith("_")})
        i += 1
    return out


# -- rollback via time travel --------------------------------------------

def rollback_best(root_or_store, checkpoints: list[dict] | None = None) -> dict:
    """Restore the best-so-far checkpoint via Circuit.rewind.

    Picks the lowest-energy checkpoint, rebuilds a Circuit whose
    journal is that iteration's journaled entries (read back from the
    store), rewinds it to the iteration's final checkpoint, and
    reports the restored state hash, the params recovered from the
    journaled RY entries, and the re-evaluated energy.
    """
    store = _as_journal_store(root_or_store)
    ckpts = checkpoints if checkpoints is not None else load_checkpoints(store)
    if not ckpts:
        raise ValueError("no VQE checkpoints to roll back to")
    best = min(ckpts, key=lambda r: r["energy"])
    i = int(best["iteration"])
    entries = [dict(e) for e in
               store.entries[i * GATES_PER_ITERATION:(i + 1) * GATES_PER_ITERATION]]
    if len(entries) != GATES_PER_ITERATION:
        raise ValueError("best iteration's journal entries not found in store")
    c = Circuit(N_QUBITS, store=store)
    c.journal = entries
    c.rewind(len(entries))  # restores state via get_checkpoint(state_hash)
    params = [float(e["params"][0]) for e in c.journal if e["gate"] == "RY"]
    return {
        "circuit": c,
        "iteration": i,
        "params": params,
        "energy": expectation(c),
        "state_hash": c.current_hash(),
        "replay": c.replay(),
    }


# -- deterministic Nelder-Mead (NumPy only) --------------------------------

class _EvalLimit(Exception):
    pass


def _nelder_mead(objective, x0, step: float = 0.5, max_evals: int = 110) -> None:
    """Textbook Nelder-Mead; every objective call is one VQE iteration.
    Deterministic: fixed start, fixed initial simplex, no randomness."""
    n = len(x0)
    simplex = [np.array(x0, dtype=float)]
    for k in range(n):
        x = np.array(x0, dtype=float)
        x[k] += step
        simplex.append(x)
    try:
        values = [objective(x) for x in simplex]
        while True:
            order = np.argsort(values)
            simplex = [simplex[k] for k in order]
            values = [values[k] for k in order]
            centroid = np.mean(simplex[:-1], axis=0)
            xr = centroid + (centroid - simplex[-1])
            fr = objective(xr)
            if fr < values[0]:
                xe = centroid + 2.0 * (centroid - simplex[-1])
                fe = objective(xe)
                simplex[-1], values[-1] = (xe, fe) if fe < fr else (xr, fr)
            elif fr < values[-2]:
                simplex[-1], values[-1] = xr, fr
            else:
                xc = centroid + 0.5 * (simplex[-1] - centroid)
                fc = objective(xc)
                if fc < values[-1]:
                    simplex[-1], values[-1] = xc, fc
                else:  # contraction failed: shrink toward the best point
                    for k in range(1, n + 1):
                        simplex[k] = simplex[0] + 0.5 * (simplex[k] - simplex[0])
                        values[k] = objective(simplex[k])
    except _EvalLimit:
        return


# -- the loop ---------------------------------------------------------------

def run_vqe(root, max_evals: int = 110) -> dict:
    """Run the VQE-style loop under ``root`` (journal at <root>/journal).

    Each evaluation resets the run's single Circuit to |00> via
    rewind(0), builds the ansatz (journaled, hash-chained), evaluates
    the exact energy, and persists a parameter checkpoint to NEDB.
    After the optimizer stops, best-so-far is restored via
    :func:`rollback_best` and the restoration is verified.
    """
    store = _as_journal_store(root)
    circuit = Circuit(N_QUBITS, store=store)
    history: list[dict] = []
    evals = 0

    def objective(params: np.ndarray) -> float:
        nonlocal evals
        if evals >= max_evals:
            raise _EvalLimit
        circuit.rewind(0)
        build_ansatz(circuit, params)
        energy = expectation(circuit)
        rec = _record_checkpoint(store, evals, params, energy,
                                 circuit.current_hash(), circuit.journal[-1]["seq"])
        history.append(rec)
        evals += 1
        return energy

    _nelder_mead(objective, np.array(START_PARAMS, dtype=float),
                 max_evals=max_evals)

    best = min(history, key=lambda r: r["energy"])
    rolled = rollback_best(store, history)
    rollback_verified = bool(
        rolled["state_hash"] == best["state_hash"]
        and np.allclose(rolled["params"], best["params"], atol=1e-12)
        and abs(rolled["energy"] - best["energy"]) < 1e-9
    )
    return {
        "history": history,
        "iterations": len(history),
        "best_iteration": int(best["iteration"]),
        "best_params": best["params"],
        "best_energy": best["energy"],
        "best_state_hash": best["state_hash"],
        "exact_ground_energy": exact_ground_energy(),
        "start_energy": history[0]["energy"],
        "final_energy": history[-1]["energy"],
        "final_params": history[-1]["params"],
        "restored_params": rolled["params"],
        "restored_energy": rolled["energy"],
        "restored_state_hash": rolled["state_hash"],
        "rollback_verified": rollback_verified,
        "rollback_replay": rolled["replay"],
        "journal_length": len(store.entries),
        "journal_chain_verify": store.verify_chain(),
        "journal_store_verify": store.verify_store(),
        "circuit": rolled["circuit"],
    }
