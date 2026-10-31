"""Circuit: a state-vector plus a journaled, rewindable, forkable history."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .journal import JournalStore, state_hash
from .state import StateVector, gate_matrix

CHECKPOINT_EVERY = 1  # slice 1: checkpoint every step (cheap at demo scale);
# a stride (e.g. every 16) drops in here without changing the API.


class Circuit:
    def __init__(self, n_qubits: int, store: JournalStore | None = None,
                 _state: StateVector | None = None, backend="numpy",
                 backend_fallback: bool = False):
        """``backend``: "numpy" (default) or "cupy". Requesting cupy when
        unavailable raises BackendUnavailableError unless
        ``backend_fallback=True`` (explicit NumPy fallback, never silent).
        Journal/checkpoints always store host NumPy arrays."""
        self.n = n_qubits
        self.store = store
        self.state = _state or StateVector(n_qubits, backend=backend,
                                           backend_fallback=backend_fallback)
        self.journal: list[dict] = []  # in-memory mirror of journal entries
        if self.store is not None:
            self._genesis_hash = self.store.put_checkpoint(self.state.to_numpy())

    @property
    def backend(self):
        return self.state.backend

    @property
    def backend_name(self) -> str:
        return self.state.backend_name

    # Backend seam: gate math goes through StateVector's backend
    # (microq.backend); Circuit and the journal APIs are unchanged.

    # -- gate application (journaled) --------------------------------------
    def _record(self, gate: str, targets: list[int], params: tuple[float, ...] = ()) -> dict:
        if self.store is not None:
            entry = self.store.append(gate, targets, list(params), self.state.to_numpy())
        else:
            seq = len(self.journal) + 1
            entry = {
                "seq": seq,
                "gate": gate,
                "targets": list(targets),
                "params": list(params),
                "state_hash": state_hash(self.state.to_numpy()),
                "timestamp": None,
            }
        self.journal.append(entry)
        return entry

    def apply(self, gate: str, *targets: int, params: tuple[float, ...] = ()) -> "Circuit":
        gate = gate.upper()
        if gate in ("CNOT", "CX"):
            if len(targets) != 2:
                raise ValueError("CNOT needs (control, target)")
            self.state.cnot(targets[0], targets[1])
        elif gate == "CZ":
            self.state.cz(targets[0], targets[1])
        elif gate == "CP":
            if len(targets) != 2:
                raise ValueError("CP needs (control, target)")
            if len(params) != 1:
                raise ValueError("CP needs exactly one angle param (theta)")
            theta = float(params[0])
            mat = np.diag([1, 1, 1, np.exp(1j * theta)]).astype(np.complex128)
            self.state.apply_2q(mat, targets[0], targets[1])
        elif gate == "SWAP":
            self.state.swap(targets[0], targets[1])
        else:
            if len(targets) != 1:
                raise ValueError(f"{gate} needs exactly one target")
            self.state.apply_1q(gate_matrix(gate, params), targets[0])
        self._record(gate, list(targets), params)
        return self

    # Convenience wrappers
    def h(self, q: int): return self.apply("H", q)
    def x(self, q: int): return self.apply("X", q)
    def y(self, q: int): return self.apply("Y", q)
    def z(self, q: int): return self.apply("Z", q)
    def s(self, q: int): return self.apply("S", q)
    def t(self, q: int): return self.apply("T", q)
    def rx(self, q: int, theta: float): return self.apply("RX", q, params=(theta,))
    def ry(self, q: int, theta: float): return self.apply("RY", q, params=(theta,))
    def rz(self, q: int, theta: float): return self.apply("RZ", q, params=(theta,))
    def cnot(self, c: int, t: int): return self.apply("CNOT", c, t)
    def cz(self, a: int, b: int): return self.apply("CZ", a, b)
    def cp(self, a: int, b: int, theta: float): return self.apply("CP", a, b, params=(theta,))
    def swap(self, a: int, b: int): return self.apply("SWAP", a, b)

    # -- observation ---------------------------------------------------------
    def probabilities(self) -> dict[str, float]:
        return self.state.prob_dict()

    def sample(self, shots: int = 1000, seed: int | None = None) -> dict[str, int]:
        rng = np.random.default_rng(seed)
        return self.state.sample(shots, rng=rng)

    def current_hash(self) -> str:
        return state_hash(self.state.to_numpy())

    # -- provenance reads (AS OF / TRACE, in journal seq) ------------------
    # `seq` here is this circuit's journal seq, NOT NEDB's global op
    # seq — see the seq-distinction note in microq/journal.py. The
    # circuit's view is bounded by its own journal (e.g. after rewind),
    # even when the backing store still holds later entries.
    def as_of(self, seq: int) -> dict | None:
        """The journal entry as of journal `seq`: seq 0 (genesis)
        returns None; out of range raises ValueError. Store-backed
        circuits source the entry from the store via
        `JournalStore.entry_as_of`."""
        JournalStore._check_journal_seq(seq, len(self.journal))
        if seq == 0:
            return None
        if self.store is not None:
            return self.store.entry_as_of(seq)
        return dict(self.journal[seq - 1])

    def trace(self, seq: int | None = None) -> dict:
        """The ordered, chain-verified journal prefix leading to `seq`
        (default: this circuit's head), in the same result shape as
        `JournalStore.trace`. Storeless circuits record no hash chain,
        so a non-empty storeless trace reports `verified` as None
        rather than making a claim it cannot back."""
        if seq is None:
            seq = len(self.journal)
        JournalStore._check_journal_seq(seq, len(self.journal))
        if self.store is not None:
            return self.store.trace(seq)
        entries = [dict(e) for e in self.journal[:seq]]
        chained = all("entry_hash" in e and "prev_hash" in e for e in entries)
        return {
            "seq": seq,
            "entries": entries,
            "verified": JournalStore._verify_entries(entries) if chained else None,
            "head_hash": entries[-1].get("entry_hash", "0" * 64) if entries else "0" * 64,
            "state_hash": entries[-1]["state_hash"] if entries else None,
        }

    # -- time travel ---------------------------------------------------------
    def _checkpoint_hash_at(self, seq: int) -> str:
        if seq == 0:
            if self.store is None:
                raise ValueError("rewind(0) needs a persistent store (genesis checkpoint)")
            return self._genesis_hash
        if not 1 <= seq <= len(self.journal):
            raise ValueError(f"seq {seq} out of journal range")
        return self.journal[seq - 1]["state_hash"]

    def rewind(self, seq: int) -> "Circuit":
        """Restore the checkpointed state at journal seq (0 = |0...0>)."""
        if self.store is None:
            raise ValueError("rewind needs a persistent JournalStore")
        h = self._checkpoint_hash_at(seq)
        psi = self.store.get_checkpoint(h)
        self.state = StateVector(self.n, psi, backend=self.state.backend)
        self.journal = self.journal[:seq]
        return self

    def fork(self, seq: int | None = None) -> "Circuit":
        """Branch a new independent circuit from the checkpoint at `seq`
        (default: current head). The fork shares the content-addressed
        checkpoint store but gets its own journal."""
        if self.store is None:
            raise ValueError("fork needs a persistent JournalStore")
        if seq is None:
            psi = self.state.to_numpy().copy()
            hist = list(self.journal)
        else:
            psi = self.store.get_checkpoint(self._checkpoint_hash_at(seq))
            hist = list(self.journal[:seq])
        child_store_root = Path(self.store.root) / f"fork-{len(list(Path(self.store.root).glob('fork-*')))}-{seq if seq is not None else 'head'}"
        child_store = JournalStore(child_store_root)
        child = Circuit(self.n, store=child_store,
                        _state=StateVector(self.n, psi, backend=self.state.backend))
        child.journal = hist
        for e in hist:  # fork history is the parent's chain prefix, persisted
            child_store.import_entry(e)
        return child

    def replay(self) -> bool:
        """Re-apply the journal from |0...0> and verify the final state hash
        matches the journaled head hash — the provenance proof."""
        if not self.journal:
            return self.current_hash() == state_hash(StateVector(self.n).to_numpy())
        fresh = Circuit(self.n, backend=self.state.backend)
        for e in self.journal:
            fresh.apply(e["gate"], *e["targets"], params=tuple(e["params"]))
        return fresh.current_hash() == self.journal[-1]["state_hash"]
