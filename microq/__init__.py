"""microquantum — classical state-vector simulation of quantum circuits,
with a NEDB-shaped execution journal (provenance, hash chain, time travel).

This is classical simulation of quantum math on NumPy. It is NOT quantum
hardware and claims no quantum speedup.
"""
from .backend import BackendUnavailableError, CupyBackend, NumpyBackend, get_backend
from .circuit import Circuit
from .debug import (EntropyWatch, FidelityWatch, ProbabilityWatch,
                    diff_states, find_divergence, inspect_at,
                    run_with_watchpoints)
from .journal import JournalStore, state_hash
from .state import StateVector

__all__ = ["Circuit", "JournalStore", "StateVector", "state_hash",
           "inspect_at", "diff_states", "find_divergence",
           "ProbabilityWatch", "FidelityWatch", "EntropyWatch",
           "run_with_watchpoints",
           "NumpyBackend", "CupyBackend", "get_backend", "BackendUnavailableError"]
__version__ = "0.1.0"
