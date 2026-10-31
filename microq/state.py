"""State-vector engine: n-qubit complex128 state, gates via a backend.

Classical simulation of quantum math. Honest framing: no quantum
hardware is involved; this is linear algebra on NumPy (default) or,
when available on a GPU host, CuPy — V100 performance is UNMEASURED.
"""
from __future__ import annotations

import numpy as np

from .backend import Backend, get_backend

# --- Single-qubit gates -------------------------------------------------
I2 = np.eye(2, dtype=np.complex128)
H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
S = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)
SDG = np.array([[1, 0], [0, -1j]], dtype=np.complex128)
TDG = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=np.complex128)

SINGLE = {"H": H, "X": X, "Y": Y, "Z": Z, "S": S, "T": T, "SDG": SDG, "TDG": TDG, "I": I2}


def RX(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)


def RY(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def RZ(theta: float) -> np.ndarray:
    return np.array(
        [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]],
        dtype=np.complex128,
    )


PARAMETRIC = {"RX": RX, "RY": RY, "RZ": RZ}


def gate_matrix(name: str, params: tuple[float, ...] = ()) -> np.ndarray:
    if name in SINGLE:
        return SINGLE[name]
    if name in PARAMETRIC:
        if len(params) != 1:
            raise ValueError(f"{name} needs exactly one angle param")
        return PARAMETRIC[name](float(params[0]))
    raise ValueError(f"unknown single-qubit gate {name!r}")


class StateVector:
    """An n-qubit pure state. Qubit 0 is the most-significant index bit.

    ``psi`` is a backend array (NumPy by default, CuPy on a GPU host).
    Use :meth:`to_numpy` for a guaranteed host array (journal, hashing).
    """

    def __init__(self, n_qubits: int, state=None, backend="numpy",
                 backend_fallback: bool = False):
        if n_qubits < 1:
            raise ValueError("need at least 1 qubit")
        self.n = n_qubits
        self.backend: Backend = get_backend(backend, backend_fallback=backend_fallback)
        if state is None:
            self.psi = self.backend.zeros(2**n_qubits)
            self.psi[0] = 1.0
        else:
            host = self.backend.to_numpy(state) if not isinstance(state, np.ndarray) else state
            psi = np.asarray(host, dtype=np.complex128)
            if psi.shape != (2**n_qubits,):
                raise ValueError("state shape mismatch")
            self.psi = self.backend.from_numpy(psi.copy())

    @property
    def backend_name(self) -> str:
        return self.backend.name

    def to_numpy(self) -> np.ndarray:
        return self.backend.to_numpy(self.psi)

    def copy(self) -> "StateVector":
        return StateVector(self.n, self.to_numpy(), backend=self.backend)

    # -- gate application (delegated to the backend) ---------------------
    def apply_1q(self, gate: np.ndarray, target: int) -> None:
        if not 0 <= target < self.n:
            raise ValueError("target out of range")
        self.psi = self.backend.apply_1q(self.psi, gate, target, self.n)

    def apply_2q(self, gate4: np.ndarray, a: int, b: int) -> None:
        if a == b or not (0 <= a < self.n and 0 <= b < self.n):
            raise ValueError("bad 2-qubit targets")
        self.psi = self.backend.apply_2q(self.psi, gate4, a, b, self.n)

    def cnot(self, control: int, target: int) -> None:
        if control == target or not (0 <= control < self.n and 0 <= target < self.n):
            raise ValueError("bad 2-qubit targets")
        self.psi = self.backend.cnot(self.psi, control, target, self.n)

    def cz(self, a: int, b: int) -> None:
        if a == b or not (0 <= a < self.n and 0 <= b < self.n):
            raise ValueError("bad 2-qubit targets")
        self.psi = self.backend.cz(self.psi, a, b, self.n)

    def swap(self, a: int, b: int) -> None:
        if a == b or not (0 <= a < self.n and 0 <= b < self.n):
            raise ValueError("bad 2-qubit targets")
        self.psi = self.backend.swap(self.psi, a, b, self.n)

    # -- observation -----------------------------------------------------
    def probabilities(self) -> np.ndarray:
        return np.abs(self.to_numpy()) ** 2

    def prob_dict(self) -> dict[str, float]:
        fmt = f"0{self.n}b"
        return {format(i, fmt): float(p) for i, p in enumerate(self.probabilities())}

    def sample(self, shots: int = 1000, rng: np.random.Generator | None = None) -> dict[str, int]:
        rng = rng or np.random.default_rng()
        outcomes = rng.choice(2**self.n, size=shots, p=self.probabilities())
        fmt = f"0{self.n}b"
        counts: dict[str, int] = {}
        for o in outcomes:
            k = format(int(o), fmt)
            counts[k] = counts.get(k, 0) + 1
        return dict(sorted(counts.items()))

    def fidelity(self, other: "StateVector") -> float:
        return float(abs(np.vdot(self.to_numpy(), other.to_numpy())) ** 2)

    def norm(self) -> float:
        return float(np.linalg.norm(self.to_numpy()))
