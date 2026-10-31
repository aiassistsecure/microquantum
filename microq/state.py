"""State-vector engine: n-qubit complex128 state, gates via reshape/transpose.

Classical simulation of quantum math on NumPy. Honest framing: no quantum
hardware is involved; this is linear algebra.
"""
from __future__ import annotations

import numpy as np

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
    """An n-qubit pure state. Qubit 0 is the most-significant index bit."""

    def __init__(self, n_qubits: int, state: np.ndarray | None = None):
        if n_qubits < 1:
            raise ValueError("need at least 1 qubit")
        self.n = n_qubits
        if state is None:
            self.psi = np.zeros(2**n_qubits, dtype=np.complex128)
            self.psi[0] = 1.0
        else:
            psi = np.asarray(state, dtype=np.complex128)
            if psi.shape != (2**n_qubits,):
                raise ValueError("state shape mismatch")
            self.psi = psi.copy()

    def copy(self) -> "StateVector":
        return StateVector(self.n, self.psi)

    # -- gate application ------------------------------------------------
    def apply_1q(self, gate: np.ndarray, target: int) -> None:
        """Apply a 2x2 gate to `target` via reshape + tensordot."""
        if not 0 <= target < self.n:
            raise ValueError("target out of range")
        t = self.psi.reshape([2] * self.n)
        t = np.tensordot(gate, t, axes=([1], [target]))
        t = np.moveaxis(t, 0, target)
        self.psi = t.reshape(-1)

    def apply_2q(self, gate4: np.ndarray, a: int, b: int) -> None:
        """Apply a 4x4 gate acting on qubits (a, b), a = control/high bit."""
        if a == b or not (0 <= a < self.n and 0 <= b < self.n):
            raise ValueError("bad 2-qubit targets")
        t = self.psi.reshape([2] * self.n)
        g = gate4.reshape(2, 2, 2, 2)
        t = np.tensordot(g, t, axes=([2, 3], [a, b]))
        # tensordot puts the two output axes first, in order (a-out, b-out)
        t = np.moveaxis(t, [0, 1], [a, b])
        self.psi = t.reshape(-1)

    def cnot(self, control: int, target: int) -> None:
        gate = np.zeros((4, 4), dtype=np.complex128)
        gate[0, 0] = gate[1, 1] = 1.0  # control 0: identity on target
        gate[3, 2] = gate[2, 3] = 1.0  # control 1: X on target
        self.apply_2q(gate, control, target)

    def cz(self, a: int, b: int) -> None:
        gate = np.diag([1, 1, 1, -1]).astype(np.complex128)
        self.apply_2q(gate, a, b)

    def swap(self, a: int, b: int) -> None:
        gate = np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
            dtype=np.complex128,
        )
        self.apply_2q(gate, a, b)

    # -- observation -----------------------------------------------------
    def probabilities(self) -> np.ndarray:
        return np.abs(self.psi) ** 2

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
        return float(abs(np.vdot(self.psi, other.psi)) ** 2)

    def norm(self) -> float:
        return float(np.linalg.norm(self.psi))
