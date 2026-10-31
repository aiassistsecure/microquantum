"""S3 backend tests: numpy parity, selection, cupy-unavailable behaviour.

CuPy tests skip (not fail) when cupy / a CUDA device is unavailable —
on this VM cupy is not installed. V100 performance is UNMEASURED.
"""
import numpy as np
import pytest

from microq import Circuit, JournalStore
from microq.backend import (
    HAS_CUPY, BackendUnavailableError, CupyBackend, NumpyBackend, get_backend,
)
from microq.state import StateVector, gate_matrix


def test_get_backend_numpy_and_instance():
    b = get_backend("numpy")
    assert isinstance(b, NumpyBackend) and b.name == "numpy" and not b.is_gpu
    assert get_backend(b) is b
    assert get_backend(None).name == "numpy"


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        get_backend("tpu")
    with pytest.raises(ValueError):
        StateVector(1, backend="tpu")


def test_backend_gate_parity_direct():
    b = NumpyBackend()
    psi = b.zeros(4); psi[0] = 1.0
    psi = b.apply_1q(psi, gate_matrix("H"), 0, 2)
    psi = b.cnot(psi, 0, 1, 2)
    host = b.to_numpy(psi)
    assert host == pytest.approx(np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2))
    # cz + swap surface via the backend directly: CZ flips |11> phase,
    # swap of the symmetric Bell state leaves the (phased) state unchanged
    psi2 = b.swap(b.cz(psi, 0, 1, 2), 0, 1, 2)
    phased = host.copy(); phased[3] *= -1
    assert np.allclose(b.to_numpy(psi2), phased)


def test_statevector_explicit_numpy_matches_default():
    a, b = StateVector(2), StateVector(2, backend="numpy")
    for sv in (a, b):
        sv.apply_1q(gate_matrix("H"), 0); sv.cnot(0, 1); sv.apply_1q(gate_matrix("RY", (0.3,)), 1)
    assert np.array_equal(a.to_numpy(), b.to_numpy())  # bit-compatible CPU path
    assert isinstance(b.psi, np.ndarray) and b.backend_name == "numpy"


def test_circuit_backend_selection_and_replay(tmp_path):
    c = Circuit(2, store=JournalStore(tmp_path / "d"), backend="numpy")
    assert c.backend_name == "numpy" and c.backend.is_fallback is False
    c.h(0).cnot(0, 1)
    assert c.probabilities()["00"] == pytest.approx(0.5)
    assert c.replay() is True
    c.rewind(1); assert len(c.journal) == 1


def test_host_backend_transfer_roundtrip():
    b = NumpyBackend()
    arr = np.array([1, 0, 0, 0], dtype=np.complex128)
    assert np.array_equal(b.to_numpy(b.from_numpy(arr)), arr)
    sv = StateVector(2, backend=b)
    sv.apply_1q(gate_matrix("X"), 1)
    assert np.array_equal(sv.to_numpy(), np.array([0, 1, 0, 0], dtype=complex))
    assert sv.copy().backend is b or sv.copy().backend_name == "numpy"


@pytest.mark.skipif(HAS_CUPY, reason="cupy available here; unavailable-path N/A")
def test_cupy_unavailable_fails_clearly():
    with pytest.raises(BackendUnavailableError):
        get_backend("cupy")
    with pytest.raises(BackendUnavailableError):
        CupyBackend()
    with pytest.raises(BackendUnavailableError):
        StateVector(1, backend="cupy")
    with pytest.raises(BackendUnavailableError):
        Circuit(1, backend="cupy")


@pytest.mark.skipif(HAS_CUPY, reason="cupy available here; fallback-path N/A")
def test_cupy_explicit_fallback_is_flagged_never_silent():
    sv = StateVector(1, backend="cupy", backend_fallback=True)
    assert sv.backend_name == "numpy"
    assert sv.backend.is_fallback is True and sv.backend.requested_name == "cupy"
    c = Circuit(1, backend="cupy", backend_fallback=True)
    assert c.backend.is_fallback is True


@pytest.mark.skipif(not HAS_CUPY, reason="cupy / CUDA device unavailable on this host")
def test_cupy_bell_parity():
    sv = StateVector(2, backend="cupy")
    sv.apply_1q(gate_matrix("H"), 0); sv.cnot(0, 1)
    expected = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    assert sv.to_numpy() == pytest.approx(expected)
