"""Array backends for the state-vector engine (S3).

Classical simulation of quantum math — a backend only chooses *where*
the linear algebra runs (host NumPy vs a CuPy/CUDA device array). It is
not quantum hardware and gives no quantum speedup.

Choice on unavailable CuPy (documented, never silent): requesting
``backend="cupy"`` raises :class:`BackendUnavailableError` unless the
caller passes ``backend_fallback=True``, in which case a NumPy backend
is returned with ``is_fallback=True`` and ``requested_name="cupy"`` so
callers can see they are NOT on GPU. V100 performance is UNMEASURED:
CuPy code here has never been run on the V100 in this project.
"""
from __future__ import annotations

import numpy as np

try:  # import-guarded: cupy is NOT installed on this VM
    import cupy as _cp
    _CUPY_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - depends on environment
    _cp = None
    _CUPY_IMPORT_ERROR = exc


def cupy_available() -> bool:
    """True only if cupy imports AND a CUDA device is visible."""
    if _cp is None:
        return False
    try:
        return int(_cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        return False


HAS_CUPY = cupy_available()


class BackendUnavailableError(RuntimeError):
    """Raised when a requested backend (e.g. cupy) is not available."""


class Backend:
    """Backend surface StateVector needs: array helpers + gate ops.

    ``psi``/gate arrays are arrays of ``self.xp``; observation helpers in
    StateVector convert to host NumPy via :meth:`to_numpy`.
    """

    name = "base"
    is_gpu = False
    is_fallback = False
    requested_name: str | None = None
    xp = np

    # -- array creation / conversion ------------------------------------
    def zeros(self, size: int):
        return self.xp.zeros(size, dtype=self.xp.complex128)

    def asarray(self, data):
        return self.xp.asarray(data, dtype=self.xp.complex128)

    def from_numpy(self, arr: np.ndarray):
        """Host NumPy array -> backend array."""
        return self.asarray(np.asarray(arr, dtype=np.complex128))

    def to_numpy(self, arr) -> np.ndarray:
        """Backend array -> host NumPy array (copy on GPU backends)."""
        return np.asarray(arr, dtype=np.complex128)

    def _gate(self, gate):
        return self.xp.asarray(gate, dtype=self.xp.complex128)

    # -- gate application (same math for every backend) ------------------
    def apply_1q(self, psi, gate, target: int, n: int):
        xp = self.xp
        t = psi.reshape([2] * n)
        t = xp.tensordot(self._gate(gate), t, axes=([1], [target]))
        t = xp.moveaxis(t, 0, target)
        return t.reshape(-1)

    def apply_2q(self, psi, gate4, a: int, b: int, n: int):
        xp = self.xp
        t = psi.reshape([2] * n)
        g = self._gate(gate4).reshape(2, 2, 2, 2)
        t = xp.tensordot(g, t, axes=([2, 3], [a, b]))
        t = xp.moveaxis(t, [0, 1], [a, b])
        return t.reshape(-1)

    def cnot(self, psi, control: int, target: int, n: int):
        gate = self.xp.zeros((4, 4), dtype=self.xp.complex128)
        gate[0, 0] = gate[1, 1] = 1.0
        gate[3, 2] = gate[2, 3] = 1.0
        return self.apply_2q(psi, gate, control, target, n)

    def cz(self, psi, a: int, b: int, n: int):
        gate = self.xp.diag(self.xp.array([1, 1, 1, -1], dtype=self.xp.complex128))
        return self.apply_2q(psi, gate, a, b, n)

    def swap(self, psi, a: int, b: int, n: int):
        gate = self.xp.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
            dtype=self.xp.complex128,
        )
        return self.apply_2q(psi, gate, a, b, n)


class NumpyBackend(Backend):
    name = "numpy"
    is_gpu = False
    xp = np

    def to_numpy(self, arr) -> np.ndarray:  # no copy semantics needed
        return np.asarray(arr, dtype=np.complex128)


class CupyBackend(Backend):
    """CuPy/CUDA backend for the V100. UNMEASURED on real hardware."""

    name = "cupy"
    is_gpu = True

    def __init__(self):
        if not HAS_CUPY:
            reason = f" ({_CUPY_IMPORT_ERROR})" if _CUPY_IMPORT_ERROR else ""
            raise BackendUnavailableError(
                "cupy backend unavailable: cupy not importable or no CUDA "
                f"device visible{reason}. Install cupy on a GPU host or "
                "pass backend_fallback=True for an explicit NumPy fallback."
            )
        self.xp = _cp

    def to_numpy(self, arr) -> np.ndarray:
        return self.xp.asnumpy(arr).astype(np.complex128, copy=False)


def get_backend(backend="numpy", *, backend_fallback: bool = False) -> Backend:
    """Resolve a backend name/instance.

    ``backend`` may be a Backend instance (returned as-is), ``"numpy"``
    or ``"cupy"``. Unknown names raise ValueError; unavailable cupy
    raises BackendUnavailableError unless ``backend_fallback=True``.
    """
    if isinstance(backend, Backend):
        return backend
    if backend is None:
        backend = "numpy"
    name = str(backend).lower()
    if name == "numpy":
        return NumpyBackend()
    if name == "cupy":
        if HAS_CUPY:
            return CupyBackend()
        if backend_fallback:
            fb = NumpyBackend()
            fb.is_fallback = True
            fb.requested_name = "cupy"
            return fb
        # CupyBackend() raises the clear, documented error.
        return CupyBackend()
    raise ValueError(f"unknown backend {backend!r} (expected 'numpy' or 'cupy')")
