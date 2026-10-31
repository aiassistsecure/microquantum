"""S6 tests: VQE-style variational loop, journaled checkpoints, rollback."""
import numpy as np
import pytest

from microq.circuit import Circuit
from microq.journal import HAS_NEDB
from microq.vqe import (
    HAMILTONIAN_TERMS,
    START_PARAMS,
    ansatz_state,
    build_ansatz,
    exact_ground_energy,
    expectation,
    hamiltonian_matrix,
    load_checkpoints,
    rollback_best,
    run_vqe,
)

pytestmark = pytest.mark.skipif(not HAS_NEDB, reason="nedb binding unavailable")

_I = np.eye(2, dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.diag([1.0, -1.0]).astype(np.complex128)
_P = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}


def _direct_matrix():
    H = np.zeros((4, 4), dtype=np.complex128)
    for coeff, s in HAMILTONIAN_TERMS:
        H = H + coeff * np.kron(_P[s[0]], _P[s[1]])
    return H


def test_hamiltonian_matrix_matches_direct_construction():
    assert np.allclose(hamiltonian_matrix(), _direct_matrix(), atol=1e-15)
    ground = float(np.linalg.eigvalsh(_direct_matrix())[0])
    assert exact_ground_energy() == pytest.approx(ground, abs=1e-12)
    # Non-trivial: ground is below every basis state's energy.
    assert ground < min(np.real(np.diag(_direct_matrix()))) - 0.5


def test_expectation_of_known_states_exact():
    # |00>: ZZ=+1, ZI=+1, IZ=+1, XX=YY=0 -> 1.0 - 0.5 + 0.2 = 0.7
    assert expectation(np.array([1, 0, 0, 0], dtype=complex)) == pytest.approx(0.7, abs=1e-12)
    # Bell (|00>+|11>)/sqrt2: ZZ=+1, XX=+1, YY=-1, Z terms 0 -> 1 + 0.6 - 0.4 = 1.2
    c = Circuit(2)
    c.h(0).cnot(0, 1)
    assert expectation(c) == pytest.approx(1.2, abs=1e-12)


def test_ansatz_zero_params_is_ground_state_of_free_qubits():
    psi = ansatz_state([0.0, 0.0, 0.0, 0.0])
    assert np.allclose(psi, [1, 0, 0, 0], atol=1e-15)
    c = Circuit(2)
    build_ansatz(c, [0.0, 0.0, 0.0, 0.0])
    assert expectation(c) == pytest.approx(0.7, abs=1e-12)
    assert [e["gate"] for e in c.journal] == ["RY", "RY", "CNOT", "RY", "RY"]


@pytest.fixture(scope="module")
def vqe_run(tmp_path_factory):
    return run_vqe(tmp_path_factory.mktemp("vqe") / "run")


def test_optimizer_converges_to_ground(vqe_run):
    ground = float(np.linalg.eigvalsh(_direct_matrix())[0])
    assert vqe_run["best_energy"] == pytest.approx(ground, abs=1e-2)
    assert vqe_run["exact_ground_energy"] == pytest.approx(ground, abs=1e-12)
    assert vqe_run["best_energy"] <= vqe_run["start_energy"]
    assert vqe_run["start_energy"] == pytest.approx(
        expectation(ansatz_state(START_PARAMS)), abs=1e-12)
    assert vqe_run["iterations"] == len(vqe_run["history"]) > 10


def test_every_iteration_checkpoint_persisted_and_reloadable(tmp_path):
    r = run_vqe(tmp_path / "run", max_evals=25)
    loaded = load_checkpoints(tmp_path / "run")  # fresh store view of same root
    assert len(loaded) == len(r["history"]) == 25
    for saved, seen in zip(r["history"], loaded):
        assert seen["iteration"] == saved["iteration"]
        assert seen["params"] == pytest.approx(saved["params"], abs=1e-12)
        assert seen["energy"] == pytest.approx(saved["energy"], abs=1e-12)
        assert seen["state_hash"] == saved["state_hash"]


def test_rollback_restores_best(vqe_run):
    assert vqe_run["rollback_verified"] is True
    assert vqe_run["rollback_replay"] is True
    assert vqe_run["restored_state_hash"] == vqe_run["best_state_hash"]
    assert vqe_run["restored_params"] == pytest.approx(vqe_run["best_params"], abs=1e-12)
    assert vqe_run["restored_energy"] == pytest.approx(vqe_run["best_energy"], abs=1e-9)


def test_rollback_returns_best_after_worse_params(tmp_path):
    r = run_vqe(tmp_path / "run", max_evals=40)
    # Deliberately evaluate much worse params AFTER the run's best.
    worse = expectation(ansatz_state([0.0, 0.0, 0.0, 0.0]))
    assert worse > r["best_energy"] + 0.5
    rolled = rollback_best(tmp_path / "run")
    assert rolled["iteration"] == r["best_iteration"]
    assert rolled["state_hash"] == r["best_state_hash"]
    assert rolled["params"] == pytest.approx(r["best_params"], abs=1e-12)
    assert rolled["energy"] == pytest.approx(r["best_energy"], abs=1e-9)
    assert rolled["replay"] is True


def test_vqe_journal_chain_verifies(vqe_run):
    assert vqe_run["journal_chain_verify"] is True
    assert vqe_run["journal_store_verify"] is True
    assert vqe_run["journal_length"] == 5 * vqe_run["iterations"]
