"""S5 tests: CP gate, QFT / inverse QFT, and quantum phase estimation."""
import numpy as np
import pytest

from microq.circuit import Circuit
from microq.qft import inverse_qft, phase_estimation, qft


def _dft_matrix(n):
    N = 2 ** n
    w = np.exp(2j * np.pi / N)
    return np.array(
        [[w ** (x * y) / np.sqrt(N) for x in range(N)] for y in range(N)],
        dtype=np.complex128,
    )


# --- CP gate ---------------------------------------------------------------

def test_cp_pi_equals_cz_and_phases_only_11():
    # On basis states CP(pi) flips only |11>, like CZ.
    for bits, sign in [("00", 1), ("01", 1), ("10", 1), ("11", -1)]:
        c = Circuit(2)
        for q, b in enumerate(bits):
            if b == "1":
                c.x(q)
        c.cp(0, 1, np.pi)
        idx = int(bits, 2)
        assert c.state.to_numpy()[idx] == pytest.approx(sign)
    # On a superposition CP(pi) and CZ produce identical states.
    a, b = Circuit(2), Circuit(2)
    for circ in (a, b):
        circ.h(0).h(1)
    a.cp(0, 1, np.pi)
    b.cz(0, 1)
    assert np.allclose(a.state.to_numpy(), b.state.to_numpy())


def test_cp_quarter_phase_and_replay():
    c = Circuit(2)
    c.x(0).x(1).cp(0, 1, np.pi / 2)
    assert c.state.to_numpy()[3] == pytest.approx(1j)
    assert c.journal[-1]["gate"] == "CP"
    assert c.journal[-1]["params"] == [pytest.approx(np.pi / 2)]
    assert c.replay() is True


# --- QFT --------------------------------------------------------------------

def test_qft_of_zero_is_uniform():
    c = Circuit(3)
    qft(c, [0, 1, 2])
    probs = c.probabilities()
    assert len(probs) == 8
    for v in probs.values():
        assert v == pytest.approx(1 / 8, abs=1e-12)
    assert c.replay() is True


def test_qft_matches_dft_matrix_on_nontrivial_state():
    c = Circuit(3)
    c.h(0).rx(1, 0.7).ry(2, -1.1).cnot(0, 2)  # non-trivial input state
    psi_in = c.state.to_numpy().copy()
    qft(c, [0, 1, 2])
    expected = _dft_matrix(3) @ psi_in
    assert np.allclose(c.state.to_numpy(), expected, atol=1e-12)
    assert c.probabilities() == pytest.approx(
        {format(i, "03b"): float(abs(a) ** 2) for i, a in enumerate(expected)},
        abs=1e-12,
    )


def test_inverse_qft_round_trip_fidelity():
    c = Circuit(3)
    c.h(0).rx(1, 0.3).cnot(1, 2).rz(0, 0.9)
    psi_in = c.state.to_numpy().copy()
    qft(c, [0, 1, 2])
    inverse_qft(c, [0, 1, 2])
    fidelity = float(abs(np.vdot(psi_in, c.state.to_numpy())) ** 2)
    assert fidelity == pytest.approx(1.0, abs=1e-12)
    assert c.replay() is True


# --- QPE --------------------------------------------------------------------

@pytest.mark.parametrize("phi", [3 / 8, 5 / 8, 1 / 4])
def test_qpe_recovers_exact_binary_phi(tmp_path, phi):
    r = phase_estimation(tmp_path / "qpe", phi, n_count=3)
    assert r["true_phi"] == phi
    assert r["estimated_phi"] == pytest.approx(phi, abs=1e-12)
    assert r["peak_probability"] > 0.99
    assert r["probabilities"][r["peak_bitstring"]] == pytest.approx(
        r["peak_probability"], abs=1e-12)


def test_qpe_journal_replay_and_chain_verify(tmp_path):
    r = phase_estimation(tmp_path / "qpe", 3 / 8, n_count=3)
    assert r["journal_length"] > 0
    assert r["replay"] is True
    assert r["journal_chain_verify"] is True
    assert r["journal_store_verify"] is True
