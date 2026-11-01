"""Debugger inspection core (D1): inspect / diff journaled states.

Classical state-vector simulation only — no quantum hardware.

Design: the state at any journal ``seq`` is obtained via
:meth:`Circuit.state_at` — *checkpoint restore preferred*: the nearest
stored checkpoint at or before ``seq`` (the genesis checkpoint for
``seq == 0``, always kept) is loaded, and only the stride gap after it
(fewer than ``checkpoint_every`` gates) is replay-forwarded; states are
never re-run from scratch when a checkpoint exists. Inspection
never mutates the passed circuit: its state and journal are only read.

Storeless circuits: there is no checkpoint store to restore from, so
:func:`inspect_at` / :func:`diff_states` / :func:`find_divergence`
raise ``ValueError`` rather than silently falling back to replaying
gates. Qubit 0 is the MSB, matching ``microq.state`` / ``microq.circuit``.
"""
from __future__ import annotations

import time

import numpy as np

from .journal import JournalStore, state_hash

_TOP_K = 5


def _state_at(circuit, seq: int) -> tuple[np.ndarray, str]:
    """(psi, state_hash) at journal `seq`.

    Resolved via :meth:`Circuit.state_at`: checkpoint restore is
    preferred and only the stride gap (< ``checkpoint_every`` gates)
    after the nearest checkpoint is replay-forwarded — never a
    re-run from scratch when a checkpoint exists."""
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
        psi = np.array(circuit.state_at(seq), dtype=np.complex128)
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


def _report(psi: np.ndarray, h: str, seq: int, n: int) -> dict:
    """The inspect_at report shape, built from an explicit state."""
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
    return _report(psi, h, seq, circuit.n)


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


def _gate_at(circuit, seq: int) -> dict | None:
    """Copy of the journal entry at `seq` (1-based), or None."""
    if 1 <= seq <= len(circuit.journal):
        return dict(circuit.journal[seq - 1])
    return None


def find_divergence(circuit_a, circuit_b, tolerance: float = 1e-9) -> dict:
    """Binary-search the first journal seq where two circuits diverge.

    Compares the states of two store-backed circuits at the same journal
    ``seq`` (checkpoint restore via :func:`_state_at` only; neither
    circuit is mutated, no gate is re-run). States diverge at a seq when
    their fidelity |<psi_a|psi_b>|^2 is ``< 1 - tolerance``. Equal state
    hashes are a fast path to fidelity 1.0, but the divergence decision
    is always the fidelity comparison, never hash inequality alone (a
    hash difference within `tolerance` still reports no divergence).

    The search over the common prefix ``0..min(len_a, len_b)`` is a real
    binary search (log-step probing). It assumes divergence, once
    present, persists to the end of the common prefix: under that
    assumption the returned seq is the first divergent one, and the seq
    before it is verified within tolerance. If two states diverged and
    then *reconverged* within tolerance, the search could in theory
    miss the transient divergence or report a later seq — no stronger
    guarantee is claimed.

    Returns a dict with ``diverged``, ``seq`` (None when no divergence),
    ``fidelity`` (at the divergent seq, or at the final compared seq
    when none; None for the length-mismatch case, where there is no
    counterpart state to compare), ``state_hash_a`` / ``state_hash_b``,
    ``gate_a`` / ``gate_b`` (journal entries at ``seq``), ``suspect_gate``
    (the entry at ``seq`` from `circuit_b`, falling back to `circuit_a`
    if b has none; None when there is no divergence or ``seq == 0``),
    ``length_a`` / ``length_b`` / ``lengths``, ``tolerance``, and
    ``reason`` / ``message``.

    Differing lengths with an identical common prefix diverge at
    ``seq = min_len + 1``: the suspect is the first extra gate of the
    longer circuit. ``seq == 0`` means the genesis states themselves
    differ (e.g. a fork, whose genesis is its fork-point state).
    Different qubit counts and storeless circuits raise ``ValueError``.
    """
    if not 0.0 <= tolerance < 1.0:
        raise ValueError(f"tolerance must be in [0, 1), got {tolerance!r}")
    if circuit_a.n != circuit_b.n:
        raise ValueError(
            f"states incomparable: {circuit_a.n} vs {circuit_b.n} qubits")
    for c in (circuit_a, circuit_b):
        if c.store is None:
            raise ValueError(
                "find_divergence needs JournalStore-backed circuits: "
                "states are restored from checkpoints, and a storeless "
                "circuit has none")
    len_a, len_b = len(circuit_a.journal), len(circuit_b.journal)
    min_len = min(len_a, len_b)
    base = {"length_a": len_a, "length_b": len_b,
            "lengths": (len_a, len_b), "tolerance": tolerance}

    def fidelity_at(seq):
        psi_a, hash_a = _state_at(circuit_a, seq)
        psi_b, hash_b = _state_at(circuit_b, seq)
        if hash_a == hash_b:
            return 1.0, hash_a, hash_b
        fid = float(abs(np.vdot(psi_a, psi_b)) ** 2)
        return min(1.0, max(0.0, fid)), hash_a, hash_b

    def result(diverged, seq, fidelity, hash_a, hash_b, reason):
        gate_a = _gate_at(circuit_a, seq) if seq else None
        gate_b = _gate_at(circuit_b, seq) if seq else None
        suspect = None
        if diverged and seq:
            suspect = gate_b if gate_b is not None else gate_a
        return {**base, "diverged": diverged, "seq": seq,
                "fidelity": fidelity, "state_hash_a": hash_a,
                "state_hash_b": hash_b, "gate_a": gate_a,
                "gate_b": gate_b, "suspect_gate": suspect,
                "reason": reason, "message": reason}

    # Genesis (seq 0) is checked explicitly: genesis states can differ
    # (e.g. a fork's genesis is its fork-point state, not |0...0>).
    fid0, ha0, hb0 = fidelity_at(0)
    if fid0 < 1.0 - tolerance:
        return result(True, 0, fid0, ha0, hb0,
                      "genesis states (seq 0) already diverge: "
                      f"fidelity {fid0:.6g} < 1 - tolerance")
    # Covers min_len == 0 too: fid_end is then the (agreeing) genesis
    # fidelity, so equal lengths report no divergence and differing
    # lengths fall into the length-mismatch case at seq 1.
    fid_end, ha_end, hb_end = fidelity_at(min_len)
    if fid_end >= 1.0 - tolerance:
        if len_a == len_b:
            return result(False, None, fid_end, ha_end, hb_end,
                          "no divergence: states agree within tolerance "
                          f"at every compared seq (final fidelity "
                          f"{fid_end:.6g} at seq {min_len})")
        seq = min_len + 1
        longer_c = circuit_b if len_b > len_a else circuit_a
        longer = "b" if len_b > len_a else "a"
        reason = (f"length mismatch: common prefix (0..{min_len}) never "
                  f"diverges but lengths differ ({len_a} vs {len_b}); "
                  f"first extra gate is circuit_{longer} seq {seq}")
        out = result(True, seq, None,
                     _state_at(circuit_a, seq)[1] if len_a >= seq else None,
                     _state_at(circuit_b, seq)[1] if len_b >= seq else None,
                     reason)
        out["suspect_gate"] = _gate_at(longer_c, seq)
        return out

    # Binary search: invariant seq `lo` is within tolerance (seq 0 was
    # verified above) and seq `hi` diverges (min_len verified above).
    lo, hi = 0, min_len
    fid_hi, ha_hi, hb_hi = fid_end, ha_end, hb_end
    while hi - lo > 1:
        mid = (lo + hi) // 2
        fid_mid, ha_mid, hb_mid = fidelity_at(mid)
        if fid_mid < 1.0 - tolerance:
            hi, fid_hi, ha_hi, hb_hi = mid, fid_mid, ha_mid, hb_mid
        else:
            lo = mid
    gate = _gate_at(circuit_b, hi) or _gate_at(circuit_a, hi) or {}
    return result(True, hi, fid_hi, ha_hi, hb_hi,
                  f"first divergence at seq {hi} (gate "
                  f"{gate.get('gate')} on {gate.get('targets')}): "
                  f"fidelity {fid_hi:.6g} < 1 - tolerance; seq {lo} "
                  f"verified within tolerance")


# -- D3: breakpoints / watchpoints ---------------------------------------
# A step is a gate spec (all applied via Circuit.apply, journaled
# normally): ("H", 0), ("CNOT", (0, 1)), ("RY", 1, 0.5),
# ("CP", (0, 1), 0.3), ("RY", (1,), (0.5,)), or the dict shape
# {"gate": "RY", "targets": [1], "params": [0.5]}.
_TWO_Q_FREE = {"CNOT", "CX", "CZ", "SWAP"}
_EPS = 1e-12  # float tolerance at a watch threshold boundary


def _parse_step(step, index: int) -> tuple[str, list[int], tuple[float, ...]]:
    """Normalise one step to (gate, targets, params); ValueError if bad."""
    try:
        if isinstance(step, dict):
            gate = step["gate"]
            raw_t = step["targets"] if "targets" in step else step["target"]
            raw_p = step.get("params", step.get("param", ()))
        elif isinstance(step, (tuple, list)) and len(step) >= 2:
            gate, rest = step[0], list(step[1:])
            if isinstance(rest[0], (tuple, list)):
                if len(rest) > 2: raise ValueError("too many fields")
                raw_t = list(rest[0])
                raw_p = rest[1] if len(rest) > 1 else ()
            elif str(gate).upper() in _TWO_Q_FREE and len(rest) == 2:
                raw_t, raw_p = rest, ()
            elif str(gate).upper() == "CP" and len(rest) == 3:
                raw_t, raw_p = rest[:2], rest[2]
            elif len(rest) == 1:
                raw_t, raw_p = rest, ()
            elif len(rest) == 2:
                raw_t, raw_p = rest[:1], rest[1]
            else:
                raise ValueError("too many fields")
        else:
            raise ValueError("step must be a dict or a (gate, ...) tuple/list")
        targets = [raw_t] if isinstance(raw_t, int) else [int(t) for t in raw_t]
        params = ((float(raw_p),) if isinstance(raw_p, (int, float))
                  else tuple(float(p) for p in raw_p))
        return str(gate).upper(), targets, params
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"step {index}: malformed gate spec {step!r}") from exc


def _check_threshold(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"threshold must be a number in [0, 1], got {value!r}")
    return float(value)


class _Watch:
    """Base watchpoint condition: reusable, describable, journal-safe
    (``to_dict`` carries no arrays — a fidelity target is hashed)."""
    kind = "watch"

    def __init__(self, threshold, name=None):
        self.threshold = _check_threshold(threshold)
        self.name = name

    def describe(self) -> str: raise NotImplementedError
    def validate(self, n: int) -> None: raise NotImplementedError
    def observed(self, psi, n) -> float: raise NotImplementedError
    def tripped(self, value) -> bool: raise NotImplementedError

    def params_dict(self) -> dict:
        return {"threshold": self.threshold}

    def to_dict(self, observed_value=None) -> dict:
        return {"kind": self.kind, "name": self.name,
                "description": self.describe(), **self.params_dict(),
                "observed_value": observed_value}


class ProbabilityWatch(_Watch):
    """Trip when P(|bitstring>) reaches/exceeds `threshold` (>=)."""
    kind = "probability"

    def __init__(self, bitstring, threshold, name=None):
        if not isinstance(bitstring, str) or not bitstring \
                or any(c not in "01" for c in bitstring):
            raise ValueError(f"bitstring must be a non-empty '0'/'1' string, "
                             f"got {bitstring!r}")
        super().__init__(threshold, name)
        self.bitstring = bitstring

    def describe(self): return f"P(|{self.bitstring}>) >= {self.threshold}"

    def validate(self, n):
        if len(self.bitstring) != n:
            raise ValueError(f"bitstring length {len(self.bitstring)} != {n} qubits")

    def observed(self, psi, n):
        return float(abs(psi[int(self.bitstring, 2)]) ** 2)

    def tripped(self, value):
        return value >= self.threshold - _EPS

    def params_dict(self): return {"bitstring": self.bitstring, "threshold": self.threshold}


class FidelityWatch(_Watch):
    """Trip when fidelity vs `target_state` drops below `threshold` (<)."""
    kind = "fidelity"

    def __init__(self, target_state, threshold, name=None):
        super().__init__(threshold, name)
        target = np.asarray(target_state, dtype=np.complex128)
        if target.ndim != 1 or target.size < 2 or target.size & (target.size - 1):
            raise ValueError("target_state must be a 1-D state vector whose "
                             "length is a power of two >= 2")
        if abs(float(np.linalg.norm(target)) - 1.0) > 1e-6:
            raise ValueError("target_state must be normalised (unit norm)")
        self.target_state = target

    def describe(self): return f"fidelity vs target < {self.threshold}"

    def validate(self, n):
        if self.target_state.size != 2 ** n:
            raise ValueError(f"target has {self.target_state.size} amplitudes != {n} qubits")

    def observed(self, psi, n):
        return float(min(1.0, max(0.0, abs(np.vdot(self.target_state, psi)) ** 2)))

    def tripped(self, value):
        return value < self.threshold - _EPS

    def params_dict(self):
        return {"threshold": self.threshold, "target_hash": state_hash(self.target_state)}


class EntropyWatch(_Watch):
    """Trip when `qubit`'s von Neumann entropy (bits) rises above
    `threshold` (>)."""
    kind = "entropy"

    def __init__(self, qubit, threshold, name=None):
        if isinstance(qubit, bool) or not isinstance(qubit, int) or qubit < 0:
            raise ValueError(f"qubit must be an int >= 0, got {qubit!r}")
        super().__init__(threshold, name)
        self.qubit = qubit

    def describe(self): return f"entropy(qubit {self.qubit}) > {self.threshold} bits"

    def validate(self, n):
        if self.qubit >= n:
            raise ValueError(f"qubit {self.qubit} out of range for {n} qubits")

    def observed(self, psi, n):
        return _entropy_bits(_reduced_state(psi, n, self.qubit))

    def tripped(self, value):
        return value > self.threshold + _EPS

    def params_dict(self): return {"qubit": self.qubit, "threshold": self.threshold}


def run_with_watchpoints(circuit, steps, conditions) -> dict:
    """Apply `steps` gate by gate, halting at the first tripped watch.

    Every step goes through ``Circuit.apply`` (journaled normally);
    after EACH gate every condition is evaluated against the live
    state (``circuit.state.to_numpy()``, a copy — no gate is re-run,
    evaluation does not mutate). All conditions/steps are validated
    before gate 1, so bad input raises ``ValueError`` untouched.

    On trip: halt immediately (remaining steps NOT applied); return
    ``tripped=True``, ``seq`` (journal seq of the trip), ``condition``
    (kind/name/description/params/observed value), and ``inspect`` —
    the full :func:`inspect_at` report at that seq (also ``report``).
    On no trip: all steps applied; ``tripped=False``, ``seq=None``
    (documented choice), final ``inspect``, and ``final_observed``
    (last value per condition). ``steps_applied``/``final_seq`` are
    in both. Threshold comparisons carry a 1e-12 float tolerance at
    the boundary (a hand-computed P = 0.5 vs a 0.5 threshold trips).

    A trip records ONE breakpoint event, separate from the gate
    journal: the store's ``debug_events`` collection/file when
    store-backed (gate journal, hash chain, verify/replay, AS OF /
    TRACE untouched), else an in-memory ``circuit.debug_events``
    list — then ``inspect`` is built from the live state in the same
    shape (no checkpoint exists to restore). No trip, no event.
    Events are returned under ``debug_events`` / ``event``.
    """
    conditions = list(conditions)
    for c in conditions:
        if not isinstance(c, _Watch):
            raise ValueError(f"not a watchpoint condition: {c!r}")
        c.validate(circuit.n)
    parsed = [_parse_step(s, i) for i, s in enumerate(steps, start=1)]
    start_len = len(circuit.journal)
    final_observed: list[dict] = []
    report = None
    for gate, targets, params in parsed:
        circuit.apply(gate, *targets, params=params)
        seq = circuit.journal[-1]["seq"]
        psi = circuit.state.to_numpy()
        final_observed = []
        for cond in conditions:
            value = cond.observed(psi, circuit.n)
            final_observed.append(cond.to_dict(value))
            if not cond.tripped(value):
                continue
            report = (inspect_at(circuit, seq) if circuit.store is not None
                      else _report(psi, state_hash(psi), seq, circuit.n))
            body = {"event_kind": "breakpoint", "kind": "breakpoint",
                    "journal_seq": seq, "seq": seq,
                    "condition_kind": cond.kind,
                    "condition": cond.to_dict(value),
                    "observed_value": value, "threshold": cond.threshold,
                    "state_hash": state_hash(psi)}
            if circuit.store is not None:
                event = circuit.store.record_debug_event(body)
            else:
                if not hasattr(circuit, "debug_events"):
                    circuit.debug_events = []
                event = {"event_seq": len(circuit.debug_events) + 1, **body,
                         "timestamp": time.time()}
                circuit.debug_events.append(event)
            return {"tripped": True, "seq": seq, "final_seq": seq,
                    "condition": cond.to_dict(value),
                    "condition_kind": cond.kind, "observed_value": value,
                    "inspect": report, "report": report,
                    "steps_applied": len(circuit.journal) - start_len,
                    "final_observed": final_observed,
                    "debug_events": [event], "event": event}
    if parsed:
        seq = circuit.journal[-1]["seq"]
        psi = circuit.state.to_numpy()
        report = (inspect_at(circuit, seq) if circuit.store is not None
                  else _report(psi, state_hash(psi), seq, circuit.n))
    return {"tripped": False, "seq": None,
            "final_seq": len(circuit.journal), "condition": None,
            "condition_kind": None, "observed_value": None,
            "inspect": report, "report": report,
            "steps_applied": len(circuit.journal) - start_len,
            "final_observed": final_observed, "debug_events": [],
            "event": None}
