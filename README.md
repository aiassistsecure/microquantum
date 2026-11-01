# microquantum

A GPU-ready quantum circuit **state-vector simulator** whose execution
journal lives in NEDB-shaped storage: every gate is journaled with a hash
chain, states are checkpointed content-addressed, and circuits support
**rewind** (time travel), **fork** (branch from a checkpoint), and
**replay** (re-execute the journal and verify the final state hash —
provenance proof).

## Honest framing

This is **classical simulation of quantum mathematics** on NumPy. There is
no quantum hardware involved and no quantum speedup: a state-vector
simulator pays exponential memory to represent what a quantum device holds
natively. Grover's algorithm below *demonstrates the amplitude-amplification
math*; on this simulator it is slower than just scanning the list, which is
exactly the point of showing it honestly.

### The V100 ceiling

The target machine is Mark's Tesla V100 with 32 GB of GPU memory. A
complex128 amplitude is 16 bytes, and an n-qubit state has 2^n amplitudes:

- 2^30 × 16 B = 17,179,869,184 B ≈ **17 GB** → ~30 qubits fits in 32 GB
  (with room for one working copy / gate temporaries, ~29–30 is the
  practical ceiling).
- 2^31 × 16 B ≈ 34 GB → 31 qubits does **not** fit.

So "microquantum on the V100" means a hard ceiling around 30 qubits —
real, useful for education, algorithm prototyping and verification, and
honestly nowhere near quantum advantage. Slice 1 runs on CPU/NumPy; the
engine sits behind a small backend surface (`apply_1q` / `apply_2q` /
`cnot` / `cz` / `swap` on an array) so a CuPy/CUDA backend can drop in for
the V100 without touching `Circuit` or the journal.

## The NEDB angle

NEDB is Mark's database whose journal *is* the product: hash-chained
writes, provenance, time-travel reads. microquantum applies the same idea
to circuit execution:

- **Journal = provenance.** Every gate application appends an entry
  `{seq, gate, targets, params, state_hash, prev_hash, entry_hash,
  timestamp}`. The `prev_hash`/`entry_hash` BLAKE2b chain makes the
  execution history tamper-evident: edit one recorded gate and
  `verify_chain()` fails.
- **Time travel.** `circuit.rewind(seq)` restores the checkpointed state
  at any journal point, including `rewind(0)` back to |0...0>.
- **Forking.** `circuit.fork(seq)` branches an independent circuit from a
  checkpoint; gates on the fork never touch the parent.
- **Replay = proof.** `circuit.replay()` re-applies the journal from
  |0...0> and checks the final state hash equals the journaled head hash.
- **Checkpoints never duplicate bytes.** Checkpoints are stored under
  `data/<demo>/checkpoints/<blake2b-of-state-bytes>.npy`; identical states
  are written once (SI's rule: never duplicate bytes).

### Slice-1 persistence status (plainly)

The journal runs on the **real NEDB engine**: `nedb-engine` from PyPI
(`NEDB(path)`), verified working on this VM in slice 1. Journal entries
are documents in NEDB's `journal` collection — NEDB seals each write in
its own hash chain (`_seq`/`_hash`, `verify()`), and each entry also
carries microquantum's own `prev_hash`/`entry_hash` chain so the journal
verifies standalone. Checkpoints stay out of the database as
content-addressed blob files, mirroring SI's split (journal in NEDB,
blobs by reference). If the `nedb` binding is ever unavailable,
`JournalStore` degrades to an append-only JSONL journal with the identical
entry schema and hash chain (`JournalStore.backend` reports which).

## Provenance reads (AS OF / TRACE)

The journal can be read back at any point in its history, not just
rewound:

- `store.entry_as_of(seq)` / `circuit.as_of(seq)` — the journal entry
  as of that point: its gate, targets, params, and the `state_hash`
  the circuit actually had then. `seq == 0` is genesis (no entry yet)
  and returns `None`; any other out-of-range seq raises `ValueError`.
- `store.trace(seq=None)` / `circuit.trace(seq=None)` — the ordered
  chain prefix leading to `seq` (default: the head), returned as
  `{seq, entries, verified, head_hash, state_hash}`, where `verified`
  is the microquantum hash-chain check run over exactly that prefix.

One seq distinction matters here. These reads take the **microquantum
journal seq** (position in this circuit's journal). NEDB's native
`get(coll, id, as_of=N)` instead takes NEDB's **global op seq**, where
writes to other collections interleave — the two numbers are not
interchangeable, and passing a journal seq to NEDB's `as_of` would
read the wrong point in time. microquantum never does that: journal
entries are keyed `str(journal_seq)` and each key is written exactly
once, so a plain per-key get *is* the as-of read for that journal seq,
and a prefix of per-key gets plus chain verification gives correct
TRACE semantics. (A storeless in-memory circuit can answer `as_of`,
but records no hash chain, so its `trace` reports `verified` as
`None` rather than claiming a check it cannot run.)

## Layout

- `microq/backend.py` — S3 backend interface: `NumpyBackend` (default)
  and import-guarded `CupyBackend`, resolution via `get_backend`.
- `microq/state.py` — state-vector engine (H, X, Y, Z, S, T, S†, T†,
  RX/RY/RZ, CNOT, CZ, SWAP via reshape/tensordot), sampling, fidelity.
- `microq/circuit.py` — journaled Circuit: apply, rewind, fork, replay,
  AS OF / TRACE provenance reads (`as_of` / `trace`).
- `microq/journal.py` — NEDB journal (JSONL fallback) + content-addressed
  checkpoints, hash-chain verification, `entry_as_of` / `trace` reads.
- `microq/demos.py` — Bell state, GHZ-5, Grover search over 16 "records",
  Demo D: Grover over records stored in NEDB (S4), plus Demo E:
  QFT + quantum phase estimation (S5).
- `microq/search.py` — S4 NEDB record store + data-driven Grover search.
- `microq/qft.py` — S5 QFT / inverse QFT + quantum phase estimation,
  built from journaled H/CP/SWAP gates (CP added to `circuit.py`).
- `microq/vqe.py` — S6 VQE-style variational loop: RY/CNOT ansatz,
  exact expectations, NumPy Nelder–Mead, journaled parameter
  checkpoints in NEDB, rollback to best-so-far via rewind.
- `microq/debug.py` — debugger core: `inspect_at` / `diff_states`
  (D1), `find_divergence` bisection (D2), and `run_with_watchpoints`
  breakpoints (D3) over journaled states (see Debugger below).
- `tests/test_microq.py` — pytest suite (see SLICE1-RESULTS.md).

## Debugger (D1–D3)

`microq/debug.py` is the core of the debugger line:

- `inspect_at(circuit, seq)` — the full state at journal `seq`:
  per-qubit reduced density matrices, Bloch vectors, and entanglement
  entropies (von Neumann, bits), the full probability distribution,
  and the top amplitudes. `seq == 0` is genesis |0...0>; out-of-range
  raises `ValueError`.
- `diff_states(circuit, seq_a, seq_b)` — fidelity
  (|⟨psi_a|psi_b⟩|^2), per-amplitude / per-probability deltas, and
  the largest movers between two journal points.
- `find_divergence(circuit_a, circuit_b, tolerance=1e-9)` (D2) —
  **binary-searches** the common journal prefix for the first `seq`
  where the two circuits' states diverge: fidelity
  |⟨psi_a|psi_b⟩|^2 `< 1 - tolerance`. It returns that `seq`, the
  fidelity and state hashes there, both gate entries, and
  `suspect_gate` — the gate entry at that seq from `circuit_b` (the
  presumed mutated circuit; falling back to `circuit_a` if b has no
  such entry) — so a planted wrong angle or wrong target is named
  directly, not just located approximately.
- `run_with_watchpoints(circuit, steps, conditions)` (D3) —
  **breakpoints / watchpoints**: applies `steps` (gate specs as
  tuples like `("RY", 1, 0.5)` / `("CNOT", (0, 1))`, or dicts like
  `{"gate": "RY", "targets": [1], "params": [0.5]}`) through
  `Circuit.apply`, so every gate is journaled normally, and after
  each gate evaluates every condition against the resulting state.
  Three reusable, describable condition types: `ProbabilityWatch`
  (trip when P(|bitstring>) reaches a threshold),
  `FidelityWatch` (trip when fidelity vs a target state drops
  below a threshold), and `EntropyWatch` (trip when a qubit's von
  Neumann entropy in bits rises above a threshold). Comparisons
  carry a 1e-12 float tolerance at the boundary.

States are obtained by **checkpoint restore** (`store.get_checkpoint`
on the journal entry's `state_hash`, the genesis checkpoint for
`seq 0`), never by re-running gates when a checkpoint exists, and
inspection never mutates either circuit. A storeless circuit has no
checkpoints to restore, so all three functions raise `ValueError`
for one rather than silently replaying. Circuits with different
qubit counts also raise `ValueError` (states incomparable).

`find_divergence` behaviour at the edges, stated plainly:

- Identical circuits (or an identical prefix of equal length) report
  `diverged=False, seq=None`. The decision is fidelity-based: equal
  state hashes are only a fast path to fidelity 1, and a hash
  difference whose fidelity stays within `tolerance` (e.g. a
  negligibly different rotation angle) still reports no divergence.
- Differing lengths with an agreeing common prefix diverge at
  `seq = min_len + 1`, with the longer circuit's first extra gate as
  the suspect and `fidelity=None` — there is no counterpart state to
  compare, so no fidelity is fabricated.
- `seq == 0` means the genesis states themselves differ (possible
  for a fork, whose genesis is its fork-point state).
- The binary search assumes divergence, once present, persists to
  the end of the common prefix. Under that assumption the returned
  seq is the first divergent one (the seq before it is verified
  within tolerance). If states diverged and then reconverged within
  tolerance, the search could miss the transient divergence or
  report a later seq — no stronger guarantee is claimed.

D3 halt semantics, stated plainly: on the first tripped condition
the run halts immediately — remaining steps are never applied, so
the gate journal's length equals the tripped `seq` — and the result
carries `tripped=True`, that `seq`, the tripped condition with its
observed value, and the full `inspect_at` report at that seq. With
no trip, all steps run: `tripped=False, seq=None`, final observed
values. A trip records exactly one breakpoint event as its **own
entry kind, outside the gate journal**: the store's `debug_events`
collection (NEDB) or `debug_events.jsonl` file, under its own
`event_seq`, referencing the journal seq, condition, observed
value, and state hash — gate entries, hash chain, `verify_chain()`,
`replay()`, and AS OF / TRACE are completely unaffected (tested).
A storeless circuit keeps events in an in-memory
`circuit.debug_events` list; its trip report is built from the live
state (no checkpoint to restore). Invalid conditions, thresholds,
or bitstrings raise `ValueError` before any gate is applied.

Limits, stated plainly: this is classical state-vector simulation
(no quantum hardware), inspection and comparison cost the same
exponential memory as the state itself, watchpoint evaluation adds
a per-gate reduced-state cost at the same scale, and V100
performance is UNMEASURED.

## Backends (S3)

Gate math runs behind `microq.backend.Backend` (`apply_1q` / `apply_2q` /
`cnot` / `cz` / `swap`, plus `zeros` / `from_numpy` / `to_numpy` transfer
helpers). `Circuit` and the journal API are unchanged; checkpoints and
hashes always use host NumPy arrays.

```python
from microq import Circuit
c = Circuit(2, backend="numpy")   # default, CPU
c = Circuit(2, backend="cupy")    # CuPy/CUDA on a GPU host (e.g. V100)
```

- `backend="cupy"` when CuPy or a CUDA device is unavailable **raises
  `BackendUnavailableError`** — it never silently pretends to be on GPU.
  Pass `backend_fallback=True` for an explicit fallback: the returned
  backend is NumPy with `is_fallback=True` and `requested_name="cupy"`.
- The CuPy backend is import-guarded; on hosts without CuPy its tests
  skip cleanly. Do not `pip install cupy` on a CPU-only host expecting
  it to work.
- **V100 performance is UNMEASURED.** No GPU benchmark has been run for
  this project; any V100 number would be fabricated, so none is given.
  This remains classical simulation — a GPU backend only runs the same
  linear algebra faster, it is not quantum hardware.

## Grover over NEDB records (S4)

`microq/search.py` seeds 16 deterministic records (`{index, name, tag}`)
into a **real NEDB** database (`records` collection, keyed by
`str(index)`, in a separate NEDB path from the circuit journal store).
`grover_search_nedb(root, target_tag)` reads the matching record back
out of NEDB, derives the marked bitstring from that record's *stored*
index, and builds the Grover oracle (the S1 oracle/diffuser, reused)
from the derived string. It returns the found record, `P(marked)`
before/after, the journal length, `replay()`, and both verify results
(record-DB `verify()` and the journal hash chain).

Demo D (`python -m microq.demos`) runs this end to end. Honest framing
again: the lookup that finds the record scans NEDB directly and is
faster than the simulated search — there is no quantum speedup here;
the point is an oracle that is genuinely data-driven from NEDB, on a
fully journaled, replay-verifiable circuit.

## QFT + phase estimation (S5)

`microq/qft.py` adds `qft` / `inverse_qft` (H + controlled-phase CP +
SWAP, all journaled — CP is a new journaled `Circuit` gate,
`diag(1, 1, 1, exp(i*theta))`) and `phase_estimation(root, phi,
n_count=3)`, which estimates the phase of `U = diag(1, exp(2*pi*i*phi))`
on its eigenstate |1>: counting register in superposition,
controlled-`U**(2**k)` as CP gates, then the inverse QFT. It returns
the true and estimated phi, the peak counting bitstring and its
probability, the full counting-register probabilities, the journal
length, and replay / hash-chain verification. Demo E
(`python -m microq.demos`) shows a QFT→inverse-QFT round trip and
recovers phi = 3/8 exactly at n_count = 3. The engine's qubit 0 is the
MSB; the tests verify the QFT unitary against the direct DFT matrix
and the QPE bit ordering empirically rather than assuming a
convention. Still classical simulation: no hardware, no speedup.

## Variational loop (S6)

`microq/vqe.py` runs a VQE-style loop on a fixed 2-qubit Hamiltonian
(`ZZ + 0.6 XX + 0.4 YY - 0.5 ZI + 0.2 IZ`, exact ground energy from
`eigvalsh`): a 4-parameter RY/CNOT ansatz built only from journaled
Circuit gates, expectation values computed **exactly** from the state
vector (no sampling), and a deterministic Nelder–Mead optimizer in
pure NumPy. Every evaluation persists a parameter checkpoint
(params + energy + state hash) into NEDB on the journal's database;
after the loop, best-so-far is restored via `Circuit.rewind` to that
iteration's checkpoint and proved by state hash, recovered params,
and re-evaluated energy. Demo F (`python -m microq.demos`) runs it
end to end. Still classical simulation of quantum math: no quantum
hardware, no speedup claims, and V100 numbers remain UNMEASURED.

## Run it

```bash
cd ~/workspace/microquantum
python3 -m venv venv && source venv/bin/activate
pip install numpy pytest nedb-engine
python -m microq.demos     # Bell, GHZ, Grover + rewind/replay checks
python -m pytest tests/ -q
```

## What's next

1. ~~CuPy/CUDA backend for the V100~~ — done in S3 (see Backends above;
   V100 numbers UNMEASURED until run at the box).
2. ~~Grover search over records actually stored in NEDB~~ — done in S4
   (see above; oracle derived from the record read back out of NEDB).
3. ~~NEDB AS OF / TRACE reads surfaced in the circuit API~~ — done
   (see Provenance reads above; journal-seq `entry_as_of` / `as_of`
   and `trace`, chain-verified prefixes, never conflated with NEDB's
   global op seq).
4. Checkpoint striding (checkpoint every k gates + replay-forward on
   rewind) once circuits get long enough that per-gate checkpoints hurt.
5. ~~More algorithms as journaled demos (QFT, phase estimation)~~ —
   done in S5 (see above; each with replay proofs). Further algorithms
   remain welcome as journaled demos.
6. ~~Variational loop (VQE-style) with journaled parameter checkpoints
   and rollback to best-so-far via time travel~~ — done in S6 (see
   above; checkpoints in NEDB, rollback proved by hash/params/energy).
