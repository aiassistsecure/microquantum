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

## Layout

- `microq/backend.py` — S3 backend interface: `NumpyBackend` (default)
  and import-guarded `CupyBackend`, resolution via `get_backend`.
- `microq/state.py` — state-vector engine (H, X, Y, Z, S, T, S†, T†,
  RX/RY/RZ, CNOT, CZ, SWAP via reshape/tensordot), sampling, fidelity.
- `microq/circuit.py` — journaled Circuit: apply, rewind, fork, replay.
- `microq/journal.py` — JSONL journal + content-addressed checkpoints,
  hash-chain verification.
- `microq/demos.py` — Bell state, GHZ-5, Grover search over 16 "records",
  Demo D: Grover over records stored in NEDB (S4), plus Demo E:
  QFT + quantum phase estimation (S5).
- `microq/search.py` — S4 NEDB record store + data-driven Grover search.
- `microq/qft.py` — S5 QFT / inverse QFT + quantum phase estimation,
  built from journaled H/CP/SWAP gates (CP added to `circuit.py`).
- `tests/test_microq.py` — pytest suite (see SLICE1-RESULTS.md).

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
3. NEDB AS OF / TRACE reads surfaced in the circuit API (rewind already
   restores checkpoints; expose `get_as_of` provenance queries next).
4. Checkpoint striding (checkpoint every k gates + replay-forward on
   rewind) once circuits get long enough that per-gate checkpoints hurt.
5. ~~More algorithms as journaled demos (QFT, phase estimation)~~ —
   done in S5 (see above; each with replay proofs). Further algorithms
   remain welcome as journaled demos.
6. Variational loop (VQE-style) with journaled parameter checkpoints
   and rollback to best-so-far via time travel.
