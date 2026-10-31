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

- `microq/state.py` — state-vector engine (H, X, Y, Z, S, T, S†, T†,
  RX/RY/RZ, CNOT, CZ, SWAP via reshape/tensordot), sampling, fidelity.
- `microq/circuit.py` — journaled Circuit: apply, rewind, fork, replay.
- `microq/journal.py` — JSONL journal + content-addressed checkpoints,
  hash-chain verification.
- `microq/demos.py` — Bell state, GHZ-5, Grover search over 16 "records".
- `tests/test_microq.py` — pytest suite (see SLICE1-RESULTS.md).

## Run it

```bash
cd ~/workspace/microquantum
python3 -m venv venv && source venv/bin/activate
pip install numpy pytest nedb-engine
python -m microq.demos     # Bell, GHZ, Grover + rewind/replay checks
python -m pytest tests/ -q
```

## What's next

1. CuPy/CUDA backend for the V100 (same backend surface, device arrays).
2. NEDB AS OF / TRACE reads surfaced in the circuit API (rewind already
   restores checkpoints; expose `get_as_of` provenance queries next).
3. Checkpoint striding (checkpoint every k gates + replay-forward on
   rewind) once circuits get long enough that per-gate checkpoints hurt.
4. More algorithms as journaled demos (QFT, phase estimation), each with
   replay proofs.
