# microquantum — Slice 1 Results

Run: 2026-10-31T16:02:27+00:00 on this VM (CPU/NumPy; no GPU on this VM).
Backend journal store: real NEDB via nedb-engine (PyPI), checkpoints as content-addressed .npy blobs.

## Environment check
nedb-engine importable: True
JournalStore.backend: nedb
bell store verify_store (NEDB verify()): True
bell store verify_chain (microq chain): True

## Demos (python -m microq.demos)
```
=== Demo A: Bell state (2 qubits) ===
  P(|00>) = 0.5000
  P(|01>) = 0.0000
  P(|10>) = 0.0000
  P(|11>) = 0.5000
  1000 shots: {'00': 503, '11': 497}
  journal length: 2 entries; head state hash efba273719bbbd31...
  replay() from |0...0> reproduces final state: True
  rewind(seq=1) -> state == checkpoint at seq 1: True

=== Demo B: GHZ state (5 qubits) ===
  P(|00000>) = 0.5000
  P(|11111>) = 0.5000
  1000 shots: {'00000': 502, '11111': 498}
  journal length: 5 entries; head state hash afba633cf46da6c8...
  replay() from |0...0> reproduces final state: True
  rewind(seq=2) -> state == checkpoint at seq 2: True

=== Demo C: Grover search (4 search qubits + 2 ancillas) ===
Searching an unsorted space of 16 'records' for marked |1011>.
Classical simulation: amplitude amplification, no quantum speedup.
  initial P(marked) = 0.0625
  after iteration 1: P(marked) = 0.4727
  after iteration 2: P(marked) = 0.9084
  after iteration 3: P(marked) = 0.9613
  1000 shots, top outcomes (6-bit: search+ancillas): [('101100', 965), ('000000', 4), ('000100', 4)]
  journal length: 424 entries; head state hash df8e907d5b8b4af3...
  replay() from |0...0> reproduces final state: True
  rewind(seq=212) -> state == checkpoint at seq 212: True
```

## Tests (python -m pytest tests/ -v)
```
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.2.0, pluggy-1.7.0 -- /home/hatch/workspace/microquantum/venv/bin/python
cachedir: .pytest_cache
rootdir: /home/hatch/workspace/microquantum
collecting ... collected 11 items

tests/test_microq.py::test_bell_probabilities PASSED                     [  9%]
tests/test_microq.py::test_ghz_probabilities PASSED                      [ 18%]
tests/test_microq.py::test_toffoli_truth_table PASSED                    [ 27%]
tests/test_microq.py::test_grover_amplification PASSED                   [ 36%]
tests/test_microq.py::test_rewind_correctness PASSED                     [ 45%]
tests/test_microq.py::test_rewind_to_genesis PASSED                      [ 54%]
tests/test_microq.py::test_fork_independence PASSED                      [ 63%]
tests/test_microq.py::test_replay_hash_equality PASSED                   [ 72%]
tests/test_microq.py::test_journal_hash_chain PASSED                     [ 81%]
tests/test_microq.py::test_checkpoint_dedup PASSED                       [ 90%]
tests/test_microq.py::test_fidelity_and_norm PASSED                      [100%]

============================== 11 passed in 2.20s ==============================
```
