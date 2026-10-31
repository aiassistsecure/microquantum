# microquantum — STATE

Manager state for the hourly dev loop. Repo: aiassistsecure/microquantum (public).
Local workdir: ~/workspace/microquantum.
Honest framing always: classical simulation of quantum math, no quantum
hardware, no fabricated benchmarks — unrun numbers are marked UNMEASURED.

## Roadmap / slice status

- S1 — state-vector engine + Bell/GHZ/Grover demos + journal + tests + README: **DONE, verified + landed 2026-10-31**
  - Manager's own verification this run: `pytest` 11 passed; Bell + GHZ demos exit 0
    (replay True, rewind True); Grover run separately: P(marked) 0.0625 → 0.4727 →
    0.9084 → 0.9613 over 3 iterations, journal 424 entries, replay True, rewind(212) True.
    Numbers match SLICE1-RESULTS.md exactly.
  - Journal backend verified live: `JournalStore.backend == "nedb"` (real nedb-engine),
    `verify_store() == True`.
- S2 — real NEDB wire-in for journal/checkpoints: **DONE in substance inside S1**
  - S1 already uses real NEDB as the primary journal store (same entry schema);
    JSONL is fallback-only when the nedb binding is unavailable. Checkpoints are
    content-addressed .npy blobs by design (SI rule: blobs by reference, journal in NEDB).
  - Remaining optional polish (not blocking): surface NEDB AS OF / TRACE provenance
    reads in the circuit API (README "What's next" #2).
- S3 — CUDA/CuPy backend for the V100 behind the backend interface, CPU-fallback
  tested here: **NEXT**. Real V100 numbers UNMEASURED until run at the box.
- S4 — Grover search over records actually stored in NEDB: pending.
- S5 — QFT + quantum phase estimation demo, journaled: pending.
- S6 — VQE-style variational loop with journaled checkpoints + rollback: pending.

## Last results

- 2026-10-31 ~13:05 EDT (manager run): S1 verified (11/11 tests, demos above),
  initial commit pushed to GitHub main (see git log / remote for head sha).
- Free disk at that run: 107G available on /home (200G total, 47% used).
