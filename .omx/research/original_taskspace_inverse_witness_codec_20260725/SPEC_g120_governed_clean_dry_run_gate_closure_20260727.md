# G120 governed clean dry-run gate closure

Date: 2026-07-27  
Task: `pact-g120-governed-clean-dry-run-gate-20260727`  
Axis: `[macOS-CPU orchestration dry-run; no scorer or score authority]`  
Parent contracts: G120-v2, G121  
Status: implementation specification

## Objective

Close the literal gate required by
`SPEC_g120_parsed_stage_production_authority_20260727.md` before the G121 live
monitor may run on a real G111 producer:

1. make G120-v2 actually resume from each physically complete prediction-batch
   receipt without repeating that scorer batch;
2. provide a governed two-process dry-run that binds the exact producer launch,
   monitor output/progress directories, SSD measurement-cache root, storage
   preflight, and the production batch-resume primitive;
3. require the G121 live monitor to reopen that exact dry-run receipt by
   externally supplied SHA-256 before any harvest call.

The dry-run is apparatus evidence only. It must execute no SegNet scorer, emit
no G120 semantic measurement, and claim no score, candidate, promotion, or
pointer movement.

## Literal invocation

Checkpoint phase:

```bash
.venv/bin/python tools/run_g120_governed_clean_dry_run.py checkpoint \
  --producer-run-dir /Volumes/VertigoDataTier/pact/G111_REAL_RUN \
  --expected-launch-manifest-sha256 MANIFEST_SHA256 \
  --monitor-output-dir /Volumes/VertigoDataTier/pact/G111_REAL_RUN/g121_harvest \
  --monitor-progress-dir /Volumes/VertigoDataTier/pact/G111_REAL_RUN/g121_progress \
  --measurement-cache-dir /Volumes/VertigoDataTier/pact/G111_REAL_RUN/g121_measurement_cache \
  --gate-dir /Volumes/VertigoDataTier/pact/G111_REAL_RUN/g120_dry_run_gate
```

Resume phase, in a distinct process, uses the same arguments with `resume` in
place of `checkpoint`. It emits the immutable completion receipt. The live
monitor invocation then adds:

```text
--g120-dry-run-receipt RECEIPT_PATH
--expected-g120-dry-run-receipt-sha256 RECEIPT_SHA256
```

## Exact custody and storage requirements

- The producer launch manifest is reopened by an externally supplied SHA-256
  and must satisfy the existing G111/G121 n600, batch-16, HOSC,
  exact-polar/self-orient-off, 384x512/no-AA, generated-Y1/V10 contract.
- Producer, monitor output, monitor progress, measurement cache, and gate
  directories are absolute, physical, non-symlink paths.
- All five paths are below one configured SSD pact root:
  `/Volumes/VertigoDataTier/pact`, falling back only to
  `/Volumes/APDataStore/pact`.
- Output, progress, cache, and gate directories are pairwise distinct.
- Storage preflight records device, free bytes, and the exact conservative
  bound derived from two full n600 prediction populations plus four maximum
  batch-16 scorer/camera working sets:

  `B_min = 2*N*384*512 + 4*16*(384*512*3 + 874*1164*3)`.

- The checkpoint phase writes one deterministic batch-16, 384x512 physical
  prediction array and its canonical receipt through the same immutable batch
  persistence primitive used by G120 production.
- The resume phase must run under a different PID, reopen the same bytes
  through the production validation primitive, and prove no scorer callback
  was available or invoked.
- The completion receipt binds the current source bytes for G120-v2, the
  dry-run gate, G121, and the live monitor. A source change invalidates the
  gate until it is rerun.

## Production batch-resume law

For batch `b` with exact execution key `E`:

`complete_b(E) = receipt_b exists AND array_b exists AND`
`reopen(receipt_b, array_b, target_b, scorer_hash_b, camera_hash_b) == exact`.

Then:

`scorer_calls_b = 0` if `complete_b(E)`, otherwise `1`.

Any malformed, mismatched, or receipt-without-array state fails closed. An
array written before a crash but lacking its receipt may be recomputed; the
immutable write requires the recomputed bytes to match.

## Files

- `src/tac/witness_dsl/g120_parsed_stage_production_authority_v2.py`
- `src/tac/witness_dsl/tests/test_g120_parsed_stage_production_authority_v2.py`
- `src/tac/witness_control/g120_governed_clean_dry_run_gate_v1.py`
- `src/tac/witness_control/tests/test_g120_governed_clean_dry_run_gate_v1.py`
- `tools/run_g120_governed_clean_dry_run.py`
- `tools/run_taskspace_g121_live_stage_harvest.py`
- `tools/tests/test_run_taskspace_g121_live_stage_harvest.py`

## Acceptance

```bash
.venv/bin/python -m ruff check \
  src/tac/witness_dsl/g120_parsed_stage_production_authority_v2.py \
  src/tac/witness_dsl/tests/test_g120_parsed_stage_production_authority_v2.py \
  src/tac/witness_control/g120_governed_clean_dry_run_gate_v1.py \
  src/tac/witness_control/tests/test_g120_governed_clean_dry_run_gate_v1.py \
  tools/run_g120_governed_clean_dry_run.py \
  tools/run_taskspace_g121_live_stage_harvest.py \
  tools/tests/test_run_taskspace_g121_live_stage_harvest.py

.venv/bin/pytest -q \
  src/tac/witness_dsl/tests/test_g120_parsed_stage_production_authority_v2.py \
  src/tac/witness_control/tests/test_g120_governed_clean_dry_run_gate_v1.py \
  tools/tests/test_run_taskspace_g121_live_stage_harvest.py
```

Required cases:

- a crash after batch 0 resumes without repeating batch 0 scorer work;
- corrupted or mismatched physical batch state fails closed;
- checkpoint and resume phases require distinct processes and exact bindings;
- non-SSD or insufficient-free-space roots fail closed;
- monitor refuses absent, stale, wrong-producer, wrong-directory, or
  source-stale gate receipts;
- no dry-run path imports or invokes the scorer.

## Do not touch

- G110 release materializer, public plugin semantics, or candidate payloads;
- pinned upstream files or scorer weights;
- the active G111 producer process;
- any exact-eval, paid dispatch, or full-n600 scorer launch.

