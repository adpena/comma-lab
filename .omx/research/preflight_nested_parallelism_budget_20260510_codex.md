# Preflight Nested Parallelism Budget — 2026-05-10

Status: LANDED-PENDING-COMMIT
Owner: codex
Scope: developer preflight wall-clock performance, no score claim

## Summary

The previous developer-preflight default launched up to 8 independent checks.
Each broad check could also ask `SourceIndex.facts_for_files()` to spawn up to
32 text-fact workers. On this repo that nested fan-out was slower than the
sequential path because all checks contend on the shared source-index caches,
filesystem, and Python thread scheduling.

This tranche bounds both layers:

- `PACT_PREFLIGHT_PARALLEL_WORKERS` defaults to 2 and clamps to `[1, 16]`.
- `PACT_SOURCE_INDEX_FACT_WORKERS` defaults to 8 and clamps to `[1, 32]`.

The knobs keep profiling flexibility for larger CI hosts while making the
normal local path deterministic and less prone to oversubscription.

## Evidence

Focused tests:

```text
.venv/bin/python -m pytest \
  src/tac/tests/test_source_index.py \
  tests/test_preflight_source_index_equivalence.py \
  src/tac/tests/test_preflight_all_clean_cache.py \
  src/tac/tests/test_preflight_proactive_checks.py \
  tests/test_harvest_a1_bias_correction_sweep.py -q

80 passed in 6.86s
```

Cold developer preflight, incremental cache disabled:

```text
before nested-budget patch:
  wall_elapsed_s=6.974
  serial_elapsed_s=48.988
  failed_steps=0

after default PACT_PREFLIGHT_PARALLEL_WORKERS=2:
  wall_elapsed_s=6.500
  serial_elapsed_s=12.662
  failed_steps=0
```

Comparison sweep from the same working tree:

```text
PACT_PREFLIGHT_ENABLE_PARALLEL=0: wall=6.482s serial=6.405s
PACT_PREFLIGHT_PARALLEL_WORKERS=2: wall=6.416-6.500s serial=12.512-12.662s
PACT_PREFLIGHT_PARALLEL_WORKERS=3: wall=6.529s serial=18.699s
PACT_PREFLIGHT_PARALLEL_WORKERS=4: wall=6.739s serial=25.374s
previous max_workers=8: wall=6.974s serial=48.988s
```

## Score-Lowering Consequence

This does not move score directly. It protects the score-lowering loop by
making routine preflight stay far under the 30s DX crash budget while preserving
the strict checks that block stale exact-ready queues, duplicate dispatches, and
custody-free score promotion.

The next score-lowering action remains active T1 Modal recovery:

- lane: `t1_balle_128k_endtoend`
- job: `t1_balle_modal_phase1_ab2d0f6_20260510T1437Z`
- call: `fc-01KR955JSYQAVTTYZA48VAV7WJ`
- current state during this tranche: `pending` / `NOT READY`

No duplicate T1 launch is allowed while that active claim remains open.
