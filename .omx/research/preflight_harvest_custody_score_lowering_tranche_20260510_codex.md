# Preflight + Harvest Custody Score-Lowering Tranche (2026-05-10)

Generated: `2026-05-10T20:40:00Z`

`research_only=true`; `score_claim=false`; `remote_dispatch_attempted=false`.

## Scope

This tranche hardened two infrastructure surfaces that directly gate score
lowering:

1. developer preflight wall-clock behavior under parallel scanners;
2. GHA CPU harvest custody for A1/PR101 inflate-time bias candidates.

It did not open a new dispatch claim and did not launch a remote/GPU job.
The active T1 Modal claim remains open and was polled, not duplicated.

## Active Dispatch Custody

Current active dispatch remains:

- lane: `t1_balle_128k_endtoend`
- job: `t1_balle_modal_phase1_ab2d0f6_20260510T1437Z`
- platform: `modal`
- call id: `fc-01KR955JSYQAVTTYZA48VAV7WJ`
- latest local recovery result: `NOT READY`, still queued or running

No same-lane duplicate dispatch was opened.

## Preflight Changes

`src/tac/source_index.py` now uses per-key single-flight locks for file lists,
pattern groups, text reads, AST parses, text facts, fact groups, and substring
group fills. This prevents parallel preflight workers from racing duplicate
cache fills while preserving the existing scanner predicates.

The previously eager source-index prewarm is now opt-in via:

```bash
PACT_PREFLIGHT_PREWARM_SOURCE_INDEX=1
```

Measured cold developer preflight with the incremental clean-cache disabled:

```text
default lazy fill:  wall_elapsed_s=7.332081, failed_step_count=0
opt-in prewarm:    wall_elapsed_s=7.826992, failed_step_count=0
```

Conclusion: default lazy single-flight is the lower wall-clock path in this
checkout. Eager prewarm remains available for future profiling but is no
longer on by default.

## PCC2 Self-Trigger Fix

The source-index prefilter needs the literal needle `overrides this stub` for
comment-only-contract detection. Holding that literal in
`src/tac/source_index.py` made the file self-trigger PCC2. The string is now
spelled as adjacent literals so runtime semantics are unchanged while the
source file no longer contains the forbidden phrase.

Focused profile after the fix:

```text
check_no_comment_only_contracts + check_codebase_drift: 1.326s, passed
```

## Harvester Custody Fix

Red-team review found that `tools/harvest_a1_bias_correction_sweep.py` could
accept a GHA `report.txt` by exact `submission_name` without proving the
evaluated artifact matched the local archive/runtime identity.

The harvester now:

- extracts workflow `headSha`;
- collects archive SHA/bytes, `inflate.py` SHA, and runtime-tree SHA from
  downloaded JSON artifacts when present;
- builds expected identity from the local rollup/build manifest;
- keeps report-only artifacts as
  `status=report_identity_incomplete`, `score_claim=false`,
  `exact_report_custody=false`;
- allows `[contest-CPU GHA Linux x86_64]` score claims only when archive,
  inflate, runtime tree, and report byte identity are all bound.

This fixes an evidence overclaim path. It does not make any existing
constrained-coordinate candidate dispatch-ready; those packets still need
runtime custody, exact smoke, no-op output-change proof, a fresh claim, and
then GHA/CUDA evaluation.

## Verification

Focused tests:

```text
93 passed in 2.57s
```

Covered:

- `src/tac/tests/test_preflight_all_clean_cache.py`
- `tests/test_preflight_source_index_equivalence.py`
- `src/tac/tests/test_source_index.py`
- `tests/test_harvest_a1_bias_correction_sweep.py`
- `src/tac/tests/test_no_comment_only_contracts.py`
- `src/tac/tests/test_dispatch_cli_shell_hazards.py`

Additional checks:

```text
git diff --check: passed
py_compile: passed
developer preflight default: passed, 7.332081s cold
developer preflight opt-in prewarm: passed, 7.826992s cold
```

## Score-Lowering Consequence

Do not dispatch the existing A1 per-pair sidecar proxy-MSE candidate; it is
already retired as a measured `[contest-CPU]` regression.

Do not dispatch broad A1/PR101 bias variants directly from the current
constrained-coordinate manifests. The next unblocked local score-lowering
work is to produce a small refined PR101-bias packet set with full archive and
runtime identity, exact smoke, and no-op output-change proof. Only then should
GHA CPU rows be harvested, and only a CPU-positive row should graduate to a
fresh exact-CUDA claim.

Optuna/CMA-ES remains useful as `score_claim=false` proxy/config search only
until the identity and runtime-consumption gaps above are closed.
