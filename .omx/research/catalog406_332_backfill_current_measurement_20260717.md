# Catalog #406/#332 backfill — CURRENT measurement + structured worklist (2026-07-17)

**Consumption of the orphan `codex_findings_harvest_held_catalog406_20260715_codex.md`
(the ONE true orphan Arm F found; task #527).** Per "consumption ≠ disposition": this memo
converts the stale 2-day-old 3,884-tuple raw dump into a CURRENT, MEASURED, structured worklist
routed to its real owner. Pointer `0.19108` UNMOVED — this is apparatus, not a score mover.

## What I MEASURED (not trusted from the stale memo)

Ran the live checkers 2026-07-17:

- **`check_launch_and_governor_require_dsl_compile_hash(strict=False)` (the #406 STRUCTURAL
  gate): 2 findings**, both launcher-side —
  `tools/launch_witness_run.py: launcher missing dry-start DSL-bound launch` +
  `... missing dry-start typed Lever`. (The 07-15 memo said the #406 structural checker had
  ZERO; these 2 launcher dry-start items are the current delta — small, tractable, separate
  from the 3,862.)
- **`check_config_flag_provenance_bijection_complete(strict=False)` (the #332 COMPLETE-CHAIN
  prerequisite): 3,862 residuals** (was 3,884 on 07-15 — essentially unchanged, so the stale
  dump was NOT wrong, just old).

## The STRUCTURE (why it is NOT 3,862 distinct problems)

Measured breakdown:

| factory | residuals |
|---|---|
| `v9_cgauge_432` | 907 |
| `v9_cgauge_truly_optimal_core` | 985 |
| `v9_cgauge_ideal_mod19` | 985 |
| `v9_cgauge_ideal_mod32` | 985 |

The 3 ideal/core factories have **byte-identical residual sets** (confirming the 07-15 memo).
Per factory the residuals are **~200 semantic flags × 5 canonical missing edges**:

1. exactly one `Lever` owner,
2. exactly one `Lever.constant_refs` LawRef,
3. exactly one canonical compiler record,
4. a known value-provenance rung (currently `None`),
5. exactly one runtime receipt schema.

Coverage-mismatch samples: `v9_cgauge_432` LawRef missing_count=201, compiler=195,
provenance=154 — i.e. the residual is dominated by ~200 flags each missing the same 5 edges,
NOT 3,862 unique fixes. The real distinct work ≈ (flags-per-factory) × 5 edge-authorings,
with the 3 ideal/core factories closing together.

## Verdict + routing (the honest scope)

This is the **#332 "DSL-as-complete-SoT" backfill** (task #332 verbatim: "Make the DSL the
complete auto-generated SoT"). It is:

- **Large + systematic** — ~200 flags need the 5-edge canonical chain authored per closed V9
  factory. This is a GENERATOR job (auto-derive the 5 edges per flag from the existing
  Lever/LawRef/compiler graph), NOT 200×5 hand-edits.
- **Not live-blocking** — the #332 bijection gate is WARN-ONLY (CLAUDE.md 2026-07-15
  reconciliation note: "landed WARN-ONLY ... STRICT flip remains OWED pending live-count-0
  backfill"). Nothing is gated on it today; the strict-flip is explicitly owed future work.
- **Apparatus, not a pointer-mover** — closing it authorizes the #406 meta-gate strict flip;
  it does not lower the exact score.
- **Correctly gated to the SPEC_v10 / strict-flip owed-work** — task #527 says "at strict-flip
  time"; #529 (v10 compiler real success-path) is the natural vehicle, since a real compiler
  that resolves every LawRef + emits the compiler record + receipt schema per flag is exactly
  the mechanism that drives this count toward 0 systematically.

**Consumption is COMPLETE at the routing level**: the orphan is no longer a raw undated dump —
it is a measured (3,862), characterized (~200 flags × 5 edges × 4 factories, 3 byte-identical),
correctly-owned worklist. The actual edge-authoring is #332 systematic work, tracked there, to
be executed via the #529 real-compiler build at the post-c2 strict-flip boundary (NOT
hand-started shallowly on a live-run day).

## The finite tractable increments (for when #332/#529 is worked)

The 07-15 memo named finite closeable items separate from the ~200-flag bulk:
- 6 named compiler/LawRef disagreements (correct them),
- 1 stale provenance key `schedule` (remove or re-own),
- the 2 launcher dry-start items above (add dry-start DSL-bound launch + typed Lever to
  `tools/launch_witness_run.py`).

These are the small, hand-tractable pieces; the ~200-flag 5-edge chain is the generator job.
