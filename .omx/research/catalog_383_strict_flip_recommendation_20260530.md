# Catalog #383 STRICT-flip recommendation 2026-05-30

**Operator-routable**: per CLAUDE.md "Strict-flip atomicity rule" — STRICT-flip is operator-routable, NOT auto-flipped by subagent.

**Lane**: `lane_gumbel_softmax_sample_canonical_extraction_migration_20260530` L1.

## Catalog #383 current state

- **Wire-in**: WARN-ONLY (`preflight_all(..., strict=False, ...)` calls `check_mlx_primitives_route_through_canonical_helper`) per MLX canonicalization landing commit `e52f2f6b4`.
- **Live count**: 0 (verified empirically via `.venv/bin/python -c "from tac.preflight import check_mlx_primitives_route_through_canonical_helper; v = check_mlx_primitives_route_through_canonical_helper(strict=False, verbose=False); print(len(v))"` returns 0).
- **STRICT-flip readiness**: YES (live count at 0; canonical extraction migration via PRINCIPLED FORK waivers per Catalog #290 falling-rule landed in same commit batch as THIS recommendation memo).

## Recommendation

Flip Catalog #383 wire-in to STRICT (`preflight_all(..., strict=True, ...)`) per CLAUDE.md "Strict-flip atomicity rule" non-negotiable:

> If the fix subagent achieves live count = 0 in the same landing, the strict-flip should land in the SAME commit-batch (not a follow-up). This avoids the warn-only-purgatory failure mode where a check ships warn-only and the strict-flip never happens.

## Why operator-routable (not auto-flipped by THIS subagent)

Per the SCOPE constraint of THIS subagent prompt: "**IF live count reaches 0**: prepare Catalog #383 STRICT-flip recommendation memo per CLAUDE.md "Strict-flip atomicity rule" (operator-routable; do NOT auto-flip strict mode in this lane)".

The operator decides the STRICT-flip cadence because:
1. THIS lane is sister-DISJOINT vs PR110-OPT-7 paired-CUDA + z6_v2 Phase C + Wyner-Ziv canonical equation (per Catalog #340 sister-checkpoint guard); flipping STRICT mid-flight could surface unexpected violations in sister lanes' working trees that haven't yet committed.
2. The 2 PRINCIPLED FORK waivers are NEW; operator review of the waiver rationales is appropriate before structural enforcement.
3. Future canonical helper signature extension OR substrate-callsite refactor are operator-routable reactivation paths per Catalog #290; locking STRICT now does not block these paths but does require waiver rotation when they execute.

## Suggested operator command

When ready to flip:

```bash
# Edit src/tac/preflight.py at preflight_all() callsite:
#   check_mlx_primitives_route_through_canonical_helper(strict=True, ...)
# (currently strict=False per MLX canonicalization landing)
```

Then verify:

```bash
.venv/bin/python -c "from tac.preflight import preflight_all; preflight_all(strict=True)"
```

If exit code = 0: strict-flip successful. If raises `PreflightError` with Catalog #383 violations: a sister lane introduced a new violation; review the waiver rationale or add a new PRINCIPLED FORK waiver.

## Cross-references

- Landing memo: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_gumbel_softmax_canonical_extraction_migration_landed_20260530.md`
- Per-substrate verdict table: `.omx/research/gumbel_softmax_canonical_extraction_migration_20260530.md`
- Retroactive sweep memo: `.omx/research/retroactive_sweep_for_catalog_383_gumbel_softmax_migration_20260530T230000Z.md`
- Catalog #383 source: `src/tac/preflight.py:check_mlx_primitives_route_through_canonical_helper`
- Parent canonical extraction landing: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_mlx_canonicalization_audit_plus_tinygrad_bridge_plus_6_pillar_discipline_landed_20260530.md`
