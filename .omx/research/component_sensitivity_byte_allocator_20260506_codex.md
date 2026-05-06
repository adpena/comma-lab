# Component Sensitivity Byte Allocator - 2026-05-06 Codex

## Scope

Hidden-gem registry key: `component_sensitivity_byte_allocator`.

This tranche adds manifest-level byte accounting to the OWV3
sensitivity-weighted renderer payload. The goal is to make local byte-plan
artifacts useful as compiler feedback without turning byte-only evidence into a
score claim.

## Patch

- `src/tac/owv3_sensitivity_weighted.py` now emits:
  - `byte_plan.action_bytes`
  - `byte_plan.fallback_reason_counts`
  - `byte_plan.fallback_reason_bytes`
- `src/tac/tests/test_owv3_sensitivity_weighted.py` verifies:
  - mixed OWV3 layers account `owv2_low_bit`, protected `asym`, and `bias`
    bytes;
  - all-protected layers account `all_channels_protected` fallback bytes;
  - diagnostic FP16 fallback remains explicit and non-promotable.

## Evidence Discipline

Evidence grade: `empirical` for local byte-plan manifest structure only.

This does not claim score movement. Any candidate selected from these byte
plans still requires exact CUDA auth eval on exact archive bytes before
promotion, ranking, or writeup score claims.
