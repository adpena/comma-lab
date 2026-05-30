# Codex Findings: z6_v2 MLX Full-Run Axis Label Review

**UTC:** 2026-05-30T21:55:51Z
**Reviewed commit:** `78c1db48b3dcf0eb37f4c3f977e53c8f3799994f`
**Scope:** partner landing for `lane_z6_v2_canonical_29650ep_mlx_local_full_run_20260530`
**Verdict:** `PROCEED_WITH_AXIS_LABEL_CAVEAT`

## Findings

1. The landing preserves the critical authority boundary: the z6_v2 result is
   explicitly advisory/research-only, not promotable, and blocked on canonical
   contest inflate output plus substrate shrinkage before exact CPU/CUDA
   consideration.

2. The canonical-equation appended anchor for the 29,650-epoch run records
   `measurement_axis="[macOS-CPU advisory]"` and
   `evidence_grade="macos_cpu_advisory"` while the actual described run is
   MLX-local (`[macOS-MLX research-signal]`). This is not a score-authority
   violation because `promotion_eligible=false` and `score_claim_valid=false`,
   but it is an axis-label ambiguity that downstream consumers must not read as
   a completed macOS-CPU auth-eval smoke.

## Required Follow-Up

- Treat the z6_v2 29,650-epoch anchor as `[macOS-MLX research-signal]` for
  acquisition and posterior routing unless a separate macOS-CPU advisory payload
  with source artifact custody is present.
- Future MLX long-run registry appenders should use an MLX provenance builder or
  an explicit `[macOS-MLX research-signal]` measurement axis instead of the
  macOS-CPU advisory builder.
- Do not promote, rank, kill, or exact-dispatch from this anchor alone.
