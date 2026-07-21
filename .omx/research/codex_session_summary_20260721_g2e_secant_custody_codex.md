# Codex Tier-0 session summary — Task #578 G2e secant custody

## Landed in this branch

- Typed candidate-arrangement realized-secant, class/bucket trust, deterministic
  rank-at-most-four QP, packet, and receipt primitives.
- A resumable SSD measurement arm integrated into the settled G2 lattice CLI,
  including fresh candidate-state 144D feature custody, #557 round-trip, hard
  Seg/Pose oracle, and n16/n64/n600 prefix checkpoints.
- Focused regression coverage, including the 34-pair n600 one-write geometry via
  deterministic zero padding of the unavailable fourth RGB direction.
- Corrected n16 real receipt, findings, DAG FEED, and REUSE MANIFEST.

## Measured result

`MEASURED_G2E_SECANT_PREFIX_N16_FAMILY_OPEN` on `[macOS-CPU advisory]`:
64 pair/column rows; usable trust regions 0/31; pair solves
`TRUST_REGION_REFUSED` 16/16; correction bytes 0; whole-description exact 0/16;
declared writes survive 0/97; mean d_seg 0.3777516682942708; mean d_pose
172.29492246623715; 121,128 total bytes with 95,094 bytes headroom.

## Exact blocker transition

Closed at n16:
`R1B2_RANK4_FIRST_ORDER_REALIZED_SECANT_CUSTODY_ABSENT`.

Active, formulation-scoped:
`R1B2_RANK4_REALIZED_SECANT_TRUST_REGION_EMPTY_N16_OPENPILOT`.

## Verification and authority

29 focused tests passed; compile, Ruff lint/format, strict receipt validation,
JSON parse, 200-case SciPy QP cross-check, receipt-corruption checks, two clean
review-tracker passes, and diff checks passed. The worktree carries no review
policy file, so no policy verdict is claimed. Pointer `0.1910828242
[contest-CPU]` is unchanged. No score, promotion, GO, or n600 claim. MAIN
landing review is required.

## Recommended next child

Remeasure paired bidirectional secants at a smaller local amplitude on n16,
retaining exact class/bucket and receiver custody. Reopen QP allocation only
after a nonempty trust region exists; keep Pose correction separate and xi
factorized.
