# DDM v18 generated-column vocabulary — DAG FEED

- lane_id: `ddm_v18_column_generation_vocabulary`
- phase: 2
- research_only: `true`
- score_claim: `false`
- pointer_moved: `false`
- MAIN landing review: `required`

## Triality

- DSL:
  `.omx/research/configs/ddm_a1_column_generated_n64_20260723.json` validated by
  `DDMA1ColumnGeneratedCorrectionConfigV1`.
- DAG: this file.
- Equation:
  `tac.canonical_equations.ddm_v18_column_pricing_law_20260723.ddm_column_reduced_cost`.

## Executable readiness DAG

```text
S0 bound v12 receipt + exact archive bytes
S1 bound G1 receipt
S2 bound v15 receipt + exact archive bytes
S3 bound v16 receipt
 └── G0 common receiver/master source closure
      ├── require one camera-resolution uint8 -> evaluator-R path
      ├── require one hybrid parse-back grammar and exact byte ownership
      └── BLOCKED_PRECONDITION_NO_COMMON_EXACT_R_MASTER
           ├── P0 hybrid receiver/archive schema [OWED TO MAIN]
           └── P1 common exact-R control remeasurement [OWED TO MAIN]

G0 PASS (future)
 └── A0 generate n64 columns
      ├── residual + Task #391 R-adjoint VJP
      ├── G1 grammar coordinates
      ├── v15/v16 template DOF
      ├── Fisher/margin EV rank
      ├── corrected inner-Jacobian step law
      └── curvelet/shearlet boundary supports
           └── M0 restricted-master LP
                ├── exact coder-byte constraint
                ├── conflict constraints
                └── actual HiGHS duals
                     └── P2 reduced-cost pricing
                          r_j = c_j - b_j*y_b - sum_k A_kj*y_k
                           └── S4 global conflict-aware selector
                                ├── beam width 32
                                └── conflict MIQP entrant
                                     └── R0 exact replay of every explored set
                                          ├── paint -> uint8 -> R -> scorers
                                          └── exact archive bytes
                                               └── C0 coder race at matched d_seg
                                                    ├── explicit indices
                                                    ├── 2-of-4 support metadata
                                                    └── shared-scale int4 MX payload
                                                         └── N0 n600 exact replay
                                                              └── four equal-byte rows
```

No edge below `G0` executed. The receipt preserves that as the resumable
`round_00_source_closure` boundary; null pricing counts cannot be interpreted
as zero negative-reduced-cost columns.

## Current canonical feed

- Receipt:
  `.omx/research/ddm_v18_column_generation_vocabulary_20260723T030000Z/ddm_a1_column_generated_correction_receipt.json`
- Receipt SHA-256:
  `f8daae958510e6cea9cf39b499ec820d6747dd968e9a51157e0a5a3e25601a96`
- Findings:
  `.omx/research/codex_premise_falsification_ddm_v18_column_generation_vocabulary_20260723_codex.md`
- Falsifier:
  **OPEN**. It requires three complete exact pricing rounds with no negative
  reduced-cost columns **and** four complete global equal-byte replays with no
  v12 beat.

## Unified-system wiring disposition

This landing is explicitly `research_only=true` and cannot feed promotion,
bit allocation, or dispatch because source closure failed before any empirical
column value existed. The reusable pricing helper is ready to consume measured
singleton objective deltas, exact coder bytes, conflicts, and dependencies.
After `G0` closes, MAIN should route accepted exact-replay rows into the
sensitivity/EV surface, Pareto constraint, bit allocator, autopilot candidate
queue, and continual-learning posterior. Until then, emitting any such row
would create an orphaned false signal.

## Operator amendment consumed

The 2026-07-23 coding-format directive is bound into the DSL as three coder
entrants. The structured 2-of-4 and MX-block modes are entrants to the existing
race, not new goalposts, and remain explicitly unmeasured.
