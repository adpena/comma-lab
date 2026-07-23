---
schema: codex_findings_ddm_v17_iterative_realized_trust_region.v1
date_utc: 2026-07-23T03:42:00Z
lane_id: ddm_v17_iterative_realized_trust_region_solve
axis: "[macOS-CPU frozen-scorer advisory]"
research_only: true
execution_allowed: false
score_claim: false
main_landing_review_required: true
---

# DDM v17 / Probe A iterative realized trust-region findings

## Outcome first

`MEASURED_ADVISORY_INITIAL_GRID_PLATEAU_FORMULATION_OPEN`.

The operator-supplemented Probe A contract was measured exactly on its eight preregistered pairs:
three template bases, lattice radii `{1,2,4,8}`, collateral caps `{0,16,32,64}`, the full
`100*d_seg + sqrt(10*d_pose) + 25*bytes/37545489` objective, and 12 solve versus 12 unique
model-disabled j2 exact receiver calls. No candidate both improved the realized objective and
stayed within `epsilon <= 64`, so the held v16 control remained selected. The pointer remains
`0.1910828242 [contest-CPU]` and no n64/n600 row is claimed.

This is an **INSTANCE** result for the measured KKT/Babai and ranked-prefix proposal grid at one
realized point. It does not close contextual templates, the direct-description family, parallel
tempering, nonlinear joint training, or a clean KKT formulation.

## Exact binding decomposition

| Basis / best relevant radius | Joint-objective delta | d_seg delta | Byte delta | Harmful flips | Disposition |
|---|---:|---:|---:|---:|---|
| `1x1_rowband_control`, r=4 | `-0.004767545957` | `-0.000050226847` | `+201` | `405` | Improves objective but exceeds epsilon=64 by 341; collateral cap binds. |
| `2x2_contextual`, r=1 | `+0.001669068881` | `+0.000016530355` | `+20` | `53` | Admissible only at epsilon=64, but exact objective worsens. |
| `boundary_normal_2x2`, r=1 | `+0.000076779716` | `+0.000000635783` | `+20` | `1` | Admissible at epsilon 16/32/64, but exact objective worsens. |

The 1x1 r=1 point also improved objective by `-0.001584977601`, but required 98 harmful flips.
Thus v15's zero-collateral extreme was genuinely too strict for some net-improving moves, while
the preregistered bounded ladder still stopped short of the first improving point. This names the
current binder without converting it into a family negative.

All three KKT solves ended `MAX_ITERATIONS_RESIDUAL_NOT_CLEAN`. The 1x1 subspace collapsed four
Babai radii to one unique point, so the runner fail-closed and used four distinct lattice points
from the damped `M^T W M`-preconditioned direction. In that arm the preconditioned sign path and
j2 sign control coincided; contextual and boundary-normal KKT candidates remained distinct.

## Validity-radius curve

The registered, basis-conditioned Fisher-margin ratio curve is:

| Lattice quanta | Candidate count | Finite rho count | Median rho | Mean rho | Negative rho count |
|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 3 | `-0.446150087525` | `-1.642513866047` | 2 |
| 2 | 3 | 2 | `-4.088996609328` | `-4.088996609328` | 1 |
| 4 | 3 | 1 | `-1.242440547478` | `-1.242440547478` | 1 |
| 8 | 3 | 0 | `N/A` | `N/A` | 0 |

At the contextual one-quantum point, rho was `-0.446150087525`; the trust controller therefore
hard-shrank to the one-quantum floor. Boundary-normal r=1/r=2 had positive rho (`0.7430` and
`0.8466`) but still worsened the real joint objective. Rho diagnoses model validity and controls
radius; it never confers acceptance.

## Solve versus model-disabled j2 control

Both forms received 12 unique exact receiver/scorer calls on the same pairs and per-basis DOF.
Neither produced an admissible hard improvement, so the aggregate comparison is a tie at zero
accepted gain. This is not evidence of equal unconstrained potential: the 1x1 paths coincide after
the explicit M-preconditioned fallback, while contextual and boundary-normal solve/control paths
are distinct and both worsen or violate the collateral cap.

## Scope, ceiling, and next information

- The measured Lane+Movable mechanism can address at most **23.404922%** of remaining in-box Seg
  debt. That is a derived ceiling, not expected gain.
- Fixed paint still leaves **60.561878%** of the Movable projection gap; this run does not convert
  that formulation qualifier into an irreducible loss claim.
- No dev winner means n64 and n600 are correctly `NOT_RUN`; no score or d_seg claim exists.
- The next non-duplicate evidence must change the proposal family: an epsilon rung above 64 with
  explicit marginal collateral pricing, a clean constrained solve, or deterministic PT/beam
  escape. Repeating the same initial grid would not create a second realized iteration.

## Triality and custody

- DSL: `.omx/research/configs/ddm_a1_bounded_collateral_realized_n64_20260723.json`, typed hash
  `2594e458c9a8567a386ab3c8bdfe6c82c825bfedfc913e0300449ef51197bed1`.
- DAG: `.omx/research/ddm_v17_iterative_realized_trust_region_DAG_FEED_20260723.json`.
- Equation: `ddm_v17_realized_validity_ratio_uint8_v1`, registered through the locked helper.
- Receipt: `.omx/research/ddm_a1_bounded_collateral_realized_n64_20260723T031500Z/ddm_a1_bounded_collateral_realized_receipt.json`, SHA-256
  `7cad1d17c5697b578234db282e6191ba3f2789b188051473874638392e18f53a`.
- The original top24 run is preserved but explicitly invalidated as pre-directive exploratory
  evidence; its interrupted n600 batches carry no score authority.
- Three-pass review found and repaired one unexecuted-path defect: a future accepted-step n600
  rung had treated a missing bound-control objective as automatically no worse. The current runner
  now binds the SHA-verified v15 n600 score row and fails closed on archive mismatch. This did not
  affect the measured plateau because zero steps were accepted and neither ladder rung ran. The
  review receipt and reverse reconstruction patch preserve both producer SHAs.

STORES CONSULTED: `CLAUDE.md`; `AGENTS.md`; `PROGRAM.md`;
`docs/operating_manual_craft_handoff.md`; v7.5/v8 operating specs; v15/v16 source receipts;
`ddm_a1_naive_verdict_audit_20260723_codex.md` at main commit `fecbefe5a5`; #579/#586/#599
crosswalks; canonical equation registry; lane/progress registries; operator inbox through
`2026-07-23T02:55:15Z`.

MAIN landing review must independently check the basis projection maps, harmful-flip definition,
all 24 exact-call rows, the lack of an admissible winner, the equation's instance boundary, and
that neither partial n600 data nor macOS advisory values move the contest pointer.
