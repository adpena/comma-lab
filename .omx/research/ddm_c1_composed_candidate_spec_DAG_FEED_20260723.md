---
title: FEED-613 DDM C1 composed candidate target architecture
date_utc: 2026-07-23
lane_id: lane_ddm_c1_composed_candidate_spec_603_613_20260723
research_only: true
execution_allowed: false
pointer: "0.1910828242 [contest-CPU]"
pointer_moved: false
---

# DAG

```text
G1 mask-space descriptions ---------\
v13 worldsheet events ---------------+--> PREDICT
J2 706-DOF lift + 270-B Lane seed ---/       |
#601/#605 transport controls ----------------|  (controls, not additive payloads)
                                                v
                                      PROJECT: one camera/uint8/R master
                                                |
                         v14 3,240,528-error exact control
                                                |
                   +----------------------------+---------------------------+
                   |                                                        |
                   v                                                        v
       REALIZE successor stage race                     v18b common-master columns
       pre-u8 FP <-> post-int8 lattice                  generated all-role vocabulary
       current v17 credit 0; ceiling 726,416
                   |                                                        |
                   +-----------------------+--------------------------------+
                                           v
                                  FINISH: J3/#366
                        lifted worldsheet + Lane + template + xi
                        residual >=2,377,273 errors after max v17
                               and d_pose <=0.00161
                                           |
                                           v
                                    CODE lower envelope
                Aurenhammer | context-arithmetic/Selfcomp | xi-delta |
                              2:4 | MX-int4
                                           |
                                           v
                       one exact archive <=200,000 bytes
                                           |
                    R0 -> R1 -> R2 -> R3 -> R4 -> R5 -> R6
                                  ^                         |
                                  | pair_convergence.jsonl  |
                                  +-------------------------+
```

# Feed contract

The composed candidate is **not fireable**. Measured receiver-closed evidence proves:

- 3,103,689 integer errors must be removed.
- Lane+Movable can own at most 726,416 of them.
- at least 2,377,273 errors therefore require a shared all-role column/joint-finish mechanism.
- the exact seeded control is 134,211 B, leaving 65,789 B to the 200,000-byte box.
- the existing 3,721-byte Pose6 stream is present but scores `d_pose=163.061327281443`;
  J3 must reach the preregistered `d_pose<=0.00161` finish gate.

The live-arm accounting law is:

`E_v17 + E_v18b + E_j3 >= 3,103,689`, `0 <= E_v17 <= 726,416`.

Credit is telescoping on one exact-R archive chain. Independent deltas over the same errors are
never added. SegNet squeeze-excite makes even local same-frame corrections nonlocal; those rows
remain `COMPUTABLE_NOT_YET_COMPUTED` and require combined exact-R replay. Each correction must
race and record its
high-resolution-pre-uint8 versus post-quantization-int8 application cost/effect.

Pending numbers are computable on demand in the frozen space: #391
`src/tac/through_r/flip_inverse.py` supplies the exact adjoint, #549
`tools/measure_realization_g2_lattice.py` / `tools/measure_joint_seg_pose_rate.py` supply the
lattice/joint solve, #580 `tools/measure_resize_full_kernel.py` supplies the projector, and
`tools/measure_arith_selfcomp_rate_coders.py` supplies real coder bytes. The current exact set is
`INFEASIBILITY_CERTIFIED`; the full successor composition is
`COMPUTABLE_NOT_YET_COMPUTED`.

R2 is a per-pair recursive state machine:

`G3 pair row + waterfill -> solve -> exact-R diff -> G4 shared/local route -> repair -> replay`.

Its terminal set is exactly `{threshold-met, infeasible-certified, budget-exhausted}`. Pair
thresholds are derived from the G3 atlas and live byte allocation, never copied from a global
constant. Shared G4 recurrence components are paid once under a stable component id; later pair
rows reference the first owner. The append-only `pair_convergence.jsonl` binds those decisions
to archive/runtime hashes, and R6 refuses an incomplete ledger or duplicate shared charge.

# Hooks

1. **Sensitivity map:** allocate realized error reductions by global/per-role/per-pair exact
   error telemetry, G3 atlas rows, and frozen-scorer Fisher/top1-top2 margin.
2. **Pareto constraint:** hard `archive_bytes<=200000`, integer errors `<=136839`, Pose present,
   and #366 finish `d_pose<=0.00161`.
3. **Bit allocator:** use `25/37,545,489` as the marginal score-byte dual; alternatives share a
   lower convex envelope.
4. **Cathedral/autopilot:** no dispatch from this feed. A consumer may route to #366/J3 only after
   the exact feasibility predicate is true and MAIN reviews the hashes.
5. **Continual-learning posterior:** the durable ledger records the infeasible measured
   composition, conditional swing thresholds, and pair-recursion telemetry schema.
6. **Probe disambiguator:** the v17 successor first races application stage per correction, then
   v17/v18b/J3 are arbitrated by sequential exact-R replay; coder entrants are raced on identical
   semantic content and exact final ZIP bytes.

# Triality

- DSL/data: `.omx/research/ddm_c1_composed_candidate_ledger_603_613_20260723.json`
- equations: `.omx/research/ddm_c1_composed_candidate_spec_canonical_equations_20260723.md`
- design/verdict: `.omx/research/ddm_c1_composed_candidate_spec_603_613_20260723.md`
- lane: `lane_ddm_c1_composed_candidate_spec_603_613_20260723`

Pointer `0.1910828242 [contest-CPU]` remains **UNMOVED**. MAIN landing review is required.
