---
title: Codex findings — DDM V16 coupled joint scorer-margin solve
utc: 2026-07-23T02:01:00Z
tasks: [603, 613, 366]
verdict: INSTANCE_SCOPED_FORK_C
verdict_scope: canonical Lane=1 eight-island operating point x configured local trust boxes
research_only: true
execution_allowed: false
score_claim: false
main_landing_review_required: true
---

# Outcome

V16 lands the reusable coupling machinery but does not lower the measured receiver. The corrected
operator is `151 x 141` (32 target rows, 23 protected rows, 96 pose rows; 72 shared-template and 69
sparse-compensation DOF). Eight sampled finite-difference entries pass the absolute-or-relative
gate with maximum absolute error `1.5648212865926325e-5`. All four local first-order/GN KKT
attempts are residual-unclean, realized correlations are negative, and both rounds select the
unchanged control.

Full n600 is `135,328 B / d_seg 0.027470296224 / d_pose 163.061327281443`; Movable is
`0.291615222639` and Lane is `0.435195521828`. All camera pixels are byte-identical to v15. The
Movable `<=0.05` fork fails; pointer `0.1910828242 [contest-CPU]` is unchanged.

## What landed

- Exact conditional level-set QP/KKT, local GN Hessian, Babai/error-bound, sampled FD validation,
  and quadratic trust-bound primitives.
- Counted v16 compiler/parser/receiver with pair-local template phases, sparse int8-safe RGB
  compensation, deterministic parse/re-encode, exact byte homes, and no scorer/GT table at decode.
- Resumable typed measurement runner with per-round and per-16-pair checkpoints, real receiver
  measurement, archive/video diffs against v15 and GT, and eight-island/n64/n600 ladder receipts.
- Canonical class-order regression, invalidation record, equation anchor, triality JSON/notes, DAG
  feed, and scoped adversarial review.

## Bounded re-derivation

```text
/Users/adpena/Projects/pact/.venv/bin/python tools/measure_ddm_v16_coupled_joint_solve.py --config .omx/research/configs/ddm_v16_coupled_joint_solve_lane_fix_20260723.json --output-directory .omx/research/ddm_v16_coupled_joint_solve_lane_fix_20260723T013500Z
```

One invocation advances one immutable round/rung; repeat until the final receipt is emitted. The
canonical receipt SHA is recorded in the adjacent SHA ledger.

## Round-1 adversarial self-review

1. The recovered runner mislabeled scorer class 4 (MyCar) as Lane. Its receipt SHA
   `170adb825d4d709e...` is invalidated for Lane rows, M geometry, fork, and downstream warm start.
   The canonical mapping is now regression-tested as Lane=1 and Movable=3.
2. The FD statement is sampled and tolerance-scoped: maximum relative error exceeds one on a
   near-zero entry, while the absolute error passes. Do not claim full-matrix parity.
3. No local KKT attempt converged. Numpy also emitted overflow/divide/invalid warnings during the
   rejected proposal solves; saved diagnostics remained finite, but the instability is part of the
   instance verdict and must not be hidden.
4. The largest rejected candidate used 12/23 sparse compensation supports (120 added program
   bytes), yet worsened aggregate d_seg and Movable. Compensation byte cost was not the primary
   blocker; local solve/linearization validity was.
5. The selected archive stores 48 placement records and no compensation. Its 1,387-byte overhead
   versus v15 changes archive bytes but no inflated pixels.

## Blocker delta versus #603

#603/v15 ended with independent zero-collateral subproblems and no joint coupling operator. V16
supplies the durable local M, counted compensation actuator, conditional equations, uint8
projection, and real-receiver recursion. The blocker has moved to an exact instance diagnosis:
the configured local KKT systems did not converge and their integer receiver realizations diverged
from the affine prediction. #366 remains open as nonlinear joint descent; M may be reviewed as a
warm-start/costate/preconditioner signal only.

## STORES CONSULTED

`CLAUDE.md`; `AGENTS.md`; `PROGRAM.md`; `docs/operating_manual_craft_handoff.md`; v7.5/v8 specs;
v14/v15 receipts, DAG feeds, and equations; #391/#549/#341/#423/#532/#586 built surfaces;
frozen n600 target cache; `reports/latest.md`; lane/task/progress state; operator directives through
2026-07-23T00:31:59Z.

MAIN landing review is required.
