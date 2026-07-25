---
title: DDM MR1 independent-approver review of RG4
date_utc: 2026-07-25
reviewer: mr1-independent-approver
reviewed_tip: cb34bbb0f119f790015c2561e2b57d0470580537
main_landing_review_required: true
score_claim: false
pointer_moved: false
---

# Verdict

`MERGE-WORTHY AFTER FIXES`. The 25 source rows rederive as 25 distinct
pair/bucket obstructions, and each row contains the exact Cartesian product of
its counted RG3 actuator magnitudes and both signed one-quantum directions.
The source-local receiver preserves exact uint8 zero identity and canonical
archive parse-back. Both candidate active-tube statements remain scoped to the
full six-dimensional quadratic; the per-dimension split is diagnostic only.

The independent pass found three enforcement gaps and fixed them in the merge
commit: missing/residual key-set equality and uniqueness were not checked,
sign-by-magnitude probe completeness was inferred rather than enforced, and an
empty active-tube batch reached reductions instead of failing closed. The
runner now also refuses malformed/nonfinite scorer geometry before pricing.

# Clean passes

Pass 1 rederived typed archive and obstruction custody. Pass 2 traced the
source-local receiver and scorer call path. Pass 3 attacked duplicate keys,
incomplete probe grids, empty batches, and nonfinite scorer outputs after the
fixes. All three passes were clean after the fix reset.

### src/tac/optimization/ddm_rg4_g3_blocks_and_active_tube.py — CLEAN

### src/tac/optimization/tests/test_ddm_rg4_g3_blocks_and_active_tube.py — CLEAN

### tools/run_ddm_rg4_g3_blocks_and_active_tube.py — CLEAN

# Verification

- `PYTHONWARNINGS=error ... pytest -q
  src/tac/optimization/tests/test_ddm_rg4_g3_blocks_and_active_tube.py`:
  5 passed.
- Actual landed receipt: 25 distinct keys; every row has two signs for every
  counted magnitude.
- Ruff check/format, `py_compile`, and `git diff --check`: clean.

# Authority boundary

No scorer run, training, paid dispatch, exact contest evaluation, archive
promotion, reseal, FIRE decision, or frontier-pointer mutation occurred.
MAIN must review the merge commit and this credential before landing.
