# L5 v2 timing-smoke allowed semantics hardening

- schema: `l5_v2_timing_smoke_allowed_semantics_hardening_v1`
- created_at_utc: `2026-05-16T22:25:03Z`
- surface: `src/tac/optimization/l5_staircase_v2.py`
- score_claim: `false`
- promotion_eligible: `false`
- ready_for_exact_eval_dispatch: `false`

## Finding

`first_anchor_timing_smoke_allowed` was overloaded as if it meant the timing
smoke artifact already existed. That created a deadlock-prone operator signal:
the field stayed `false` until after `.omx/state/tt5l_first_anchor_timing_smoke.json`
was valid, even though the next action at that point is precisely to materialize
the timing-smoke custody artifact.

This was the third finding from the L5 v2 adversarial review. It is lower
blast-radius than the architecture-lock split-authority bug, but still matters:
operator briefing, Cathedral, or a dispatcher should be able to distinguish
"the timing-smoke run is now allowed" from "the timing-smoke artifact is already
valid."

## Fix

`first_anchor_timing_smoke_allowed` now means the prerequisite set for running
or materializing the first-anchor timing smoke is satisfied:

- Dykstra score-axis sanity valid;
- TT5L move-level feasibility valid;
- side-info gate evidence valid;
- paired side-info effect curve valid;
- C1/Z5/TT5L probe gate valid;
- paired CPU/CUDA axis plan valid.

The separate `first_anchor_timing_smoke_artifact_valid` field remains the
custody flag for whether the timing-smoke artifact already exists and passes
validation. Architecture lock still requires that custody flag plus the anchor
pair; this change does not grant score, rank, promotion, or dispatch authority.

`tools/all_lanes_preflight.py` now validates the allowed flag against the
prerequisite set only, not against the artifact it is supposed to enable.

## Regression

`src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_tt5l_first_anchor_timing_requires_probe_and_paired_axis_plan`
now asserts `first_anchor_timing_smoke_allowed=true` when all prerequisites are
met but the timing artifact is still missing, while `architecture_lock_allowed`
remains false.

`src/tac/tests/test_all_lanes_operator_briefing_gate.py::test_operator_briefing_dispatch_gate_allows_tt5l_timing_before_artifact`
verifies the operator briefing preflight accepts that exact state and does not
mistake a missing timing artifact for a reason to reject the timing-smoke
materialization step.
