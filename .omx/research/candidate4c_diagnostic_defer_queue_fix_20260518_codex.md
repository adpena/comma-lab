# Candidate 4c Diagnostic-DEFER Queue Fix - 2026-05-18

## Status

Candidate 4c's Modal training recipe is intentionally disabled as a paid
contest-CUDA launch surface after the diagnostic-only split:

```text
recipe: .omx/operator_authorize_recipes/substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch.yaml
dispatch_enabled=false
smoke_validation_contract=training_artifact_v1
dispatch_blocker=candidate4c_modal_training_recipe_is_diagnostic_only_exact_cuda_handoff_required
```

The queue previously classified this as `NEEDS_FIX`, which made the active
frontier loop keep returning to a recipe that should not be fixed by flipping
`dispatch_enabled=true`. That was a false-actionability bug. The 600-pair
zero-epoch full/identity pair has already been exact-evaluated on both axes,
and the trained Candidate 4c path still needs a separate archive-producing
training surface before any new exact-eval handoff.

## Patch

`tools/asymptotic_pursuit_candidate_readiness_assessment.py` now treats the
specific pair:

```text
RECIPE_dispatch_enabled=false
RECIPE_DISPATCH_BLOCKER:candidate4c_modal_training_recipe_is_diagnostic_only_exact_cuda_handoff_required
```

as `DEFER`, not `NEEDS_FIX`. Generic recipe flag blockers still classify as
`NEEDS_FIX`; this is deliberately scoped to Candidate 4c's diagnostic exact-eval
handoff contract.

Regression coverage:

```text
src/tac/tests/test_asymptotic_pursuit_candidate_readiness.py
```

## Queue Result

Refreshed queue artifact:

```text
.omx/state/asymptotic_pursuit/dispatch_queue_20260518T114641Z.json
```

Key queue fields:

```text
ready_for_paid_dispatch_count=0
immediately_runnable_paid_dispatch_count=0
candidate4c.readiness_verdict=DEFER
candidate4c.ready_for_paid_dispatch=false
top_1_substrate=time_traveler_l5_z7_lstm_predictive_coding
top_1_readiness_verdict=DEFER
```

This prevents a no-op loop around Candidate 4c paid-training enablement and
surfaces the next real queue blocker: Z7's missing trainer/substrate package
plus council state.

## Verification

```bash
.venv/bin/python -m py_compile \
  tools/asymptotic_pursuit_candidate_readiness_assessment.py \
  src/tac/tests/test_asymptotic_pursuit_candidate_readiness.py
# rc=0

.venv/bin/python -m pytest -q \
  src/tac/tests/test_asymptotic_pursuit_candidate_readiness.py
# 64 passed in 14.56s

.venv/bin/python tools/asymptotic_pursuit_dispatch_queue.py --write-artifact --json \
  > /tmp/pact_queue_after_candidate4c_defer.json
# wrote .omx/state/asymptotic_pursuit/dispatch_queue_20260518T114641Z.json
```

## Result Classification

- classification: queue false-actionability bug fixed
- provider_dispatch_attempted: false
- lane_claim_opened: false
- score_claim: false
- promotion_eligible: false
- reactivation criterion: build or harvest a trained Candidate 4c 600-pair
  archive/runtime packet, then run a fresh paired exact-eval handoff; do not
  flip the diagnostic Modal recipe into contest-CUDA dispatch authority.
