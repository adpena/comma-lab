# Z7-Mamba-2 Queue Visibility And Axis Fix - 2026-05-18

## Status

Z7-Mamba-2 already had WIP surfaces:

```text
recipe: .omx/operator_authorize_recipes/substrate_time_traveler_l5_z7_mamba2_modal_a100_dispatch.yaml
trainer scaffold: experiments/train_substrate_time_traveler_l5_z7_mamba2.py
canonical helper: src/tac/optimization/mamba2_predictor.py
tests: src/tac/tests/test_z7_mamba2_scaffold.py
design memo: .omx/research/z7_mamba2_substrate_design_memo_20260518.md
lane: lane_top5_2_z7_mamba2_scaffold_design_20260518
```

But it was not in `CANONICAL_CANDIDATES`, and direct assessment fell through
to a wrong `_modal_t4_` recipe name. That meant the queue hid an active WIP
substrate and reported `RECIPE_MISSING`/`LANE_REGISTRY_NOT_REGISTERED` if
queried by substrate id.

## Patch

`tools/asymptotic_pursuit_candidate_readiness_assessment.py` now registers
Z7-Mamba-2 as an explicit canonical candidate and maps it to:

```text
substrate_id=time_traveler_l5_z7_mamba2
recipe=substrate_time_traveler_l5_z7_mamba2_modal_a100_dispatch
trainer=train_substrate_time_traveler_l5_z7_mamba2.py
alias=lane_top5_2_z7_mamba2_scaffold_design_20260518
```

During this patch, a second evidence bug was found and fixed: loose design-memo
regex extraction was able to capture a `predicted_delta_s_band` and then label
it as a `predicted_score_band`. The readiness assessment now prefers explicit
recipe `predicted_band` metadata when `predicted_band_kind` is present. That
keeps score bands and delta-S bands on separate axes.

## Queue Result

Refreshed queue artifact:

```text
.omx/state/asymptotic_pursuit/dispatch_queue_20260518T115056Z.json
```

Z7-Mamba-2 row:

```text
substrate_id=time_traveler_l5_z7_mamba2
readiness_verdict=DEFER
predicted_score_band=[0.167, 0.184]
predicted_band_axis=contest-CPU
horizon_class=frontier_pursuit
first blockers:
  CATALOG_240_FULL_MAIN_BLOCKED:RAISES_NotImplementedError
  RECIPE_research_only=true_OPERATOR_NEEDS_TO_FLIP_AFTER_PHASE_2_COUNCIL
  RECIPE_DISPATCH_BLOCKER:z7_mamba2_trainer_full_main_raises_NotImplementedError_per_catalog_240
  RECIPE_DISPATCH_BLOCKER:z7_mamba2_substrate_module_absent_pre_build_per_z7_symposium_revision_6
```

This is not dispatch authority. It is a no-signal-loss queue representation of
the active scaffold and a corrected axis-labelled planning prior.

## Verification

```bash
.venv/bin/python tools/asymptotic_pursuit_candidate_readiness_assessment.py \
  --candidates time_traveler_l5_z7_mamba2 --json
# predicted_score_band=[0.167, 0.184], axis=contest-CPU, verdict=DEFER

.venv/bin/python -m pytest -q \
  src/tac/tests/test_asymptotic_pursuit_candidate_readiness.py \
  src/tac/tests/test_z7_mamba2_scaffold.py
# 101 passed, 1 warning in 16.28s

.venv/bin/python tools/asymptotic_pursuit_dispatch_queue.py --write-artifact --json \
  > /tmp/pact_queue_after_z7_mamba2.json
# wrote .omx/state/asymptotic_pursuit/dispatch_queue_20260518T115056Z.json
```

## Result Classification

- classification: WIP visibility repair plus axis-label evidence bug fix
- provider_dispatch_attempted: false
- lane_claim_opened: false
- score_claim: false
- promotion_eligible: false
- next real Z7-Mamba-2 gate: implement `_full_main` and substrate package only
  after Z7-GRU Wave 2 / operator override and C6 beta-anchor dependencies are
  deliberately resolved.
