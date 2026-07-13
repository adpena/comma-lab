# DAG FEED — #484 PRE-SE multi-source composition

Date: 2026-07-13  
Feed id: `FEED-484-pre-se-multi-source-retained-mass-kill`  
Research only: `true`  
Canonical append: `DEFERRED-SIBLING-HELD-DIRTY-SHARED-DAG`

## Proposed append

```text
NODE FEED-484-pre-se-multi-source-retained-mass-kill
  authority = [macOS-CPU advisory; NumPy-fp64 convex fit; CPU-Torch nonlinear]
  receipt = experiments/results/pre_se_multi_source_reopen_20260713/receipt.json
  receipt_sha256 = a092dd5cf791ab060a4300ac3b9c1d49a196ddd83b158121b70fae6a130dc643
  verdict = RETAINED-MASS-FAMILY-KILL
  verdict_scope = FAMILY x CHEAP-PRE-SE-LOCALIZATION x SINGLE-AND-MULTI-SOURCE x CONVEX-AND-NONLINEAR-RUNGS x FIXED-n600-REPLAY x 4.70%-AREA
  tileability_modulo_cheap_globals = MEASURED_CONFIRMED
  retained_mass_convex = MEASURED 0.11225888402810756
  retained_mass_nonlinear = MEASURED 0.31562159104967574
  retained_mass_bar = SOURCE 0.47
  same_area_oracle = MEASURED_INHERITED_AND_REPRODUCED 0.5278150212253758
  route = #455 whole-teacher DISTILLED student
  pointer_delta = NONE

EDGE protected-pre-se-n600-receipt -> FEED-484-pre-se-multi-source-retained-mass-kill
  relation = supplies immutable 480 compact targets plus 120 heldout costate hashes
EDGE block2-pre-se + block3-pre-se + shallow-chart -> FEED-484-pre-se-multi-source-retained-mass-kill
  relation = 476-column composition under unchanged convex/nonlinear rungs
EDGE upstream-SE-gates-once -> FEED-484-pre-se-multi-source-retained-mass-kill
  relation = seven unique reductions and 864 gates broadcast across local tiles
EDGE FEED-484-pre-se-multi-source-retained-mass-kill -| #484-cheap-localization-reopen
  relation = retained-mass bar remains binding despite tileability pass
EDGE FEED-484-pre-se-multi-source-retained-mass-kill -> #455-whole-teacher-distilled-student
  relation = surviving family does not require tileability
```

## Reactivation criterion

Do not reopen this frozen PRE-SE family for a new concatenation or another small head. Reactivate only when a feature provider outside shallow+block2+block3 frozen PRE-SE charts adds a concrete source of target-ordering information and is preregistered against the same n600, 4.70%-area, 0.47 retained-mass gate. Whole-teacher distillation #455 is the routed successor rather than a reactivation of this killed family.

## Triality

- DSL: `src/tac/witness_dsl/pre_se_multi_source_reopen_policy_20260713.py`
- Equation: `src/tac/canonical_equations/pre_se_multi_source_reopen_20260713.py`
- Evidence: `experiments/results/pre_se_multi_source_reopen_20260713/receipt.json`

No shared DAG bytes were changed in this landing.
