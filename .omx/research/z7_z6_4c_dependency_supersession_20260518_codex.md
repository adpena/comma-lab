# Z7 Z6 4c Dependency Supersession - 2026-05-18

## Scope

This ledger records a readiness-surface repair, not a score claim. Z6 Candidate
4c now has paired `[contest-CUDA]` and `[contest-CPU]` zero-epoch full-vs-identity
exact-eval evidence in
`.omx/research/z6_candidate4c_identity_archive_pair_disambiguator_20260518_codex.json`.
That closes only the Z7 sequencing dependency
`z7_dispatch_requires_z6_wave2_candidate_4c_paired_exact_eval_outcome`.

The Z7 recipe remains non-dispatchable. This supersession must not be treated as
operator spend authority, promotion evidence, or a claim that Z7 is implemented.

## Code Artifact

Readiness assessment now computes recipe-blocker supersessions from the exact
Z6 Candidate 4c disambiguator artifact:

- `tools/asymptotic_pursuit_candidate_readiness_assessment.py`
- `tools/asymptotic_pursuit_dispatch_queue.py`
- `src/tac/tests/test_asymptotic_pursuit_candidate_readiness.py`

The queue row exposes `dispatch_blocker_supersessions` separately from
`blocking_issues`, so stale predecessor dependencies do not hide the remaining
Z7 work.

## Current Queue Evidence

Generated artifact:

```text
.omx/state/asymptotic_pursuit/dispatch_queue_20260518T112213Z.json
```

Targeted jq review:

```text
substrate_id=time_traveler_l5_z7_lstm_predictive_coding
readiness_verdict=DEFER
dispatch_blocker_supersessions=[
  z7_dispatch_requires_z6_wave2_candidate_4c_paired_exact_eval_outcome
]
```

Remaining blockers:

```text
TRAINER_MISSING
CATALOG_240_FULL_MAIN_BLOCKED:TRAINER_FILE_MISSING
CATALOG_315_COUNCIL_PROCEED_WITH_REVISIONS_NEEDS_ITERATION_TO_OPTIMAL_FORM
RECIPE_research_only=true_OPERATOR_NEEDS_TO_FLIP_AFTER_PHASE_2_COUNCIL
RECIPE_DISPATCH_BLOCKER:z7_trainer_module_absent_verified_by_symposium_pv2
RECIPE_DISPATCH_BLOCKER:z7_substrate_package_absent_verified_by_symposium_pv2
RECIPE_DISPATCH_BLOCKER:z7_dispatch_requires_wave_n_plus_1_council_after_z6_4c_outcome
RECIPE_DISPATCH_BLOCKER:z7_beta_ib_parameter_requires_c6_ibps_phase2_empirical_beta_anchor
RECIPE_DISPATCH_BLOCKER:z7_wave2_probe_requires_paired_exact_eval_json_from_probe_z7_temporal_coherence_vs_static_capacity_disambiguator
RECIPE_DISPATCH_BLOCKER:z7_requires_same_archive_bytes_identity_disambiguator_before_full_dispatch
```

## Verification

```bash
.venv/bin/python -m py_compile \
  tools/asymptotic_pursuit_candidate_readiness_assessment.py \
  tools/asymptotic_pursuit_dispatch_queue.py \
  src/tac/tests/test_asymptotic_pursuit_candidate_readiness.py
# rc=0

.venv/bin/python -m pytest -q \
  src/tac/tests/test_asymptotic_pursuit_candidate_readiness.py
# 62 passed in 13.17s

.venv/bin/python tools/asymptotic_pursuit_dispatch_queue.py --write-artifact --json \
  > /tmp/pact_dispatch_queue_after_z6_supersession.json
# wrote .omx/state/asymptotic_pursuit/dispatch_queue_20260518T112213Z.json
```

## Result Classification

- classification: readiness-surface repair
- provider_dispatch_attempted: false
- lane_claim_opened: false
- score_claim: false
- promotion_eligible: false
- evidence_axis: planning/readiness artifact with inline `[contest-CUDA]` and
  `[contest-CPU]` predecessor-evidence checks only

Next score-moving Z7 action is still substantive: build the trainer/substrate
or land a smaller byte-closed probe that can satisfy the remaining same-archive
identity disambiguator and Wave N+1 council gates.
