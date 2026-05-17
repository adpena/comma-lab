# Master-Gradient Optimal-Plan Loader False-Authority Review

Date: 2026-05-17
Owner: codex
Lane: lane_q2_q3_batched_catalog_319_gate_plus_autopilot_reweight_v2_20260517
research_only: false
score_claim: false
promotion_eligible: false

## Finding

The green Q3 autopilot reweight WIP had a false-authority hole:
`tac.master_gradient_consumers.load_optimal_plan_for_archive()` selected the
newest `optimal_plan_<sha[:12]>_*.json` sidecar by filename prefix and returned
the payload without checking that the payload itself belonged to the requested
archive or that it remained a non-authoritative prediction.

That would let a malformed or misfiled sidecar replace the cathedral
autopilot's `predicted_delta` for the wrong archive. The failure would be
silent and rank-affecting, so it belongs in the same custody class as Catalog
#319's Venn reward proof gate.

## Fix

`load_optimal_plan_for_archive()` now skips candidate sidecars unless all of
these are true:

- `archive_sha256` exactly matches the requested full archive SHA-256.
- `consumer_id == "per_pair_optimal_treatment_plan_via_lagrangian_dual"`.
- `catalog_consumer_id == 15`.
- `evidence_grade == "predicted"`.
- `score_claim`, `promotion_eligible`, and
  `ready_for_exact_eval_dispatch` are all explicitly `false`.

The optimal-plan writer now emits those custody fields directly, so the loader
does not need to infer authority from filename shape or default JSON tags.

## Evidence

Focused WIP suite:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_autopilot_reweight_v2_lagrangian_derived.py \
  src/tac/tests/test_master_gradient_consumers.py \
  src/tac/tests/test_master_gradient_consumers_rashomon.py \
  src/tac/tests/test_per_pair_optimal_treatment_plan.py \
  src/tac/tests/test_check_319_venn_reweight_requires_deliverability_proof.py \
  src/tac/tests/test_session_20260517_cli_flag_additions.py \
  src/tac/tests/test_sensitivity_map_wyner_ziv_wire_in.py \
  src/tac/tests/test_wyner_ziv_deliverability_prober.py \
  src/tac/tests/test_wyner_ziv_deliverability_proof_builder.py \
  -q
```

Result: `236 passed in 9.15s`.

New adversarial cases:

- Filename prefix collision/misfiled payload with mismatched `archive_sha256`
  returns `None`.
- Authority-bearing payloads with `score_claim=true` or
  `evidence_grade="contest-CUDA"` return `None`.
- Valid newest matching predicted payload still wins.

This is a planning-ranker custody fix only. It creates no score claim, no
dispatch claim, and no promotion claim.
