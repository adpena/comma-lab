# Autopilot Advisory-Negative Integration - 2026-05-08

Scope: integrate the latest A1 and cross-paradigm macOS CPU advisory negatives
into the machine-readable autopilot evidence feed without promoting either as a
contest score.

## Evidence rows added

`reports/cathedral_autopilot_evidence.jsonl` now has fail-closed rows for:

- `phase_a1_score_gradient_pr101_finetune_modal_config_20260508T230020Z`
  - archive bytes: `205879`
  - archive SHA-256:
    `cb9de2b71133929b0c2df00b0e511b9c306939d62438ffb348e947aef719e185`
  - runtime-tree SHA-256:
    `dcdc5b995993fb455989e743bd55eba9987648675a9cb2c5da6e3975c451bc7c`
  - advisory score: `3.7216542390470915`
  - components: `seg=0.02248664`, `pose=0.17846388`, `rate_term=0.1370865`
  - status: `measured_config_retired_macos_cpu_advisory_negative`

- `cross_paradigm_admm_continuous_k_plus_op1_finalizer`
  - archive bytes: `153513`
  - archive SHA-256:
    `7bbba307b1432d8d885e22533fdda9ab5cc87a6025510b2d5098084895284897`
  - runtime-tree SHA-256:
    `4a3fdcb6fbe8aed4263b283da89a96ec6f0dff8dba1efdcd3811fda5228ecdea`
  - advisory score: `0.32844434076752543`
  - components: `seg=0.00188570`, `pose=0.00014180`, `rate_term=0.10221800`
  - status: `measured_config_retired_macos_cpu_advisory_negative`

Both rows explicitly set:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `exact_cuda_auth_eval=false`
- `cuda_eval_worth_testing=false`
- `family_falsified=false`
- `method_family_retired=false`
- `falsification_scope=measured_configuration_only_macos_cpu_advisory`
- `dispatch_blockers` including
  `reactivation_required_before_new_dispatch`
- non-empty `reactivation_criteria`

## Guard fixed

Direct B6 redispatch scanning found an older lossy-coarsening exact-negative
row that already had `reactivation_required_before_new_dispatch` but lacked a
non-empty `reactivation_criteria` list. The row now carries the same scoped
reactivation criteria as the canonical later reviewed row, so retired configs
cannot be silently redispatched.

## Generated operator surfaces

Regenerated from `reports/cathedral_autopilot_evidence.jsonl`:

- `reports/cathedral_autopilot_catalog_updated_20260508.json`
- `reports/cathedral_autopilot_plan_pr103_pr106_to_019_reconciled_20260508.json`
- `reports/cathedral_autopilot_plan_pr103_pr106_to_0155_reconciled_20260508.json`
- `reports/cathedral_meta_lagrangian_ranking_pr103_pr106_to_019_reconciled_20260508.json`
- `reports/cathedral_meta_lagrangian_ranking_pr103_pr106_to_0155_reconciled_20260508.json`

The updated validation queue classifies both new advisory negatives as
`unknown_not_cuda_worth_testing_until_reactivated` with zero
`potential_score_delta_if_validated`.

## Verification

Commands run:

```bash
.venv/bin/python -m pytest tests/test_cathedral_autopilot_advisory_negatives.py -q
.venv/bin/python tools/check_predispatch_retired_config_warning.py --strict
.venv/bin/python tools/cathedral_autopilot.py evidence-update \
  --prior-evidence reports/cathedral_autopilot_evidence.jsonl \
  --output reports/cathedral_autopilot_catalog_updated_20260508.json
.venv/bin/python tools/cathedral_autopilot.py plan \
  --label pr103_pr106_A++_reconciled \
  --d-seg 0.00067082 \
  --d-pose 3.36e-05 \
  --archive-bytes 185578 \
  --target-score 0.190 \
  --output reports/cathedral_autopilot_plan_pr103_pr106_to_019_reconciled_20260508.json \
  --prior-evidence reports/cathedral_autopilot_evidence.jsonl
.venv/bin/python tools/cathedral_autopilot.py plan \
  --label pr103_pr106_A++_reconciled \
  --d-seg 0.00067082 \
  --d-pose 3.36e-05 \
  --archive-bytes 185578 \
  --target-score 0.155 \
  --output reports/cathedral_autopilot_plan_pr103_pr106_to_0155_reconciled_20260508.json \
  --prior-evidence reports/cathedral_autopilot_evidence.jsonl
.venv/bin/python tools/cathedral_autopilot_meta_lagrangian_bridge.py \
  --plan-json reports/cathedral_autopilot_plan_pr103_pr106_to_019_reconciled_20260508.json \
  --output reports/cathedral_meta_lagrangian_ranking_pr103_pr106_to_019_reconciled_20260508.json
.venv/bin/python tools/cathedral_autopilot_meta_lagrangian_bridge.py \
  --plan-json reports/cathedral_autopilot_plan_pr103_pr106_to_0155_reconciled_20260508.json \
  --output reports/cathedral_meta_lagrangian_ranking_pr103_pr106_to_0155_reconciled_20260508.json
```

The expected cathedral warnings remain the pre-existing model-spec caveats for
`tiny_nn_pmf_predictor` and `lossy_int4_quantization`; neither is introduced by
this integration.
