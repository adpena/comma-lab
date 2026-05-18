---
date_utc: 2026-05-18T00:12:00Z
lane_id: lane_catalog324_predicted_band_recipe_closure_20260518
repair_class: predicted_band_false_authority_closure
horizon_class: frontier_protecting
score_claim: false
promotion_eligible: false
evidence_grade: diagnostic
axis: "[diagnostic; operator-recipe predicted-band provenance]"
---

# Catalog #324 Predicted-Band Recipe Closure

## Finding

The first Catalog #324 audit artifact at
`.omx/state/predicted_band_audit_20260518T000608Z.json` reported:

- in-scope recipe bands: 19
- pass: 9
- fail: 10

The failures were all operator-authorize recipes with `predicted_band` but no
`predicted_band_validation_status`. This preserves the same false-authority
shape as the C6 IBPS 22x miss: a score band can look dispatch-ready even when
the Tier-C density basis was not measured on the post-training artifact.

## Repair

Added `predicted_band_validation_status: pending_post_training` and explicit
`predicted_band_reactivation_criteria` to the nine still-dispatchable planning
recipes:

- `.omx/operator_authorize_recipes/substrate_a1_plus_lapose_modal_a100_dispatch.yaml`
- `.omx/operator_authorize_recipes/substrate_a1_plus_wavelet_residual_modal_t4_dispatch.yaml`
- `.omx/operator_authorize_recipes/substrate_sabor_boundary_only_renderer_modal_t4_dispatch.yaml`
- `.omx/operator_authorize_recipes/substrate_sar_coherent_pose_pairs_modal_t4_dispatch.yaml`
- `.omx/operator_authorize_recipes/substrate_siren_modal_a100_dispatch.yaml`
- `.omx/operator_authorize_recipes/substrate_wyner_ziv_cooperative_receiver_modal_a100_dispatch.yaml`
- `.omx/operator_authorize_recipes/substrate_z3_balle_hyperprior_bolton_modal_t4_dispatch.yaml`
- `.omx/operator_authorize_recipes/substrate_z3_g1_scorer_softmax_hyperprior_gating_modal_t4_dispatch.yaml`
- `.omx/operator_authorize_recipes/substrate_z4_cooperative_receiver_loss_modal_t4_dispatch.yaml`

For the empirically falsified C6 recipe
`.omx/operator_authorize_recipes/substrate_c6_e4_mdl_ibps_modal_t4_dispatch.yaml`,
the repair is stricter:

- keep the historical `[0.113, 0.163]` band visible for debrief/debugging;
- add `predicted_band_validation_status: pending_post_training`;
- add reactivation criteria requiring post-training Tier-C density and a revised
  score band or explicit operator waiver;
- set `dispatch_enabled: false`.

This is not a C6 kill. It is a Catalog #324 deauthorization until the random-init
Tier-C basis is replaced with post-training evidence from the smoke archive.

## Verification

```bash
.venv/bin/python -m pytest src/tac/tests/test_check_324_predicted_band_post_training_validation.py -q
# 43 passed in 0.59s

.venv/bin/python tools/audit_predicted_band_provenance.py --strict
# Recipes scanned: 72
# In-scope: 19
# PASS: 19
# FAIL: 0
```

No contest score claim, CPU claim, CUDA claim, promotion claim, or readiness
claim is made. The repair only removes predicted-band dispatch authority until
post-training Tier-C evidence exists.
