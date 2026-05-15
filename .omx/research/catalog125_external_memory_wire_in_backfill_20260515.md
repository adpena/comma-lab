# Catalog #125 external-memory wire-in backfill - 2026-05-15

## Scope

Strict preflight surfaced 25 post-cutover Claude landing memos under
`~/.claude/projects/-Users-adpena-Projects-pact/memory/` that were missing one
or more unified-Lagrangian wire-in declarations required by Catalog #125.

This backfill did not change any score, archive, lane status, dispatch claim,
or implementation. It appended conservative per-hook `N/A - <rationale>` lines
to the historical memos so the strict gate can distinguish "historical memo did
not create this hook" from "silent orphan signal".

## Verification

Command:

```bash
PYTHONPATH=src .venv/bin/python - <<'PY'
from tac.preflight import check_subagent_landing_has_solver_wire_in
check_subagent_landing_has_solver_wire_in(strict=True, verbose=True)
PY
```

Result:

```text
[subagent-landing-wire-in] OK (309 post-cutover memo(s) scanned, 27 research-only opt-out, 0 missing)
```

## Backfilled files

- `feedback_across_class_campaign_push_8_parallel_fanout_landed_20260514.md`
- `feedback_autopilot_tier_c_integration_catalog_227_landed_20260514.md`
- `feedback_c6_mdl_ibps_landed_20260514.md`
- `feedback_c6_next_wave_landed_20260514.md`
- `feedback_catalog_241_backfill_29_trainers_landed_20260515.md`
- `feedback_d1_segnet_margin_polytope_landed_20260514.md`
- `feedback_d8_l1_l2_promotion_canonical_strict_preflight_landed_20260514.md`
- `feedback_deep_math_geometry_manifolds_research_landed_20260514.md`
- `feedback_dp1_phase_2_landed_20260514.md`
- `feedback_grand_council_ad_hoc_dispatch_unblock_landed_20260514.md`
- `feedback_grand_council_maximize_value_landed_20260514.md`
- `feedback_grand_council_omnibus_design_decisions_landed_20260514.md`
- `feedback_hdm8_film_grain_selector_dispatch_landed_20260514.md`
- `feedback_ibps1_canonical_surface_landed_20260514.md`
- `feedback_ibps1_parser_wave_p0_landed_20260514.md`
- `feedback_mdl_ablation_tier_c_ibps1_landed_20260514.md`
- `feedback_mdl_density_gate_and_autopilot_ranker_landed_20260514.md`
- `feedback_meta_layer_adversarial_review_round_1_2_landed_20260515.md`
- `feedback_oss_v0_2_0_rc1_release_prep_audit_landed_20260514.md`
- `feedback_r2_medium_fix_wave_selfcomp_mackay_landed_20260515.md`
- `feedback_recovery_2_c6_finish_and_modal_harvest_landed_20260514.md`
- `feedback_tier_c_pr106_dp1_extension_landed_20260514.md`
- `feedback_yucr_substrate_landed_20260514.md`
- `feedback_z1_mdl_ablation_landed_20260514.md`
- `feedback_z4_atick_redlich_minimum_viable_landed_20260515.md`
