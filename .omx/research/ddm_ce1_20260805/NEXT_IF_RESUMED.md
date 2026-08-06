# NEXT IF RESUMED - DDM CE1

1. Re-run the current full scanners before editing:
   - `.venv/bin/python tools/check_subset_default_scope_fields.py --json --write-baseline .omx/research/ddm_ce1_20260805/ce1_subset_scope_inventory_YYYYMMDD.json`
   - `.venv/bin/python tools/check_no_silent_cap_defaults.py --json --write-baseline .omx/research/ddm_ce1_20260805/ce1_cap_inventory_YYYYMMDD.json`

2. Do not edit `tools/run_taskspace_r10_feature_texture_relay.py` until its untracked ownership is resolved. If released, add `pair_selection` with `n`, `population`, `selection_mode=video_order_prefix`, pair indices, and `axis_bias_caveat`.

3. Next cap cures should target owner-approved/scorer-lane files first:
   - `experiments/ddm_q3x_q3_convergence_measurement.py`
   - `experiments/ddm_et1_block16_realization.py`
   - `experiments/ddm_sq1_pose_null_constrained_paint.py`
   Each should emit `CapStopReceipt` and must not launch or reinterpret prior capped rows without owner/scorer authorization.

4. Next vacuity cures should target:
   - `tools/check_gate*.py` zero-examined OK paths
   - prose-only `_finish(ok_detail=...)` emitters
   - runtime adapter / launcher bare-Python surfaces
   Add explicit examined/declared/population denominators and fail closed on zero examined unless explicitly acknowledged.

5. Keep frontier statement unchanged unless an exact evaluator row moves it: `S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]`; contest pointer borrowed/unmoved.

