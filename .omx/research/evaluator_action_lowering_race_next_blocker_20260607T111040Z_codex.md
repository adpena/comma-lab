# EvaluatorActionLoweringRace Next Blocker

schema: tac.evaluator_action_lowering_race_next_blocker.v1
generated_at_utc: 2026-06-07T11:10:40Z
artifact_dir: /Volumes/VertigoDataTier/pact/experiments/results/evaluator_action_lowering_race_20260607T111040Z_codex
support_codec_artifact_dir: /Volumes/VertigoDataTier/pact/experiments/results/support_codec_router_20260607T110540Z_codex
score_claim: false
promotion_eligible: false
ready_for_exact_eval_dispatch: false

## Verdict

- action_id: `path_tube:p0:c0:a0f24babfd13cfe4`
- support_sha256: `a0f24babfd13cfe4c086cb05dbbbf1a58ed11c01c97338b13898f7396b665298`
- best_lowering: `none`
- first_failing_surface: `PARSEBACK_FAILED`
- backend_status: `BACKEND_REALIZATION_FAILED`
- sidecar_status: `PARSEBACK_FAILED`
- composite_status: `COMPOSITE_NOT_MEASURED`
- semantic_pose_status: `SEMANTIC_PRIMITIVE_MISSING`

## Candidate

- lowering_target: `byte_priced_sidecar`
- support_encoding: `rle`
- support_encoded_bytes: `2378`
- action_payload_bytes: `0`
- metadata_bytes: `0`
- delta_bytes: `2378`
- delta_score_total: `0.0015834125905245236`
- value_per_byte: `-6.658589531221714e-07`
- parseback_survived: `false`
- inflate_survived: `false`

## Next Work

The next useful work is same-action sidecar parse-back/inflate measurement for
the selected `rle` support, or a measured pose-compensated composite. Backend
actuator expansion remains blocked for this action until a backend-realization
ActionEffect exists with receiver-surface motion and exact byte accounting.
