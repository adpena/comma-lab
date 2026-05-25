# Codex Findings: Materializer Archive-Delta Feedback

- utc: `2026-05-25T06:16:23Z`
- lane_id: `codex_materializer_archive_delta_feedback_20260525`
- scope: inverse-steganalysis water-bucket feedback, materializer archive economics, dry downstream replanning
- authority: planning-only; no score claim, no promotion, no rank/kill authority

## Findings

1. A receiver-correct inverse-scorer materialization can still add archive bytes. Treating that as a generic successful queue row loses the key economic signal and lets the next water-bucket pass refill the same bad bucket.
2. The first feedback pass blocked only the matched pose bucket, but the second direct materializer run exposed an identity gap: replanned `byte_shaving_unit_*` atom IDs did not intersect the original `inverse_surface_*` / `inverse_action_*` materializer IDs.
3. The byte-shaving campaign planner treated a saturated action surface with zero selected cells as an exception. That is not a valid autonomous terminal state; it must be a typed dry no-op plan.
4. The materializer campaign runner feedback command auto-discovered family-agnostic materializer observations, but not direct materializer archive-delta manifests. That left the new feedback flag manual and easy to orphan.

## Landed Guards

- Materializer chain/direct manifests now normalize to `materializer_chain_archive_delta` observations with `realized_saved_bytes`, rate outcome, source/candidate archive bytes and SHA fields, parity flags, blockers, and false-authority boundaries.
- Rate-nonpositive materializer observations set observed gains to zero and block only matching buckets unless `quality_spend_allowed=true`.
- Byte-shaving ranked-unit provenance now preserves inverse-action/source atom identity, so materializer feedback survives replanning identity rewrites.
- Empty inverse-action water buckets now become dry byte-shaving campaign plans instead of exceptions.
- The materializer runner now auto-discovers direct archive-delta manifests and wires them into feedback replans via `--materializer-chain-manifest`.

## Proof Artifacts

- `.omx/research/high_level_byte_shaving_runner_smoke_20260524T050723Z/campaign3_ias1aware_repair_feedback/inverse_steganalysis_action_functional.feedback.json`
- `.omx/research/high_level_byte_shaving_runner_smoke_20260524T050723Z/campaign3_ias1aware_repair_feedback/byte_shaving_campaign_plan.feedback.json`
- `.omx/research/high_level_byte_shaving_runner_smoke_20260524T050723Z/campaign3_ias1aware_repair_feedback/followup_materializer_queue_exec2/materializer_outputs/materializer_work_materializer_work_queue_required_inverse_scorer_cell_candidate_v1_scorer_inverse_surface_cell_materialize_inverse_scorer_cell_candidate_inverse_scorer_cell_candidate_adapter.json`
- `.omx/research/high_level_byte_shaving_runner_smoke_20260524T050723Z/campaign3_ias1aware_repair_feedback/followup_materializer_queue_exec2/inverse_steganalysis_action_functional.feedback.json`
- `.omx/research/high_level_byte_shaving_runner_smoke_20260524T050723Z/campaign3_ias1aware_repair_feedback/followup_materializer_queue_exec2/byte_shaving_campaign_plan.feedback.json`

## Current Verdict

The two tested high-level smoke inverse cells are structurally materializable but rate-negative on this archive packaging path:

- chain repair pose bucket: `realized_saved_bytes=-1764`
- direct follow-up rate-null-space bucket: `realized_saved_bytes=-1805`

The feedback loop now preserves both as typed negative planning signal. The exec2 action surface has `cell_count=1`, `materializer_archive_delta_blocked_cell_count=1`, and `selected_count=0`; the downstream byte-shaving campaign plan has `units=0`, `prefixes=0`, and `combinations=0`.

## Next Work

- Widen the inverse-surface candidate generator beyond these two smoke cells; this local pocket is dry, not the global floor.
- Add the same archive-delta feedback contract to additional materializers as they land for HNeRV, HNeRV boltons, NeRV-family variants, and non-NeRV substrates.
- Move from leaf-cell materialization toward compiled multi-operation receiver transforms once the operation-set compiler emits byte-closed candidates.
