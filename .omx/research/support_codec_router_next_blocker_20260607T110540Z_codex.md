# SupportCodecRouter Next Blocker

schema: tac.support_codec_router_next_blocker.v1
generated_at_utc: 2026-06-07T11:05:40Z
artifact_dir: /Volumes/VertigoDataTier/pact/experiments/results/support_codec_router_20260607T110540Z_codex
source_path_action_artifact: /Volumes/VertigoDataTier/pact/experiments/results/path_action_producer_v3_20260607T105331Z_codex
score_claim: false
promotion_eligible: false
ready_for_exact_eval_dispatch: false

## Selected Codec

- action_id: `path_tube:p0:c0:a0f24babfd13cfe4`
- support_sha256: `a0f24babfd13cfe4c086cb05dbbbf1a58ed11c01c97338b13898f7396b665298`
- selected_support_encoding: `rle`
- selected_total_cost_bytes: `2378`
- candidate_queue_rows: `1`
- selected_action_effect_rows: `1`
- ActionEffect validation: `passed`

## Alternatives

- `rle`: `2378` bytes, selected
- `tile_set_16x16_bitmap`: `7691` bytes, dominated
- `tile_set_8x8_bitmap`: `10809` bytes, dominated
- `path_tube`: `18230` bytes, dominated
- `bitmap_bitset`: `24589` bytes, dominated
- `coordinate_list_u16`: `176541` bytes, dominated
- `semantic_grammar`: unavailable
- `latent_derived`: unavailable

## Next Blockers

- `path_action_parseback_survival_missing`
- `path_action_inflate_survival_missing`
- `inverse_scorer_receiver_surface_motion_missing`
- `support_codec_unavailable` for semantic and latent-derived support codecs

The router has converted the path-tube falsification into reusable planner
signal: downstream menu/search sees only the selected `rle` support row for this
action/support hash, while dominated alternatives stay in the comparison report.
