# PathActionProducer Next Blocker

schema: tac.path_action_producer_next_blocker.v1
generated_at_utc: 2026-06-07T10:53:31Z
artifact_dir: /Volumes/VertigoDataTier/pact/experiments/results/path_action_producer_v3_20260607T105331Z_codex
score_claim: false
promotion_eligible: false
ready_for_exact_eval_dispatch: false

## Evidence

- Real hard-region input: `/Volumes/VertigoDataTier/pact/experiments/results/hinerv_target_region_birth_real_smoke_20260606/hinerv_witness_readiness_short_smoke_v41_lateall_wall_normal_forced_region_20260607T094500Z/hi_nerv_mlx_training/hi_nerv_hard_region_miner_inputs.npz`
- Emitted ActionEffect rows: `1`
- Emitted candidate queue rows: `1`
- Action kind: `frame1_seg_margin_frontier_path`
- ActionEffect validation: `passed`
- Focused tests: `62 passed`
- Ruff: `passed`

## Byte Signal

- support_cardinality: `44132`
- bitmap_bytes: `24576`
- coordinate_list_bytes: `176528`
- path_tube_bytes: `18230`
- rle_bytes: `1773`
- best_encoding: `rle`

The live region proves path-tube support identity and ActionEffect byte pricing,
but it also says this particular component should route to RLE before path-tube
inside a byte compiler.

## Next Blockers

- `path_action_trajectory_receiver_proof_missing`
- `path_action_parseback_survival_missing`
- `path_action_inflate_survival_missing`
- `path_action_support_without_wrong_to_target_is_not_birth`

PathActionProducer rows are candidate material only. They do not clear launch,
score, rank, promotion, exact replay, or queue admission until the same action
has receiver-surface movement plus parse-back and inflate survival.
