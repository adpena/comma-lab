# HiNeRV/SNeRV Interpreter Burndown - 2026-06-07T15:45Z

## Scope

This slice tightens the evaluator-witness interpreter path without approving long
training. It fixes two concrete blocker classes:

- HiNeRV wall-normal branch receipts now keep one canonical branch per fixed
  action/support: direct teacher, backend fit, selected receiver-surviving
  sidecar. Dominated sidecar grammar rows stay in comparison reports but no
  longer poison the receipt.
- HiNeRV launch-gate archive closure now accepts a later valid same-run
  receiver-consumed target-region action proof with exact archive bytes instead
  of letting stale target-region telemetry rows veto the selected tile payload.

## HiNeRV Evidence

Selected real branch bundle:

`/Volumes/VertigoDataTier/pact/experiments/results/actioneffect_inverse_scorer_20260607Tcodex_tile_brotli_full_branch_v2`

- `summary.json` sha256: `573ae76e817fa59092e2ebd963f501001f3ecc782c5ba367c0586fa0e0c0cfca`
- `wall_normal_branch_receipt.json` sha256: `bb15b314d1606d6e8f8185145d26d810aaceb4838d0b1f654289860e29d65d48`
- `wall_normal_branch_action_effect_rows.jsonl` sha256: `40fdc71bca68bdf9c3508e964d51625e8fc19d77c56bfc41245e1b2edf698d31`
- `wall_normal_branch_lowering_race.json` sha256: `5c309423242ee9ce3740e340b1788d6941c957779c750409ffc502e7182276fc`
- `pr110_k16_baseline_validation.json` sha256: `8c6bbdd82dfa1bb9a794b138f40ccab5d600a79eaa158f89b6c3d3390cb86723`

Real smoke-root gate after the stale-telemetry fix:

`/Volumes/VertigoDataTier/pact/experiments/results/hinerv_witness_readiness_short_smoke_current_main_20260607Tcodex_v3_export_wall_normal_support/hinerv_long_run_gate_after_tile_branch_gatefix.json`

Remaining blockers are now only:

- `real_video_birth_receipt_archive_unclosed`
- `source_qualified_metrics_missing`

The retired false blockers were stale target-region action hash/support-encoding
and meta-materialization blockers. Long HiNeRV training is still not approved:
the sidecar survives as an interpreter program, while backend realization remains
the hard crux.

## SNeRV Evidence

Materialized one real source-bound SNeRV triplet from the contest video for the
official source-forward lane:

`/Volumes/VertigoDataTier/pact/artifacts/snerv_source_triplets/source_frame_triplets_pairs_0000_20260607T154146Z.npy`

Manifest:

`/Volumes/VertigoDataTier/pact/artifacts/snerv_source_triplets/source_frame_triplets_pairs_0000_20260607T154146Z.npy.manifest.json`

- manifest sha256: `b695d4c1b10876663e8d9b90b76a2438b9a613c36d07d5b7a45c92b99e004da1`
- npy sha256: `fc5520b4aa59b6ea694ef03fc4348a22236785f3d437030f4171cb912dac753b`
- shape: `[1, 3, 3, 874, 1164]`
- triplet order: current frame `2p+1`, previous frame `2p`, next frame `2p+2`
- source video sha256: `2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9`

This does not clear MFU/HFR/TUB source-forward parity. It removes fixture-origin
input ambiguity for the next source-forward proof.

## Verification

- `uv run pytest src/tac/tests/test_nerv_long_run_launch_gate.py::test_target_region_archive_evidence_rejects_stale_tile_support_encoding src/tac/tests/test_nerv_long_run_launch_gate.py::test_target_region_archive_evidence_uses_valid_receiver_tile_over_stale_rows src/tac/tests/test_inverse_scorer_actions.py::test_wall_normal_fixed_scope_keeps_selected_receiver_bound_sidecar src/tac/tests/test_inverse_scorer_actions.py::test_generate_inverse_evaluate_actions_cli_scopes_wall_normal_receipt_to_fixed_action -q`
- `uv run pytest src/tac/tests/test_materialize_snerv_source_triplets.py -q`
- `uv run pytest src/tac/tests/test_hinerv_target_region_action_comparison.py::test_hinerv_action_comparison_decomposes_receiver_survived_sidecar src/tac/substrates/hi_nerv/tests/test_hi_nerv_roundtrip.py::test_target_region_action_payload_uses_receiver_decodable_compression src/tac/substrates/hi_nerv/tests/test_hi_nerv_roundtrip.py::test_target_region_action_payload_uses_tile_brotli_when_support_is_canonical src/tac/substrates/hi_nerv/tests/test_hi_nerv_roundtrip.py::test_target_region_action_payload_does_not_mislabel_noncanonical_support_as_tile src/tac/substrates/hi_nerv/tests/test_hi_nerv_roundtrip.py::test_target_region_action_payload_uses_split_brotli_when_streams_win -q`
- `uv run ruff check src/tac/analysis/nerv_long_run_launch_gate.py src/tac/tests/test_nerv_long_run_launch_gate.py tools/generate_inverse_evaluate_actions.py src/tac/tests/test_inverse_scorer_actions.py src/tac/analysis/hinerv_target_region_action_comparison.py src/tac/substrates/hi_nerv/target_region_actions.py`
- `uv run ruff check tools/materialize_snerv_source_triplets.py src/tac/tests/test_materialize_snerv_source_triplets.py`

