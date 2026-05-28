# Codex Findings - Repair Stage-Disciplined Archive Adapters

UTC: 2026-05-28T06:52Z

## Result

Landed the next repair tranche as executable code rather than planning text:

- added archive-family detection to the repair byte-transform executor;
- added a grammar-aware `FP11` + `FEC6` fixed-Huffman K16 selector mutation adapter;
- added coder-boundary packet-member recompress as a separate transform path;
- kept ZIP repack as the post-container fallback;
- wired the multi-archive runner to emit a blocked exact-dispatch dry-run plan after runtime custody closes;
- preserved false-authority semantics: MLX/local rows remain advisory, no score claim, no promotion, no exact dispatch authority.

## Live Evidence

Live artifact:

`.omx/research/repair_multi_archive_autonomous_stage_disciplined_psv3_fec6_20260528T0648Z/runner_summary.json`

Inputs:

- PSV3 live archive: `experiments/results/pact_nerv_selector_v3_mlx_local_long_2000ep_32pairs_20260528T045801Z/archive.zip`
- FEC6 live archive: `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/archive.zip`

Observed:

- `archive_count=2`
- `typed_response_count=10`
- `ready_experiment_count=10`
- `exact_ready_bridge_candidate_count=10`
- `exact_ready_bridge_runtime_content_tree_custody_proven_count=10`
- `posterior_appended_count_total=10`
- `blocked_exact_dispatch_authorized_candidate_count=0`
- `blocked_exact_dispatch_blocked_candidate_count=1`
- `stop_reason=strictly_better_archive_bound_candidate_exact_axis_blocked`

Stage-disciplined adapter selection:

- FEC6 scorer-repair families selected `fec6_selector_payload_mutation`.
- FEC6 `entropy_boundary_probe` selected `packet_member_entropy_boundary_recompress`.
- PSV3 was detected as `pact_nerv_selector_v3_packet` and stayed on ZIP-safe transforms until a PSV3 parser adapter exists.

## Review Finding Fixed Before Seal

The first live run revealed a real stage-order bug: `entropy_boundary_probe` was allowed to select the FEC6 semantic selector mutation path. The executor now refuses semantic FEC6 mutation for families not explicitly enabled in the scorer-repair selector map, so entropy-boundary probes remain at the coder/container layer.

## Verification

- `ruff` passed on touched code/tests/tools.
- `pytest src/tac/tests/test_repair_family_materializers.py src/tac/tests/test_repair_autonomous_multi_archive_runner.py src/tac/tests/test_repair_campaign_materialization_queue.py -q`
  - `30 passed`
- `tools/review_gate_hook.py`
  - exit code `0`
- recursive adversarial review: final content-hash seal recorded in the
  canonical recursive review ledger; bundle id is `4a3df9fdeea272e8`, with
  three clean senior-engineer passes on the final code/artifact scope.

## Remaining Exact-Axis Blocker

This tranche emits byte-closed, runtime-closed, archive-bound candidates and a blocked exact-dispatch plan. It still intentionally refuses exact dispatch and score authority until a contest CPU/CUDA exact-axis queue row passes the existing exact-readiness and lane-claim gates.
