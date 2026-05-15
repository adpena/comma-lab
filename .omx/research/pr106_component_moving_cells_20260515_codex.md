# PR106 Component-Moving Cell Plan

- schema: `pr106_component_moving_cell_plan_v1`
- label: `pr106_public_r2_latest_component_moving_cells_20260515`
- kind: `latent_sidecar`
- source_score_table_axis: `[provider-CUDA:kaggle advisory score-table]`
- score_claim: `false`
- promotion_eligible: `false`
- ready_for_exact_eval_dispatch: `false`

## Summary

- rows: `600`
- candidates: `113`
- component_improving_cells: `28485`
- net_improving_cells: `28442`
- cell_byte_delta: `2.0` (cli_cell_byte_delta)
- best_net_score_delta_charged: `-0.007331343441071264`

## Source Custody

- manifest_source_archive_path: `inputs/pr106_archive.zip`
- manifest_source_archive_sha256: `cb9976bd33468475aac54a98c3baff996101c144b00e8d7e2c5107c86cda6182`
- manifest_source_archive_member_name: `None`
- manifest_source_zero_bin_sha256: `7f2cc905b7611ae8d7bced72be24e2266b0aa341f90cfeccbb0854fd8fc01eb7`
- local_source_archive_path: `experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip`
- local_source_archive_sha256: `3fefbe5dfdd738179a55ca5c995ff8f63ec2755662d60684706f20d313913f58`
- local_source_member_name: `0.bin`
- local_source_member_sha256: `7f2cc905b7611ae8d7bced72be24e2266b0aa341f90cfeccbb0854fd8fc01eb7`
- source_archive_payload_match: `True`
- source_archive_blockers: `[]`
- source_archive_warnings: `['source_archive_zip_sha256_differs_but_payload_sha256_matches']`

## Top Cells

| rank | row | pair | frame | candidate | component_delta | byte_delta | net_delta |
|---:|---:|---:|---:|---|---:|---:|---:|
| 1 | 545 | 545 |  | `dim=24,delta_q=2` | -0.007332675 | 2.000 | -0.007331343 |
| 2 | 518 | 518 |  | `dim=5,delta_q=-2` | -0.007075064 | 2.000 | -0.007073732 |
| 3 | 513 | 513 |  | `dim=16,delta_q=-2` | -0.006889522 | 2.000 | -0.006888190 |
| 4 | 460 | 460 |  | `dim=17,delta_q=2` | -0.006861240 | 2.000 | -0.006859908 |
| 5 | 108 | 108 |  | `dim=17,delta_q=2` | -0.006760225 | 2.000 | -0.006758893 |
| 6 | 374 | 374 |  | `dim=2,delta_q=2` | -0.006735660 | 2.000 | -0.006734328 |
| 7 | 524 | 524 |  | `dim=7,delta_q=-2` | -0.006663717 | 2.000 | -0.006662386 |
| 8 | 518 | 518 |  | `dim=8,delta_q=2` | -0.006602123 | 2.000 | -0.006600792 |
| 9 | 330 | 330 |  | `dim=27,delta_q=2` | -0.006585263 | 2.000 | -0.006583931 |
| 10 | 518 | 518 |  | `dim=27,delta_q=2` | -0.006559461 | 2.000 | -0.006558130 |
| 11 | 142 | 142 |  | `dim=5,delta_q=-2` | -0.006554358 | 2.000 | -0.006553026 |
| 12 | 263 | 263 |  | `dim=4,delta_q=2` | -0.006529838 | 2.000 | -0.006528506 |
| 13 | 532 | 532 |  | `dim=4,delta_q=2` | -0.006486282 | 2.000 | -0.006484950 |
| 14 | 127 | 127 |  | `dim=8,delta_q=-2` | -0.006407723 | 2.000 | -0.006406391 |
| 15 | 142 | 142 |  | `dim=10,delta_q=2` | -0.006396569 | 2.000 | -0.006395238 |
| 16 | 410 | 410 |  | `dim=23,delta_q=2` | -0.006341413 | 2.000 | -0.006340081 |

## Blockers

- `planning_artifact_only`
- `requires_byte_closed_archive_materialization`
- `requires_lane_dispatch_claim_before_exact_eval`
- `requires_paired_contest_cuda_auth_eval`
- `requires_paired_contest_cpu_auth_eval`
- `requires_adjudicated_component_recompute_before_score_claim`
