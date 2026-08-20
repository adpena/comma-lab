# 01 — Codebase-grounded findings

Pinned ref: `69b4c523d2696978cae4423d95488d15d851e8cd`.

| ID | File/function | Current behavior | Exact math term | Exact actuator/telemetry | Missing proof | Patch target |
|---|---|---|---|---|---|---|
| F-001 | `src/tac/training/long_training_canonical.py::_export_live_ema_archive_selection` | Exports live + EMA archives, checks SHA, composes `proxy_score` from local components + bytes, then selects min. | `proxy = local seg/pose components + bytes * 25/N` | `score_components`, `proxy_score`, `archive_bytes` | No built-packet parse-back or inflate/replay components participate in selection. | Add optional `archive_replay_components()` hook and `archive_selection_replay_required`. |
| F-002 | `src/tac/training/long_training_canonical.py::_archive_selection_health_sort_key` | Sorts by target coverage/min-ratio/occupied classes before proxy score. | SegNet collapse guard, not exact score. | `selection_health_segnet_direct_live_candidate_target_class_min_ratio` | Does not directly sort by score-weighted unsolved argmax mass even though `loss.py` emits it. | Add `score_weighted_total_unsolved_argmax_mass` to health keys and sort key before coverage. |
| F-003 | `src/tac/substrates/_shared/mlx_score_aware/loss.py::_segnet_target_min_ratio_floor_loss_and_metrics` | Computes target region hard support, score-weighted unsolved mass, crossing loss, frontier margin, seed island loss. | `100 * target_fraction * (1-region_ratio)` | `*_score_weighted_unsolved_argmax_mass`, `*_target_region_crossing_loss`, `*_frontier_margin` | Telemetry exists; no guarantee a scoped output-head/late-grid actuator consumes the worst atom. | Add HiNeRV birth actuator consuming worst class rows. |
| F-004 | `src/tac/substrates/_shared/mlx_score_aware/loss.py::_direct_live_posenet_distillation_loss_and_metrics` | Direct-live PoseNet path computes `sqrt(10*raw_mse)` and now exposes score marginal. | `sqrt(10*d_pose)`, `5/sqrt(10*d_pose)` | `pose_direct_live_score_marginal_wrt_raw_mse`, residual L2 mean/max | No trust-region consumes pose marginal during SegNet edits. | Add joint Seg/Pose acceptance rule. |
| F-005 | `.omx/research/snerv_official_mfu_hfr_tub_forward_parity_20260605T125926Z.json` | MFU/HFR primitive fixture parity is proven; TUB primitive has partial proof; full stack false. | Source-forward parity, not score term. | `source_forward_parity_proven`, `full_tub_source_forward_parity_proven` | Trained checkpoint, TUB encoder/decoder, output2 mapping missing. | Close `snerv_official_tub_source_forward_replay.py` and tests. |
| F-006 | `src/tac/substrates/hi_nerv/mlx_renderer.py` | Has output-head bias/contrast bootstrap and scorer-domain bootstrap. | Degenerate renderer prevention. | `output_head_target_bias_init`, `output_head_target_contrast_init`, `fit_scorer_domain_bootstrap_from_targets` | Bootstrap can improve mean/std without hard class-region argmax birth. | Restrict/update late grids + `head_rgb_1` against worst-region direct-live SegNet VJP. |
| F-007 | `src/tac/training/long_training_canonical.py` | Per-axis decomposition is emitted from sampled batches. | `seg`, `pose`, `archive_bytes=0` during training | `PerEpochMetrics.per_axis_decomposition` | Archive bytes are not per-step and parse-back score is absent until export. | Add post-export replay row and value-per-byte ledger. |

## Current positives to preserve

- Pose telemetry at `69b4c523d2696978cae4423d95488d15d851e8cd` has been corrected to score units, not raw MSE.
- Direct-live SegNet target-min-ratio floor already emits score-weighted unsolved mass.
- HiNeRV renderer has QAT stage control and waterfill-to-fake-quant hooks.
- SNeRV receiver primitive proof already blocks scorer imports and carries runtime markers; do not discard it, but do not call it full official parity.

## Current false-authority boundaries

- Current live/EMA archive selection explicitly labels authority as `local_training_proxy_false_authority`.
- Current SNeRV source-forward parity artifact explicitly has `source_forward_parity_proven=false` for full TUB.
- Current TrainingArtifact rejects promotion/readiness flags by construction.
