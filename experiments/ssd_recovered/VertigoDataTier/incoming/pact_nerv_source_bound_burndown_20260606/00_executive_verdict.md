# 00 — Executive verdict

Pinned repository ref: `69b4c523d2696978cae4423d95488d15d851e8cd`.

## Current long-run status

| Stack | Verdict | Smallest blocker set |
|---|---|---|
| HiNeRV | **Blocked** | Target-region hard birth remains unproven; min-ratio can be zero while auxiliary terms move. Needs output-head/late-feature-grid birth actuator, parse-back selection, joint Seg/Pose trust region, and value-per-byte ledger. |
| SNeRV | **Blocked** | Official MFU/HFR/TUB full source-forward closure missing. Current proof is primitive/receiver-bound, not full trained official graph bound. |
| Shared L2 trainer | **Blocked for PR95-faithful selection** | Current live-vs-EMA archive selection is false-authority local proxy, not archive parse-back/replay. |

## Exact score math

S = 100*d_seg + sqrt(10*d_pose) + 25*bytes/37,545,489.

Local admission:

Delta S approx 100*Delta d_seg
  + (5/sqrt(10*d_pose))*Delta d_pose
  + (25/37,545,489)*Delta bytes.

Byte price = 6.658589531221714e-7 score/byte.

## Highest-EV code changes

1. `src/tac/training/long_training_canonical.py`
   - Add `archive_selection_replay_required`.
   - Add optional adapter hook `archive_replay_components(archive_path, batch)`.
   - Prefer parse-back/replay proxy over local live proxy when present.
   - Fail closed for long-run configs when hook absent.

2. `src/tac/substrates/hi_nerv/mlx_renderer.py`
   - Add `fit_target_region_birth_from_segnet()` or tighten existing scorer-domain bootstrap to update only `head_rgb_1` and late `feature_grids`.
   - Input: worst class/region rows emitted by existing `target_min_ratio_floor` telemetry.
   - Output: direct change to charged model tensors, no sidecar.

3. `src/tac/substrates/_shared/mlx_score_aware/adapter.py`
   - Add joint Seg/Pose trust region around accepted update/step metrics.
   - Required telemetry: `seg_gain_score_units`, `pose_harm_score_units`, `joint_delta_score_units`, `seg_pose_trust_ratio`.

4. `src/tac/analysis/snerv_official_tub_source_forward_replay.py`
   - Close official TUB source-forward by mapping temporal encoder and output2 decoder weights.
   - Emit `full_tub_source_forward_parity_proven=true` only after official Torch ↔ portable NumPy/MLX scorer-atom parity.

## Stop doing

- Do not accept class coverage as class birth.
- Do not accept live proxy archive selection as source-faithful.
- Do not treat SNeRV receiver primitive decode as official graph parity.
- Do not turn on byte pressure before scorer-space birth unless score-unit value is positive.
