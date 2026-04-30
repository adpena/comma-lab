# Dispatch Queue (HTD agent — High-Throughput-Dispatch)

Maintained by HTD agent. Other agents append to "PARADIGM-AGENTS-PENDING" via `/tmp/dispatch_request_<paradigm>.json`.

## Format

`| priority | lane_id | script | predicted_band | kill_criteria | status | dispatched_at |`

## READY-NOW (have impl + scaffold + tests)

| priority | lane_id | script | predicted_band | est_cost | est_eta | kill_criteria | status |
|---|---|---|---|---|---|---|---|
| P0 | lane_omega_w_v2 | scripts/remote_lane_omega_w_v2_stack.sh | [0.95, 1.02] | $0.50 | ~1h | score > 1.05 | DISPATCHING |
| P0 | lane_pose_delta_pd_v2 | scripts/remote_lane_pd_pose_deltas.sh | [-0.05, -0.02] delta vs Lane G v3 (1.00-1.03 absolute) | $0.50 | ~1h | score > 1.05 | DISPATCHING |
| P0 | lane_j_nwc | scripts/remote_lane_j_nwc_neural_weight_compression.sh | [0.92, 1.02] | $0.50 | ~1.5h | score > 1.05 | DISPATCHING |
| P0 | lane_omega_w_water_filling | scripts/remote_lane_omega_w_water_filling.sh | [0.94, 1.04] | $0.50 | ~1h | score > 1.05 | DISPATCHING |
| P1 | lane_stc_clean_source | scripts/remote_lane_stc_clean_source.sh | [0.85, 1.05] | $1.00 | ~2h | score > 1.10 | QUEUED |
| P1 | lane_sa_segmap_clone | scripts/remote_lane_sa_segmap_clone.sh | [0.85, 1.20] | $1.00 | ~2h | score > 1.30 | QUEUED |
| P1 | lane_si_v2_learnable_threshold | scripts/remote_lane_si_v2_learnable_threshold.sh | [0.95, 1.10] | $0.80 | ~1.5h | score > 1.20 | QUEUED |
| P1 | lane_pa_pose_as_affine | scripts/remote_lane_pa_pose_as_affine.sh | [0.95, 1.10] | $0.50 | ~1h | score > 1.20 | QUEUED |
| P1 | lane_lr_v2_learnable_rank | scripts/remote_lane_lr_v2_learnable_rank.sh | [0.95, 1.10] | $0.80 | ~1.5h | score > 1.20 | QUEUED |
| P1 | lane_w_v2_learnable_hardness | scripts/remote_lane_w_v2_learnable_hardness.sh | [0.95, 1.10] | $0.80 | ~1.5h | score > 1.20 | QUEUED |

## BLOCKED (need impl/fix first)

| lane_id | reason | unblocker_owner |
|---|---|---|
| lane_sc_plus_plus_v6 | block_fp_codec.verify_roundtrip crash (V5 killed) — needs codec tolerance fix | (paradigm-ε agent) |
| lane_pint12_pca | not yet implemented | (paradigm-β agent) |

## PARADIGM-AGENTS-PENDING

| paradigm | lane_id | ready_at | dispatch_request_path |
|---|---|---|---|
| α2 | wavelet_mask_codec | (waiting) | /tmp/dispatch_request_alpha.json |
| α3 | vqvae_mask_codec | (waiting) | /tmp/dispatch_request_alpha.json |
| α4 | grayscale_lut_refine | (waiting) | /tmp/dispatch_request_alpha.json |
| β | omega_w_v3 | (waiting) | /tmp/dispatch_request_beta.json |
| γ | joint_admm_real_archive | (waiting) | /tmp/dispatch_request_gamma.json |
| δ | joint_renderer_scorer | (waiting) | /tmp/dispatch_request_delta.json |
| ε | self_compress_nn | (waiting) | /tmp/dispatch_request_epsilon.json |
| ζ | additional | (waiting) | /tmp/dispatch_request_zeta.json |
