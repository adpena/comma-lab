# Codex Findings: HPRC Week 1 Train/Export Adapter And PR95 Full-Frame Parity

Date: 2026-05-31T23:40:00Z

## Landed implementation

- Added `tac.substrates.hprc.training_adapter.HprcCompactReceiverLongTrainingAdapter`.
- The adapter fits compact HPRC receiver components from low-resolution RGB frames, trains the decode RDO gain triple through `tac.training.long_training_canonical.run_long_training`, and exports byte-closed HPRC archives through the existing HPRC archive/runtime bridge.
- Added `tools/run_hprc_compact_receiver_training.py`, a storage-waterfall operator runner for frame `.npy` inputs. It defaults to SSD-tier custody, records source frame bytes/SHA-256, emits canonical long-training artifacts, and keeps score/exact-readiness false.
- Updated the HPRC campaign manifest so `hprc_v1_train_export_archive` is no longer scaffold-blocked; it is routed to the compact receiver long-training adapter.

## PR95 parity proof

Artifact: `.omx/research/pr95_full_frame_inflate_parity_week1_20260531T233547Z.json`

- Source archive SHA-256: `e976acd5fe565c94fb9a8c62e5200c949919f76150e84599f268d6a58588440a`
- Expected raw bytes: `3662409600`
- Public runtime raw bytes: `3662409600`
- Direct PyTorch reference: byte-exact with public `inflate.sh`
- MLX GPU optimized render: not byte-exact with public `inflate.sh`
- MLX changed byte count: `101504006`
- MLX changed byte fraction: `0.027715088448872566`
- MLX max abs uint8 delta: `3`
- MLX mean abs uint8 delta: `0.027715891745150514`
- Verdict: `mlx_decoder_or_mlx_bridge_arithmetic_drift`

## Blocker classification

This is not a PR95 source-runtime parser mismatch. PyTorch direct render exactly matches public inflate output, so the remaining PR95 full-frame parity gap is localized to MLX decoder/bridge arithmetic under the optimized GPU path.

Exact-readiness remains refused for PR95 MLX control-arm promotion until either:

1. the MLX render path reaches full-frame byte parity, or
2. a documented CPU/CUDA exact replay path establishes the candidate under the appropriate contest axis without claiming MLX parity authority.

## Next action

Run bounded MLX drift-reduction probes over the PR95 full-frame proof tool (`fixed_fp32`, `kahan_fp32`, and layer override presets), then route the best local candidate into HPRC as the control arm while HPRC compact receiver training moves to real low-resolution frame inputs and Z8 residual sidecar composition.
