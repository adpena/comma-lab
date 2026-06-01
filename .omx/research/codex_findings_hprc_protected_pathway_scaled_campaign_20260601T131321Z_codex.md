# HPRC protected-pathway 32/128/600 campaign landed

Date: 2026-06-01T13:13:21Z  
Agent: Codex  
Axis: local MLX train/export, queue-owned local rate gate; no contest CPU/CUDA score claim.

## What changed

- Added executable `RESIDUAL_RC` v2 support for HPRC compact receivers:
  coarse residual tokens plus an optional high-resolution protected residual
  sidecar.
- The numpy receiver renders the sidecar, semantic mutation proof preserves the
  v2 header, and rate-collapse/transforms preserve protected residual tokens.
- Added CLI and queue controls:
  `--enable-protected-residual-pathway`,
  `--protected-residual-grid-h`, `--protected-residual-grid-w`.
- Added full-video P18/P19 protection-surface prefix projection: one full600
  `residual_protection.npy` can drive 32/128/600 campaign slices without
  per-scale ad hoc files.

## Queue artifact

- Queue:
  `/Volumes/VertigoDataTier/pact/experiments/results/hprc_compact_receiver_training_queue/hprc_protected_pathway_32_128_600_20260601T130707Z/hprc_queue.json`
- Plan:
  `/Volumes/VertigoDataTier/pact/experiments/results/hprc_compact_receiver_training_queue/hprc_protected_pathway_32_128_600_20260601T130707Z/hprc_plan.json`
- State:
  `.omx/state/experiment_queue_hprc_compact_receiver_campaign_hprc_protected_pathway_32_128_600_20260601T130707Z.sqlite`
- Queue SHA-256:
  `16e1bc10e2418c972480013a838adf290b18368390963e86f0a5ea3302dae97d`
- Plan SHA-256:
  `a61aa7dbdcaaa399cb5dae8f1bf8bb41ee9a7848415c9257c361b36c42fdce76`

## Executed queue steps

All executed through `tools/experiment_queue.py`, not manually.

| pairs | MLX train | rate collapse | follow-up | local replay gate |
|---:|---:|---:|---:|---|
| 32 | passed, 4.30s | passed, 4.49s | passed | not full-video gate |
| 128 | passed, 8.52s | passed, 14.99s | passed | not full-video gate |
| 600 | passed, 19.23s | passed, 81.60s | passed | blocked by rate |

## Byte verdict

The protected pathway works technically, but it is not rate-competitive at this
48x64 sidecar operating point.

| pairs | train archive bytes | rate-collapsed bytes | rate term |
|---:|---:|---:|---:|
| 32 | 320,563 | 227,932 | 0.1517705629 |
| 128 | 1,220,079 | 824,400 | 0.5489341210 |
| 600 | 5,727,226 | 3,969,285 | 2.6429839547 |

600-pair gate verdict:

```json
{
  "local_replay_recommended": false,
  "blockers": ["archive_rate_term_not_below_target_before_distortion"],
  "archive_zip_bytes": 3969285,
  "archive_rate_term": 2.642983954743538,
  "rate_gate_threshold": 0.1519853363
}
```

Section profile explains the blocker. At 600 pairs after rate collapse:

- `residual_rc`: 3,929,982 bytes
- `decoder_qw`: 15,005 bytes
- `manifest_json`: 1,642 bytes
- `rdo_plan`: 966 bytes
- `latents_rc`: 868 bytes

The new high-res sidecar correctly removes the false demand that a low-res
interior grid carry PoseNet geometry, but it moves the binding problem to
protected residual payload rate. This is useful evidence, not a frontier
candidate.

## Next actions

1. Add a sparse/procedural protected pathway variant: store only protected cells
   or connected components, not a dense 48x64x3 sidecar for every frame.
2. Make protected-sidecar grid/rate a queue sweep variable:
   24x32, 32x48, 48x64, sparse-topk, run-length mask, and pair-adaptive grids.
3. Move the sidecar from dense int8 raster to token grammar:
   `(pair/frame, cell_id, rgb_delta)` with entropy coding and optional
   openpilot-color/class-conditioned fill.
4. Only re-enable MLX prefilter/local CPU replay after the archive-rate gate
   clears. The current 600-pair artifact is fail-closed and should update the
   HPRC posterior as "protected dense sidecar too expensive."

