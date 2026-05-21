# DP1 Streamer No-Chunk-IDs Dispatch Failure

## Finding

Both required DP1 first-anchor Modal runs reached Stage 4 full training and failed before training with:

```text
ValueError: Comma2k19LocalStreamer has no chunk ids; pass an explicit chunk_ids list or populate the streamer's dataset_sha256_manifest
```

Affected calls:

- baseline: `fc-01KS480WY6S90VFXX54SC7V209`, label `substrate_pretrained_driving_prior_original_baseline_modal_t4_paired_dispatch_20260521T030640Z`, rc=1, elapsed about 21s
- procedural: `fc-01KS484S3Z8YZBRVMCTQ6SX8MV`, label `substrate_pretrained_driving_prior_procedural_codebook_modal_t4_paired_dispatch_20260521T030858Z`, rc=1, elapsed about 20s

Smoke-stage archives were emitted, but they are structural smoke artifacts only; they are not valid first-anchor paired-harvest candidates because full training did not complete.

## Root Cause

The recipes selected `DPP_USE_STREAMER: "1"` for a real `comma2k19` full run. The current trainer can construct `Comma2k19LocalStreamer`, but no recipe/CLI path supplies a real `dataset_sha256_manifest` or explicit chunk-id list. Real-mode streamer therefore has an empty chunk-id set.

## Fix

- Changed the DP1 baseline/procedural/null-control recipes to use the already-wired Comma2k19 local-cache source:
  - `DPP_USE_STREAMER: "0"`
  - `DPP_CACHE_DIR: /root/.cache/tac/comma2k19_chunks`
- Added a local pre-deploy guard that rejects DP1 full-run recipes selecting `DPP_USE_STREAMER=1` for `comma2k19` until a real streamer manifest/chunk-id path is wired.
- Added regression tests for the rejected streamer configuration and accepted cache-source configuration.

## Status

The corrected baseline and procedural recipes need fresh Modal dispatches. The failed calls remain recorded as non-score, non-promotional failed training evidence.
