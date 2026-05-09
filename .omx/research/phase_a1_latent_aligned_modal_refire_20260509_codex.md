# Phase A1 Latent-Aligned Modal Refire - 2026-05-09

## Status

Dispatch is in flight. This is not a score claim and is not promotion-eligible.

## Why This Refire Exists

The prior A1 Modal artifact collapsed on macOS CPU advisory score
`3.7216542390470915`. The follow-up bug hunt found a training/deploy contract
bug: non-smoke score-gradient training sampled random latent vectors while the
PR101 archive builder preserved the original `latent_blob + sidecar_blob`.

Commit `133f1286` fixed the contract by decoding PR101 archive latents through
`tac.pr101_archive_state_loader.load_pr101_archive_latents()` and feeding those
rows into `RealPairBatchSource` during non-smoke training.

## Dispatch

- Lane: `track1_phase_a1_score_gradient`
- Platform: Modal T4
- Instance/job id: `track1_phase_a1_score_gradient_latentalign_lr2e6_20260509T011929Z_modal`
- Modal call id: `fc-01KR553TPH27G73HMHH56MDZH0`
- Modal app run: `https://modal.com/apps/adpena/main/ap-88kd5YBTtxSP2ZVmMhnTVK`
- Dispatched at: `2026-05-09T01:20:01Z`
- Predicted ETA: `2026-05-09T03:49:59Z`
- Estimated cost: `$1.47`
- Evidence grade: `[advisory only - dispatch in flight]`
- Score claim: `false`
- Promotion eligible: `false`

## Inputs

- PR101 archive: `178258` bytes,
  SHA-256 `b83bf3488625dbd73adeddff91712994197ab53098e578e91327a0c6e49efb3e`
- PR101 source zip snapshot: `19137` bytes,
  SHA-256 `cf7853a09a08654daa5a6363eba0e36f2b5d2ac9060999f7b799d3d99f8a6a17`
- Video: `37545489` bytes,
  SHA-256 `2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9`

## Conservative Configuration

- Epochs: `40`
- Steps per epoch: `8`
- Batch size: `4`
- Learning rate: `2e-6`
- Max frames: `1200`
- Auxiliary KL weight: `0.2`
- Auxiliary pixel L1 weight: `0.01`
- `continue_after_nvdec_failure`: `true`

The run is intentionally smaller and lower-learning-rate than the retired
`20260508T230020Z` configuration. If Modal DALI/NVDEC preflight fails again,
the run may still produce a training/build archive, but it must not claim a
CUDA score.

## Exact Recovery Commands

```bash
.venv/bin/python experiments/modal_phase_a1_score_gradient_pr101.py recover \
  --label track1_phase_a1_score_gradient_latentalign_lr2e6_20260509T011929Z_modal
```

or:

```bash
.venv/bin/python tools/harvest_modal_calls.py
```

## Claim Ledger

The dispatcher recorded an active claim row in the ignored live custody ledger:

```text
2026-05-09T01:19:59Z | claude:modal_phase_a1 | track1_phase_a1_score_gradient | modal | track1_phase_a1_score_gradient_latentalign_lr2e6_20260509T011929Z_modal | 2026-05-09T03:49:59Z | active_dispatching
```

When the run is recovered, append a terminal row through
`tools/claim_lane_dispatch.py` or the recover tool's built-in terminal claim
logic.

## Next Review Packet

On recovery, do not promote from training loss or macOS-only advisory. The
minimum review packet is:

- archive bytes and SHA-256 if a packet was built;
- runtime-tree SHA;
- command/log paths;
- Modal call id and terminal claim row;
- exact CUDA component fields if DALI/NVDEC succeeds;
- macOS CPU advisory only as a collapse screen;
- measured-config classification and reactivation criteria if it regresses.
