# Local Smoke: Cool-Chic/C3/Self-Compression -- 2026-04-25

## Scope

This is a wiring and reproducibility smoke, not a quality claim. It uses an 8-frame slice from `precomputed_local/gt_frames.pt` and exercises real scorer loading, SegNet mask extraction, QAT, eval-roundtrip scorer loss, FP4 checkpoint save, and int4+LZMA2 self-compression export.

Output root:

`experiments/results/local_smoke_coolchic_c3_20260425/`

## Bugs Caught And Fixed

1. **Eval-roundtrip full-resolution GT reshape bug.**
   - Symptom: scorer phase failed reshaping full-resolution GT frames to renderer resolution.
   - Root cause: `train_renderer.py` used rendered-frame height/width to reshape the GT roundtrip tensor without first resizing GT to renderer resolution.
   - Fix: added `resize_pair_hwc()` and resize GT before eval-roundtrip scorer loss.

2. **MPS FP4 QAT NaNs.**
   - Symptom: Cool-Chic pretrain loss became `nan`; QAT forward produced infinities/NaNs from zero-initialized decoder weights.
   - Root cause: FP4 parametrization buffers stayed on CPU when the base model was already on MPS.
   - Fix: move `QATRendererFP4(model)` to the training device after attaching parametrizations.

## Smoke Results

| Lane | Params | FP4 checkpoint | Uniform int4+LZMA2 | Mixed latents8+LZMA2 | Smoke status |
|------|--------|----------------|--------------------|----------------------|--------------|
| Cool-Chic renderer | 37,170 | 56,525 B | 16,509 B | 20,590 B | Passed |
| C3 residual renderer | 36,492 | 67,743 B | 16,877 B | 19,921 B | Passed |

Both 2-epoch smokes used:

- `seed=42`
- `deterministic=True`
- `eval_roundtrip=True`
- one pretrain epoch
- one scorer epoch
- FP4 QAT enabled
- MPS device

Both lanes produced the same tiny-slice FP4 scorer value (`93.6397`). That number is intentionally not meaningful as a leaderboard proxy because the dataset has only 8 frames and training ran for one scorer epoch.

## Reproducibility

Cool-Chic replay produced identical metadata and scorer values:

- `scorer=93.63971843249183`
- `pose=183.6695556640625`
- `seg=0.5078303217887878`

The FP4 files were not byte-identical on MPS. Dequantized state comparison showed max absolute delta `4.57763671875e-05` across six tensors. Interpretation: local MPS replay is metric-stable for this smoke, but not byte-stable. CUDA/CPU replay must be checked before any reproducibility claim stronger than metric stability.

## Self-Compression Smoke

Existing tiny-frame self-compression tests passed. The mixed-precision exporter smoke passed. Exporting the two experimental checkpoints showed that uniform int4+LZMA2 is substantially smaller than the current FP4 smoke artifacts. The `latents8` mixed allocation is larger, as expected, but creates a useful precision-control lane for later scorer-sensitive bit allocation.

## Next Gates

1. Repeat smokes on CPU and CUDA/T4 to separate MPS nondeterminism from model nondeterminism.
2. Run a longer 20-50 epoch local smoke on 32-64 frames to determine whether loss decreases.
3. Wire archive/inflate support for both variants only after local loss trend is sane.
4. Add scorer-sensitive bit allocation instead of the crude `latents8` mixed allocation.
5. Do not deploy to Vast.ai until the archive path can score decoded outputs end to end.
