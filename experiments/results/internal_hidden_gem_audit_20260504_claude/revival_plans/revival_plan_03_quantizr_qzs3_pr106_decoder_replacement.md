# Revival plan: Lane Q-V3: QZS3 grouped-FP4 codec replacing PR106 decoder packing

**Gem**: `src/tac/quantizr_qzs3_codec.py`
**ID**: `03_quantizr_qzs3_pr106_decoder_replacement`

## Current state

Level-4. Shipped in Lane G v3 1.05 [contest-CUDA RTX 4090]. Quantizr 0.33 paradigm (88K-DSConv FP4). PR106 decoder is brotli-packed FP32 (170KB).

## Files touched

- experiments/extract_pr106_decoder_state.py (new)
- experiments/quantize_decoder_to_fp4.py (new)
- src/tac/quantizr_qzs3_codec.py (no changes)
- submissions/apogee_q/* (new sibling submission)

## Integration sketch

1. Load HNeRV decoder via PR106's compatible inflate path.
2. Per-tensor: compute fp4_max → quantize → group FP4 packer.
3. QZS3 layout: header + group exponents + packed FP4 stream + arithmetic-coded residuals.
4. Inflate: reverse pack → dequantize → load into HNeRV decoder.
5. Verify forward output Δ < 1e-3 against original FP32.
6. Score on T4.

## Test plan

- Quantization fidelity: load decoder, quant, dequant, assert max-frame Δ < 1e-3.
- Compression: target ≤140KB packed (saves ~30KB).
- Score: must be ≤ 0.20946 to ship.

## Predicted score basis

PR106 decoder 170KB brotli FP32 → ~100-130KB QZS3 FP4. Saved bytes 40-70KB → rate Δ ≈ -27 to -47 × 6.66e-7 = -2.7e-5 to -3.1e-5… NEGLIGIBLE direct rate impact unless quantization reduces SegNet/PoseNet too. Risk: FP4 quantization may degrade decoder beyond pose/seg recoverable threshold.

## What would change my mind

If decoder forward Δ > 5e-3, pose/seg will degrade enough to wipe out the rate gain. Abandon if quantization fidelity check fails.

## Blockers resolved in plan

- Need FP4 quantization-aware finetune of HNeRV decoder (~6h CUDA) — deferred to dispatch budget.

## Skunkworks council deliberation

Quantizr/Selfcomp endorse. Shannon: 'rate gain alone is small per byte; the win must come from joint quantization+decoder co-training'.

**Verdict**: VOTE 6/10 GO conditional on QAT finetune; 4 dissents (Shannon/MacKay/Ballé/Boyd want a hyperprior+arithmetic alternative tested first).
