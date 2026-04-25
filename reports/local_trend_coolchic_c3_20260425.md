# Local Trend Smoke: Cool-Chic/C3 -- 2026-04-25

## Scope

This extends the 8-frame smoke to a 32-frame, 20-epoch local trend run. It is still not a leaderboard proxy. The goal is to test whether the new renderer families can train through the real scorer path and whether compressed checkpoints improve, not to estimate final quality.

Output root:

`experiments/results/local_trend_coolchic_c3_20260425/`

Schedule:

- 32 frames
- 16 adjacent pairs
- 20 epochs
- 5 pretrain epochs
- 15 scorer epochs
- eval every 5 epochs
- FP4 QAT enabled
- MPS device
- `seed=42`
- `deterministic=True`

## Results

| Lane | Params | Best FP4 scorer | Best epoch | FP4 bytes | Uniform int4+LZMA2 | Mixed latents8+LZMA2 |
|------|--------|-----------------|------------|-----------|--------------------|----------------------|
| Cool-Chic renderer | 37,170 | 93.4409 | 5 | 58,839 B | 16,295 B | 20,368 B |
| C3 residual renderer | 36,492 | 93.4409 | 5 | 70,969 B | 16,493 B | 19,533 B |

The FP4 scorer did not improve after epoch 5 in either lane. That is a negative result for the compressed checkpoint at this tiny scale.

## Training-Loss Trend

Cool-Chic:

- epoch 5: loss 94.3579, pose 180.4322, seg 0.5189
- epoch 19: loss 93.7085, pose 177.2254, seg 0.5162

Interpretation: slight float-path improvement, but no compressed-checkpoint improvement.

C3 residual:

- epoch 5: loss 92.3028, pose 147.0910, seg 0.5399
- epoch 19: loss 68.7140, pose 168.9101, seg 0.2763

Interpretation: C3 residual learns strongly in the float/scorer path, mainly through SegNet. FP4 evaluation does not preserve that gain yet, which makes quantization/export robustness the next blocker.

## Self-Compression

Uniform int4+LZMA2 is substantially smaller than FP4 for both trend checkpoints. The crude `latents8` mixed allocation increases bytes, so the next mixed-precision pass should be scorer-sensitive rather than manually assigning all latents to 8-bit.

## Determinism

An 8-frame Cool-Chic CPU replay produced almost identical scorer metadata to the MPS run:

- MPS scorer: `93.63971843249183`
- CPU scorer: `93.63971694237571`

The saved FP4 states were not tensor-stable across devices:

- MPS vs CPU max dequantized delta: `0.01471710205078125`
- MPS replay vs MPS replay max dequantized delta: `4.57763671875e-05`

Determinism tier for current local evidence: scorer-stable, not byte-stable or tensor-stable.

## Next Gate

Before Vast.ai:

1. Make FP4/int4 evaluation preserve the C3 float-path SegNet gain.
2. Add scorer-sensitive mixed precision and compare scorer delta, not only byte size.
3. Run the same 32-frame trend on CUDA/T4.
4. Only then attempt archive/inflate integration.
