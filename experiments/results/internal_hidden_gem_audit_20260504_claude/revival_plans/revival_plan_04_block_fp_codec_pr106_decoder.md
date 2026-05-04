# Revival plan: Lane BlockFP-V2: block-FP weight codec on PR106 HNeRV decoder

**Gem**: `src/tac/block_fp_codec.py`
**ID**: `04_block_fp_codec_pr106_decoder`

## Current state

Level-3 measured in Selfcomp 0.36 lineage. Sibling of QZS3. Block-FP shares a single exponent across a block of weights → small overhead per block.

## Files touched

- experiments/profile_block_fp_pr106.py (new)
- src/tac/block_fp_codec.py (no changes)
- submissions/apogee_blockfp/* (new sibling)

## Integration sketch

1. Same as QZS3 but use block_fp_codec instead.
2. Block size sweep: {16, 32, 64, 128} → pick smallest reconstruction error per byte.
3. Pack to ≤140KB; verify decoder forward Δ.
4. Score on T4.

## Test plan

- Block size sweep: choose Pareto-best per-block-size.
- Compression: target ≤140KB.
- Score: must be ≤ 0.20946.

## Predicted score basis

Selfcomp 1.017 bpw effective on his renderer → on PR106 decoder roughly 130-150KB. Lower than QZS3 only if block-FP exponents are more compressible than QZS3 group exponents. Same risk profile as QZS3 (quantization fidelity).

## What would change my mind

Same fidelity-failure exit criterion as QZS3.

## Blockers resolved in plan

- Same QAT finetune blocker as QZS3.

## Skunkworks council deliberation

Selfcomp/Quantizr: 'try both QZS3 + block-FP in parallel; let measurement decide'. Hotz: 'pick whichever is cheaper to ship'.

**Verdict**: VOTE 7/10 GO as parallel sibling to QZS3; 3 dissents (same as QZS3).
