# Revival plan: Lane Selfcomp-LUT-V2: replace PR106 mask channel with shared-latent + Gaussian-softmax-LUT

**Gem**: `src/tac/mask_grayscale_lut.py`
**ID**: `06_mask_grayscale_lut_pr106_mask_replace`

## Current state

Level-3 from Selfcomp 0.36. Reverse-engineered in `reference_pr56_selfcomp_blob_byte_layout_proper_reverse_engineering_20260501.md`. Σ=15, CLASS_TARGETS=[0,255,64,192,128]. Replaces the entire SegNet-mimicking encoder with fixed lookup + zero params.

## Files touched

- experiments/build_lut_replacement_for_pr106.py (new)
- src/tac/mask_grayscale_lut.py (no changes)
- submissions/apogee_lut/* (new sibling)

## Integration sketch

1. Train shared-latent encoder (3.6KB) + per-frame affine table (7KB) on contest video.
2. Replace PR106 mask-channel decoder with LUT lookup over shared latent.
3. Pack: shared latent + affine table + (PR106 pose) ≤170KB total.
4. Inflate: latent → affine → softmax → LUT → mask. No neural decoder for mask.
5. Score: target ≤0.20.

## Test plan

- Smoke: LUT round-trip — softmax(σ=15) over class targets — verify reproduces Selfcomp paradigm.
- Compression: target archive ≤170KB.
- Score: must be ≤ 0.20946.

## Predicted score basis

Selfcomp 0.36 with this paradigm. PR106 0.21. Stack potential: -0.005 to -0.020 ÷ 5h. The LUT is a paradigm replacement — risk is total loss of HNeRV's expressivity for mask channel.

## What would change my mind

If PR106's mask channel is already extracting latent structure better than LUT can capture (likely), this won't beat. Test with smoke first.

## Blockers resolved in plan

- Affine table training needs CUDA — deferred.

## Skunkworks council deliberation

Selfcomp/Mallat: 'LUT is paradigm-shift'. Quantizr: 'PR106 already converged near LUT in latent capacity'. Shannon: 'LUT is RD-optimal under known mask distribution; HNeRV is general-purpose'.

**Verdict**: VOTE 5/10 GO; 5 dissents — split on whether LUT applies after HNeRV pivot.
