# Revival plan: Lane UNIWARD-V9: textured-region embedding delta on PR106 mask

**Gem**: `src/tac/uniward_delta.py`
**ID**: `05_uniward_delta_pr106_mask_channel`

## Current state

Level-3 from Lane UNIWARD v8 1.14 [contest-CPU advisory]. Needs CUDA confirm. Fridrich UNIWARD principle: errors concentrated in textured regions are undetectable by SegNet.

## Files touched

- experiments/build_uniward_delta_for_pr106.py (new)
- src/tac/uniward_delta.py (no changes)
- src/tac/uniward_texture.py (no changes)

## Integration sketch

1. Compute per-pixel UNIWARD weight = 1 / local_variance over the contest video.
2. For each PR106 reconstructed frame, identify SegNet error pixels.
3. Encode error correction sidecar as UNIWARD-weighted delta (low-bit-depth in textured pixels, high-fidelity in low-variance pixels).
4. Bundle as charged sidecar in PR106 archive (under 5KB target).
5. Inflate: decode HNeRV → apply UNIWARD-weighted delta → eval.

## Test plan

- Adversarial test: SegNet on baseline vs +UNIWARD delta — assert seg_dist reduction > 0.005 per 5KB.
- Compliance: pass `--allow-pending-compliance` gate (per CLAUDE.md `check_uniward_delta_has_attestation_gate`).
- Score: must be ≤ 0.20946.

## Predicted score basis

PR106 seg contribution 0.067142. UNIWARD delta in 5KB charged → expected -0.005 to -0.015 score reduction per Lane UNIWARD v8 lineage. CUDA-CPU drift tolerance: v8 was 1.14 [contest-CPU advisory] — needs CUDA gate.

## What would change my mind

If CUDA replay shows v8 advisory was a CPU-only artifact (per MPS-falsification rule equivalent for CPU), abandon.

## Blockers resolved in plan

- UNIWARD attestation gate requires `--allow-pending-compliance` flag; documented in revival plan.

## Skunkworks council deliberation

Fridrich/Yousfi LEAD endorse (Fridrich's own principle). Contrarian: 'CUDA confirm of v8 1.14 first — don't compose untested primitives'.

**Verdict**: VOTE 7/10 GO conditional on CUDA confirm of v8 baseline; 3 dissents (Contrarian/Shannon/Boyd).
