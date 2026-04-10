# next experiments — 2026-04-10 evening

## promoted floor: 1.33 (dilated h=64, Modal A10G, 905 epochs)

## active fleet

| Lane | Experiment | Config | Status |
|------|-----------|--------|--------|
| Local MPS | Dilated + dual saliency + STE | `--variant dilated --use-dual-saliency --alpha-seg 5000 --use-ste --boundary-weight 5.0 --alpha 5` | RUNNING (PID 11446) |
| Lightning T4 | Dilated + KL distill | `VARIANT=dilated LOSS=kl_distill TEMP_START=5.0 TEMP_END=1.0` | DEPLOYING (fix uploaded) |
| Modal A10G #1 | h=96 standard | ep 759, scorer 0.940 | RUNNING |
| Modal A10G #2 | Dilated h=64 | Produced 1.33 checkpoint | RUNNING |
| bat00 RTX 2070 | Standard h=64 | Just launched | REASSIGN to dilated + dual sal when user is at machine |

## stopped (superseded by dilated 1.33)

- Local MPS standard h=64 v5 (was ep ~190, scorer 1.452) — KILLED
- Local MPS standard h=64 temp anneal — KILLED
- Lightning standard h=64 (was ep ~53, scorer 1.455) — REPLACING with KL distill

## priority queue (council-endorsed, nothing abandoned)

### Tier 1 — In flight or next

1. **Dual saliency** (alpha_seg=5000 + STE boundary=5): Running locally. The saliency
   inversion is the single highest-EV finding. SegNet has 590x marginal leverage.

2. **KL distill** (T=5→2→1 stepwise): Deploying to Lightning. Soft SegNet targets
   bypass the hard argmax discontinuity.

3. **Per-channel quantization**: Already implemented in tac v0.9.0. All new runs use it.

4. **Hard-frame upsampling**: Needs ~30 lines. Precompute per-pair SegNet disagreement,
   oversample worst 20%. Next code change.

### Tier 2 — After Tier 1 results

5. **DualHead architecture** (1x1 seg + 3x3 pose heads): Same param budget, architectural
   inductive bias for split objectives. Needs implementation.

6. **Test-time optimization**: 5 Adam steps per frame at inflate time against frozen scorer.
   Highest variance/highest EV. Inflate budget: ~5 min available.

7. **Multi-pass inflate**: Run CNN twice. Free within inflate budget. Train with chained
   loss for best second-pass quality.

8. **Pair-aware 6ch**: Implemented in tac. Deploy when Tier 1 saturates.

### Tier 3 — Scaling

9. **h=96 dilated**: Already running on Modal. Scaling law extrapolation.

10. **LSQ learned step size**: Implemented but undeployed. `apply_lsq(model)` in Trainer.

## theoretical minimum

SegNet 0.003 + PoseNet 0.001 + rate 0.023 → score 0.975

## cycle rules

- At most 3 serious lanes in flight
- Nothing abandoned — every technique attacks a different score component
- Compounding gains: techniques are additive, not exclusive
- Always use dilated architecture as base (proven 5.6x PoseNet gain)
- Always use per-channel quantization (free precision)
