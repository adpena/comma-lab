# Findings

## 2026-04-10 promoted floor

- Track B promoted floor is **1.33** authoritative (compliant ~1.356 with .pt inside archive).
- Variant: `dilated_h64`, standard loss.
- Platform: `modal_a10g`, 905 epochs.
- PoseNet `0.00218374`, SegNet `0.00609921`, rate `0.02301653`.
- Archive: 903KB (877KB video + 46KB compressed checkpoint).
- Canonical score/report mirrors are generated from `.omx/state/promoted_result.json`.

## 2026-04-10 KL distillation declared DEAD

Two independent authoritative evaluations confirmed KL distillation does not transfer from proxy to authoritative scoring:

1. **First eval** (sw=100, PoseNet gradient cap): proxy 1.25, authoritative **1.85**. SegNet improved 19% but PoseNet regressed 26x (0.00218 to 0.05725).
2. **Second eval** (sw=30, no cap): proxy 1.43, authoritative **2.05**. Even without the gradient cap, PoseNet regressed 37x (0.00218 to 0.081).

Root cause: KL distillation's gradient signal structurally dominates PoseNet MSE at any segnet_weight above ~5. The proxy does not replicate the inflate pipeline's amplification of PoseNet-sensitive texture degradation.

Conclusion: standard loss is the ONLY loss mode that transfers reliably. KL distill is permanently retired.

## 2026-04-10 adaptive weight formula declared DEAD

The adaptive weight formula `w_s*(p, T) = 20*sqrt(p/0.1)/T^2` was found to be vacuous:

- The Hinton T^2 correction was already inside the KL loss function, so dividing by T^2 in the weight formula double-corrected, making the weight temperature-independent by construction.
- The compound invariant `w_s*T^2` was trivially constant (T^2 cancels), not by any physical property.
- The formula produced w_s=0.80 when the empirical winner used w_s=100 (125x mismatch).
- Lean 4 proofs are mathematically correct but verify a vacuous identity.

The score sensitivity analysis (d(score)/d(seg) = 100, d(score)/d(pose) = 5/sqrt(10p)) remains useful as a diagnostic.

## Key lessons (cumulative)

- Standard loss + dilated convolutions is the only proven technique.
- Proxy scores are untrustworthy for any technique involving KL distillation.
- 15+ proxy scorer bugs accounted for ~0.5 points of phantom score (fixed in tac v1.0.0).
- The int8 quantization gap (2.25x on PoseNet) is the single largest deployment hazard.
- All preprocessing (blur, denoise, chroma subsampling) kills PoseNet.
- PoseNet gradient caps/clamps cause silent regression.
