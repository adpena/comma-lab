# Wavelet / Telescopic Foveation Reactivation Note (2026-05-09)

<!-- generated_at: 2026-05-09T05:45:00Z, evidence_grade: [research-ledger; no score claim] -->

## Operator Prompt

The operator noted that existing wavelet, telescopic foveation, and related
geometry work may still be useful even if we are not committed to any one lane.
This note records the current disposition so future score-lowering work can
reuse signal without re-promoting exhausted proxies.

## Current Evidence

- `reports/phase_a_pareto_20260508.md` records A3-alt Mallat wavelet
  importance as a byte-anchor only: `156,344 B`, `-21,800 B`, but
  `incremental_improvement_insufficient`; it was a weight-domain sensitivity
  proxy, not scorer-domain evidence.
- `reports/latest.md` records the same Phase-A conclusion: Xavier-L2 and
  Mallat weight-domain proxies were exhausted on PR101 and should not be
  promoted as score evidence.
- `.omx/research/lapose_foveation_archive_consumption_closure_20260506_codex.md`
  records LA-POSE/telescopic foveation as runtime-consumer-gated: useful only
  when charged foveation bytes modify a scorer-visible output path and no-op
  controls pass.
- `.omx/research/wr01_static_packet_custody_20260506_codex.md` records WR01
  HNeRV wavelet apply-transform as byte-custody-backed but not exact-score
  promoted; decoded-output changes require component-response or exact eval
  before status changes.

## Disposition

Do not kill wavelets or foveation as families. Retire only the measured
weight-domain proxy configurations that were already exhausted.

Wavelet and foveation signals should be reused in three narrower roles:

1. **Training-time scorer-domain priors.** Use `tac.wavelet_variance` and
   foveation fields as spatial masks inside T10/IB-Lagrangian or score-gradient
   training, where gradients flow through SegNet/PoseNet or their calibrated
   surrogates.
2. **Runtime-consumed charged atoms.** Use WR01/LFV1 only when archive manifests
   prove charged bytes are consumed by `inflate.sh` and no-op controls show the
   decoded scorer-visible output changed.
3. **Meta-Lagrangian ranking features.** Feed wavelet/foveation features into
   atom ranking as priors, not as promotion evidence. Exact CUDA/CPU artifacts
   remain the only score anchors.

## Reactivation Criteria

- A score-domain map replaces weight-domain Mallat/Xavier proxies
  (`score_gradient`, Hessian trace, Fisher-Rao/Sinkhorn SegNet surrogate, or
  exact component-response marginal).
- A packet contains charged wavelet/foveation bytes, the runtime consumes them,
  and a no-op control proves the changed bytes alter scorer-visible output.
- A paired exact eval records archive SHA-256, runtime-tree SHA, component
  distances, sample count, hardware, command, logs, and dispatch-claim closure.

## Immediate Engineering Hook

Prefer reusing these lanes inside the representation → prediction →
quantization → hyperprior → arithmetic → pack stack as:

`score-domain saliency map -> Lagrangian bit allocation -> runtime-consumed packet`

not as:

`weight-domain texture proxy -> byte-only promotion`.

This preserves the useful Mallat/Yousfi/telescopic intuition while closing the
Track-4 bug class where plausible proxies are anti-correlated with the true
score surface.
