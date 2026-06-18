# Label-noise floor — RESOLUTION via the frontier existence-proof (2026-06-18)

**Audit-the-auditor correction.** The label-noise-floor measurement (#141,
`label_noise_floor_and_margin_saliency_20260618.md`) concluded "sub-0.15 d_seg NOT reachable by the decoder
→ rate-led forced." My custody audit found that conclusion **OVERSTATED** — it trusted the proxy without
checking the frontier existence-proof. All `[advisory]`; pointer UNMOVED 0.19110.

## The measurement (sound as far as it goes)
On 96 GT pairs / real frozen SegNet / bc20 basin: the basin's d_seg flips land at GT top-2 margin median
**0.1501** (vs 5.82 over all pixels) — i.e. the basin's residual flips are at pixels where the detector is
itself near a class wall. The "label-noise floor" declares flips at GT-margin < τ "unwinnable"; the
achievable-floor (= unwinnable mass = current × ambiguous_frac) is τ-dependent:
τ=0.05 → 0.00055 (0.055 S) · τ=0.137 → 0.00123 (0.123 S) · τ=0.30 → 0.00195 (0.195 S).
(The subagent's "winnable d_seg" column was the *removable* mass, not the floor — mislabeled; the floor is
`current − that`. Conclusion direction unaffected.)

## The decisive correction: the frontier already beats the proxy floor
The 0.19110 frontier vehicle (177,169 B → rate 0.1180): `100·d_seg + sqrt(10·d_pose) = 0.0731`. For ANY
plausible d_pose split, the frontier's **d_seg ≈ 0.00015–0.00056** — BELOW the proxy's τ=0.137 floor (0.00123)
and at/below even the τ=0.05 floor. **A real existing vehicle already achieves d_seg the proxy called
unwinnable.** Therefore the proxy "GT-margin < τ ⟹ unwinnable" is too pessimistic: it conflates "the detector
is unsure on the GT" with "the decoder cannot fix it." A higher-capacity decoder renders RGB that resolves the
detector toward the GT class even at low-GT-margin pixels — the frontier does exactly this.

## Resolved verdict
1. **sub-0.15 d_seg is NOT fundamentally blocked.** The frontier (d_seg ~0.0003) is the existence proof; the
   sub-0.15 d_seg target 0.000322 sits right in the frontier's achieved range. CLAUDE.md S_floor=0.11797
   STANDS (the label-noise floor does NOT contradict it).
2. **The real bc20 question is CAPACITY, not a fundamental wall.** bc20 (83K params) is at d_seg 0.0026 and its
   remaining flips are genuinely hard (low GT-margin) — it MAY be capacity-walled above the frontier's d_seg.
   The test is the running margin-hinge long-train: does bc20 get below ~0.001? (If it walls at ~0.0012-0.0026,
   that's a bc20-capacity ceiling, and the sub-0.15 vehicle needs more capacity OR a better representation —
   NOT that sub-0.15 is impossible.)
3. **Rate-led is a valid HEDGE, not forced.** Rate has no label-noise wall, so FP-shrink (#136) is more
   valuable than before. But the d_seg path is alive (capacity-dependent), so keep both: margin-hinge drives
   bc20's d_seg toward its capacity floor; FP-shrink + the stem-Nyquist-blind-band allocation cut rate.
4. **The margin-saliency asset (`tac.margin_saliency_map`, built by #141) is the keeper** — flip-targeted
   boundary saliency ratio 3.15 validates it as the cost for the weighted margin-hinge + the FP-shrink
   allocation (Yousfi's unified-map design #141 stands).

## What this corrects in the SoT
The SoT headline "sub-0.15(CPU) ≈ 0.127 via d_seg→0.000322" is **NOT falsified** (the frontier proves 0.000322
is reachable with capacity) but is **bc20-capacity-CONDITIONAL**: bc20 specifically may wall above it. The
honest headline: sub-0.15 is reachable in principle (frontier-grade d_seg + FP-shrink rate); whether the
*bc20 small basis* is the vehicle that gets there is the open capacity question the long-train + FP-shrink
settle. Do NOT propagate the subagent's "sub-0.15 not reachable" — that was the too-pessimistic proxy.

Cross-ref: `label_noise_floor_and_margin_saliency_20260618.md` (the measurement), `yousfi_council_checkin_unified_margin_saliency_20260618.md`,
`SESSION_SYNTHESIS_SoT_20260617_20260618.md`, the G3 exact row (frontier-component basis).
