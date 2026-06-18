# Score-aware taper / channel-allocation — the second water-fill axis of the pivot (2026-06-18)

**Operator: "may need to update the taper too / channel alloc."** Correct — and it's a first-class lever, not
a detail. The capacity pivot is NOT "uniform base_ch bump + QAT"; it's "higher capacity, with the per-stage
CHANNEL ALLOCATION (taper) re-solved score-aware." This memo captures the design; it folds into the running
pivot build (`capacity_rd_score_aware_qat_pivot`) at its $0 desk-calc gate (before any long train).
All `[advisory]`; pointer UNMOVED 0.19110. Subsumes/updates task #121 (d_seg-aware taper).

## The two water-fill axes (the unifying frame)
The rate↔d_seg allocation has TWO orthogonal axes, both governed by the master gradient (∂S/∂d_seg=100):
1. **Channel allocation (the TAPER):** how many weights per decoder stage. = capacity AND bytes per spatial band.
2. **Bit allocation (QAT):** how many bits per weight. = bytes per weight at fixed capacity.
Both must be **score-aware water-filling**: put resolution (channels × bits) where d_seg sensitivity lives,
starve where the detector is blind. QAT alone (the prior plan) only does axis 2; the taper does axis 1.

## The spatial geometry (why the taper is score-aware)
HNeRV decoder stages (×2 each from 6×8): **6×8 → 12×16 → 24×32 → 48×64 → 96×128 → 192×256 → 384×512** (7 stages).
The frozen SegNet preprocesses to 512×384 then stride-2 stem → ~256×192 first feature grid, so it DECIDES on
~256×192-and-coarser. Therefore:
- **The 96×128–192×256 stages ARE the d_seg-decision band** (where the 882×-concentrated boundary flips live)
  → CONCENTRATE channels here (capacity to resolve the boundary = lower d_seg).
- **The final 384×512 stage is ABOVE the SegNet stem-Nyquist → d_seg-BLIND** → STARVE its channels (saves
  bytes at ~0 d_seg cost). CAVEAT (Yousfi): the 384×512 stage also feeds PoseNet (YUV6, may see higher freq)
  + recon — so certify the starve against d_pose too, not just d_seg. Starve only to the d_pose-safe point.
- Memory anchor (#121 d_seg-aware realloc): current allocation ~81% params low-freq / ~1.75% at the 192×256
  flip band → the realloc moves MORE channels to the 192×256 decision band, fewer to the d_seg-blind 384×512.

## How it composes with the capacity pivot + QAT
S(p, taper, bits) = 100·d_seg(p, taper) + √(10·d_pose) + 25·B(p, taper, bits)/B₀.
- **Capacity p:** total budget (raise until d_seg→~0.0004).
- **Taper:** distributes p across stages → put it in the 192×256 decision band (lowers d_seg per param), pull
  it from 384×512 (lowers bytes per param at ~0 d_seg).
- **QAT bits:** per-weight precision by margin-saliency (the |∂margin/∂input| map), int4 on the d_seg-blind.
The three are a JOINT water-fill: at the optimum, the marginal ΔS per resource (channel, bit) is equalized
across stages — i.e. don't add a channel/bit to the 384×512 d_seg-blind stage when the same resource at the
192×256 band buys more d_seg. The taper re-solve is the channel-axis half; without it, a uniform base_ch bump
wastes capacity+bytes on the d_seg-blind final stage (the operator's exact concern).

## Fold-in plan (when the build's $0 desk-calc gate returns)
1. The desk calc (existing d_seg(p), QAT byte model) is taper-agnostic → fine to run as-is.
2. At the gate (before the long train), re-solve the taper at the chosen capacity: parameterize the 7-stage
   channel vector, bias toward the 96×128–192×256 stages, starve 384×512 to the d_pose-safe floor. Train the
   higher-capacity decoder with THAT taper (not uniform), margin-hinge seg, stable trunk-pose, warm-LR.
3. Then score-aware QAT (bit axis) on top. Byte-close + measure S.
A cheap pre-train check: at the chosen capacity, A/B two tapers (uniform vs d_seg-band-concentrated) for a
bounded budget; keep the lower-d_seg-per-byte one (the #121 A/B, now at the new capacity).

## Cross-refs
`campaign_math_review_dynamics_and_optimization_20260618.md` (the master-gradient + capacity↔rate stationarity),
`yousfi_council_checkin_unified_margin_saliency_20260618.md` (the stem-Nyquist-blind band + d_pose caveat),
`tac.margin_saliency_map`, task #121 (d_seg-aware taper, now folded here at the new capacity), #136 (QAT), #141.
