# Label-noise floor + margin-saliency map — Yousfi #1 decisive $0 measurement (task #141)

**2026-06-18. `[contest-CPU advisory]` / `[macOS-CPU advisory]` — NON-PROMOTABLE. Pointer UNMOVED 0.19110.
$0, no GPU, CPU-only (MPS owns the live train + would 2x-corrupt the SegNet).** Gates the long-train
sub-0.15 d_seg thesis from the Yousfi council check-in
(`.omx/research/yousfi_council_checkin_unified_margin_saliency_20260618.md`, REDIRECT 3).

NO FAKE: real frozen SegNet (`load_frozen_distortion_net(device='cpu')`), real basin parse-back
(`best/best_archive.bin` = the contest-visible bytes, base_ch=20 latent_dim=28 **89,136 archive bytes**),
real GT via `frame_utils.yuv420_to_rgb`. Reuses the proven `probe_yousfi_detector_cost_blindspot_b` plumbing
(no duplication) + a new `tac.margin_saliency_map` producer helper. 96 GT pairs, exact eval pipeline
(native 384×512 → bicubic camera 874×1164 → uint8 → SegNet preprocess bilinear 384×512 → argmax-flip).
JSON: `.omx/research/label_noise_floor_and_margin_saliency_20260618.json`.

## The question (well-posedness of the d_seg half of the sub-0.15 path)
Canonical arithmetic (`feedback_small_basis_rate_headroom_..._20260616`): `S = 100·d_seg + √(10·d_pose) + 25·B/37.5M`;
the bc20 small basis rate+pose floor = **0.1178** (rate 0.05935 + pose 0.05845) → sub-0.15 needs
**100·d_seg < 0.0322 → d_seg < 0.000322** (8× from the basin's 0.00260). Yousfi argues some of the basin's
residual flips are the SegNet's OWN label ambiguity (comma10k labelers disagree on the lane-paint edge —
64% road↔lane) → UNWINNABLE by the decoder. **Winnable d_seg = basin d_seg − (flips where the SegNet's own
GT-frame top-2 margin < τ).** If the label-noise floor is high, sub-0.15 d_seg is unreachable by the decoder
and the FP-shrink RATE lever becomes load-bearing.

## What was measured (real, 96 pairs)
- **Basin d_seg (advisory) = 0.002625** (S contribution 0.2625), 49,543 flip pixels. Matches the canonical
  basin 0.00260 — the parse-back is faithful.
- **GT-frame top-2 margin:** median over ALL pixels = **5.82** (the detector is decisive almost everywhere);
  but AT the flip pixels the GT margin median = **0.1501**, mean 0.2223 — i.e. the basin's flips land almost
  exclusively on pixels where the detector itself is on a class wall (confirms Yousfi's ~0.137 anchor).

## THE LABEL-NOISE FLOOR CURVE (the headline)
Fraction of current flips at pixels where the detector's OWN GT-frame margin < τ (it is barely sure on the
ground truth itself), and the resulting winnable d_seg vs the sub-0.15 target 0.000322:

| τ | ambiguous flip frac | winnable d_seg | vs target | sub-0.15 reachable? | implied S (rate+pose+winnable d_seg) |
|---|---|---|---|---|---|
| 0.05  | 21.0% | 0.002073 | **6.44×** | NO | 0.3251 |
| 0.10  | 37.2% | 0.001648 | **5.12×** | NO | 0.2826 |
| **0.137** | **47.0%** | **0.001392** | **4.32×** | **NO** | 0.2570 |
| 0.20  | 60.1% | 0.001048 | **3.26×** | NO | 0.2226 |
| 0.30  | 74.1% | 0.000680 | **2.11×** | NO | 0.1858 |

**The decisive read:** at EVERY plausible label-noise threshold the winnable d_seg is ABOVE the 0.000322
target. Even the *conservative* τ=0.05 (declaring only the deepest-ambiguity 21% of flips unwinnable) leaves
6.4× of headroom. Even the *generous* τ=0.30 (declaring 74% of flips label-noise) still leaves 2.1×. There is
**no τ at which the residual is small enough that the remaining winnable flips alone reach sub-0.15 d_seg.**

## VERDICT — is sub-0.15 d_seg reachable by the decoder?
**MODERATE-to-HIGH label-noise floor; sub-0.15 d_seg is NOT comfortably reachable by the decoder alone, and
becomes UNREACHABLE under any honest label-noise accounting.** The d_seg half of the sub-0.15 path is
**ill-conditioned, not well-posed-and-easy:**

- ~47% of the basin's flips (at Yousfi's τ=0.137) are at pixels where the SegNet can't reliably classify the
  GROUND TRUTH itself — those are the detector's label ambiguity, not the decoder's reconstruction error, and
  no RGB the decoder emits will reliably fix them (you can't impose `argmax(SegNet)`, only RGB; sister of the
  36.9%-survival wall).
- To reach sub-0.15 d_seg via the decoder you must win **essentially ALL** of the remaining winnable flips
  (4.3× reduction of the winnable mass at τ=0.137, 6.4× at the conservative τ=0.05). That is far beyond the
  long-train's observed ep30 0.00225 (which is only 1.16× below the basin, and 7× ABOVE target).

**This re-routes the campaign.** The d_seg long-train is NOT a clean sub-0.15 path on its own — it chases a
mass that is ~half label-noise and whose winnable remainder is still multiples above target. **The FP-shrink
RATE lever is load-bearing** and should run in parallel / take precedence: shedding bytes off the
stem-Nyquist-blind, low-saliency band drops the rate term (0.05935) directly and un-amortized, with no
label-noise wall in the way. The honest sub-0.15 path is **rate-led, with d_seg as a parallel grind toward the
winnable remainder — not d_seg-led.**

Caveat (intellectual honesty): "ambiguous-at-GT-margin-< τ" is a PROXY for true comma10k labeler disagreement,
not a direct measurement of it. A pixel where the detector is uncertain on the GT is *consistent with* label
noise but could also be a genuinely hard boundary the detector under-fits. Either way the operational
conclusion is the same — those flips are not reliably decoder-winnable, because the decoder can only move RGB,
not the detector's own uncertainty. The curve's monotone steepness (winnable d_seg stays multiples above
target across the whole τ sweep) makes the routing robust to where exactly the "label-noise" line is drawn.

## The margin-saliency map (the unified-lever asset task #141 consumes)
Built `tac.margin_saliency_map.compute_margin_saliency_map` — the PRODUCER of the detector's own
`|∂margin/∂input|` map via autograd through the frozen SegNet (`preprocess_input` is a clean slice + bilinear,
verified gradient-reachable; no `@torch.no_grad`/in-place/round). This is the Z8 P18 saliency on the right
vehicle (bc20). The consumer side already exists (`tac.logit_margin_sensitivity_weighted` — a
sensitivity-weighted logit-margin loss that *takes* a per-pixel sensitivity tensor; this module *produces* it).

Characterization (6 pairs):
- **Global map:** mean 0.926, gini 0.491 (moderately concentrated), boundary/interior saliency ratio **1.70**;
  percentiles p50/p90/p99 = 0.60 / 2.03 / 5.05 (a fat tail → a clear low-saliency band exists for rate-shed).
- **Flip-targeted map** (margin summed over the actual flip pixels = "what input change repairs the flips"):
  boundary/interior saliency ratio **3.15** — sharply higher than the global 1.70. **The detector's own
  gradient concentrates exactly at the low-margin pixels where the flips live.** This is the sanity check the
  Yousfi seam needed: the saliency map aligns with the d_seg-relevant pixels, so it is a valid detector-informed
  cost for (a) the weighted margin-hinge (`w(p)=exp(−margin/τ)` × saliency) and (b) the FP-shrink allocation
  (shed bytes into the COMPLEMENT — the low-saliency band — which must ALSO be d_pose-checked before shedding,
  per the Yousfi audit refinement).

## Disposition / wire-in (6-hook per Catalog #125)
- **#1 sensitivity-map:** ACTIVE — `tac.margin_saliency_map` is a new sensitivity-map producer (the detector's
  own `∂margin/∂input`), the asset for the weighted margin-hinge + FP-shrink allocation.
- **#2 Pareto:** ACTIVE (advisory) — quantifies the d_seg-axis feasible floor (winnable d_seg ≥ ~0.0007–0.0021)
  vs the sub-0.15 target; constrains the d_seg long-train's achievable region.
- **#3 bit-allocator:** ACTIVE (the lever) — the low-saliency complement is the certified rate-shed band the
  FP-shrink QAT / dead-zone allocator should target.
- **#4 cathedral autopilot dispatch:** N/A — diagnostic, no archive emitted, NON-PROMOTABLE.
- **#5 continual-learning posterior:** N/A — `[macOS-CPU advisory]` non-promotable per Catalog #341/#192.
- **#6 probe-disambiguator:** ACTIVE — this probe IS the disambiguator (d_seg-led long-train vs rate-led path);
  verdict = rate-led, d_seg parallel.

Cross-ref: `yousfi_council_checkin_unified_margin_saliency_20260618.md` (REDIRECT 3 source);
`feedback_small_basis_rate_headroom_..._20260616.md` (the 0.1178 floor + 0.000322 target arithmetic);
`probe_yousfi_detector_cost_blindspot_b.py` (the reused plumbing); the FP-shrink lever (#136) the verdict
elevates; #137 survival cert (the saliency map certifies which repairs survive the downsample).
