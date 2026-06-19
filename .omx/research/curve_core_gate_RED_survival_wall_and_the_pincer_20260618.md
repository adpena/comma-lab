---
title: Curve-core gate VERDICT — RED (survival wall) + the sub-0.15 PINCER
authority: "[contest-CPU advisory] — NON-PROMOTABLE; exact pointer UNMOVED at 0.19110"
score_claim: false
promotable: false
frontier_pointer_moved: false
mission_contribution: frontier_breaking_enabler
date: 2026-06-18
verdict: RED_CURVE_CORE_HITS_SURVIVAL_WALL_LIKE_STATIC_STORE
wall_diagnosis: SURVIVAL_WALL
cross_refs:
  - experiments/results/curve_core_dseg_feasibility_gate/gate_state.json
  - experiments/probe_curve_core_dseg_feasibility_gate.py
  - .omx/research/factored_lf_core_capacity_gate_20260618T233913Z.md
  - .omx/research/campaign_inflection_three_paths_capped_concentrated_saliency_20260618.md
  - .omx/research/SESSION_SYNTHESIS_SoT_20260617_20260618.md
  - .omx/research/generative_axis_dseg_core_design_20260618.md
---

# Curve-core gate — VERDICT: RED (SURVIVAL WALL) + the sub-0.15 PINCER

**The differentiable-curve geometry family (the LAST static-description test) is RED.** All
`[contest-CPU advisory]` NON-PROMOTABLE; exact pointer UNMOVED at **0.19110** — stated plainly per
the GOAL firewall: this unit did NOT move the pointer. Its value is a measured RED that, combined
with the factored-LF RED, **closes a pincer** around byte-cheap sub-0.15 d_seg. $0, no paid GPU, no PR.

## The measured table (5 of 6 complexities; mp256 not measured — see §daemon-death; verdict robust)
| mp | ctrl pts | geo_recon | geo_seg | **realized** | bnd_flip | rate (term) | S |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8   | 171 | 0.05748 | 0.01231 | 0.01228 | 0.262 | 0.00107 | 1.287 |
| 16  | 270 | 0.01010 | 0.00980 | 0.00994 | 0.217 | 0.00165 | 1.053 |
| 32  | 401 | 0.00335 | 0.00704 | 0.00714 | 0.172 | 0.00241 | 0.775 |
| 64  | 625 | 0.00194 | 0.00717 | 0.00722 | 0.168 | 0.00373 | 0.784 |
| 128 | 838 | **0.00106** | 0.00668 | **0.00673** | 0.161 | 0.00497 | 0.736 |

`geo_recon` = pure combinatorial polygon-vs-L*; `geo_seg` = painted-frame SegNet argmax, no roundtrip;
**`realized` = hard-painted frame through the EXACT uint8 roundtrip + real SegNet vs L* = the AUTHORITY.**
S recomputed from components (mp128): 100·0.00673 + √(10·0.00034) + 0.00497 = 0.736 ✓ (matches stored —
no label inversion, NO-FAKE clean). GREEN rows (rate<0.05 ∧ realized<0.0012 ∧ S<0.15): **NONE**.

## The mechanism (measurement-first; corrects the mid-sweep param-explosion guess)
- **The geometry fits fine.** `geo_recon` falls monotonically to **0.00106 at mp128** — BELOW the
  GREEN d_seg threshold (0.0012). The decimated polygon CAN represent L* with enough control points.
  So this is **NOT param-explosion** (a 2-point mid-sweep extrapolation wrongly suggested it was; the
  full data corrected it — exactly why measurement-first beats extrapolation).
- **But `realized` plateaus at ~0.0067 from mp32 onward** and does NOT follow `geo_recon` down. The
  **survival_gap = realized − geo_recon = 0.00567** (a 6.3× gap at mp128). `boundary_band_flip`
  plateaus ~0.16 — **~16% of boundary-band pixels flip through the roundtrip regardless of geometry
  fidelity OR the differentiable color/boundary-offset pre-compensation** (which the static store
  lacked — and it STILL didn't beat the wall).
- **realized floors at 0.00673 = 2.62× the frontier d_seg and 1.05× the static partition-store's
  survival wall (0.00641).** The differentiable curve-core lands at essentially the SAME place as the
  non-differentiable static store. **VERDICT: RED_CURVE_CORE_HITS_SURVIVAL_WALL_LIKE_STATIC_STORE.**

## ⚠️ MECHANISM CORRECTION (2026-06-19, appended) — the wall is TEXTURE-dependence, not the resize
The §below attributes the wall to "bilinear boundary-band color mixing" (the resize). The curve agent's
MEASURED 3-way decomposition refutes that: at mp128 the **resize contributes +0.00005** (negligible) and the
**flat-colour→SegNet step contributes +0.00562** (the entire wall). The true mechanism: a flat-per-class-colour
frame lands OUTSIDE the GT's per-pixel argmax polytope at boundary pixels because EfficientNet-B2's features key
on TEXTURE, not region identity — independent of the resize. The §below's "resize mixing" framing is SUPERSEDED;
keep the RED verdict + the pincer (both hold), but read the mechanism as texture-dependence. Canonical:
`eval_roundtrip_deep_math_pr95_handling_and_exploits_20260619.md` §3 + the curve agent's
`curve_core_dseg_feasibility_gate_20260619T005432Z.md` (the measured decomposition).

## The deep finding — (SUPERSEDED MECHANISM; see correction above) the SURVIVAL WALL for flat-region painting
Painting a region partition with flat per-class colors, then the eval roundtrip (camera-res bicubic-874
→ bilinear-384 → round) → SegNet, **linearly blends the two regions' colors in the 1px boundary band**,
creating intermediate colors SegNet argmaxes to the wrong side → realized d_seg floors ~**0.0067**.
This holds for ANY geometry (static partition store OR fitted polylines) and is NOT fixable by
geometry fidelity or by backpropagating colors/offsets through the roundtrip. The frontier's learned
decoder reaches 0.00056 ONLY because it paints **continuous texture** the resize preserves — not flat
regions.

## THE PINCER (the campaign's terminal structure for byte-cheap sub-0.15 d_seg)
Two measured REDs close a pincer; sub-0.15 d_seg must live in a corner, and both corners are walled:
1. **Flat-region representations** (any geometry: partition-store RED S≈0.84, curve-core RED S≈0.74):
   **survival-walled at realized d_seg ~0.0067** (the boundary-band roundtrip mixing).
2. **Continuous-texture representations** (learned pixel decoders): **capacity-walled at
   d_seg ∼ 29.3·params^−0.71** (factored-LF RED); frontier-grade d_seg (~0.0006) needs ~4–10M params
   = bytes (forfeits the rate). The frontier sits ON this curve (0.00056 @ 161KB decoder).
**The only unwalled corner left:** a representation that paints **continuous texture** (corner 2, to
beat the survival wall) AND is **byte-cheap** (escapes the capacity wall) — i.e. something that breaks
`d_seg ∼ params^−0.71`. A one-shot decoder cannot (that's the law it obeys). The single remaining
candidate mechanism is the **generative/iterated** axis (operator reframe 2026-06-18): an iterated
rule with weight-sharing that gets effective-depth-N detail from rule-size bytes.

## IMPLICATION FOR THE GENERATIVE-AXIS (NCA) GATE — must grow CONTINUOUS TEXTURE, not a partition
The NCA d_seg-core gate (`generative_axis_dseg_core_design_*` / subagent) MUST grow **continuous-tone
texture** (the only representation type that beats 0.0067), testing whether **iteration escapes the
capacity wall** the factored-LF one-shot decoder hit. **A flat-region NCA partition would just
re-measure the survival wall (~0.0067) — that is NOT the question.** The decisive generative test:
*can a few-KB iterated rule grow continuous texture whose SegNet argmax = L* through the roundtrip
(the thing the frontier does at 161KB) — i.e. does weight-shared iteration break params^−0.71?*
- GREEN ⇒ the generative axis is the one corner that escapes the pincer ⇒ the sub-0.15 path.
- RED ⇒ the pincer is airtight across static AND generative families ⇒ the terminal finding:
  byte-cheap sub-0.15 d_seg is unreachable; the frontier ~0.19 is near the real floor for the
  learned-renderer regime; CLAUDE.md S_floor=0.11797 over-counts (it assumed d_seg→0 byte-cheaply,
  which the pincer falsifies) → a goal/floor re-frame for the operator.

## Daemon death (per crash-resume discipline)
The probe daemon (PID 36794, `--train-device cpu`) died during **mp256** (the largest complexity,
~1200+ control points) before writing `final_verdict`. Its stdout log was buffered (the probe omitted
`python -u`) so no traceback was captured — likely an mp256 memory spike. **The verdict is robust to
the missing mp256 row:** realized d_seg survival-plateaued from mp32 (0.00714 → 0.00722 → 0.00673), so
mp256 would only lower `geo_recon` further and raise `rate`, with realized staying survival-walled
~0.0067. The verdict was finalized post-hoc by the parent from the 5 decisive rows using the probe's
own verdict logic (transparently flagged `finalized_post_hoc` in `gate_state.json`). **Process lesson
(self-protect):** $0 probe daemons MUST use `python -u` + write tracebacks to a durable file so a
crash leaves a diagnosable record — folded into the NCA gate's launch instruction.

## Cross-refs
`factored_lf_core_capacity_gate_*` (the capacity-wall corner of the pincer), `partition_store_realization_gate_DEFER_*`
(the static-store survival wall, S≈0.84), `campaign_inflection_three_paths_capped_*` (the prior 6 families),
`SESSION_SYNTHESIS_SoT_*` (the inflection table — update curve-core #7 → RED survival wall),
`generative_axis_dseg_core_design_*` (the NCA gate this routes). Tasks #142 (this verdict), #143 (the NCA gate).
