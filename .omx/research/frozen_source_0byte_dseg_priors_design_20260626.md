# Frozen-source 0-byte d_seg priors — measurement + fold design (2026-06-26)

**Evidence grade:** `[macOS-CPU advisory]` — geometry/mechanism only. `promotable=false`,
`score_claim=false`. No GPU, no MPS. Measured on the deterministic frozen-SegNet GT argmax
cache (the EXACT contest authority surface; `no_fake_verification` argmax-vs-teacher disagree
= 2.4e-5 fp16-tie).

**Source cache:** `/Volumes/VertigoDataTier/pact/lever_b_score_native_argmax_smoke_20260610/targets_n600/`
(`gt_segnet_argmax.u8` 600×384×512 uint8 + `gt_segnet_margin.f16`). Pipeline-correctness gate
PASSED (class histogram matches `targets_meta.json` exactly).
**Artifact:** `.omx/research/frozen_source_0byte_dseg_priors_20260626.json`.

## What "0-byte prior" means here (NO-FAKE boundary)
The witness produces frames; SegNet reads them. The witness has NO GT SegNet at decode (loading
the scorer at inflate is FORBIDDEN). So any static-region clamp is either: (a) a STORED table
precomputed at compress-time and read by inflate.py (COUNTED bytes, no scorer at decode), (b) a
DETERMINISTIC geometric prior regenerated at decode (free algorithm + tiny counted side-info), or
(c) GT-SegNet-derived per-frame content (NOT byte-closeable). The static map is video-derived →
it is **(a) — counted, but measured cheap** (NOT free-from-thin-air; calling it "0-byte" is only
true for the parametric-band form below, ~8 bytes).

## STATIC-REGION MEASUREMENT (temporal: GT argmax constant across ALL 600 frames)
- **static_frac = 72.4%** of pixels never change class across all 600 last-frames. dynamic = 27.6%.
- Static-class mix: class 2 (sky/undrivable) 60.0%, class 4 (ego/hood) 33.5%, class 0 (road) 6.5%.
  **Classes 1 (lanes) and 3 (movable) are NEVER static** (inherently dynamic — the binding set).
- Spatial bands: **sky rows 0–96 = 100% static** (class 2); **hood rows 336–384 = 100% static**
  (class 4); hood rows 288–384 = 96.6% static; lower-mid 192–288 = only 19% static (the road
  surface ahead — the moving content). The clean parametric clamps:
  **sky rows 0–154 (>99% static class 2, 40.4% px)** + **hood rows 297–383 (>99% static class 4,
  22.7% px)** = **63% of pixels** clampable by 2 row-thresholds.
- Boundary annulus: **per-frame codim-1 boundary = 2.19% of pixels** (confirms the ~2.26% annulus).
  Width-1 union annulus over time = 13.9% px and **captures 88.6% of temporal-flip mass** (confirms
  "≈89% of d_seg debt in the annulus"). Width-2 → 93.0%, width-3 → 94.9%.
- Margin: static px mean margin 5.96 (clamp-safe); the lowest-5% margin px carry 50.8% of flip-mass,
  lowest-22.6% carry 91.7%. Top-5% most-unstable px carry 64% of flip-mass; top-10% carry 86%.

## d_seg-CAPACITY BENEFIT (allocation gain — re-routing non-binding capacity)
The d_seg debt lives entirely in the dynamic low-margin annulus; the static 72.4% is trivially
correct (constant, high-margin). Reallocating the SAME witness capacity off the static interior
onto the binding pixels gives an effective-capacity multiplier:
- Stage 1 — clamp temporal-static 72.4% → **3.6× capacity per dynamic pixel** (1/0.276).
- Stage 2 — route within dynamic to the width-1 union annulus (13.9% px, 88.6% of flip-mass) →
  **~7.2× over uniform** (1/0.139).
- Stage 3 — per-frame boundary (2.19% px) is the tightest target → ~45×, but GT-derived per-frame
  (needs the task-space vehicle's directional basis + stored 8-dim coords, not a static map).
Caveat (honest): capacity→d_seg is NOT linear; this is a capacity-reallocation factor, not a
guaranteed d_seg ratio. It is consistent with the ALREADY-MEASURED levers in CLAUDE.md (directional
all-class basis −48% d_seg; +capacity-routing −64% combined). The static clamp is the ENABLER of
those levers, not a standalone d_seg mover. **This is the resolution of the capacity-vs-rate
trilemma: more effective capacity at equal/lower bytes by spending it on the 5–14% that binds.**

## 0-BYTE FOLD DESIGN (byte-closed, no scorer at decode)
1. **Parametric band clamp (recommended first):** store 2 row-thresholds + 2 class ids
   (`sky: rows<155→cls2`, `hood: rows≥297→cls4`) ≈ **8 bytes** → clamps 63% of pixels. inflate.py
   reads the 4 ints and forces those rows. Pure table; no GT SegNet at decode.
2. **Full static map (optional +9.4% road-interior):** single 384×512 5-class map w/ sentinel-255
   for dynamic, brotli-q11 = **1,948 bytes** (= 0.0013 score-units) → clamps 72.4%.
3. **Consumer:** the witness/generator treats the clamp as a capacity-routing MASK — the
   coordinate-INR / Fourier basis spends 0 basis functions on clamped pixels, all on the dynamic
   28% (Stage-1 gain). R-survival note: static regions are high-margin (5.96) → robust to the
   uint8/resize/parse-back roundtrip.

## BYTE-CLOSEABILITY VERDICT (per prior)
| Prior | Verdict | Cost |
|---|---|---|
| Ego-hood band (rows≥297→cls4) | **byte-closeable, COUNTED-trivial** (camera-geometry constant; stored row threshold; no scorer at decode) | ~4 B |
| Sky/top band (rows<155→cls2) | **byte-closeable, COUNTED-trivial** | ~4 B |
| Full temporal-static map (72.4%) | **byte-closeable, COUNTED-cheap** (stored 384×512 table) | 1,948 B |
| Road-interior static residual (9.4%) | byte-closeable, COUNTED (needs per-px stored map; subset of full map) | ~1,920 B |
| Boundary-annulus / directional-basis targeting (−48% lever) | **NOT a static fold** — GT-derived per-frame; byte-closeable ONLY via the task-space vehicle (free basis orientation + stored ~8-dim lane coords) | hundreds of B (coords) |
| comma10k label conventions | **free, but NOT a per-pixel prior** (public class definitions; informs loss/class-weighting only) | 0 B |
| openpilot ground-plane homography | **byte-closeable, COUNTED-tiny** IF calibration stored/derived-from-pose (3×3+intrinsics); predicts road/lane geometry as a PRIOR, NOT the full partition; expands via the generator | tens of B |

## RECOMMENDATION
- **Fold NOW into the witness as a capacity-routing clamp (byte-closed, ~8 B parametric):** the
  sky+hood parametric bands. They free 63% of pixels for ~8 bytes — essentially free — and let the
  coordinate-INR spend its whole basis budget on the binding 28%. Optionally add the full static
  map (+9.4% road interior) for ~2KB if the road-interior capacity matters.
- **Realize the d_seg drop via the task-space vehicle (NOT a standalone fold):** the directional
  all-class boundary basis + margin-saliency capacity-routing (the measured −48%/−64% levers) are
  what actually move d_seg; the static clamp is their enabler. Annulus targeting is byte-closeable
  through the generator (free oriented basis + counted 8-dim coords), never as a stored per-frame map.
- **Geometric side-info to consider for the task-space vehicle:** the openpilot ground-plane
  homography (tiny stored calibration / pose-derived) as a deterministic road/lane prior the
  generator expands — but it is a prior, not the partition.
- **Do NOT** smuggle the video-derived static map into inflate.py disguised as "code" to dodge the
  rate term (NO-FAKE #6/#7) — it is stored data (counted), and at ~8 B–2 KB the rate cost is
  already negligible, so there is no incentive to fake it.

Cross-refs: CLAUDE.md "THE CURRENT FRONTIER … WITNESS CAPSTONE" (the measured d_seg levers; chroma
as a d_seg lever) + "inflate.py is a FREE interpreter" (the free-algorithm/counted-data boundary) +
task #138 (lane-polynomial/ground-plane IoU) + task #139 (ego-hood static region — re-measured here:
hood = rows≥297, 100% static class 4 below row 336).
