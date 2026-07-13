# DIRECTIVE (operator-routed, 2026-07-13) → cheapen_real95_tilehalo_fp16: Lever A is a WATERFILL, not a binary mask

Operator verbatim: "we know precisely which pixels aren't seen at all and which are harder and more sensitive."
Supersedes the binary-annulus framing in the original prompt. Consume the MEASURED sensitivity stratification we
already own — do NOT re-derive any of it:

## The measured pixel-tier inventory (all n600, all committed)
- TIER 0 — NEVER SEEN: #401 blind coordinates — 230,904 camera px/frame (22.70%) with provably ZERO weight through
  R for BOTH scorers (bit-identity proven n600; blind_coordinate_rate_lever memo). PLUS #139 hood static core
  (class-4 MyCar, 25.6% of area, temporal IoU 0.994 — static across the clip).
- TIER 1 — SEEN, INSENSITIVE: the large-margin interior. The margin field IS the continuous per-pixel sensitivity:
  Fisher curvature <-> (−margin), Pearson 0.978 (L1 unification). #141's ONE margin-saliency map
  (∂margin/∂input on the 512×384 grid) is the canonical object.
- TIER 2 — HARD/SENSITIVE: the ~4.7%-area boundary annulus carrying ~97% of d_seg (#333), plus the exact
  render-px → scorer-px influence map from the #391 composite-R kernel adjoint flip ledger.

## The directive
Lever A = a THREE-TIER COMPUTE WATERFILL over the margin-saliency map (compute-per-pixel ∝ measured sensitivity),
exactly as #157/#336 waterfill bits-per-tensor: {TIER 0 never-compute · TIER 1 low-cadence/coarse refresh
(pre-registered cadence K, state it) · TIER 2 every-step full precision with exact halos}. Compute allocation =
rate allocation = the SAME Fisher/margin object — one sensitivity field, three consumers (bytes, loss weights,
FLOPs). Derive the expected speedup from the MEASURED tier areas (22.7% + static-core never; large-margin interior
low-cadence; ~5-10% every-step), not from the binary annulus alone. Exactness discipline unchanged: TIER 2
evaluated pixels bit-compare vs full-frame; TIER 1 staleness is a TRAINING-path tolerance (1-thread principle),
never an authority claim. If you have already built the binary-annulus version, keep it as the ablation control
and add the waterfilled arm — the A/B between them is itself signal.

## ADDENDUM (operator, same day): the CLASS-LEVEL axis — waterfill = margin × class-pair weight, cadence per class-pair
Operator: "We also have class level analysis." The spatial tiers under-resolve a MEASURED ~120× per-area sensitivity
spread across classes. Consume (all measured, committed — do not re-derive):
- Canonical class order + stats (gt_n600/gt_n96, MEMORY class-index law): 0=Road 22.9% area IoU 0.955 · 1=Lane 0.59%
  area IoU 0.263 (THE unstable orbit) · 2=Undrivable 49.3% IoU 0.995 · 3=Movable 1.56% IoU 0.903 · 4=MyCar 25.6%
  IoU 0.994 (static core).
- d_seg flip mass: ~50% Road / 19% Lane / 13% Undrivable → flip-density per area-share: Lane ≈ 32× · Road ≈ 2.2× ·
  Undrivable ≈ 0.26× · hood ≈ 0 — a ~120× dynamic range the binary annulus flattens.
- Per-class-pair machinery already built: σ_cc′ anisotropic surface tension (#382) · per-class λ sensors (#315,
  #253/#255 attribution) · per-class-λ costate arm (#433, aniso formulation −18%) · L83 ground-plane law (horizon
  97.5% temporally coherent).
DIRECTIVE REFINEMENT: the compute waterfill weight = margin-saliency × class-pair sensitivity (flip-density), and
the TIER-1 refresh CADENCE is PER CLASS-PAIR from measured temporal IoU: Lane-adjacent tiles every step; Movable
tiles per its 0.903 IoU; Road↔Undrivable horizon at low cadence; hood never. State the derived per-class-pair
cadences with their IoU provenance. This is the same per-class decomposition v8 carriers use for BYTES, applied to
FLOPs — third consumer of the same object.
