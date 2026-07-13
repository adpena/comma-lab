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
