# Contest-domain-optimal correction representation space (operator riff, 2026-06-10)

**Operator:** *"perhaps the residual is worth it as well"* + *"might be a way to accomplish more finely
grained or detailed or at another level or dimension; may be other ways of representing and calculating
or applying as well that are more contest and domain optimal."*

This memo is the durable MENU of how to represent / calculate / apply the d_seg correction (the residual
that fixes SegNet-argmax flips), ranked by contest-domain optimality. It enriches the running coders
(#72 margin-conditional residual coder, #54 cross-pair waterfilled corrector) and is the v2 reactivation
if their first representation doesn't beat the 1.27 B/flip waterline. It is also the seg-side actuator
representation for the lever-C distortion-closure campaign.

## The governing realization
The d_seg residual is **NOT a set of pixel-flips — it is a PARTITION-DELTA** (the set difference between
the renderer's SegNet-argmax partition and GT's). A partition's information lives on its **1-D boundary**,
not its 2-D area. Every contest-optimal representation follows from this + the exact `evaluate.py` /
`modules.py` structure (SegNet resizes frame1 to 384×512 THEN argmaxes over 5 classes; d_seg = per-pixel
argmax-flip rate). The naive per-pixel sidecar (1.525 B/flip, DORMANT) failed because it codes the area,
not the boundary, and unconditionally.

## The representation axes (each lowers the per-flip cost below the 1.27 waterline)

### 1. Dimension drop (2-D pixels → 1-D contour) — the biggest lever
Code the CONTOUR of each argmax-disagreement region + its target class, not the pixels. A disagreement
region of `A` pixels has a boundary of ~`√A` steps → contour coding is ~`√A/A` = `1/√A` cheaper per
flip. A 100-pixel disagreement region: ~10 contour steps vs 100 positions = 10× cheaper. Chain-code /
crack-edge / the `boundary_math.contour_codec` (#52). This alone can blow past 1.27 B/flip for any
NON-isolated flip (the isolated salt-and-pepper ones stay expensive — waterfill them out).

### 2. The effective grid is 384×512 (post-resize), NOT 874×1164 (camera)
SegNet bilinear-resizes frame1 to (384,512) then argmaxes. The scored argmax lives on 196,608 cells at
384×512. Specify corrections THERE — camera-resolution detail is in the resize null space (certified
free, #47/#49). Correcting at camera res wastes bits on invisible DOF.

### 3. Per-class-pair structure
Flips are specific confusions (class A↔B at specific boundary types). Code by (confusion-type, region)
— flips of the same class-pair at the same boundary share a model; the confusion-type is ~log2(pairs)
< log2(5 classes) and is spatially predictable.

### 4. Multi-scale / hierarchical (Daubechies domain)
Quadtree / wavelet the partition-delta: coarse corrections fix many pixels at once (whole-region class
flips), fine corrections refine the boundary. Hierarchical-coarse-gates-fine (the canonical wavelet
discipline) — pay fine bits only where the boundary is fine.

### 5. Alternative APPLICATION operators (not additive RGB delta)
The correction need not be an additive pixel delta. Lower-dimensional operators the decoder applies:
- **Boundary WARP** — a 1-D displacement field that moves the renderer's argmax contour onto GT's
  contour. The correction is the displacement along the boundary (1-D), not the recolor (2-D).
- **Region LUT / remap** — one parameter recolors a whole region to flip its argmax class.
- **Morphological snap** — dilate/erode/open/close the partition to fix systematic boundary bias (a few
  structuring-element params fix many flips).
- **Parametric boundary** — fit the GT boundary with a spline/polygon; store the low-D spline params.

### 6. Better-conditioned CALCULATING (the margin-polytope solve at the contour level)
Compute the correction via the margin-polytope closed-form (the minimum perturbation per flip,
`boundary_solver`'s Gα≥b, #55) — but apply it at the CONTOUR/region level, conditioned on the margin
prior the decoder REGENERATES for free from the renderer's own output (it has the renderer → recompute
the SegNet margin field → know the flip-prone contour). The sidecar stores only the conditional residual.

## The contest-domain-optimal synthesis (the target representation)
Code the d_seg residual as: the **contour of the partition-delta at 384×512** (dimension drop), labeled
**per class-pair**, **multi-scale** (quadtree/wavelet), applied via **boundary-warp or region-remap**
(low-D operator), conditioned on the **margin prior** (decoder-regenerated, free side-info), with the
**isolated salt-and-pepper flips waterfilled OUT** (they stay above 1.27 — don't store them). The score
move: fixing the contour-codeable subset drops d_seg toward 0 at a byte cost far below the per-pixel
floor; net ΔS = −(seg drop) + 25·(contour bytes)/D, byte-closed + exact-eval-verified.

## Routing
- **#72** (margin-conditional residual coder, RUNNING) — tests axes 1+6 (contour + margin-conditional).
  If it beats 1.27 for a subset → exact d_seg drop. If not → this menu's axes 2-5 are the v2.
- **#54** (cross-pair waterfilled corrector, RUNNING) — the λ* allocator; consumes this menu's per-flip
  cost model for the seg side; cross-pair for pose.
- **Lever-C distortion-closure** (gated on #63) — this menu IS the seg-side actuator representation once
  a contiguous-residual base exists.
- **v2 reactivation** — if #72's contour coder doesn't beat 1.27, the alternative application operators
  (warp/LUT/morphological/parametric, axis 5) + multi-scale (axis 4) are the next representations to try.

## NO-FAKE discipline
Every representation must (a) ACTUALLY drop exact d_seg when applied (decode → exact SegNet argmax →
verify the stored flips fixed, zero new bad flips), (b) beat the 1.27 B/flip waterline for the stored
subset (else waterfill it out), (c) recompute S from components on the exact scorer. A representation
that codes cheaply but doesn't verifiably drop exact d_seg is a fake.
