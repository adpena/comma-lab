"""tac.witness_curriculum — LADDER-style difficulty-homotopy operators for rare-class
island birth in the level-set witness (task #323).

The rare-class birth failure (measured 2026-07-06, memory
``why_mod32cap_baseline_has_zero_lane_movable_islands``): cross-entropy will not birth
a <1%-area class into the SegNet argmax. LADDER's cure (arXiv:2503.00735) — make the
hard thing WINNABLE in an easier variant, birth it, anneal the assistance back — is,
formally, our costate controller's difficulty gradient (LADDER ⊂ our Hamiltonian
costate law; memory ``ladder_costate_optimal_difficulty_gradient_lane_movable``). This
package supplies the SPATIAL target-easing operators with a CONTINUITY +
MANIFOLD-PRESERVATION guarantee so a witness trained on the eased target transfers back
to the true target as the easing anneals to zero.

Two geometries, MEASURED + adversarially reviewed (n600 cached GT, 2026-07-06):

* BLOB classes (movable): ``sdf_dilation_eased`` — SDF is 1-Lipschitz ⇒ the eased-target
  path is Hausdorff-continuous in the radius ⇒ bounded per-step d_seg change; the family
  is a nested filtration; dilation = forward-Euler of the outward level-set flow
  ∂ₜu=+|∇u|. Transfer follows from Lipschitz flow-dependence. MEASURED GO (birthable at
  r≈3, coherence 0.80, continuous path).

* CURVE classes (lane): the SPATIAL easer is ``oriented_width_eased`` — dilate ALONG the
  per-component tangent so every eased target stays a (thin) CURVE, never becoming a
  blob (isotropic dilation of a curve LEAVES the curve manifold ⇒ no transfer = the
  measured NO-GO). BUT the PRIMARY lane lever is NOT spatial: lane is already ~81%
  coherent (≥50px segments); its birth barrier is AREA/MARGIN, so the lane lever is the
  LOSS-SPACE per-class-λ over-weight→anneal (the costate per-class λ, ``tac.witness_control``,
  task #315). Adversarial finding (do-not-fake): oriented dilation does NOT bridge dash
  gaps better than isotropic (the per-dash tangent is too local to span inter-dash gaps);
  its only validated virtue is manifold-preservation (stays thin/on-curve), so it is a
  SECONDARY birth-gradient widener, not a dash-bridger.
"""
from tac.witness_curriculum.eased_targets import (
    birthability,
    oriented_width_eased,
    sdf_dilation_eased,
)

__all__ = ["birthability", "oriented_width_eased", "sdf_dilation_eased"]
