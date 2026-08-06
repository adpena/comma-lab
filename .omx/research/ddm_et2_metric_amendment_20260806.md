# et2 Amendment 1 — seg-metric oblique projection (operator steer 2026-08-06 "Euclidean is not optimal")

All advisory, score_claim=false. Derivation (MAIN), consumed by the live ddm_et2 arm's
fire-order 1 (Arm E vs Arm M A/B).

## The structural fact

The Q3 pose-null property is membership in ker(A) (A = the 6×12 per-2×2-block yuv6 constraint
matrix, trainer `pose_null_projector_np`): any δ with Aδ=0 yields bit-identical PoseNet input
at float, REGARDLESS of how δ was selected. The projector's orthogonality metric therefore
does NOT affect pose-nullity — it only chooses WHERE in the null space the correction lands.

## The defect (m65 dual-metric law + operator steer)

The live projector P = I − pinv(A)A is Euclidean-orthogonal in RGB coordinates: it returns the
null-space point closest in RGB ℓ₂ — the wrong currency. m65 (Euclid-vs-Fisher cosine
SIGN-FLIP, measured) already established Euclidean ordering is not seg-effect ordering; the
measured symptom is the −44% η tax of project-after on the band.

## The cure (first-order solve-within)

For SPD M, the M-orthogonal (oblique in RGB) projector onto ker(A):
    P_M = I − M⁻¹Aᵀ (A M⁻¹ Aᵀ)⁻¹ A
satisfies A·P_M = 0 exactly and P_M² = P_M. Choosing M = GᵀG + λI, G = block-local seg-effect
rows (margin-saliency ∂margin/∂input, #141; or ms3/ms4 margin-Fisher row-Gram), makes P_M δ*
the solution of  argmin_{Aδ=0} ‖G(δ − δ*)‖²  — the pose-null correction preserving maximum
first-order seg effect. This is exactly the linearized form of the ffm1 consumer clause
(constrained conditioning, never raw projection) and DOMINATES the Euclidean choice in the
currency et1/et2's η measures: η(P_M) ≥ η(P) (both feasible; P_M optimizes the measured
objective). Identical float pose-nullity; uint8 leakage re-measured per arm (direction-
dependent — the ~4% Euclidean figure does not transfer).

Scope: applies to STATIC export-time corrections (et2). Does NOT reopen training-mode Q3
(refuted at formulation scope, la1 appendix 4 — the disarmament mechanism is orthogonal to
the projection metric). Consumers: ddm_et2 fire-orders 1/1b/2 (amendment appended to its
charter); any future ker(A) placement surface.
