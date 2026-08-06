# ddm_ffm1 PREDICTIONS - Q3 projection-vs-conditioning check

Status: PRE-REGISTERED before the `jd8q3_window` endpoint receipt was observed by this arm.

This prediction is a literature-derived diagnostic, not a score claim. It does not authorize a scorer run, launch, candidate promotion, or pointer update.

## Source Mechanism

Paper source: Lennon J. Shikhman, "Discretization and Statistical Consistency of Functional Flow Matching," arXiv:2608.04531v1, submitted 2026-08-05.

The paper's Section 9.2 constructs a trace-class Gaussian observation model where two operations do not commute:

- Projecting/restricting the continuum conditional field onto the observed boundary direction gives multiplier `0`.
- Conditioning after the finite observation gives multiplier `0.72`.

This is not a theorem about Pact's Q3 projector. The transferable mechanism is narrower: a linear projected vector field can be an order-wrong surrogate for the conditional or constrained finite target when the hidden covariance/constraint geometry does not commute with the projection.

## Pact Analogy

Live Pact surface: `jd8q3_window` from `.omx/state/main_hot_state.md`.

Current Q3 implementation uses `--seg-grad-q3-project`, a linear projection of the seg gradient into the frame_1 pose-null subspace. The exact constrained analogue would optimize the seg objective restricted to the pose-neutral, uint8/R-surviving manifold, including curvature and finite-lattice effects. Those two objects need not match.

This arm predicts only an observable class:

> If the projection-vs-conditioning gap transfers, Q3 should hold pose but show a material seg-yield deficit versus the matched jd7-OFF continuation window. The deficit is the signal; the paper's `0.72` is not imported as the Pact threshold.

## Registered Quantities

Use the endpoint probe requested in `main_hot_state.md` for the Q3 arm:

- Baseline: ep1766 EMA shipping basis.
- Control: jd7-OFF window from the banked la1 adjudication.
- Q3 arm: `tr1_jd4_cont_ep1766_q3on`, final tag `final_ep1886_q3on`.

Definitions:

- `off_seg_gain = 0.000386` in d_seg magnitude, from the live-board preregistered OFF comparison.
- `off_pose_giveback = 0.003212` in EMA d_pose, from the live-board preregistered OFF comparison.
- `q3_seg_gain = d_seg(ep1766) - d_seg(final_ep1886_q3on)` on the same EMA endpoint basis.
- `q3_pose_delta = d_pose(final_ep1886_q3on) - d_pose(ep1766)` on the same EMA endpoint basis.
- `seg_retention = q3_seg_gain / off_seg_gain`.

Pose-held condition:

- `q3_pose_delta <= max(0.00015, 0.05 * off_pose_giveback) = 0.0001606`.

Prediction support:

- `seg_retention <= 0.75` while the pose-held condition passes.
- Strong support: `seg_retention <= 0.50` while the pose-held condition passes.

Prediction falsifier for this surface:

- `seg_retention >= 0.90` while the pose-held condition passes.

Ambiguous zone:

- `0.75 < seg_retention < 0.90`, or any endpoint where pose is not held. In that case Q3 may still be useful or harmful, but this literature-derived projection-vs-conditioning prediction is not cleanly adjudicated.

## Consumer

The immediate consumer is the `jd8q3_window` endpoint read. If prediction support lands, the follow-on is not "kill Q3." It is: treat linear projection as an instance-level constrained-descent approximation and require any next Q3 child to compare against a constrained or curvature-corrected solve rather than only against raw projected-gradient descent.

If the falsifier lands, this paper contributes no live Q3 caution beyond the already-known integer leakage caveat, and Q3 can be judged by its ordinary seg/pose/rate endpoint arithmetic.
