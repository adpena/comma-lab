# Codex session summary — DDM v17 realized trust region

- Consumed the 2026-07-23T02:55:15Z Probe A supplement and replaced the initial top24 contract
  with the exact three-basis, four-epsilon, four-radius typed preregistration.
- Landed integer-preserving basis projections, unique Babai/M-preconditioned proposals, classical
  rho trust control, exact realized-only acceptance, and a 12-vs-12 matched j2 control.
- Measured one exact plateau iteration: zero accepted steps; best net-improving 1x1 point required
  405 harmful flips, while the best admissible boundary-normal point worsened objective.
- Registered `ddm_v17_realized_validity_ratio_uint8_v1`; preserved the pre-directive run with an
  explicit no-authority invalidation; pointer `0.1910828242 [contest-CPU]` unchanged.
- Repaired the review-found future n600 comparator so absent control-objective custody fails closed;
  the executed zero-acceptance measurement path was unaffected and its historical producer is
  exactly reconstructable from the landed reverse patch.
- MAIN review is required before merge. A future continuation should add new proposal information
  (clean constrained solve, epsilon>64 pricing study, or bounded deterministic escape), not replay
  the same initial grid.
