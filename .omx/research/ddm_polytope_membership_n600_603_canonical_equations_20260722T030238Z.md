---
schema: canonical_equation_candidate_note.v1
utc: 2026-07-22T03:02:38Z
task: 603
lane_id: lane_ddm_polytope_membership_n600_603_20260722
research_only: true
registry_promotion: false
---

# Candidate equations — DDM cell membership

Let `A(z)` be the counted archive receiver, `y*` the exact C1 solved member, `S` the frozen SegNet
under one fixed arithmetic/batch contract, and `P` the evaluated frame-1 sites. Define

```text
c*(p) = argmax_k S_k(y*; p)
c_z(p) = argmax_k S_k(A(z); p)
M_cell(z; y*) = |P|^-1 sum_{p in P} 1[c_z(p) = c*(p)]
E_rgb(z; y*) = |P|^-1 sum_{p in P} 1[A(z)_p = y*_p in all RGB channels]
Delta_member = M_cell - E_rgb
R_slack = sum_p 1[not E_rgb(p) and c_z(p) = c*(p)] / sum_p 1[not E_rgb(p)]
C_pose(z) = stored_pose6_coordinates / (6 * described_pairs)
B(z) = exact final ZIP bytes
```

The bounded curve is `(B(z), M_cell(z;y*), C_pose(z))`. `M_cell` is a membership statistic, not
`d_seg`: there is no source-vs-candidate contest evaluation here. `C_pose` is counted-code
completeness, not PoseNet distortion or pose-tube satisfaction.

The tie-first RGB input-channel argmax disagreement `H_rgb` is a separate apparatus diagnostic. At
n256, `H_rgb = 0.229040582975` and `1-M_cell = 0.505538980166`; therefore `H_rgb` is empirically not
an estimator of cell escape for this grammar.

Because `S` has a spatial receptive field, equality of a local RGB pixel does not imply equality of
its scorer cell. Consequently `Delta_member` is a descriptive net delta, while `R_slack` is the
strict inexact-but-member rescue fraction. Both are retained per target class, target margin band,
and boundary/interior stratum.

This note is not appended to the canonical equation registry. Promotion requires MAIN review of the
measurement definition, cache-crosscheck semantics, and whether a future solver consumes the typed
receipt without converting advisory membership into score authority.
