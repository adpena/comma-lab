---
schema: canonical_equations.v1
date_utc: 2026-07-24T19:16:23Z
lane_id: ddm_ev1_campaign_evidence_joins
research_only: true
execution_allowed: false
score_claim: false
main_landing_review_required: true
---

# DDM EV1 campaign evidence-join equations

Let `p in {0,...,599}` index the 600 two-frame evaluator inputs.  For a
receiver endpoint `x`, let `R(x)_p` be its exact uint8 camera pair, `L(x)_p`
the frozen SegNet frame-1 argmax cells, and `P(x)_p` the frozen PoseNet
six-vector.  No per-pair score is inferred from another pair or from a proxy.

## V19 receiver-closed join

For control `c` and candidate `v`, each joined row contains the exact
receiver/scorer observations

```text
e_x,p = sum_q 1[L(x)_p,q != L*_p,q]
u_x,p = ||P(x)_p - P*_p||_2^2 / 6
Delta e_p = e_v,p - e_c,p
Delta u_p = u_v,p - u_c,p.
```

The counted-rate term has one global home only:

```text
Delta B_V19 = bytes(archive_v) - bytes(archive_c)
sum_p Delta B_V19,p is forbidden
home(Delta B_V19) = one shared archive-delta object.
```

Thus V19 closes only when the pair ids are exactly `0..599`, every row is
receiver-closed, and the single global byte delta is counted once.

## RD1 exact candidate edges and scorer metric

For each of three ordered endpoint edges `j`, the only admitted distortion
coordinate is the evaluator composite

```text
Delta D_j
  = 100 * (E_after,j - E_before,j) / (600 * 384 * 512)
    + sqrt(10 * d_pose_after,j) - sqrt(10 * d_pose_before,j).
```

Semantic dimensions are partitioned by exact target class and G4 temporal
class.  The Pose6 dimension is split across G4 classes by measured receiver
absolute-step mass.  Euclidean-naive rows are inadmissible: the semantic
surface is the exact rank-4 margin-Fisher geometry and the Pose surface is
the composite-R quadratic/Hessian geometry.

For typed home `h=(j,stratum,visibility,g4)`, define nonnegative measured
weight `w_h=|Delta D_h|`.  Largest-remainder allocation forms an exact
integer partition of the candidate byte delta:

```text
b*_h = Delta B_j * w_h / sum_g w_g
b_h  = floor(b*_h) + largest-remainder correction
sum_h b_h = Delta B_j
interior(range_h) intersect interior(range_g) = empty, h != g
union_h range_h = [0, Delta B_j).
```

This is an exclusive accounting home, not a claim that a ZIP byte is
physically separable.  EV1 never writes an RD1 dual price; all 162
`lambda_bytes_per_D_dimension` values remain owned by
`ddm_ms2r_tolerance_capped_solve_r2`.

## G4 shared-field amortization

G4 measures the exact recurrence table

```text
k(q,a,b) = sum_p 1[L_p(q)=a and L*_p(q)=b and a!=b].
```

For an aggregated G4 class `g`, let `M_g` be event mass and `U_g` the number
of once-coded loci or tracks.  Its effective reach and amortized home cost are

```text
k_g = M_g / U_g
b_amortized,h = b_h / k_g.
```

`STATIC_IN_IMAGE` uses G4 exact-k recurrent loci (`k>=2`);
`STATIC_IN_XI_PROXY` uses measured ξ-proxy track length; `TRANSIENT` has
`k=1` after removing the two ξ-proxy events.  This is a measured G4
class-level reuse prior applied to typed RD1 homes, not a fresh edge-specific
k measurement.  Because the aggregated recurrent bucket mixes multiple exact
`k` values and G4 has no exclusive `k=600` aggregate, it is
`shared_k_frames`, not falsely promoted to `shared_clip`.

Generic decoder, solver, and transport interpreter code has zero counted
bytes.  Irreducible video-derived residual and ξ statistics remain counted
exactly once; the G4 ξ source is a target-cache metric proxy and is not
promoted to independently observed physical BEV.

## Receiver-step histograms

For every home, exact uint8 absolute steps form a 256-bin histogram

```text
H_h[s] = count(|R(after)-R(before)| = s), s in {0,...,255}
H_h[0] = 0
changed_h = sum_s H_h[s]
step_sum_h = sum_s s H_h[s].
```

All 162 histograms are packed in typed-key order and round-trip through real
Brotli quality 11 byte-identically.  Histograms are receiver evidence; they
do not replace the scorer metric.

Pointer remains `0.1910828242 [contest-CPU]`.  This local macOS-CPU
frozen-scorer evidence is research-only and requires MAIN landing review.
