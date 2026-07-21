# DAG FEED — full-screw advection and chart-level coefficient price

`FEED-ADVECTED-SCREW6-CHARTLEVEL-20260721` · `research_only=true` ·
`[macOS-CPU advisory]` · `score_claim=false` · `promotion_eligible=false` ·
`pointer=0.19108 [contest-CPU] UNMOVED`

```text
hash-bound gt_n600.gt_poses (six coordinates; already-counted pose sidecar)
       |
       +-- LawRef s_t=0.16 (MEASURED pose-carry anchor)
       +-- LawRef s_r=1.0 (DERIVED tac.lie identity convention)
       `-- hash-pinned G1 pitch
                         |
        xi = log_SE3(T(exp_SO3(s_r*w), s_t*v))
                         |
                 all six xi coordinates active
                         |
          +--------------+----------------+
          |                               |
 #549 solved frame0 RGB          PPCS five-class chart0
          |                               |
 openpilot ground homography       one-hot full-screw warp
          |                               |
          +------ ground-stratified base--+
                         |
         per-class median RGB residual coefficients
                         |
        int8[5,3]/pair + float16 scale/pair
                         |
 strict canonical packet -> SHA/CRC -> Brotli-11
                         |
             decode/apply/parseback exact
                         |
        factor-2 realization + hard CPU PoseNet/SegNet
                         |
      n16 two-axis gate: pose better AND bytes non-worse?
                         |
          pose YES (187.7895 -> 181.8322)
          rate NO  (269 B -> 348 B)
                         X
                n64/n600 REFUSED
```

## Triality

- DSL/schema: `predict_project_counted_full_screw_xi.v1` and
  `predict_project_chart_rgb_coefficients.v1` strictly bind complete motion and
  receiver-consumed chart coefficients.
- DAG: the executed path above. Sixteen immutable pair stages and two chunk
  checkpoints are preserved under the SSD evidence root.
- Equations: `ego_motion_cumulative_se3_bspline_v1`,
  `xi_advected_prior_per_class_chart_reconciliation_v1`,
  `lane_band_ego_factorization_source_reparam_v1`, and
  `realization_breakeven_bytes_v1`.

## Terminal edge and routing

Full-screw `xi_l2` is genuinely nontrivial (full-n600 median `4.9830`, max
`5.6088`, versus planar max `0.05226`), but the tested one-ground-depth framing
only reduced n16 `d_pose` by 3.17% while multiplying `d_seg` by 2.31. The
full-screw coefficient stream was 348 B versus 269 B static (+79 B, +29.37%).
The composed action therefore lost `+0.272859` at the canonical lambda.

Route the point `(348 B, d_pose=181.987147, d_seg=0.016764005)` to U1 as a
rejected R-D point and to G-pose #603 as a formulation-negative row, not a
waterfill coefficient. The next disambiguator is depth-stratified screw action:
ground homography, sky rotation-only, hood identity, object/movable motion, and
Fisher-margin boundary coefficients. Advection as a family remains open.

## STORES CONSULTED

PPCS B2 seed/decoder; `gt_n600` RGB and stored six-coordinate pose; #549 target
memo and deterministic source-plane reconstruction; predecessor planar receipt;
hash-pinned G1 worldsheet receipt; `tac.lie`; native upstream CPU-Torch scorer;
lane registry; delegation inbox; SSD stage/checkpoint tree. No quarantined
archive bytes were consumed. MAIN review is required before landing.
