# REUSE MANIFEST — BEV staticity v2 absolute trajectory

| required surface | reused implementation / artifact | disposition |
|---|---|---|
| frozen RGB/f1-label/within-pose source | hash-pinned `gt_n600.npz` | reused via ZIP_STORED mmap |
| frozen singleton SegNet | `tac.boundary_math.seg_core` + upstream weights | reused; f1/cache equality fail-closes |
| frozen singleton PoseNet | upstream `DistortionNet.posenet` | reused for exact cross targets |
| G1 calibration | `load_g1_worldsheet_motion` + LawRef/receipt hashes | values reused; stale transition proxy superseded only in v2 custody |
| SE(3) exp/log/compose/inverse | `tac.lie._se3_numpy` | reused; dual f0/f1 phase charts added |
| horizon/camera/openpilot IPM | `tac.clip_profile`; `lane_sdf_component.image_to_ground` | reused with `v_h=174`, `cam_h=1.22` |
| class-order detection | `tac.clip_profile.detect_class_order` | reused; n64 canonical map verified |
| subpixel/Fisher orientation | registered `t=M_p/(M_p+M_q)` shallow-side law | reused unchanged |
| staticity/events/polynomials | v1 probe helpers | reused after D0 correction |
| xi B-spline helpers | `fit_se3_bspline_controls`, `bspline_fit_error_curve` | wired but correctly not executed after D1/D2 negative |
| raw Gaussian K | no stable discrete estimator admitted | refused; directrix/ruling residual retained |

## Minimum-new-code justification

The v1 probe already provided SSD-resumable boundary extraction, static-segment summaries, and
fail-closed D3. Searches across `src/`, `tools/`, the v1 artifacts, G1 worldsheet tooling, and frozen
GT-cache builders found no reusable composition that simultaneously:

1. rebuilds f0 labels under the cache's exact singleton scorer geometry and checks f1 equality;
2. scores the missing cross-pair PoseNet transition directly;
3. constructs phase-consistent absolute `A_f0` and `A_f1` charts;
4. isolates the largest bottom-connected MyCar hood before Road/Lane; and
5. refuses n600/D1-D3 unless the exact-source n64 receipt passes.

`tools/measure_bev_staticity_developability.py` therefore extends the v1 instrument at those custody
and gating seams. It does not fork a scorer, evaluator, Lie algebra, IPM, class detector, B-spline
implementation, DSL compiler, launcher, or archive receiver.

## Storage and cleanup

The SSD evidence tree is rebuildable from the recorded cache/scorer/implementation hashes and exact
argv. Per-frame atomic stages are the resume surface. No transient virtualenv, decoded PNG tree,
profiler trace, copied archive, or local bulk scratch was created; therefore no destructive cleanup
or cold-store move is authorized or needed.

