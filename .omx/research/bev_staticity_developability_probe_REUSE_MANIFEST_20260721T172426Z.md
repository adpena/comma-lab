# REUSE MANIFEST — BEV staticity / developability probe

| required surface | reused implementation / artifact | disposition |
|---|---|---|
| #327 horizon + camera | `tac.clip_profile`, #327 audit; `v_h=174`, `cam_h=1.22` | reused, value-custodied |
| openpilot IPM | `tac.boundary_math.lane_sdf_component.image_to_ground` | reused with reconciled arguments |
| SE(3) convention/composition | `tac.lie._se3_numpy` | reused, translation-first |
| solved ξ(t) | `predict_project_receiver.counted_planar_xi_series` over the solved PPCS seed | reused; no probe-side fit |
| #549 solved boundary cells | hash-pinned `gt_n600.npz[lstars,margins]` | reused via ZIP_STORED mmap |
| self-detected class strata | `tac.clip_profile.detect_class_order` | reused; canonical map verified |
| subpixel/Fisher orientation | `separatrix_asymmetry_t_subpixel_boundary_localizer_v1`; `t=M_p/(M_p+M_q)` | reused equation; shallow side only |
| ξ B-spline estimate | `fit_se3_bspline_controls`, `bspline_fit_error_curve` | wired but correctly not executed after C1 failure |
| symmetric boundary helper | `segnet_boundary_marginals.boundary_mask_from_labels` | inspected, not used: loses side orientation and subpixel t |
| raw Gaussian K | no reusable stable discrete estimator accepted | refused per C3; ruling residual substituted |

## New-code failed-search justification

Searches across `src/`, `tools/`, and the equation registry found G1 projective transport,
lane-IPM fitting, symmetric boundary masks, and proxy ξ measurements, but no existing instrument that
simultaneously:

1. consumes the frozen solved PPCS ξ without re-estimation;
2. extracts per-pair **oriented shallow-side** subpixel boundaries;
3. applies the C1 hood gate before any Road/Lane interpretation;
4. reports n64 directional and n600 load-bearing segmented ruling residuals; and
5. fail-closes D3 matched-distortion bytes on a control failure.

`tools/measure_bev_staticity_developability.py` is therefore the minimum new composition.  It does
not fork an evaluator, scorer, IPM, Lie algebra, ξ estimator, class detector, B-spline helper, DSL,
or launcher.
