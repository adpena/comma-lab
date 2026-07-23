---
schema: ddm_m1_missed_optimum_dag_feed.v1
date_utc: 2026-07-23
lane_id: lane_ddm_m1_missed_optimum_hunt_20260723
research_only: true
execution_allowed: false
score_claim: false
verdict: TWO_GENUINELY_OPEN_REPRESENTATION_RACES_FOUND
verdict_scope: "DESIGN x current DDM vehicle and finite cited forms; no family, launch, score, or promotion verdict"
pointer: "0.1910828242 [contest-CPU Linux x86_64]"
pointer_moved: false
main_landing_review_required: true
---

# DDM M1 missed-optimum DAG feed

## Pointer delta

`0.1910828242 [contest-CPU Linux x86_64] -> unchanged`.

No build, measurement, archive, launch, dispatch, exact eval, score claim, or promotion occurred.

## Finding routed to the live DAG

```text
current exact v19b master
  137,825 B / d_seg .026594424778 / d_pose 163.061176604795
  3,000,367 errors above target
  |
  +-> existing live path (do not duplicate)
  |     v19b joint correction
  |       -> #366/J5 trunk descent
  |         -> #604 complete-description minimizer
  |           -> c1 common-master waterfill
  |
  +-> OPEN RACE M1-A: kinetic anisotropic Laguerre cell complex
  |     Stage A: target labels -> one evolving regular triangulation
  |       gate: <=136,839 errors AND <=100,099 complete bytes
  |     Stage B: scorer-free RGB pullback -> uint8 -> R -> Seg/Pose
  |       gate: <=200,000 B, d_seg<=.001159998576, d_pose<=.00161
  |     fail: scoped formulation close; retain broader generator family
  |
  +-> OPEN RACE M1-B: conditional frame-0 Pose preimage
        freeze frame 1 bytes/cells exactly
          -> reuse xi only under chart-selection custody
            -> luma and joint-YUV6 finite bases
              gate: d_pose<=.00161 in <=7,195 complete bytes
        fail: scoped finite-basis close; retain broader conditional generator
```

## Coverage decisions

- **Power diagram:** literal few-site/global Euclidean and packet-only forms remain closed; the
  kinetic, spatially explicit, shared-edge formulation is new and open.
- **Two-frame factorization:** source structure is settled; current-vehicle minimal frame-0 rate
  is not measured. R1 is a comparator, not a transferable component.
- **Curvelet:** target-boundary exact byte race remains open but already belongs in c1/A1 column
  generation; no duplicate arm.
- **BEV:** v2 G1-PoseNet chart is far-field confounded; true-trajectory v3 family stays open, with
  <=9.6841% Lane ceiling. It is not the top pending probe.
- **Global n600 solve:** #604/#366 cover it. The missing end-to-end bridge is execution debt, not a
  newly discovered representation.
- **Interaction:** v19b's measured positive non-additivity is a binding admission rule. Future
  components are replayed jointly and never summed from isolated receipts.

## Triality

- DSL/data:
  `.omx/research/configs/ddm_m1_kinetic_laguerre_at_tolerance_probe_20260723.json` and
  `.omx/research/configs/ddm_m1_conditional_frame0_pose_preimage_probe_20260723.json`.
- DAG: this feed.
- Equations:
  `.omx/research/ddm_m1_missed_optimum_hunt_canonical_equations_20260723.md`.
- Findings:
  `.omx/research/ddm_m1_missed_optimum_hunt_20260723_codex.md`.

## MAIN merge-boundary review

Re-derive source incidence, receipt hashes, target/residual arithmetic, and all rate scopes. Ensure
the later v19b artifacts are present before this feed is accepted. Preserve the probe configs as
`execution_allowed=false`; neither absent runner may be treated as built, runnable-now, or fired.
