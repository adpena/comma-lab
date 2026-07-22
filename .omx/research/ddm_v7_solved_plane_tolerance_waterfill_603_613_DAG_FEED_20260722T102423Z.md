---
schema: dag_feed.v1
task: 603
feeds_task: 613
master_task: 578
lane_id: ddm_v7_solved_plane_tolerance_waterfill
research_only: true
execution_allowed: false
evidence_axis: "[macOS-CPU frozen-scorer advisory]"
main_landing_review_required: true
---

# FEED-603 -> 613: solved-plane tolerance waterfill DAG

```text
SHA-bound full-precision C1 target chunks (SSD read-only)
  + SHA-bound v6 fixed_ar1_hold24 predictor archive
  + v5 self-detected role permutation (never luma-sort)
  -> boundary-precedence partition
       Road, Lane, Undrivable, Movable, MyCar, Boundary
  -> seven per-stratum rungs
       exact, q4, q16, q64, fixed_hold24, xi_hold24, drop
  -> Brotli-Q11 versus LZMA-XZ-9e exact tournament per section
  -> nine typed composed policies
  -> receiver-closed archive
       no scorer weights
       no GT argmax table
       external stratum semantics
       x2 compiler + parse/re-encode + deterministic replay
  -> batch16 frozen SegNet + official-YUV6 PoseNet advisory bridge
  -> exact final-ZIP byte homes + class/topology/margin d_seg decomposition
  -> measured discrete Pareto envelope
  -> marginal Delta(100*d_seg+sqrt(10*d_pose))/Delta(bytes)
  -> stop at 25/37,545,489
  -> n64 + n256 exact-correction formulation falsifier
  -> immutable receipt/register row -> FEED-613
```

## Edge-state delta

| edge | state | evidence |
|---|---|---|
| solved-plane exact receiver | GREEN_LOCAL_ADVISORY | bit-identical target replay n64 and n256 |
| per-stratum tolerance ladder | GREEN_LOCAL_ADVISORY | 7 rungs x 6 sections, all atomic checkpoints preserved |
| joint Seg/Pose bridge | GREEN_LOCAL_ADVISORY | every candidate has frozen-SegNet d_seg and official-YUV6 PoseNet d_pose |
| `d_seg <= 0.00116` | GREEN_EXACT_ONLY_LOCAL_ADVISORY | n64 .000171423; n256 .000154535 |
| exact correction `<=200 KB` | RED_FORMULATION_SCOPE | n64 43,112,153 B; n256 171,332,654 B |
| useful constrained knee | RED | only feasible rows are 215.6x/856.7x the rate box |
| n600 | RED_TIME_BOUND | not run after n256 required 2,831.91 s and falsified rate by 856.7x |
| contest CPU/CUDA / score | REFUSE | unauthorized; no candidate archive or pointer move |

## Unified-stack wire-in

1. Sensitivity map: every candidate receipt carries target-class, boundary/interior, and margin-band
   d_seg. The rate binder (bulk Undrivable/MyCar/Road) stays separate from the evaluator binder
   (Lane/boundary).
2. Pareto constraint: Task #613 consumes only the measured byte-monotone envelope. Mixed policies
   dominated in both windows are excluded without deleting their receipts.
3. Bit allocator: exact ZIP homes and the first rate break are durable inputs. n64 stops before
   q16; n256 stops before xi_hold24. Neither stop point satisfies d_seg, so this formulation is not
   allocatable into the 200 KB box.
4. Cathedral/autopilot: no launch edge is emitted. A strict successor guard should refuse n600
   opaque site/value corrections and require a structured/learned description family.
5. Continual learning: cross-window receipt
   `64658a05a8975707f98db308223cefff78b5352975bb59cc2aa8a4ff2f8d50fb` is the empirical anchor;
   the append-only register draft leaves canonical 603 at 8/19 pending MAIN review.
6. Probe disambiguator: opaque corrections versus structured analytic/learned solved-plane carriers
   remain distinct formulations. The next probe must preserve both modes and arbitrate by exact
   receiver-closed bytes plus joint advisory evaluator debt.

Pointer `0.1910828242 [contest-CPU]` unchanged.
