---
task: 603
feeds_task: 613
lane_id: lane_ddm_v5_route_fix_compose_603_613_20260722
research_only: true
triality_leg: DAG
main_landing_review_required: true
---

# FEED-603 -> 613: evaluator bridge and temporal-amortization DAG

```text
SHA-bound v5 composed archive + receipt
  -> exact v5 recompile equality
  -> semantic chart + structured S4 receiver sources
  -> candidate fanout with preserved Pose6
       exact control
       unit-root AR(1) hold, fixed keys every 24 pairs
       unit-root AR(1) hold, counted-Pose6/xi reverse-waterfilled keys
       safe-zero chart residuals + one-key dynamic S4 control
  -> existing entropy tournament + existing composed archive compiler
  -> exact final ZIP homes + parse/recompile equality + deterministic receiver replay
  -> batch16 bounded scorer bridge
       SegNet(last frame) argmax vs gt_n600.lstars
       PoseNet(official two-frame YUV6) first-six MSE vs gt_n600.gt_poses
  -> per-pair + target-class + topology + margin receipts
  -> n64/n256 cross-window marginal table
  -> Task #613 rate/efficacy gates
       marginal <=300 B/pair: PASS
       d_seg <=0.016 and <=0.00116: FAIL_FORMULATION_SCOPED
  -> immutable receipt/register row; no frontier promotion
```

## Readiness state

| node | state | evidence / reason |
|---|---|---|
| source boundary | PASS | v5/S4/target/cache hashes bound; SSD sources read-only |
| typed DSL | PASS | v6 config and compile hash; execution_allowed=false |
| receiver closure | PASS | exact v5 control equality, candidate compile x2, receiver replay |
| evaluator d_seg | PASS_LOCAL_ADVISORY | actual frozen-SegNet GT disagreement, all strata and pairs |
| evaluator d_pose | PASS_LOCAL_ADVISORY | official PoseNet YUV6 path; not payload-completeness proxy |
| static-once | PASS | Road PXQ1, Lane LBND2, MyCar hood have 0 B/added-pair |
| temporal marginal | PASS | 107.9375-110.2448 B/pair vs 1067.3177 control |
| absolute d_seg | BLOCKED_FORMULATION_SCOPED | best measured 0.038300534089, above 0.016 and 0.00116 |
| n600 | OWED | not run inside bounded arm |
| contest CPU/CUDA | REFUSE | not authorized; advisory rows cannot move pointer |

## Unified-stack wire-in

1. Sensitivity map: persist target-class, boundary/interior, margin-band, and ordered per-pair debt;
   Movable/Lane/Road boundaries dominate.
2. Pareto constraint: use exact archive bytes, actual `d_seg`, actual `d_pose`, and `score_claim=false`;
   a rate-only win cannot pass admission.
3. Bit allocator: feed the measured per-home marginal table and Fisher/margin strata to reverse
   waterfill; do not spend on already-low MyCar/Undrivable bulk.
4. Cathedral/autopilot: the amortized carrier is eligible only as a rate substrate for a new
   evaluator-debt actuator, never as a candidate archive.
5. Continual learning: append the scoped rate-green/efficacy-red row with both receipt hashes.
6. Probe disambiguator: fixed versus xi keying remains a measured two-mode interface; xi is slightly
   better on n256 `d_seg`, fixed is 33 bytes smaller.  Let the next actual evaluator anchor arbitrate.

Pointer honesty: `0.1910828242 [contest-CPU]`, unchanged.
