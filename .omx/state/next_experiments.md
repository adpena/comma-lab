# next experiments

## promoted floor: 1.33 (dilated_h64, standard loss)

## Pre-Registered Three-Arm Experiment (Council Approved 2026-04-10)

### Gate Criterion (pre-registered, not to be moved)
At epoch 300-400: **seg < 0.00590 AND pose < 0.00250**
Early termination if trajectory clearly failing at epoch 300.

### Arm A: PCGrad gradient surgery
- **Profile**: `pareto_pcgrad`
- **Hypothesis**: Non-opposing gradient projection decouples PoseNet/SegNet
- **Platform**: Modal A10G
- **Counterfactual**: Arm B (same architecture, different loss)

### Arm B: Simple reweighting (control)
- **Profile**: `reweight_ablation` (seg_weight=200, standard loss)
- **Hypothesis**: SegNet regression is loss-driven, not architectural
- **Platform**: Kaggle P100
- **Purpose**: If B beats A, gradient surgery is unnecessary

### Arm C: Spatial gate architecture (Collier's proposal)
- **Profile**: `gated_dilated_smoke` (400 epochs)
- **Hypothesis**: Spatial allocation of correction capacity is the root cause
- **Platform**: Kaggle P100
- **Purpose**: If C beats A and B, problem is architectural

### Predictions (recorded for posterity)
- **Contrarian** (50%): PCGrad fails gate. B beats A. C beats B.
- **Collier**: Agrees with Contrarian's ordering.
- **Euler**: Worst case (all fail), rate-only gives 1.25-1.30, still #1.

### If ALL three fail the gate:
Pivot to rate-only optimization:
- CRF 35 + current postfilter (confirmed 6.6% file reduction = 0.038 pts)
- CRF 36 + current postfilter (estimated ~0.08 pts)
- Projected score: 1.25-1.30

## Parallel: Rate Optimization (orthogonal, runs on local M5 Max)
| Priority | Experiment | Platform | Status |
|----------|-----------|----------|--------|
| 1 | CRF 35 re-encode | Local | CRF 35 done: 791KB (6.6% savings) |
| 2 | CRF 36 re-encode | Local | Encoding (slow, SVT-AV1 preset 0) |

## Parallel: Writeup Artifacts
| Priority | Experiment | Platform | Status |
|----------|-----------|----------|--------|
| 3 | extreme_posenet | Kaggle | QUEUED |
| 3 | extreme_segnet | Kaggle | QUEUED |

## Decision Point: April 21
Lock top-2 candidates. No new experiments after this date.
Phase 4 (Apr 21-25): Authoritative eval of candidates.
Phase 5 (Apr 25-May 3): Packaging + paper.
