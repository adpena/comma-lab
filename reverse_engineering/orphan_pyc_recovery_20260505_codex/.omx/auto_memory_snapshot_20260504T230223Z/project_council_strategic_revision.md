---
name: Council Strategic Revision (2026-04-10)
description: Rate error corrected, SegNet priority confirmed, revised experiment ranking, council bugs found
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Rate Analysis Correction
Council originally claimed rate was 43% wasted. WRONG — baseline also has video rate.
CNN adds only 0.031 points (46KB / 37.5MB). Filter saves 0.173 on distortion. Net +0.14.
But CRF sweep IS high-value: CRF 35 saves 0.038 points on rate (confirmed 791KB vs 847KB).

## Strategic Reversal: SegNet Matters More
- SegNet weight 100x is CORRECT per comma's product priorities
- SegNet has NO backup sensor in openpilot (PoseNet has IMU/wheel odometry)
- Our submission: PoseNet 5.6x better, SegNet 5.2% WORSE = wrong priority
- Each 0.001 SegNet improvement = 0.10 score points (vs 0.003 for PoseNet at current)

## Revised Experiment Priority (council consensus)
1. CRF 35/36 rate sweep (0.03-0.08 pts, low risk)
2. MRS-adaptive alone (proven_baseline + adaptive_rebalance=True)
3. SWA validation (free, already implemented)
4. Non-opposing gradient with static weight (test pcgrad alone)
5. Two-phase training (freeze conv1/2, fine-tune conv3)
6. Resolution sweep (520x390, 530x398)
7. Chained training (train double, deploy single)

## Council Bugs Found and Fixed
1. "pcgrad" missing from Literal type — FIXED
2. PCGrad projection formula wrong — FIXED (new non-opposing formula)
3. MRS weight unclamped — FIXED (clamped [10, 150])
4. PCGrad + accum_steps interaction — DOCUMENTED
5. Scale can zero SegNet entirely — FIXED (min scale 0.01)

## Karpathy's Rule
Do NOT combine multiple untested things. Test one at a time:
1. First: proven_baseline + adaptive_rebalance only
2. Second: proven_baseline + pcgrad only
3. Only if both help: combine

**Why:** Corrects strategic assumptions. SegNet recovery is now the priority.
**How to apply:** Always decompose score changes into 3 components. SegNet first.
