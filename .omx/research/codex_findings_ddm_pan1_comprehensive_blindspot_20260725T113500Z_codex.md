---
title: "Codex findings: PAN1 comprehensive Pantheon blind-spot hunt"
date_utc: "2026-07-25T11:35:00Z"
lane_id: "ddm_pan1_comprehensive_pantheon_blindspot"
research_only: true
execution_allowed: false
score_claim: false
promotion_eligible: false
verdict: "VETO_CURRENT_FRONTIER_SUCCESS_CRITERION_PROCEED_WITH_RESCOPED_EXACT_SCORE_GATE"
verdict_scope: "ARITHMETIC x declared targets x C1 post-CC3 bytes; no family, launch, promotion, or score verdict"
pointer_before: "0.19108282419209976 [contest-CPU]"
pointer_after: "0.19108282419209976 [contest-CPU]"
pointer_delta: 0
main_review_required: true
---

# Findings

The highest-priority blind spot is a falsified scoreboard premise. The #613 box
with the banked pose fallback cannot beat the pointer:

- pose contribution at d_pose .00161: `0.1268857754`;
- rate contribution at 130,789 B: `0.0870870266`;
- pose plus rate at d_seg=0: `0.2139728020`;
- #613 corner score: `0.3299728020`;
- pointer: `0.1910828242`.

The campaign's d_pose around 163 is not a vehicle-scaled unit. Source inspection
shows both the launcher and contest use mean squared error over 600 pairs x six
PoseNet outputs. The required 163.061-to-.00161 reduction is about 101,280x.

The next four findings are:

1. one clean step-1 n600 point cannot identify the 450-step decay law, and its
   constant extrapolation misses stage 1;
2. the 100,099-byte predictor is 76.5% of the post-CC3 plan, so representation
   replacement has much higher upside than another coder micro-race;
3. PC1 admission is not pose efficacy (`descent_was_run=false`);
4. no current DDM exact contest-CPU calibration closes advisory ordering.

## Durable outputs

- T3 council:
  `feedback_pantheon_comprehensive_blindspot_pan1_20260725.md`;
- exact arithmetic:
  `ddm_pan1_scoreboard_receipt_20260725.json`;
- twelve-row typed register:
  `ddm_pan1_ranked_blindspot_register_20260725.json`;
- routing:
  `ddm_pan1_comprehensive_blindspot_DAG_FEED_20260725.md`;
- online dedup:
  `papers_checked_pan1_minimum_description_video_20260725.md`;
- tracked anchor payload:
  `ddm_pan1_comprehensive_blindspot_council_anchor_receipt_20260725.json`.

## Own adversarial review

The review challenged three tempting overreaches:

- It rejected treating advisory-axis versus contest-axis as a pose unit
  conversion. Units are identical; hardware/implementation authority is not.
- It restricted the veto to success arithmetic. DDM, #366, W_seg-perp, PC1,
  and family-(d) remain open.
- It kept literature gains external. The online sweep only shapes same-object
  prototypes and predicts no contest delta.

No live surface was mutated, no launch or dispatch occurred, and no old-lineage
bytes or weights were composed. The 0.188044 and R1 rows are signal-only.

## MAIN review

MAIN must recompute the table, inspect launcher lines 709-741, adjudicate the
veto before attempt-6 fire, verify no route collides with ct1/j10/la1/ws4, and
append the anchor payload only after branch review. Terminal disposition must
name the FEED as findings consumer.
