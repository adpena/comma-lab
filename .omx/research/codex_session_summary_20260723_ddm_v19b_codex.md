---
title: Codex session summary - DDM v19b joint remeasurement
date_utc: 2026-07-23
lane_id: ddm_v19b_joint_remeasure_stack
tier: TIER-0
research_only: true
execution_allowed: false
score_claim: false
evidence_axis: "[macOS-CPU frozen-scorer advisory]"
main_landing_review_required: true
---

# Landed in this isolated branch

- typed v19b config and resumable common-master measurement runner;
- ten immutable per-candidate joint-remeasurement checkpoints and archives;
- n64 four-batch and n600 38-batch exact receiver/scorer ladders;
- per-move Lane+Movable versus residual attribution;
- c1 16,384-byte downstream budget handoff;
- atom-order gauge accounting at the current fixed-width coder surface;
- DSL/DAG/equation triality and adversarial findings memo.

# Result

`MULTI_MOVE_JOINT_STACK_ADMITTED_N600_ADVISORY`

Final n600: 137,825 B,
SHA `74ede4194ed0eb6c2716f1033a47569c87a7afdf0c21fb1ee23ac059beff3aae`,
`delta_d_seg=-0.0008758714460000011`,
`delta_d_pose=-0.00015067664799062186`,
`delta_S=-0.08501960537266746`, and 103,322 net Seg flips.

# Authority boundary

This is local macOS-CPU frozen-scorer advisory research. There was no paid
dispatch, remote/GPU launch, live-run actuation, contest score, promotion, or
pointer movement. MAIN review is mandatory before landing.

The lane is L2. The global lane validator's 110 missing legacy evidence paths
are pre-existing and do not include v19b; they remain separate repository debt.

# Next consumer

MAIN should review and land the commits, then feed the exact 137,825-byte state
and 3,137,206-error residual to c1 v18b and #366/J3. Their credits remain
sequential and non-additive.
