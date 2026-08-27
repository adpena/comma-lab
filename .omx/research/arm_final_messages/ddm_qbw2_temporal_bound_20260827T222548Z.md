Step 2 did not fire. The best gate-eligible full-n600 result was **188,860 B**, far above both the 84,910-B allowance and 68,000-B schema-v2 fire bar.

| Leg | n32 projection | Full n600 | Status |
|---|---:|---:|---|
| Joint temporal context | 212,081 B | **188,860 B** | Eligible; best result |
| GB1 carrier-state translation | 712,998 B | 635,029 B | Ineligible QA39 proxy |
| Road topology + exceptions | 482,949 B | 455,936 B | Eligible |

Joint context removed 43.84% from QBW1’s prior 336,286-B projection, but still missed the allowance by 103,950 B. Road topology determines 93.2331% of interfaces, yet its exact representation remains much too large.

The strict QA39 leg was not faked: GB1 carries 12-dimensional pose-carrier coefficients, not geometric Pose6. The retained translation proxy is explicitly excluded from the gate.

Artifacts and verification:

- Verdict: [ddm_qbw2_temporal_bound_verdict_20260827.md](/Users/adpena/Projects/pact/.omx/research/ddm_qbw2_temporal_bound_verdict_20260827.md)
- Custody: `/Volumes/APDataStore/pact/ddm_qbw1_boundary_event_quotient/qbw2_temporal_bound/`
- 824 retained files, 70,657,919 logical bytes.
- Replayed 102 coder receipts and verified all 612 primary/repeat payloads.
- Five focused tests passed; payload-retention findings: zero.
- Two named review-tracker passes completed for both Python files.
- Serializer commit: `fbf582b09652aabfa2b67e56602c4afb89296b69`.
- Existing unrelated AU1, lane-registry, and WD3 changes were preserved and excluded.

No scorer, Metal, Modal, RGB realization, distortion, or exact evaluation ran. The result closes only the measured explicit representations on current GB1.

Own-vehicle frontier: **gb1 — S 0.14811799921260607 @ 180,215 B [contest-CUDA T4 n600]**. UNMOVED.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER — QBFLOW scorer-free first rung.** Owner: `MAIN object-family router`. Consumer store: `/Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow/`. Fire trigger: consume this closure, confirm no duplicate lane, and preregister a retained n32 packet whose counted weights, latents, and headers project the complete archive to at most 137,986 B; no training or scorer dispatch before that serialized-rate gate clears.

## LIVE-HYPOTHESES

- Continuous implicit boundary flow remains plausible because decoder-generated interfaces avoid serializing the explicit crack and mask mass that dominates all three measured streams.
- Genuine decoder-owned geometric transport remains open because pinned GB1 lacks Pose6 and therefore could not test the prescribed QA39 homography.
- Road topology may still help as free generator conditioning because 93.2331% of interfaces touch Road, even though serializing the exact topology is too expensive.

## DEAD-ENDS

- QBW1 schema v2 on this bound: closed because 188,860 B exceeds the 68,000-B fire bar by 120,860 B.
- Another context-coder, header, or dictionary pass on the same explicit object: closed because the measured 43.84% context cut still misses allowance by 103,950 B.
- Exact Road mask plus categorical exceptions: closed at 455,936 B.
- Relabelling GB1’s 12-D carrier coefficients as Pose6: forbidden because they are not geometric pose.
- Summing credits from the three overlapping legs: closed because they are substitutes, not additive savings.