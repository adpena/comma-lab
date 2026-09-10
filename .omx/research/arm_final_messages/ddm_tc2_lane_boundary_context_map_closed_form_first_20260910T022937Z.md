Measured on **move37, all 600 frames**:

| Result | Gain |
|---|---:|
| GT-boundary reference, ideal codelength | 5,480.40 B |
| Minimal causal predictor, ideal codelength | 82.77 B |
| Actual twin encode, including new weights | **78 B** |

Both payloads match. Conditional parse-back reproduces all **117,964,800 tokens** exactly. The equation is registered and payloads retained. No archive or scorer ran; standalone public-receiver and cross-host proof remain unmeasured.

[Memo](/Users/adpena/Projects/pact/.omx/research/ddm_tc2_lane_boundary_context_map_20260910.md) · [Final handoff](/Users/adpena/Projects/pact/.omx/research/ddm_tc2_20260910/FINAL_HANDOFF.json)

Git writes were denied. Verified bundle **83645a01e395a9dd2a0a0a94dc8f36731bc3fc42** contains all 20 intended files; **main is unlanded**. The staged index is unchanged.

Live frontier: **S 0.13766931482209038 @ 180,186 B [contest-CUDA T4 n600]**. TC2 did not move it; sub-0.12 remains unmet.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER — MAIN:** build the better predictor. Consumer: [predictor order](/Volumes/VertigoDataTier/pact/ddm_tc2_lane_context_map/handoff/better_predictor/FIRE_ORDER.json). Trigger: harvest, deduplicate ownership, and pin the current field.
- **QUEUED-WITH-A-FIRE-ORDER — MAIN:** consume the tiny positive tail through the seal chain. Consumer: [seal intake](/Volumes/VertigoDataTier/pact/ddm_tc2_lane_context_map/handoff/seal_intake/FIRE_ORDER.json). Trigger: harvest; require current-field compatibility, public receiver proof, and positive actual archive pricing.
- **QUEUED-WITH-A-FIRE-ORDER — MAIN:** land the verified bundle and final receipts. Consumer: [landing handoff](/Users/adpena/Projects/pact/.omx/research/ddm_tc2_20260910/FINAL_HANDOFF.json). Trigger: Git-writable context and required commit checks.

## LIVE-HYPOTHESES

- Coherent lane/run tracking may recover continuity discarded by independent slots.
- Joint mixer calibration may help; the measured bound freezes the original weights. Both leads are folded into the predictor order.

## DEAD-ENDS

- This minimal predictor as a kilobyte-scale improvement: measured gain is only **78 B**.
- GT distance as a universal geometry ceiling: quantization and nonnested predictors invalidate that inference.
- Treating conditional token identity as public-receiver or exact-score proof: those paths were not executed.