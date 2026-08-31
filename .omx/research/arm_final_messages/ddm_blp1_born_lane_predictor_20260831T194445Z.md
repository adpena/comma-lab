CLOSED-AT-FLOOR. The retained r10 born-Lane predictor costs **60,191 B** before any residual, so it cannot clear the D3B reopening trigger.

Measured: three real coders, exact decode, deterministic repeats, and receiver-dependency controls. Not measured: checkpoint inference, 32/568 residual race, scorer components, or score. Scope is this checkpoint instance—not the predictor family.

Artifacts: [verdict memo](/Users/adpena/Projects/pact/.omx/research/ddm_blp1_born_lane_predictor_20260831.md), [Stage 0 receipt](/Volumes/APDataStore/pact/ddm_blp1/stage0/STAGE0_RESULT.json). Committed as `8ce32946a680c9dc18c99819cc5c74e3abe83452`.

`[contest-CUDA T4 n600] own-vehicle frontier: AFR1 — S=0.14797617125559104 at 180,002 B; BLP1 did not move it.`

## NEXT_IF_RESUMED

- `QUEUED-WITH-A-FIRE-ORDER` — owner: `MAIN`; consumer store: `/Volumes/APDataStore/pact/ddm_blp1/`; fire trigger: ltg1 publishes its terminal real-coder receipt. Join both rows for the family-scope disposition; do not run BLP1 Stage 1 unless a new weight serialization first clears the trigger.

## LIVE-HYPOTHESES

- A shared or generically generated predictor may remain viable because this instance is dominated by learned-weight cost, not the estimated residual.
- The ltg1 topology/event sibling may avoid that weight tax with a materially different counted object.
- Held-out collapse remains untested and matters only if a future predictor first clears the weight floor.

## DEAD-ENDS

- This retained r10 coarse/flow/step predictor: overweight before residual.
- Charging only the final Lane matrix: omits receiver-consumed learned content.
- Running Stage 1 on this checkpoint: even a zero-byte residual cannot reverse Stage 0.
- Treating the entropy estimate as a coded residual or claiming a held-out ratio: neither was measured.

