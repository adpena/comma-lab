Implemented, measured, retained, reviewed twice, and committed as `fb732e7579`.

Artifacts: [evidence memo](/Users/adpena/Projects/pact/.omx/research/ddm_ec1_event_coordinate_producer_20260812.md), [producer](/Users/adpena/Projects/pact/experiments/ddm_ec1_event_coordinate_producer.py), and [tests](/Users/adpena/Projects/pact/experiments/tests/test_ddm_ec1_event_coordinate_producer.py).

Verdict: sparse CP135→HY1 events are viable; complete adjacent-frame temporal event coding loses to intra on both matched n600 objects.

### Event-vs-intra pricing per class

These are exact typed-coordinate streams. Each class raced Brotli q11, raw LZMA1, and SMEVR. Ratios use TF1’s complete 356,636 B intra denominator.

| Object | Class | Sites | Winner | Bytes | × intra |
|---|---|---:|---|---:|---:|
| CP135→HY1 | Road | 12,245 | LZMA1 | 27,074 | 0.076 |
| CP135→HY1 | Lane | 3,927 | Brotli | 11,027 | 0.031 |
| CP135→HY1 | Undrivable | 4,807 | LZMA1 | 10,872 | 0.030 |
| CP135→HY1 | Movable | 2,476 | LZMA1 | 6,920 | 0.019 |
| CP135→HY1 | MyCar | 3,896 | LZMA1 | 8,332 | 0.023 |
| CP135 temporal | Road | 714,007 | LZMA1 | 366,359 | 1.027 |
| CP135 temporal | Lane | 412,271 | LZMA1 | 342,554 | 0.961 |
| CP135 temporal | Undrivable | 265,574 | LZMA1 | 94,731 | 0.266 |
| CP135 temporal | Movable | 117,243 | LZMA1 | 72,940 | 0.205 |
| CP135 temporal | MyCar | 154,889 | Brotli | 31,645 | 0.089 |
| HY1 temporal | Road | 714,487 | LZMA1 | 366,549 | 1.028 |
| HY1 temporal | Lane | 412,271 | LZMA1 | 342,801 | 0.961 |
| HY1 temporal | Undrivable | 265,917 | LZMA1 | 94,935 | 0.266 |
| HY1 temporal | Movable | 117,427 | LZMA1 | 73,153 | 0.205 |
| HY1 temporal | MyCar | 154,961 | Brotli | 31,609 | 0.089 |

The optimal-form reused SP1 curve/event containers were:

| Object | Bytes | × intra | × XOR | F1 |
|---|---:|---:|---:|---|
| CP135→HY1 | 44,410 | 0.125 | 0.098 | Not fired |
| CP135 temporal | 633,441 | 1.776 | 1.397 | Fired, INSTANCE |
| HY1 temporal | 633,606 | 1.777 | 1.397 | Fired, INSTANCE |

SP1 improved the temporal raster-coordinate baseline from roughly 909 KB to 633 KB, but fragmentation still leaves it well above intra.

### Receiver effectiveness

Measured through the shipped CP135 semantic receiver → camera resize → uint8 → scorer-lattice resize, without invoking SegNet or PoseNet:

- Inactive replay: 32/32 stratified pairs byte-identical.
- Active C1 layer: 32/32 pairs changed receiver output.
- Proposals: 200 attempts, 200 receiver-effective.
- Custody: 200 unique proposal IDs and payload hashes; all consumer artifacts reverified against size and SHA-256.
- Mix: 151 boundary-offset, 48 lane-program, and 1 island-death proposal.
- F2: not fired.
- F3: not fired.
- Acceptance and score improvement: not measured.

Proposal store:

`/Volumes/VertigoDataTier/pact/ddm_js5_20260812/authoritative_seeded/follow_on/realized_acceptance_200`

No candidate archive or exact evaluation was run. Effective frontier remains CP135 at `0.16195513827824176 @ 186,252 B [contest-CUDA T4, n600]`.

## NEXT_IF_RESUMED

- **Action:** JS5 realized acceptance over EC1 event-coordinate proposals. **Disposition:** QUEUED-WITH-A-FIRE-ORDER. **Owner:** MAIN training-leg router. **Consumer store:** `/Volumes/VertigoDataTier/pact/ddm_js5_20260812/authoritative_seeded/follow_on/realized_acceptance_200`. **Fire trigger:** MAIN owns the training leg, observes the sole n600 scorer slot free, verifies the EC1 store state and source archive SHA, then runs the existing JS5 pose-gated robust-improvement acceptance loop without regenerating proposal payloads.
- **Action:** HY1 terminal whole-container replacement. **Disposition:** QUEUED-WITH-A-FIRE-ORDER. **Owner:** HY1/js1 whole-container builder. **Consumer store:** `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/hy1_solved_carriage/`. **Fire trigger:** ps135 emits its terminal same-parent pose carrier; replace the stale T1R1 carrier, reuse the EC1 exact base-to-HY1 coordinate receipt for proposal ordering, rebuild the complete archive, and prove independent decode before any scorer request.
- **Action:** local event-coordinate prior in HPAC. **Disposition:** QUEUED-WITH-A-FIRE-ORDER. **Owner:** ddm_cl1_capacity MAIN executor/harvester. **Consumer store:** `/Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/`. **Fire trigger:** the existing CL1 lambda-1 twin equality and decode gates pass; condition only on the retained EC1 local event coordinates, race the complete model-plus-token package against the no-context control, and retain both payloads.

## LIVE-HYPOTHESES

- Sparse CP135→HY1 events may contain joint-negative steps: all 200 tested edits survive the receiver, and their target came from task descent. Scorer acceptance remains untested.
- Sparse event coordinates may help HPAC conditioning even though standalone global temporal coding loses, because the model can selectively exploit predictable events.
- A terminal same-parent pose carrier could make the solved C1 composition scoreable without T1R1’s stale-carrier confound.

## DEAD-ENDS

- Full adjacent-frame raster-coordinate events: closed at `INSTANCE ×2`; exact containers are 908,293–909,111 B.
- Full adjacent-frame SP1 curve/event coding: closed at `INSTANCE ×2`; 633,441–633,606 B is still about 1.78× intra.
- Receiver effectiveness as a substitute for score acceptance: closed; changed uint8/scorer-lattice input proves actuator survival only.
- EC1 frontier movement: none. Own-vehicle frontier remains LC2 `S=0.16959899569230852 @ 187,226 B [contest-CUDA T4, n600]`.