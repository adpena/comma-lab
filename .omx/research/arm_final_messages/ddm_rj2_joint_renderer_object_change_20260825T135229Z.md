RJ2 is complete and committed as `3c51b038a2`. The mechanism passed every required gate, but the bounded candidate was refused.

Measured `[macOS-CPU advisory n1; engineering smoke]`:

- Archive: 179,274 B, shedding 1,094 B from DX2.
- Primary/repeat SHA-256: `82bda77dd2eb582a21fa607c0473c636857f040d603b493b6c88a15c947af12a`.
- Exact DX2 scope: S `0.17024397523303644`.
- RJ2 final scope: S `0.3530185565980673`.
- Delta: `+0.18277458136503086 S`; therefore no admission or pointer move.
- Compensation improved d_pose from `0.0061619673` to `0.0033970401`, recovering 45.07% of the d_pose gap, short of the predicted majority.
- Candidate remains 41,288 B above the 137,986-byte current-distortion cap.

All required mechanisms passed: both frozen scorers, MF1 training input, EMA/stage checkpoints, final-object compensation, full carrier re-solve/re-encode, real CAP1→DX2→RR5→Brotli coding, receiver parseback, and matching primary/repeat archives. No Metal, Modal, n600 scorer, full inflate, or contest evaluator ran.

QS2 was re-priced and fired at n1. RE1, EC1, and LD1 were re-priced but remain queued on the future moved-field object. AE1, OE1, and HPAC rows remain unchanged and closed.

Artifacts:

- [Handoff memo](/Users/adpena/Projects/pact/.omx/research/ddm_rj2_joint_renderer_object_change_20260823.md)
- [Implementation](/Users/adpena/Projects/pact/experiments/ddm_rj2_joint_renderer_object_change.py)
- [Tests](/Users/adpena/Projects/pact/src/tac/tests/test_ddm_rj2_joint_renderer_object_change.py)
- [Final retained receipt](/Volumes/APDataStore/pact/ddm_rj2_joint_renderer_object_change/reviewed_replay_r1/SMOKE_RESULT.json)

Verification: 84 focused tests passed, Ruff passed, two 41-entity review-tracker passes completed, and the serializer committed only the three owned files. The wider developer preflight still has unrelated pre-existing findings, enumerated in the memo. Only regenerable bytecode/AppleDouble residue was removed; scientific payloads were preserved and `upstream/` was not altered.

Own-vehicle frontier remains **S 0.14811799921260607 @ 180,215 B `[contest-CUDA T4, n600]`**, GB1 SHA `ba1f3830cd51b820d7f9b834a1dcc12e8776a0260f9da57a4e8e0944b988e3a4`.

## NEXT_IF_RESUMED

- **Disposition:** `QUEUED-WITH-A-FIRE-ORDER / S1-STAGE-A-ADAPTER`; **owner:** MAIN-designated WD3/S1 implementer; **consumer store:** `/Volumes/APDataStore/pact/ddm_s1_trained_renderer_diagonal/stage_a/`; **fire trigger:** coherent RJ1 custody plus reviewed RJ2 adapters ported to the exact GB1 body with both registered seeds.
- **Disposition:** `QUEUED-BEHIND-STAGE-A / MOVED-FIELD-AND-JG2`; **owner:** MAIN-designated moved-field producer; **consumer store:** `/Volumes/APDataStore/pact/ddm_s1_trained_renderer_diagonal/stage_b/`; **fire trigger:** stage A retains an n600 moved renderer/runtime and receiver-consumed moved token field.
- **Disposition:** `QUEUED-BEHIND-STAGE-B / EXACT-OBJECT-COMPENSATION`; **owner:** MAIN-designated QS5/RJ2 implementer; **consumer store:** `/Volumes/APDataStore/pact/ddm_s1_trained_renderer_diagonal/stage_c/`; **fire trigger:** stage B fingerprints the final archive, runtime, realized field, and Pose6 targets.
- **Disposition:** `QUEUED-BEHIND-RECEIVER-AND-BYTE-GATES`; **owner:** MAIN scorer-lane router; **consumer store:** `/Volumes/APDataStore/pact/ddm_s1_trained_renderer_diagonal/admission/`; **fire trigger:** stages A–C are receiver-closed, repeat-identical, payload-verified, and exact arithmetic indicates a plausible negative composed delta.

## LIVE-HYPOTHESES

- Longer multi-pair W96 training may recover more pose because this smoke allowed one update, while compensation alone materially reduced d_pose.
- A genuinely moved token field may reveal conditional coding structure absent from the unchanged DX2 field.
- RJ2’s initializer and compensation adapters may close S1’s missing GB1 interfaces, provided every container binding is re-proved.
- RE1, EC1, and LD1 may price differently on a receiver-consumed moved field because their old scorer effects belong to the old renderer object.

## DEAD-ENDS

- Generic CAP1 predictor refitting changed the production object despite decoding correctly.
- Encoding RR5 before DX2 used the inverse pipeline order.
- Brotli q11 and q9/lgwin24 failed exact stream identity; DX2 requires q9/lgwin16.
- Reusing a runtime copy containing the source archive is incompatible with immutable sealing.
- The 1,094-byte renderer saving is not an admission: its rate credit is overwhelmed by measured joint damage.
- AE1, OE1, and HPAC perturbations are not reopened by a renderer-only move because their field and probability object did not change.