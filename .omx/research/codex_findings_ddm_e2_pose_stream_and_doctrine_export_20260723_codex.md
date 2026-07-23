# Codex findings — DDM E2 pose stream and doctrine export — 2026-07-23

`research_only=true` · `score_claim=false` · pointer unchanged · MAIN landing review required.

## Verdict

**PASS exporter/doctrine and sensitivity apparatus; BLOCKED compact pose closure.** The pose stream is absent from the counted packet, not counted-but-inert. E2 removes duplicated semantic paint from frame 0 and improves the official-harness-local advisory score by `0.05557250`, but `d_pose=162.58094788` remains outside the tube because the 3,600-byte Pose6 target has no compact code-to-photometry inverse.

## Receiver-closed measurements

| Row | FIRST-RUNG | Bytes | d_seg | d_pose | Seg term | Pose term | Rate term | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E1 official harness local advisory | yes | 339,094 | 0.02861482 | 163.05291748 | 2.86148200 | 40.37981148 | 0.22578878 | 43.46708225 |
| E2 official harness local advisory | yes | 343,466 | 0.02861482 | 162.58094788 | 2.86148200 | 40.32132784 | 0.22869991 | 43.41150975 |
| E2 independent frozen scorer | yes | 343,466 | 0.028614807129 | 162.580958694146 | 2.86148071 | 40.32132918 | 0.22869991 | 43.41150981 |

E2 adds 4,372 bytes and gains `0.05557250 S`, or `1.27110018e-5 S/byte`: 19.09× the `25/37,545,489` rate dual. This is a local FIRST-RUNG positive, not a score claim or promotion.

At the C1 waterfilled 160,000-byte ceiling, the rate term alone is `0.10653743`, leaving only `0.08454539` for both distortion terms under pointer `0.1910828242`. E2's zero-pose floor is already `3.09018191`; no nonnegative byte budget closes this formulation.

## Root cause and #417

- Counted packet members: manifest, chart, semantic. Pose member: none.
- The nested `uint8[600,6]` Pose6 object is consumed only before export.
- Counted pose→output: not applicable at zero pose bytes.
- Output pose effect→owner: fail; compact inverse owner is absent.

Verdict scope: current E1/E2 compact packet boundary only. The exact 409,526,925-byte lattice control keeps the DDM/exact-solve family open.

## Four-clause export doctrine

Every counted description stream carries the scorer-visibility, sensitivity-priced-tolerance, and descriptive→compact→coder audit triple. The exporter and receiver both fail closed on a missing/malformed row or incomplete ordered-pair matrix. Facts have one dimension home, frame 0 is structurally seg-free, and corrections own zero bytes.

| Ordered row | FIRST-RUNG | Standalone B | Conditioned B | Redundancy B |
|---|---:|---:|---:|---:|
| chart → semantic | yes | 315,033 | 318,169 | -3,136 |
| semantic → chart | yes | 18,412 | 18,461 | -49 |

Both Brotli-Q11 contexts hurt; there is no measured cross-stream gain to deduct. Verdict scope: these two stream formulations/coder only. Candidate admission remains false because per-coordinate chart tolerances and semantic boundary tolerances are not measured.

The alleged E1 “remaining 177 KB” is an accounting error: `339094-134211-28108` subtracts an input state archive and free runtime source from the counted packet even though neither is a packet member. E1's counted semantic member alone occupied 315,153 home bytes.

## Typed perturb → realize → score API

`DDMRuntimePerturbationV1` now reopens the strict packet, edits one chart or semantic coordinate, serializes the actual `.ddb`/`.dds` member, parses it back, realizes RGB, and measures batch-16 SegNet/PoseNet deltas and measured byte deltas. It fails on receiver-inert edits and binds each output effect to one owner stream.

A real E2 semantic smoke changed one scorer-cell code, grew `composed.dds` from 315,102 to 315,108 bytes, and changed 27 camera RGB values. This closes the apparatus gap for g2/p581r costate rows; it does not supply their candidate measurements.

The official-harness Seg realization gap is `0.02861482 - 0.027470296224 = 0.001144523776`, so the previously `GS=0` `J_paint` rank-1 duty now has a measured receiver-realization signal.

## Verification

- Standalone exact raw: 3,662,409,600 bytes, SHA-256 `4871b1c19074041e56294093cacce6a6df9875e6215ab7802aec21252b0867c7`; 38 preserved stage checkpoints; 222.31 seconds total.
- Frozen upstream harness: PASS on local CPU advisory axis; archive SHA-256 `8891012e4019e474d1e8ae7578104d74f27c25838c7b68a3798af35853469819`.
- Ruff: PASS.
- Focused tests: 14 passed. Full nearby DDM suite: 72 passed in 81.07 seconds.

Canonical machine receipt: `.omx/research/ddm_e2_pose_stream_and_doctrine_export_receipt.json`.
