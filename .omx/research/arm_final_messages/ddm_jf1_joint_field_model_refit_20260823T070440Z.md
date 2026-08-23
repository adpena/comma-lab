The epoch-2 byte leg is complete and negative at the narrow supported scope.

- Shipped control reproduced exactly: 113,777 B, matching SHA-256 byte-for-byte.
- Null refit deficit: **+7,554 stream bytes**; model savings only 167 B; combined **+7,387 B**.
- Four receiver-closed diagonal rows all lost:

| Rung | Refit contribution | Model + stream delta |
|---|---:|---:|
| `k002500` | +6,634 B | +6,615 B |
| `k010000` | +7,103 B | +7,701 B |
| `k040000` | +4,542 B | +5,000 B |
| `k060000` | +2,861 B | **+2,715 B** |

Best early row is `k060000` at 130,007 B versus the 127,292 B baseline. Both preregistered byte predictions failed at epoch 2.

This is only `SCOPE_REDUCTION_EPOCH_2_OF_60 / BYTE-LEG ONLY`. Realized per-class d_seg, d_pose, and net ΔS were not measured because AP1 owns the exclusive n600 scorer lane. The scorer follow-on is durably queued.

Artifacts:

- [Research memo](/Users/adpena/Projects/pact/.omx/research/ddm_jf1_joint_field_model_refit_20260823.md), SHA-256 `66c60a18…652dd`
- [Aggregate byte receipt](/Users/adpena/Projects/pact/.omx/tmp/arm_receipts_local/ddm_jf1_joint_field_model_refit/BYTE_DIAGONAL_SCOPE_E0002.json), SHA-256 `ac48c600…cf15e0`
- [Scorer fire order](/Users/adpena/Projects/pact/.omx/tmp/arm_receipts_local/ddm_jf1_joint_field_model_refit/SCORER_FIRE_ORDER.json), SHA-256 `70e12f6b…d1d4b4`

All payloads were retained locally; no `/Volumes` writes or `upstream/` edits occurred. Verification passed: 21 tests and Ruff.

The seven sealed full-reference fits remain `running`; last verified at epoch 6 with immutable checkpoints. The memo is not committed: two serializer attempts failed at Git object insertion with `Operation not permitted`. The shared index remains empty. Implementation commits already landed as `7ceacbd126` and `b5c8edf63b`.

## NEXT_IF_RESUMED

- **FIRED** — owner: JF1 governed CPU launchers; consumer store: `.omx/tmp/arm_receipts_local/ddm_jf1_joint_field_model_refit/training/<tag>/`; fire trigger: automatic continuation until all seven epoch-60 checkpoints and terminal reports exist.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: JF1/MAIN byte custodian; consumer store: `.omx/tmp/arm_receipts_local/ddm_jf1_joint_field_model_refit/rows/`; fire trigger: all seven epoch-60 hashes validate and local storage preflight passes; then physically encode, pack, receiver-close, and finalize `BYTE_DIAGONAL.json`.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN exclusive scorer custodian; consumer store: `.omx/tmp/arm_receipts_local/ddm_jf1_joint_field_model_refit/scorer/`; fire trigger: AP1 terminal, explicit lane transfer, no active n600 scorer, terminal candidate hashes validated, and local storage preflight passed.
- **BLOCKED-LANDING** — owner: operator or future Git-authorized agent; consumer store: repository history; fire trigger: writable Git object store with memo SHA-256 still `66c60a18…652dd`, then rerun the serializer.

## LIVE-HYPOTHESES

- The full 60-epoch fit could reverse the early underfit deficit. This remains plausible because `k060000` is already 4,672 stream bytes below the epoch-2 null refit, even though both remain above their shipped-model controls.
- Scorer distortion could change the net ordering because the rungs alter Lane→Road labels. Rate-only losses do not determine net ΔS without DALI-aligned SegNet and PoseNet measurements.

## DEAD-ENDS

- An incumbent reproduction mismatch is closed: the shipped stream reproduced exactly.
- The epoch-2 prediction of more than 2,000 recovered bytes is closed: every measured refit contribution was positive.
- The epoch-2 prediction of a combined total below 127,292 B is closed: every measured total exceeded it.
- Trainer estimates as byte authority are closed: every reported row uses physical model packing, real RC64 bytes, and receiver identity.
- The missing-`PYTHONHASHSEED` launch and niceness wrapper are closed: both self-refused before producing training artifacts.
- Serializer landing in the current sandbox is closed after two identical Git object-store permission failures; the completed memo remains untracked.

Own-vehicle frontier: **dx2 — S 0.14821987563243377 @ 180,368 B [contest-CUDA T4, n600]**, unchanged.