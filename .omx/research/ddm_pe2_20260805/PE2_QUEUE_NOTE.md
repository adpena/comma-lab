# PE2 n600 Scorer Queue Note

Status: **QUEUED-WITH-FIRE-ORDER / NOT RUN BY PE2**.

Fire condition: MAIN confirms `sq2 [SCORER]` has released the single scorer slot, then claims one scorer-slot batch for the three candidate rows below.

Exact fire command from repo root: `bash /Users/adpena/Projects/pact/.omx/research/ddm_pe2_20260805/stage_pe2_three_candidate_scorer_batch.sh cpu`

Axis warning: running the command on this Mac is `[macOS-CPU advisory]`; contest authority still requires the contest-CPU or contest-CUDA host.

| candidate | receiver-copy archive bytes | receiver-copy archive sha256 | submission dir |
|---|---:|---|---|
| PE1 full explicit_curve_k8 | `478612` | `51e2e5b78d2c83b3cb357206c2e3a006ba3a51e2ba5661fcee364c39762a0416` | `/Volumes/VertigoDataTier/pact/ddm_pe2_20260805/sub_auto_pairbit_pe2_pe1_full_explicit_curve_k8_receiver` |
| PE1 surgical generator-pair waterfill 75kb | `425627` | `90de4c14887156fe462ead76163303bacefc3bbffec54bea10296ea104ff929a` | `/Volumes/VertigoDataTier/pact/ddm_pe2_20260805/sub_auto_pairbit_pe2_pe1_surgical_generator_pair_waterfill_75kb_receiver` |
| BF1 lane-crop r3 | `563256` | `4741bfc91e3c013ea63435edd578933ef346540e998b71c368ff88f0b7bfa13a` | `/Volumes/VertigoDataTier/pact/ddm_pe2_20260805/sub_auto_pairbit_pe2_bf1_lane_crop_r3_receiver` |

Batch contract:

- One fire handles PE1 full, PE1 surgical, and BF1 in sequence.
- All three candidates are qo1-base IX2 archives with one optional receiver-consumed section family.
- The staged script uses the canonical byteclose/evaluate wrapper per exact archive; this preserves exact-archive semantics.
- No scorer was run while PE2 generated this note.
