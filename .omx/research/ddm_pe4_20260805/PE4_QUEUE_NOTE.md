# PE4 Fourth-Candidate Scorer Queue Note

Status: **QUEUED-WITH-FIRE-ORDER / NOT RUN BY PE4**.

Fire condition: MAIN harvests the active PE2 three-candidate batch, confirms the single scorer slot is free, then claims one follow-on scorer job for the PE3 75KB hybrid.

Exact fire command from repo root: `bash /Users/adpena/Projects/pact/.omx/research/ddm_pe4_20260805/stage_pe4_fourth_candidate_scorer_batch.sh cpu`

Axis warning: running the command on this Mac is `[macOS-CPU advisory]`; contest authority still requires the contest-CPU or contest-CUDA host.

| candidate | receiver-copy archive bytes | receiver-copy archive sha256 | submission dir |
|---|---:|---|---|
| PE3 hybrid 75KB | `432428` | `3f08c7fdd1c2746fa456ef8b6d8005e850d1a3acac5665a5d08b2ef17585b5e0` | `/Volumes/VertigoDataTier/pact/ddm_pe4_20260805_r2/sub_auto_pairbit_pe4_pe3_hybrid_75kb_receiver` |

Batch contract:

- This is the fourth candidate after PE2's PE1 full, PE1 surgical, and BF1 batch.
- The archive bytes are the PE3 75KB hybrid archive; the receiver copy uses the PE4 PE3EDGE1-consuming runtime.
- The staged script uses the canonical byteclose/evaluate wrapper per exact archive.
- No scorer was run while PE4 generated this note.
