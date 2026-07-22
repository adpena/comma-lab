---
title: DAG FEED - stream-specific optimal coders for DDM v3
date_utc: 2026-07-22T04:47:13Z
task: 603
feeds_task: 613
lane_id: lane_truly_optimal_coder_survey_603_613_20260722
research_only: true
main_landing_review_required: true
---

# Trajectory

`SHA-bound S4/PPCS/Pose sources -> semantic stream extraction with legacy-alias warning -> source-shape statistics -> exact generic recodes + strict AQC1 parse-back -> exact Golomb/Elias/colex counts + KT ceilings -> per-stream argmin -> framed final-archive probe owed -> Task 603 v3 receipt -> Task 613 waterfill`

# Typed nodes

- `truly_optimal_coder_survey_measurement.v1`: source hashes, source-shape statistics, complete measured
  ladders, analytic-rate labels, winner assignments, budget diagnostics, and scoped negatives.
- `CodecAssignmentV1` successor interface: `(stream, transform, model, coder, parameters,
  complete_framing_bytes, decoded_sha256, final_archive_delta_bytes)`.
- `CodecSelectionTagV1`: deterministic tag used only when the complete framed minimum across two or
  more real encodes pays for its own tag.
- `RepresentationBlockerV1`: event/PPCS object whose best exact entropy backend cannot meet the cap;
  routes upstream to a changed semantic representation rather than another coder-only loop.

# Registerable assignments

| Stream | Preregistered v3 candidates | Current measured selector |
|---|---|---|
| static ground coefficients | `{brotli_q11}` | Brotli, 610 B on real PXQ1 mask payload |
| xi-curve knots | `{lzma1_raw_1MiB, brotli_q11}` | LZMA 204 B versus Brotli 205 B on compact real trajectory wire; exact final payload selects |
| Pose6 dxi residuals | `{raw, temporal_delta_rice(k)}` | Rice `k=6`, 3,509 B on real ordinal-code proxy; actual dxi remeasurement mandatory |
| sparse events | `{lzma1_raw_1MiB}` | settled LZMA, 181,904 B on decoded real PCE3 |
| entropy state | `{brotli_q11}` | Brotli, 1,086 B on real manifest |
| exceptions | `{brotli_q11_global, lzma1_raw_1MiB_global}` | global Brotli, 80,478 B; final archive delta and receiver proof owed |

# Unified solver wire-in

1. **Sensitivity map:** emit no distortion sensitivity from a lossless recode. Once exact final-ZIP
   deltas exist, attach `delta_archive_bytes` to the unchanged semantic candidate only.
2. **Pareto constraint:** admission requires byte-identical decoded semantics, fresh receiver success,
   Pose completeness, and strict final-archive reduction. A stream-local win alone is not Pareto
   evidence.
3. **Bit allocator:** reserve codec/model/tag/header bytes first. Route residual bit allocation only
   after the chosen lossless syntax has a complete byte price.
4. **Cathedral/autopilot:** enqueue the global exception recode as the cheapest local v3 probe; keep
   all execution, dispatch, score, promotion, and pointer gates closed until MAIN review.
5. **Continual learning:** register the per-stream winners and the `EVENT_REPRESENTATION_DOMINATES`
   blocker. Do not learn a global “Brotli wins” or “arithmetic wins” rule.
6. **Probe disambiguation:** xi and Pose explicitly ship two candidate interpretations and let exact
   final bytes arbitrate. The exception probe compares global Brotli and LZMA under the same framing.

# Blocker trajectory

- `FIXED_WIDTH_RATE_GRADIENT_WALL`: addressed at design level by variable-length candidates; no v3
  archive has yet proven a negative byte delta.
- `EXCEPTION_STREAM_BACKEND_UNKNOWN`: red to measured hypothesis (`99,718 B` isolated saving); remains
  red for exact final-ZIP delta and receiver closure.
- `EVENT_STREAM_WITHIN_STRICT_BOX`: hard red. Best measured event bytes `181,904 > 154,524` before all
  other streams and framing. This is a representation blocker, not evidence against LZMA or event
  coding as a family.
- `PPCS_SEMANTIC_REMAINDER_PRICED`: red. The 204-byte trajectory is not the whole `78,969 B` PPCS
  diagnostic and may not be substituted for it.
- `POSE6_REAL_DXI_RATE`: red. The current Rice row is a real Pose-derived ordinal proxy; actual v3
  signed dxi payload and receiver completeness remain owed.
- `TASK_613_CODER_ONLY_PATH`: formulation-negative on measured objects. Wider representation work
  remains open.

# Handoff condition

The live `mdl_member_solve_v3_entropy` arm should consume only the preregistered candidates and scoped
byte hypotheses above. It must report complete final-archive bytes, not add isolated savings, and it
must retain all prior stream semantics or name a representation change with fresh receiver evidence.

0.1910828242 [contest-CPU] — unchanged by construction.
