# Codex Findings: SNeRV Mixed Decoder Payload

UTC: 2026-06-02T03:03Z

## Verdict

Mixed per-kernel decoder precision is now a receiver-visible grammar, not a
post-hoc packing idea. It is useful for scorer-loop/QAT assignment, but the
current magnitude heuristic is not a rate win and is not promotion evidence.

## Landed Signal

- Added `snerv_decoder_payload.v3`.
- Added `mixed_magnitude_symmetric` decoder payload codec.
- Per-kernel modes are carried by packed 3-bit mode codes:
  `zero`, `int2`, `int4`, `int8`, and `fp16`.
- Int modes use symmetric per-kernel quantization plus fp16 scales.
- `fp16` mode stores the 3x3 kernel weights directly as little-endian float16.
- `zero` mode stores no scale and no q payload.
- Advisory JSON now includes the validated decoder payload header, preserving
  mode histograms and payload byte accounting without exposing raw packet bytes.

## Smoke Evidence

Artifact:
`.omx/research/snerv_decoder_payload_codec_mixed_header_receiver_scored_1pair_smoke_20260602T0300Z.json`

Axis: `[macOS-CPU advisory]`, non-promotable.

Key fields:

- `decoder_payload_codec=mixed_magnitude_symmetric`
- `decoder_payload_header.schema=snerv_decoder_payload.v3`
- `decoder_payload_header.mode_histogram={fp16: 2, int4: 1}`
- `decoder_payload_header.payload_bytes=45`
- `decoder_bytes=837`
- `receiver_archive_replay_verified=true`
- `receiver_archive_packet_bytes=456123`
- `beats_frontier_rate=false`

Interpretation: the mixed grammar replayed correctly and slightly improved the
1-pair pose advisory versus the prior int8 smoke, but the whole archive is still
rate-worse than the PR101 frontier at this operating point. This is a grammar
and instrumentation step, not a score step.

## Next Integration

- Replace the magnitude heuristic with scorer-loop or QAT assignment that can
  choose zero/int2/int4/int8/fp16 under PoseNet and SegNet guards.
- Keep mode histograms in every advisory artifact so decoder precision choices
  can feed allocator and rate-model updates.
- Do not promote SNeRV from this result. Promotion still requires full-600
  byte-closed receiver proof plus paired contest CPU/CUDA pass.
