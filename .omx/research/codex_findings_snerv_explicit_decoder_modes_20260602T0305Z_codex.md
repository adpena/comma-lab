# Codex Findings: SNeRV Explicit Decoder Modes

UTC: 2026-06-02T03:05Z

## Verdict

SNeRV mixed decoder precision is now explicitly assignable through the
receiver-visible payload grammar. This converts the previous magnitude heuristic
into a controllable target for scorer-loop/QAT search. It is not promotion
evidence.

## Landed Signal

- `encode_decoder_payload(..., mixed_modes=...)` accepts one explicit mode per
  decoder level/subband kernel when `codec="mixed_magnitude_symmetric"`.
- Legal modes remain the v3 receiver grammar modes:
  `zero`, `int2`, `int4`, `int8`, and `fp16`.
- The encoder fails closed when explicit modes are supplied to non-mixed codecs,
  when the mode count is wrong, or when an unknown mode appears.
- Advisory CLI now exposes `--decoder-payload-mixed-modes`.
- The payload header records `mode_assignment_source`, so artifacts distinguish
  `explicit` optimizer/scorer assignments from the baseline
  `magnitude_heuristic`.

## Smoke Evidence

Artifact:
`.omx/research/snerv_decoder_payload_explicit_modes_receiver_scored_1pair_smoke_20260602T030144Z.json`

Axis: `[macOS-CPU advisory]`, non-promotable.

Command shape:

```bash
tools/run_snerv_inverse_steg_advisory.py \
  --n-pairs 1 \
  --levels 1 \
  --decoder-payload-codec mixed_magnitude_symmetric \
  --decoder-payload-mixed-modes fp16,int4,int4
```

Key fields:

- `decoder_payload_header.schema=snerv_decoder_payload.v3`
- `decoder_payload_header.mode_assignment_source=explicit`
- `decoder_payload_header.mode_histogram={fp16: 1, int4: 2}`
- `decoder_payload_header.payload_bytes=34`
- `decoder_bytes=862`
- `receiver_archive_replay_verified=true`
- `receiver_archive_packet_bytes=456201`
- `d_pose_mean_linf=0.16987484693527222`
- `score_linf=2.9824514626437555`
- `beats_frontier_rate=false`

Interpretation: explicit mode assignment materially changes the detector
response while staying receiver-replayable. The LF payload still dominates rate,
and the 1-pair operating point is still far from promotion.

## Next Integration

- Build a local scorer-loop mode assignment probe over this explicit interface:
  candidate modes should be evaluated through receiver-decoded weights and
  PoseNet/SegNet deltas.
- Add QAT hooks that optimize toward the explicit mode plan rather than relying
  on magnitude thresholds.
- Keep every mode assignment artifact fail-closed with `score_claim=false` until
  full-600 receiver proof and paired contest CPU/CUDA pass.
