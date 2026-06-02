# Codex Findings: SNeRV Decoder Payload Codec

UTC: 2026-06-02T02:52Z

## Verdict

Receiver-visible mixed-precision decoder payload grammar is useful and now wired,
but it is a rate/control primitive only. It does not make SNeRV promotion-ready,
and it must be evaluated through receiver-decoded weights because int8/int4/int2
decoder protection is lossy.

## Landed Signal

- Added `snerv_decoder_payload.v2` for symmetric per-kernel decoder quantization:
  `int8_symmetric`, `int4_symmetric`, and `int2_symmetric`.
- Reused the shared fixed-width integer packer in
  `tac.substrates._shared.int_stream_codec` instead of adding a duplicate bit
  packing path.
- Preserved legacy `snerv_decoder_payload.v1` / `float32_lzma` as the default
  codec.
- Wired `--decoder-payload-codec` through the SNeRV advisory CLI and report JSON.
- Vendored `tac.substrates._shared.int_stream_codec` into the generated SNeRV
  contest runtime package so archive-bound receiver proof is self-contained.

## False-Authority Finding And Fix

The first 1-pair int8 smoke exposed a real false-authority risk:

- Artifact:
  `.omx/research/snerv_decoder_payload_codec_int8_1pair_smoke_20260602T024507Z.json`
- Result:
  `receiver_archive_replay_verified=false`
- Error:
  `decoder_0_LH_mismatch`

Root cause: the advisory scored frames decoded with the training-time fp64/fp32
decoder while the archive carried quantized receiver decoder bytes.

Fix: `run_snerv_advisory` now encodes the decoder payload immediately after
fitting, decodes the receiver-visible payload back to a decoder, and uses that
receiver-decoded decoder for both L-inf and L2 advisory reconstructions. The
score path now measures the archive-consumable decoder, not an encoder-only
object.

Corrected smoke:

- Artifact:
  `.omx/research/snerv_decoder_payload_codec_int8_receiver_scored_1pair_smoke_20260602T0250Z.json`
- `decoder_payload_codec=int8_symmetric`
- `decoder_bytes=638`
- `receiver_archive_replay_verified=true`
- `receiver_archive_packet_bytes=455913`
- `beats_frontier_rate=false`
- Axis:
  `[macOS-CPU advisory]`, non-promotable

## Interpretation

The decoder codec is now a legitimate receiver grammar knob for scorer-loop or
QAT work. It is not enough on its own: the current 1-pair least-squares operating
point is still rate-worse than PR101 frontier bytes and remains blocked by
full-600 receiver proof plus paired contest CPU/CUDA eval.

## Next Integration

- Add scorer-loop or QAT training that chooses decoder precision under PoseNet
  and SegNet guards instead of post-hoc quantizing an LS decoder.
- Extend the codec portfolio from uniform whole-decoder selection toward
  per-kernel or per-component precision assignment only when the receiver header
  explicitly carries the grammar.
- Keep this rate primitive paired with archive receiver replay; any lossy decoder
  codec with replay mismatch is a blocker, not a score result.
