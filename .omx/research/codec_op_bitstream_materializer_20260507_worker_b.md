# CodecOp Bitstream Materializer - Worker B - 2026-05-07

## Scope

Owned write scope:

- `src/tac/codec_op_bitstream_materializer.py`
- `tools/materialize_codec_op_bitstream.py`
- `src/tac/tests/test_codec_op_bitstream_materializer.py`
- `.omx/research/codec_op_bitstream_materializer_20260507_worker_b.md`

This is a byte-custody bridge from CodecOp ADMM/search planning rows to a
concrete deterministic bitstream artifact. It is not an archive substitution,
score claim, promotion claim, or exact-eval dispatch packet.

## Implementation

Added `codec_op_bitstream_materializer.v1`, a deterministic envelope format:

- magic: `COBM1`
- prefix: magic plus 4-byte big-endian canonical-header length
- header: canonical JSON with candidate id, payload SHA-256, payload bytes,
  deterministic CodecOp params, and roundtrip status
- payload: the exact CodecOp-produced byte stream supplied inline, by hex/base64,
  by manifest path, or by explicit CLI `--payload`

The manifest records:

- charged blob path, bytes, SHA-256, header SHA-256, payload offset, and payload
  SHA-256
- codec magic and format details
- deterministic params from CodecOp/ADMM/search-style fields
- roundtrip/decode proof status
- archive identity presence
- source evidence semantics, CPU-only detection, exact-CUDA status
- fail-closed blockers and remaining exact-eval blockers

## Fail-Closed Policy

Hard failure before writing:

- payload bytes missing
- payload byte count or SHA-256 mismatch
- decode/roundtrip proof absent or failed
- malformed materialized envelope during internal parseback

Manifest blockers, with artifact still usable as planning/golden-vector
evidence:

- CPU-only source evidence
- missing exact CUDA auth eval
- missing source archive SHA-256/byte identity
- source-side score or dispatch claims are ignored and revalidated

All materializer outputs keep:

- `score_claim=false`
- `dispatchable=false`
- `ready_for_exact_eval_dispatch=false`
- `promotion_eligible=false`
- `score_affecting_payload_changed=false`
- `charged_bits_changed=false`

The blob can only become score-relevant after a reviewed archive-substitution
candidate consumes it and exact CUDA auth eval validates the resulting archive.

## Test Coverage

Focused tests cover:

- deterministic envelope and golden-vector manifest creation
- stable blob bytes across output paths
- missing payload fail-closed behavior
- failed decode/roundtrip rejection
- CPU-only plus missing archive identity blockers
- payload custody mismatch rejection
- CLI materialization from manifest custody plus explicit payload path

## Current Findings

`src/tac/codec_op_admm_adapter.py` keeps full-decode custody and payload SHA,
but deliberately does not retain raw blob bytes in planning rows. This
materializer therefore requires actual payload bytes again and verifies them
against any available `bytes_out`/`blob_sha256` custody fields rather than
trusting hash-only rows.

`tools/codec_op_cma_search.py` and `tools/codec_op_optuna_search.py` remain
CPU planning surfaces. Their rows can feed this materializer only when actual
payload bytes are supplied separately; the resulting materialized artifact
still carries CPU-only blockers and cannot claim score.

`src/tac/submission_packet_compiler.py` already has identity/golden-vector
packet discipline. The new materializer mirrors that posture at the CodecOp
payload layer, below full archive packet compilation.
