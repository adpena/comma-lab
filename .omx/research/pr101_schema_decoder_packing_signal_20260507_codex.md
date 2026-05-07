# PR101 Schema-Driven Decoder Packing Signal

timestamp_utc: 2026-05-07T15:04:19Z

## Signal

PR101's engineering win should be treated as schema-driven decoder packing, not
generic Brotli luck. The key public-intake target is the decoder storage layout:

- `DECODER_STORAGE_ORDER`
- `DECODER_STREAM_ENDS = (1, 2, 22, 23, 26, 27, 28)`
- `DECODER_BYTE_MAPS`

The reported byte-level signal is about 8 KB saved on the decoder stream versus
PR106's flat Brotli decoder layout.

## Immediate Use

This should be the next HNeRV decoder-packing tranche after the HDM3 q-Brotli
split candidate. HDM3 proved a small fixed-schema q-stream/raw-scale section
win on the current PR106x low-level Brotli archive:

- source archive bytes: 186080
- candidate archive bytes: 186066
- decoder section: 170127 -> 170113
- raw decoder SHA-256 preserved:
  `f22eb6be56499fa5785f47f85d2bef7f71246f29674691fd3e06af733c8c0703`

The PR101-style target is larger: reconstruct the schema storage order and byte
maps, prove raw decoder equivalence, then build a runtime-adapter-backed archive
candidate. It remains planning/forensic until the submission runtime consumes
the schema-packed stream and exact CUDA custody passes.

## Required Next Artifacts

1. Extract PR101 decoder schema constants into a clean `tac` parser/encoder.
2. Produce a PR101-vs-PR106 decoder anatomy table:
   source section bytes, raw SHA-256, q-stream grouping, stream-end cuts, and
   byte-map transforms.
3. Build a deterministic PR106x schema-packed candidate stream.
4. Prove raw decoder equivalence against the current PR106x low-level Brotli
   archive and record old/new section SHA-256.
5. Emit a fail-closed archive candidate manifest with runtime-adapter blockers,
   as HDM3 now does.
6. Only after runtime parity and strict compliance, claim a lane and dispatch
   exact CUDA.

## Evidence Grade

evidence_grade: external_operator_signal + local_hdm3_byte_equivalence_context
score_claim: false
dispatch_attempted: false
ready_for_exact_eval_dispatch: false
