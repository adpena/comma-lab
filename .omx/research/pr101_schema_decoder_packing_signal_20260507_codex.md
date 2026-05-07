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

## 2026-05-07T15:26:06Z Update - PR106x Schema Candidate And fp16 Probe

Implemented the exact-safe f32-scale PR101 schema fixture and a separate
planning-only fp16-scale probe:

- implementation: `src/tac/hnerv_pr101_schema_packer.py`
- CLI: `tools/build_hnerv_pr101_schema_candidate.py`
- tests: `src/tac/tests/test_hnerv_pr101_schema_packer.py`
- real manifest:
  `experiments/results/hnerv_pr101_schema_candidate_20260507_codex/manifest.with_fp16_probe.json`

Measured on
`experiments/results/hnerv_lowlevel_repack_pr106x_20260506_codex/pr106x_hnerv_brotli_repack_candidate.zip`:

- source archive bytes: 186080
- source decoder section bytes: 170127
- f32-scale raw-equivalent schema section bytes: 170091
- f32-scale archive bytes: 186044
- f32-scale archive SHA-256:
  `a927bd957b34d18a85230ac1a784331c6b6ad9b25bde7532d646c875b7548d6e`
- raw decoder SHA-256 preserved:
  `f22eb6be56499fa5785f47f85d2bef7f71246f29674691fd3e06af733c8c0703`

The PR101-native fp16-scale probe is deliberately not raw-equivalent:

- fp16 probe decoder section bytes: 170037
- fp16 probe archive bytes: 185990
- fp16 probe archive SHA-256:
  `0928bf4692bd8055ba312cab2c14619c55177627318aa1ca61aeaa9648c4e4c9`
- q-stream roundtrip: true
- raw scale bytes equal: false
- max absolute scale error: `4.408881068229675e-06`
- max relative scale error: `0.0003232605350608376`
- max absolute weight-error bound: `0.0005599278956651688`

The safe byte-equivalent win is only 36 bytes; the fp16 scale probe adds another
54 bytes versus f32 schema. This falsifies the idea that PR101's full reported
decoder saving comes from the schema constants alone when applied to PR106x
bytes. The remaining PR101 gap is likely native trained decoder distribution,
latent microcodec, and/or arithmetic/context coding, not just f32-to-fp16 scale
serialization.

No score claim. Both candidates remain blocked for exact eval until the
submission runtime adapter, inflate-output parity, strict compliance, lane
claim, and exact CUDA auth eval exist.
