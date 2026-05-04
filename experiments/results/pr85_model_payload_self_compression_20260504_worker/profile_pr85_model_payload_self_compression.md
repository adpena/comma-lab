# PR85 Model Payload Self-Compression Profile

- planning_only: true
- score_claim: false
- dispatch_performed: false
- archive: `experiments/results/public_pr85_intake_20260503_codex/archive.zip`
- archive_sha256: `eb18df2f1b364e513f36933116ec0c1011c6d3bbf8022e16495e0508c6258f5e`
- model_bytes: 57074
- model_sha256: `c9194c19d44d3c32c4e588bbe4f9986ed0c562779078f0f4d6b073f0799b5dbc`
- encoded_container: brotli_stream
- decoded_payload_kind: pr85_qh0_joint_frame_model
- decoded_bytes: 61590

## Static Probes

- encoded entropy bits/byte: 7.995683133524
- encoded best recompression delta bytes: 4
- decoded best recompression delta bytes: -4363

## Candidate Routes

### qh0_record_level_repack_or_serializer

- route_type: score_affecting_model_payload_rewrite
- estimated_model_segment_bytes_saved: None
- dispatchable_now: False
- recommendation: Top byte-saving route when lossless Brotli recode is non-improving: build a deterministic QH0 record-level serializer/repacker and target low-entropy decoded model records under runtime parity.

### lossless_brotli_recode_decoded_model_segment

- route_type: lossless_container_recode
- estimated_model_segment_bytes_saved: 0
- dispatchable_now: False
- recommendation: Do not prioritize: generic Brotli recoding did not beat the current charged bytes.

### generic_wrapper_noop_guard

- route_type: negative_guardrail
- estimated_model_segment_bytes_saved: None
- dispatchable_now: False
- recommendation: Do not wrap the charged model segment in another generic compressor unless the decoded-byte recode proves a smaller byte-closed model segment.

## Top Implementable Route

`qh0_record_level_repack_or_serializer`: Top byte-saving route when lossless Brotli recode is non-improving: build a deterministic QH0 record-level serializer/repacker and target low-entropy decoded model records under runtime parity.
