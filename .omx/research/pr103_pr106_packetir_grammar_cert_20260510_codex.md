# PR103/PR106 PacketIR grammar cert (2026-05-10)

generated_at_utc: `2026-05-10T18:45:00Z`
research_only: true
score_claim: false
dispatch_attempted: false
remote_gpu_run: false
lane_claim_created: false

## Scope

Local PacketIR custody work only. No GPU jobs were dispatched and no score is
claimed. The goal was to keep score-lowering PR103/PR106 byte work away from
raw destructive deletion by adding a parser-section candidate-pair certifier.

## Landing

- Added `tac.packet_section_transform.certify_hnerv_grammar_preserving_candidate_pair`.
- Added `tools/certify_hnerv_packet_transform_candidate.py`.
- Extended `src/tac/tests/test_packet_section_transform.py` with positive
  raw-equivalent PR106 recode coverage, raw-mismatch rejection coverage, and
  CLI coverage.

The certifier builds source/candidate `PacketIR`, compares parser-proven
sections, accepts PR106 len24 header updates only when they match the new
decoder section length, requires raw-equivalent Brotli recodes for changed
PR106 decoder/latent sections, blocks fixed-layout PR103 section-length changes
without a runtime adapter, and always leaves
`ready_for_exact_eval_dispatch=false`.

## Candidate status

### Accepted for local archive preflight only

- Candidate: `pr106_exhaustive_152byte_brotli`
- Source archive:
  `experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip`
- Candidate archive:
  `experiments/results/hnerv_lowlevel_repack_pr106_exhaustive_20260510_codex/pr106_hnerv_brotli_repack_candidate.zip`
- PacketIR cert:
  `experiments/results/pr103_pr106_packetir_cert_20260510_codex/pr106_exhaustive_152byte_brotli_packetir_cert.json`
- Archive delta: `-152` bytes
- Predicted rate-only delta if components are unchanged:
  `-0.000101210561`
- `grammar_preserving=true`
- `ready_for_archive_preflight=true`
- `ready_for_exact_eval_dispatch=false`

PacketIR changed sections:

- `packed_header_ff_len24`: byte-neutral control update; len24 matches the
  new decoder section length.
- `decoder_packed_brotli`: `-152` bytes; Brotli decompressed raw SHA-256 is
  unchanged.
- `latents_and_sidecar_brotli`: bytes unchanged; offset moves because the
  decoder section shrank; Brotli decompressed raw SHA-256 is unchanged.

### Rejected as non-grammar-preserving

- Candidate: `pr103_ac_hidden_gem_raw_deletion_control`
- Source archive:
  `experiments/results/lightning_batch/pr103_pr106_ac_repack_exact_eval_t4_20260507T181300Z/archive.zip`
- Candidate archive:
  `experiments/results/hnerv_hidden_gem_pr103_ac_candidate_20260510_agent/archive.zip`
- PacketIR cert:
  `experiments/results/pr103_pr106_packetir_cert_20260510_codex/pr103_hidden_gem_raw_deletion_rejected_packetir_cert.json`
- Archive delta: `-4` bytes
- `grammar_preserving=false`
- `ready_for_archive_preflight=false`
- Blocker:
  `brotli_raw_equivalence_unavailable:decoder_packed_brotli:brotli: decoder failed`

This keeps the already-negative PR103 raw range-stream deletion out of the
grammar-preserving path. Future PR103 bitstream work needs a symbol-roundtrip
or runtime-adapter proof, not raw word deletion.

## Verification

```bash
.venv/bin/python -m pytest src/tac/tests/test_packet_section_transform.py -q
# 9 passed

.venv/bin/python -m pytest \
  src/tac/tests/test_packet_section_transform.py \
  src/tac/tests/test_hnerv_packet_sections.py -q
# 20 passed
```

## Solver-stack wire-in disposition

`research_only=true`; this is a local custody/readiness gate and does not add a
new empirical anchor.

- Sensitivity-map contribution: N/A, no component score or scorer trace.
- Pareto constraint: non-binding until exact CUDA adjudication.
- Bit-allocator hook: N/A, no tensor importance policy changed.
- Cathedral autopilot dispatch hook: not wired; the certifier explicitly
  refuses exact dispatch authority.
- Continual-learning posterior update: N/A, no empirical anchor.
- Probe-disambiguator: N/A, one deterministic certification policy.

## Dispatch disposition

Do **not** exact-dispatch `pr106_exhaustive_152byte_brotli` as a standalone
score-lowering lane. It is useful PacketIR/certifier evidence, but it is
dominated by the active floor.

The closest sibling already has terminal Modal T4 exact-CUDA evidence:

- `pr106_q10_151byte_brotli`
- archive SHA:
  `626b1c76d318eaed45198dc26aea7ee98c8a05f685b840356cf5b621bcddeea7`
- canonical score:
  `0.20936498680571203`
- active floor:
  `0.2089810755823297`
- gap:
  `+0.00038391122338233`

This PacketIR candidate saves only one additional byte versus that evaluated
q10 sibling. The official rate term per byte is approximately
`25 / 37,545,489 = 0.000000665858`, so even the best rate-only improvement over
the terminal q10 result leaves a gap of roughly `0.000383245`. Because the
certifier proves raw-equivalent Brotli recoding, it does not provide a
component-distortion improvement mechanism. Therefore:

1. keep the cert as a conformance and custom-packet-compiler proof;
2. do not claim or dispatch `pr106_exhaustive_152byte_brotli` unless it is
   composed with a byte-different/runtime-different mechanism that can plausibly
   improve components or remove hundreds more bytes;
3. move score-lowering effort to T1/T6/HNeRV-parity training, PR101/A1
   runtime/sidecar mechanisms with real component movement, or larger
   tensor-level custom-codec transforms.
