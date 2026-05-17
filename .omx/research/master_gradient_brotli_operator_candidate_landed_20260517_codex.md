# Master-Gradient Brotli Operator Candidate Landed - 2026-05-17

## Verdict

The master-gradient apparatus now has its first executable grammar-aware
operator row: a PR106-format Brotli-section recompression tournament.

This is a packet-valid archive mutation, not a score claim. It proves that a
parser-proven logical section can be modified, repacked, CRC-checked, and
reparsed without using raw archive-byte derivatives.

## Landed Surface

- Reusable builder:
  `src/tac/master_gradient_brotli_operator_candidate.py`
- CLI:
  `tools/build_master_gradient_brotli_operator_candidate.py`
- Tests:
  `src/tac/tests/test_master_gradient_brotli_operator_candidate.py`

The builder supports `pr106_ff_packed_hnerv` sections:

- `decoder_packed_brotli`
- `latents_and_sidecar_brotli`

It refuses unsupported grammars, refuses no-op replacements by default, and
delegates archive rebuilding to `tac.monolithic_packet_candidate` so PR106
headers, section offsets, deterministic ZIP metadata, and CRC are handled by
the existing packet-candidate path.

## Local Materialized Candidate

Command:

```bash
.venv/bin/python tools/build_master_gradient_brotli_operator_candidate.py \
  --source-archive experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip \
  --output-dir experiments/results/master_gradient_brotli_operator_pr106_decoder_20260517_codex \
  --candidate-id pr106_decoder_brotli_q10_lgwin18_operator_20260517 \
  --target-section decoder_packed_brotli
```

Result:

- source archive:
  `experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip`
- candidate archive:
  `experiments/results/master_gradient_brotli_operator_pr106_decoder_20260517_codex/archive.zip`
- candidate archive SHA-256:
  `33ee2d9c5f2b0dade3e9c5cf85c91336f60f5b3e5de423543224832e56ed5aa3`
- archive byte delta: `-151`
- changed section:
  `decoder_packed_brotli`
- section byte delta: `170278 -> 170127` (`-151`)
- best grid row:
  Brotli `quality=10`, `lgwin=18`
- rate-only score delta if components are unchanged:
  `-0.00010054470192144788`

The materialized artifact remains in ignored `experiments/results/`; this
ledger preserves the durable signal without committing rebuildable binary
artifacts.

## Proof State

Proven:

- `repacked_archive=true`
- `updated_zip_headers=true`
- `updated_zip_crc=true`
- `parser_reparse_success=true`
- `brotli_decode_success=true`
- `structural_non_noop_section_changed=true`

Still missing:

- `inflate_success_proof`
- `runtime_byte_consumption_noop_detector`
- active lane claim
- paired exact CPU/CUDA result review

Therefore:

- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_provider_dispatch=false`
- `ready_for_exact_eval_dispatch=false`
- `dispatch_attempted=false`

## Interpretation

This is not a return to PR106 local-basin obsession. It is a reusable executable
operator class for the master-gradient rewrite. PR106-format packets are simply
the first supported grammar because their logical sections are already
Brotli-decodable and `tac.monolithic_packet_candidate` already knows how to
rewrite their header and validate both Brotli streams.

The score-lowering signal is small (`151` bytes, about `1e-4` score on rate
alone) but real as a packet mutation. It establishes the path for future
operator rows:

1. section parser proves the coordinate;
2. operator mutates the section through a valid codec transform;
3. archive builder repacks and updates headers/CRC;
4. parser re-proves the new layout;
5. runtime/inflate proof closes byte-consumption;
6. paired exact CPU/CUDA eval decides whether the row belongs in a response
   matrix.

## Next Highest-EV Follow-Up

Build the equivalent operator class for the current FEC6 CPU frontier:

- parse `FP11` + `FEC6` selector payload;
- mutate selector grammar with a component-aware rule, not generic byte flips;
- rebuild the PR101/FEC6 wrapper and ZIP packet;
- prove selector decode and runtime consumption;
- only then route to paired exact eval.

The FEC6 path is the score-relevant one. The PR106 Brotli operator is the first
working template for the apparatus.
