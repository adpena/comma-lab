# PR101/FEC6 Sidecar Re-Encode Probe

## Verdict

`pr101_sidecar_grammar.encode_ranked_no_op_sidecar` now re-encodes the live PR101/FEC6 607-byte sidecar byte-identically.

PR106 packet adapters keep their legacy wider framing explicitly via encoder
width overrides, so the PR101/FEC6 correction does not silently rewrite the
existing PR106 format-0x02/0x05-0x0C materialization contracts.

This is a grammar-custody correction, not a score claim:

- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- axis: `[predicted]`

## Empirical anchor

Input archive:

- `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/archive.zip`
- archive SHA-256: `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`

Observed sidecar:

- sidecar bytes: `607`
- sidecar SHA-256: `6c2946e323bbbc6f8d906ef6c68989e8acbd8d60332c87da8fe8147f1ea7b12f`
- decoded valid corrections: `597`
- decoded no-op pairs: `3`

Before the fix, the reusable encoder emitted `608` bytes after the base-28 width correction, or `623` bytes before it. The remaining 1-byte excess came from using worst-case `ceil(log2(C(N,k)))` no-op-rank width. The live PR101 huff-enum sidecar stores the actual no-op rank in the minimal byte width needed by this packet: `3` bytes.

After the fix:

- re-encoded bytes: `607`
- re-encoded SHA-256: `6c2946e323bbbc6f8d906ef6c68989e8acbd8d60332c87da8fe8147f1ea7b12f`
- byte-identical to live sidecar: `true`

## Interpretation

The current PR101/FEC6 sidecar is already at the canonical ranked-Huffman/no-op grammar representation available in `tac.packet_compiler.pr101_sidecar_grammar`. A sidecar-only grammar re-encode has `0` byte savings on the current payload.

The next byte-moving sidecar work must therefore change the sidecar semantics or runtime adapter, then prove full runtime visibility and exact eval. A pure re-encode of the existing sidecar grammar is custody-hardening only.

## Verification

```text
.venv/bin/python -m pytest src/tac/tests/test_packet_compiler_pr101_sidecar_grammar.py src/tac/tests/test_pr101_fec6_runtime_consumption.py src/tac/tests/test_pr101_fec6_candidate_queue.py -q
62 passed
```

PR106 compatibility was also checked after the width override was added:

```text
.venv/bin/python -m pytest src/tac/tests/test_packet_compiler_pr106_sidecar_packet.py -q
34 passed
```
