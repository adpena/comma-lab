# HNeRV Brotli section recode scan (2026-05-10)

Generated: `2026-05-10T18:35:00Z`

`score_claim=false`; `dispatch_attempted=false`; `ready_for_exact_eval_dispatch=false`.

## Why this exists

The packet-section bridge can compile PR106 `0xff + len24` section recodes, but
PR101/A1 and PR103 use different runtime grammars. A raw byte win in a fixed
layout section is not a contest candidate unless the corresponding inflate
runtime consumes the changed section boundaries.

This scan makes that distinction explicit so byte-level work does not become a
proxy leak or no-op candidate.

## Artifact

```text
experiments/results/hnerv_brotli_section_recode_opportunities_20260510_codex/opportunities.json
```

Command:

```bash
.venv/bin/python tools/scan_hnerv_brotli_section_recode_opportunities.py \
  --quality 9 --quality 10 --quality 11 \
  --lgwin default --lgwin 18 --lgwin 20 --lgwin 22 --lgwin 24 \
  --lgblock default --lgblock 16 \
  --jobs 8 \
  --json-out experiments/results/hnerv_brotli_section_recode_opportunities_20260510_codex/opportunities.json \
  --fail-if-blocked
```

## Result

- archives scanned: PR101, PR103, PR106 public HNeRV artifacts
- parser sections: 14
- Brotli-decompressible sections: 7
- rate-positive sections under this grid: 1
- existing-bridge-compilable sections: 1
- runtime-adapter-required rate-positive sections: 0
- best byte delta: `-152` bytes

The only rate-positive result is PR106 `decoder_packed_brotli`:

```text
source_bytes=170278
candidate_bytes=170126
byte_delta=-152
runtime_contract=pr106_len24_header_recomputed_by_packet_section_compiler
```

This matches the earlier exhaustive PR106 recode candidate SHA path and is not a
new score claim. It is a reproducible way to regenerate the same byte-positive
candidate through the typed packet-section bridge.

## Adversarial classification

Top-level scan output is not an archive candidate and must remain
`ready_for_archive_preflight=false`. The scan may report
`ready_for_candidate_build=true` when at least one section can be compiled by an
existing bridge. A later candidate builder must still emit changed archive
bytes, run archive preflight, claim the lane before GPU, and get exact CUDA
evidence before any score status changes.

PR101/PR103 do not expose a Brotli recode win under this bounded grid. That
does not falsify arithmetic/range-coding, section regrouping, runtime adapter,
or train-time substrate work; it only says generic per-section Brotli recoding
is saturated for these public bytes at this grid.

## Next

1. Build PR103 arithmetic-section transform planning around the existing
   range-code parser, not generic Brotli recoding.
2. Wire the typed bridge output into exact-eval packet readiness so archive
   candidates and scan artifacts cannot be confused.
3. Keep T1 Ballé Modal harvest as the first score-lowering dispatch result to
   classify when the running job exits.
