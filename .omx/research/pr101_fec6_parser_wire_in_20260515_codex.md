# PR101 FEC6 Parser Wire-In - 2026-05-15

score_claim: false
promotion_eligible: false
ready_for_exact_eval_dispatch: false

## Why This Landed

PR101 FEC6 is the nearest live CPU-axis packet:

- archive: `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip`
- archive_sha256: `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`
- bytes: `178517`
- `[contest-CPU]`: `0.1920513168811056`
- paired `[contest-CUDA/T4]`: `0.22621002169349796`

The CPU gap to `<0.192` is about `78` bytes if components are unchanged.
Generic entropy xray found only about `30` bytes of headroom, so the next
credible move must inspect the actual FP11/FEC6 selector structure, not treat
the packet as opaque PR101 bytes.

## Code Delta

- `src/tac/analysis/hnerv_packet_sections.py`
  - adds parser `pr101_fec6_fixed_huffman_selector` with alias `pr101_fec6`;
  - auto-detects `FP11` wrapper payloads;
  - splits the member into eight sections:
    - `fp11_magic`
    - `source_len_u32le`
    - `source_decoder_compact_brotli_streams`
    - `source_latents_raw_lzma_delta_u8`
    - `source_sidecar_dim_delta_huffman_enum`
    - `selector_len_u16le`
    - `selector_fec6_fixed_huffman_k16_header`
    - `selector_fec6_fixed_huffman_k16_bitstream`
- `src/tac/tests/test_hnerv_packet_sections.py`
  - adds fail-closed coverage for FP11 plus FEC6 fixed-Huffman K16 selector custody.

## Real Packet Probe

Command:

```bash
.venv/bin/python tools/build_hnerv_packet_section_manifest.py \
  --archive fec6=experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip \
  --parser pr101_fec6 \
  --json-out /tmp/pr101_fec6_section_manifest.json
```

Result:

- `parser_section_gate.ready=true`
- parser: `pr101_fec6_fixed_huffman_selector`
- member bytes: `178417`
- section count: `8`
- FEC6 selector bitstream bytes: `243`
- FEC6 selector bitstream entropy: `7.030745092222` bits/byte
- score claim: false

## Next Byte-Closed Prototype

`pr101_fec6_k16_microcodec_x` should reuse existing packet/compiler surfaces:

- PR101 section constants and sidecar grammar: `tac.packet_compiler.pr101_sidecar_grammar`
- PR103 arithmetic/range coding: `tac.packet_compiler.pr103_arithmetic_coding`
- PR106 exact-radix helpers: `tac.packet_compiler.pr106_sidecar_packet`
- PR84 adaptive-context symbol coding: `tac.packet_compiler.pr84_adaptive_mask`

Stop condition for CPU-only byte work: if a candidate cannot save at least
`79` charged bytes or move components on exact `[contest-CPU]`, do not spend
paired eval. Promotion still requires exact `[contest-CPU]` and exact
`[contest-CUDA/T4]` on the same archive SHA and runtime tree.

## Verification

```bash
.venv/bin/python -m pytest -q src/tac/tests/test_hnerv_packet_sections.py
.venv/bin/ruff check src/tac/analysis/hnerv_packet_sections.py src/tac/tests/test_hnerv_packet_sections.py
```

Results:

- `13 passed in 0.34s`
- `All checks passed!`
