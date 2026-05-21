# Parser-Safe Null-Byte Subset Smoke Landed

**Date (UTC):** 2026-05-21T01:45:17Z
**Lane:** `lane_wave_3_parser_safe_subset_smoke_20260520`
**Artifact:** `experiments/results/parser_safe_subset_smoke_20260521T014517Z/`
**Verdict:** `PARSER_SAFE_SUBSET_EMPTY`
**Axis:** `[macOS-CPU advisory]`; `score_claim=false`, `promotion_eligible=false`, `ready_for_exact_eval_dispatch=false`

## Summary

This smoke closes the immediate follow-up to the PR101/FEC6 master-gradient
null-byte result. It asks whether any of the 16,292 master-gradient-null bytes
are also downstream of parser dispatch and therefore plausible byte-removal
targets inside the existing fec6 archive grammar.

Result: none are. Every null-gradient byte falls inside a parser-essential
region:

| Region | Kind | Byte range | Null bytes |
|---|---|---:|---:|
| `A_fp11_outer_wrapper` | struct field | `[0, 8)` | 8 |
| `B_pr101_decoder_brotli` | Brotli stream | `[8, 162172)` | 39 |
| `C_pr101_latent_lzma` | LZMA stream | `[162172, 177559)` | 15,387 |
| `D_pr101_sidecar_brotli` | Brotli stream | `[177559, 178166)` | 607 |
| `E_fec6_selector_len_uint16` | struct field | `[178166, 178168)` | 2 |
| `F_fec6_selector_payload` | fixed Huffman bitstream | `[178168, 178417)` | 249 |

The parser-safe subset size is `0 / 16,292`. The 4-variant auth-eval smoke is
therefore skipped by construction: there are no bytes in the existing fec6
member that are both null-gradient and parser-safe.

## Why This Matters

This turns the null-exploit cascade into an exact failure classification for
the current fec6 grammar rather than a vague negative. The previous smoke
showed direct mutation of all null-gradient bytes breaks inflate. This smoke
localizes why: the null-gradient bytes are score-opaque but parser-essential.

The engineering consequence is narrow and useful:

- Do not spend on removal/replacement of current fec6 null bytes.
- Do use the null-gradient signal for future archive designs that expose
  intermediate-transform regions by construction.
- Keep canonical equation #26 in-domain only for real replacement surfaces
  such as DP1 codebook bytes, VQ-VAE codebook bytes, GLV2 chroma LUT bytes,
  ATW V2 quantizer/CDF tables, or other parser-visible but scorer-low-leverage
  intermediate transforms.

## Verification

Commands run:

```bash
.venv/bin/python tools/run_parser_safe_subset_smoke.py --dry-run
.venv/bin/python -m pytest -q src/tac/tests/test_parser_safe_subset_smoke.py
.venv/bin/python -m py_compile tools/run_parser_safe_subset_smoke.py src/tac/tests/test_parser_safe_subset_smoke.py
.venv/bin/python tools/lane_maturity.py validate
git diff --check
```

Artifact hashes:

```text
174302a44aeaebf94fa2c429152610425c3f174c4ab9143c42c24f38e6a843c2  experiments/results/parser_safe_subset_smoke_20260521T014517Z/smoke_result.json
9e67cfcc75505422553b1e1f81aeca8c609b1f83500a85667c23f90fc02967e5  experiments/results/parser_safe_subset_smoke_20260521T014517Z/smoke_result.md
```

## Next Action

Route frontier effort away from current fec6 byte removal and toward
parser-visible replacement substrates. The nearest concrete paths are:

1. ATW V2 procedural variant design and byte-count audit.
2. DP1/VQ-VAE procedural replacement paired smoke when operator-gated dispatch
   is available.
3. GLV2 grayscale-LUT grammar if the grayscale scaffold graduates from
   append-only envelope to real replacement surface.
