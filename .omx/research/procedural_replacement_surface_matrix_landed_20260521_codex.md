# Procedural Replacement Surface Matrix Landed

**Date (UTC):** 2026-05-21
**Lane:** `lane_procedural_replacement_surface_matrix_20260521`
**Artifact:** `experiments/results/procedural_replacement_surface_matrix_20260521T020000Z/`
**Axis:** `[predicted]`; `score_claim=false`, `promotion_eligible=false`, `ready_for_exact_eval_dispatch=false`

## Summary

This landing turns the FEC6 `PARSER_SAFE_SUBSET_EMPTY` result into a
substrate-facing routing matrix. The key distinction is now explicit:
null-gradient is not enough; a candidate must be parser-visible as a
whole-section replacement surface or must first land a procedural-aware archive
adapter.

The current ranked matrix is:

| Rank | Substrate | Surface | Status | Predicted bytes saved | Predicted dS | Blocker |
|---:|---|---|---|---:|---:|---|
| 1 | `pretrained_driving_prior_dp1` | `codebook_blob` | `READY_TO_PAIR_SMOKE` | 4,064 | -0.002706051 | Paired contest-CUDA + contest-CPU smoke not yet landed |
| 2 | `atw_codec_v2` | `cdf_table_blob` | `DESIGN_READY_DEFERRED` | 2,528 | -0.001683291 | D4 predecessor verdict and Variant-C scoping gate paid dispatch |
| 3 | `atw_codec_v2` | `class_prior_table_blob` | `REQUIRES_SIGNAL_PRESERVATION_PROBE` | 19,168 | -0.012763184 | High signal-risk; probe after CDF table |
| 4 | `vq_vae` | `codebook_inside_decoder_blob` | `ADAPTER_REQUIRED` | 8,160 | -0.005433409 | Canonical VQV1 stores codebook inside brotli'd decoder blob |
| 5 | `grayscale_lut_glv1` | `chroma_lut` | `BLOCKED_NO_CURRENT_SURFACE` | 224 | -0.000149152 | GLV2 explicit LUT grammar required |
| 6 | `pr101_fec6_frontier` | `master_gradient_null_bytes` | `BLOCKED_PARSER_SAFE_SUBSET_EMPTY` | 0 | 0.0 | All null-gradient bytes are parser-essential |

## What Landed

- `src/tac/procedural_replacement_surfaces.py`: reusable static matrix with
  explicit parser-visible, raw-byte-safe, whole-section-replacement, adapter,
  and canonical equation #26 domain fields.
- `tools/plan_procedural_replacement_surfaces.py`: operator CLI that emits
  `surface_matrix.json` and `surface_matrix.md`.
- `src/tac/tests/test_procedural_replacement_surfaces.py`: focused tests for
  DP1, ATW2, VQ-VAE, grayscale-LUT, and PR101/FEC6 negative-control routing.

## Verification

```bash
.venv/bin/python -m pytest -q src/tac/tests/test_parser_safe_subset_smoke.py src/tac/tests/test_procedural_replacement_surfaces.py
.venv/bin/python -m py_compile src/tac/procedural_replacement_surfaces.py tools/plan_procedural_replacement_surfaces.py src/tac/tests/test_procedural_replacement_surfaces.py
git diff --check
.venv/bin/python tools/plan_procedural_replacement_surfaces.py --output-dir experiments/results/procedural_replacement_surface_matrix_20260521T020000Z
```

Artifact hashes:

```text
a5f0844b92da5468f724c70b2eddbb822d7ca2c4ce1e73ac712b68756c71bd61  experiments/results/procedural_replacement_surface_matrix_20260521T020000Z/surface_matrix.json
7ba2b59d43678d761e7aed820bdcbc6af945fecf287be507bd6c1136a632def9  experiments/results/procedural_replacement_surface_matrix_20260521T020000Z/surface_matrix.md
```

## Next Action

The next frontier-moving action is DP1 paired smoke, because it is the top
ranked `READY_TO_PAIR_SMOKE` surface. ATW2 CDF table is next once the D4 /
Variant-C gate is resolved. VQ-VAE and GLV1 should not be treated as raw
parser-visible surfaces without their procedural adapters.
