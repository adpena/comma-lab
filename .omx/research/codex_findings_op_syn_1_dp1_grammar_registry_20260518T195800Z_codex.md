# Codex Findings - OP-SYN-1 DP1 Grammar Registry Slice

Date: 2026-05-18 19:58:00 UTC
Author: Codex

## Scope

Directive:
`.omx/research/codex_routing_directive_op_syn_1_master_gradient_six_archive_extension_20260518.md`.

This lands a bounded premise-verified slice, not the full 6-archive extractor
campaign.

## Patch

- Added `parse_dp1_archive_layout(...)` to `tools/extract_master_gradient.py`.
- DP1 uses the canonical
  `tac.substrates.pretrained_driving_prior.archive.parse_dp1_archive_bytes`
  section parser, so the extractor does not hand-roll DP1 offsets.
- DP1 is registered as `fail_closed_detection_only` with required projector
  `dp1_pretrained_driving_prior_schema_projector`.
- Added `--list-grammars`, returning the operator-facing grammar registry with
  authority contracts for all known extractor grammars.

## Authority Boundary

Current anchor-emitting grammars:

- `fec6_fp11_selector`
- `pr101_lc_v2`
- `a1_finetuned`

Current detection-only grammars:

- `dp1_pretrained_driving_prior`
- `pr106_format0d`
- `pr106_ff_packed_hnerv`
- `hnerv_lc_v2_length_prefixed`
- `pr107_apogee_length_prefixed`

DP1 section detection is now available for xray/routing, but master-gradient
anchor emission remains blocked until the DP1 renderer/codebook/residual
projector exists.

## Verification

- `[empirical:PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q src/tac/tests/test_extract_master_gradient.py]`
  - Result: `36 passed`
- `[empirical:PYTHONDONTWRITEBYTECODE=1 .venv/bin/ruff check tools/extract_master_gradient.py src/tac/tests/test_extract_master_gradient.py]`
  - Result: `All checks passed`
- `[empirical:PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/extract_master_gradient.py --list-grammars]`
  - Result: `grammar_count=8`, `dp1_pretrained_driving_prior` is detection-only,
    `anchor_emission_allowed=false`, `score_claim_allowed=false`.
