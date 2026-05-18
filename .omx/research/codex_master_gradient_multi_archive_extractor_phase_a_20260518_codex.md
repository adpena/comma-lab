# Codex Master-Gradient Multi-Archive Extractor Phase A - 2026-05-18

## Scope

Task:
`codex_routing_directive_inflate_py_plus_wyner_ziv_compliance_plus_master_gradient_extension_20260518::ITEM_3`.

Phase A extends `tools/extract_master_gradient.py` with a typed archive grammar
detection and boundary/xray surface. This is intentionally not a full
gradient-authority claim for every grammar.

## Landed Surface

- Added `ArchiveSection` and `ArchiveLayout` records that preserve:
  charged archive SHA/bytes, inner member SHA/bytes, gradient subject domain,
  logical sections, parser notes, and `gradient_projection_supported`.
- Added `--detect-grammar-only`, which prints layout JSON and exits before
  importing scorer/runtime modules.
- Added parsers for:
  - `fec6_fp11_selector`
  - `a1_finetuned`
  - `pr101_lc_v2` fixed-offset HNeRV microcodec, constrained to the
    known public fixed-offset byte family instead of an arbitrary
    length-only fallback
  - `pr106_format0d`
  - `pr106_ff_packed_hnerv`
  - `hnerv_lc_v2_length_prefixed`
  - `pr107_apogee_length_prefixed`
- Heavy extraction now fails closed before codec/scorer import for grammars
  where byte-gradient projection is not implemented.
- ZIP payload extraction is strict-single-member only. Multi-member ZIPs now
  fail closed so charged archive bytes cannot silently diverge from the
  gradient subject byte domain.
- Normal anchor-emitting runs still require an explicit `--axis`; only
  `--detect-grammar-only` can omit it.

## Authority Boundary

`gradient_projection_supported=true` only for the already-supported
fec6/PR101-compatible split-Brotli fixed-section path. A1, PR106 direct,
PR106 format0d, true hnerv_lc_v2, and PR107 Apogee are layout/xray only until
a grammar-aware projector is wired.

This avoids the false-authority class where a single `x` member with enough
bytes is misread as PR101 fixed-offset and then emitted as a master-gradient
anchor.

## Verification

- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider src/tac/tests/test_extract_master_gradient.py`
  - `33 passed`
- `.venv/bin/ruff check tools/extract_master_gradient.py src/tac/tests/test_extract_master_gradient.py`
  - `All checks passed`
- `git diff --check`
  - clean
- Real local fixture probes:
  - FEC6: `fec6_fp11_selector`, `projection_supported=true`
  - public PR101: `pr101_lc_v2`, `projection_supported=true`
  - PR106 format0d: `pr106_format0d`, `projection_supported=false`
  - public PR106 direct: `pr106_ff_packed_hnerv`, `projection_supported=false`
  - public PR100 true lc_v2: `hnerv_lc_v2_length_prefixed`, `projection_supported=false`
  - PR107 Apogee: `pr107_apogee_length_prefixed`, `projection_supported=false`

## Remaining Work

ITEM_3 should remain `in_progress` until the next phase materializes the
requested per-pair fp64 master-gradient tensor for a supported archive and
adds grammar-aware projection for any additional packet family that will
emit anchors. This phase is the false-authority guard and xray boundary layer.
