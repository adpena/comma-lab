# PR106 Stacked Sister Anchor Custody - 2026-05-06

## Context

`experiments/build_pr106_stacked.py` composes pre-built PR106 sister
sidechannels into one stacked archive. Stack correctness requires every sister
archive to have been built on the same PR106 anchor payload as the stack.

## Finding

Wavelet sister extraction already checked its embedded PR106 payload against
the stack anchor. Latent, yshift, and LRL1 extraction validated wrapper magic
and section framing but did not compare the embedded PR106 bytes against the
anchor before taking their sidechannel blobs.

Evidence grade: `empirical` composition custody hardening, not score evidence.

## Change

- `extract_latent_section_blob(...)`, `extract_yshift_section_blob(...)`, and
  `extract_lrl1_section_blob(...)` now accept `expected_pr106_bytes`.
- Main stack composition passes the loaded PR106 anchor to all sister extractors.
- Each extractor rejects embedded-anchor mismatch and PR106-length overruns
  before returning section bytes.
- Added regression tests for latent, yshift, and LRL1 anchor mismatch.

## Verification

Focused:

```text
.venv/bin/python -m pytest \
  src/tac/tests/test_pr106_stacked.py \
  src/tac/tests/test_dispatch_dryrun_pr106_sidechannels.py -q
```

Result: `42 passed`.

Full preflight:

```text
.venv/bin/python tools/all_lanes_preflight.py --timings
```

## Promotion Status

This does not dispatch CUDA work and does not claim a score. It prevents
cross-anchor sibling composition before any exact CUDA auth eval.
