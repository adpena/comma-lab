# Codex Session Summary: PR107 Apogee CD1 Projector

Timestamp UTC: 2026-05-19T06:37:00Z
Session: 019de465
Scope: OP_SYN_1 continuation after PR106 format0d projector landing.

## Landed

- Moved `pr107_apogee_length_prefixed` from fail-closed detection-only to
  anchor-ready grammar for its decoder section.
- Added `parse_pr107_apogee_projector_layout` and `CD1` tensor-span mapping in
  `tools/extract_master_gradient.py`.
- Wired PR107's runtime camera-space channel offsets into the differentiable
  roundtrip before scorer evaluation.
- Updated `tac.master_gradient_archive_parsers` so downstream consumers see
  PR107 Apogee as anchor-emitting.
- Added tests for synthetic PR107 detection, registry status, extract-all
  counts, and `CD1` scale/mantissa offsets.
- Ran live PR107 one-pair `[diagnostic-CPU]` no-anchor extraction against the
  public PR107 intake archive and source inflate path.

## Still Open

- OP_SYN_1 is not complete until DP1 and/or explicitly scoped remaining
  projector blockers are closed or retired with evidence.
- PR107 latent-byte sensitivity is not yet differentiated; latent payloads are
  treated as zero-gradient v1 surfaces.
- No contest-axis anchor was written. The smoke used `--no-anchor-write`.
- Partner WIP in provider, MPS, preflight, cathedral, and state surfaces was
  left untouched.

## Next Best Action

Continue OP_SYN_1 with DP1 renderer/codebook/residual projector if live churn is
stable; otherwise switch to a non-overlapping score-lowering task from the
canonical task queue.
