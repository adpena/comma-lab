# Codex Session Summary: OP-SYN-1 Extract-All Manifest Runner

timestamp_utc: 2026-05-18T20:29:47Z
agent: codex
score_claim: false

## Landed

- Added `extract-all --manifest ... [--output ...] [--strict]` to
  `tools/extract_master_gradient.py`.
- Preserved the existing flat extractor CLI and added `list-grammars` as a
  command-style alias for the existing `--list-grammars` flow.
- Added tests proving the batch runner covers all eight registered synthetic
  grammar contracts, never writes anchors, and fails strict mode on
  detection-only grammars.
- Closed OP-SYN-1's `extract_all_batch_cli_missing` blocker without promoting
  DP1/PR106/PR107 parser success to anchor authority.

## Still Open

OP-SYN-1 remains in progress until the detection-only grammars receive real
projectors:

- DP1 schema projector
- PR106 format0d primary payload projector
- PR107 Apogee schema projector

Next best slice: pick one projector only after verifying the real archive
grammar and consumer needs; do not reuse the PR101 fixed-section projector for
packed or length-prefixed layouts.
