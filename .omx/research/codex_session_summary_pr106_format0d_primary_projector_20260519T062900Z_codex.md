# Codex Session Summary - PR106 Format0d Primary Projector

timestamp_utc: 2026-05-19T06:29:00Z
agent: codex
score_claim: false

## Landed

- Promoted `pr106_format0d` from detection-only to anchor-ready for the
  primary packed-HNeRV decoder payload.
- Preserved explicit zero-gradient v1 semantics for format0d base/extra
  sidecar bytes; no sidecar score-response authority was inferred.
- Added synthetic and live-fixture tests for the projector contract and byte
  mapping.
- Proved a real one-pair diagnostic extractor smoke with `--no-anchor-write`;
  no `.omx/state/master_gradient_anchors.jsonl` row was written.

## Still Open

- OP-SYN-1 remains `in_progress`: DP1 and PR107 projectors are still missing.
- PR106 sidecar mutation authority still needs packet-valid operator-response
  rows and exact runtime proofs.
- The current diagnostic smoke is `[diagnostic-CPU]`, not contest authority.

## Next

Continue OP-SYN-1 with one more real projector only after checking live churn.
Preferred next slice: PR107 Apogee length-prefixed schema if its source codec is
available and can be decoded without false authority; otherwise DP1 projector
with renderer/codebook/residual section separation.
