# Codex Findings - SNeRV Fresh-Eyes Guardrail Hardening

UTC: 2026-06-02T02:32Z
Axis: code guardrail / false-authority hardening
Authority: no score authority

## Trigger

Fresh-eyes explorer `Archimedes` flagged four SNeRV false-authority risks:

1. Adaptive/waterfill step-map packets could carry unconsumed payload bytes or
   duplicate map ownership and still decode.
2. SNeRV rate reports compare SNAR1 packet bytes to PR101 archive.zip bytes
   before contest packaging overhead is closed.
3. Rate adjudication treated any non-empty `receiver_archive_sha256` as enough
   custody shape.
4. CLI score output made it too easy to forget that `score_linf` and `score_l2`
   share one charged archive `rate_term`.

## Landed Guardrails

- `src/tac/analysis/snerv_step_map_coder.py`
  - adaptive decode now rejects duplicate map indices;
  - adaptive decode now rejects out-of-range map indices;
  - adaptive decode now rejects constant groups that carry payload bytes;
  - adaptive decode now verifies non-constant payload ranges are contiguous and
    consume the entire payload.
- `src/tac/tests/test_snerv_step_map_coder.py`
  - added duplicate-map-ownership regression;
  - added trailing-payload regression.
- `src/tac/analysis/snerv_rate_adjudication.py`
  - receiver archive parser closure now requires a 64-hex SHA-256-shaped digest;
  - invalid/missing receiver archive SHA leaves parser blockers in place;
  - all rate rows now carry
    `snar1_packet_bytes_not_contest_archive_zip_bytes`.
- `src/tac/tests/test_snerv_rate_adjudication.py`
  - upgraded replay-verified examples to full SHA-256-shaped digests;
  - added explicit `receiver_archive_sha256="abc"` regression.
- `tools/run_snerv_inverse_steg_advisory.py`
  - CLI now labels `rate_term` as the shared charged archive term used by both
    L-inf and L2 advisory scores.

## Verification

```bash
/Users/adpena/Projects/pact/.venv/bin/ruff check \
  src/tac/analysis/snerv_step_map_coder.py \
  src/tac/tests/test_snerv_step_map_coder.py \
  src/tac/analysis/snerv_rate_adjudication.py \
  src/tac/tests/test_snerv_rate_adjudication.py

/Users/adpena/Projects/pact/.venv/bin/python -m pytest \
  src/tac/tests/test_snerv_step_map_coder.py \
  src/tac/tests/test_snerv_rate_adjudication.py
```

Result: ruff passed; 26 tests passed.

## Remaining Constraints

- The SNAR1-vs-archive.zip comparison is still advisory. A true rate win needs
  contest archive.zip packaging, byte-closed receiver proof, and paired contest
  CPU/CUDA replay.
- The shared-rate CLI label prevents misreading, but a deeper future cleanup
  should separate detector-distortion components from archive-rate components
  in any table that compares L-inf and L2 objectives.
- PR101 CPU recovery was still pending during this hardening, so no new exact,
  full-video, or CUDA launch was authorized.
