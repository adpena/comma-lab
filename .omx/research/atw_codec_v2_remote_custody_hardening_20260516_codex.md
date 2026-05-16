# ATW Codec V2 Remote Custody Hardening

- date: `2026-05-16`
- agent: `codex`
- trigger: post-push adversarial review of `scripts/remote_lane_substrate_atw_codec_v2.sh`
- score_claim: `false`
- promotion_eligible: `false`

## Finding

The ATW v2 remote driver required `DISPATCH_INSTANCE_JOB_ID` and the claims
ledger path, but it did not verify that the lane/job pair had an active
`tools/claim_lane_dispatch.py` claim before continuing. It also emitted a
`LANE_ATW_CODEC_V2_DONE [contest-*]` marker when an auth-eval JSON merely
existed, rather than parsing the JSON as a custody-valid score claim. The
completion line pointed at `0.bin`, while the scored archive boundary is
`archive.zip`.

Classification: launch-path false authority / custody overclaim risk. No score
or lane result changes.

## Fix

1. Added active claim verification via `tools/claim_lane_dispatch.py summary`
   before NVDEC/bootstrap/training.
2. Added terminal claim closure on all script exits with statuses split between
   claim-verification failure and remote-driver failure.
3. Changed completion-marker logic so `[contest-CUDA]` / `[contest-CPU]` is
   emitted only after `tac.auth_eval_result.parse_auth_eval_score_claim` accepts
   the auth-eval JSON with component recomputation.
4. Changed completion logging to name both `archive_zip=$OUTPUT_DIR/archive.zip`
   and `payload_0bin=$OUTPUT_DIR/0.bin`.
5. Added ATW v2 regression tests covering active-claim verification, terminal
   closure, and parser-gated contest markers.

## Verification

- `bash -n scripts/remote_lane_substrate_atw_codec_v2.sh`
- `.venv/bin/python -m ruff check src/tac/substrates/atw_codec_v2/tests/test_atw_codec_v2.py`
- `.venv/bin/python -m pytest src/tac/substrates/atw_codec_v2/tests/test_atw_codec_v2.py -q` -> `26 passed`
- `git diff --check`
- `.venv/bin/python tools/lane_maturity.py validate` -> `767 lane(s) validated cleanly`
