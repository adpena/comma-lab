# Q10 exact-eval unblock audit - 2026-05-07T17:15:43Z

## Result

- candidate_id: `pr106_q10_151byte_brotli`
- job_name: `exact_eval_pr106_q10_151byte_brotli_20260507`
- archive: `experiments/results/hnerv_lowlevel_repack_pr106_q10_20260506_codex/pr106_hnerv_brotli_repack_candidate.zip`
- archive_sha256: `626b1c76d318eaed45198dc26aea7ee98c8a05f685b840356cf5b621bcddeea7`
- archive_bytes: `186088`
- byte_delta_vs_source_archive: `-151`
- static_packet_ready: `true`
- ready_for_submit: `false`
- score_claim: `false`
- dispatch_attempted: `false`
- remote_gpu_run: `false`

## Remaining Blockers

The candidate is statically ready but fail-closed for submit. Current blockers
are operator/environment and coordination blockers, not method failure:

- `missing_lightning_environment`
- `missing_active_lane_dispatch_claim`

Missing Lightning env vars in this shell:

- `LIGHTNING_SSH_TARGET`
- `LIGHTNING_REMOTE_PACT`
- `LIGHTNING_UPSTREAM_DIR`
- `LIGHTNING_TEAMSPACE`
- `LIGHTNING_STUDIO`
- `LIGHTNING_SDK_USER`

The claim ledger contains a terminal audit row at `2026-05-07T15:42:13Z` with
`claim_status=refused_dispatch_missing_lightning_env`. That row records that no
remote job was submitted from the missing-env shell. It is not an active claim
and it is not method or archive evidence.

## Verified Surfaces

- Packet: `experiments/results/hnerv_lowlevel_repack_pr106_q10_packet_20260507_codex/hnerv_lowlevel_exact_eval_packet.json`
- Candidate manifest: `experiments/results/hnerv_lowlevel_repack_pr106_q10_packet_20260507_codex/manifest.json`
- Dispatch readiness: `experiments/results/hnerv_lowlevel_repack_pr106_q10_packet_20260507_codex/dispatch_readiness_preflight.json`
- Public replay preflight: `experiments/results/hnerv_lowlevel_repack_pr106_q10_packet_20260507_codex/public_replay_preflight.json`
- Release report: `experiments/results/hnerv_lowlevel_repack_pr106_q10_packet_20260507_codex/release_surface/report.txt`

Current runtime tree for exact-eval custody:

- `2a3bbb259179152ca38e659370afdeeabf670bf1f96e2bc91568a7c9cfc93b47`

Operator briefing reports q10 as `ready_for_submit=false` with blockers
`missing_lightning_environment` and `missing_active_lane_dispatch_claim`.
Field-meta selection reports `field_selection_ready_for_exact_eval_dispatch=false`
with the same environment blockers and next local non-GPU action
`verify_lightning_env`.

## Copy-Safe Next Action

Run this first. It does not write repo state and does not dispatch remote/GPU
work:

```bash
.venv/bin/python -c 'import os,sys; missing=[k for k in sys.argv[1:] if not os.environ.get(k)]; raise SystemExit(('"'"'FATAL: missing Lightning env: '"'"'+'"'"', '"'"'.join(missing)) if missing else 0)' LIGHTNING_SSH_TARGET LIGHTNING_REMOTE_PACT LIGHTNING_UPSTREAM_DIR LIGHTNING_TEAMSPACE LIGHTNING_STUDIO LIGHTNING_SDK_USER
```

Only after that command exits `0`, run the packet's ordered steps:

1. refresh static packet without dispatch
2. claim the lane with `tools/claim_lane_dispatch.py`
3. refresh with operator exact-CUDA approval
4. submit exact CUDA
5. harvest with adjudication

The first remote/GPU action is `submit_exact_cuda`; do not run it until
`ready_for_submit=true` in the packet.

## Verification Commands

```bash
.venv/bin/python tools/build_hnerv_lowlevel_exact_eval_packet.py --candidate-result experiments/results/hnerv_lowlevel_repack_pr106_q10_20260506_codex/manifest.json --archive experiments/results/hnerv_lowlevel_repack_pr106_q10_20260506_codex/pr106_hnerv_brotli_repack_candidate.zip --archive-sha256 626b1c76d318eaed45198dc26aea7ee98c8a05f685b840356cf5b621bcddeea7 --archive-bytes 186088 --baseline-json experiments/results/lightning_batch/exact_eval_public_pr106_belt_and_suspenders_adapter_t4_20260504T1330Z/contest_auth_eval.adjudicated.json --inflate-sh experiments/public_runtime_adapters/pr106_belt_and_suspenders_adapter/inflate.sh --upstream-dir upstream --result-dir experiments/results/hnerv_lowlevel_repack_pr106_q10_packet_20260507_codex --release-surface-dir experiments/results/hnerv_lowlevel_repack_pr106_q10_packet_20260507_codex/release_surface --lane-id pr106_q10_151byte_brotli --job-name exact_eval_pr106_q10_151byte_brotli_20260507 --claims-path .omx/state/active_lane_dispatch_claims.md --claim-ttl-hours 24 --agent codex:gpt-5.5 --now-utc 2026-05-07T17:15:43Z --json-out experiments/results/hnerv_lowlevel_repack_pr106_q10_packet_20260507_codex/hnerv_lowlevel_exact_eval_packet.json --operator-approved-exact-cuda
.venv/bin/python -m pytest src/tac/tests/test_hnerv_lowlevel_exact_eval_packet.py
.venv/bin/python tools/build_field_meta_dispatch_selection.py --manifest experiments/results/hnerv_lowlevel_repack_pr106_q10_packet_20260507_codex/hnerv_lowlevel_exact_eval_packet.json --claims-path .omx/state/active_lane_dispatch_claims.md --now-utc 2026-05-07T17:15:43Z --operator-approved-exact-cuda --json-out /tmp/q10_field_selection.json
.venv/bin/python tools/operator_briefing.py --json
```
