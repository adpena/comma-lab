# Q10 Frontier Dispatch Readiness

Date: 2026-05-07
Operator approval context: exact-CUDA spend approved, but dispatch remains blocked until Lightning environment variables are present and a matching active Level-2 lane claim exists.

## Result

`pr106_q10_151byte_brotli` is the current field-selected exact-eval packet among the static HNeRV rate candidates.

- Candidate archive: `experiments/results/hnerv_lowlevel_repack_pr106_q10_20260506_codex/pr106_hnerv_brotli_repack_candidate.zip`
- Packet: `experiments/results/hnerv_lowlevel_repack_pr106_q10_packet_20260507_codex/hnerv_lowlevel_exact_eval_packet.json`
- Archive bytes: `186088`
- Archive SHA-256: `626b1c76d318eaed45198dc26aea7ee98c8a05f685b840356cf5b621bcddeea7`
- Byte delta: `-151` versus the PR106 HNeRV source archive
- Rate-only score delta: `-0.00010054470192144788`
- Evidence grade: empirical archive candidate until exact CUDA auth eval lands
- Score claim: `false`
- Dispatch attempted: `false`

## Hardening Landed

- `tools/build_hnerv_lowlevel_exact_eval_packet.py` now refreshes standalone dispatch readiness with explicit lane-claim context and writes an outer fail-closed readiness artifact when the active claim is missing.
- The q10 packet artifacts were refreshed so tracked custody no longer says `ready_for_exact_eval_dispatch=true` without an active claim.
- `tools/operator_briefing.py` now lists q10 in Phase 1 exact-eval packets ahead of the dominated one-byte `lgblock16` candidate.
- `tools/build_frontier_roadmap_status.py --operator-approved-exact-cuda` selects q10 and reports `needs_active_lane_claim_before_dispatch`, not a ready dispatch.

## Current Blockers

- `missing_lightning_environment`
- `missing_active_lane_dispatch_claim`

No phantom lane claim was created because the Lightning submit environment is not present. Per AGENTS.md, the claim should be created only immediately before the real dispatch attempt.

## Next Exact-Eval Procedure

1. Verify Lightning environment:

```bash
.venv/bin/python -c 'import os,sys; missing=[k for k in sys.argv[1:] if not os.environ.get(k)]; raise SystemExit(("FATAL: missing Lightning env: "+", ".join(missing)) if missing else 0)' LIGHTNING_SSH_TARGET LIGHTNING_REMOTE_PACT LIGHTNING_UPSTREAM_DIR LIGHTNING_TEAMSPACE LIGHTNING_STUDIO LIGHTNING_SDK_USER
```

2. Claim the q10 lane only when ready to submit:

```bash
.venv/bin/python tools/claim_lane_dispatch.py claim --lane-id pr106_q10_151byte_brotli --platform lightning --instance-job-id exact_eval_pr106_q10_151byte_brotli_20260507 --agent codex:gpt-5.5 --status active_exact_eval --notes 'pr106_q10_151byte_brotli HNeRV low-level Brotli exact CUDA eval; byte_delta=-151; source=PR106; member=0.bin; archive_sha256=626b1c76d318eaed45198dc26aea7ee98c8a05f685b840356cf5b621bcddeea7 bytes=186088; static_packet=experiments/results/hnerv_lowlevel_repack_pr106_q10_packet_20260507_codex/hnerv_lowlevel_exact_eval_packet.json'
```

3. Refresh the packet with operator approval and the active claim, then submit using the packet command from:

```bash
.venv/bin/python tools/operator_briefing.py --skip-dashboard --skip-reconciler --top 3
```

4. Harvest with adjudication required and record exact `contest_auth_eval.adjudicated.json` before making any score claim.

## Next Comprehensive Tranche

After q10 dispatch is either queued or blocked by environment, continue the highest-EV non-GPU work:

- Wave-Ω/SJ-KL: prove a charged runtime-consumed archive path, not only planning metadata.
- HDC2 entropy: reduce context-table overhead only if the byte mass can become archive-negative against the current HNeRV payload.
- Categorical/openpilot labels: keep PR91/HPM1 fail-closed until full decode/reencode parity and sidecar-free runtime consumption are proven.
- LA-pose/foveation: keep as calibrated planning signals until there is a charged artifact and runtime consumer.
