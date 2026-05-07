# Q10 Exact Eval Dispatch Blocked - Missing Lightning Environment

timestamp_utc: 2026-05-07T15:42:51Z
score_claim: false
dispatch_attempted: false
remote_job_submitted: false

## Target

- lane_id: `pr106_q10_151byte_brotli`
- job_name: `exact_eval_pr106_q10_151byte_brotli_20260507`
- packet:
  `experiments/results/hnerv_lowlevel_repack_pr106_q10_packet_20260507_codex/hnerv_lowlevel_exact_eval_packet.json`
- archive bytes: `186088`
- archive SHA-256:
  `626b1c76d318eaed45198dc26aea7ee98c8a05f685b840356cf5b621bcddeea7`
- byte delta: `-151`
- expected rate-only score delta: `-0.00010054470192144788`

## What Happened

The q10 packet is static-ready and operator-approved for exact CUDA after a
Level-2 lane claim. The current shell cannot submit it because all required
Lightning environment variables are missing:

- `LIGHTNING_SSH_TARGET`
- `LIGHTNING_REMOTE_PACT`
- `LIGHTNING_UPSTREAM_DIR`
- `LIGHTNING_TEAMSPACE`
- `LIGHTNING_STUDIO`
- `LIGHTNING_SDK_USER`

`gws` is present, but it is Google Workspace CLI, not a GPU dispatch backend.
No remote GPU job was submitted.

## Coordination Ledger

Recorded a terminal non-dispatch row in ignored local state:

```text
.venv/bin/python tools/claim_lane_dispatch.py claim --lane-id pr106_q10_151byte_brotli --platform lightning --instance-job-id exact_eval_pr106_q10_151byte_brotli_20260507 --agent codex:gpt-5.5 --status refused_dispatch_missing_lightning_env --notes 'Operator approved q10 exact CUDA, but this shell has no LIGHTNING_SSH_TARGET, LIGHTNING_REMOTE_PACT, LIGHTNING_UPSTREAM_DIR, LIGHTNING_TEAMSPACE, LIGHTNING_STUDIO, or LIGHTNING_SDK_USER; no remote job submitted.'
```

## Next Unblock

Export or otherwise provide the six Lightning variables above, then rerun the
packet's command sequence:

1. verify env
2. claim active lane
3. refresh packet with operator approval
4. submit exact CUDA
5. harvest with expected archive SHA/bytes and adjudication required

Until then, the highest-value unblocked local work is PR102 custody correction
and PR101/PR103 runtime-adapter design, not creating phantom active claims.
