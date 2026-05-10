# PR103 arithmetic-schema refresh (2026-05-10)

Generated: `2026-05-10T18:45:00Z`

`score_claim=false`; `dispatch_attempted=false`; `ready_for_exact_eval_dispatch=false`.

## Artifact

```text
experiments/results/hnerv_pr103_lc_ac_schema_refresh_20260510_codex/manifest.json
experiments/results/hnerv_pr103_lc_ac_schema_refresh_20260510_codex/manifest.md
```

Command:

```bash
.venv/bin/python tools/profile_hnerv_pr103_lc_ac_schema.py \
  --source-archive experiments/results/public_pr_intake_full/public_pr103_intake_20260505_auto/archive.zip \
  --source-label 'PR103 hnerv_lc_ac 20260510 refresh' \
  --exact-adjudication-log experiments/results/lightning_batch/exact_eval_public_pr103_hnerv_lc_ac_t4_20260504T1245Z/adjudication.log \
  --replay-fidelity-json experiments/results/public_hnerv_frontier_payload_profiles_20260504_codex/pr103_replay_fidelity.json \
  --json-out experiments/results/hnerv_pr103_lc_ac_schema_refresh_20260510_codex/manifest.json \
  --md-out experiments/results/hnerv_pr103_lc_ac_schema_refresh_20260510_codex/manifest.md
```

## Result

- source archive bytes: `178223`
- source archive SHA-256:
  `31881b2d23d027e6619f2d8df2fe35d4d207d08882ec673d6c1b7ff119f18c30`
- merged AC stream bytes: `153856`
- decoded symbols: `237561`
- reencoded byte-identical: `true`
- `ready_for_schema_review=true`
- `ready_for_archive_preflight=false`
- `ready_for_exact_eval_dispatch=false`

Top arithmetic targets by estimated model gap:

| rank | stream | symbols | model gap bytes | model cross-entropy floor bytes |
|---:|---|---:|---:|---:|
| 1 | `stem.weight` | 48384 | 46 | 31627 |
| 2 | `blocks.1.weight` | 46656 | 45 | 32478 |
| 3 | `blocks.0.weight` | 46656 | 33 | 32340 |
| 4 | `blocks.2.weight` | 34992 | 25 | 25462 |
| 5 | `blocks.3.weight` | 19440 | 9 | 14031 |

## Adversarial classification

The schema contract is useful but not a score candidate. Remaining blockers are:

- `replay_fidelity:public_leaderboard_score_mismatch`
- `candidate_archive_missing`
- `old_new_archive_sha256_pair_missing`
- `candidate_runtime_adapter_missing`
- `strict_pre_submission_compliance_json_missing`
- `lane_dispatch_claim_missing`
- `exact_cuda_auth_eval_missing`

This reinforces the 2026-05-10 Brotli scan: generic per-section Brotli
recompression is not the next PR103 path. The next PR103 work must be an
arithmetic-runtime adapter or a substrate conversion that consumes a
byte-different merged AC stream.

## Next implementation target

Build a no-score PR103 arithmetic transform planner that takes one decoded
symbol stream target, emits a candidate stream mutation proposal with expected
byte accounting, and refuses archive preflight until an inflate runtime adapter
can parse and consume the changed stream.
