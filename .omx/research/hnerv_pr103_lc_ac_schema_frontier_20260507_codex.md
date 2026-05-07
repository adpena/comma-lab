# HNeRV PR103 lc_ac Arithmetic Schema Frontier - 2026-05-07 Codex

## Scope

Implemented a disjoint PR103 `hnerv_lc_ac` arithmetic-schema custody surface.
This does not edit the active PR101 schema packer files and does not claim a
new score.

## Artifact

- manifest JSON:
  `experiments/results/hnerv_pr103_lc_ac_schema_20260507_codex/manifest.json`
- manifest MD:
  `experiments/results/hnerv_pr103_lc_ac_schema_20260507_codex/manifest.md`
- source archive:
  `experiments/results/public_pr_intake_full/public_pr103_intake_20260505_auto/archive.zip`
- source archive bytes: `178223`
- source archive SHA-256:
  `31881b2d23d027e6619f2d8df2fe35d4d207d08882ec673d6c1b7ff119f18c30`

## Arithmetic Schema Result

- merged AC section bytes: `153856`
- merged AC section SHA-256:
  `08f0a219b395e6783c522bacc8239f01dcb3d27d9f7f8d2291a7478d04859de7`
- decoded symbols: `237561`
- constriction decoder exhausted: `true`
- re-encoded stream byte-identical: `true`
- schema review gate: `ready_for_schema_review=true`
- score claim: `false`
- dispatch attempted: `false`
- ready for exact eval dispatch: `false`

Top model-gap targets from the decoded stream:

| rank | stream | symbols | model gap bytes | model floor bytes |
|---:|---|---:|---:|---:|
| 1 | `stem.weight` | 48384 | 46 | 31627 |
| 2 | `blocks.1.weight` | 46656 | 45 | 32478 |
| 3 | `blocks.0.weight` | 46656 | 33 | 32340 |
| 4 | `blocks.2.weight` | 34992 | 25 | 25462 |
| 5 | `blocks.3.weight` | 19440 | 9 | 14031 |

Interpretation: PR103's published arithmetic queue stream is now
byte-parity-provable as a source schema. The remaining decoded-model entropy
gaps are small inside PR103 itself, so the immediate frontier value is not a
blind PR103 stream polish. The useful next step is using the PR103 arithmetic
contract as a bounded source for a PR101/PR106x schema candidate only after an
archive builder and runtime adapter close old/new byte custody.

## Blockers

- `replay_fidelity:public_leaderboard_score_mismatch`
- `candidate_archive_missing`
- `old_new_archive_sha256_pair_missing`
- `candidate_runtime_adapter_missing`
- `strict_pre_submission_compliance_json_missing`
- `lane_dispatch_claim_missing`
- `exact_cuda_auth_eval_missing`

The public PR body reports about `0.195`, while the local exact T4 adjudication
for the same archive SHA recomputes `0.2277649714224471`. Treat PR103 as a
source-schema hidden gem, not a frontier score row, until a byte-different
candidate is exact CUDA evaluated.

## Verification

```bash
.venv/bin/python -m pytest src/tac/tests/test_hnerv_pr103_lc_ac_schema.py -q
.venv/bin/ruff check src/tac/hnerv_pr103_lc_ac_schema.py src/tac/tests/test_hnerv_pr103_lc_ac_schema.py tools/profile_hnerv_pr103_lc_ac_schema.py
.venv/bin/python -m py_compile src/tac/hnerv_pr103_lc_ac_schema.py tools/profile_hnerv_pr103_lc_ac_schema.py src/tac/tests/test_hnerv_pr103_lc_ac_schema.py
.venv/bin/python tools/profile_hnerv_pr103_lc_ac_schema.py --source-archive experiments/results/public_pr_intake_full/public_pr103_intake_20260505_auto/archive.zip --source-label PR103-hnerv-lc-ac --exact-adjudication-log experiments/results/lightning_batch/exact_eval_public_pr103_hnerv_lc_ac_t4_20260504T1245Z/adjudication.log --replay-fidelity-json experiments/results/public_hnerv_frontier_payload_profiles_20260504_codex/pr103_replay_fidelity.json --json-out experiments/results/hnerv_pr103_lc_ac_schema_20260507_codex/manifest.json --md-out experiments/results/hnerv_pr103_lc_ac_schema_20260507_codex/manifest.md
```

Results:

- pytest: `3 passed`
- ruff: `All checks passed`
- py_compile: passed
- tool manifest emitted with `score_claim=false`,
  `dispatch_attempted=false`, and `ready_for_exact_eval_dispatch=false`
