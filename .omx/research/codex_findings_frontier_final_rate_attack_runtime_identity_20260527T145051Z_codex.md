# Codex Findings: Frontier Final Rate Attack Runtime Identity

Date: 2026-05-27T14:50:51Z

## Scope

Queue-owned final-rate attack against the current canonical contest-CPU frontier
archive, focused on materializer-chain harvest and observer revalidation
false-authority surfaces.

## Current Frontier Input

- Axis: `[contest-CPU]`
- Score: `0.19202062679074616`
- Archive: `experiments/results/v14_v2_dqs1_plus_fec10_substituted_20260526T023000Z/submission_dir/archive.zip`
- Archive SHA-256: `0a3abfe645c4fac0df9ea89237f25dd9bfc6b2471b897c36d7437795d27d1403`
- Archive bytes: `178546`

## Findings

1. The frontier queue builder could not reliably run from the canonical pointer
   because the default archive resolver only searched auth-eval request files
   unless the operator manually supplied an archive. It now also performs a
   bounded default search over promoted local submission packets:
   `experiments/results/*/submission_dir/archive.zip`,
   `experiments/results/*/submission/archive.zip`, and `submissions/*/archive.zip`.

2. The default results-root selector treated an existing external SSD path as
   usable even when macOS denied child creation under that path. It now probes
   child creation and falls back to local `experiments/results` when external
   storage is not actually writable.

3. The earlier observer refusal was valid: harvested source-runtime/materializer
   identity rows were carrying `runtime_adapter_ready=true` without a concrete
   runtime adapter tree or file identity. The harvest boundary now preserves the
   split:
   - `receiver_contract_satisfied=true` means the local receiver proof passed.
   - `runtime_adapter_ready=true` requires a concrete runtime adapter directory
     or adapter file identity.

## Queue Evidence

Successful run:

- Queue id: `frontier_final_rate_attack_runtime_identity_fix_local_20260527T1505Z`
- Queue artifacts: `.omx/research/frontier_final_rate_attack_20260527T144824Z/`
- Result artifacts:
  `experiments/results/frontier_final_rate_attack/frontier_final_rate_attack_runtime_identity_fix_local_20260527T1505Z/`
- Worker status: 10 succeeded, 0 failed
- Observer status: healthy, 0 blockers

Materializer outcomes on the current frontier:

| target_kind | max_saved_bytes | rate_positive_count | verdict |
| --- | ---: | ---: | --- |
| `packet_member_recompress_v1` | 0 | 0 | demote for matching archive class |
| `packet_member_zip_header_elide_v1` | 0 | 0 | demote for matching archive class |

Both harvested source queues now have:

- `receiver_contract_satisfied=true`
- `runtime_adapter_ready=false`
- `readiness_blockers=["candidate_not_rate_positive"]`
- no observer runtime-identity blocker

## Next Integration

The rate-only campaign should now consume these negative rows as posterior
signal and move away from same-position no-op outer ZIP work on this archive.
The next high-EV queue expansion is to run executable DFL1/merge/section
targets where receiver inputs are available, then feed positive rate deltas
into distortion-budget queues for grouped PoseNet-null and SegNet-region
waterfill cascades.
