# Main source-of-truth push and recovered safety-branch disposition

Generated: 2026-05-10

## Scope

This ledger records the adversarial review performed before pushing `main` as
the sole source of truth after the operator requested:

- commit and push all to `main`;
- merge all to `main` as the sole source of truth;
- preserve all signal with no acceptable loss;
- continue bit-level, categorical, arithmetic, and score-lowering work from the
  canonical current codebase.

This is not a score claim and no remote training, eval, or GPU dispatch was
started by this ledger.

```yaml
score_claim: false
dispatch_attempted: false
main_source_of_truth: true
branch_review_policy: preserve_signal_without_stale_reverse_merge
```

## Current main custody

- Branch: `main`
- HEAD before this ledger: `dbc0e79a0ee6941cb6bd8408405f15daff56dd86`
- Local status before this ledger: clean, `main` ahead of `origin/main` by 47
  commits
- Worktree inventory: one worktree at `/Users/adpena/Projects/pact`

Recent frontier-oriented commits already on `main` before this ledger included:

- `632a6834` add typed packet section transform bridge
- `9c2c6a65` add packet section transform CLI
- `c79bbbc7` scan hnerv brotli section recode opportunities
- `dbc0e79a` record pr103 arithmetic schema refresh

## Unmerged local branch review

Four unmerged branches were present locally. They are recovered safety/stash
refs, not active development branches:

| branch | tip | classification | disposition |
| --- | --- | --- | --- |
| `safety/stash-recovered-20260505T052046Z-stash0` | `1d9e7329` | recovered stash/snapshot ref | preserve as historical ref; do not merge into current `main` as code |
| `safety/stash-recovered-20260505T052046Z-stash1` | `710bd3a2` | recovered stash/snapshot ref | preserve as historical ref; do not merge into current `main` as code |
| `safety/stash-recovered-20260505T052046Z-stash2` | `e8ca384e` | recovered stash/snapshot ref | preserve as historical ref; do not merge into current `main` as code |
| `safety/stash-recovered-20260505T052046Z-stash3` | `c242e5dc` | recovered stash/snapshot ref | preserve as historical ref; do not merge into current `main` as code |

Adversarial finding: normal-merging these refs into the current `main` would
not preserve signal. Their `main..safety/...` diffs are stale recovered
snapshots with massive reverse deltas relative to current canonical `main`,
including thousands of apparent deletions of the modern system. Treating those
refs as code to merge would risk erasing the latest source-of-truth state.

Correct disposition:

1. Keep current `main` as the only promoted code/source-of-truth branch.
2. Preserve recovered refs as historical evidence refs, not as merge sources.
3. Record their existence and classification in this ledger so no signal is
   silently lost.
4. Continue all bit-level, arithmetic-coding, categorical, packet-compiler,
   HNeRV parity, and score-lowering work from current `main`.

## Score-lowering implications

The stale recovered refs do not change the current highest-EV score-lowering
sequence. The current canonical work remains:

- push and protect the current packet deconstruction and transform bridge;
- keep arithmetic/range/ANS, bit packing, categorical sidecars, tensor-section
  recoding, and deterministic packet compiler lanes active;
- avoid arbitrary inflate-time local-minimum shaving unless exact bytes,
  no-op proof, and custody show real score movement;
- continue T1/Ballé end-to-end, HNeRV PR95 parity, differentiable eval
  roundtrip, YUV6 gradient reachability, and archive-closed training;
- keep `[contest-CPU]`, `[contest-CUDA]`, proxy/MPS, and advisory curves
  separated in all claims.

## Result classification

```yaml
classification: custody_and_source_of_truth_guard
status: durable_signal_preserved
main_merge_action: no_stale_branch_merge
reason: recovered_stash_refs_would_reverse_delete_current_canonical_work
next_action: push_current_main_and_optionally_push_safety_refs_as_historical_refs
```
