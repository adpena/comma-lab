# PR103 Static Release Runtime Custody Hardening

Date: 2026-05-07
Owner: Codex
Evidence grade: engineering custody check
Score claim: false
Dispatch attempted: false

## Summary

The pre-submission compliance gate now checks the submitted `inflate.sh`
runtime tree against the runtime tree recorded by exact CUDA auth eval. This
closes a stale release-surface bug class: an archive SHA alone is not enough
when a static release wrapper delegates into repo-local runtime code.

## Current PR103-on-PR106 Finding

The existing PR103-on-PR106 exact CUDA score remains an exact runtime/archive
custody artifact for:

- archive SHA-256:
  `ec0890c2d2317dcad903ed37ffddb2794cd19c1df9effa057cb7f05af205e1ce`
- archive bytes: `185578`
- exact-eval runtime tree:
  `54db9e5ddee85ae7f486fae900ff3907932efb1c8d3062bc264b0e5c7456d8f6`

However, rerunning the stricter contest-final compliance gate against the
tracked `exact_eval_static_release_surface/` wrapper fails closed:

- `auth_eval_explicit_promotable_stamp`
- `submission_runtime_tree_matches_auth_eval`

Root cause: the tracked release wrapper has a different `inflate.sh` hash and
delegates to `submissions/pr103_pr106_final_runtime/inflate.sh`. The exact CUDA
run was performed against the delegated runtime, not the wrapper surface.

## Disposition

Do not treat the static wrapper surface as a contest-final releasable packet
until one of these is true:

1. A self-contained release runtime is exact-evaluated directly on CUDA and
   emits a fresh auth-eval JSON with explicit `promotion_eligible=true`,
   `score_claim_valid=true`, `evidence_grade=A++`, and matching runtime tree.
2. A reviewed delegate-root proof is added to the compliance checker, the
   wrapper/delegate relationship is encoded in the exact-eval runtime manifest,
   and the exact CUDA run is repeated through that wrapper.

Until then, the PR103-on-PR106 result remains a score anchor for rate-frontier
planning, not a final upload packet.
