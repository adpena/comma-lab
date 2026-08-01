# G14b — semantic frontier continuity for resumable long runs

Status: `DESIGN_FROZEN_FROM_REAL_FAILURE`; `research_only=true`; no score,
candidate, evaluation, promotion, or dispatch claim.

## Real failure that forced this contract

During the real resumable G14 n2 interaction run, the canonical upstream
leaderboard refresh changed the pointer artifact from SHA-256
`940491eaa6cd81fc66da5f93a497104e83426f306fd25a0d6c8f43bf2a311851`
to `2a61b052be496d3a9a1be1a9c230c8d179a788e61fd03472e50fc85832da94c6`.
Both reopened pointers independently derived the same live competitive target:

- score `0.172`;
- axis `official_leaderboard`;
- source `upstream_official_leaderboard`;
- submission `semantic-pose-HPAC_CPR1`, PR 130.

Only refresh/artifact identity changed. Nevertheless,
`verify_dynamic_frontier_target_snapshot()` compares the complete dataclass,
including pointer bytes, inode, mtime, and refresh timestamps. A refresh during
the allocator's double-decode therefore raised
`DynamicFrontierTargetError: canonical frontier pointer identity or derived
target changed after snapshot` and stopped the process. The run retained all
stage checkpoints and resumed from disk against the refreshed pointer; no
measurement or artifact was silently accepted and no completed stage was lost.

This is a real type error: exact pointer-artifact custody and competitive-target
continuity are different identities and must not share one equality operator.

## Two non-substitutable identity types

### `PointerArtifactIdentityV1`

Exact provenance of one opened canonical pointer object:

`(absolute_path, bytes, sha256, device, inode, mtime_ns, last_refreshed_utc,
source_snapshot_at_utc)`.

This identity belongs in start/resume/final receipts. It proves which bytes were
consulted. It is intentionally expected to change after an atomic refresh.

### `CompetitiveTargetIdentityV1`

The authority-bearing, recomputed control state used for admission:

`(target_score, selected_axis, selected_source, selected_source_kind,
selected_score_precision, selected_custody, selected_evidence_grade,
selection_rule, selected_archive_sha256, selected_lane_id,
selected_hardware_substrate, selected_submission_name, selected_pr_number,
selected_pr_url, selected_leaderboard_rank)`.

Every value must be derived from the currently reopened pointer constituents;
the serialized `effective_frontier` cache is never trusted. Official identity
fields are required when the selected source is the leaderboard. Local archive,
lane, and hardware identities are required when a local exact row is selected.
Refresh timestamps, file metadata, and the pointer SHA are not members of this
semantic identity.

## Required guards

The implementation must expose two explicit operations; a boolean mode is too
easy to call incorrectly.

1. `verify_exact_pointer_artifact(snapshot)` retains today's exact-object
   behavior for short custody-critical operations.
2. `verify_competitive_target_continuity(snapshot)` safely reopens the current
   pointer, verifies its exact file-read and freshness invariants, recomputes the
   selected row, and compares `CompetitiveTargetIdentityV1`.

The continuity guard may accept a new pointer artifact only when the semantic
target identity is exactly equal. It returns the newly reopened snapshot and a
typed `METADATA_REFRESH_SAME_TARGET` edge binding both artifact identities.
Callers preserve that edge in the run's pointer observation directory.

If score, selected source, authority axis, selected submission/archive, custody,
or selection rule changes, the current intra-stage operation refuses. The
resumable runner records `SEMANTIC_FRONTIER_REBASE_REQUIRED`, preserves all
completed checkpoints, reloads the new target at the next stage boundary, and
recomputes every target-dependent admission decision. It must never relabel an
old admission decision under a new frontier.

The original snapshot's timestamps need not remain fresh after a compatible
refresh; the currently reopened pointer and its selected source snapshot must
be fresh. This is necessary for runs longer than the 24-hour pointer freshness
window.

## Acceptance matrix

The permanent regression suite must prove:

1. exact same pointer object passes both guards;
2. metadata-only official refresh with the same score and PR passes continuity,
   fails exact-object equality, and emits a refresh edge;
3. same displayed score but a different official PR/name fails continuity;
4. a lower official score fails continuity and requests a resumable rebase;
5. a newly qualifying lower local CPU or CUDA row fails continuity and requests
   a resumable rebase;
6. a non-selected constituent update that leaves the selected target identity
   unchanged passes continuity;
7. a stale current pointer or stale selected official snapshot fails;
8. a forged cached `effective_frontier`, forged snapshot target, symlink,
   non-regular file, in-read mutation, or atomic name swap still fails;
9. a compatible refresh between the two receiver replays does not abort or
   change decoded bytes;
10. a semantic change between receiver replays aborts before measurement
    admission and preserves the prior stage checkpoint;
11. start, every compatible refresh, every semantic rebase, and final pointer
    artifacts remain separately content-addressed in the receipt.

## Landing order

The currently running G14 stable contract hashes
`dynamic_frontier_target.py`, `taskspace_whole_archive_allocator.py`, and the
runner. Do not mutate those files until that resumable run reaches a terminal
receipt. Then land the new identity type and guards, update the allocator to use
semantic continuity within a stage, update long-run resume/finalization to
record refresh/rebase edges, and run the existing adversarial pointer tests plus
the matrix above. One-shot custody tools may retain exact-object verification.

## Triality

- DSL: `PointerArtifactIdentityV1`, `CompetitiveTargetIdentityV1`, and typed
  refresh/rebase edges.
- DAG: stable read -> constituent recomposition -> freshness -> semantic
  comparison -> continue or checkpointed rebase.
- Equation: admission always uses
  `S_candidate < min(S_local_CPU, S_local_CUDA, S_upstream_official)` from the
  current pointer; artifact refresh is zero change only when the complete
  selected-target identity is unchanged.

HISTORICAL_PROVENANCE: first contract derived from the real G14 interruption
caused by a metadata-only canonical pointer refresh during double decode.
