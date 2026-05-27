# Codex Findings - Frontier Rate Attack Target Scope Contract

UTC: 2026-05-27T15:52Z

## Finding

The final-rate queue builder was still archive-centric: it could run materializer
sweeps against the current frontier archive, but it did not carry the declared
video/corpus optimization target through the generated queue. That made the
contest-video overfit policy operationally implicit and made future corpus runs
too easy to fork into ad hoc command conventions.

## Landing

- Added reusable `frontier_rate_attack_target_profile` scheduler module for
  `contest_video_overfit`, `corpus_generalization`, and
  `hybrid_contest_plus_corpus` target binding.
- Refactored the existing feedback target-profile helper to import that module,
  preserving the prior public feedback API while removing the duplicate local
  implementation.
- Wired `tools/build_frontier_final_rate_attack_queue.py` to build a target
  profile by default and to accept `--target-video`, `--target-mode`,
  `--target-profile-id`, and `--target-corpus-manifest`.
- Wired `build_frontier_rate_attack_payloads(...)` to propagate compact target
  profile metadata into bootstrap artifacts, materializer contexts, backlog rows,
  and each experiment's metadata.
- Added a fail-closed `require_target_profile_ready` gate so corpus/generalization
  runs cannot silently execute without their declared video/corpus inputs unless
  the operator explicitly requests blocker-bearing planning artifacts.

## Smoke

Queue: `frontier_final_rate_attack_target_scope_20260527Tscope`

Artifacts:
- `.omx/research/frontier_final_rate_attack_target_scope_20260527Tscope/frontier_rate_attack_bootstrap.json`
- `.omx/research/frontier_final_rate_attack_target_scope_20260527Tscope/experiment_queue.json`
- `.omx/research/frontier_final_rate_attack_target_scope_20260527Tscope/final_observation.json`

Results root:
`/Volumes/VertigoDataTier/experiments/results/frontier_final_rate_attack/target_scope_20260527Tscope`

Observer summary:
- `healthy=true`
- `blockers=[]`
- `status_counts={"succeeded": 10}`
- `succeeded_artifact_steps=6`
- `succeeded_artifact_failure_steps=0`
- target profile: `contest_video_overfit`, `profile_ready=true`

The smoke reproduced the current PR110-class frontier archive as a local-only
materializer campaign. `packet_member_zip_header_elide_v1` and
`packet_member_recompress_v1` both closed through harvest, submission closure,
exact-readiness bridge, and local dispatch-plan construction. Both candidates
were skipped for exact dispatch because they were zero-delta / non-rate-positive.

## Authority

This landing does not claim score movement. All emitted target-profile,
materializer, dispatch-plan, and queue artifacts remain false-authority local
automation signals until an exact contest auth eval axis accepts a byte-closed
candidate.

## Next

Use this target scope contract as the binding layer for grouped rate/distortion
campaigns:

1. Keep default contest runs overfit to `upstream/videos/0.mkv`, explicitly and
   reproducibly.
2. Add corpus manifests for comma2k19 / comma10k19-style runs and run the same
   queue compiler with `--target-mode corpus_generalization` or
   `hybrid_contest_plus_corpus`.
3. Promote inverse-steg and repair-budget operations from contest-only leaf
   actions into target-profile-aware operation groups so discoveries can transfer
   across videos while preserving contest-specific overfit authority.
