# Codex Findings: Archive ZIP Repack Materializer

Generated: 2026-05-27T16:19:09Z

## Scope

This pass verifies and closes the remaining queue-intelligence layer around
archive-wide ZIP repacking as a typed, queue-owned, family-agnostic
materializer:
`archive_zip_repack_v1`.

The target is intentionally compatible with contest-video overfit campaigns and
with corpus-general campaigns. The target profile records whether the campaign
is optimizing the contest video, a broader corpus, or a hybrid, but the
materializer itself operates on archive/runtime contracts rather than on
PR110-specific scripts.

## Verified Wiring

- Verified `archive_zip_repack_v1` is present on the family-agnostic
  materializer surface.
- The materializer explores uniform ZIP compression plans and a deterministic
  greedy per-member compression plan.
- The emitted runtime-consumption proof verifies payload identity for every ZIP
  member and preserves false-authority semantics.
- Verified the target is wired through:
  - materializer registry;
  - materializer work queue;
  - grouped archive-state chaining;
  - frontier final-rate bootstrap defaults;
  - CLI runner and empirical sweep runner;
  - observation feedback;
  - entropy-position mapping at P15;
  - materializer chain harvest support;
  - queue observer schema recognition.
- Closed a small remaining feedback gap so blocker-bearing
  `archive_zip_repack_candidate.v1` artifacts participate in queue-feedback
  replan detection.

## Empirical Anchor

Queue smoke:

`frontier_final_rate_attack_archive_repack_local_20260527T1621Z`

Source archive:

- axis: `[contest-CPU]`
- score: `0.19202062679074616`
- archive bytes: `178546`
- archive SHA-256:
  `0a3abfe645c4fac0df9ea89237f25dd9bfc6b2471b897c36d7437795d27d1403`

Result:

- queue execution failed commands: `0`
- final queue status counts: `{"succeeded": 15}`
- materializer targets:
  - `archive_zip_repack_v1`
  - `packet_member_recompress_v1`
  - `packet_member_zip_header_elide_v1`
- observation count per target: `1`
- receiver contract satisfied for every target: `true`
- saved bytes for every target: `0`
- blocker for every target: `candidate_not_rate_positive`
- dispatch-ready rows: `0`
- target profile: `contest_video_overfit`, declared target video
  `upstream/videos/0.mkv`, target video SHA-256
  `2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9`

Interpretation: the current CPU frontier archive is already at this ZIP repack
fixed point for the tested stored/deflated levels and at the member-recompress
and ZIP-header-elide fixed point for the current adapter settings. This is a
useful negative because the queue now learns the archive-level repack floor
automatically instead of relying on a manual conclusion.

## Storage Note

First attempt:

`frontier_final_rate_attack_archive_repack_20260527T161914Z`

This correctly built the queue, but execution failed because the preferred
external results root under `/Volumes/VertigoDataTier/experiments/results/`
returned `Operation not permitted` for the new run directory. The local rerun
above is the authoritative materializer proof for this pass. Next storage work
should repair result-root permissions or route the water-bucket allocator to a
writable VertigoDataTier prefix before longer campaigns.

## Follow-Up

Run the widened final-rate queue after this target lands on `main`:

- default executable target set, including `archive_zip_repack_v1`;
- derived section manifests enabled;
- derived merge contracts enabled;
- exact-readiness follow-up enabled;
- target mode `contest_video_overfit`;
- local CPU/MLX calibration rows harvested into feedback.

If the widened run remains zero-delta at P15/P16, shift acquisition toward
pre-entropy and scorer-entropy positions: DFL1 grammar overhead, section recode
with real section manifests, tensor/codebook transforms, and PoseNet-null plus
SegNet-region water filling.
