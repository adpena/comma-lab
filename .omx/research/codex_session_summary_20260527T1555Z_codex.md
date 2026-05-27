# Codex Session Summary - Target-Scoped Final Rate Attack Automation

UTC: 2026-05-27T15:55Z

## Scope

This landing canonicalizes contest-video overfitting as declared queue data,
not hidden tool behavior. The same target profile can bind a single contest
video, a corpus manifest, or a hybrid contest+corpus run, so final-rate attack
materializers remain reusable for other videos and future local MLX substrate
training outputs.

## Landed Implementation

- Added `frontier_rate_attack_target_profile.v1`, including target video SHA,
  corpus manifest reference, declared overfit policy, portability contract, and
  false-authority fields.
- Wired the profile through final-rate attack bootstrap contexts, backlog rows,
  queue metadata, experiment metadata, and bootstrap artifacts.
- Added CLI support to `tools/build_frontier_final_rate_attack_queue.py`:
  `--target-profile-id`, `--target-mode`, `--target-video`,
  `--target-corpus-manifest`, and `--allow-unready-target-profile`.
- Refactored the older feedback target-profile helper into the reusable
  scheduler module so the feedback loop and final-rate queue consume one
  canonical implementation.

## Verification

- `ruff check` passed on the edited scheduler/tool/test files.
- `pytest src/tac/tests/test_frontier_rate_attack_bootstrap.py src/tac/tests/test_frontier_rate_attack_feedback.py -q`
  passed: 72 tests.
- Queue-owned smoke:
  `.venv/bin/python tools/build_frontier_final_rate_attack_queue.py --queue-id frontier_final_rate_attack_target_scope_20260527Tscope --output-dir .omx/research/frontier_final_rate_attack_target_scope_20260527Tscope --results-root /Volumes/VertigoDataTier/experiments/results/frontier_final_rate_attack/target_scope_20260527Tscope --allow-materializer-overwrite --include-exact-readiness-followup --local-cpu-concurrency 2 --max-steps 8 --max-parallel 2 --execute`
  then a second bounded worker pass for the remaining dispatch-plan steps.
- Final observation: healthy, zero blockers, `status_counts={"succeeded": 10}`,
  zero failed steps, zero artifact-revalidation failures.

## Empirical Result

The target-scoped smoke found no rate-positive candidate on the current
`[contest-CPU]` frontier archive for the two executable member-level materializers
that applied to this archive:

- `packet_member_zip_header_elide_v1`: zero archive delta, exact-readiness skipped
  with `candidate_not_rate_positive`.
- `packet_member_recompress_v1`: zero archive delta, exact-readiness skipped
  with `candidate_not_rate_positive`.

Four materializers were correctly preserved as explicit target omissions rather
than silent absences:

- `packet_member_merge_v1`: requires at least two members.
- `renderer_payload_dfl1_v1`: missing `renderer.bin`, `masks.mkv`,
  `optimized_poses.pt`.
- `archive_section_entropy_recode_v1`: requires section manifest.
- `tensor_factorize_v1`: requires tensor manifest and rank/contract.

No score claim, promotion claim, rank/kill authority, or dispatch authority was
created.

## Next Codex Actions

1. Make `targeted_component_correction_queue` return a frozen, explicit
   no-actionable-rows queue instead of `None`, so response harvest and waterfill
   see a canonical blocker rather than a missing artifact.
2. Add target profile metadata to grouped inverse-steganalysis/waterfill queues,
   so PoseNet-null and SegNet-region attacks are declared against the same video
   or corpus scope as the rate attack.
3. Promote omitted materializers by generating the missing section/tensor
   manifests from receiver/runtime contracts, then rerun the same queue-owned
   campaign with all materializers executable.
4. Feed the zero-delta member-level results into the materializer acquisition
   model so future sweeps do not spend wall-clock on saturated single-member ZIP
   operations unless the archive grammar changes.
