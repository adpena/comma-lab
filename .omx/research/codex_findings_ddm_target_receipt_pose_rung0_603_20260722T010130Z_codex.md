---
title: Codex findings — Task 603 target receipt and Pose rung zero
utc: 2026-07-22T01:34:04Z
task: 603
lane_id: lane_ddm_target_receipt_pose_rung0_603_20260722
review_round: 1
disposition: CLEAN_AFTER_FIX_FOR_TWO_SCOPED_BLOCKERS
research_only: true
score_claim: false
---

# Findings

## F1 — strict target-receipt JSON reload rejected canonical arrays — FIXED

`DirectDescriptionTargetPlaneReceiptV1` is strict and tuple-typed. The first reload path decoded JSON
to ordinary Python lists and then used object-mode validation, so the exact freshly materialized
receipt refused at `scorer_hw`, `chunks`, `subset_pair_ids`, and `pose6_source_shape`. The fix validates
directly from canonical JSON bytes, preserving strict JSON-to-tuple semantics, and adds a synthetic
50-chunk schema round-trip regression.

## F2 — checkpoint config/envelope reload had the same boundary defect — FIXED

Checkpoint construction revalidated its JSON-mode config dict through object-mode validation, and
checkpoint reload did likewise for tuple fields. The first real rung refused before stage completion.
Both paths now validate canonical JSON bytes. The regression runs a real bounded stage, writes the
immutable checkpoint, reloads it, and checks its continuation cursor and semantic argv.

## F3 — rung receipt lacked direct producer-code custody — FIXED

Deterministic output alone did not bind the producing implementation. The final receipt now contains
the exact module and CLI paths, source SHA-256s, and their last full git SHAs. The producer helper
refuses if runtime bytes do not match the claimed committed blob. After landing the fix, the rung was
re-materialized; all archive and checkpoint bytes matched the earlier run exactly.

## Re-derived invariants

- Target receipt SHA is exact and every referenced source file, aggregate plane stream, cache,
  archive, upstream snapshot, projection, and Pose6 code hash revalidates before optimization.
- The Pose stage is active in the objective and strictly reduces its registered debt without plane
  regression.
- Every stage checkpoint is immutable, primary and resumed checkpoint bytes match, and final archive,
  receiver output, and objective are identical after disk continuation.
- Labels remain `research_only=true`, `score_claim=false`, `candidate_archive=false`; the 1,585-byte
  artifact is explicitly `.not_a_candidate.zip`.
- No evidence supports any of the remaining 13 red blockers or pointer movement.

## Review disposition

`CLEAN_AFTER_FIX_FOR_TWO_SCOPED_BLOCKERS`. This is not a PRIMARY, n600 same-artifact, scorer, CPU/CUDA,
completion, promotion, or score seal. MAIN landing review is required.
