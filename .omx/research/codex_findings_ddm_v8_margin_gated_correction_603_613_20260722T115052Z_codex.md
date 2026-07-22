---
schema: codex_findings.v1
task: 603
feeds_task: 613
review_round: 1
reviewer: codex:gpt-5.6-sol
research_only: true
main_landing_review_required: true
---

# Round-1 findings — DDM v8 margin-gated correction

## Disposition

`PASS_AFTER_THREE_CUSTODY_FIXES`, research-only. Ruff and 51 focused tests pass. The measured
finite-tau formulation verdict is valid; no family, contest-score, or promotion claim is valid.

## Finding 1 — settled typed config was revalidated in the wrong data mode

- Severity: high, resume blocker.
- Observed: the first n64 launch refused before scorer work because strict Python-mode Pydantic
  parsing rejected JSON arrays for settled tuple fields.
- Fix: `52f02d8fd1` revalidates the SHA-bound embedded v7 config through strict JSON mode.
- Regression: the settled config dumps to JSON arrays and round-trips exactly.

## Finding 2 — inherited evaluator bridge schema was read flat

- Severity: high, final-report blocker.
- Observed: all four n64 candidates completed, then the final receipt hit `KeyError: d_seg`; v7
  stores distances under `segmentation` and `pose`.
- Fix: `53c66f4ee6` centralizes validated nested extraction and applies an explicit joint Seg/Pose
  guard to the inherited endpoint.
- Regression: flat bridge rows fail closed; the settled nested row is jointly feasible.

## Finding 3 — completed-receipt validation under-bound preserved stages

- Severity: high, P0 resumability/custody gap.
- Observed: initial validation hashed candidate archives and an arbitrary-length tau digest list but
  did not bind the candidate table to checkpoint rows, exactly six frames, per-stream ZIP homes,
  resume paths, or the settled v7 endpoint.
- Fix: `46d3a79370` validates every one of those surfaces and rejects weakened receiver manifests.
- Proof: after preserving pre-round1 receipts, both final receipts regenerated from checkpoints;
  sealed validation completes in 5.21/17.79 seconds.

## Re-derived measurement invariants

- Tight masks cover 4.53%/4.02% of sites and collapse v7 exact bytes 93.90%/94.54%.
- The collapsed archives still cost 2,629,076/9,360,569 bytes and miss both evaluator gates.
- Increasing tau improves Pose but monotonically worsens d_seg on both windows.
- Boundary dominates added margin payload; Movable/Road/Boundary dominate tight archive bytes.
- Only the inherited exact endpoint is joint feasible, at 43,112,153/171,332,654 bytes.
- n600 and contest CPU/CUDA remain unmeasured; pointer unchanged.

Canonical #603 remains 8/19 until MAIN reviews the draft row and fifth-anchor equations note.
