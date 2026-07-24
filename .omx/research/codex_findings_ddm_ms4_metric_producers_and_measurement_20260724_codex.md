# Codex findings — DDM MS4 metric producers and measurement

Evidence is `[macOS-CPU frozen-scorer advisory]`; `score_claim=false`; pointer
`0.1910828242 [contest-CPU]` is unchanged. MAIN landing review is required.

1. **CRITICAL — three requested bucket producers are not identified by the
   landed PF2 atlas.** All 1,200 rows preserve the requested semantic keys, but
   zero rows identify `pair_ids`, `receiver_actuator_id`, and `direction_id`.
   A semantic bucket is not a scorer input. Duplicating a pair-level tensor over
   all keys would be a false per-bucket measurement. Scope:
   `INSTANCE(current PF2 atlas) < FORMULATION < FAMILY < PARADIGM`.

2. **COMPLETE — Pose output-space geometry is now real n600/batch32 custody.**
   The governed run measured ordered pair IDs 0..599 with four Torch threads,
   19 immutable full-n600 blocks, exact rank-6 factors `I/sqrt(6)`, and positive
   `0.05` contest-budget tubes. All 600 analytic quadratics converged. The
   existing strict loader accepts the Pose component as `COMPLETE`.

3. **MEASURED — batch geometry was not silently treated as invariant.** Fresh
   batch32 centers differ from registered cached centers by maximum absolute
   `1.9073486328125e-05` (mean of per-pair maxima
   `3.608719756205877e-06`). This is small but nonzero, so the old batch16
   lineage was correctly retained as control-only.

4. **PARTIAL — the bundle remains honestly inactive.** Seg, composite-R, and
   dual diagnostics each carry `PF2_BUCKET_INPUT_ASSIGNMENT_ABSENT`; the strict
   loader reports `BUNDLE-PARTIAL`, `scorer_metric_active=false`, and
   `pose_tube_active=false`. MS2 rerun, PF2R, and RD1 duals are not fireable.

5. **P0 resumability and order were exercised.** The run preserved top24,
   top64, stratified-control24, and full-n600 stage receipts plus 23 scorer
   blocks across the four stages (1 + 2 + 1 + 19). The top24 checkpoint was
   reused unchanged during the full run. Bulk JSON lives on VertigoDataTier;
   no source or cache bytes were moved or deleted.

6. **FIRST RUNG — land the missing assignment schema, not another metric
   approximation.** Each PF2 row needs a SHA-bound 1:1 measurement assignment
   naming pair IDs 0..599, receiver actuator, and direction. That single edge
   unlocks honest hard-tail-first Seg, composite-R, and matched dual producers.

## Round-1 adversarial review

The review found and fixed three custody bugs before the full run: the first
governor check looked for a nonexistent attestation variable instead of
`TAC_GOVERNED_ADMISSION`; volatile free-space telemetry was initially embedded
in immutable resume identity; and advancing stages initially targeted one
immutable receipt path. The final implementation uses the canonical governed
marker, keeps scientific input custody stable, and materializes stage-scoped
bundle receipts.

The remaining limitation is scientific, not mechanical: no current artifact
maps the 1,200 PF2 keys to receiver inputs and perturbation directions.
