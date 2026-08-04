# NEXT-IF-RESUMED - ddm_uf1

1. Before any consumer reads `r9m_advisory_to_contest_cpu_calibration_prior` for qo1 or a later archive,
   fire a paired contest-CPU row for that exact archive, or keep the projection labeled queued.
2. When the scorer slot is free, refresh PF2, MS3-MS6, G3, G4, G2F, margin/saliency, and
   sensitivity-bitalloc rows only if a named consumer is about to use them.
3. For #891, first locate durable H_ab and mixed-partial inputs for a stale `(a,b)` row. If they are not
   available, route the row to full recompute; do not invent transport inputs.
4. If `gt_n600.npz` or the d2 pose trace sha changes, re-run the #931 prefix ratio derivation before
   using prefix negatives for pose-family decisions.
5. Preserve the scorer-free boundary: UF1 did not run `upstream/evaluate.py`, atlas jobs, or Modal jobs.
