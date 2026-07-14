# RGB cargo-cult scrutiny — DAG FEED — 2026-07-14

Feed id: `FEED-500-rgb-cargocult-taskspace-replacements`  
Lane: `rgb_cargocult_scrutiny_optimal_replacement`  
Research only: `true`  
Pointer: `0.1910828242 [contest-CPU Linux x86_64]`, unchanged

## Nodes

| Node | State | Input | Output / owner |
|---|---|---|---|
| RGB-BOUNDARY | KEEP / HARD-EARNED | exact RGB frame, RGB-to-YUV6, R, uint8, scorer cells | exact scorer input; byte-close/evaluator |
| METRIC-HELPER | IN-FLIGHT SIBLING | winner/rival/centered-logit decision Jacobian, renderer pullback, optimizer preconditioner, density-ratio custody | `metric_id=argmax_native_vjp_fidelity_v1`; per-state `receipt_schema=reachable_decision_geometry_fidelity.v1`; owner `surrogate_vjp_fidelity_metric` |
| METRIC-SELECT-N600 | NO-VERDICT-DATA-CUSTODY | real n600 logits/costates/perturbations and on-policy density/support | selected metric receipt; same sibling owner |
| CARRIER-RANK-N600 | QUEUED `$0` | saved #205/v9 state, exact R/scorers, current carrier bytes | decision/Pose Jacobian spectrum + task-rank/rate receipt; next build lane |
| DECISION-CARRIER | DESIGN | partition `phi`, tie residual, `xi`, low-rank Pose/chroma tangents | render-only RGB carrier; byte-close consumer |
| PALETTE-CHROMA-SOLVE | DESIGN | 15-D palette/chroma finite differences, winner/rival Fisher, Pose trust | decision-optimal init/loss; training-loss sibling consumes metric |
| BASIS-COMPILER | IN-FLIGHT SIBLING | Task497 family IDs + selected metric id | `tac.witness_dsl.optimal_basis_20260714`; owner `optimal_basis_beyond_fourier_20260714` |
| SELECTOR-REWEIGHT | DESIGN | HPRC/HiNeRV/shared-loss candidate rows | score-debt-per-byte ordering; substrate owners after metric selection |
| EXACT-ROW | GOVERNED TERMINAL | byte-closed n600 candidate, exact archive | upstream CPU/CUDA row; only node permitted to move pointer |

## Edges

```text
RGB-BOUNDARY ───────────────┐
                            ├─> METRIC-HELPER ─> METRIC-SELECT-N600
saved state + carrier bytes ┘                         │
                                                     ├─> CARRIER-RANK-N600 ─> DECISION-CARRIER ─┐
                                                     ├─> PALETTE-CHROMA-SOLVE ──────────────────┤
                                                     ├─> BASIS-COMPILER ─────────────────────────┤
                                                     └─> SELECTOR-REWEIGHT ──────────────────────┤
                                                                                                 v
                                                                                         EXACT-ROW
```

## Trigger / stop rules

1. Do not duplicate `src/tac/scorer_surrogate/vjp_fidelity.py`, the stage-metric DSL, or the Task497 basis compiler.
2. `METRIC-SELECT-N600` may not consume the retained n120 receipt as n600 authority. Missing live/cache density ratios remain distribution-custody absence, not evidence against the family.
3. The Task497 warm-start observation (`OFF 0.004244 / 109559 values`, along8 `0.0042590586 / 111095`, along26 `0.004286 / 111095` at ep675) scopes only those formulations/states. Fresh-start/different-frame basis families remain open.
4. R-MTF amplitude evidence scopes only ambient isotropic RGB amplitude inversion. Task-directed tie/preimage correction remains open.
5. Every learned carrier landing must be resumable, stage-checkpointed, byte-closed, storage-preflighted, and auto-cleaned under the v7.5 operating contract.
6. No pointer update without `EXACT-ROW`; advisory metric improvements cannot promote.

## Six-hook wire-in

1. Sensitivity map: decision pullback/Fisher replaces RGB saliency as the common field.
2. Pareto: exact `(d_seg,d_pose,archive_bytes)` remains terminal; task-rank/RD receipts are acquisition-only.
3. Bit allocator: allocate by score-units-per-byte in the selected decision geometry.
4. Autopilot: route only after metric receipt/family id is custody-complete.
5. Continual learning: assumption split registered under T1 id `rgb_cargocult_scrutiny_20260714_t1`.
6. Probe disambiguator: metric/basis/palette/chroma alternatives remain multiple typed modes until n600 math arbitrates. The selection receipt uses `reachable_decision_preconditioner_selection.v1`, distinct from the law id and per-state schema.

## Source finding

Full table and file:line evidence: `.omx/research/codex_findings_rgb_cargocult_scrutiny_optimal_replacement_20260714_codex.md`.
