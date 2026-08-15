# WD3 scorer-aware width distillation — build handoff

Disposition: **QUEUED-WITH-A-FIRE-ORDER**. The real build apparatus is landed and reviewed; no cache,
scorer, Metal, training, n60/n120/n600 evaluation, or contest evaluation ran. The exact pointer did not
move.

## Outcome and boundaries

- **MEASURED [scorer-free build apparatus]:** the tiny deterministic uniform packet is 3,569 B and the
  one-group adaptive packet is 3,570 B. Both payloads are retained with SHA-256; pack/unpack/repack is
  byte-identical. These are apparatus packets, not candidate archive rates.
- **VERIFIED VIA SOURCE INSPECTION AND TESTS:** the extension implements a real `WD3Q` sub-int16 adaptive
  packet consumed by a retained runtime, fixed-frame0 plus student-frame1 pairing, exact frozen-scorer
  call sites, all-n600 repeated cache materialization, original-target and teacher constraints, adaptive
  duals, per-cell/per-edge selection, measured quantization re-score, scorer-free non-W0 birth
  checkpoints, full resume state, retained evaluation/archive paths, seeded-n120 instance verdicts, and
  hard n600 same-instrument admission.
- **NOT MEASURED:** scorer quality, teacher-cache determinism, training convergence, any architecture
  verdict, archive-size saving on a trained WD3 model, n120/n600 score, contest score, or pointer movement.
- **AUTHORITY BOUNDARY:** all future local/MPS results remain advisory. Only the exact retained archive on
  contest-CPU or contest-CUDA can promote.
- **CUSTODY BOUNDARY:** bulk-producing entrypoints re-hash pinned inputs, require APDataStore storage,
  retain every payload, and refuse absent resume/lane/authorization state. The scorer-free verifier used
  injected pin facts; it did not re-hash the multi-GB inputs or invoke a scorer.

## Build receipt

- Implementation: `experiments/ddm_wd3_scorer_aware_width_distillation.py`
- Receiver: `experiments/ddm_wd3_student_receiver.py`
- Tests: `experiments/tests/test_ddm_wd3_scorer_aware_width_distillation.py`
- Machine receipt: `build_v1/BUILD_RECEIPT.json`
- Retained payload inventory: `build_v1/RETENTION_INVENTORY.json`
- Governed queue: `FIRE_ORDER.json`
- Verification: 31 WD2+WD3 tests passed; Ruff passed; two post-fix review-tracker passes recorded for all
  three Python files; P0 payload-retention detector found 0 findings across the 3-file changed-Python
  scope.

## RECALL EVIDENCE

Sources searched:

- full `.omx/research/` memo/receipt corpus, canonical research indexes, and the `sub015_DAG_*` graph with
  content queries `width distill`, `scorer-aware`, `adaptive quant`, `sub-int16`, `Road-Lane`, `selective`,
  `surgical`, `Schur`, `#816`, `zero-moment`, and `prefix bias`;
- canonical equations registry via
  `.venv/bin/python tools/list_canonical_equations.py --json`, queried for distillation, quantization,
  region/cell, selection, surgical work, and reset-state evidence;
- current task/hot-state and live lane ledger surfaces, plus the actual WD2 builder/receiver and scorer
  loader call graph.

What was found beyond the charter seeds and what changed:

- Older Hinton-distillation anchors use fixed distillation and pose weights. Those vehicle-specific
  constants were not transferred; WD3 uses T=2 structure with adaptive nonnegative constraint duals,
  including teacher-Pose matching, and exact nonlinear original-target Pose pricing.
- DAG FEED-gc15/#816 independently confirms that fresh-vs-warm confounds optimizer state with a measured
  zero-moment step excursion. This kept W0 warm and W0 reset as separate arms and made the reset-ramp
  bound executable rather than documentary.
- Older witness surfaces report much stronger Road-Lane dominance than the current WD3/pc2 surface.
  Those numbers were not transferred as WD3 measurements. They changed the implementation only by
  requiring generic unordered class-pair telemetry and an explicit Road-Lane gate, not a hard-coded
  Road-Lane-only selector.
- Prior selective-decoder and residual-sidecar rows show that a named surgical mechanism is not a score
  win by itself. The QS2/QS5 seam therefore stays fail-closed until a receiver-consumed repeat-identical
  Schur receipt and a measured residual edge map exist.

## Remaining governed work

The exact sequence and triggers are machine-readable in `FIRE_ORDER.json`. PID 63183 was observed absent
with `kill -0` in this process namespace on 2026-08-15, but MAIN must re-verify it at fire time. No current
WD3-owned live scorer or Metal lane claim was found in the bounded ledger search.

Vehicle frontier unchanged: **S 0.15959729295498598 @ 182,759 B [contest-CUDA T4, n600]**, archive SHA
`80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e`.
