# MLX per-pair master-gradient extractor — DE-ORPHANED INTO A PIPELINE — LANDED 2026-05-27

**Lane:** `lane_master_gradient_pipeline_automation_wire_in_20260527` (L1: impl_complete + strict_preflight + memory_entry)
**Subagent:** `mlx-master-gradient-pipeline-automation`
**Cost:** $0 — MLX-local/CPU; NO paid GPU dispatch.
**Evidence grade:** `macOS-MLX research-signal` (NON-PROMOTABLE per CLAUDE.md "MLX portable-local-substrate authority" + Catalog #192/#127/#323).

## Operator directive

> *"Shouldn't that be automated and wired and integrated into a pipeline rather than orphaned tool?"*

Plus the 7th AUTOMATED+COMPOUNDING+OPTIMAL standing directive + CLAUDE.md
"Results must become system intelligence" non-negotiable. The MLX per-pair
master-gradient extractor (`tac.master_gradient_mlx_extractor` /
`tools/extract_master_gradient_mlx.py`, landed `mlx_per_pair_master_gradient_authoritative_artifacts_landed_20260527.md`)
was an ORPHANED TOOL: it produced authoritative-shape `(N_bytes, N_pairs, 3)`
heuristic-prior artifacts but nothing AUTO-TRIGGERED it on frontier change,
nothing GOLDEN-VECTOR-guarded its math against drift, and no cathedral consumer
INGESTED its signal. This landing closes all three gaps.

## What landed — the pipeline (NOT an orphan tool)

### 1. Auto-trigger seam (canonical wire-in)

`src/tac/master_gradient_mlx_pipeline.py` (~470 LOC):
`auto_schedule_mlx_per_pair_extraction_for_frontier(...)` is the canonical
seam. It is **wired into the Catalog #343 frontier-pointer refresh path**:
`tac.canonical_frontier_pointer.auto_refresh_canonical_frontier_after_dispatch_outcome`
now calls it (fail-quiet) AFTER a successful pointer refresh. When the frontier
archive sha changes, the MLX per-pair extraction is auto-scheduled so the 5D
canvas / Dykstra Pareto solver / bit_allocator always have per-pair
heuristic-prior signal for the CURRENT frontier, not a stale one.

- **Deterministic + idempotent:** the fcntl-locked extraction-state ledger
  `.omx/state/mlx_per_pair_extraction_state.jsonl` (APPEND-ONLY per Catalog
  #110/#113; latest-row-wins per `(archive_sha256, n_pairs)` key) records a
  `scheduled` / `completed` / `no_op` / `error` row per trigger. A sha that
  already has a `scheduled` OR `completed` row is a no-op (`force=True`
  overrides). An already-landed artifact (found via the canonical manifest) is
  recorded `completed` and de-duped. Verified live: re-firing against the
  current frontier returns `no_op` with no duplicate row.
- **Fast tier default (64-pair); deep tier (full-600).** Default emits a
  `scheduled` row + the canonical CLI command to run out-of-band ($0
  MLX-local; the deep extraction is heavy and stays off the frontier-refresh
  hot path). `run_now=True` runs inline (opt-in).
- **Fail-quiet on the hot path:** any failure is captured in the verdict's
  `error` field + an `error` state row, NEVER raised from the
  dispatch-outcome / frontier-refresh seam (the canonical write already
  succeeded; the MLX schedule is downstream observability-only).
- Sister of the Catalog #1100 `tac.master_gradient.append_anchor_locked`
  post-anchor consumer fan-out pattern (same fail-quiet downstream-trigger
  contract).

### 2. Golden-vector regression test

`src/tac/tests/test_master_gradient_mlx_pipeline.py` — a deterministic
`TensorByteSpan` + sensitivity-dict fixture → a HAND-COMPUTED
`(N_bytes, N_pairs, 3)` tensor locks the
`project_per_tensor_sensitivity_to_per_byte` canonical math
(`per_byte = sens * |fp16_scale| / numel`, uniform across the mantissa span,
rate axis identically zero, per-pair axis preserved, span-past-archive-end
clamped, zero/non-finite scale skipped). The extractor's per-byte projection
output can no longer silently drift. 14 pipeline tests + 6 consumer tests = 20
pass [empirical:`.venv/bin/python -m pytest src/tac/tests/test_master_gradient_mlx_pipeline.py src/tac/cathedral_consumers/mlx_per_pair_master_gradient_consumer/tests/` → 20 passed 2026-05-27].

### 3. Cathedral consumer (Catalog #335 + auto-discovery + fires in main())

NEW auto-discovered package
`src/tac/cathedral_consumers/mlx_per_pair_master_gradient_consumer/`.
Canonical contract (`CONSUMER_NAME` / `CONSUMER_VERSION` /
`CONSUMER_HOOK_NUMBERS` + `update_from_anchor` + `consume_candidate`). It is
**DISTINCT from the sister `master_gradient_per_pair_consumer`** package: that
sister reads the PyTorch-AUTHORITY `master_gradient_anchors.jsonl` surface;
THIS consumer reads the MLX HEURISTIC-PRIOR surface
(`.omx/state/mlx_research_signal_manifest.jsonl` NON-PROMOTABLE rows, which
REFUSE anchor authority by construction).

- **Tier-A observability-only** per Catalog #341 + #357: every return value
  carries `predicted_delta_adjustment=0.0` + `promotable=False` +
  `axis_tag="[predicted]"`. The MLX heuristic prior CANNOT leak into a score
  signal.
- **Verified it auto-discovers AND fires:** `discover_compliant_consumer_modules()`
  returns 71 consumers including `mlx_per_pair_master_gradient_consumer`;
  `invoke_cathedral_consumers_on_candidates([CandidateRow(archive_sha256=fec6_frontier_sha)])`
  (the invoker called from `main()` at lines 8190/8360 per Catalog #336/#337)
  produces the consumer's row with `delta_adj=0.0 / promotable=False /
  axis_tag=[predicted]` and the correct landed-artifact annotation. Catalog
  #335 + #341 STRICT gates both pass with 0 violations on the new package.

### 4. Canonical equation #344 (RECALIBRATE_ON_NEW_ANCHORS preserved)

`mlx_per_pair_master_gradient_per_byte_fd_v1` was already registered with
`next_recalibration_trigger=when_3+_new_empirical_anchors_in_domain`
(`RECALIBRATE_ON_NEW_ANCHORS`). I augmented its producer/consumer lists
(append-only re-register via `register_canonical_equation`, latest-row-wins):

- `canonical_producers += tac.master_gradient_mlx_pipeline`
- `canonical_consumers += tac.cathedral_consumers.mlx_per_pair_master_gradient_consumer`, `tac.canonical_frontier_pointer`

The Catalog #371 `auto_recalibrate_from_continual_learning_posterior` path
recognizes the `RECALIBRATE_ON_NEW_ANCHORS` trigger and will consume future
contest-axis anchors for this equation.

## RIGOR-GATING CLAUSE (explicit — signal-AGNOSTIC plumbing)

Consumption trust level is gated by the rigor-review verdict
(`master_gradient_analysis_rigor_signal_review_*`). A sister RIGOR-REVIEW
subagent is concurrently adjudicating whether the MLX per-pair signal is a
genuine heuristic prior or a uniform-mantissa-projection ARTIFACT. **If that
review finds projection-artifact bias, the pipeline consumer remains Tier-A
observability-only and a PyTorch-autograd authority cross-check
(`tools/extract_master_gradient.py`) gates any promotion.** The pipeline is
built signal-AGNOSTIC: it is SAFE regardless of the rigor verdict because it
NEVER promotes the signal to a score adjustment (every verdict + consumer
return carries `promotable=False` + `score_claim=False` +
`predicted_delta_adjustment=0.0`). It does NOT bake the "class-shift is the
only lever" conclusion into the pipeline as fact. The consumer's annotation
text surfaces the rigor-gating clause inline so a downstream reader cannot
mistake the heuristic prior for authority.

## 6-hook wire-in (Catalog #125)

1. **Sensitivity-map contribution** — ACTIVE: the per-pair heuristic-prior
   artifact IS a per-pair per-axis sensitivity map; the consumer declares
   `HookNumber.SENSITIVITY_MAP` and the 5D canvas populator consumes the
   artifact downstream.
2. **Pareto constraint** — N/A directly (the artifact's per-pair seg/pose/rate
   decomposition feeds the Dykstra Pareto polytope solver via the 5D canvas,
   not via a new direct Pareto constraint in this landing).
3. **Bit-allocator hook** — N/A directly (per-byte per-pair sensitivity is the
   bit-allocator's signal, consumed via the 5D canvas downstream; no new
   bit-allocator registration in this landing).
4. **Cathedral autopilot dispatch hook** — ACTIVE: the NEW
   `mlx_per_pair_master_gradient_consumer` auto-discovers (Catalog #335) AND
   fires in `main()` via `invoke_cathedral_consumers_on_candidates` (Catalog
   #336/#337), Tier-A observability-only.
5. **Continual-learning posterior update** — ACTIVE: canonical equation
   `mlx_per_pair_master_gradient_per_byte_fd_v1` carries
   `RECALIBRATE_ON_NEW_ANCHORS`; the pipeline is a registered producer and the
   consumer + frontier-pointer are registered consumers; future contest-axis
   anchors recalibrate it (Catalog #371).
6. **Probe-disambiguator** — N/A (single defensible interpretation: per-tensor
   FD projected per-byte; the rigor-gating clause IS the disambiguation between
   heuristic-prior vs projection-artifact, adjudicated by the sister review).

## Sister coordination (Catalog #314/#340)

- **RIGOR-REVIEW sister** (READ-ONLY on the extractor + registry; owns its
  verdict memo `master_gradient_analysis_rigor_signal_review_*`). I own the
  pipeline module + the new consumer package + the equation row augmentation +
  the frontier-pointer seam edit. The rigor-gating clause makes my plumbing
  SAFE regardless of its verdict.
- **PACT-NeRV sister** owns `*pact_nerv*` — NOT touched.
- Catalog #340 sister-checkpoint guard + POST-EDIT `--expected-content-sha256`
  protect any collision at commit.

## Files

- `src/tac/master_gradient_mlx_pipeline.py` (NEW — auto-trigger seam)
- `src/tac/canonical_frontier_pointer.py` (MODIFIED — Catalog #343 seam wire-in)
- `src/tac/cathedral_consumers/mlx_per_pair_master_gradient_consumer/__init__.py` (NEW — Tier-A consumer)
- `src/tac/cathedral_consumers/mlx_per_pair_master_gradient_consumer/tests/__init__.py` (NEW)
- `src/tac/cathedral_consumers/mlx_per_pair_master_gradient_consumer/tests/test_mlx_per_pair_master_gradient_consumer.py` (NEW — 6 tests)
- `src/tac/tests/test_master_gradient_mlx_pipeline.py` (NEW — 14 tests incl. golden-vector)
- `.omx/state/canonical_equations_registry.jsonl` (APPEND — equation producer/consumer augmentation)
