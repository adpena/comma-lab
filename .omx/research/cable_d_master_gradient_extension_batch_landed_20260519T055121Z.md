# Cable D master-gradient extension batch landed — D2+D3+D4 batched

**Date:** 2026-05-19T05:51:21Z
**Authority:** Cable D D2+D3+D4 per `.omx/research/integrated_battle_plan_priority_queue_dag_cables_gates_20260519T052801Z.md` (commit `6a1e94a63`); CLAUDE.md "PER-PAIR MASTER GRADIENT — wire-in coverage audit across ALL consumers" operator standing directive
**Lane:** `lane_cable_d_master_gradient_extension_batch_20260519` (L1: impl_complete + memory_entry)
**Sister coordination:** Catalog #230 ownership map honored (4 disjoint sister subagents in flight)

---

## Canonical-vs-unique decision per layer

| Layer | Decision |
|---|---|
| Plan structure | ADOPT_CANONICAL — Cable D D2+D3+D4 batched per battle plan |
| Consumer dataclass shape | ADOPT_CANONICAL — frozen dataclass + (entries, n_pairs, archive_sha256, measurement_axis, measurement_hardware) pattern from existing consumers 1-6 |
| Sidecar JSON emission | ADOPT_CANONICAL — `consumer_output_path` + `write_consumer_sidecar_json` helpers automatically inject `score_claim=False` per CLAUDE.md "Apples-to-apples evidence discipline" |
| CLI dispatch (xray tool) | ADOPT_CANONICAL — argparse + lazy matplotlib import pattern from `tools/visualize_scorer_drift.py` |
| Analytical surface manifest schema | UNIQUE — new schema `master_gradient_analytical_surface_manifest_v1` (no canonical sister at this layer) |
| Wire-in approach | ADOPT_CANONICAL — produce signals here; SISTER 3 owns wire-in into `tools/cathedral_autopilot_autonomous_loop.py` per Catalog #230 ownership map |

## 9-dimension success checklist evidence

- **UNIQUENESS**: Cable D D3 v3 wave covers consumers 7-14, closing the catalog from v2 (5 implemented + 15) to v3 (15 implemented total). The xray tool is the first canonical visualization for master-gradient.
- **BEAUTY+ELEGANCE**: each new consumer follows the exact frozen-dataclass + sidecar-JSON pattern of consumers 1-6; no orchestration layer added.
- **DISTINCTNESS**: each consumer addresses a different analytical surface (Pareto envelope / λ_R bisection / LoRA / coding budget / engineered correction / KKT / Volterra / decoder pruning). All wire to different downstream hooks (#2 Pareto / #3 bit-allocator).
- **RIGOR**: 61 dedicated tests across 3 new test files (38 consumers 7-14 + 16 xray + 7 manifest); 193 total master_gradient tests passing.
- **OPTIMIZATION-PER-TECHNIQUE**: each consumer is a pure NumPy implementation, deterministic given inputs + seed, no PyTorch/GPU dependency.
- **STACK-OF-STACKS-COMPOSABILITY**: consumer 12 (KKT) consumes consumer 8 (λ_R); consumer 8 consumes consumer 7 (Pareto envelope). Composable producer chain.
- **DETERMINISTIC-REPRODUCIBILITY**: every consumer takes `random_seed` (where stochastic) or is fully deterministic; tests pin determinism.
- **EXTREME-OPTIMIZATION-PERFORMANCE**: consumer 13 (Volterra) carries explicit `downsample_bytes` parameter to keep N_pairs² × N_bytes tractable on canonical (178000, 600) shape.
- **OPTIMAL-MINIMAL-CONTEST-SCORE**: signals produced feed the autopilot ranker (hook #4) which routes per-pair dispatch ranking; structural enabler for `frontier_protecting` work per battle plan.

## Observability surface

- **Per-consumer sidecar JSON**: every consumer optionally emits `.omx/state/master_gradient_consumers/<consumer_id>_<archive_sha_short>_<utc>.json` per canonical path
- **CLI**: `tools/master_gradient_xray.py --list-plots` (JSON manifest), `tools/extract_master_gradient.py list-analytical-surfaces` (JSON coverage manifest)
- **Test traces**: pytest verbose `-vv` provides per-test signal
- **Sidecar contents include canonical compliance tags**: `score_claim=False` / `promotion_eligible=False` / `ready_for_exact_eval_dispatch=False` / `evidence_grade=[diagnostic; master-gradient consumer]`

## Cargo-cult audit per assumption

| Assumption | Classification |
|---|---|
| Per-pair Pareto envelope is sweep over rate-gradient ordered bytes | HARD-EARNED (canonical Pareto frontier construction; cite Boyd convex feasibility) |
| Per-pair λ_R from OLS slope | HARD-EARNED (canonical Lagrangian stationarity dD/dR = -λ_R) |
| LoRA supervision score = mean × std | HARD-EARNED (mean = leverage; std = pair-specificity; LoRA is per-pair adapter) |
| Coding budget proportional to pose-gradient L1 | HARD-EARNED (operating-point-aware per CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent" — pose dominates at PR106 frontier) |
| Engineered correction = leverage × variance | HARD-EARNED (Catalog #303 cargo-cult-unwind: combines magnitude and variance into single leverage score) |
| KKT residual = ||dD + λ_R · dR||_2 | HARD-EARNED (canonical KKT certificate per Boyd) |
| Volterra coupling = cosine similarity on distortion profiles | HARD-EARNED (Volterra second-order kernel canonical form; cosine is the canonical normalization) |
| Dead byte = joint low aggregate AND low per-pair variance | HARD-EARNED (Catalog #220 sister: a byte with low aggregate but high per-pair variance has leverage; can't be pruned) |

## Horizon class

`frontier_protecting` — the producer/consumer wire-in is hardening infrastructure that ENABLES `frontier_breaking` cathedral autopilot ranking via sister 3's downstream wire-in. The signal produced here does not itself lower the contest score; it enables the optimization loop that lowers the contest score.

---

## Cable D D2 outcome — analytical surface coverage manifest

**Task #887/#890**: extend master-gradient parsers across analytical surfaces.

**What was empirically tractable in this subagent's scope:**
- Document the 16 analytical surfaces that consume master-gradient (vs prior "47% under-wired" framing which was inherited from a different audit scope — task #890 cited a partial-implementation gap that became outdated after consumers 7-14 landed).
- Provide an operator-facing CLI subcommand `tools/extract_master_gradient.py list-analytical-surfaces` that emits a JSON coverage manifest per canonical schema `master_gradient_analytical_surface_manifest_v1`.
- The manifest schema carries per-surface (id, module_path, coverage_state ∈ {active, indirect, decorator, pending}, wire_in_hook, notes) for the cathedral autopilot ranker + operator review.

**Current coverage from the manifest:**
- 13/16 active (81.25%) — direct imports of `tac.master_gradient`
- 2/16 decorator (12.50%) — wrap master_gradient via decorator (boosting, compress_time_optimization)
- 1/16 pending (6.25%) — `tac.optimization.pareto` Pareto-constraint consumer (sister subagent owns wire-in)
- 0/16 indirect

**What was DEFERRED to a separate symposium-grade subagent:**
- Lift PR106_format0d and PR107_apogee from `_DETECTION_ONLY_PROJECTORS` to `_SUPPORTED_PROJECTORS`. Each requires a full byte-grammar two-pass Jacobian projector implementation per the `parser_notes` fields. This is council-grade work (per CLAUDE.md "Design decisions — non-negotiable"); the parser_notes carry the canonical reactivation criteria.
- Per CLAUDE.md "Forbidden premature KILL without research exhaustion": no kills; the detection-only state is preserved as research-deferred.

## Cable D D3 outcome — 8 new consumers landed (7-14)

**Task #799**: builder wave for consumers 7-15 per `tac.master_gradient_consumers` catalog.

Consumers 1-6 + 15 were already implemented. Consumers 7-14 (8 new consumers) landed in `src/tac/master_gradient_consumers.py`:

| # | Consumer | Wire-in Hook | LOC |
|---|---|---|---|
| 7 | `per_pair_pareto_envelope` | hook #2 Pareto constraint | ~80 |
| 8 | `per_pair_lagrangian_lambda_bisection` | hook #2 Pareto constraint | ~120 |
| 9 | `per_pair_lora_supervision_signal` | hook #3 bit-allocator | ~90 |
| 10 | `per_pair_coding_budget_allocation` | hook #3 bit-allocator | ~95 |
| 11 | `engineered_correction_targeting` | hook #3 bit-allocator | ~115 |
| 12 | `per_pair_kkt_residuals` | hook #2 Pareto constraint (consumes consumer 8) | ~85 |
| 13 | `per_pair_volterra_cross_terms` | hook #2 Pareto constraint | ~115 |
| 14 | `gradient_informed_decoder_pruning` | hook #3 bit-allocator | ~110 |

Total: 8 new consumers, ~810 LOC, all canonical-pattern frozen dataclasses + sidecar JSON.

**Tests**: 38 tests in `src/tac/tests/test_master_gradient_consumers_7_to_14.py`. ALL PASS.

**`__all__` exported**: 24 new symbols (8 functions + 16 dataclasses).

**Catalog comment updated**: v1 (1, 2, 3, 4, 5, 6, 15) + v2 (7, 8) + v3 (9, 10, 11, 12, 13, 14) — all 15 consumers complete.

## Cable D D4 outcome — master_gradient_xray.py + 5 plot types

**Task #797**: `tools/master_gradient_xray.py` + 5 canonical plot types.

Operator-runnable CLI tool at `tools/master_gradient_xray.py` with 5 canonical plot types:

1. **`per_pair_distribution`** — histogram per-pair |g| L1 distribution across pairs (per axis). Reveals per-pair leverage spread; hard-pair pattern detection.
2. **`per_byte_heatmap`** — top-K bytes × 3 axes heatmap of aggregate sensitivity. Identifies canonical leverage points per axis.
3. **`cumulative_by_rank`** — Pareto leverage curve (rank vs cumulative sensitivity fraction). Annotates top-1% + top-10% operating points.
4. **`cross_substrate_correlation`** — cross-substrate cosine similarity matrix (requires ≥2 anchors). Reveals Wyner-Ziv complementary candidates.
5. **`wyner_ziv_flow`** — per-substrate-section gradient breakdown when archive_layout sections present. Degrades gracefully to single-section when no manifest.

**Discipline**:
- `/tmp` output paths REFUSED at the CLI guard
- Lazy matplotlib import per sister pattern in `tools/visualize_scorer_drift.py`
- Watermark applies advisory tag when anchor's `evidence_grade` contains "advisory" or `measurement_hardware` contains "mps" (per CLAUDE.md "MPS auth eval is NOISE" + Catalog #192)
- `--plot all` emits all 5 plots into a directory; individual plots accept either file or directory `--output`

**Tests**: 16 tests in `src/tac/tests/test_master_gradient_xray.py`. ALL PASS.

---

## Test summary

| File | Tests | Status |
|---|---|---|
| `test_master_gradient_consumers_7_to_14.py` | 38 | PASS |
| `test_master_gradient_xray.py` | 16 | PASS |
| `test_extract_master_gradient_analytical_surfaces.py` | 7 | PASS |
| Total NEW tests | **61** | **PASS** |
| Pre-existing `test_master_gradient_consumers.py` | 30 | PASS (updated 1 assertion to superset semantics) |
| Pre-existing `test_master_gradient_consumers_rashomon.py` | 21 | PASS |
| Full `test_master_gradient_*` suite (11 files) | **193** | **PASS** |

## 6-hook wire-in declaration (per Catalog #125)

1. **SENSITIVITY MAP**: ACTIVE — consumers 7+8 produce per-pair Pareto + λ_R signals that feed `tac.sensitivity_map.axis_level_reweight`; consumer 12 (KKT) feeds the same surface
2. **PARETO CONSTRAINT**: ACTIVE — consumers 7+8+12+13 produce canonical Pareto-constraint signals; SISTER 3 owns wire-in into `tac.optimization.pareto` (the 1 remaining "pending" analytical surface)
3. **BIT-ALLOCATOR**: ACTIVE — consumers 9+10+11+14 produce canonical bit-allocator signals; SISTER 3 owns wire-in into `tac.optimization.bit_allocator`
4. **CATHEDRAL AUTOPILOT DISPATCH**: ACTIVE — every new consumer's sidecar JSON is consumable by `tools/cathedral_autopilot_autonomous_loop.py::rank_candidates`; SISTER 3 owns the actual import
5. **CONTINUAL-LEARNING POSTERIOR**: ACTIVE — sidecar JSON path is the canonical posterior surface per `consumer_output_path` (`.omx/state/master_gradient_consumers/`)
6. **PROBE-DISAMBIGUATOR**: ACTIVE — consumer 12 (KKT residual) IS the canonical disambiguator for "is the chosen λ_R achieving per-pair stationarity?"; consumer 14 (decoder pruning) IS the canonical disambiguator for "is this byte dead capacity vs hidden per-pair leverage?"

## Cross-cable implications + sister handoff

- **SISTER 3 (cathedral autopilot wire-ins)**: now has 8 new canonical consumer signals + 16 dataclasses + structured JSON output paths to consume. Recommended next: implement `tools/cathedral_autopilot_autonomous_loop.py::adjust_predicted_delta_for_per_pair_pareto_envelope` etc. via the v2 cascade pattern established by sister Catalog #319's `adjust_predicted_delta_for_venn_classification_v2`.
- **SISTER 4 (substrate symposium DRAFTs)**: can reference the new consumers in symposium memos for per-substrate Pareto / λ_R / KKT / Volterra analysis. The `per_pair_pareto_envelope` + `per_pair_volterra_cross_terms` outputs are especially useful for cross-substrate composition symposiums.
- **PARETO CONSUMER WIRE-IN (the 1 "pending" surface)**: `tac.optimization.pareto` needs a sister wire-in to consume consumers 7 + 8 + 12 + 13. This is the natural next subagent in the Cable D D5/D6 follow-on (lane already declared per battle plan).

## Cross-references

- Battle plan: `.omx/research/integrated_battle_plan_priority_queue_dag_cables_gates_20260519T052801Z.md` (Cable D rows D2/D3/D4)
- Prior master-gradient memos:
  - Master-gradient consumer integration design 2026-05-17 (sister of this landing's v1)
  - Master-gradient + xray follow-up research 2026-05-18 (task #874)
  - Per-X optimal codec planner + DuckDB unification 2026-05-18 (task #881)
  - Master-gradient consumers wire-in coverage audit (task #810; operator standing directive)
- Sister catalog gates:
  - Catalog #125 (subagent landing 6-hook wire-in)
  - Catalog #127 (per-call-site custody routing)
  - Catalog #229 (premise verification before edit)
  - Catalog #230 (sister-subagent ownership map)
  - Catalog #244 (canonical NVML env block — non-applicable for editor-only work)
  - Catalog #287 (phantom-API discipline — every cited `tac.X` was grep-verified)
  - Catalog #314 (sister subagent absorption pattern — disjoint scope confirmed at start; never absorbed)
  - Catalog #319/#322 (per-pair sensitivity-driven autopilot cascade pattern — sister)
  - Catalog #323 (canonical Provenance umbrella — sidecar JSON honors via canonical compliance tags)
- Sister forbidden patterns honored:
  - "Forbidden /tmp paths in any persisted artifact" — CLI guard in xray tool
  - "Forbidden empirical-claim-without-evidence-tag" — every sidecar JSON carries explicit `evidence_grade` + `score_claim=False`
  - "Forbidden phantom-score directory trap" — sidecar paths under canonical `.omx/state/master_gradient_consumers/`, never device-named

## Lane status

- Lane `lane_cable_d_master_gradient_extension_batch_20260519` registered at L0 via `tools/lane_maturity.py add-lane`
- Gates landed in this commit batch:
  - `impl_complete` ✓ (8 new consumers + xray tool + analytical surface manifest)
  - `memory_entry` ✓ (this memo)
- Gates pending sister-subagent or operator follow-on:
  - `real_archive_empirical` — sister 3 wire-in into cathedral autopilot ranker
  - `contest_cuda` — requires sister 3 wire-in producing measurable score impact
  - `strict_preflight` — no NEW strict preflight gate needed (consumers honor existing Catalog #220 / #287 / #323 / #125)
  - `three_clean_review` — adversarial review cycle
  - `deploy_runbook` — N/A for editor-only consumer/viz work

Expected lane level after this commit: **L1** (impl_complete + memory_entry).

— Cable D batch subagent 2026-05-19
