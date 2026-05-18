# Canonical helper pattern audit — master_gradient + 7 sister helpers

**Lane:** `lane_op_routable_5_quantizr_5_stage_staircase_canonical_20260517`
**Subagent:** op_routable_5_quantizr_staircase_20260517
**Date:** 2026-05-17
**Scope:** Op-routable #5 PART 2 per operator standing directive 2026-05-17. Read-only across existing modules; this audit identifies wire-in opportunities for the next-session priority queue.

## Mission alignment

Per CLAUDE.md "Mission alignment — non-negotiable": this audit is `apparatus_maintenance` (Phase-1 of an integration-coherence cycle). The downstream wire-ins identified here are `frontier_breaking` candidates ranked at the end of this memo. The audit itself is the cheap (~30 min reading, $0 GPU) prerequisite for the operator to pick which wire-ins to fund next.

## Methodology

For each canonical helper:

1. Read public surface (class/function signatures + schema_version + axis tag support)
2. Diff schema against `tac.master_gradient.MasterGradient` to find overlapping fields + gaps
3. Identify producer / consumer wire-in surfaces
4. Classify ALREADY-COHERENT / NEEDS-WIRE-IN / NEEDS-SCHEMA-EVOLUTION
5. Estimate wall-clock to land (engineering only; no GPU)

Premise verified pre-write per Catalog #229: read all 8 source files (and grep-counted schema fields) before drafting findings.

## 1. `tac.master_gradient` — the canonical 4-layer pattern (status quo)

**Status:** ALREADY-COHERENT for Layer 1 (in-memory + JSONL ledger) + Layer 4 (autopilot lens). Layers 2 + 3 declared TODO in the helper docstring.

**Schema:** `MasterGradient` dataclass carries (`archive_sha256`, `OperatingPoint(d_seg, d_pose, rate, score)`, `gradient_array_path` sidecar, `n_bytes`, `measurement_method`, `measurement_axis="[contest-CPU]"`/`"[contest-CUDA]"`, `measurement_hardware`, `measurement_call_id`, `measurement_utc`, `pareto_facets`, `rashomon_disagreement_score`). Ledger at `.omx/state/master_gradient_anchors.jsonl` (fcntl-locked, append-only, schema_version `master_gradient_anchor_v1`).

**Mirrors:** Catalog #245 (Modal call_id ledger) 4-layer exemplar: (1) canonical helper, (2) CLI extractor [TODO], (3) STRICT preflight gate refusing stale citations [TODO], (4) autopilot rerank wire-in [LANDED at `tools/cathedral_autopilot_autonomous_loop.py:3706 rerank_candidates_via_master_gradient`].

**Gaps identified:**

- **G1 (Layer 2 missing):** `tools/extract_master_gradient.py` CLI does NOT exist. The autopilot lens at line 3706 lazily imports `tac.master_gradient.latest_anchor_for_archive`, but if the ledger is empty (no anchors yet measured), every candidate falls through to `[predicted, no-master-gradient-anchor, ...]` tag. The Layer 2 CLI is what produces the FIRST anchor (autograd projection at ~12 min wall-clock per the symposium memo §3.6).

- **G2 (Layer 3 missing):** No STRICT preflight gate refusing stale gradient citations. Operating point is local; per the symposium §3.6 re-measurement cadence, an anchor older than ~7 days at a different operating point produces predictions that miss the cliff (especially as `d_pose → 0` where pose_marginal diverges). Without a staleness gate, the autopilot lens at line 3706 could silently consume an anchor from 30 days ago and report `[predicted, master-gradient-anchor-present, ...]` with high confidence even though the operating point has drifted.

- **G3 (autopilot lens consumes operating-point score, not per-byte projection):** Line 3786 — `op_score = float(anchor.get("operating_point", {}).get("score", 0.0))`. The lens reports the OPERATING-POINT score itself, not the per-byte ΔS projection. The actual `MasterGradient.predict_delta_s(byte_modifications)` method exists (line 165) but the autopilot ranker has no `byte_modifications` field on its `CandidateRow` to feed it. Per the symposium memo §3 use-case #8 ("magic codec per-stream selection"), this is the high-value wire-in: candidate produces per-stream byte modification spec, lens projects ΔS via gradient.

**Classification:** ALREADY-COHERENT (Layers 1+4); NEEDS-WIRE-IN (Layers 2+3 + per-byte projection at Layer 4).

**Wire-in estimate:**
- G1 (Layer 2 CLI extractor): ~6 hours engineering + $0.50-2.00 first Modal CPU run.
- G2 (Layer 3 staleness gate): ~3 hours engineering; STRICT-from-byte-one once a real anchor exists.
- G3 (per-byte projection): ~2 hours to extend `CandidateRow` with `byte_modifications` field + wire into lens; depends on per-candidate ranking surface having a byte-grain spec to begin with (currently only `predicted_score_delta` scalar).

## 2. `tac.sensitivity_map` (Catalog #232) — axis-level reweighting

**Status:** NEEDS-SCHEMA-EVOLUTION to consume MasterGradient as per-byte source.

**Current schema:** Operates at PER-LAYER granularity (`{"<module>.weight" -> Tensor[O]}` where O is Conv2d output-channel count). The axis-weights sister surface (`tac.sensitivity_map.axis_weights`) computes the closed-form `pose_marginal / seg_marginal` ratio at a NAMED operating point (via `axis_weights_for_named_operating_point`). Carries `source_archive_sha256` + `baseline_archive_sha256` fields for cite-chain. NO measurement_axis tag (the sensitivity-map contract requires CUDA per the docstring "Authoritative maps must be produced on CUDA").

**Data-flow gap:** sensitivity_map currently consumes:
- per-layer scorer-gradient projection (compress-time CUDA forward+backward through SegNet/PoseNet)
- operating-point axis-weights (closed-form `pose / seg` ratio)

It does NOT yet consume:
- per-byte gradient (MasterGradient's strict superset granularity)
- archive-byte-level sensitivity (which bytes in the inflated archive contribute most to ΔS)

**Per-byte is a strict superset of per-layer:** a per-byte gradient projected onto the per-layer aggregation = `sum(|G[byte_i, :]| for byte_i in layer_X_bytes)`. The sensitivity-map could consume MasterGradient as a more granular source, aggregating to its per-layer contract.

**Wire-in surface:** New function `tac.sensitivity_map.from_master_gradient(gradient: MasterGradient, model: nn.Module) -> dict[str, Tensor]` that walks the model's conv layers, locates the byte ranges in the gradient array corresponding to each Conv2d weight tensor, and reduces along the byte axis to produce the per-layer sensitivity vector. The byte-range mapping requires a `tac.packet_compiler.parser_section_manifest` (which exists per Catalog #101) to translate archive bytes → parameter tensors.

**Classification:** NEEDS-SCHEMA-EVOLUTION (sensitivity_map's current contract is per-layer; consuming per-byte requires either an aggregation helper OR a parallel `tac.sensitivity_map.per_byte` namespace).

**Wire-in estimate:** ~4 hours engineering for the from_master_gradient bridge + tests; presumes Catalog #101 parser_section_manifest is available for the target substrate.

## 3. `tac.cost_band_calibration` (Catalog #175/#177) — dispatch cost posterior

**Status:** ALREADY-COHERENT pattern-wise; NEEDS-SCHEMA-EVOLUTION for cite-chain.

**Pattern coherence:** Both `master_gradient_anchors.jsonl` and `cost_band_posterior.jsonl` are fcntl-locked JSONL append-only stores with per-row `schema_version` (`master_gradient_anchor_v1` vs `1`/`2` integer for cost-band's `outcome` migration per R4 finding Z-4.1 2026-05-13). Both carry an `outcome`-like terminal field (master_gradient implicitly always-successful since it only writes after a measurement completes; cost-band has explicit `VALID_OUTCOMES = {successful_dispatch, failed_dispatch, timed_out, harvested_partial}`). Both load via lenient + strict paths per Catalog #138.

**Schema divergence:** cost-band keys on `(platform, gpu)` tuples (`PLATFORM_RATES_USD_PER_HOUR` registry); master_gradient keys on `archive_sha256`. The two share no natural join key. The closest overlap is `measurement_call_id` (master_gradient) ↔ `call_id` in cost-band's outcome anchors — but cost-band does NOT carry `archive_sha256` so a master_gradient measurement can't be cost-attributed back to a single dispatch by SHA alone.

**Wire-in surface:** When the Layer 2 CLI extractor (G1 above) fires a Modal CPU run to compute a master-gradient anchor, the dispatch SHOULD register itself via `tac.deploy.modal.call_id_ledger.register_dispatched_call_id` per Catalog #245 AND the resulting `MasterGradient.measurement_call_id` field SHOULD reference that call_id. Then `cost_band_calibration` can be cross-joined via call_id to attribute the wall-clock + cost of master-gradient measurement to its own posterior. This requires no schema evolution on either side, only correct producer-side wiring in the CLI extractor.

**Classification:** ALREADY-COHERENT (pattern); NEEDS-WIRE-IN (cite-chain producer-side discipline at G1 CLI extractor).

**Wire-in estimate:** ~1 hour as part of G1 CLI extractor work (no incremental cost; just thread the call_id through).

## 4. `tac.autopilot_rudin_daubechies` Rashomon ensemble (Catalog #252) — K=8 SLIM scorers

**Status:** NEEDS-WIRE-IN to ingest MasterGradient anchors.

**Current schema:** `RashomonEnsembleRanker(K=8)` constructs K=8 bootstrap-diverse SLIM rankers. `update_all(panel, store_path=...)` consumes a `ProxyPanel` (operator-readable gate verdict panel) and trains each scorer on a different bootstrap sample. Disagreement queue is the `K=8` scorers' max-pairwise rank distance per candidate.

**Wire-in path:** A MasterGradient anchor is a NEW row that the Rashomon ensemble could consume as a bootstrap-diverse training row, IF the ensemble were extended to consume gradient anchors as a feature class (in addition to gate verdicts). Specifically:

- Each MasterGradient anchor carries `pareto_facets` (sequence of `(i, j)` byte-pair tuples on the Pareto frontier) and `rashomon_disagreement_score` (already named to indicate the intended consumer).
- `rashomon_disagreement_score: float | None` field on MasterGradient is currently UNPOPULATED at write time (no producer sets it). The Layer 2 CLI extractor SHOULD compute it as `max_pairwise_rank_distance_of_byte_modifications(gradient, K=8_bootstrap_samples)` and persist alongside.
- Once populated, `RashomonEnsembleRanker.update_from_master_gradient_anchor(anchor)` ingests the per-byte gradient projection as 8 distinct bootstrap rows (one per K).

**Classification:** NEEDS-WIRE-IN (both producer-side: G1 CLI extractor must compute `rashomon_disagreement_score`; and consumer-side: ensemble needs `update_from_master_gradient_anchor` method).

**Wire-in estimate:** ~5 hours engineering total (~2 hours producer + ~3 hours consumer including K=8 stress test).

## 5. `tac.frontier_scan` (Catalog #316) — anchor inventory

**Status:** ALREADY-COHERENT; NO contention.

**Pattern coherence:** Reads from `continual_learning_posterior.json` + `active_lane_dispatch_claims.md` + `modal_call_id_ledger.jsonl` + `reports/latest.md`. Filters to `QUALIFYING_HARDWARE` (1:1 contest-compliant). Builds per-axis best-anchor lookup.

**Contention check with `master_gradient_anchors.jsonl`:** None. `frontier_scan.collect_all_anchors` only reads SCORE anchors (axis-tagged numerical scores on real archives). master_gradient anchors are GRADIENT PROJECTIONS — they carry an `operating_point.score` field but that score is METADATA about WHERE the gradient was measured, not an independent score claim. The gradient anchor's score field should NOT be promoted to frontier candidates.

**Recommendation:** If a future revision of frontier_scan extends its source list, it MUST explicitly exclude `master_gradient_anchors.jsonl` (or any ledger whose score is operating-point metadata, not a score claim). This is canonical to CLAUDE.md "Forbidden score claims" — a gradient is a derivative, not a measurement of S itself.

**Classification:** ALREADY-COHERENT.

**Wire-in estimate:** 0 hours (verified clean today; documentation note in future revisions).

## 6. `tac.deploy.modal.call_id_ledger` (Catalog #245) — Modal call_id audit trail

**Status:** ALREADY-COHERENT; producer-side discipline pending at G1 CLI extractor.

**Pattern coherence:** Both ledgers are the canonical 4-layer pattern (canonical fcntl-locked JSONL + register/update helpers + strict-load discipline per Catalog #138 + autopilot ranker wire-in). master_gradient's Layer 1 explicitly mirrors call_id_ledger's Layer 1.

**Schema overlap:** `MasterGradient.measurement_call_id: str | None` is the FK to `MODAL_CALL_ID_LEDGER_PATH` rows. Currently always-None because no producer (G1 CLI extractor) writes it. The wire-in at G1 (above) makes this FK live.

**Classification:** ALREADY-COHERENT (schema); NEEDS-WIRE-IN (producer discipline at G1).

**Wire-in estimate:** 0 incremental (covered by G1).

## 7. `tac.continual_learning` (Catalog #128) — auth-eval posterior

**Status:** ALREADY-COHERENT for archive_sha256 join; NEEDS-WIRE-IN for cite-chain queries.

**Schema overlap:** Both keyed on `archive_sha256`. `continual_learning.ContestResult.archive_sha256` matches `MasterGradient.archive_sha256` exactly. A future helper `tac.master_gradient.cross_join_with_auth_eval(sha)` could return `(MasterGradient, ContestResult)` tuples — useful for "what's the empirical score AND the predicted-ΔS-per-byte at the same archive?".

**Wire-in surface:** Two query helpers:
- `tac.master_gradient.latest_anchor_with_auth_eval(sha)` returns (gradient, auth_eval) joined on sha.
- `tac.continual_learning.contest_results_with_master_gradient_present(min_age_days=N)` returns auth-eval rows for archives that ALSO have a master-gradient anchor (the candidates eligible for per-byte ΔS prediction).

**Classification:** ALREADY-COHERENT (schema overlap); NEEDS-WIRE-IN (query helpers).

**Wire-in estimate:** ~2 hours engineering + tests.

## 8. `tac.council_continual_learning` (Catalog #300) — council deliberation posterior

**Status:** ALREADY-COHERENT pattern-wise; OPTIONAL wire-in for cite-chain.

**Schema overlap:** `CouncilDeliberationRecord.related_deliberation_ids: tuple[str, ...]` is the cite-chain field. The T4 symposium memo `grand_reunion_t4_symposium_orthogonal_optimization_master_gradient_20260517.md` is the canonical authority for the master-gradient nickname + 8 in-training uses. A future council deliberation that consumes master-gradient anchors SHOULD cite the T4 symposium memo's deliberation_id in `related_deliberation_ids`.

**Wire-in surface:** The next council deliberation that references the master_gradient helper MUST include the T4 symposium memo's deliberation_id. This is a documentation convention, not a structural change.

**Classification:** ALREADY-COHERENT; OPTIONAL documentation convention.

**Wire-in estimate:** 0 incremental (covered by next council deliberation memo author).

## Ranked op-routable list (for operator next-session selection)

Ranked by `|predicted ΔS impact / cost|`. Cost in {engineering hours, $GPU}.

| Rank | Op-routable | ΔS impact (qualitative) | LOC | Wall-clock | $GPU | Notes |
|------|-------------|-------------------------|-----|------------|------|-------|
| **1** | G1: `tools/extract_master_gradient.py` Layer 2 CLI | HIGH — produces the FIRST anchor; without it, the autopilot lens line 3706 always falls through to no-anchor predictions | ~200 | 6 h | $0.50-2.00 first run | Unblocks Rashomon ingestion (#4) + cost-band cite-chain (#3) + sensitivity_map bridge (#2) by producing the first anchor row |
| **2** | G3: per-byte projection wire-in on autopilot ranker | HIGH — converts lens from "anchor present?" to "predicted ΔS per candidate byte_modifications spec" | ~150 | 2 h | $0 | Requires `CandidateRow.byte_modifications` field on autopilot; depends on candidates having a per-stream byte spec (currently only have scalar `predicted_score_delta`) |
| **3** | G2: Layer 3 STRICT preflight gate for stale anchors | MEDIUM — prevents silent consumption of stale gradients across operating-point drift; cliff risk as `d_pose → 0` | ~120 | 3 h | $0 | STRICT-from-byte-one once a real anchor exists; cite Symposium §3.6 cadence (7-day staleness) |
| **4** | #2: `tac.sensitivity_map.from_master_gradient(gradient, model)` bridge | MEDIUM — promotes per-byte to per-layer; consumed by composition_ranking_json autopilot bridge | ~180 | 4 h | $0 | Depends on Catalog #101 parser_section_manifest being available for the target substrate |
| **5** | #4: Rashomon ensemble + producer `rashomon_disagreement_score` | MEDIUM — opens K=8 ensemble's consumption of gradient anchors as bootstrap-diverse training rows | ~150 producer + ~200 consumer | 5 h | $0 | Producer-side at G1 CLI extractor (~2h); consumer-side at `tac.autopilot_rudin_daubechies.rashomon_ensemble.update_from_master_gradient_anchor` (~3h) |
| **6** | #7: cite-chain query helpers `latest_anchor_with_auth_eval` + `contest_results_with_master_gradient_present` | LOW — operator UX improvement for cross-querying gradient + auth-eval | ~80 | 2 h | $0 | Sister of `tac.frontier_scan.collect_all_anchors`; pure SHA-join |
| **7** | PR-101 + PR-106 + PR-107 master-gradient measurement campaign | HIGH — applies G1 to the 3 canonical anchors to seed the ledger with 3 known-good entries | 0 (just 3 invocations of G1) | 0.5 h + 3× wait | $1.50-6.00 | Depends on G1 landing first |

## Cross-validation against existing CLAUDE.md non-negotiables

- **"Apples-to-apples evidence discipline":** every wire-in respects axis tag (`measurement_axis` propagated end-to-end).
- **"Forbidden score claims":** the autopilot lens at line 3706 already tags `[predicted, master-gradient-projection]`; gates G2 + G3 preserve this; gradient is not promoted to a score itself.
- **"Bugs must be permanently fixed AND self-protected against":** G2 is the STRICT preflight self-protection for the bug class "stale gradient cited as fresh".
- **"Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY":** the master_gradient module itself is OPERATIONAL (the autopilot lens consumes it at runtime; the ledger persists across sessions). The 8 in-design uses per symposium §3.6 are all wire-in TODOs; each is a distinct lane.
- **"Mission alignment":** every wire-in is `frontier_breaking` or `frontier_protecting` (G2 protects the frontier from stale-gradient regression).

## What this audit deliberately does NOT recommend

Per Hotz's revision and the operator's standing directive against premature generalization:

- **No substrate-trainer adoption of `QuantizrFiveStageStaircase`.** Wait until a SECOND substrate demonstrably needs a different schedule. Today's landing is Quantizr-specific.
- **No new helper modules beyond the 7 wire-ins above.** Each wire-in extends an EXISTING canonical helper; no new packages.
- **No master-gradient + cost-band cross-join helper.** The two ledgers are already cross-joinable via call_id; a dedicated helper would be premature aggregation.

## Cross-references

- `tac.master_gradient` (the helper this audit is about)
- `.omx/research/grand_reunion_t4_symposium_orthogonal_optimization_master_gradient_20260517.md` (the canonical authority for the 8 in-design uses)
- `.omx/research/cpu_frontier_master_gradient_campaign_plan_20260517.md` §1.3 (op-routable #5 source)
- CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L7 (bolt-on vs substrate-engineering split that informs Hotz's revision)
- CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" (the canonical-vs-unique decision framework this audit consumes)
- `src/tac/training_curriculum/quantizr_5_stage_staircase.py` (PART 1 of this op-routable; today's landing)
- `src/tac/tests/test_quantizr_5_stage_staircase.py` (42/42 tests pass at landing)
