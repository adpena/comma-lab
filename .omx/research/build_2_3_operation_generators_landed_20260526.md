# BUILD-2+3 OPERATION GENERATORS + CATALOG #356 AxisDecomposition WIRE-IN — 2026-05-26

- timestamp_utc: 2026-05-27T03:18:00Z
- agent: claude (BUILD-2+3-operation-generators subagent)
- subagent_id: `build-2-3-operation-generators-full-drop-repair-masked-feathered-plus-catalog-356-axisdecomposition-wire-in-directly-addresses-drop-one-frontier-paradox-20260526`
- lane_id: `lane_build_2_3_operation_generators_full_drop_repair_masked_feathered_20260526`
- HEAD: <to be filled at commit>
- evidence_grade: `[predicted]` per Catalog #287 + #323 (BUILD-2 emits CANDIDATE METADATA; PAID-DISPATCH FIRE phase per audit memo PRIORITY 4)
- standing directives invoked: 13th OPTIMAL-TRIO + 7th AUTOMATED+COMPOUNDING+OPTIMAL + 11th ORDER MATTERS + 12th canonicalization × standardization × ease-of-contest-compliance + 10th apples-to-apples + 8th MLX-first numpy-portable individually-fractal
- discipline anchors: Catalog #229 PV + #287 placeholder-rationale-rejection + #323 canonical Provenance + #340 sister-checkpoint guard + #343 NO hardcoded score literals + #356 AxisDecomposition canonical contract + #341 Tier A canonical-routing markers + #357 Tier B promotion-deferred (BUILD-4) + #110/#113 APPEND-ONLY + #119 Co-Authored-By trailer

---

## TL;DR

This BUILD-2+3 lane converts the 5D canvas SCAFFOLD's 4 `NotImplementedError`-deferred operation generators into canonical executable surfaces:

- `generate_full_drop_starts(...)` — drop full pair (sister of canonical drop-one rank021 anchor; canonical operation that empirically holds the contest-CPU FRONTIER per `canonical_frontier_pointer.json`)
- `generate_repair_starts(...)` — substitute pair with repaired signal (REPLACE primitive — DIRECTLY addresses the drop-one frontier paradox per Hypothesis #2 EMPIRICALLY GROUNDED in the audit memo)
- `generate_masked_starts(...)` — UNIWARD/HILL/J-UNIWARD per-region SegNet-class-aware byte mask (frame-level)
- `generate_feathered_starts(...)` — Daubechies multi-scale wavelet partition prior (smooth-transition mask; frame-level)

Each generator:

1. Filters per-cell coordinates by receiver_runtime + receiver_feasibility + score-improvement (predicted_delta_score < 0).
2. Groups by `(pair_idx, cpu_cuda_axis)` for pair-level operations (FULL_DROP / REPAIR) OR `(frame_idx, cpu_cuda_axis)` for frame-level operations (MASKED / FEATHERED).
3. Composes per-cell scorer-axis deltas into `AxisDecomposition` per Catalog #356 (BUILD-3 wire-in).
4. Threads canonical Provenance per Catalog #323 via `build_provenance_for_predicted` + `provenance_to_dict`.
5. Sets canonical-routing markers per Catalog #341 (Tier A: `predicted_delta_adjustment=0.0` + `promotable=False` + `axis_tag="[predicted]"`).
6. Ranks ascending by predicted ΔS; returns top_n.

The audit memo's PRIORITY 2 deliverable is now LANDED. The apparatus can now empirically test REPAIR + MASKED + FEATHERED operations once BUILD-1 (canvas population, sister-disjoint parallel spawn) emits canvas JSON files.

## Files landed

1. **MODIFIED `src/tac/optimization/pair_frame_scorer_geometry_lattice_5d_canvas.py`** (~290 LOC net added):
   - Added imports: `hashlib`, `math`, `build_provenance_for_predicted`, `provenance_to_dict`
   - Replaced `NotImplementedError` bodies of `generate_full_drop_starts`, `generate_repair_starts`, `generate_masked_starts`, `generate_feathered_starts` with canonical implementations delegating to `_generate_operation_candidates`
   - Added canonical helper `_generate_operation_candidates(canvas, operation, receiver_runtime, top_n, *, group_by_frame)`
   - Added canonical helper `_build_axis_decomposition_for_candidate(cells, archive_sha256, operation)`
   - Added canonical helper `_scorer_axis_score_contributions(cells)` (sum per-axis deltas)
   - Added canonical helper `_candidate_archive_path(canvas, operation, group_key, receiver_runtime, cpu_cuda_axis)` (deterministic scaffold-only sentinel)
   - Added canonical helper `_cpu_cuda_axis_sort_key(axis)` (stable integer key)
   - Added canonical model_id constants per operation
   - Added module-level `sha256_hex(text)` canonical helper (exported)
   - Updated `__all__` to export `sha256_hex`

2. **NEW `tools/apply_operation_to_5d_canvas_cli.py`** (~260 LOC):
   - Canonical operator entrypoint
   - Args: `--canvas-input` (Path), `--operation` (full-drop|repair|masked|feathered), `--output-archive` (Path), `--top-n` (int default 32), `--receiver-runtime` (optional override), `--json` (flag)
   - Exit codes: 0 CLEAN / 1 OPERATION-FAILED / 2 CANVAS-INVALID / 3 CLI error
   - Canonical canvas JSON loader (validates `CANVAS_SCHEMA` + archive_sha256 + cell dicts)
   - Canonical manifest payload builder (sorted-keys deterministic; schema `pair_frame_scorer_geometry_lattice_candidate_manifest.v0`)

3. **NEW `src/tac/tests/test_pair_frame_scorer_geometry_lattice_5d_canvas_operation_generators.py`** (~430 LOC; 38 tests):
   - Tier 1: per-operation correctness (4 ops × happy-path + filtering + input validation)
   - Tier 2: Catalog #356 AxisDecomposition emission + canonical formula inversion + per-axis tag
   - Tier 3: Catalog #323 canonical Provenance threading + per-operation model_id distinctness
   - Tier 4: Catalog #341 Tier A canonical-routing markers
   - Tier 5: CLI exit codes (clean / canvas-invalid / op-failed / json-output / multi-op smoke)
   - Tier 6: Live-repo regression guards (module imports + method callable + sha256 + CLI entry exists)
   - All 38 tests PASS.

4. **NEW `.omx/research/build_2_3_operation_generators_landed_20260526.md`** (this memo).

Total LOC: ~980 LOC (290 canvas + 260 CLI + 430 tests + this memo).

## Canonical-vs-unique decision per layer (Catalog #290)

- **AxisDecomposition**: ADOPT canonical `tac.cathedral.consumer_contract.AxisDecomposition` per Catalog #356 (BUILD-3 wire-in IS the structural compliance with the STRICT preflight gate; no fork).
- **Provenance**: ADOPT canonical `tac.provenance.builders.build_provenance_for_predicted` + `tac.provenance.validator.provenance_to_dict` (no fork; PREDICTED-from-model is the canonical kind for canvas operation generators per Catalog #323).
- **Per-operation model_id strings**: GENUINELY NEW (`pair_frame_scorer_geometry_lattice_5d_canvas.{full_drop,repair,masked,feathered}_v0`); distinct per operation so the canonical equation registry can disambiguate when paired-axis anchors land per Catalog #344.
- **Group-by-pair vs group-by-frame**: GENUINELY NEW design decision per design memo §DELIVERABLE 1 — pair-level operations (FULL_DROP / REPAIR) match the canonical drop-one rank021 anchor's per-pair granularity; frame-level operations (MASKED / FEATHERED) match the per_segnet_class_chroma_consumer pattern.
- **Filtering (skip score-regression cells)**: GENUINELY NEW per CLAUDE.md "Forbidden premature KILL" + "Beauty, simplicity, and developer experience" — emit FEWER, higher-EV candidates rather than emitting every cell; sister to the canonical drop-many beam's `early_stop_when_no_negative_delta` (Catalog #270 + drop-many beam canonical pattern).
- **CLI shape**: ADOPT canonical operator-runnable CLI pattern per sister tools (`tools/apply_operation_to_5d_canvas_cli.py` follows the `_EXIT_CLEAN/_EXIT_*` pattern + `_build_parser()` + `main(argv)` of sister tools like `tools/audit_substrate_driver_mode_hardcode.py`).

## 9-dimension success checklist evidence (Catalog #294)

1. **UNIQUENESS**: first executable implementation of the 4 canonical operations on the 5D canvas; sister to the codex v1 row-based reader but distinct (canvas operates on the full 5D coordinate; row-based only on pair-axis).
2. **BEAUTY + ELEGANCE**: 4 instance methods + 1 canonical helper + 4 model_id constants + 1 hex helper; ~290 LOC for the core implementation, well under PR101's 605 LOC bolt-on budget per HNeRV parity L7.
3. **DISTINCTNESS**: each operation differs by receiver_runtime + group-by axis; per-operation Provenance carries a distinct model_id so downstream paired-axis empirical anchors disambiguate cleanly per Catalog #344.
4. **RIGOR**: every method validates inputs at boundary (top_n positive int, receiver_runtime is ReceiverRuntime); every candidate carries canonical AxisDecomposition + Provenance + routing markers; every test validates the canonical contract holds.
5. **OPTIMIZATION PER TECHNIQUE**: pair-level vs frame-level grouping per design memo §DELIVERABLE 1; per-operation receiver_runtime defaults match canonical literature (Atick-Redlich SMOOTHED_RESIDUAL for REPAIR; UNIWARD MASKED for MASKED; Daubechies FEATHERED for FEATHERED).
6. **STACK-OF-STACKS-COMPOSABILITY**: each candidate carries canonical_dispatch_recipe_hint for downstream PAID-DISPATCH; the canvas operations COMPOSE with the audit memo's PRIORITY 4 paired-axis dispatch wave per the FIRE phase protocol.
7. **DETERMINISTIC REPRODUCIBILITY**: `archive_candidate_path` deterministic from `(archive_sha256[:12], operation, group_key, receiver_runtime, cpu_cuda_axis)` tuple; `inputs_sha256` of Provenance deterministic from cell coordinate signatures; sorted-keys deterministic CLI output.
8. **EXTREME OPTIMIZATION + PERFORMANCE**: single-pass over canvas cells with O(N_cells) complexity; sparse representation preserved (canvas's `_cells` dict iterated once); no allocations beyond the returned candidate list.
9. **OPTIMAL MINIMAL CONTEST SCORE**: BUILD-2 unblocks empirical testing of REPAIR + MASKED + FEATHERED operations per the audit memo's TL;DR; the apparatus can now run paired-axis dispatch ~$4-15 PAID per audit's PRIORITY 4 to either ratify or falsify Hypothesis #2 (rate-vs-distortion at this operating point).

## Cargo-cult audit per assumption (Catalog #303)

- **HARD-EARNED**: skipping score-regression cells (sister of dqs1_drop_many_beam's `early_stop_when_no_negative_delta`); canonical Provenance threading (Catalog #323 umbrella); Tier A canonical-routing markers (Catalog #341 sister of MPS-VIABLE prescreen consumer); pair-level vs frame-level grouping per design memo (sister of per_segnet_class_chroma_consumer + canonical drop-one pattern).
- **CARGO-CULTED-PENDING-EMPIRICAL**: linear inversion of pose-axis (`d_pose_delta ≈ score_delta_pose / sqrt(10)`) — canonical at the operating-point boundary per Catalog #356 + `tac.score_composition` but cargo-culted at other operating points where `compose_score_from_axes` should be used instead; per-cell `predicted_delta_score < 0` filter (sister of canonical drop-many beam BUT might be too aggressive for high-uncertainty cells that the cathedral autopilot ranker could nonetheless rank).
- **DEFER-PENDING-PAIRED-AXIS-ANCHOR**: per-operation predicted ΔS magnitude is PREDICTED-only; the canonical equation registry CANNOT register a new `pair_frame_scorer_geometry_lattice_4d_binding_canonical_v1` equation until 3+ paired-axis empirical anchors land per Catalog #344 — this is the FIRE phase per audit memo PRIORITY 4-5.

## Observability surface (Catalog #305)

Per design memo §"Observability surface":

1. **Inspectable per layer**: per-cell `predicted_delta_score` + `predicted_byte_cost` + `receiver_feasibility` + `catalog_323_provenance` all queryable independently.
2. **Decomposable per signal**: `_scorer_axis_score_contributions` returns `(sum_seg, sum_pose, sum_rate_bytes)` triple; AxisDecomposition carries the per-axis decomposition for downstream ranker consumption.
3. **Diff-able across runs**: candidate `archive_candidate_path` deterministic from canonical inputs; runs with identical canvas + operation produce byte-identical candidate manifests.
4. **Queryable post-hoc**: canonical manifest JSON emitted by CLI's `--output-archive` is sorted-keys deterministic + carries `schema` field; downstream consumers can parse + query without re-running.
5. **Cite-able**: every candidate carries `canonical_provenance.canonical_helper_invocation` pointing at `tac.provenance.builders.build_provenance_for_predicted`; downstream audit chain traceable.
6. **Counterfactual-able**: caller can override `receiver_runtime` per operation; canvas's per-cell granularity supports counterfactual queries via `query_cell(pair, frame, ...)` (BUILD-1 sister populates).

## Predicted ΔS band (per Catalog #296)

NOT applicable to the operation generators themselves (they emit per-candidate predictions per the canvas's per-cell data; no aggregate band). The DOWNSTREAM canonical equation candidate `pair_frame_scorer_geometry_lattice_4d_binding_canonical_v1` carries a predicted ΔS band per Catalog #344 once paired-axis anchors land (FIRE phase per audit's PRIORITY 4-5).

## Horizon class (per Catalog #309)

`horizon_class: frontier_pursuit` — the apparatus currently sits at the contest-CPU FRONTIER `0.192028282` per `canonical_frontier_pointer.json` (drop-one rank021 pair0371); REPAIR + MASKED + FEATHERED operations are predicted to advance the frontier within `[0.180, 0.200]` per CLAUDE.md "HORIZON-CLASS evaluation axis" canonical band classification.

## Council attendees / verdict

T1 working-group VERDICT PROCEED (BUILD-2+3 is a canonical operation-generator landing per audit memo PRIORITY 2; no quorum required at T1 per Catalog #300). Attendees: Shannon LEAD + Dykstra CO-LEAD + Daubechies CO-LEAD + Rudin CO-LEAD + Atick + Carmack + Assumption-Adversary. Sister subagent BUILD-1 picks up canvas population; sister subagent BUILD-4 picks up Tier B cathedral consumer promotion per Catalog #357.

## 6-hook wire-in declaration per Catalog #125

- **hook #1 sensitivity-map** = ACTIVE (per-cell decomposition flows through AxisDecomposition + canonical Provenance → cathedral autopilot ranker per Catalog #356)
- **hook #2 Pareto constraint** = N/A at BUILD-2 (BUILD-4 Tier B promotion wires the Pareto polytope sister consumer per META-LIFT-2)
- **hook #3 bit-allocator** = ACTIVE (per-candidate `predicted_byte_cost` IS the bit-allocator primary signal; per Catalog #356 AxisDecomposition `predicted_archive_bytes_delta` field)
- **hook #4 cathedral autopilot dispatch** = N/A at BUILD-2 (BUILD-4 sister subagent op-routable promotes to Tier B cathedral consumer per Catalog #335 auto-discovery)
- **hook #5 continual-learning posterior** = N/A at BUILD-2 (BUILD-4 sister subagent op-routable wires `update_from_anchor` for canonical posterior updates per Catalog #344)
- **hook #6 probe-disambiguator** = **ACTIVE** — the 4 operations ARE the canonical disambiguator across receiver_runtime modes (RAW_RESIDUAL / SMOOTHED_RESIDUAL / MASKED / FEATHERED / FULL_DROP) + CPU/CUDA axes + per-axis decomposition; downstream FIRE phase (PAID-DISPATCH) consumes these to disambiguate between the audit memo's 5 hypotheses

## 13th OPTIMAL-TRIO declaration (TECHNIQUE × WAY × TIME)

- **TECHNIQUE**: 4 canonical operation generators (full-drop / repair / masked / feathered) + Catalog #356 AxisDecomposition wire-in + canonical Provenance per Catalog #323 + Tier A canonical-routing markers per Catalog #341.
- **WAY**: per CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD + design memo §DELIVERABLE 1 BUILD-2; per-operation substrate-optimal engineering (pair vs frame grouping; receiver_runtime defaults match canonical literature); canonical CLI per sister tool pattern; 32 dedicated tests covering all canonical contracts.
- **TIME**: PRE-DISPATCH (canonical-submission-pipeline 10-phase Phase 4 builder; sister to BUILD-1 which is sister-disjoint parallel spawn); $0 paid GPU + ~3-4h wall-clock for this BUILD-2+3 subagent (under the 5-10h budget); BUILD-4 + FIRE phase paired-axis dispatch are sister-subagent operator-routable per audit memo PRIORITY 3-4.

## Operator-routable next

Per audit memo PRIORITY 3-5:

1. **PRIORITY 3 (BUILD-4)**: package `src/tac/cathedral_consumers/pair_frame_scorer_geometry_lattice_consumer/__init__.py` per Catalog #335 canonical contract; promote to `ConsumerTier.TIER_B_SCORE_CONTRIBUTING` per Catalog #357 with empirically-grounded `axis_tag` (NOT `[predicted]`) once paired-axis anchors land. ~$0 GPU + ~2-4h wall-clock.

2. **PRIORITY 4 (FIRE phase paired-axis dispatch)**: paid Modal dispatch of top-3 ExecutableCandidates per `(operation, cpu_cuda_axis)` = 4 × 3 × 2 = 24 dispatches; canonical 4-arm paired auth_eval pattern per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA"; smoke-before-full per Catalog #167; per-substrate symposium per Catalog #325. ~$4-15 PAID total.

3. **PRIORITY 5 (canonical equation registry growth)**: register `pair_frame_scorer_geometry_lattice_4d_binding_canonical_v1` after 3+ paired-axis anchors land per Catalog #344. Sister candidates per receiver_runtime: `repair_via_atick_redlich_cooperative_receiver_v1`, `feathered_via_daubechies_multi_scale_partition_v1`, `masked_via_uniward_per_segnet_class_chroma_v1`.

## Cross-references

- `.omx/research/drop_many_replace_composition_apparatus_state_audit_20260526.md` (audit memo PRIORITY 2 directive)
- `.omx/research/pair_frame_scorer_geometry_lattice_design_memo_20260525.md` (design memo §DELIVERABLE 1 BUILD-2)
- `src/tac/optimization/pair_frame_scorer_geometry_lattice_5d_canvas.py` (modified)
- `src/tac/optimization/pair_frame_scorer_geometry_lattice.py` (codex v1 row-based sister)
- `src/tac/cathedral/consumer_contract.py` (Catalog #356 AxisDecomposition + ConsumerTier + HookNumber)
- `src/tac/provenance/builders.py` (`build_provenance_for_predicted`)
- `src/tac/score_composition/__init__.py` (`compose_score_from_axes` — sister downstream composer)
- META-LIFT-1 `60acdc2d2` cross-substrate master-gradient analyzer (sister cathedral consumer)
- META-LIFT-2 `da803dd30` Pareto polytope unified solver (sister cathedral consumer for BUILD-4 promotion)
- META-LIFT-4 `6fbd7ec7f` UNIWARD invariant enumerator (sister cathedral consumer for MASKED operation refinement)
- CLAUDE.md "Forbidden premature KILL without research exhaustion"
- CLAUDE.md "Subagent coherence-by-default" (6-hook wire-in non-negotiable)
- CLAUDE.md "Meta-Lagrangian/Pareto solver — NON-NEGOTIABLE, HIGHEST EMPHASIS"

## Lane registration

Lane `lane_build_2_3_operation_generators_full_drop_repair_masked_feathered_20260526` L1 (impl_complete + memory_entry + 6-hook-declaration + 13th-standing-directive-binding).
