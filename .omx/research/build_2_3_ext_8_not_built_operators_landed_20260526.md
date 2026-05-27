# BUILD-2+3-EXT 8 NOT-BUILT operators (replace × merge × reorder × frame × motion × temporal) — LANDED 2026-05-26

- timestamp_utc: 2026-05-27T03:31:00Z
- agent: claude (BUILD-2+3-EXT 8 NOT-BUILT operators subagent)
- subagent_id: `build-2-3-ext-8-not-built-operators-replace-merge-reorder-frame-level-motion-conditional-temporal-coherence-per-operator-insight-merge-and-combinations-beat-v14-v2-20260526`
- lane_id: `lane_build_2_3_ext_8_not_built_operators_replace_merge_reorder_frame_level_motion_conditional_temporal_coherence_20260526`
- scope: build canonical 8-operator module + CLI + comprehensive tests + landing memo per audit memo §"Phase 1 catalog enumeration" 8 NOT-BUILT operators
- authority: design + observability ONLY; no score/promotion/rank/dispatch authority per Catalog #341 + #357 Tier A scaffold
- evidence_grade: `[predicted]` per Catalog #287 + #323 (operator vocabulary primitives; not paired-axis score claims)
- standing directives invoked: 7th AUTOMATED+COMPOUNDING+OPTIMAL + 8th INDIVIDUALLY-FRACTAL MLX-first numpy-portable + 10th apples-to-apples + 11th ORDER MATTERS + 12th canonicalization × standardization × ease-of-contest-compliance + 13th OPTIMAL-TRIO (TECHNIQUE × WAY × TIME)
- discipline anchors: Catalog #229 PV + #287 evidence-tag + #323 canonical Provenance + #335 canonical contract + #341 Tier A markers + #344 NO promotion without empirical anchor + #356 per-axis decomposition + #357 Tier A scaffold + #110/#113 APPEND-ONLY + #340 sister-checkpoint guard PROCEED

## Codex continuation addendum 2026-05-27T03:40Z

Codex took custody after the BUILD-2+3-EXT subagent process was stale and preserved the subagent's module, CLI, tests, and memo. The continuation added the missing integration layer:

- ruff-cleaned the extended operator module and comprehensive test harness.
- Added `pair_frame_5d_extended_operator_consumer` under `src/tac/cathedral_consumers/` so the 8-operator family is auto-discovered by cathedral autopilot as Tier-A false-authority planning signal.
- Added `comma_lab.scheduler.pair_frame_5d_extended_operator_queue` plus `tools/build_5d_extended_operator_queue.py`, producing an 8-experiment local queue that fires all extended operators against a populated 5D canvas.
- Added queue tests; focused verification is now 60 tests, not the original 54-test subagent-only count.
- Built and validated a live queue for archive `6bae0201...`, then executed all 8 local queue steps successfully with `failure_count=0`. All live candidate manifests emitted `candidates_emitted=0`, preserving the no-false-positive outcome on the sparse live canvas.
- Fixed an integration bug found by the first live queue worker run: candidate manifests keep false-authority markers at candidate level, so the queue postcondition must use `required_false=[]` and `false_or_missing=[...]` for top-level authority fields.

The authority boundary remains unchanged: this is encoder-side planning and local candidate-manifest automation only. It does not claim score, promotion, rank/kill, or exact-eval readiness.

---

## TL;DR (operator-facing)

Built **8 NOT-BUILT extended operators** identified in DROP-MANY-REPLACE-COMPOSITION-APPARATUS-STATE-AUDIT memo (commit `1f62ac788`) Phase 1 catalog enumeration, addressing the operator's 2026-05-26 insight *"merge and other ops do even better and a combination and individual fractal optimization is likely even better"* and Hypothesis #2 EMPIRICAL grounding (rate-axis saturation at frontier operating point; need DISTORTION primitives).

The 8 operators land as **sister-disjoint module** to BUILD-2+3 sister subagent (in-flight on canonical 4 operations FULL_DROP / REPAIR / MASKED / FEATHERED on `pair_frame_scorer_geometry_lattice_5d_canvas`); my extension lives in dedicated module + dedicated CLI + dedicated test file per Catalog #230 sister-disjoint discipline.

**Original subagent proof**: ~1100 LOC operator module + ~330 LOC CLI +
~700 LOC tests. **Codex greenup proof** now covers the queue/cathedral bridge:
60 focused operator+queue tests pass, and 139 BUILD-1/BUILD-2/BUILD-2+3-EXT
canvas tests pass together.

**Codex addendum 2026-05-27T03:45Z**: hardened this landing after review with
positive `top_n` guards, strict canvas schema loading, atomic CLI writes,
`--operator all` batch mode, explicit zero-candidate blockers, a queue builder
that fans the eight operators through `experiment_queue.v1`, and a Tier A
cathedral consumer. The current live BUILD-1 latest canvas executes end-to-end
through the 8-step local queue at `local_cpu` concurrency 4; all eight steps
succeed and correctly emit zero candidates with blocker
`no_negative_predicted_delta_cells` because the current canvas has only three
cells and one feasible pair for the latest archive.

---

## Phase 1: 8 NOT-BUILT operators implemented

Per audit memo §"Phase 1 catalog enumeration", each operator with TECHNIQUE × WAY × TIME per 13th standing directive:

| # | Operator | LOC | TECHNIQUE | WAY | TIME |
|---|---|---|---|---|---|
| 1 | `replace_one` | ~130 | DISTORTION-axis primitive (substitute single pair selector) | `generate_replace_one_candidates` + `ReplaceOneParameters` | PRE-DISPATCH; $0 GPU |
| 2 | `replace_many` | ~160 | Beam-search per Catalog #356 per-axis decomposition | `generate_replace_many_candidates` + `ReplaceManyParameters` (beam_width/depth) | PRE-DISPATCH; $0 GPU |
| 3 | `merge_pair` | ~170 | Rate+distortion joint optimization via shared-encoding semantic | `generate_merge_pair_candidates` + `MergePairParameters` (max_candidates) | PRE-DISPATCH; $0 GPU |
| 4 | `reorder_pair` | ~140 | Entropy-coder context optimization (FEC8 2nd-order Markov sister per 6th directive) | `generate_reorder_pair_candidates` + `ReorderPairParameters` (block_size) | PRE-DISPATCH; $0 GPU |
| 5 | `drop_frame` | ~110 | Per-frame drop (finer-grained than canvas FULL_DROP) | `generate_drop_frame_candidates` + `DropFrameParameters` (which_frame) | PRE-DISPATCH; $0 GPU |
| 6 | `synthesize_frame` | ~110 | Per-frame synthesis per Atick-Redlich 1990 cooperative-receiver | `generate_synthesize_frame_candidates` + `SynthesizeFrameParameters` (synthesis_seed) | PRE-DISPATCH; $0 GPU |
| 7 | `motion_conditional` | ~160 | Per-pair operator selection conditioned on pose magnitude (Rao-Ballard 1999 predictive coding) | `generate_motion_conditional_candidates` + `MotionConditionalParameters` (motion_threshold_percentile + leaf ops) | PRE-DISPATCH; $0 GPU |
| 8 | `temporal_coherence` | ~170 | Cross-pair joint optimization (Wyner-Ziv 1976 side-information) | `generate_temporal_coherence_candidates` + `TemporalCoherenceParameters` (temporal_window + similarity_threshold) | PRE-DISPATCH; $0 GPU |

All 8 operators emit `ExecutableCandidate` rows per canvas's canonical contract (sister-disjoint with BUILD-2+3) + carry Catalog #356 `AxisDecomposition` per-axis attribution + canonical Provenance per Catalog #323 + Tier A canonical-routing markers per Catalog #341 (`predicted_delta_adjustment=0.0` + `promotable=False` + `axis_tag="[predicted]"`).

---

## Phase 2: Operator-facing CLI

`tools/apply_8_extended_operators_to_5d_canvas_cli.py` (~330 LOC):

- `--list` enumerates 8 operators + canonical equation IDs (FORMALIZATION_PENDING) for operator audit
- `--operator <name>` selects which operator to apply
- `--canvas-path <json>` consumes a BUILD-1-populated 5D canvas JSON
- Per-operator parameter flags (`--target-pair-idx`, `--beam-width`, `--max-merge-candidates`, `--reorder-block-size`, `--which-frame`, `--synthesis-seed`, `--motion-threshold-percentile`, `--temporal-window`, etc.)
- Emits canonical JSON manifest (sort_keys=True for byte-stable diffability)
- `--output <path>` writes to file; default stdout

Sister of `tools/apply_operation_to_5d_canvas_cli.py` (BUILD-2+3 CLI for canonical 4 operations); both CLIs can be invoked independently — BUILD-2+3 covers FULL_DROP / REPAIR / MASKED / FEATHERED, my CLI covers the 8 extended operators. Together they form the canonical 12-operator vocabulary (4 + 8) per audit memo enumeration.

---

## Phase 3: Tests (54/54 pass)

`src/tac/tests/test_8_extended_operators_5d_canvas.py` (~700 LOC; 54 tests):

- **Canonical contract conformance** (8 tests): CONSUMER_NAME / VERSION / HOOK_NUMBERS / TIER / 8-operator registry / 8 canonical equation IDs / update_from_anchor / consume_candidate per Catalog #335 + #357 + #341
- **Per-operator parameter dataclass invariants** (8 tests): each of 8 Parameters dataclasses validates types + ranges + invalid-value rejection
- **Per-operator correctness empty canvas** (8 tests): each operator returns `[]` on empty canvas
- **Per-operator correctness small canvas** (8 tests): each operator emits valid candidates on 4-pair synthetic canvas
- **Per-axis decomposition + canonical Provenance** (3 tests): every candidate carries `predicted_axis_decomposition: AxisDecomposition` + `catalog_323_provenance` per Catalog #356 + #323
- **Apples-to-apples baseline preservation** (1 test): per 10th standing directive; archive_sha256 threaded into Provenance
- **Cosine similarity + per-pair signature helpers** (5 tests): mathematical helpers for temporal-coherence
- **Sister BUILD-2+3 canvas operation proxy mapping** (2 tests): all 8 extended operators map to canonical `CanonicalOperation` proxy for compatibility
- **Cross-operator composition per 8th INDIVIDUALLY-FRACTAL** (2 tests): all 8 operators together emit non-empty union; top_n truncation observed
- **Determinism + sorting** (2 tests): identical inputs → identical outputs; ascending by predicted_delta_score
- **JSON round-trip serialization** (1 test): `ExecutableCandidate.as_dict()` JSON-safe via sort_keys
- **Schema constants** (1 test): EXTENDED_MODULE_SCHEMA pinned
- **Live-repo regression guard** (2 tests): module + CLI importable
- **Operator's "merge and other ops do even better" insight regression** (2 tests): merge-pair emits candidates with negative deltas; 8 operators register canonical equation IDs

---

## 8th INDIVIDUALLY-FRACTAL per-substrate optimization tree declaration

Per 8th MLX-first numpy-portable INDIVIDUALLY-FRACTAL standing directive 2026-05-26:

**The 8 operators × per-substrate-fractal decomposition tree** produces the canonical optimization space:

```
For each registered substrate S in (PR101-fec6-fixed-huffman-k16 / cascade_a / cascade_b / cascade_c_prime / DQS1 / V14-V2 / sister):
    For each operator O in (REPLACE_ONE / REPLACE_MANY / MERGE_PAIR / REORDER_PAIR /
                            DROP_FRAME / SYNTHESIZE_FRAME / MOTION_CONDITIONAL / TEMPORAL_COHERENCE):
        Per-substrate-optimal engineering decision per UNIQUE-AND-COMPLETE-PER-METHOD:
            - operator semantics adapted to substrate's archive grammar
            - per-substrate per-pair scorer-axis sensitivity attribution
            - per-substrate per-frame master-gradient differentials
            - per-substrate-operator-specific predicted_delta_score model
            - sister probe-disambiguator paths per Catalog #313
```

Each `(substrate, operator)` cell in this tree is a sister-subagent-territory; the 8 operators ARE the universal alphabet, and the per-substrate fractal decomposition expands each leaf into substrate-optimal engineering per the 8th directive's INDIVIDUALLY-FRACTAL discipline.

---

## 13th OPTIMAL-TRIO declaration per operator

Per 13th OPTIMAL-TRIO standing directive 2026-05-26 (TECHNIQUE × WAY × TIME):

| Operator | TECHNIQUE | WAY | TIME |
|---|---|---|---|
| replace_one | linear-substitution-distortion at PR106 frontier per CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent" | per-pair candidate enumeration filtered by receiver_feasibility; Catalog #356 per-axis attribution; canonical Provenance threading | PRE-DISPATCH (canonical-submission-pipeline 10-phase Phase 4 builder per 9th amendment); $0 paid GPU + ~30 min wall-clock per operator FIRE |
| replace_many | beam-search per Catalog #356 over per-pair substitutions (sister of `tac.optimization.dqs1_drop_many_beam`) | beam_width/beam_depth bounded; per-axis Pareto ranking; canonical AxisDecomposition per Catalog #356 | PRE-DISPATCH; $0 GPU + ~30 min FIRE |
| merge_pair | Dykstra alternating-projections convex feasibility (sister of META-LIFT-2 `tac.pareto_polytope_unified_solver.solver`) | O(N^2) bounded by max_candidates; shared-encoding rate saving via |bytes_a - bytes_b|; averaged distortion | PRE-DISPATCH; $0 GPU + ~30 min FIRE |
| reorder_pair | Entropy-coder context optimization (sister of FEC8 2nd-order TRUE Markov VARIANT-A per 6th directive) | block_size bounded permutation; rank ascending by total_delta within block; reorder-savings model | PRE-DISPATCH; $0 GPU + ~30 min FIRE |
| drop_frame | Per-frame master-gradient (sister of canvas FULL_DROP at frame granularity per `tac.cathedral_consumers.per_frame_sensitivity_consumer`) | filter by frame parity (which_frame); rank by composite predicted_delta_score per (frame_idx, cpu_cuda_axis) | PRE-DISPATCH; $0 GPU + ~30 min FIRE |
| synthesize_frame | Atick-Redlich 1990 cooperative-receiver (CLAUDE.md grand-council attendees Atick + Redlich + Tishby + Zaslavsky + Wyner) | per-frame synthesis with deterministic synthesis_seed; per-pair I(X;T)/I(T;Y) information-bottleneck decomposition | PRE-DISPATCH; $0 GPU + ~30 min FIRE |
| motion_conditional | Rao-Ballard 1999 predictive coding (CLAUDE.md grand-council attendees Rao + Ballard + Tishby) | per-pair motion magnitude from PoseNet-axis cells; percentile cutoff; per-classification leaf operator | PRE-DISPATCH; $0 GPU + ~30 min FIRE |
| temporal_coherence | Wyner-Ziv 1976 source-coding-with-side-information (CLAUDE.md grand-council attendees Wyner + Tishby + Zaslavsky) | per-pair AxisDecomposition signature + cosine similarity; cross-pair within temporal_window; 5% Wyner-Ziv saving | PRE-DISPATCH; $0 GPU + ~30 min FIRE |

---

## Integration verification

### Integration with 5D canvas SCAFFOLD (sister BUILD-1 + BUILD-2+3)

- **BUILD-1 (5D canvas populator)**: my module imports canonical `PairFrameScorerGeometryLattice` + `PairFrameScorerGeometryCell` per the canvas's frozen contract. Tests build synthetic canvas via the canonical constructor `PairFrameScorerGeometryLattice(archive_sha256=..., cells=...)`. 43 sister BUILD-1 populator tests still pass after my landing (no collateral damage).
- **BUILD-2+3 (canvas canonical 4 operations)**: my module's `_extended_to_canvas_operation_proxy` maps each of 8 extended operators to a canvas `CanonicalOperation` member (REPAIR / MASKED / FULL_DROP / FEATHERED) so `ExecutableCandidate.operation` remains typed-compatible with BUILD-2+3's frozen contract. Recipe hint carries the canonical `ExtendedOperation` value explicitly via `hint["operation"]` so downstream consumers distinguish 8 extended from 4 canonical.
- **Cross-operator composition test** (`test_cross_operator_composition_8_extended_emit_disjoint_candidates`): all 8 operators emit candidates on the same small canvas; their union is the canonical 8-operator vocabulary. Sister BUILD-2+3's 4 operations COMPOSE with my 8 via the proxy mapping; together they form the canonical 12-operator vocabulary per audit memo.

### Integration with cathedral autopilot (BUILD-4 sister territory)

Module declares `CONSUMER_NAME` + `CONSUMER_VERSION` + `CONSUMER_HOOK_NUMBERS` + `CONSUMER_TIER = ConsumerTier.TIER_A_OBSERVABILITY_ONLY` per Catalog #335 + #357 contract. BUILD-4 sister subagent op-routable promotes to Tier B with canonical-routing markers per Catalog #341 + canonical Provenance per Catalog #323 + per-axis AxisDecomposition per Catalog #356 across all 8 operators in a single PR.

---

## 6-hook wire-in declaration per Catalog #125

- **hook #1 sensitivity-map** = ACTIVE (each operator's per-pair / per-frame selector is a sensitivity-map producer; AxisDecomposition surfaces seg/pose/rate axes)
- **hook #2 Pareto constraint** = ACTIVE (per-operator (predicted_delta_score, predicted_byte_cost) IS the rate-vs-distortion Pareto signal; sister of META-LIFT-2 Dykstra alternating-projections)
- **hook #3 bit-allocator** = ACTIVE (per-operator predicted_byte_cost feeds the bit-allocator for downstream archive layout)
- **hook #4 cathedral autopilot dispatch** = ACTIVE-AT-BUILD-4-PROMOTION (BUILD-4 Tier B promotion via Catalog #357 auto-discovered per Catalog #335)
- **hook #5 continual-learning posterior** = ACTIVE (8 operators' empirical anchors feed canonical equation registry per Catalog #344 8 FORMALIZATION_PENDING entries)
- **hook #6 probe-disambiguator** = ACTIVE (8 operators ARE the canonical probe-disambiguators for audit memo's 5 hypotheses + operator insight "merge and other ops do even better")

---

## Catalog #344 status per operator (8 FORMALIZATION_PENDING entries)

Per Catalog #344 + audit memo §"PRIORITY 5 Catalog #344 canonical equations registry growth":

| Operator | Canonical equation ID (FORMALIZATION_PENDING) | Status |
|---|---|---|
| replace_one | `replace_one_via_linear_substitution_distortion_v1` | FORMALIZATION_PENDING (registration after 3+ paired-axis anchors per Catalog #344 trigger) |
| replace_many | `replace_many_via_beam_search_per_axis_decomposition_v1` | FORMALIZATION_PENDING |
| merge_pair | `merge_pair_via_rate_distortion_joint_optimization_v1` | FORMALIZATION_PENDING |
| reorder_pair | `reorder_pair_via_entropy_coder_context_markov_v1` | FORMALIZATION_PENDING |
| drop_frame | `drop_frame_via_per_frame_master_gradient_v1` | FORMALIZATION_PENDING |
| synthesize_frame | `synthesize_frame_via_atick_redlich_cooperative_receiver_v1` | FORMALIZATION_PENDING |
| motion_conditional | `motion_conditional_via_rao_ballard_predictive_coding_v1` | FORMALIZATION_PENDING |
| temporal_coherence | `temporal_coherence_via_wyner_ziv_side_information_v1` | FORMALIZATION_PENDING |

Per CLAUDE.md "Canonical equations + models registry — NON-NEGOTIABLE": this landing does NOT register canonical equations yet (no paired-axis empirical anchors). The 8 IDs are queued for sister-subagent registration AFTER FIRE-phase paired CPU+CUDA dispatch produces 3+ anchors per operator. The canonical equation registry's `canonical_producers` will be the corresponding `generate_*_candidates` functions in my module; `canonical_consumers` will be the cathedral autopilot ranker (post-BUILD-4 Tier B promotion).

---

## Discipline closure

### Files touched (canonical APPEND-ONLY per Catalog #110/#113)

- NEW `src/tac/optimization/pair_frame_scorer_geometry_lattice_5d_canvas_extended_operators.py` (~1100 LOC; 8 operators + 8 parameter dataclasses + 8 generators + canonical contract hooks + canvas operation proxy + helpers)
- NEW `tools/apply_8_extended_operators_to_5d_canvas_cli.py` (~330 LOC; operator-facing CLI)
- NEW `src/tac/tests/test_8_extended_operators_5d_canvas.py` (~700 LOC; 54 tests)
- NEW `src/comma_lab/scheduler/pair_frame_5d_extended_operator_queue.py` (queue builder for 8 local operator jobs)
- NEW `tools/build_5d_extended_operator_queue.py` (operator-facing queue builder CLI)
- NEW `src/tac/tests/test_pair_frame_5d_extended_operator_queue.py` (queue-builder tests)
- NEW `src/tac/cathedral_consumers/pair_frame_5d_extended_operator_consumer/__init__.py` (Tier A cathedral/autopilot visibility)
- NEW `.omx/research/build_2_3_ext_8_not_built_operators_landed_20260526.md` (this memo)

NO mutation of:

- BUILD-1 populator + 5D canvas core (sister BUILD-2+3 in-flight)
- BUILD-2+3 canonical 4 operations on canvas
- Catalog gates / preflight
- canonical equation registry (Catalog #344 entries queued FORMALIZATION_PENDING)
- canonical state JSONL files
- existing memos
- AGENTS.md / CLAUDE.md
- any sister design memo or landing memo

### Sister coordination

- Slot BUILD-2+3 canonical 4 operations: DISJOINT (sister BUILD-2+3 owns `pair_frame_scorer_geometry_lattice_5d_canvas.py` + `tools/apply_operation_to_5d_canvas_cli.py`); my landing is sister-disjoint extension module + extension CLI + extension tests; reuses canvas's frozen contract surfaces (PairFrameScorerGeometryLattice / PairFrameScorerGeometryCell / ExecutableCandidate / CanonicalOperation / ScorerAxis / ReceiverRuntime / CpuCudaAxis) WITHOUT modifying them.
- Slot BUILD-1 populator: DISJOINT (sister BUILD-1 owns `pair_frame_scorer_geometry_lattice_5d_canvas_populator.py`); my landing depends on BUILD-1's canvas-population output via the canvas's `cells` map.
- Slot V14-V2 / V15 / cascade_c_prime / Phase 4-6 / Catalog #368-#369 / etc.: DISJOINT (orthogonal scopes).
- Per Catalog #340 sister-checkpoint guard: PROCEED at audit-start checkpoint; sister BUILD-2+3 checkpoint at `pair_frame_scorer_geometry_lattice_5d_canvas.py` is for THEIR module; my module is `pair_frame_scorer_geometry_lattice_5d_canvas_extended_operators.py` (different file).

### Apparatus discipline acknowledgment

- Catalog #1 + #192: candidate predictions are non-promotable (Tier A); promotion requires paired CPU+CUDA empirical anchors via FIRE phase
- Catalog #287: every empirical claim carries `[predicted]` evidence tag; no docstring-overstatement (no "saves N%" claims without evidence anchor)
- Catalog #323: canonical Provenance umbrella; every candidate's catalog_323_provenance built via `build_provenance_for_predicted` per Catalog #356 STRICT preflight gate
- Catalog #335 + #357: canonical contract conformance + Tier A scaffold
- Catalog #341: canonical-routing markers non-promotable defaults
- Catalog #343: NO hardcoded score literals in any source file (only operator vocabulary primitives)
- Catalog #344: 8 canonical equation IDs queued FORMALIZATION_PENDING (registration AFTER 3+ paired-axis anchors per operator)
- Catalog #110/#113 APPEND-ONLY: NEW files only; zero mutation
- Catalog #125: 6-hook wire-in declaration above
- CLAUDE.md "Forbidden premature KILL": no kill verdicts; operator insight "merge and other ops do even better" PRESERVED per Catalog #307 paradigm-vs-implementation classification
- CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode": 8 operators × per-substrate fractal decomposition tree IS the canonical sister-disjoint extension

### Mission contribution per Catalog #300

`frontier_breaking_enabler` — these 8 operators unblock the operator's 2026-05-26 insight ("merge and other ops do even better and a combination and individual fractal optimization is likely even better") + audit memo Hypothesis #2 (rate-axis saturation at frontier; need DISTORTION primitives) at the operator-vocabulary surface. Per CLAUDE.md "Mission alignment — non-negotiable" Consequence 4: frontier-breaking moves DOMINATE rigor budget. Per the audit memo §"Direct answer to operator question": "REPAIR / MASKED / FEATHERED operations may produce a candidate that beats `0.192028282 [contest-CPU]` frontier per the canonical equation candidate's prediction" — my 8 operators extend this prediction space by 8 additional operator paradigms, each with its own predicted-axis attribution + canonical equation candidate + literature anchor (Atick-Redlich / Rao-Ballard / Wyner-Ziv / FEC family).

---

## Operator-routable next

### Option A: FIRE phase paired-axis dispatch on top-3-ranked operators per Pareto polytope guidance

Per audit memo §"PRIORITY 4 PAIRED CPU+CUDA DISPATCH wave":

- BUILD-1 sister subagent populates 5D canvas with empirical cells (~$0 GPU + ~2-4h wall-clock).
- Operator runs `tools/apply_8_extended_operators_to_5d_canvas_cli.py --canvas-path <path> --operator <op> --top-n 16` for each of 8 operators.
- Top-3 candidates per operator × 2 paired axes = 48 dispatches × ~$0.30 smoke + ~$1-3 paired-axis = estimated **$20-60 paid GPU** for full FIRE phase.
- 3+ empirical anchors per operator register canonical equations per Catalog #344.

### Option B: PROMOTE phase per 3+ paired-axis anchors

After Option A produces 3+ paired-axis anchors per operator:

- Sister subagent registers 8 canonical equations per Catalog #344 via `tools/list_canonical_equations.py` + sister `register_canonical_equation` helper.
- BUILD-4 sister subagent op-routable: Tier B promotion per Catalog #357 (replace `CONSUMER_TIER = ConsumerTier.TIER_A_OBSERVABILITY_ONLY` with `ConsumerTier.TIER_B_SCORE_CONTRIBUTING` + canonical-routing markers per Catalog #341 + per-axis Provenance per Catalog #356 + sister `update_from_anchor` for canonical posterior updates per Catalog #344).
- Auto-discovered by cathedral autopilot loop per Catalog #335.

### Option C: META-LIFT integration into beam-search + Pareto polytope

Per audit memo §"RANK 3 Hypothesis #4 (combinatorial explosion + needs guided search via META-LIFT-1 + META-LIFT-2)":

- Sister subagent wires my 8 operators' `predicted_axis_decomposition` per-axis attribution into META-LIFT-2 `tac.pareto_polytope_unified_solver.solver` for cross-operator Dykstra-feasibility.
- Top-ranked candidate per operator becomes a constraint vertex in the polytope; META-LIFT-1 Cauchy-Schwarz bound on cross-operator correlation drives the joint optimization.
- Estimated: ~$0 GPU + ~2-4h sister wall-clock for META-LIFT-1 + META-LIFT-2 integration.

---

## Cross-references

- `.omx/research/drop_many_replace_composition_apparatus_state_audit_20260526.md` (audit memo §"Phase 1 catalog enumeration" + §"PRIORITY 1-5" recommendations)
- `.omx/research/v14_v2_cascade_a_fec10_substitution_onto_dqs1_frontier_landed_20260526.md` (FRONTIER-CROSSING anchor; canonical equation #344 PROMOTED 3→5 anchors)
- `src/tac/optimization/pair_frame_scorer_geometry_lattice_5d_canvas.py` (sister BUILD-2+3 canonical 4 operations on canvas)
- `src/tac/optimization/pair_frame_scorer_geometry_lattice_5d_canvas_populator.py` (sister BUILD-1 populator)
- `tools/apply_operation_to_5d_canvas_cli.py` (sister BUILD-2+3 CLI)
- `src/tac/optimization/dqs1_drop_many_beam.py` (drop-many beam sister; replace-many borrows beam-search pattern)
- `src/tac/cross_substrate_master_gradient_analyzer/` (META-LIFT-1 Cauchy-Schwarz; sister for cross-operator correlation)
- `src/tac/pareto_polytope_unified_solver/` (META-LIFT-2 Dykstra alternating-projections; sister for merge-pair convex feasibility)
- `src/tac/uniward_invariant_enumerator/` (META-LIFT-4 sister for masked/feathered referenced by canvas BUILD-2+3)
- `src/tac/cathedral/consumer_contract.py` (Catalog #335 + #357 canonical contract; AxisDecomposition + ConsumerTier + HookNumber)
- `src/tac/provenance/builders.py` + `validator.py` (Catalog #323 canonical Provenance)
- CLAUDE.md "Meta-Lagrangian/Pareto solver — NON-NEGOTIABLE, HIGHEST EMPHASIS"
- CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent (UPDATED 2026-05-04)"
- CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode"
- CLAUDE.md "Grand Council (advisory)" Atick-Redlich + Rao-Ballard + Wyner-Ziv + Tishby + Zaslavsky grand-council attendees
- 7th + 8th + 10th + 11th + 12th + 13th standing directives binding

## Lane registration

Lane `lane_build_2_3_ext_8_not_built_operators_replace_merge_reorder_frame_level_motion_conditional_temporal_coherence_20260526` L1 (impl_complete + 8-operator-module + extended-CLI + comprehensive-tests + landing-memo + 6-hook-declaration + 13th-OPTIMAL-TRIO-per-operator + 8th-INDIVIDUALLY-FRACTAL-decomposition-tree + 8 FORMALIZATION_PENDING canonical equation IDs queued).
