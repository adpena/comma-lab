# Orphan-signal audit for score-lowering — 2026-05-15

**Lane**: `lane_orphan_signal_audit_20260515` (L1; research_only=true at this turn)
**Subagent**: `ORPHAN-SIGNAL-AUDIT-AND-WIRE-IN-FOR-SCORE-LOWERING-20260515`
**Operator directive verbatim**: *"need another wiring and integration and signal pass, especially if there is anything we can leverage in our score lowering from past work; for example i totally forgot about the sensitivity shards or whatever"*

Per CLAUDE.md "Subagent coherence-by-default" + "Mandatory wire-in for every landing
(no orphaned signals)", every producer surface in the repo must wire into at
least one of the 6 canonical solver hooks (sensitivity-map / Pareto / bit-allocator
/ cathedral-autopilot / continual-learning / probe-disambiguator) OR be tagged
`research_only=true`. Producers in this audit lacking BOTH a production consumer
AND `research_only=true` are ORPHAN signals — past investment that is not
currently affecting score-lowering decisions.

---

## TL;DR

| Class | Count | Examples |
|---|---:|---|
| WIRED-AND-CONSUMED | 8 | sensitivity_map_artifacts (path enumerated) / continual_learning posterior / cost_band_calibration / cathedral_autopilot ranker / Catalog #243 local_pre_deploy_check / Modal call_id ledger / macos_cpu_advisory_signal / register_substrate decorator |
| WIRED-BUT-DORMANT | 4 | DARTS supernet (only `search_time_traveler_supernet.py`) / theoretical_floor_substrate_refresh (sister-only) / cooperative_receiver_integration (manifest-builder-only) / cross_paradigm_composition_examples |
| **PRODUCED-NOT-CONSUMED (ORPHAN)** | **15** | **xray.wire_in (37 primitives across 6 hooks)** / **all 11 composition primitives** (frontier / wbce_mera / bregman / sinkhorn / distillation_chain / hypernetwork / product_of_experts / td_lora / adapters) / **Rust crates 6/7** (qma / stbm1br / residual / raw_writer / zipwire / tac_packet_compiler) / **bit_allocator_end_to_end** (no substrate-trainer consumer) / **field_equation_planner** (tools-only) / **lapose_foveation_atoms** (manifest-builder-only) / **joint_admm_coordinator** (sister-tools only) / **codec_op_admm_adapter** / **hyperbolic_foveation** / **JSCC entropy coder** / **scorer_conditional_mdl** (xray-internal-only) / **predictive_coding_hierarchy** (xray-internal-only) / **score_lipschitz** (xray-internal-only) / **vq_codebook_coverage** (xray-internal-only) |
| CODE-EXISTS-NEVER-RUN | 0 | (every producer has at least tests + a sister tool that imports it) |

**Headline finding**: the entire `tac.xray.wire_in` package — designed explicitly as
the canonical wire-in surface for the 6 hooks per CLAUDE.md "Subagent coherence-by-default"
— has **ZERO production consumers**. The 37 xray primitives across 6 hooks
(`bit_allocator: 10`, `sensitivity_map: 11`, `cathedral_autopilot: 3`,
`continual_learning: 1`, `pareto_constraint: 3`, `probe_disambiguator: 9`)
are produced but the canonical bundle iterator (`wire_in_for_hook`,
`XRayWireInBundle`, `discover_primitives_by_hook`, `aggregate_hook_evidence_grade`)
is consumed only by its own tests.

This is the meta-canonical wire-in surface. If it is orphaned, every individual
xray primitive is also orphaned (the wire-in bundle was the consumer that was
supposed to make them all reachable).

---

## Per-producer matrix (18+ surfaces audited)

### 1. Sensitivity shards / sensitivity-map artifacts (operator's "i totally forgot")

| Sub-surface | Path | Status | Notes |
|---|---|---|---|
| `experiments/modal_component_sensitivity_shards.py` | dispatch surface | WIRED-BUT-DORMANT | Generated 2 historical artifact dirs (`component_sensitivity_pfp16_a_plus_plus_cpu_smoke_20260501` + `modal_component_sensitivity`); not invoked in current dispatch path |
| `src/tac/component_sensitivity_artifact.py` (70KB, 70 functions) | producer + helpers | WIRED-BUT-DORMANT | 9 sister consumers (lightning batch_jobs, lagrangian_per_tensor_allocation, hidden_gems, sensitivity_weighted tools, build_a2_sensitivity_weighted_pr101_packet); but NO substrate trainer + NO cathedral autopilot consumer |
| `src/tac/sensitivity_map/` package | producer | WIRED partial | `discover_sensitivity_map_artifacts` enumerates `.pt` files; but `axis_weights_for_named_operating_point` (16KB module) consumed only by its own tests |
| `experiments/build_sensitivity_map_pr106.py` | dispatch surface | WIRED | produced live artifact `experiments/results/posenet_sensitivity_v5/sensitivity_map.pt` |
| `experiments/profile_component_sensitivity.py` | producer | WIRED-BUT-DORMANT | profiling pass; no consumer in current dispatch path |
| `experiments/certify_component_sensitivity_maps.py` | producer | WIRED-BUT-DORMANT | certification pass; no consumer in current dispatch path |

**Net assessment**: the operator's recall is correct — sensitivity shards are
produced but are functionally orphaned at the dispatch path. The cathedral
autopilot enumerates sensitivity-map paths (W/I/A I-3 wire-in 2026-05-12) but
does NOT actually CONSUME the per-tensor importance weights to inform ranking.
**HIGH-EV wire-in target**: cathedral autopilot ranker should LOAD a
sensitivity-weighted prior and apply it to each candidate's `predicted_score_delta`
proportional to the candidate's per-tensor footprint.

### 2. Composition primitives (`src/tac/composition/`)

| Primitive | LOC | Status | Notes |
|---|---:|---|---|
| `darts_supernet.py` | 35.8K | WIRED-BUT-DORMANT | only `search_time_traveler_supernet.py` (1 dispatch wrapper) + `diagnose_supernet_ranking.py` (1 tool) consume |
| `frontier_primitives.py` | 21.2K | **ORPHAN** | only intra-package consumers (sister composition modules) + tests |
| `wbce_mera.py` | 25.8K | **ORPHAN** | only intra-package + tests + xray_substrate_classifier (tool) |
| `bregman_mixing.py` | 17.5K | **ORPHAN** | only intra-package + tests |
| `sinkhorn_ot_mixing.py` | 14.8K | **ORPHAN** | only intra-package + tests |
| `distillation_chain.py` | 12.1K | **ORPHAN** | only intra-package + tests |
| `hypernetwork.py` | 12.2K | **ORPHAN** | only intra-package + tests |
| `product_of_experts.py` | 14.3K | **ORPHAN** | only intra-package + tests |
| `td_lora.py` | 18.4K | **ORPHAN** | only intra-package + tests |
| `adapters.py` | 23.0K | **ORPHAN** | only intra-package + tests |
| `distillation.py` | 10.1K | WIRED-marginal | `experiments/train_distill.py` consumes; but no substrate trainer route |
| `enumerate.py` | 28.7K | WIRED | `tools/build_composition_ranking_json.py` + `tools/rank_composition_cells_by_ev.py` consume |
| `registry.py` | 63.5K | WIRED | `tools/build_composition_ranking_json.py` + `tools/cathedral_autopilot_autonomous_loop.py` + `tools/rank_composition_cells_by_ev.py` consume |
| `stack_of_stacks/__init__.py` | (pkg) | WIRED | `experiments/train_substrate_stack_of_stacks.py` consumes |

**Net assessment**: ~9 of 14 composition modules (~150K LOC) are orphan signals.
They are the canonical "stacking sweep" / "cross-paradigm composition" math
substrate the operator built last year + this Spring. The autopilot ranker
consumes the registry but NOT any composition primitive — meaning composition
candidates are RANKED but never EXECUTED via the canonical primitives.

### 3. Solver-stack primitives (`src/tac/optimization/`)

| Primitive | Status | Notes |
|---|---|---|
| `bit_allocator_end_to_end.py` (34K) | **ORPHAN at substrate boundary** | consumed by `cross_paradigm_composition_examples` + `theoretical_floor_substrate_refresh` + `tac.xray.shannon_vector_r_d` + `tac.xray.wire_in` (all sister-modules); ZERO substrate trainer routes |
| `field_equation_planner.py` (29.9K) | TOOLS-ONLY | `tools/build_field_meta_dispatch_selection.py` + `tools/build_field_equation_plan.py` consume; NO substrate trainer; NO autopilot consumer |
| `autopilot_dispatch_ranking.py` (19.5K) | WIRED | cathedral autopilot consumes |
| `theoretical_floor_substrate_refresh.py` (20.9K) | **ORPHAN** | only `bit_allocator_end_to_end` + own tests; no production consumer |
| `cooperative_receiver_integration.py` | WIRED-BUT-DORMANT | `tools/operator_briefing.py` + `tools/build_cooperative_receiver_integration_manifest.py` consume; no substrate trainer route |
| `cross_paradigm_composition_examples.py` | **ORPHAN** | only own tests |
| `substrate_composition_matrix.py` | WIRED | cathedral autopilot consumes via Catalog #227 ranker |
| `macos_cpu_advisory_signal.py` | WIRED | per Catalog #192 |

### 4. Xray primitives package (`src/tac/xray/`) — the meta-canonical orphan

| Primitive | Status |
|---|---|
| `wire_in.py` (canonical bundle iterator) | **ORPHAN** — ZERO production consumers; only tests |
| `registry.py` (37 xray primitive specs) | WIRED-BUT-DORMANT (sister wire_in.py reads it; nothing reads wire_in's output) |
| `mdl_scorer_conditional.py` | WIRED-via-tools | consumed by `tac.analysis.scorer_conditional_mdl` |
| `posenet_se3_lie_algebra.py` | **ORPHAN** | only intra-package + tests |
| `unified_action_principle.py` | **ORPHAN** | only intra-package + tests |
| `wavelet_hf_energy.py` | **ORPHAN** | only intra-package + tests |
| `per_pair_score_decomposition.py` | **ORPHAN** | only intra-package + tests |
| `shannon_vector_r_d.py` | **ORPHAN** | only intra-package + tests |
| `score_lipschitz.py` | **ORPHAN** | only intra-package + tests |
| `predictive_coding_hierarchy.py` | **ORPHAN** | only intra-package + tests |
| `bilinear_resize_nullspace.py` | **ORPHAN** | only intra-package + tests |
| `vq_codebook_coverage.py` | **ORPHAN** | only intra-package + tests |
| `segnet_margin_polytope.py` | WIRED-via-substrate | D1 substrate consumes |
| `foveation_ego_motion.py` | **ORPHAN** | only intra-package + tests |
| `yuv6_sublattice_geometry.py` | **ORPHAN** | only intra-package + tests |

### 5. Foveation primitives

| Module | Status |
|---|---|
| `lapose_foveation_atoms.py` | WIRED-BUT-DORMANT | `tools/build_lapose_foveation_atom_manifest.py` consumes; substrate trainer `train_substrate_a1_plus_lapose.py` does NOT import lapose_foveation_atoms (recipe exists, but consumer-side wire missing) |
| `lapose_foveation_payload_candidate.py` | WIRED-BUT-DORMANT | `tools/build_lapose_foveation_payload_archive.py` + `tools/build_lapose_foveation_tuple_payload.py` consume |
| `lapose_foveation_runtime_skeleton.py` | **ORPHAN** | only own tests |
| `hyperbolic_foveation.py` | **ORPHAN** | only `tools/audit_hyperbolic_foveation_readiness.py` (audit-only) + `c067_hotspot_mask_geometry_compiler` (1 dispatch surface) |
| `foveation_field.py` | **ORPHAN** | only own tests |

### 6. Joint-ADMM coordinator family

| Module | Status |
|---|---|
| `joint_admm_coordinator.py` | WIRED-BUT-DORMANT | sister tools consume; no substrate trainer co-optimization route |
| `joint_admm_proximal_water_filling_v2.py` | WIRED-BUT-DORMANT | sister tools |
| `joint_admm_proximal_pose_delta.py` | WIRED-BUT-DORMANT | sister tools |
| `codec_op_admm_adapter.py` | **ORPHAN** | sister tool only |
| `codec_pipeline_joint_admm.py` | **ORPHAN** | sister tool only |
| `paradigm_delta_epsilon_zeta/joint_lagrangian_admm.py` | WIRED | T1 Balle endtoend trainer consumes |

### 7. Rust native primitives (`runtime-rs/crates/`)

| Crate | Python consumer | Status |
|---|---|---|
| `qma-codec` | `experiments/profile_qma9_native_decode.py` (probe-only) | **ORPHAN** at substrate boundary |
| `stbm1br-codec` | 5 build_pr85_*.py + `experiments/profile_stbm1br_rust_decode.py` (probe-only) | **ORPHAN** at substrate boundary; PR85 candidates were built but not in current dispatch queue |
| `residual-codec` | None | **ORPHAN** |
| `raw-writer` | None | **ORPHAN** |
| `zipwire` | None | **ORPHAN** (used internally by inflate-cli only) |
| `tac-packet-compiler` | tools/{contest_packet_compiler,submission_packet_compiler}.py | WIRED |
| `tac-packet-compiler-wasm` | None (browser-target) | research_only |
| `python-ast-indexer` | None | **ORPHAN** |
| `inflate-cli` | only as Cargo binary; no `submissions/*/inflate.py` invokes Rust | **ORPHAN** at submission runtime |

**Net assessment**: 6 of 9 Rust crates are orphan at the substrate / submission
boundary. Inflate.py paths in submissions don't invoke the Rust runtime; the
Rust path is only activated by tools/*.py compilers. This means the substantial
performance work invested in Rust crates is not lowering inflate-time scores OR
enabling new submission grammars.

### 8. JSCC + scorer-conditional codec primitives

| Module | Status |
|---|---|
| `tac.codec.jscc.archive_format` | **ORPHAN** | only own tests + sister jscc/entropy_coder |
| `tac.codec.jscc.entropy_coder` | **ORPHAN** | only own tests |
| `tac.codec.cooperative_receiver` | WIRED-BUT-DORMANT | `tac.composition.registry` references; no substrate trainer route |

### 9. DARTS supernet + time-traveler (`tac.composition.darts_supernet`)

WIRED-BUT-DORMANT. Only `experiments/search_time_traveler_supernet.py` invokes
it; `tools/diagnose_supernet_ranking.py` is a sister diagnostic tool. Per
Catalog #227 the time_traveler L5 substrate trainer exists; verify it routes
through DARTS.

### 10. Hidden gems / forensics

| Module | Status |
|---|---|
| `tac.hidden_gems` | WIRED-BUT-DORMANT | sister consumers (lightning batch_jobs, lagrangian_per_tensor_allocation); no substrate / autopilot route |
| `tac.forensics` | WIRED via sister sensitivity-map module |

---

## HIGH-EV wire-in targets (defer to follow-up wave; this turn lands audit only)

This turn lands the AUDIT memo + lane registration ONLY. Code-side wire-ins
deferred because the working tree contains 79 unresolved merge conflicts from
concurrent sister subagents (NEM-7-PR101-INTAKE / WAVE-recipe-canon-backfill /
others). Per CLAUDE.md "Race-mode rigor inversion" + "Subagent coherence-by-default"
sister-subagent ownership map (Catalog #230), introducing additional code edits
into a 79-conflict working tree would compound the integration cost.

The 5 HIGH-EV wire-ins below are operator-routable and become the basis of
follow-up subagent dispatches when the working tree is clean:

### Op-routable #1 — CRITICAL — cathedral autopilot consumes xray.wire_in bundles

**Producer**: `tac.xray.wire_in.discover_primitives_by_hook()` + `wire_in_for_hook(hook=H, targets=...)` returning `XRayWireInBundle` with 37 primitives across 6 hooks.
**Consumer**: `tools/cathedral_autopilot_autonomous_loop.py::load_planner_posterior_for_loop` — should add `xray_wire_in_inventory` key to the planner context payload (sister of the existing `sensitivity_map_inventory` key).
**Predicted score impact**: ~3-5% movement via better candidate ranking — the autopilot ranker would weight candidates by how many xray primitives they engage per hook (high-engagement candidates have more empirical grounding).
**Effort**: ~50 LOC in cathedral_autopilot_autonomous_loop.py + new context-payload field.
**Self-protection**: NEW Catalog #247 (`check_cathedral_autopilot_payload_includes_xray_wire_in_inventory`) — refuses any state of `tools/cathedral_autopilot_autonomous_loop.py::load_planner_posterior_for_loop` that drops the `xray_wire_in_inventory` field. CLAIMED 2026-05-15 commit `c2d538f7d`.

### Op-routable #2 — HIGH — composition primitives reach DARTS supernet routing

**Producer**: 9 orphan composition primitives (`frontier_primitives`, `wbce_mera`, `bregman_mixing`, `sinkhorn_ot_mixing`, `distillation_chain`, `hypernetwork`, `product_of_experts`, `td_lora`, `adapters`).
**Consumer**: `tac.composition.darts_supernet.SuperNet` should be able to enumerate composition cells via these primitives during architecture search. Currently DARTS only sees the registry.
**Predicted score impact**: ~5-10% via cross-paradigm stacking that the operator architected but never executed end-to-end.
**Effort**: medium (~200 LOC); council-grade per CLAUDE.md "Design decisions" because composition primitive ordering changes the Dykstra-Pareto frontier shape.

### Op-routable #3 — HIGH — substrate trainers route through bit_allocator_end_to_end

**Producer**: `tac.optimization.bit_allocator_end_to_end` (34KB, ~1k LOC of canonical bit-allocator math).
**Consumer**: every substrate trainer's `pack_archive` callsite. Currently each substrate hand-rolls its own bit allocator (per CLAUDE.md "tac stays clean" + the `axis_weights` per-tensor allocation discipline). Council-grade per the WIRE-AND-INTEGRATE audit's "deferred-to-deterministic-packet-compiler" note.
**Predicted score impact**: ~2-5% via uniform bit-allocator behavior across substrates (consistency dividend).
**Effort**: large (~500 LOC); council-grade.

### Op-routable #4 — MEDIUM — sensitivity-weighted prior in cathedral autopilot ranker

**Producer**: `experiments/results/posenet_sensitivity_v5/sensitivity_map.pt` + 9 sister sensitivity-weighted helpers.
**Consumer**: `tools/cathedral_autopilot_autonomous_loop.py::rank_candidates` should LOAD a sensitivity-weighted prior and apply it to `predicted_score_delta` proportional to the candidate's per-tensor footprint.
**Predicted score impact**: ~1-3% via better candidate-vs-budget ordering.
**Effort**: medium (~150 LOC).

### Op-routable #5 — MEDIUM — Rust runtime activation for 1+ submission inflate.py

**Producer**: 6 orphan Rust crates with mature lib.rs implementations.
**Consumer**: `submissions/<lane>/inflate.py` — currently 0 inflate.py paths invoke Rust; all the runtime-rs work is dormant at submission boundary.
**Predicted score impact**: indirect — Rust runtime enables stricter LOC budgets (Catalog #4 ≤100 LOC inflate.py limit) by externalizing decompression to the Rust binary; no direct score lowering, but unblocks substrate classes currently size-blocked.
**Effort**: large (council-grade); requires `inflate.sh` contract revision per CLAUDE.md exact-current-non-edit rule.

### Op-routable #6 — MEDIUM — joint_admm_coordinator co-optimization route

**Producer**: `tac.joint_admm_coordinator` + 3 sister proximal-step primitives.
**Consumer**: substrate trainers that produce (rate, seg, pose) triples should route through the joint-ADMM coordinator for principled rate-distortion-component trade-offs.
**Predicted score impact**: ~3-7% via Pareto-aware joint optimization vs uniform Lagrangian weights.
**Effort**: large (council-grade).

### Op-routable #7 — LOW — lapose_foveation_atoms wired into A1+lapose substrate trainer

**Producer**: `tac.lapose_foveation_atoms` (canonical analytical atoms).
**Consumer**: `experiments/train_substrate_a1_plus_lapose.py` — recipe exists but does NOT import lapose_foveation_atoms. The trainer hand-rolls lapose without consulting the canonical atom library.
**Predicted score impact**: ~1-2% via canonical atom geometry vs hand-rolled.
**Effort**: small (~50 LOC).

### Op-routable #8 — LOW — xray primitives `mdl_scorer_conditional` route to cost-band posterior

**Producer**: `tac.xray.mdl_scorer_conditional` (Tier C MDL substrate-class probe) — already used per Catalog #227 ranker, but the `score_lipschitz`, `predictive_coding_hierarchy`, `vq_codebook_coverage` siblings are all ORPHAN at the dispatch boundary.
**Consumer**: cost-band posterior reseeders (`harvest_modal_calls.py`).
**Predicted score impact**: ~1% via better cost-band priors.
**Effort**: medium.

---

## 6-hook wire-in declaration (this audit landing)

Per CLAUDE.md "Subagent coherence-by-default" mandatory wire-in:

1. **Sensitivity-map**: ACTIVE — audit catalogs ALL sensitivity-related modules + identifies HIGH-EV op-routable #4 to wire sensitivity-weighted priors into cathedral autopilot ranker (production consumption beyond path-enumeration).
2. **Pareto constraint**: ACTIVE — audit identifies HIGH-EV op-routable #6 to wire joint_admm_coordinator into substrate trainers for principled (rate, seg, pose) Pareto-aware co-optimization.
3. **Bit-allocator hook**: ACTIVE — audit identifies HIGH-EV op-routable #3 to consolidate per-substrate bit-allocator math into `tac.optimization.bit_allocator_end_to_end`. Council-grade per CLAUDE.md "Deterministic packet compiler" non-negotiable.
4. **Cathedral autopilot dispatch hook**: ACTIVE — audit identifies CRITICAL op-routable #1 to wire xray.wire_in bundle inventory into the planner context payload (37 primitives across 6 hooks become structurally visible to the autopilot ranker).
5. **Continual-learning posterior**: ACTIVE — audit identifies LOW op-routable #8 to route the orphan xray primitives (score_lipschitz / predictive_coding_hierarchy / vq_codebook_coverage) into cost-band posterior reseeders.
6. **Probe-disambiguator**: N/A with rationale — this is an audit + landing memo; no design tension requiring 2+ defensible interpretations. The orphan vs WIRED classification is empirically observable from the import graph.

---

## Apples-to-apples evidence axis labels

All status claims in this audit (`WIRED`, `WIRED-BUT-DORMANT`, `ORPHAN`,
`CODE-EXISTS-NEVER-RUN`) are sourced from **static import-graph analysis** via
`grep -rln "from <module>"` on Python source files; they are NOT empirical
runtime claims and carry **no axis label** because no exact eval was run.

The HIGH-EV / MEDIUM-EV / LOW-EV predicted-score-impact bands are **prior
predictions** (`[prediction]` axis) based on the operator's accumulated knowledge
of which primitives historically moved score and the recursive-review history.
Empirical confirmation requires the wire-in to land + a follow-up
`[contest-CUDA]` or `[contest-CPU]` measurement.

---

## Cross-references

- `feedback_consolidate_everything_into_meta_layer_or_canonical_helpers_standing_directive_20260515.md` — the operator's standing directive on consolidation (this audit IS a "consolidation discoverability pass")
- `feedback_grand_council_fields_medal_omnibus_20260515.md` — Council Decision 7 (Wave 2 actuator queue) + S1 HIGH (zero consumer-side integration)
- `feedback_meta_layer_adversarial_review_round_1_2_landed_20260515.md` — META layer landing memo
- `feedback_modal_call_id_ledger_canonical_landed_20260515.md` — exemplar producer/consumer wire-in
- `.omx/research/cross_stack_wire_in_audit_20260515.md` — sister WIRE-AND-INTEGRATE-ALL audit (this orphan audit extends it with score-lowering-prioritized recall)
- `feedback_unified_lagrangian_action_principle_GR_style_20260509.md` — the canonical 6-hook charter
- CLAUDE.md non-negotiables: "Subagent coherence-by-default" + "Meta-Lagrangian/Pareto solver" + "Frontier target" + "Bugs must be permanently fixed AND self-protected against"
- Catalog #247 reservation (this turn): `check_cathedral_autopilot_payload_includes_xray_wire_in_inventory` (planned strict gate; implementation deferred to op-routable #1 wire-in subagent)
