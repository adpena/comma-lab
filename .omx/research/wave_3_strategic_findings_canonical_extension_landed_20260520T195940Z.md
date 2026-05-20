# WAVE-3-STRATEGIC-FINDINGS-CANONICAL-EXTENSION — Landing Memo

`lane_wave_3_strategic_findings_canonical_extension_20260520` L1 (impl_complete + memory_entry).

**Source:** operator routing 2026-05-20 verbatim *"this is very interesting and important as well"* + *"ensure all are integrated and wired"*. Sister of WAVE-3-CROSS-CANDIDATE-SENSITIVITY-COMPARISON-DIAGNOSTIC (commit `af727e3c1`) and PACT-NERV-ULTIMATE (task #1095; in-flight; this task disjoint — produces standalone canonical artifacts).

## 9-dimension success checklist evidence

| Dim | Evidence |
|---|---|
| 1. UNIQUENESS | Cross-codec orthogonality + cross-hardware leverage + backbone saturation are NEW canonical equation surfaces. No sister consumer covers Equation 7 (cross-hardware) or Equation 8 (backbone saturation) prior to this landing. |
| 2. BEAUTY + ELEGANCE | 1 consumer (~330 LOC) + 3 predictor callables + 17 tests; reviewable in 30 seconds per PR101 model. |
| 3. DISTINCTNESS | Distinct from `cross_substrate_similarity_consumer` (predecessor; observability of matrix entries) — this consumer PREDICTS from the matrix via 3 named equations. |
| 4. RIGOR | Catalog #229 PV (read predecessor + builtins + consumer source) + Catalog #287 placeholder-rejection + Catalog #323 canonical Provenance on every anchor + Catalog #344 canonical equation registration. |
| 5. OPTIMIZATION PER TECHNIQUE | Each equation is independently calibratable + recalibratable from the canonical anchors registry via `auto_recalibrate_from_continual_learning_posterior`. |
| 6. STACK-OF-STACKS COMPOSABILITY | Equation 9 directly predicts cross-codec composition_alpha for stack-of-stacks via Catalog #322 sister discipline. |
| 7. DETERMINISTIC REPRODUCIBILITY | All 3 predictor callables are pure functions of `(k_percent, axis, substrate_label, codec_family_pair, top_k_jaccard, per_axis_pearson_seg)`. Empirical anchors are seed-pinned via UTC timestamp + archive sha prefix. |
| 8. EXTREME OPTIMIZATION + PERFORMANCE | 17 tests run in 0.09s. Equation lookup is O(1) hash + O(4) anchor scan; cathedral consumer side-effect-free. |
| 9. OPTIMAL MINIMAL CONTEST SCORE | Indirect — backbone-saturation finding (Equation 8) constrains future substrate work to selector/microcodec overlay; the 7 L0 SKETCH selector-paradigm-extension lanes are the score-lowering candidate pool surfaced by this landing. |

## Cargo-cult audit per assumption

| Assumption | Classification | Rationale |
|---|---|---|
| Cross-hardware drift on the SAME archive bytes is a structural ranking signal | HARD-EARNED | Empirically anchored at 6.41% [advisory] vs 11.11% [CUDA T4] on the SAME fec6 archive sha `6bae0201` per predecessor commit `af727e3c1` |
| HNeRV backbone is saturated across the medal cluster | HARD-EARNED | Per-axis Pearson seg ρ=0.961, pose ρ=0.971 on 178158 shared bytes between PR101 GOLD and fec6 frontier; +259-byte selector accounts for entire +0.000794 advantage |
| 4 SUPER_ADDITIVE cross-codec pairs predict ALL cross-codec compositions | CARGO-CULTED-PENDING-EVIDENCE | We have 4 anchors from advisory-derived matrix; production decision requires paired-CUDA backbone-equivalence anchor; confidence floor 0.60 for observation-derived predictions reflects this honestly |
| The `_state_dir()` walker correctly finds `.omx/state/` from any consumer location | HARD-EARNED | Sister `cross_substrate_similarity_consumer` (commit `af727e3c1`) uses the same pattern; tested in 17/17 consumer tests |

## Observability surface

| Facet | State |
|---|---|
| Inspectable per layer | YES — each predictor callable returns a structured Mapping with per-field rationale |
| Decomposable per signal | YES — top-K leverage / backbone saturation / cross-codec orthogonality are 3 independent signals |
| Diff-able across runs | YES — empirical anchors live in `.omx/state/canonical_equations_registry.jsonl` APPEND-ONLY; future anchor refits diffable via `query_equations_by_*` |
| Queryable post-hoc | YES — `tools/list_canonical_equations.py --json` per Catalog #344 + canonical query helpers in `tac.canonical_equations.registry` |
| Cite-able | YES — every empirical anchor carries `source_artifact` path + `measurement_axis` + `hardware_substrate` per Catalog #323 |
| Counterfactual-able | YES — callers can supply `(top_k_jaccard, per_axis_pearson_seg)` to `predict_cross_codec_super_additivity` to ask "what if pair X classifies as Y?" |

## 6-hook wire-in declaration (Catalog #125)

| Hook | State | Rationale |
|---|---|---|
| #1 sensitivity-map contribution | ACTIVE | Cross-hardware top-K leverage + cross-codec orthogonality are sensitivity-map artifacts |
| #2 Pareto constraint | ACTIVE | Equation 8 backbone saturation constrains the Pareto frontier for HNeRV-class candidates |
| #3 bit-allocator hook | ACTIVE | Equation 9 cross-codec orthogonality gives per-codec-pair priors for stack-of-stacks budget allocation |
| #4 cathedral autopilot dispatch | ACTIVE PRIMARY | New consumer auto-discovered via Catalog #335 paradigm |
| #5 continual-learning posterior | ACTIVE | 3 new equations + 7 anchors landed in canonical registry per Catalog #344 + fcntl-locked APPEND-ONLY per Catalog #131/#138/#245 |
| #6 probe-disambiguator | ACTIVE | Cross-hardware drift is the canonical disambiguator between advisory-only top-K and CUDA-paired top-K |

`mission_predicted_contribution = frontier_breaking_enabler` per Catalog #300.

## Canonical-vs-unique decision per layer

| Layer | Decision | Rationale |
|---|---|---|
| Cathedral consumer Protocol | ADOPT_CANONICAL_BECAUSE_SERVES | `tac.cathedral.consumer_contract.HookNumber` + Tier A markers; same pattern as 40+ sister consumers |
| Canonical equation registration | ADOPT_CANONICAL_BECAUSE_SERVES | `tac.canonical_equations.register_canonical_equation` per Catalog #344 |
| fcntl-locked JSONL APPEND-ONLY persistence | ADOPT_CANONICAL_BECAUSE_SERVES | Sister of Catalog #131/#138/#245 — no substrate-specific reason to fork |
| Empirical anchor schema | ADOPT_CANONICAL_BECAUSE_SERVES | `EmpiricalAnchor` per Catalog #344 + canonical Provenance per Catalog #323 |
| Predictor callable shapes | FORK_BECAUSE_PRINCIPLED_MISMATCH | Each predictor returns a domain-specific Mapping (saturation vs leverage vs SUPER_ADDITIVE classification); a generic shape would lose per-equation semantics |
| Lane registry seeds | ADOPT_CANONICAL_BECAUSE_SERVES | `tools/lane_maturity.py add-lane` per Catalog #90 |

## Cross-task integration declarations

* **PACT-NERV-ULTIMATE (#1095, in-flight)** — Disjoint scope. This task produces standalone canonical artifacts; #1095 integrates findings into its variant-taxonomy memo independently.
* **Predecessor `af727e3c1` (sensitivity comparison diagnostic)** — Sister extends Catalog #344 `per_byte_leverage_uniformly_distributed_v1` to a NEW v2 equation with cross-hardware factor. APPEND-ONLY per Catalog #110.
* **Sister tasks in queue** (#1096 + #1098) — Disjoint by lane_id design; no overlap.

## Discipline summary

* Catalog #110/#113 APPEND-ONLY (predecessor memo unmutated; v1 equation preserved)
* Catalog #117/#157/#174 canonical serializer + POST-EDIT --expected-content-sha256
* Catalog #119 Co-Authored-By trailer (auto-emitted)
* Catalog #125 6-hook wire-in declaration
* Catalog #206 crash-resume (3+ checkpoint events)
* Catalog #229 PV (read predecessor commit + memo + builtins + consumer source)
* Catalog #287 placeholder-rationale rejection (NO `<rationale>` / `<reason>` literals)
* Catalog #299 quota brake (NO new STRICT preflight gate)
* Catalog #313 probe-outcomes ledger row appended
* Catalog #323 canonical Provenance on every anchor
* Catalog #335 cathedral consumer canonical contract satisfied (17/17 tests pass)
* Catalog #340 sister-checkpoint guard PROCEED at staging time
* Catalog #341 Tier A canonical-routing markers in every `consume_candidate`
* Catalog #344 canonical equations registry (3 equations registered)

## Sister-collision verdicts

* **#1095 PACT-NERV-ULTIMATE** — DISJOINT (different deliverables; both consume the same predecessor findings; no file collision)
* **#1096 + #1098 queued** — DISJOINT by lane_id design
* **`tools/check_sister_files_recently_landed.py`** — PROCEED at staging time (verified)

## Operator-routable next steps

1. Review the strategic findings memo at `.omx/research/cross_candidate_strategic_findings_canonical_extension_20260520T195940Z.md`.
2. Optional sister-subagent: promote one or more of the 7 L0 SKETCH selector-paradigm lanes to L1 SCAFFOLD per the design-memo discipline cluster.
3. Optional sister-subagent: fire a $0.30 Modal T4 paired-CUDA smoke for backbone-equivalence (PR101 GOLD + fec6 frontier) to strengthen Equation 8 to apples-to-apples CUDA↔CUDA per CLAUDE.md "Apples-to-apples evidence discipline".
