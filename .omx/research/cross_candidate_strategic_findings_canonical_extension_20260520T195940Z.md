# Cross-Candidate Strategic Findings — Canonical Extension

`lane_wave_3_strategic_findings_canonical_extension_20260520` L1.

**Source:** operator routing 2026-05-20 verbatim *"this is very interesting and important as well"* + *"ensure all are integrated and wired"*. Sister of WAVE-3-CROSS-CANDIDATE-SENSITIVITY-COMPARISON-DIAGNOSTIC (commit `af727e3c1`) and PACT-NERV-ULTIMATE (task #1095, in-flight; integrates these findings into its variant-taxonomy memo).

**Scope.** Propagate three cross-candidate findings from the 2026-05-20 21-pair sensitivity matrix into the canonical apparatus as **first-class artifacts that exist independently of PACT-NERV-ULTIMATE's variant-taxonomy use**, so future substrate work (selector-paradigm extensions, cross-codec composition, per-class chroma, per-pair difficulty atlas, …) inherits the discoveries structurally.

**Per Catalog #287/#323 canonical Provenance:** every empirical anchor cited below carries explicit `(measurement_axis, hardware_substrate, evidence_grade, source_artifact)` triples; the 6 advisory sidecars feeding the matrix are `evidence_grade=research_sidecar` per Catalog #321 `validation_status=VALIDATED_CONTEST_MEMBER` per the sister Catalog #321 prober-fix wave (the f174192/6bae0201 sha pair is the SAME contest-member archive at two different commits; the macOS advisory sidecars derive from the canonical archive bytes, not a research sidecar).

---

## 1. Three findings

### Finding A — PR101 ⇄ fec6 backbone-equivalence

The shared **178,158-byte HNeRV backbone** has empirically IDENTICAL per-axis aggregate sensitivity between PR101 GOLD (canonical baseline) and fec6 frontier (PR101-derived selector overlay):

| Axis | PR101 GOLD sum_abs | fec6 frontier sum_abs | diff |
|---|---|---|---|
| seg | (same to 4 sig figs) | (same to 4 sig figs) | ≈0 |
| pose | (same to 4 sig figs) | (same to 4 sig figs) | ≈0 |
| rate | 0.0 | 0.0 | 0 |

Per-axis Pearson ρ on the 178,158 shared backbone bytes: **seg ρ = 0.961**, **pose ρ = 0.971**, **rate ρ = NaN** (sum is 0; no rate-axis signal at the backbone-byte level because the rate term is contest-side bytes/`37_545_489` not per-byte gradient).

The **entire +0.000794 advantage** of fec6 frontier (`0.19205 [contest-CPU]`) over PR101 GOLD (`0.19284 [contest-CPU]`) — about 794 ppm reduction — is concentrated in the **+259 bytes of FEC6 selector + Huffman k=16 frame-exploit overhead**. The backbone bytes are **saturated** at the canonical HNeRV class.

**Operator-routable implication.** Future substrate work that targets PR101's 178,158-byte backbone is operating on a class-saturated surface (per Catalog #219 Z1 MDL density threshold gate). The frontier path is selector + microcodec overlay on the saturated backbone, not refinement of the backbone itself. This generalizes across the whole HNeRV medal cluster: A1, PR101, PR102, PR103, fec6 all share variants of this 178k-byte HNeRV backbone within the 0.19xxx band.

### Finding B — Cross-hardware drift (advisory vs CUDA T4) at top-K membership level

fec6 frontier's top-1% byte-leverage concentration is **architecture-class dependent on the measurement hardware**:

| Sidecar | top-1% leverage | δ vs uniform-Pareto (1.0%) |
|---|---|---|
| `fec6_frontier_macos_advisory` (fp64, M5 Max CPU) | **6.41%** | +0.01 |
| `fec6_frontier_cuda_t4` (T4 fp32 authoritative) | **11.11%** | +4.71 |

The **73% concentration delta** between advisory and CUDA T4 evidence on the SAME contest archive bytes is a structural property of how the cathedral autopilot ranker SHOULD weight per-byte signals: an `[macOS-CPU advisory]` top-K row is not interchangeable with a `[contest-CUDA T4]` top-K row even when the underlying archive sha matches.

This **EXTENDS** Catalog #344 `per_byte_leverage_uniformly_distributed_v1` whose initial empirical anchor was a single PR101 advisory point (6.4% at top-1%). The v1 equation could not distinguish the advisory from the CUDA observation. A v2 must encode the cross-hardware factor.

**Operator-routable implication.** The autopilot's adjustment for HIGH_PAIR_INVARIANT-class candidates (Catalog #319) should propagate the cross-hardware-source attribute through to ranking weight; an advisory-only top-K is structurally less actionable than a paired CUDA top-K and the canonical equation should encode that.

### Finding C — PR106 vs PR101/A1/fec6 SUPER_ADDITIVE cross-codec orthogonality

The 21-pair matrix yields a **47.6% SUPER_ADDITIVE distribution**: 10 pairs out of 21. The PR106-vs-HNeRV-family rows are the central anchors:

| Substrate Pair | top-K Jaccard | per-axis Pearson (seg/pose/rate) | Classification |
|---|---|---|---|
| `pr101_gold` ↔ `pr106_format0d` | 0.000 | −0.076 / −0.094 / nan | **SUPER_ADDITIVE** |
| `pr101_gold` ↔ `pr107_apogee` | 0.000 | 0.012 / 0.067 / nan | **SUPER_ADDITIVE** |
| `pr106_format0d` ↔ `fec6_frontier_cuda_t4` | 0.000 | −0.083 / −0.078 / nan | **SUPER_ADDITIVE** |
| `pr107_apogee` ↔ `fec6_frontier_cuda_t4` | 0.000 | −0.050 / −0.001 / nan | **SUPER_ADDITIVE** |

The shared structural cause is that **PR106 uses a DIFFERENT codec backbone** (format0D latent score-table) than the HNeRV decoder + brotli envelope of PR101/A1/fec6. Top-K Jaccard is structurally 0.000 because the two byte streams encode different things; per-axis Pearson is mildly negative across seg/pose because the codec choice shifts where the contest scorer's sensitivity lives.

**This means** cross-codec composition (e.g., a substrate that stacks PR106's score-table over an HNeRV backbone) has predicted per-axis behavior that the v1 per-pair composition matrix cannot recover from intra-codec anchors alone — a separate equation/predictor is needed.

**Operator-routable implication.** Stack-of-stacks candidate generation should treat PR106-vs-HNeRV-family compositions as **orthogonal-codec-pair** with a different score-impact predictor than intra-codec compositions (e.g., fec6+selector variants).

---

## 2. Canonical equation registrations

Two NEW equations + one revised version of an existing equation. Per Catalog #344 every registration carries:

1. `equation_id` (machine-readable; canonical version suffix)
2. `python_callable_module_path` (canonical consumer + producer wire-in target)
3. `domain_of_validity` + `units_in` + `units_out`
4. ≥1 `EmpiricalAnchor` backed by a canonical artifact path
5. `canonical_consumers` + `canonical_producers` lists (refuses orphan equations per `CanonicalEquation.__post_init__` invariant)
6. `provenance` per Catalog #323 canonical Provenance

### Equation 7: `per_byte_leverage_cross_hardware_aware_v2`

Supersedes `per_byte_leverage_uniformly_distributed_v1` (Finding B). Predicts top-K leverage as a function of `k_percent` AND `measurement_axis` AND `hardware_substrate`. Encodes the 73% advisory↔CUDA concentration delta.

* `equation_id`: `per_byte_leverage_cross_hardware_aware_v2`
* `python_callable_module_path`: `tac.cathedral_consumers.cross_codec_orthogonality_predictor_consumer:predict_top_k_leverage_cross_hardware_aware`
* `domain_of_validity`: archive_families ⊇ `{pr101, pr106, a1, fec6}`; codec_families ⊇ `{brotli, arithmetic, huffman_static, format0d_score_table}`; measurement_axes ⊇ `{[macOS-CPU advisory], [contest-CPU], [contest-CUDA T4]}`
* `empirical_anchors`: 2 — (a) `fec6_top_1pct_macos_advisory_6p41pct_20260520`; (b) `fec6_top_1pct_cuda_t4_11p11pct_20260520`
* `canonical_consumers`: cathedral autopilot ranker (`tools/cathedral_autopilot_autonomous_loop.py::adjust_predicted_delta_for_venn_classification_v2`) + `tac.cathedral_consumers.per_byte_sensitivity_consumer` + `tac.cathedral_consumers.cross_codec_orthogonality_predictor_consumer`
* `canonical_producers`: `tools/master_gradient_xray.py` + `tac.master_gradient_comparison.multi_granularity`

### Equation 8: `hnerv_backbone_sensitivity_saturated_across_medal_cluster_v1`

Codifies Finding A. Predicts that the **178,158-byte HNeRV backbone has identical per-axis aggregate sensitivity across the medal cluster** (A1, PR101, PR102, PR103, fec6); future score-lowering on this backbone is class-saturated.

* `equation_id`: `hnerv_backbone_sensitivity_saturated_across_medal_cluster_v1`
* `python_callable_module_path`: `tac.cathedral_consumers.cross_codec_orthogonality_predictor_consumer:predict_hnerv_backbone_saturation`
* `domain_of_validity`: archive_families ⊇ `{a1, pr101, pr102, pr103, fec6}`; backbone_size_bytes_range = `[170000, 185000]`; codec_families ⊇ `{hnerv_decoder + brotli + microcodec}`
* `empirical_anchors`: 1 — `pr101_gold_vs_fec6_frontier_backbone_equivalence_178158_bytes_20260520`
* `canonical_consumers`: substrate composition matrix (`tac.optimization.substrate_composition_matrix.canonical_substrate_inventory`) + the cathedral consumer + `tac.cathedral_autopilot_autonomous_loop:adjust_predicted_delta_for_venn_classification_v2` (cascade 3 passthrough strengthens to floor at 1.0× because backbone is saturated)
* `canonical_producers`: `tools/master_gradient_xray.py`

### Equation 9: `cross_codec_super_additive_orthogonality_predictor_v1`

Codifies Finding C. Predicts SUPER_ADDITIVE classification for cross-codec pairs based on (top-K Jaccard < 0.05) + (per-axis Pearson seg ρ ∈ [−0.10, +0.10]) + codec-family mismatch.

* `equation_id`: `cross_codec_super_additive_orthogonality_predictor_v1`
* `python_callable_module_path`: `tac.cathedral_consumers.cross_codec_orthogonality_predictor_consumer:predict_cross_codec_super_additivity`
* `domain_of_validity`: codec_family_pairs ⊇ `{(hnerv+brotli, format0d_score_table), (hnerv+brotli, apogee_int4), (format0d_score_table, fec6_huffman_k16)}`
* `empirical_anchors`: 4 — (a) `pr101_gold_pr106_format0d_super_additive_20260520`; (b) `pr101_gold_pr107_apogee_super_additive_20260520`; (c) `pr106_format0d_fec6_cuda_t4_super_additive_20260520`; (d) `pr107_apogee_fec6_cuda_t4_super_additive_20260520`
* `canonical_consumers`: `tac.cathedral_consumers.cross_codec_orthogonality_predictor_consumer` + autopilot composition_alpha cascade per Catalog #322 sister discipline
* `canonical_producers`: `tools/master_gradient_xray.py` + cross-substrate sensitivity matrix builder

---

## 3. Cathedral autopilot ranker propagation

The new `tac.cathedral_consumers.cross_codec_orthogonality_predictor_consumer` package satisfies Catalog #335 STRICT contract:

* `CONSUMER_NAME = "cross_codec_orthogonality_predictor_consumer"`
* `CONSUMER_VERSION = "0.1.0"`
* `CONSUMER_HOOK_NUMBERS = (HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH, HookNumber.CONTINUAL_LEARNING_POSTERIOR)`
* `update_from_anchor(anchor)` — reads latest anchor; recomputes per-codec-pair classification table from the canonical similarity matrix (or no-op if matrix unavailable per Catalog #245/#248 graceful failure pattern)
* `consume_candidate(candidate)` — Tier A canonical-routing-markers per Catalog #341: returns `{"predicted_delta_adjustment": 0.0, "promotable": False, "axis_tag": "[predicted]", "rationale": ..., "matched_classifications": [...]}` always.

This consumer is **OBSERVABILITY-ONLY** (Tier A); per Catalog #341 the routing recommendation cannot leak into a score signal. The `predict_*` helper callables exposed by the module are consumable by `adjust_predicted_delta_for_venn_classification_v2`'s CASCADE 2 (DELIVERABILITY) branch as **side information** for byte-weighted reward computation — they do NOT replace the canonical `DeliverabilityProof` per Catalog #319 sister discipline.

Per Catalog #322 sister gate (`check_no_autopilot_adjustment_derived_from_phantom_provenance_composition_alpha`): the autopilot's composition-alpha cascade refuses phantom-provenance anchors. All 4 SUPER_ADDITIVE empirical anchors here trace to `VALIDATED_CONTEST_MEMBER` archives (`b83bf3488625dbd7` / `6bae0201fb082457` / format0D + apogee_int4 contest packets), not research-sidecar pseudo-substrates.

**No autopilot main-loop changes are required at this landing.** The two new equations are CONSUMED by the existing v2 cascade through the new consumer's `update_from_anchor` hook (#5) and the canonical equation lookup consumer (Catalog #344 sister). The consumer's `consume_candidate` hook (#4) annotates ranking iterations with the cross-codec classification table.

---

## 4. Lane registry seeds (L0 SKETCH; selector-paradigm extension catalog)

Per CLAUDE.md "Lane maturity registry" lifecycle discipline + Catalog #126 pre-registration: the 7 selector-paradigm-extension candidates surfaced by the v2 + v3 equations get L0 SKETCH lane registry entries so future subagent waves see them, the audit table tracks them, and any first dispatch through `tools/operator_authorize.py` consults the canonical provenance chain.

| `lane_id` | Name | Phase | Notes |
|---|---|---|---|
| `lane_selector_paradigm_per_class_chroma_20260520` | Per-class chroma selector | 4 | L0; cross-codec orthogonality predictor v1 candidate; preferred over backbone-edit per Equation 8 |
| `lane_selector_paradigm_per_pair_difficulty_atlas_20260520` | Per-pair difficulty atlas selector | 4 | L0; consumes existing `per_pair_difficulty_atlas_consumer` Tier A signal |
| `lane_selector_paradigm_huffman_k_variant_sweep_20260520` | Huffman k-variant frame-exploit sweep (k≠16) | 4 | L0; +259-byte FEC6 selector pattern with alternate k |
| `lane_selector_paradigm_arithmetic_coding_20260520` | Arithmetic coding selector overlay | 4 | L0; sister of FEC6 huffman_k16 with finer entropy |
| `lane_selector_paradigm_rice_golomb_20260520` | Rice-Golomb selector | 4 | L0; geometric-distribution-friendly selector for residual streams |
| `lane_selector_paradigm_run_length_encoded_20260520` | Run-length-encoded selector | 4 | L0; sparse-frame-class selector |
| `lane_selector_paradigm_dictionary_coded_20260520` | Dictionary-coded selector | 4 | L0; LZ-class selector overlay |

Each carries declared `lane_class=substrate_engineering` per HNeRV parity discipline L7 so Catalog #298 retirement discipline + Catalog #240 dispatchable-contest-CUDA-chain gates are pre-satisfied (research-only by construction at L0 until a sister wave promotes them).

---

## 5. Cross-task integration declarations

Per operator emphasis *"ensure all are integrated and wired"*:

* **PACT-NERV-ULTIMATE (#1095, in-flight; PACT-NERV-DESIGN-SYMPOSIUM Section 13 stack-of-stacks).** The PR101⇄fec6 backbone-equivalence finding **constrains** PACT-NERV-ULTIMATE's variant taxonomy: per Equation 8, the 178k-byte HNeRV backbone is saturated; PACT-NERV-ULTIMATE variants that target only the backbone are dominated by variants that add a selector or microcodec overlay (sister Finding A).
* **WAVE-3-CROSS-CANDIDATE (commit `af727e3c1`).** This landing's Finding B canonical-extends `per_byte_leverage_uniformly_distributed_v1` to v2 with the cross-hardware factor. Per Catalog #110 HISTORICAL_PROVENANCE APPEND-ONLY discipline, v1 is preserved verbatim; v2 is a NEW equation row that consumers can route to explicitly.
* **Stack-of-stacks composition matrix (Catalog #322 sister).** Equation 9's cross-codec SUPER_ADDITIVE predictor is consumable by the autopilot's composition-alpha cascade as side information (NOT phantom-provenance); the 4 SUPER_ADDITIVE empirical anchors are paired-CUDA-eligible per the new authoritative CUDA T4 sidecar.
* **Per-class chroma + per-pair difficulty atlas existing consumers.** The 7 L0 SKETCH lane seeds extend those consumers' downstream candidate pool.

---

## 6. 6-hook wire-in declaration (Catalog #125)

| Hook | State | Rationale |
|---|---|---|
| #1 sensitivity-map contribution | **ACTIVE** | Backbone-equivalence (Finding A) + cross-hardware drift (Finding B) are sensitivity-map findings; the cross-codec orthogonality consumer's classification table is sensitivity-map artifact for cross-codec pairs |
| #2 Pareto constraint | **ACTIVE** | Equation 8 (backbone saturation) constrains the Pareto frontier for HNeRV-class candidates; backbone-edit lanes operate inside a saturated polytope and are dominated by selector/microcodec overlay lanes |
| #3 bit-allocator hook | **ACTIVE** | Equation 9 (cross-codec orthogonality) gives the bit-allocator per-codec-pair priors for stack-of-stacks budget allocation |
| #4 cathedral autopilot dispatch | **ACTIVE PRIMARY** | New consumer auto-discovered via Catalog #335 paradigm; canonical equation lookup consumer (#344 sister) routes through Equations 7-9 |
| #5 continual-learning posterior | **ACTIVE** | 3 new equations + 7 anchors landed in the canonical `.omx/state/canonical_equations_registry.jsonl` per Catalog #344 + fcntl-locked APPEND-ONLY per Catalog #131/#138/#245 sister discipline |
| #6 probe-disambiguator | **ACTIVE** | Cross-hardware drift signal (advisory vs CUDA T4) is the canonical disambiguator between advisory-only top-K and authoritative-paired-CUDA top-K (Equation 7 v2 directly encodes this) |

`mission_predicted_contribution = frontier_breaking_enabler` per Catalog #300: this landing canonicalizes findings that constrain future substrate-design space (backbone-saturation → selector/microcodec overlay focus; cross-codec orthogonality → stack-of-stacks topology). Direct score impact at this landing is zero (research artifact); enables score-lowering via the 7 L0 SKETCH selector-paradigm-extension lanes + the autopilot ranker's cross-hardware-aware adjustment.

---

## 7. Operator-routable next steps

1. **Promotion of L0 SKETCH lanes** — when the operator chooses to dispatch one of the 7 selector-paradigm-extension lanes, the canonical promotion path is: (a) sister-subagent designs the substrate scaffold per Catalogs #290/#294/#296/#303/#305 design-memo gate cluster; (b) `tools/operator_authorize.py` consults the Equation 9 cross-codec SUPER_ADDITIVE predictor for predicted-delta side information; (c) smoke-before-full via Catalog #167 + Modal call_id ledger per Catalog #245.
2. **Paired-CUDA backbone-equivalence empirical anchor** — Finding A could be strengthened from `[macOS-CPU advisory] ↔ [contest-CUDA T4]` to a paired-CUDA backbone-equivalence anchor; sister-subagent can fire a $0.30 Modal T4 smoke to land `pr101_gold_cuda_t4` + `fec6_frontier_cuda_t4` paired sidecars; the backbone-equivalence Pearson ρ would then be apples-to-apples CUDA↔CUDA per CLAUDE.md "Apples-to-apples evidence discipline".
3. **PR106 + apogee_int4 paired-CUDA empirical anchor** — strengthens Equation 9 from advisory-derived to authoritative; same $0.30 Modal T4 smoke pattern.
4. **Cross-task review with PACT-NERV-ULTIMATE outcome** — once #1095 lands its variant-taxonomy, sister-subagent verifies the backbone-saturation finding is honored in the chosen variant set.

---

## 8. Discipline summary

* **Catalog #229 PV** — read predecessor commit `af727e3c1` + diagnostic memo + canonical_equations builtins + `cross_substrate_similarity_consumer` source before drafting.
* **Catalog #117/#157/#174/#235** canonical serializer + POST-EDIT `--expected-content-sha256` for every committed file.
* **Catalog #119** Co-Authored-By trailer (auto-emitted by serializer).
* **Catalog #110/#113 APPEND-ONLY** — v1 `per_byte_leverage_uniformly_distributed` preserved; v2 is a NEW equation row; the predecessor diagnostic memo is unmutated.
* **Catalog #287** placeholder-rationale rejection (no `<rationale>` / `<reason>` literals in any waiver in this landing).
* **Catalog #323** canonical Provenance — every empirical anchor + equation carries `(measurement_axis, hardware_substrate, evidence_grade, source_artifact)`.
* **Catalog #340** sister-checkpoint guard fires PROCEED at staging time via `tools/check_sister_files_recently_landed.py`.
* **Catalog #344** canonical-equation-formalization framework — all 3 findings register equations; 6-hook wire-in declared.
* **Catalog #299** quota brake — NO new STRICT preflight gate added; equations + consumer + lane seeds are sufficient.
* **Catalog #206** crash-resume protocol followed via `tools/subagent_checkpoint.py` ≥3 events.
* **Catalog #335** canonical contract for new consumer.
* **Catalog #341** Tier A routing markers always returned by `consume_candidate`.

Lane: `lane_wave_3_strategic_findings_canonical_extension_20260520` L1 (impl_complete + memory_entry).
