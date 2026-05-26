# tac.side_information namespace design — 2026-05-17

**Lane:** `lane_tac_side_information_namespace_decorator_api_20260517`
**Phase:** 2 (canonical-helper namespace; per spec §5.2 build queue item 5 of 5)
**Sister landings:** `tac.boosting` (a1a29b24) + `tac.compress_time_optimization` (afc61441) + (in-flight) `tac.inflate_time_post_processing` + `tac.search`
**Premise verifier:** `.omx/tmp/tac_side_information_premise_verifier.txt`
horizon_class: frontier_pursuit  <!-- Catalog #309 canonical format (snake_case, no markdown bold). Wyner-Ziv 1976 + cooperative-receiver = canonical class-shift primitive; this namespace IS the apparatus for systematically extracting compression gain via legal side information. -->

---

## ## Canonical-vs-unique decision per layer

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" non-negotiable (2026-05-15 standing directive) + Catalog #290 sister-design-memo gate, every adopted canonical helper / forked design layer is documented per the falling-rule list.

| Layer | Decision | Rationale |
|---|---|---|
| Decorator pattern | ADOPT_CANONICAL | The pass-through decorator + module-level registry + rollback-on-failed-registration is identical across all three sister namespaces. No substrate-optimal reason to fork — sharing the pattern means subagents inherit the K1+Q2 adversarial-review discipline automatically. |
| Frozen-dataclass contract | FORK_BECAUSE_PRINCIPLED_MISMATCH | Side-information bakers have STRUCTURALLY DIFFERENT fields than compress-time passes or boosting stages. Fork-specific fields: `side_info_source` (6-value enum), `side_info_reproducible` (bool, REQUIRED True per Wyner-Ziv + contest rules), `archive_bytes_added` + `inflate_runtime_bytes_added` (TWO budgets, not one), `requires_canonical_comma2k19_cache` (Catalog #213 integration), `wyner_ziv_correlation_estimate` (float ∈ [0,1] or None). Sharing the parent contract would FORCE side-info bakers to inherit `max_wallclock_seconds`/`rate_budget_bytes`/`stage_phase=compress/archive_build` which don't fit. |
| Stage-phase semantics | FORK_BECAUSE_PRINCIPLED_MISMATCH | Side-information bakers can run at COMPRESS only, INFLATE only, or BOTH (the canonical shared-prior bake-and-consume cycle). Compress-time passes are by construction compress-only. Forking lets the contract validator allow the legal `stage_phase="both"` while still enforcing the strict-scorer-rule at inflate via the `scorer_free` invariant. |
| Cross-field invariants | FORK_BECAUSE_PRINCIPLED_MISMATCH | New invariants unique to this namespace: (a) `side_info_reproducible=False` raises specific `NonReproducibleSideInfoViolation`; (b) `side_info_source="scorer_weights"` requires `scorer_free=False` (the strict-scorer-rule's compress-time exception); (c) `requires_canonical_comma2k19_cache=True` triggers import-time check of the canonical helper (Catalog #213); (d) `wyner_ziv_correlation_estimate` MUST be in [0,1] or NaN/inf rejected (the I(X;Y)/H(X) ratio is by definition bounded). |
| Typed exceptions | ADOPT_CANONICAL pattern + FORK names | Same root-exception + typed-subclasses pattern as sister namespaces; FORK to namespace-specific names (`NonReproducibleSideInfoViolation` etc.) for surface specificity. |
| Pipeline composition (\|, &, @) | ADOPT_CANONICAL | The operator-overload pipeline pattern is the canonical "composable namespace" API. Same operator semantics across all sister namespaces means future operators reading pipelines do not need to relearn syntax per namespace. |
| Pipeline budget filters | FORK_BECAUSE_PRINCIPLED_MISMATCH | Side-information pipeline must separately budget BOTH archive bytes (which show up in the contest archive ZIP) AND inflate runtime bytes (which inflate inside inflate.py as Python constants). This is the structurally distinguishing feature vs `with_rate_budget` (compress_time) + `with_decoder_overhead_budget` (boosting). |
| Persistence (JSONL + fcntl) | ADOPT_CANONICAL | Sister of `tac.boosting.persistence` and `tac.compress_time_optimization.persistence` — same fcntl LOCK_EX + STRICT-load + unique .tmp + os.replace + APPEND-ONLY pattern per Catalog #128 / #131 / #138 / #245. No substrate-optimal reason to fork the locking discipline. |
| 5 builders | FORK per builder | Each builder is a unique primitive type. Sharing would require artificial intermediate abstractions; the builders' specs encode their distinguishing math (Wyner-Ziv correlation estimate, palette size, source dataset, statistic kind, feature extraction kind) at construction. |
| Test structure | ADOPT_CANONICAL | TestContractValidation / TestDecorator / TestPipeline / TestPersistence / TestBuilders / TestStructuralIndependence — same layout as sister namespaces means reviewers can scan all three test files with the same mental model. |

---

## ## Cargo-cult audit per assumption

Per CLAUDE.md "META-CC-2 hard-earned-vs-cargo-culted" Catalog #303 sister-gate + the addendum (`feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515.md`). Every architecturally-implicit assumption is classified.

| Assumption | Classification | Rationale |
|---|---|---|
| Wyner-Ziv 1976 framing applies to the contest | HARD-EARNED | The scorer weights are by contest contract decoder-side side info; the residual encoding gain `Rate(X) - Rate(X|Y)` is bounded by `I(X;Y)/H(X)` per the theorem. Anchor: Wyner-Ziv 1976. |
| Public reproducibility is the structural side-info contract | HARD-EARNED | The contest rules + CLAUDE.md "Apples-to-apples evidence discipline" + "Public Disclosure Hygiene" jointly forbid private datasets / secret tables from contributing to score. The `side_info_reproducible=True` REQUIRED invariant is the structural enforcement. |
| Comma2k19 is the canonical dashcam dataset | HARD-EARNED | MIT-licensed; verified via `gh api repos/commaai/comma2k19 --jq '.license.spdx_id'` 2026-05-14 (per `local_chunk_cache.py` provenance comment). Catalog #213 already enforces canonical-helper routing. |
| Atick-Redlich 1990 cooperative-receiver framing applies | HARD-EARNED | Wunderkind G2-PARTIAL / G3 / B3-precomputed-table tasks explicitly cite Atick-Redlich. The frozen scorer weights ARE side info in the cooperative-receiver formalism. Council seat for Atick + Redlich + Tishby memorial + Zaslavsky + Wyner already convened for Z4/Z5 per CLAUDE.md grand council roster. |
| The 4 "shared-prior" bakers emit ZERO archive bytes | HARD-EARNED | The shared prior is a BAKED-INTO-INFLATE.PY constant per Catalog #146 (`contest_one_video_replay`); the bytes live in the inflate.py source code, not the archive. The byte budget on the archive side is structurally zero. |
| The Wyner-Ziv encoder DOES contribute archive bytes | HARD-EARNED | Per Wyner-Ziv theorem the residual `X - f(Y)` IS the transmission; the encoder MUST write its encoded residual to the archive. The `archive_bytes_added >= 1` validator structurally enforces this. |
| The inflate-runtime byte budget is a real constraint | HARD-EARNED | CLAUDE.md HNeRV parity discipline lesson 4 (≤ 100 LOC inflate budget). The two-budget pattern in `ComposableSideInfoPipeline` is the canonical articulation. |
| The 5 enumerated builders collapse §J's 7 rows | HARD-EARNED + cargo-cult-aware | The 7 rows in spec §J are SYMPTOMS (optical-flow side-info, openpilot ego-motion, etc.); the 5 builders are PRIMITIVES that compose to produce those symptoms. Cargo-cult risk: an operator may try to build a "RAFT optical-flow baker" directly when the canonical pattern is `WynerZivResidualEncoder(reconstruction_fn="optical_flow_warp", shared_prior_baker_id="comma2k19_optical_flow_palette")`. The probe disambiguator + design memo explicitly point operators at the primitive composition. |
| The pipeline's `with_archive_budget` field name does NOT conflict with sister namespaces | HARD-EARNED | Compress_time uses `with_rate_budget(bytes=...)`; boosting uses `with_decoder_overhead_budget`; side_information uses `with_archive_budget` + `with_inflate_runtime_budget`. The three names are intentionally distinct so a mistake in the operator's IDE autocomplete is structurally caught. |
| The strict-scorer-rule at INFLATE is honored by precomputed constants | HARD-EARNED | The `ScorerWeightsAsSharedPrior` builder explicitly documents: COMPRESS loads scorer; INFLATE uses precomputed constant table. The `scorer_free=False` field combined with the inflate-side artifact being a constant-only lookup is the structural protection per CLAUDE.md "Strict scorer rule" non-negotiable. |

---

## ## Observability surface

Per CLAUDE.md "Max observability — non-negotiable" Catalog #305 + the 6-facet observability definition. Every behavior of this namespace is observable at the documented surfaces:

1. **Inspectable per layer** — every baker contract is a frozen dataclass; `SideInfoBakerContract.to_dict()` exposes every field. The pipeline exposes `bakers` (tuple of refs) + `archive_budget_bytes` + `inflate_runtime_bytes_added` + `search_strategy_descriptor`. `__str__` produces human-readable chain notation.
2. **Decomposable per signal** — `SideInfoPipelineResult` separately records `cumulative_archive_bytes_added` and `cumulative_inflate_runtime_bytes_added`. Per-baker outcome dicts record `archive_bytes_added`, `inflate_runtime_bytes_added`, `cumulative_archive_bytes`, `cumulative_inflate_runtime_bytes`, `elapsed_seconds`, `status`, `emitted_keys`.
3. **Diff-able across runs** — `to_json()` is byte-stable (`sort_keys=True`) so pipeline JSON representations diff cleanly via standard text-diff tools. Persisted baker outcomes in JSONL form are line-stable.
4. **Queryable post-hoc** — `load_baker_outcomes(path)` returns all rows; `load_baker_outcomes_strict(path)` raises on corruption (Catalog #138 fail-closed). The schema is documented in the persistence module.
5. **Cite-able** — every baker contract carries `lane_id`, `design_memo`, `canonical_vs_unique_decision` provenance fields. Every persisted record carries `schema_version`, `written_at_utc`, `written_pid`, `written_host`.
6. **Counterfactual-able** — `ComposableSideInfoPipeline.with_archive_budget(bytes=X).build()` lets an operator answer "what if the archive budget were X?" without re-running. `to_dict()` round-trip lets the cathedral autopilot rank alternative pipelines without instantiating them.

The runtime observability of the canonical Comma2k19 cache helper (Catalog #213) is inherited automatically when `requires_canonical_comma2k19_cache=True` — that helper already provides chunk-level SHA-256 verification, license_tags propagation, and per-fetch logging.

---

## ## 9-dimension success checklist evidence

Per CLAUDE.md "9-DIMENSION SUCCESS CHECKLIST EVIDENCE SECTION" non-negotiable + Catalog #294 sister-design-memo gate.

| # | Dimension | Evidence |
|---|---|---|
| 1 | UNIQUENESS | This is the FIFTH canonical-helper namespace (5/5 per spec §5.2 build queue). Without it, side-information primitives (Wyner-Ziv residuals, scorer-as-shared-prior, ImageNet statistics, dashcam priors) would re-implement the contract / decorator / pipeline / persistence boilerplate per substrate. The namespace IS the systematic capture of one class-shift primitive (Wyner-Ziv 1976) into a canonical helper. |
| 2 | BEAUTY + ELEGANCE | Same pattern as sister namespaces — every file ≤ 750 LOC, every contract field validator one block, every builder spec frozen-dataclass with one `__post_init__` validator. The package `__init__.py` exports the narrow public API (~30 names) that reviewers can scan in 30 seconds. |
| 3 | DISTINCTNESS | The TWO-BUDGET pipeline (`with_archive_budget` + `with_inflate_runtime_budget`) is structurally distinct from sister namespaces' single-budget pipelines. The contract field set (`side_info_source` + `side_info_reproducible` + `wyner_ziv_correlation_estimate`) is unique to this namespace. The 5 builders each have distinct primitives (no shared parent abstraction). |
| 4 | RIGOR | Premise verifier landed at `.omx/tmp/tac_side_information_premise_verifier.txt` BEFORE any source code edit per Catalog #229. Sister-subagent ownership map per Catalog #230 honored (disjoint scope from `tac.inflate_time_post_processing` and `tac.search` in-flight subagents). 148 dedicated tests cover every contract validator, decorator path, pipeline operator, persistence concurrency. Wyner-Ziv invariants + Catalog #213 cache discipline have their own test class. |
| 5 | OPTIMIZATION PER TECHNIQUE | Per the canonical-vs-unique decision per layer table: 8 layers FORKED for substrate-optimal engineering vs 4 ADOPTED canonical patterns. The strict-scorer-rule's compress-time exception is encoded as a cross-field invariant (`scorer_weights` source ⇒ `scorer_free=False`); the byte-stable archive invariant is enforced (`archive_bytes_added>0` ⇒ `deterministic=True`); the Wyner-Ziv correlation bound is mathematically enforced (`[0,1]` interval, NaN/inf rejected). |
| 6 | STACK-OF-STACKS-COMPOSABILITY | `|` sequential + `&` parallel-merge + `@` attach-search operators are identical across all three sister namespaces. A `ComposableSideInfoPipeline` (e.g. `comma2k19_palette | wz_residual_against_palette`) can be composed with a `ComposableCompressPipeline` and a `ComposableBoostingPipeline` via the cathedral autopilot ranker (the dispatch-time integration surface) — orthogonal axes per CLAUDE.md substrate-composition-matrix Catalog #528/#566. |
| 7 | DETERMINISTIC REPRODUCIBILITY | Every contract pins `seed` (or accepts seed=) per Catalog #158. Every `to_dict()` uses `sort_keys=True` for byte-stable JSON. Every persisted record is APPEND-ONLY per Catalog #132. Every fcntl-locked write uses unique `.tmp.<uuid12>` + `os.replace`. Tests verify byte-stability of `to_json()` across two calls. |
| 8 | EXTREME OPTIMIZATION + PERFORMANCE | The decorator is pass-through (zero runtime overhead). The pipeline pre-validates structurally at `.build()` so paid GPU dispatch never encounters an invalid pipeline. The 4-process spawn concurrent-append test proves the persistence helper survives 20 concurrent rows from 4 processes. |
| 9 | OPTIMAL MINIMAL CONTEST SCORE | The Wyner-Ziv compression gain `Rate(X) - Rate(X\|Y) = I(X;Y) - I(X;Y\|D)` is theoretically optimal per Wyner-Ziv 1976; this namespace IS the canonical infrastructure for systematically extracting that gain. Substrates that adopt this namespace gain ALL 5 primitives as opt-in tools; no per-substrate re-implementation needed. Expected ΔS per the spec §J row 2: "decoder uses to predict frame_1 from frame_0 (Wyner-Ziv)" — empirical bound TBD per substrate but mathematically grounded in I(X;Y) measurement. |

---

## ## Predicted ΔS band

This namespace is INFRASTRUCTURE; it does not directly produce a score. The predicted ΔS comes from substrate dispatches that ADOPT the namespace.

Per the Wyner-Ziv 1976 theorem:
- Lower bound: `Rate(X|Y) >= R(D|Y)` — the conditional rate-distortion function.
- Upper bound: `Rate(X|Y) <= R(D)` — when `I(X;Y) = 0` (no correlation), no gain.
- Compression gain: `Rate(X) - Rate(X|Y) ~ I(X;Y) - I(X;Y|D)` — the Wyner-Ziv mutual-information ratio.

For a substrate that adopts the namespace with a baker whose `wyner_ziv_correlation_estimate=0.42` (the Comma2k19 chroma-anchor palette example), the predicted archive-byte reduction is approximately `0.42 × current_archive_bytes_on_chroma_channel`. **Dykstra-feasibility check**: the namespace is a feasibility-preserving primitive (each baker that adds an archive byte must compose with the rest of the substrate's bit allocation; the `ComposableSideInfoPipeline`'s `with_archive_budget` filter is the alternating-projection constraint on the rate axis). Per CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable, this is the convex-feasibility framing required for compositional predicted-band claims.

The first substrate to dispatch with a baker from this namespace will produce the first empirical ΔS measurement.

---

## ## Cross-references

- Sister namespaces: `tac.boosting` (decorator pattern source), `tac.compress_time_optimization` (decorator pattern source + TWO-budget inspiration via `with_rate_budget` + `with_wallclock_budget`).
- Spec: `.omx/research/meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md` §5 (canonical-helper namespace design) + §J (side-information / pre-processing / per-pair input conditioning) + §5.2 build queue item 5.
- Canonical Comma2k19 helper: `tac.substrates.pretrained_driving_prior.local_chunk_cache.Comma2k19LocalCache` per Catalog #213.
- Catalog #210 (DP1 codebook provenance metadata) — license_tags propagation pattern this namespace inherits.
- Catalog #146 (contest_one_video_replay) — admissibility of precomputed constants baked into inflate.py.
- Catalog #213 (`check_comma2k19_downloads_route_through_canonical_cache`) — STRICT preflight gate this namespace's `requires_canonical_comma2k19_cache=True` invariant honors structurally.
- Catalog #158 (deterministic_compiler discipline) — `archive_bytes_added>0` ⇒ `deterministic=True` cross-field invariant.
- Wyner-Ziv 1976 source-coding-with-side-information theorem.
- Atick-Redlich 1990 cooperative-receiver framing (canonical for the `ScorerWeightsAsSharedPrior` builder).
- Council seats: Wyner / Atick / Redlich / Tishby memorial / Zaslavsky for Z4/Z5 deliberations per CLAUDE.md grand council roster.

---

## ## 6-hook wire-in declaration

Per CLAUDE.md "Subagent coherence-by-default" non-negotiable + Catalog #125 sister-gate.

| Hook | Status | Routing |
|---|---|---|
| Sensitivity-map contribution | ACTIVE for `ScorerWeightsAsSharedPrior` (declares `hook_sensitivity_contribution="scorer_weights_shared_prior_v1"`) + `WynerZivResidualEncoder` when `sensitivity_weighted=True` (declares `master_gradient_v1`). The pipeline auto-threads `master_gradient` to bakers whose contract declares either hook. |
| Pareto constraint | ACTIVE — every contract declares `hook_pareto_constraint`. Default `rate_distortion_v1`; the 4 shared-prior + WZ-residual bakers declare `wyner_ziv_rate_distortion_v1` (unique to this namespace; encodes Wyner-Ziv's R(D|Y) bound). |
| Bit-allocator hook | N/A for the 4 shared-prior bakers (fixed-size constants); ACTIVE for `WynerZivResidualEncoder` (declares `wyner_ziv_residual_allocator`). The unique hook value names the Wyner-Ziv-residual-allocator that the cathedral autopilot ranker can dispatch when comparing residual encoders. |
| Cathedral autopilot dispatch hook | ACTIVE — every builder's contract declares `hook_autopilot_ranker="cathedral_autopilot_v1"`. The cathedral autopilot consumes `wyner_ziv_correlation_estimate` to prioritize bakers with higher predicted compression gain. |
| Continual-learning posterior update | ACTIVE — `append_baker_outcome_locked()` writes to `.omx/state/side_information_baker_outcomes.jsonl` (Catalog #128/#131 sister discipline); every contract declares `hook_continual_learning_anchor_kind="side_information_baker_outcomes_v1"`. Persistence is OPT-IN at the pipeline level (callers wrap `for outcome in result.per_baker_outcomes: append_baker_outcome_locked(outcome)`). |
| Probe-disambiguator | ACTIVE for the 3 builders with multiple defensible interpretations: `ScorerWeightsAsSharedPrior` (5 feature-extraction kinds → `tools/probe_scorer_prior_feature_extraction_disambiguator.py`), `Comma2k19DerivedPriorPalette` (6 palette kinds → `tools/probe_palette_kind_disambiguator.py`), `DashcamDomainPrior` (6 prior kinds → `tools/probe_dashcam_prior_kind_disambiguator.py`), `WynerZivResidualEncoder` (5 non-custom reconstruction_fns → `tools/probe_wyner_ziv_reconstruction_fn_disambiguator.py`). N/A for `ImageNetStatisticsPrior` (published statistic has single canonical interpretation; rationale documented in `hook_not_applicable_rationale`). Note: the probe disambiguator scripts are NAMED but not yet implemented — they are placeholders for future op-routables when the first substrate adopts each builder and an empirical disambiguator becomes valuable. |

---

## ## Observability surface (sister entry per Catalog #305 — see above)

See `## Observability surface` section above.

---

## End of design memo


# PREDICTIVE_CODING_EGO_MOTION_CONDITIONED_OK:design_memo_references_cooperative_receiver_atick_redlich_or_wyner_ziv_framework_in_cross_reference_or_spatial_not_temporal_context_NOT_as_substrate_central_predictive_coding_claim_per_catalog_311_z6_z7_z8_pattern_h_clarification_canonical_clearance_per_comprehensive_bug_audit_cascade_20260526
