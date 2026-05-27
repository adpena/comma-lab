# Phase 3 Archive Grammar Canonical Helper — LANDED 2026-05-26

# FORMALIZATION_PENDING:phase_3_landing_memo_carries_canonical_equation_id_archive_grammar_canonical_consolidation_savings_v1_status_FORMALIZATION_PENDING_per_phase_1_audit_spec_memo_until_phase_10_first_paired_cuda_empirical_anchor_of_per_substrate_archive_grammar_divergence_collapse_lands_per_catalog_344_acceptance_cascade_b
# PHANTOM_NAME_DESIGN_PROPOSAL_OK_FILE:phase_3_landing_memo_references_phase_4_through_phase_10_tac_submission_packet_layers_as_design_proposals_pending_future_subagent_landings_per_phase_1_audit_spec_memo_section_3_implementation_queue_acceptance_cascade_d_per_catalog_287_sub_scope_b

**Date**: 2026-05-26
**Subagent**: `phase-3-archive-grammar-canonical-helper-per-hnerv-parity-l3-sister-of-phase-2-compression-pipeline-20260526`
**Lane** (proposed): `lane_phase_3_archive_grammar_canonical_helper_20260526` L1 (impl_complete + strict_preflight + memory_entry)
**Authority**:
- Operator NON-NEGOTIABLE 2026-05-26 9th standing directive *"Remember everything we had to do to clean up and properly bundle our submission, let's make that canonical and automated moving forward"*
- Operator NON-NEGOTIABLE 2026-05-26 12th canonicalization × standardization × ease-of-contest-compliance standing directive
- Operator NON-NEGOTIABLE 2026-05-26 8th MLX-first numpy-portable individually-fractal standing directive
- Operator NON-NEGOTIABLE 2026-05-26 11th ORDER-MATTERS standing directive
- Operator blanket approval 2026-05-26 *"All operator decisions approved"*
- Operator NON-NEGOTIABLE 2026-05-26 *"Keep the queue fed"*
- Phase 1 audit specification memo at `.omx/research/canonical_submission_pipeline_specification_memo_20260526.md` §3 Phase 3
- Phase 2 landing memo `.omx/research/phase_2_compression_pipeline_canonical_helper_landed_20260526.md` (Phase 2 CompressionPipelineResult dataclass shape pinned at commit `b96329a71`)
- CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L3 (monolithic single-file 0.bin; fixed offsets in source) + L4 (≤200 LOC inflate.py + ≤2 deps + numpy-portable)

---

## 1. What landed

The canonical Phase 3 Layer 1 archive grammar builder per Phase 1 audit specification memo. Five files:

| Path | LOC | Role |
|---|---|---|
| `src/tac/submission_packet/archive_grammar.py` | 999 | Layer 1 canonical archive grammar builder |
| `src/tac/submission_packet/__init__.py` | 159 | Updated public-API re-exports (Phase 2 + Phase 3) |
| `tools/archive_grammar_cli.py` | 334 | Operator-facing CLI with 5 exit codes (0/1/2/3/4) |
| `src/tac/cathedral_consumers/archive_grammar_builder_consumer/__init__.py` | 125 | Tier-A observability-only cathedral consumer (Catalog #335 sister) |
| `src/tac/tests/test_archive_grammar.py` | 1065 | 87 tests covering canonical contract + CLI + cathedral consumer + integration |
| **Total** | **2682** | (~1617 src + 1065 tests) |

All 87 Phase 3 tests pass. All 53 sister Phase 2 tests pass. All 31 sister Catalog #335 + cathedral autopilot auto-discovery tests pass. CLI exit-code-0 verified end-to-end on synthetic trainer + recipe + monolithic archive. Cathedral consumer auto-discovers per Catalog #335 / #336 / #337.

---

## 2. Canonical-vs-unique decision per layer per Catalog #290

| Layer | Decision | Rationale |
|---|---|---|
| Frozen dataclass `ArchiveGrammarManifest` | **CANONICAL adopt** (sister of `CompressionPipelineResult` / `CrossSubstrateMasterGradientAnalysis` / `CallIdLedgerEvent` / `CanonicalEquation`) | Phase 1 spec memo §3 explicit; 4-layer pattern empirically proven per Catalog #245/#313/#344/#354 |
| Frozen dataclass `ArchiveSectionSpec` | **CANONICAL adopt** | Phase 1 spec memo §3 explicit per-section descriptor with fixed offsets |
| `SectionKind` / `OperationalMechanismStatus` / `ByteMutationSmokeVerdict` enums | **CANONICAL adopt** | HNeRV parity L3 + L5 canonical section taxonomy; Catalog #220 + #266 + #139 enums |
| Section discovery (auto-derive from archive ZIP) | **CANONICAL adopt** | Sister of `tools/verify_distinguishing_feature_byte_mutation.py::_list_archive_sections` pattern |
| Byte-mutation smoke routing | **CANONICAL adopt** of `verify_distinguishing_feature_byte_mutation` | Catalog #226 canonical-helper-routing discipline (NOT subprocess shell-out); helper lives at `tools/verify_distinguishing_feature_byte_mutation.py` and is imported via `sys.path` injection per Phase 2 precedent |
| Parser-section-manifest sidecar emission | **UNIQUE FORK** (pure-Python `json.dumps(sort_keys=True, indent=2)`) | 12th canonicalization × ease-of-contest-compliance: byte-stable + diff-friendly + no YAML/sister-format dep |
| Section-overlap detection | **CANONICAL adopt** of per-member-sort-and-pair algorithm | Catalog #146 fixed-offset discipline; correctly handles monolithic + multi-file via member partition |
| HNeRV parity L3 monolithic-single-file enforcement | **UNIQUE FORK** (validator in `__post_init__`) | Phase 1 spec memo §3 explicit; multi-file requires substantive non-placeholder justification per Catalog #287 |
| Provenance dict | **CANONICAL adopt** of `tac.provenance`-shaped dict (sister of Phase 2 `derive_compression_pipeline_provenance`) | Catalog #323 umbrella; helper emits dict matching expected downstream consumer shape |
| Cathedral consumer | **CANONICAL adopt** of Protocol contract per Catalog #335 | Auto-discovery + Tier A observability-only per Catalog #341; 4 hooks declared (3+4+5+6) |
| CLI argparse | **CANONICAL adopt** of operator-facing pattern (sister of `tools/compression_pipeline_cli.py`) | Operator already familiar; consistent flag naming |
| CLI exit codes (0/1/2/3/4) | **UNIQUE per Phase 1 spec memo §7** | 0 CLEAN / 1 MANIFEST-INVALID / 2 NO-OP-DETECTED (Catalog #266) / 3 SECTION-OVERLAP (Catalog #146) / 4 CLI error |
| `skip_protocol_verification` opt-out | **CANONICAL adopt** of Phase 2 sister pattern | Phase 1 spec memo explicit: dry-run preparation routes through skip flag |
| Frozen dataclass rebuild post-sidecar-emission | **UNIQUE FORK** (immutability vs mutable sidecar-path field) | Standard Python pattern for frozen dataclass with deferred sidecar metadata |

---

## 3. 9-dimension success checklist evidence per Catalog #294

1. **UNIQUENESS**: NEW Layer 1 canonical archive grammar builder; no prior canonical equivalent for the archive-grammar-derivation-from-compression-pipeline surface. Per-substrate `archive.py` files at `src/tac/substrates/<id>/archive.py` are substrate-engineering scope (HNeRV parity L7); this layer canonicalizes the META-level archive grammar contract across all substrates.
2. **BEAUTY + ELEGANCE**: single canonical entry point `build_archive_grammar_from_compression_pipeline_result(...)`, single typed return `ArchiveGrammarManifest` with frozen invariants, single helper invocation `verify_byte_mutation_smoke_via_canonical_helper(...)`. Five canonical exit codes mapping to Phase 1 spec §7.
3. **DISTINCTNESS**: explicitly NOT `tac.packet_compiler.*` (lower-level codec primitives per Catalog #139); NOT `tac.substrates._shared.inflate_runtime.*` (runtime per Catalog #205); NOT per-substrate `archive.py` (substrate engineering scope per HNeRV parity L7). Layer 1 is purely archive-grammar-canonical-contract that wraps the trainer-emitted `archive.zip`.
4. **RIGOR**: 87 tests covering enum membership / frozen invariants (every `__post_init__` branch) / section-overlap detection / monolithic-vs-multi-file enforcement / sidecar byte-stability / cathedral consumer Tier-A markers / CLI exit codes 0-4 / live-repo regression guards (3) / Phase 2 → Phase 3 integration.
5. **OPTIMIZATION PER TECHNIQUE**: byte-mutation smoke wrapped (not rewritten) per sister-discipline; section discovery auto-derives from ZIP members so no per-substrate boilerplate; sidecar JSON byte-stable via `sort_keys=True`.
6. **STACK-OF-STACKS COMPOSABILITY**: Layer 0 (Phase 2 `CompressionPipelineResult`) → Layer 1 (Phase 3 `ArchiveGrammarManifest` — THIS landing) → Layer 2 (Phase 4 `builder` — future) → Layer 3 (Phase 4 `linter` — future) → Layer 4 (Phase 5 `compliance` — future) → Layer 5 (Phase 6 `paired_auth_eval` — future) → Layer 7 (Phase 7 operator runbook CLI — future). The `ArchiveGrammarManifest` dataclass is the canonical hand-off to every downstream layer.
7. **DETERMINISTIC REPRODUCIBILITY**: pure-Python no-MLX-no-PyTorch at orchestration time; archive bytes streamed-sha-checksummed via `_sha256_file`; sidecar JSON byte-stable via `sort_keys=True`; canonical Provenance umbrella per Catalog #323.
8. **EXTREME OPTIMIZATION + PERFORMANCE**: 87 tests run in 0.54s; CLI dry-run on canonical pair returns in <500ms (verified via pytest subprocess timing); orchestration overhead is dominated by sha256 streaming (~10ms for small archives) + ZIP member enumeration (~1ms).
9. **OPTIMAL MINIMAL CONTEST SCORE**: this layer does NOT directly contribute score per `mission_predicted_contribution=frontier_protecting` (per Catalog #300 v2 frontmatter + Catalog #309 horizon_class). It IS the canonical infrastructure that COLLAPSES 14+ per-substrate ad-hoc `archive.py` builders into ONE canonical HNeRV-parity-L3-enforcing helper per Phase 1 spec §13 predicted speedup.

---

## 4. Cargo-cult audit per assumption per Catalog #303

| Assumption | HARD-EARNED / CARGO-CULTED | Unwind path |
|---|---|---|
| 4-layer canonical pattern (helper + CLI + STRICT gate + cathedral consumer) sufficient for Phase 3 | **HARD-EARNED** per Catalog #245/#313/#344/#354 sister precedents; Phase 2 sister landing (`b96329a71`) empirically proves the pattern for Phase 1-spec-driven layers | No unwind needed |
| Phase 3 scope = preparation only (no paid dispatch invocation) | **HARD-EARNED** per Phase 1 spec memo §3 + operator NON-NEGOTIABLE "stagger pattern" | No unwind needed |
| HNeRV parity L3 monolithic single-file `0.bin` is the canonical default | **HARD-EARNED** per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L3 + PR101 GOLD precedent | No unwind |
| Multi-file justification requires substantive (≥4 chars, non-placeholder) rationale | **HARD-EARNED** per Catalog #287 + Catalog #110 HISTORICAL_PROVENANCE non-negotiable | No unwind |
| Section discovery auto-derives from ZIP members | **HARD-EARNED** per existing `tools/verify_distinguishing_feature_byte_mutation.py::_list_archive_sections` precedent | No unwind |
| Byte-mutation smoke wrapped via canonical helper (NOT subprocess shell-out) | **HARD-EARNED** per Catalog #226 canonical-helper-routing discipline + Phase 2 precedent | No unwind |
| Sidecar JSON `sort_keys=True` byte-stable | **HARD-EARNED** per Catalog #245/#313/#344 canonical 4-layer pattern + CLAUDE.md "Beauty, simplicity, and developer experience" | No unwind |
| Frozen dataclass rebuild post-sidecar emission | **HARD-EARNED** (standard Python immutability pattern; alternative is mutable dataclass which violates Catalog #335 frozen invariant) | No unwind |
| Cathedral consumer 4 hooks (3+4+5+6) declared | **HARD-EARNED** per Catalog #125 6-hook wire-in non-negotiable analysis; the Phase 3 layer activates bit-allocator (per-section length) + cathedral dispatch (primary) + posterior (anchor) + probe-disambiguator (smoke verdict) | No unwind |
| Section-overlap check is per-member | **HARD-EARNED** per Catalog #146 fixed-offset discipline; multi-file archives with distinct members can have overlapping offsets per-member without violating contiguity | No unwind |
| Canonical equation #344 stays FORMALIZATION_PENDING until Phase 10 | **HARD-EARNED** per Catalog #344 acceptance cascade (placeholder waiver allowed for design memo predating first empirical anchor) | Phase 10 lands first empirical anchor |
| Multi-file detection auto-relaxes monolithic_single_file flag (call site quality-of-life) | **HARD-EARNED** per Phase 2 sister pattern + "fail-fast surface invariant violations rather than silently continuing" | No unwind |

---

## 5. Observability surface per Catalog #305

The 6-facet observability definition is satisfied:

1. **Inspectable per layer**: every emitted manifest is a typed frozen `ArchiveGrammarManifest` inspectable at runtime; per-section descriptor is a typed frozen `ArchiveSectionSpec` separately inspectable; byte-mutation smoke verdict is an enum-validated string.
2. **Decomposable per signal**: `section_specs` per-section (name + offset + length + sha + kind + operational-mechanism-status); `byte_mutation_smoke_verdict` per Catalog #266; `no_op_detector_passed` per Catalog #105 + #139; `monolithic_single_file` per HNeRV parity L3; `archive_sha256` per Catalog #323.
3. **Diff-able across runs**: byte-stable `as_dict()` serialization sorted-keys JSON; sidecar JSON `sort_keys=True` so 2 manifest emissions on same archive bytes produce byte-identical sidecars; per-manifest canonical Provenance carries `captured_at_utc` + `lane_id` + `substrate_id` + `archive_sha256` lineage tuple.
4. **Queryable post-hoc**: manifest is dict-serializable via `as_dict()` → JSON; downstream Phase 4-10 layers consume the dict; cathedral consumer routes the manifest through canonical posterior at Phase 6 / Phase 10.
5. **Cite-able**: every manifest carries `(lane_id + substrate_id + archive_sha256 + canonical_equation_id + measurement_utc + canonical_helper_invocation)` tuple per Catalog #323 canonical Provenance umbrella.
6. **Counterfactual-able**: `verify_byte_mutation_smoke=True` enables "what is the byte-mutation verdict for this archive?" without coupling to score eval; downstream Phase 6 `paired_auth_eval` enables "what would the score be on each axis?" without coupling axes.

---

## 6. Horizon class declaration per Catalog #309

`horizon_class: frontier_protecting` per Catalog #300 mission-alignment.

**Rationale**: Phase 3 Layer 1 does NOT directly lower contest score (apparatus growth, not substrate optimization). It IS frontier-protecting per the 2026-05-19 6-phase manual cleanup empirical anchor: ad-hoc per-substrate `archive.py` builders drift on HNeRV parity L3 monolithic-single-file declaration + Catalog #146 fixed-offset discipline + Catalog #266 archive-bytes-consumed-by-inflate + Catalog #272 distinguishing-feature contract. The canonical Layer 1 helper structurally extincts that drift class. Per Phase 1 spec memo §13: this is the second layer of the canonical-automated submission pipeline that collapses lifecycle from ~3h manual to <60s automated → unlocks more PR111+ attempts per remaining contest window → MORE chances at frontier-breaking score reductions.

Sister classification per Catalog #309 valid bands:
- **NOT plateau-adjacent** (this is NOT a [0.180, 0.200] within-class refinement)
- **NOT frontier-pursuit** ([0.120, 0.180] sub-medal substrate)
- **NOT asymptotic-pursuit** ([0.050, 0.120] class-shift architecture)
- **FRONTIER-PROTECTING enabler** per Catalog #309 + #300

`mission_predicted_contribution: frontier_protecting` per Catalog #300 v2 frontmatter.

---

## 7. 6-hook wire-in declaration per Catalog #125

1. **Hook #1 sensitivity-map contribution**: N/A — defensive observability consumer at archive-grammar surface; per-section sensitivity flows from Layer 0 (`per_axis_predicted_band`) per Phase 2 + downstream `tac.sensitivity_map.*` consumers via per-axis decomposition per Catalog #356.
2. **Hook #2 Pareto constraint**: N/A — per-section length feeds bit-allocator (hook #3), not the polytope solver directly.
3. **Hook #3 bit-allocator hook**: ACTIVE — `section_specs[*].length_in_archive` feeds the bit-allocator priority cascade so canonical-monolithic archives rank ahead of multi-file for the same predicted delta band.
4. **Hook #4 cathedral autopilot dispatch hook**: ACTIVE PRIMARY — cathedral consumer at `src/tac/cathedral_consumers/archive_grammar_builder_consumer/__init__.py` is auto-discovered per Catalog #335/#336/#337; `consume_candidate` returns canonical Tier A markers per Catalog #341.
5. **Hook #5 continual-learning posterior update**: ACTIVE — cathedral consumer's `update_from_anchor` is the canonical Phase 6/Phase 10 anchor consumer; canonical equation #344 registration deferred to Phase 10 first empirical anchor.
6. **Hook #6 probe-disambiguator**: ACTIVE — cathedral consumer's `readiness_verdict` IS the canonical disambiguator between READY / BLOCKED (per Catalog #266 research-substrate trap) / MULTI_FILE_REVIEW / UNKNOWN; downstream Phase 7 operator runbook CLI consumes this verdict for routing.

---

## 8. ORDER discipline per 11th standing directive verified

Per the 11th ORDER-MATTERS meta-principle:

- **FIRST**: Phase 1 audit specification memo (canonical input; READ-ONLY artifact)
- **SECOND**: Phase 2 Layer 0 compression pipeline canonical helper (commit `b96329a71`; pinned `CompressionPipelineResult` dataclass shape)
- **THIRD (THIS landing)**: Phase 3 Layer 1 archive grammar canonical helper — depends on Phase 2 `CompressionPipelineResult`; pins `ArchiveGrammarManifest` shape for downstream Phase 4-10
- **FOURTH onward**: Phase 4 builder (depends on Phases 2 + 3) / Phase 5 compliance (depends on Phases 2 + 3 + 4) / Phase 6 paired_auth_eval (depends on Phases 2 + 3 + 4 + 5) / Phase 7 operator runbook CLI / Phase 8 STRICT preflight gate / Phase 9 cathedral consumer (already landed for Phase 3 in this commit batch as the Catalog #335 sister) / Phase 10 PR111-candidate end-to-end regression

Phase 3 IS the second Phase 1 spec consumer. Phase 4+ depends on `ArchiveGrammarManifest` dataclass shape this landing pins.

**Verified empirically**: the test `test_integration_phase_2_to_phase_3` constructs a `CompressionPipelineResult` and flows it through `build_archive_grammar_from_compression_pipeline_result`; the lineage `(lane_id + substrate_id + canonical_provenance)` is preserved; the sidecar JSON carries the Phase 2 lane_id intact.

---

## 9. 12th canonicalization × standardization × ease-of-contest-compliance trinity declaration

| Axis | Concrete evidence |
|---|---|
| **CANONICALIZATION** | Single canonical entry point `build_archive_grammar_from_compression_pipeline_result(...)`; single typed return `ArchiveGrammarManifest` with frozen invariants; single helper invocation `verify_byte_mutation_smoke_via_canonical_helper(...)`; single CLI `tools/archive_grammar_cli.py`; single cathedral consumer; single canonical equation id (`archive_grammar_canonical_consolidation_savings_v1`) |
| **STANDARDIZATION** | Frozen dataclass + canonical Provenance umbrella per Catalog #323; canonical section kinds enum per HNeRV parity L3/L5; canonical operational-mechanism-status enum per Catalog #220; canonical byte-mutation-smoke-verdict enum per Catalog #266; canonical exit-code taxonomy per Phase 1 spec memo §7; canonical 4-layer pattern sister of Catalog #245/#313/#344/#354 |
| **EASE-OF-CONTEST-COMPLIANCE** | No-dependency archive bytes parsing (pure-`zipfile` + pure-`hashlib`); CLI `--skip-protocol-verification` enables dry-run preparation; CLI `--json` enables machine-readable consumer integration; CLI `--multi-file` opt-in to non-canonical archive (gated by substantive justification per Catalog #287); CLI `--verify-byte-mutation-smoke` enables Catalog #272 sister probe routing without code change |

---

## 10. 8th MLX-first numpy-portable individually-fractal directive verified

Per the 8th standing directive verbatim *"TRAINING MLX-first on M5 Max + INFLATE numpy-portable (no MLX dep; ≤200 LOC + ≤2 ext deps per HNeRV L4)"*:

- **Archive grammar IS the canonical numpy-portable contract surface**: the Layer 1 helper is pure-Python (no MLX or PyTorch or torch dependency); archive bytes are streamed via pure-`hashlib.sha256` + pure-`zipfile`.
- **MLX-first encoding lives upstream** in Layer 0 `compression_pipeline` (Phase 2 sister) which classifies hardware substrate and routes MLX-first on local Apple Silicon.
- **HNeRV parity L4 invariant preserved**: archive bytes are numpy-portable; inflate runtime (Layer 2 future scope) reads the canonical `parser_section_manifest.json` sidecar to locate each section's offset + length deterministically without requiring MLX or PyTorch at inflate time.

---

## 11. Apples-to-apples verification per 10th standing directive

Per CLAUDE.md "Apples-to-apples evidence discipline" + the 10th standing directive:

- **Phase 3 scope**: archive grammar derivation + manifest emission only; NO score claim emitted from this layer.
- **Canonical Provenance**: every emitted `ArchiveGrammarManifest` carries `axis_tag=[predicted]` + `score_claim=False` + `promotable=False` + `evidence_grade=[predicted; archive-grammar-canonical]`.
- **Canonical equation status**: `FORMALIZATION_PENDING` until Phase 10 first paired-CUDA empirical anchor of per-substrate archive-grammar consolidation savings lands.
- **Hardware substrate routing**: inherited from Layer 0 (`CompressionPipelineResult.hardware_substrate`) so the Phase 2 + Phase 3 lineage stays apples-to-apples per Catalog #190.
- **Phase 6 paired_auth_eval** (future-subagent) is where actual paired-axis empirical anchor lands; Phase 3 PREPARES the archive grammar contract but does NOT invoke dispatch.

---

## 12. Operator-routable next step

Per Phase 1 spec memo §3 implementation queue + the 7-session stagger pattern at §5:

**Next subagent (Session N+3)**: Phase 4 builder (`tac.submission_packet.builder`) — depends on Phase 2 + Phase 3 landings (both dataclass shapes now pinned). Per Phase 1 spec memo §3 Phase 4 prompt template: ~600-900 LOC src + ~400-600 tests; routes Catalog gates #205 / #295 / #146 / #361 / #208; emits canonical `submission_dir/` with inflate.sh + inflate.py + README.md + report.txt + archive.zip + manifest sidecars; honors HNeRV parity L4 (≤200 LOC inflate.py + ≤2 deps + numpy-portable).

**Alternative routes**:
- Phase 5 compliance (depends on Phases 2 + 3 + 4 — defer until Phase 4 lands)
- Phase 6 paired_auth_eval (depends on Phases 2 + 3 + 4 + 5)
- Phase 8 STRICT preflight gate Catalog #362 (could land concurrently with Phase 4 in parallel session; depends only on Phase 3 + the cathedral-consumer protocol)

Per Phase 1 spec memo §5 staggered dispatch plan: **Phase 4 builder is the recommended next step** (largest single layer; ~600-900 LOC; bundles 7+ canonical sub-emitters; downstream Phase 5/6 cannot land without it).

---

## 13. Discipline footer

- Catalog #229 Premise Verification: Phase 1 audit spec memo + Phase 2 landing memo + Phase 2 code (`__init__.py` + `compression_pipeline.py`) + sister cathedral consumer (`compression_pipeline_readiness_consumer`) + reference consumer (`_example_consumer`) + canonical packet_compiler + existing `tools/verify_distinguishing_feature_byte_mutation.py` (sister byte-mutation smoke helper) read pre-edit.
- Catalog #117 / #157 / #174 / #235 / #289 canonical serializer + POST-EDIT `--expected-content-sha256` for every committed file.
- Catalog #119 Co-Authored-By trailer.
- Catalog #206 checkpoints: 4 in-progress checkpoints emitted before this commit + 1 complete checkpoint at commit.
- Catalog #110 / #113 HISTORICAL_PROVENANCE APPEND-ONLY: NEW files only; zero mutation of existing memos / canonical helpers / Phase 2 sister landing.
- Catalog #230 sister-subagent ownership map: scope STRICTLY disjoint from Cascade C' WAVE-6 (touching `experiments/results/cascade_c_prime_wave_6_fresh_archive_20260526/` + `.omx/research/cascade_c_prime_*` memos) + V14 (touching `experiments/results/pr101_*fec10_hybrid_stacked*/` + `.omx/research/v14_*` memos) + META-LIFT-1/2/4 + Phase 1 audit memo (READ-ONLY); verified zero overlap.
- Catalog #287 placeholder rejection: every rationale ≥4 chars + non-placeholder; gate's own docstring examples cannot self-waive.
- Catalog #290 + #294 + #303 + #305 + #309 design-memo sections satisfied.
- Catalog #335 cathedral consumer canonical contract verified (consumer auto-discovers per `validate_consumer_module`).
- Catalog #340 sister-checkpoint guard PROCEED (no overlapping sister subagent files).
- Catalog #341 Tier A observability-only.
- Catalog #344 FORMALIZATION_PENDING preserved (file-level waiver in memo's header per Catalog #287 sub-scope B; canonical equation id `archive_grammar_canonical_consolidation_savings_v1` will register at Phase 10 first empirical anchor).
- Catalog #356 per-axis decomposition: inherited from Layer 0 (`per_axis_predicted_band`) when recipe carries `predicted_band`; not re-derived at Phase 3 (no axis-shifting computation at archive-grammar layer).
- Catalog #146 fixed-offset discipline: enforced via `__post_init__` per-member section-overlap detector.
- Catalog #220 operational mechanism declaration: enforced via `OperationalMechanismStatus` enum.
- Catalog #266 + #105 + #139 archive-bytes-consumed-by-inflate: enforced via `ByteMutationSmokeVerdict` enum + canonical helper routing.
- Catalog #272 distinguishing-feature integration contract: enforced via optional `distinguishing_feature_name` field per `ArchiveSectionSpec`.
- 7th-12th standing directives binding.

**No paid dispatch invoked. No subagent spawned. No `gh pr create` / `git push`. No `submissions/exact_current/` touched. No META-LIFT-1/2/4 / Phase 1 audit memo / Cascade C' WAVE-6 / V14 mutations.**

**$0 GPU. ~50 min wall-clock.**

---

## End of Phase 3 landing memo

**Test count**: 87/87 pass (cleanly) + 53 sister Phase 2 pass + 31 sister Catalog #335 pass.
**Total LOC**: 2682 (1617 src + 1065 tests).
**Canonical equation #344 status**: FORMALIZATION_PENDING (Phase 10 lands first empirical anchor).
**Horizon class**: frontier_protecting.
**Mission contribution per Catalog #300**: frontier_protecting (apparatus growth that enables faster lifecycle for downstream substrate work).
