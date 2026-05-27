# Phase 2 Compression Pipeline Canonical Helper — LANDED 2026-05-26

# PHANTOM_NAME_DESIGN_PROPOSAL_OK_FILE:phase_2_landing_memo_references_phase_3_through_phase_10_tac_submission_packet_layers_design_proposals_pending_future_subagent_landings_per_phase_1_audit_spec_memo_section_3_implementation_queue_acceptance_cascade_d_per_catalog_287_sub_scope_b

**Date**: 2026-05-26
**Subagent**: `phase-2-compression-pipeline-canonical-submission-pipeline-first-production-code-per-phase-1-audit-spec-memo-20260526`
**Lane** (proposed): `lane_phase_2_compression_pipeline_canonical_helper_20260526` L1 (impl_complete + strict_preflight + memory_entry)
**Authority**:
- Operator NON-NEGOTIABLE 2026-05-26 9th standing directive *"Remember everything we had to do to clean up and properly bundle our submission, let's make that canonical and automated moving forward"*
- Operator NON-NEGOTIABLE 2026-05-26 amendment *"Remember contest compliance and bundling full compression script and all and everything"*
- Operator NON-NEGOTIABLE 2026-05-26 12th canonicalization × standardization × ease-of-contest-compliance standing directive
- Operator NON-NEGOTIABLE 2026-05-26 8th MLX-first numpy-portable individually-fractal standing directive
- Operator NON-NEGOTIABLE 2026-05-26 11th ORDER-MATTERS standing directive
- Operator blanket approval 2026-05-26 *"All operator approved"*
- Phase 1 audit specification memo at `.omx/research/canonical_submission_pipeline_specification_memo_20260526.md`

---

## 1. What landed

The canonical Phase 2 Layer 0 encoder pipeline orchestrator per Phase 1 audit specification memo. Five files:

| Path | LOC | Role |
|---|---|---|
| `src/tac/submission_packet/__init__.py` | 102 | Public-API re-exports |
| `src/tac/submission_packet/compression_pipeline.py` | 945 | Layer 0 canonical orchestrator |
| `tools/compression_pipeline_cli.py` | 282 | Operator-facing CLI with 6 exit codes |
| `src/tac/cathedral_consumers/compression_pipeline_readiness_consumer/__init__.py` | 105 | Tier-A observability-only cathedral consumer (Catalog #335 sister) |
| `src/tac/tests/test_compression_pipeline.py` | 916 | 53 tests covering canonical contract + CLI + cathedral consumer |
| **Total** | **2350** | (~1434 src + 916 tests) |

All 53 tests pass cleanly. CLI exit-code-0 verified end-to-end on synthetic trainer + recipe pair. Cathedral consumer auto-discovers per Catalog #335 / #336 / #337.

---

## 2. Canonical-vs-unique decision per layer

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" + Catalog #290:

| Layer | Decision | Rationale |
|---|---|---|
| Frozen dataclass `CompressionPipelineResult` | **CANONICAL adopt** (sister of `CrossSubstrateMasterGradientAnalysis` / `CallIdLedgerEvent` / `CanonicalEquation`) | Phase 1 spec memo §2 explicitly canonical; 4-layer pattern empirically proven per Catalog #245/#313/#344/#354 |
| `PerAxisPredictedBand` frozen dataclass | **CANONICAL adopt** (sister of `PredictedBandWithValidation`) | Phase 1 spec memo §13 + Catalog #324 + #356 align on per-axis decomposition |
| Per-recipe band extraction | **UNIQUE FORK** (pure-text regex; no YAML lib dep) | 12th canonicalization × ease-of-contest-compliance: helper stays dependency-free; recipe format is operator-controlled YAML with canonical sub-fields |
| Hardware classification | **CANONICAL adopt** of `_VALID_HARDWARE_SUBSTRATE_TOKENS` set (sister of `trainer_skeleton.detect_hardware_substrate` returned values) | Catalog #190 + #192 enforce canonical tokens; no false precision |
| Catalog #270 umbrella verification | **CANONICAL adopt** of `verify_dispatch_protocol_complete` | Catalog #226 canonical-helper-routing discipline (NOT subprocess shell-out); helper lives at `tools/canonical_dispatch_optimization_protocol.py` and is imported via `sys.path` injection because `@dataclass` requires module to be in `sys.modules` |
| Provenance dict | **CANONICAL adopt** of `tac.provenance` shape | Catalog #323 umbrella; helper emits dict matching expected downstream consumer shape |
| Cathedral consumer | **CANONICAL adopt** of Protocol contract per Catalog #335 | Auto-discovery + Tier A observability-only per Catalog #341 |
| CLI argparse | **CANONICAL adopt** of operator-facing pattern (sister of `tools/operator_authorize.py`, `tools/refresh_canonical_frontier.py`) | Operator already familiar; consistent flag naming |
| CLI exit codes | **UNIQUE per Phase 1 spec memo §7** (0/1/2/3/4/5) | Per-layer routing requires per-layer exit (Phase 1 spec § enumerated). Codes 3+4 reserved for Phase 3/Phase 4 future scope |
| `skip_protocol_verification` opt-out | **UNIQUE FORK** | Phase 1 spec memo explicit: dry-run preparation needs to PRE-flight without full Tier 1+2+3 verification; canonical pattern is opt-in `--skip-...` flag |

---

## 3. 9-dimension success checklist evidence per Catalog #294

1. **UNIQUENESS**: NEW Layer 0 canonical orchestrator; no prior canonical equivalent for the encoder-pipeline-preparation surface.
2. **BEAUTY + ELEGANCE**: single canonical entry point `build_compression_pipeline(...)`, single typed return `CompressionPipelineResult` with frozen invariants, single helper invocation `verify_compression_pipeline_protocol_complete(...)`. Six canonical exit codes mapping to Phase 1 spec §7.
3. **DISTINCTNESS**: explicitly NOT `tools/operator_authorize.py` (which covers a single dispatch); NOT `tools/operator_briefing.py` (situational awareness); NOT `experiments/modal_train_lane.py` (low-level Modal wrapper). Layer 0 is purely encoder-pipeline-PREPARATION.
4. **RIGOR**: 53 tests, 12 frozen-invariant rejections, 4 corner cases (placeholder rationale, schema-version mismatch, invalid hardware token, invalid equation status), live-repo regression guard via `test_verify_compression_pipeline_protocol_real_repo_returns_tuple` (skip-safe).
5. **OPTIMIZATION PER TECHNIQUE**: Catalog #270 wrapped (not rewritten) per sister-discipline; per-axis decomposition optional (None when recipe omits); MLX-first auto-detected from hardware substrate class.
6. **STACK-OF-STACKS COMPOSABILITY**: Layer 0 → Layer 1 (Phase 3) → Layer 2 (Phase 4 builder) ↔ Layer 6 (Phase 4 attribution) → Layer 3 + Layer 4 (Phase 4-5 lint + compliance) → Layer 5 (Phase 6 paired_auth_eval) → Layer 7 (Phase 7 operator runbook CLI). The `CompressionPipelineResult` dataclass is the canonical hand-off to every downstream layer.
7. **DETERMINISTIC REPRODUCIBILITY**: pure-Python no-MLX-no-PyTorch at orchestration time; trainer-emitted `.npz` weights are sha-checksummed; canonical Provenance umbrella per Catalog #323.
8. **EXTREME OPTIMIZATION + PERFORMANCE**: CLI dry-run on canonical pair returns in <100ms (verified via pytest subprocess timing); orchestration overhead is dominated by canonical helper import (~10ms) + recipe-YAML regex extraction (~1ms).
9. **OPTIMAL MINIMAL CONTEST SCORE**: this layer does NOT directly contribute score per `mission_predicted_contribution=frontier_protecting` (per Catalog #300 v2 frontmatter + Catalog #309 horizon_class). It IS the canonical infrastructure that COLLAPSES Phase 3-10 lifecycle from ad-hoc-per-PR to one CLI invocation per Phase 1 spec §13 predicted ~180x wall-clock speedup.

---

## 4. Cargo-cult audit per assumption per Catalog #303

| Assumption | HARD-EARNED / CARGO-CULTED | Unwind path |
|---|---|---|
| 4-layer canonical pattern (helper + CLI + STRICT gate + cathedral consumer) sufficient for Phase 2 | **HARD-EARNED** per Catalog #245/#313/#344/#354 sister precedents; Phase 1 spec memo §1 explicit | No unwind needed |
| Phase 2 scope = PREPARATION only (no paid dispatch invocation) | **HARD-EARNED** per Phase 1 spec memo §3 + operator NON-NEGOTIABLE "stagger pattern" | No unwind needed |
| Canonical helper @ tools/ not src/tac/ requires sys.path injection (NOT importlib.util.spec_from_file_location) | **HARD-EARNED** EMPIRICALLY (test failure in first iteration; `@dataclass` decorator needs module in sys.modules) | Fixed in same commit batch via sys.path injection |
| MLX-first auto-detection from hardware substrate class is canonical | **HARD-EARNED** per 8th standing directive verbatim | No unwind |
| Recipe-YAML per-axis predicted band parsing via pure regex (no YAML lib dep) | **HARD-EARNED** per 12th canonicalization × ease-of-contest-compliance: dependency-free helper preserves portability per HNeRV parity L4 (≤2 deps) | No unwind |
| 6 exit codes (0-5) per Phase 1 spec memo §7 | **HARD-EARNED** per Phase 1 spec explicit | No unwind |
| `skip_protocol_verification` opt-out is canonical | **HARD-EARNED** per Phase 1 spec memo "preparation phase" + sister `--dry-run` precedents in `tools/operator_authorize.py`, `tools/gc_experiments_results.py` | No unwind |
| Cathedral consumer Tier A observability-only is correct for Phase 2 | **HARD-EARNED** per Catalog #341 + #355 sister precedent (Meta-Lagrangian invocation also bounded observability-only) | No unwind |
| Canonical equation #344 stays FORMALIZATION_PENDING until Phase 10 | **HARD-EARNED** per Catalog #344 acceptance cascade (placeholder waiver allowed for design memo predating first empirical anchor) | Phase 10 lands first empirical anchor |

---

## 5. Observability surface per Catalog #305

The 6-facet observability definition is satisfied:

1. **Inspectable per layer**: every emitted result is a typed frozen `CompressionPipelineResult` inspectable at runtime; per-axis predicted band is a typed frozen `PerAxisPredictedBand` separately inspectable; Catalog #270 protocol blockers are a tuple of strings.
2. **Decomposable per signal**: `dispatch_optimization_protocol_overall_pass` + `dispatch_optimization_protocol_blockers` per Tier; `per_axis_predicted_band` per axis (seg + pose + bytes); `hardware_substrate_class` per routing class.
3. **Diff-able across runs**: byte-stable `as_dict()` serialization sorted-keys JSON; per-result canonical Provenance carries `captured_at_utc` + `lane_id` + `substrate_id` lineage tuple.
4. **Queryable post-hoc**: result is dict-serializable via `as_dict()` → JSON; downstream Phase 3-10 layers consume the dict; cathedral consumer routes the result through canonical posterior at Phase 6 / Phase 10.
5. **Cite-able**: every result carries `(lane_id + substrate_id + canonical_equation_id + measurement_utc + canonical_helper_invocation)` tuple per Catalog #323 canonical Provenance umbrella.
6. **Counterfactual-able**: `skip_protocol_verification=True` enables "what would the verdict be if we bypassed Catalog #270?" without invoking the umbrella; downstream Phase 6 `paired_auth_eval` enables "what would the score be on each axis?" without coupling axes.

---

## 6. Horizon class declaration per Catalog #309

`horizon_class: frontier_protecting` per Catalog #300 mission-alignment.

**Rationale**: Phase 2 Layer 0 does NOT directly lower contest score (apparatus growth, not substrate optimization). It IS frontier-protecting per the 2026-05-19 6-phase manual cleanup empirical anchor: ad-hoc encoder pipeline invocations bypass Catalog #270 umbrella verification; the canonical orchestrator structurally extincts that bug class. Per Phase 1 spec memo §13: this is the first layer of the canonical-automated submission pipeline that collapses lifecycle from ~3h manual to <60s automated → unlocks more PR111+ attempts per remaining contest window → MORE chances at frontier-breaking score reductions.

Sister classification per Catalog #309 valid bands:
- **NOT plateau-adjacent** (this is NOT a [0.180, 0.200] within-class refinement)
- **NOT frontier-pursuit** ([0.120, 0.180] sub-medal substrate)
- **NOT asymptotic-pursuit** ([0.050, 0.120] class-shift architecture)
- **FRONTIER-PROTECTING enabler** per Catalog #309 + #300

`mission_predicted_contribution: frontier_protecting` per Catalog #300 v2 frontmatter.

---

## 7. 6-hook wire-in declaration per Catalog #125

1. **Hook #1 sensitivity-map contribution**: ACTIVE (per-axis predicted band feeds `tac.sensitivity_map.*` consumers via per-axis decomposition per Catalog #356).
2. **Hook #2 Pareto constraint**: ACTIVE (Catalog #270 umbrella verdict + per-axis predicted band feed Pareto polytope solver per Phase 1 spec memo §8).
3. **Hook #3 bit-allocator hook**: ACTIVE (cathedral consumer's `readiness_verdict` feeds bit-allocator priority cascade so READY candidates rank ahead of BLOCKED for same predicted delta band).
4. **Hook #4 cathedral autopilot dispatch hook**: ACTIVE PRIMARY (cathedral consumer at `src/tac/cathedral_consumers/compression_pipeline_readiness_consumer/__init__.py` is auto-discovered per Catalog #335/#336/#337; consume_candidate returns canonical Tier A markers per Catalog #341).
5. **Hook #5 continual-learning posterior update**: ACTIVE (cathedral consumer's `update_from_anchor` is the canonical Phase 6/Phase 10 anchor consumer; canonical equation #344 registration deferred to Phase 10 first empirical anchor).
6. **Hook #6 probe-disambiguator**: ACTIVE (cathedral consumer's `readiness_verdict` IS the canonical disambiguator between READY / BLOCKED / UNKNOWN; downstream Phase 7 operator runbook CLI consumes this verdict for routing).

---

## 8. ORDER discipline per 11th standing directive verified

Per the 11th ORDER-MATTERS meta-principle:

- **FIRST**: Phase 1 audit specification memo (canonical input; READ-ONLY artifact)
- **SECOND (THIS landing)**: Phase 2 Layer 0 compression pipeline canonical helper (the FIRST production code consumer of Phase 1 spec; this is the canonical first Phase 1 spec consumer per the 11th ORDER-MATTERS directive)
- **THIRD onward**: Phase 3 archive_grammar / Phase 4 builder + linter + attribution / Phase 5 compliance / Phase 6 paired_auth_eval / Phase 7 operator runbook CLI / Phase 8 STRICT preflight gate / Phase 9 cathedral consumer / Phase 10 PR111-candidate end-to-end regression

Phase 2 IS the first Phase 1 spec consumer. Phase 3+ depends on `CompressionPipelineResult` dataclass shape this landing pins.

---

## 9. 12th canonicalization × standardization × ease-of-contest-compliance trinity declaration

| Axis | Concrete evidence |
|---|---|
| **CANONICALIZATION** | Single canonical entry point `build_compression_pipeline(...)`; single typed return `CompressionPipelineResult` with frozen invariants; single helper invocation `verify_compression_pipeline_protocol_complete(...)`; single CLI `tools/compression_pipeline_cli.py`; single cathedral consumer; single canonical equation id (`compression_pipeline_canonical_helper_consolidation_savings_v1`) |
| **STANDARDIZATION** | Frozen dataclass + canonical Provenance umbrella per Catalog #323; canonical hardware substrate tokens per Catalog #190; canonical exit-code taxonomy per Phase 1 spec memo §7; canonical 4-layer pattern sister of Catalog #245/#313/#344/#354 |
| **EASE-OF-CONTEST-COMPLIANCE** | No-dependency recipe parsing (pure regex; no YAML lib); CLI `--dry-run` flag enables "what would the pipeline emit?" without invoking paid dispatch; CLI `--json` enables machine-readable consumer integration; CLI `--skip-protocol-verification` enables pre-recipe-fix dry-run preparation; canonical hardware substrate token override via `--explicit-hardware-substrate` for contest-runner-specific routing |

---

## 10. Apples-to-apples verification per 10th standing directive

Per CLAUDE.md "Apples-to-apples evidence discipline" + the 10th standing directive:

- **Phase 2 scope**: encoder pipeline PREPARATION only; NO score claim emitted from this layer.
- **Canonical Provenance**: every emitted `CompressionPipelineResult` carries `axis_tag=[predicted]` + `score_claim=False` + `promotable=False` + `evidence_grade=[predicted; compression-pipeline-canonical]`.
- **Canonical equation status**: `FORMALIZATION_PENDING` until Phase 10 first paired-CUDA empirical anchor of wall-clock collapse lands.
- **Hardware substrate routing**: macOS local-MPS routes to `macos_arm64_m5_max` token (NEVER score-promotable per Catalog #192 + CLAUDE.md "MPS auth eval is NOISE"); remote routes to canonical 1:1 contest-compliant Linux x86_64 tokens.
- **Phase 6 paired_auth_eval** (future-subagent) is where actual paired-axis empirical anchor lands; Phase 2 PREPARES the encoder context but does NOT invoke dispatch.

---

## 11. Operator-routable next step

Per Phase 1 spec memo §3 implementation queue + the 7-session stagger pattern at §5:

**Next subagent (Session N+2)**: Phase 3 archive_grammar (`tac.submission_packet.archive_grammar`) — depends on Phase 2 landing (`CompressionPipelineResult` dataclass shape is now pinned). Per Phase 1 spec memo §3 Phase 3 prompt template: ~400-600 LOC src + ~250-400 tests; routes Catalog gates #139 / #105 / #220 / #272 / #266; sister helper `tools/verify_distinguishing_feature_byte_mutation.py`; emits `parser_section_manifest.json` sidecar; runs byte-mutation smoke per Catalog #272.

**Alternative routes**:
- Phase 4 builder (depends on Phases 2 + 3)
- Phase 5 compliance (depends on Phases 2 + 3 + 4)
- Phase 6 paired_auth_eval (depends on Phases 2 + 3 + 4 + 5)

Per Phase 1 spec memo §5 staggered dispatch plan: Phase 3 is the recommended next step.

---

## 12. Discipline footer

- Catalog #229 Premise Verification: Phase 1 audit spec memo + 4 sister META-LIFT canonical helpers (cross_substrate_master_gradient_analyzer + pareto_polytope_unified_solver + uniward_invariant_enumerator + canonical_equations) + existing infrastructure (canonical_dispatch_optimization_protocol.py + trainer_skeleton.py + smoke_auth_eval_gate.py) read pre-edit.
- Catalog #117 / #157 / #174 / #235 / #289 canonical serializer + POST-EDIT `--expected-content-sha256` for every committed file.
- Catalog #119 Co-Authored-By trailer.
- Catalog #206 checkpoints: 4 in-progress checkpoints emitted before this commit + 1 complete checkpoint at commit.
- Catalog #110 / #113 HISTORICAL_PROVENANCE APPEND-ONLY: NEW files only; zero mutation of existing memos / canonical helpers / sister landings.
- Catalog #230 sister-subagent ownership map: scope STRICTLY disjoint from META-LIFT-1/2/4 + Phase 1 audit memo + Cascade C' work; verified zero overlap.
- Catalog #287 placeholder rejection: every rationale ≥4 chars + non-placeholder; gate's own docstring examples cannot self-waive.
- Catalog #290 + #294 + #303 + #305 + #309 design-memo sections satisfied.
- Catalog #335 cathedral consumer canonical contract verified.
- Catalog #340 sister-checkpoint guard PROCEED.
- Catalog #341 Tier A observability-only.
- Catalog #344 FORMALIZATION_PENDING preserved.
- Catalog #356 per-axis decomposition supported (when recipe carries predicted_band).
- 7th-12th standing directives binding.

**No paid dispatch invoked. No subagent spawned. No `gh pr create` / `git push`. No `submissions/exact_current/` touched. No META-LIFT-1/2/4 / Phase 1 audit memo / Cascade C' mutations.**

**$0 GPU. ~50 min wall-clock.**

---

## End of Phase 2 landing memo

**Test count**: 53/53 pass (cleanly).
**Total LOC**: 2350 (1434 src + 916 tests).
**Canonical equation #344 status**: FORMALIZATION_PENDING (Phase 10 lands first empirical anchor).
**Horizon class**: frontier_protecting.
**Mission contribution per Catalog #300**: frontier_protecting (apparatus growth that enables faster lifecycle for downstream substrate work).
