# Phase 4 — `tac.submission_packet.builder` canonical helper LANDED (Layer 2 per Phase 1 spec)

# PHANTOM_NAME_DESIGN_PROPOSAL_OK_FILE:phase_4_landing_memo_references_canonical_equation_id_FORMALIZATION_PENDING_per_catalog_344_until_phase_10_first_paired_cuda_empirical_anchor_lands_per_phase_1_spec_memo_phase_10_acceptance_criteria

**Date**: 2026-05-26
**Subagent**: `phase-4-builder-canonical-inflate-runtime-bundler-per-phase-1-spec-memo-layer-2-largest-single-layer-unblocks-phase-5-6-downstream-20260526`
**Lane**: `lane_phase_4_submission_bundle_canonical_helper_20260526` L1 (impl_complete + memory_entry)
**Scope**: Implement Layer 2 of the 7-layer canonical-automated submission pipeline per Phase 1 audit specification memo at `.omx/research/canonical_submission_pipeline_specification_memo_20260526.md` Layer 2.

**Authority**:
- Operator NON-NEGOTIABLE 2026-05-26 *"Remember everything we had to do to clean up and properly bundle our submission, let's make that canonical and automated moving forward"* (9th standing directive)
- Operator NON-NEGOTIABLE 2026-05-26 amendment *"Remember contest compliance and bundling full compression script and all and everything"*
- Operator blanket approval 2026-05-26 *"All operator decisions approved"* + *"No need to stagger now"* + *"Keep the queue fed"* (Phase 4 unblock)
- META frame: AUTOMATED + COMPOUNDING + OPTIMAL (7th META standing directive)

---

## 1. What landed

Per Phase 1 audit specification memo §3 Phase 4 prompt template (consumed expanded for Layer-2-only scope per operator's `--no-stagger` directive; Layers 3 + 6 deferred to subsequent sister-subagents per the original 10-phase queue's natural ordering):

- **NEW** `src/tac/submission_packet/builder.py` (~1345 LOC source; largest single layer per Phase 1 spec; includes embedded inflate.py + select_inflate_device template strings per HNeRV parity L4 self-contained-emit invariant)
- **NEW** `tools/submission_bundle_cli.py` (~313 LOC; operator-facing CLI sister of Phase 2 + Phase 3 CLIs)
- **NEW** `src/tac/cathedral_consumers/submission_bundle_builder_consumer/__init__.py` (~145 LOC; canonical contract-compliant cathedral consumer per Catalog #335)
- **NEW** `src/tac/tests/test_submission_bundle.py` (~617 LOC; **49 tests pass** covering all 9 dimensions per Catalog #294)
- **MODIFIED** `src/tac/submission_packet/__init__.py` (re-exports Phase 4 public API; sister wire-in to Phase 2 + Phase 3 lineage)

**Total LOC committed (new + modified)**: ~2420 LOC.

Tests pass count: **49** in new test module + **383** in sister cathedral/consumer/bundle suite (`src/tac/tests/test_submission_bundle.py` + `src/tac/cathedral_consumers/` + `src/tac/tests/test_dual_tier_consumer_architecture.py`).

---

## 2. Canonical helper public API

The Layer 2 canonical API per Phase 1 spec memo:

```python
from tac.submission_packet import (
    SubmissionBundleResult,
    SubmissionBundleError,
    DependencyClosureManifest,
    SelectInflateDeviceRouting,
    PythonpathSelfContainmentStatus,
    build_submission_bundle,
    build_dependency_closure_manifest,
    derive_submission_bundle_provenance,
    DEFAULT_INFLATE_PY_LOC_BUDGET,  # 200 per HNeRV parity L4
    DEFAULT_INFLATE_DEPS_BUDGET,    # 2 per HNeRV parity L4
    NUMPY_PORTABLE_INFLATE_DEPS,    # frozenset({"numpy"})
    HNERV_CLASS_INFLATE_DEPS,       # frozenset({"torch", "numpy"})
)

result = build_submission_bundle(
    compression_pipeline_result=phase_2_result,
    archive_grammar_manifest=phase_3_manifest,
    output_dir=Path("submissions/pr111_candidate/"),
    declared_dependencies=("numpy",),  # canonical numpy-portable default
    vendor_pythonpath_self_containment=True,  # Catalog #295
    select_inflate_device_routing=SelectInflateDeviceRouting.INLINE_WITH_WAIVER.value,  # Catalog #205
)
# result.submission_dir is a contest-compliant bundle ready for Phase 5+ consumers.
```

The bundler emits **7 canonical components** into `submission_dir/`:

| Component | Catalog gates routed |
|---|---|
| `inflate.sh` | Catalog #146 (3-arg signature) + `set -euo pipefail` |
| `inflate.py` | HNeRV parity L4 (≤200 LOC + ≤2 deps + numpy-portable) + Catalog #205 (canonical select_inflate_device mirror with INLINE_DEVICE_FORK_OK waiver) + Catalog #295 (PYTHONPATH self-containment) |
| `README.md` | PR 95 medal-class precedent + Catalog #208 (no local-absolute-paths) + public-PR hygiene (Claude/Anthropic refused) |
| `report.txt` | Canonical evaluator-output placeholder; Phase 6 paired_auth_eval rewrites with real bytes |
| `archive.zip` | Copied with `os.utime(dst, None)` per Catalog #361 (Modal artifact-filter mtime-fresh) |
| `archive_manifest.json` | Per-member identity sidecar (PR101/PR102/PR103 precedent) |
| `parser_section_manifest.json` | Phase 3 ArchiveGrammarManifest sidecar (consumed; not re-emitted) |

---

## 3. Canonical-vs-unique decision per layer

| Layer | Decision | Rationale |
|---|---|---|
| Frozen dataclasses | ADOPT_CANONICAL_BECAUSE_SERVES | Sister of Phase 2 + Phase 3 `CompressionPipelineResult` + `ArchiveGrammarManifest`; same `__post_init__` invariant pattern; canonical `as_dict()` round-trip. |
| Canonical Provenance | ADOPT_CANONICAL_BECAUSE_SERVES | Catalog #323 umbrella; every persisted row carries axis_tag + evidence_grade + score_claim + promotable + canonical_helper_invocation + captured_at_utc. |
| Cathedral consumer | ADOPT_CANONICAL_BECAUSE_SERVES | Catalog #335 Protocol contract; sister of `archive_grammar_builder_consumer` + `compression_pipeline_readiness_consumer`. |
| CLI exit codes | FORK_BECAUSE_PRINCIPLED_MISMATCH | Phase 4 exit-code semantics (HNeRV parity L4 LOC + deps + PYTHONPATH) differ from Phase 2 + Phase 3; per Phase 1 spec memo §7 each Layer has distinct exit-code taxonomy. |
| inflate.py template body | FORK_BECAUSE_SUPPRESSES | Canonical scaffold is per-substrate UNIQUE-AND-COMPLETE; canonical-only would suppress substrate-optimal decode logic (HNeRV parity L7 substrate-engineering split). The bundler accepts `inflate_body=...` kwarg for substrate-specific decode interpolation. |
| select_inflate_device mirror | ADOPT_CANONICAL_BECAUSE_SERVES | Catalog #205 canonical mirror byte-identical to `tac.substrates._shared.inflate_runtime.select_inflate_device` (modulo torch-import guard); INLINE_DEVICE_FORK_OK waiver per the canonical self-contained inflate-runtime contract. |

---

## 4. 9-dimension success checklist evidence

1. **UNIQUENESS**: Phase 4 Layer 2 is structurally unique — no prior canonical helper at this surface; 14 per-substrate ad-hoc submission_dir emitters this canvas consolidate to ONE canonical helper.
2. **BEAUTY + ELEGANCE**: Single canonical entry point `build_submission_bundle(...)` returns one typed result; downstream Phase 5-10 layers compose without re-deriving conventions.
3. **DISTINCTNESS**: Differs from Phase 2 (compression orchestrator) + Phase 3 (archive grammar) — Phase 4 emits the runtime closure files (inflate.sh + inflate.py + README + report.txt + manifest sidecars).
4. **RIGOR**: 49 dedicated tests + 383 sister tests pass; canonical Provenance per Catalog #323; HNeRV parity L4 invariants ENFORCED at `__post_init__` (refuses over-budget without waiver); Catalog #208 + public-PR hygiene scanners refuse leaks at emit time.
5. **OPTIMIZATION PER TECHNIQUE** (Catalog #290): canonical-vs-unique decision per layer documented above; FORK chosen for inflate.py template body to honor UNIQUE-AND-COMPLETE-PER-METHOD; ADOPT for everything else.
6. **STACK-OF-STACKS-COMPOSABILITY**: orthogonal to Phase 5 (compliance) + Phase 6 (paired_auth_eval) + Phase 7 (operator runbook CLI); each consumes `SubmissionBundleResult` as canonical input.
7. **DETERMINISTIC REPRODUCIBILITY**: `as_dict()` sorted-keys + canonical SHA verification + canonical Provenance + `archive_manifest.json` per-member identity (PR101/PR102/PR103 precedent).
8. **EXTREME OPTIMIZATION + PERFORMANCE**: bundle emit < 1 second on M5 Max; sister of Phase 2 + Phase 3 sub-second helpers.
9. **OPTIMAL MINIMAL CONTEST SCORE**: Phase 4 is INFRASTRUCTURE (frontier_protecting per Catalog #300); does not directly improve score but extincts the bug class that produces incorrect / non-contest-compliant submissions (PR101 / NSCS06 v5 historical anchors).

---

## 5. Cargo-cult audit per assumption

| Assumption | Classification | Rationale |
|---|---|---|
| HNeRV parity L4 ≤200 LOC inflate.py is reviewable-in-30s | HARD-EARNED | PR 95 / PR 100 / PR 101 / PR 102 / PR 103 medal-class precedent empirically validates 30-sec-review per reviewer's manual scan time. |
| Catalog #205 inline select_inflate_device mirror | HARD-EARNED | A1 council Round 1 F1/F11 + Catalog #205 landing memo proved silent inline device-fork produces +0.0335 CPU/CUDA gap on SAME bytes. |
| Catalog #295 PYTHONPATH self-containment | HARD-EARNED | NSCS06 v5 Modal dispatch failure `fc-01KRQMAQ7V41AFYMJH5HRK9P10` empirically proved bare `from tac.*` imports crash on Modal worker. |
| Catalog #146 3-arg inflate.sh signature | HARD-EARNED | Upstream contest evaluator + PR 95 / PR 100 / PR 101 medal-class precedent verifies this signature. |
| Catalog #361 Modal artifact filter mtime-fresh | HARD-EARNED | OVERNIGHT-CC commit `99d06f967` empirical falsification of DP1 vendored modules due to source-mtime preservation by `shutil.copy2`. |
| `report.txt` placeholder format | HARD-EARNED | Live A1 + PR101 + PR102 `report.txt` empirical inspection shows the canonical format; Phase 6 paired_auth_eval rewrites with real values. |
| `archive_manifest.json` per-member identity | HARD-EARNED | A1 submission_dir + PR 101 + PR 102 medal-class precedent carry this sidecar. |
| numpy-portable inflate canonical default | CARGO-CULTED-PENDING-PHASE-6 | 8th MLX-first standing directive *predicts* numpy-only inflate; PR 101 baseline empirically requires torch (HNeRV decoder). Phase 6 paired_auth_eval + Phase 10 first-PR-through-canonical-pipeline regression empirically tests whether canonical scaffold passes auth_eval (predicted: yes for new substrates emitting NumPy-only decoders; HNeRV-class substrates declare `declared_dependencies=("numpy", "torch")` waiver). |
| Default report.txt placeholder is harmless | HARD-EARNED | Phase 6 paired_auth_eval OVERWRITES the placeholder with real evaluator output BEFORE any PR submission; placeholder serves canonical-shape validation only. |

---

## 6. Observability surface

Per Catalog #305 6-facet observability definition:

1. **Inspectable per layer** — every emitted file (inflate.sh / inflate.py / README.md / report.txt / archive_manifest.json) is readable post-emit; `SubmissionBundleResult.as_dict()` exposes per-layer paths + sizes + LOCs.
2. **Decomposable per signal** — `DependencyClosureManifest` decomposes deps into `(declared_dependencies, budget, within_budget, numpy_portable, waiver_rationale)`; `inflate_py_loc` exposed separately from `inflate_py_loc_budget`.
3. **Diff-able across runs** — canonical SHA verification + `archive_manifest.json` sorted-keys serialization + `parser_section_manifest.json` byte-stable; two bundles for the same `(CompressionPipelineResult, ArchiveGrammarManifest)` input produce byte-identical outputs (modulo mtime which is intentionally fresh per Catalog #361).
4. **Queryable post-hoc** — `SubmissionBundleResult.as_dict()` → JSON; CLI `--json` flag emits canonical machine-readable output; cathedral consumer publishes `readiness_verdict` ∈ `{READY, BLOCKED, REVIEW_REQUIRED, UNKNOWN}` per Catalog #341 markers.
5. **Cite-able** — every result carries `canonical_helper_invocation="tac.submission_packet.build_submission_bundle"` + `canonical_equation_id` per Catalog #323/#344; `(lane_id, substrate_id, archive_sha256)` triple uniquely identifies the bundle.
6. **Counterfactual-able** — caller can substitute `inflate_body=...` to test different substrate decoders; CLI flag set covers every dimension; bundle reproducible per byte from inputs.

---

## 7. Predicted ΔS band (HARD-EARNED-PER-DYKSTRA-FEASIBILITY-DEFERRED)

Per Catalog #296 + #324: Phase 4 is INFRASTRUCTURE (frontier_protecting per Catalog #300); does NOT directly emit a `predicted_band` for any specific archive. Phase 6 paired_auth_eval + Phase 10 first-PR-through-canonical-pipeline regression land the first empirical-band evidence. The canonical equation `submission_bundle_canonical_helper_consolidation_savings_v1` is FORMALIZATION_PENDING per Catalog #344 until Phase 10's first paired-CUDA + Linux x86_64 CPU empirical anchor lands.

---

## 8. Horizon class

`apparatus_maintenance` per CLAUDE.md "Mission alignment — non-negotiable" + Catalog #309. Phase 4 collapses 14 per-substrate ad-hoc submission_dir emitters to ONE canonical helper; unblocks Phase 5 (compliance) + Phase 6 (paired_auth_eval) + Phase 7 (operator runbook) + Phase 8 (Catalog #362 STRICT gate) + Phase 9 (cathedral consumer) + Phase 10 (first-PR-through-canonical-pipeline regression).

---

## 9. 6-hook wire-in declaration per Catalog #125

1. **SENSITIVITY_MAP** = N/A (defensive infrastructure helper; no signal contribution at orchestration time)
2. **PARETO_CONSTRAINT** = N/A (no Pareto-relevant signal at Phase 4)
3. **BIT_ALLOCATOR** = ACTIVE — `DependencyClosureManifest.numpy_portable` + `inflate_py_loc` feed downstream bit-allocator priority cascade so canonical numpy-portable HNeRV parity L4 bundles rank ahead of multi-dep heavyweight bundles for the same predicted-delta band; consumed by the cathedral consumer's `readiness_verdict`
4. **CATHEDRAL_AUTOPILOT_DISPATCH** = ACTIVE PRIMARY — `submission_bundle_builder_consumer` is the Tier-A observability-only consumer; auto-discovered per Catalog #335/#336/#337; emits `readiness_verdict` per candidate without mutating ranker predicted delta (Catalog #341 invariant)
5. **CONTINUAL_LEARNING_POSTERIOR** = ACTIVE — per-bundle readiness anchor feeds the canonical posterior so Phase 6/Phase 10 empirical anchor landings inherit the apriori bundle-readiness signal via `tac.canonical_equations.update_equation_with_empirical_anchor` per Catalog #344
6. **PROBE_DISAMBIGUATOR** = ACTIVE — per-bundle `pythonpath_self_containment_status` ∈ `{CLEAN, VENDORED_WITH_EXPLICIT_WAIVER, SCAFFOLD_PENDING}` IS the canonical disambiguator between bundle readiness states per Catalog #295

---

## 10. ORDER discipline per 11th standing directive

Phase 4 consumes Phase 3 `ArchiveGrammarManifest` dataclass shape + Phase 2 `CompressionPipelineResult` dataclass shape per their canonical contracts (lane_id + substrate_id + archive_sha256 lineage verified at `build_submission_bundle` entry). Phase 4 PINS `SubmissionBundleResult` dataclass shape for Phase 5 (compliance) + Phase 6 (paired_auth_eval) + Phase 7 (operator runbook) + Phase 9 (cathedral consumer) downstream. Order verified empirically by 49 tests + 383 sister tests passing on full lineage.

---

## 11. 12th canonicalization × standardization × ease-of-contest-compliance trinity

- **CANONICALIZATION**: ONE entry point (`build_submission_bundle`) + ONE return shape (`SubmissionBundleResult`) + ONE bundle-emission protocol (7 canonical components per `submission_dir/`)
- **STANDARDIZATION**: per Catalog #146 contest-compliant 3-arg inflate.sh + HNeRV parity L4 ≤200 LOC inflate.py + Catalog #205 select_inflate_device + Catalog #295 PYTHONPATH self-containment + Catalog #361 Modal mtime-fresh
- **EASE-OF-CONTEST-COMPLIANCE**: single CLI invocation `tools/submission_bundle_cli.py --lane-id <...> --archive-path <...> --output-dir <...>` emits a contest-compliant `submission_dir/` ready for Phase 5 + Phase 6 + Phase 7 + final operator-trigger `gh pr create`

---

## 12. 13th OPTIMAL-TRIO declaration (techniques × ways × times)

- **TECHNIQUE**: canonical 4-layer pattern (Layer 1 helper module + Layer 2 CLI + Layer 3 STRICT gate placeholder + Layer 4 cathedral consumer) per Catalog #245/#313/#344/#354 precedent
- **WAY**: synthesize one bundle per call; canonical numpy-portable scaffold is the default WAY; substrate-specific `inflate_body=...` kwarg is the substrate-engineering opt-out WAY
- **TIME**: Phase 4 lands NOW (after Phase 2 + Phase 3, before Phase 5 + Phase 6 + Phase 7 + Phase 8 + Phase 9 + Phase 10) per the operator's `--no-stagger` directive + the dependency chain in Phase 1 spec memo §3

---

## 13. 8th MLX-first numpy-portable verified

The bundler operates at orchestration time on archive bytes; no MLX nor PyTorch dependency at orchestration time (canonical default per the 8th standing directive). The emitted canonical numpy-portable scaffold inflate.py has `declared_dependencies=("numpy",)` (canonical default); HNeRV-class substrates declare `declared_dependencies=("numpy", "torch")` (≤2 deps per HNeRV parity L4) via the canonical opt-out kwarg. MLX-first encoder training lives INSIDE the trainer module the Phase 2 pipeline wraps; numpy-portable decoder INSIDE the Phase 4 emitted bundle. Bridge contract: encoder emits `.npz` per Phase 2 `CompressionPipelineResult.weights_export_path`; Phase 4 inflate.py reads `.npz` + decodes via numpy primitives (substrate-specific decode logic interpolated via `inflate_body=...`).

---

## 14. Sister coordination + Catalog #230 disjoint scope verified

Per Catalog #230 sister-subagent ownership map: my scope is NEW `tac.submission_packet.builder` + sister CLI + cathedral consumer + tests + landing memo. Disjoint from:
- Cascade C' WAVE-7 (different substrate)
- V14-V2 + ORDER-gates + DROP-MANY-audit + META-LIFT work (sister surfaces)
- Phase 2 + Phase 3 sister landings (consumed; not modified except `__init__.py` re-exports)

No modifications to `submissions/exact_current/` (CLAUDE.md mutation frontier honored).

Catalog #340 sister-checkpoint guard PROCEED expected at commit time.

---

## 15. Operator-routable next

Per Phase 1 audit spec memo §3 + the operator's `--no-stagger` + `keep-the-queue-fed` directive, the operator-routable next-step queue is (Catalog #299 quota brake at #361/400 well under budget):

1. **Phase 5 linter** (`tac.submission_packet.linter`) per Phase 1 spec memo Layer 3 — wraps forbidden-token grep + first-person-plural grep + emdash audit + inflate.py LOC budget verification + archive.zip sha/size validation + tone audit into single typed verdict; ~400-600 LOC src + ~300-500 LOC tests
2. **Phase 6 compliance** (`tac.submission_packet.compliance`) per Phase 1 spec memo Layer 4 — typed wrapper over `scripts/pre_submission_compliance_check.py --contest-final --strict`; ~300-500 LOC src + ~200-400 LOC tests
3. **Phase 8 STRICT gate** (`check_pr_submission_packet_canonical`, Catalog #362) per Phase 1 spec memo §3 Phase 8 — STRICT preflight gate refusing non-canonical `submissions/*/` directories; ~200-400 LOC gate + ~300-500 LOC tests; WARN-ONLY initially per CLAUDE.md "Strict-flip atomicity rule"; PARALLEL-SAFE with Phase 5 + Phase 6 (different surfaces)

Per the operator's `--no-stagger` directive: Phase 5 OR Phase 6 OR Phase 8 can spawn in parallel; the dependency chain at the dataclass shape level is satisfied by Phase 4's `SubmissionBundleResult` landing in this commit batch.
