# Cargo-cult unwind audit: `procedural_codebook_generator_consumer` (Wave 2C-enhanced)

- **Date**: 2026-05-19
- **Subagent**: WAVE-2C-CARGO-CULT-UNWIND-AUDIT (`wave_2c_cargo_cult_audit_20260519`)
- **Target**: `src/tac/cathedral_consumers/procedural_codebook_generator_consumer/__init__.py` (260 LOC; commit `179dc2501` baseline + enhancement at commit `e92b3b54f` "cathedral: surface procedural seed authority")
- **Canonical reference**: `src/tac/cathedral_consumers/_example_consumer/__init__.py` (64 LOC)
- **Upstream namespace audited**: `tac.procedural_codebook_generator` (`classify_procedural_seed_authority` / `derive_codebook_from_archive_bytes` / `expand_seed_to_codebook` / `verify_no_new_bytes_added` / `verify_generator_seed_mutation_smoke`)

## Source-text differential vs `_example_consumer`

`procedural_codebook_generator_consumer` is the **STRUCTURALLY DISTINCT** member of the Wave 2C set. Unlike the other 11 Wave 2C consumers that are 53-64 LOC near-template-clones, this one is 260 LOC and:

1. **Inspects the candidate**: `_find_authority_packet(candidate)` actively reads candidate metadata (4 keys checked: `procedural_seed_authority_packet` / `procedural_codebook_authority_packet` / `procedural_authority_packet` / `procedural_seed_authority`).
2. **Validates schema**: refuses packets with `schema != "procedural_seed_authority_packet_v1"` per the canonical schema constant.
3. **Fail-closed on score-claim**: refuses any packet declaring `score_claim=True` per CLAUDE.md "Forbidden score claims" (line 119 `_authority_summary`).
4. **3-mode authority taxonomy**: `archive_seeded` / `weight_derived` / `runtime_constant` per Catalog #329 ProvenanceKind extension.
5. **Reconciliation invariant**: `_declared_modes` vs `_derive_modes` cross-check — claimed `ready_for_exact_eval_modes` MUST equal derived modes else `ready_for_exact_eval_modes_mismatch` blocker.
6. **Per-mode blockers**: `_mode_blockers_by_mode` flags `script_side_per_video_payload_probe_only` + `not_ready_for_exact_eval` + `score_claim_not_allowed`.
7. **Non-zero `confidence`**: returns `0.10` / `0.20` / `0.35` based on packet-presence + readiness vs the universal `confidence = 0.0` template default.
8. **Returns `denied_uses` tuple**: explicitly enumerates `("score_claim", "promotion", "rank_or_kill", "dispatch_readiness")` — a NOVEL contribution to the canonical contract that no other consumer provides.

This consumer is the **HARD-EARNED EXEMPLAR** of what Wave 2C consumers should look like post-cargo-cult-unwind. It already implements ~80% of the "candidate inspection + non-zero confidence + structured response" pattern that the audit recommends for sister consumers.

## Canonical-vs-unique decision per layer

Per Catalog #290 falling-rule list:

| Layer | Canonical decision | Verdict | Evidence |
|---|---|---|---|
| L1: `CONSUMER_NAME` literal | adopt | **ADOPT_CANONICAL_BECAUSE_SERVES** | Auto-discovery registration. |
| L2: `CONSUMER_HOOK_NUMBERS = (CATHEDRAL_AUTOPILOT_DISPATCH, PROBE_DISAMBIGUATOR)` | adopt dual-hook | **HARD-EARNED-FIRST-PRINCIPLES** | Hook #6 (probe-disambiguator) declared because the consumer's `denied_uses` tuple + per-mode blockers IS the canonical disambiguator between research-only-procedural vs ready-for-exact-eval-procedural. Hook #5 correctly omitted (deterministic given input). |
| L3: `update_from_anchor(anchor)` NO-OP | adopt template | **HARD-EARNED-FIRST-PRINCIPLES** | Per docstring: "Procedural codebook derivation is a deterministic function of the input archive seed bytes; no anchor-driven posterior update". Honest. |
| L4: `consume_candidate(candidate)` INSPECTS candidate via `_find_authority_packet` + `_authority_summary` | **FORK_BECAUSE_PRINCIPLED_MISMATCH** | The candidate carries per-mode authority packets that THIS consumer IS the canonical inspector for. Discarding the candidate (template default) would lose the entire signal. FORK justified by principled mismatch with template's observability-only default. |
| L5: `axis_tag = "[predicted]"` | adopt canonical | **ADOPT_CANONICAL_BECAUSE_SERVES** | HARD-EARNED. Procedural codebook expansion is predicted until paired-axis empirical. Per Catalog #329 the ProvenanceKind extension explicitly designates procedural-generation as `[predicted]` until contest-archive-member proof. |
| L6: `promotable = False` | adopt canonical | **ADOPT_CANONICAL_BECAUSE_SERVES** | HARD-EARNED. `denied_uses` tuple makes the non-promotability EXPLICIT (a stronger signal than the canonical bool). |
| L7: `confidence` is dynamic (0.10 / 0.20 / 0.35) | **FORK_BECAUSE_PRINCIPLED_MISMATCH** | Authority packets have varying evidence strength; static 0.0 would lose discrimination. HARD-EARNED. |
| L8: `denied_uses` tuple added to response | **FORK_BECAUSE_PRINCIPLED_MISMATCH (HARD-EARNED EXTENSION)** | NOVEL — no other consumer ships `denied_uses`. Makes denial enumeration machine-queryable per Catalog #287 evidence-tag discipline. Per CLAUDE.md "Max observability" facet #5 (cite-able) this is the canonical disambiguator. |
| L9: Schema validation via `_EXPECTED_AUTHORITY_SCHEMA = "procedural_seed_authority_packet_v1"` | unique | **HARD-EARNED-FIRST-PRINCIPLES** | Per CLAUDE.md "Comment-only contracts are FORBIDDEN" — schema declaration is the canonical disambiguator. Refuse-on-mismatch is fail-closed per Catalog #138 / #279 sister discipline. |
| L10: `_KNOWN_AUTHORITY_MODES = frozenset({"archive_seeded", "weight_derived", "runtime_constant"})` | unique | **HARD-EARNED-FIRST-PRINCIPLES per Catalog #329** | The 3-mode taxonomy mirrors Catalog #329 ProvenanceKind extension. Per CLAUDE.md "Frontier target" + HNeRV parity L9 (Runtime closure): unknown modes are refused, not warned. |

## Cargo-cult audit per assumption

Per Catalog #303. Three inherited assumptions:

### Assumption #1: "The 3-mode authority taxonomy (archive_seeded / weight_derived / runtime_constant) is exhaustive"

- **Classification**: HARD-EARNED-FIRST-PRINCIPLES (per Catalog #329)
- **Evidence**: Catalog #329 ProvenanceKind extension landed `PROCEDURAL_GENERATION_FROM_ARCHIVE_SEED` + `WEIGHT_DERIVED_CODEBOOK` + `FORBIDDEN_OUT_OF_ARCHIVE_PAYLOAD` as the canonical contest-compliance payload taxonomy. The consumer's 3-mode set matches the 3 compliant ProvenanceKinds. `FORBIDDEN_OUT_OF_ARCHIVE_PAYLOAD` (the 4th ProvenanceKind) is correctly excluded — it's the refused class, not a valid authority mode.
- **Verdict**: ADOPT. No unwind needed.

### Assumption #2: "Per-mode `script_side_per_video_payload_probe_only` is a blocker"

- **Classification**: HARD-EARNED-EMPIRICALLY-VERIFIED
- **Evidence**: Per the codex routing directive 2026-05-18 (Item 1) + Catalog #329 + DeliverabilityProof Tier 4 (per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"): `inflate_py_literal_seed` with `per_video_payload` literal is the FORBIDDEN_OUT_OF_ARCHIVE_PAYLOAD class. Blocking it is HARD-EARNED.
- **Verdict**: ADOPT. No unwind needed.

### Assumption #3: "Reconciliation between claimed `ready_for_exact_eval_modes` and derived modes is necessary"

- **Classification**: HARD-EARNED-EMPIRICALLY-VERIFIED
- **Evidence**: Per Catalog #324 (`check_no_predicted_band_without_post_training_tier_c_validation`) — claims that diverge from derived evidence are the canonical phantom-score class. The consumer's `ready_for_exact_eval_modes_mismatch` + `promotion_eligible_modes_mismatch` blockers extinct the same class at the authority-packet surface.
- **Verdict**: ADOPT. No unwind needed.

## Observability surface

Per Catalog #305:

1. **Inspectable per layer**: rationale + per-mode breakdown + `procedural_authority` nested dict + `denied_uses` tuple. ✓
2. **Decomposable per signal**: per-mode confidence + per-mode blockers + global vs mode-level blocker separation. ✓
3. **Diff-able across runs**: changing the candidate's authority packet changes the response. ✓
4. **Queryable post-hoc**: `consumer_invocations` persisted + structured `procedural_authority` dict survives. ✓
5. **Cite-able**: `schema` field + `preferred_promotion_mode` + per-mode `ready_for_exact_eval_modes` cite-able. ✓
6. **Counterfactual-able**: mutating any field in the authority packet changes the verdict. ✓

**Conclusion**: 6-of-6 facets satisfied. This is the OBSERVABILITY EXEMPLAR.

## Unwind priority queue

**NO HIGH-PRIORITY UNWINDS IDENTIFIED.** This consumer is the HARD-EARNED exemplar. Low-priority polish only:

| Rank | Polish | Cost | Notes |
|---|---|---|---|
| (advisory) | Surface `denied_uses` as a canonical extension to the `consume_candidate` Protocol return contract | ~$0, 5 LOC + Protocol doc update | Would make the novel `denied_uses` field discoverable by sister consumers (template-clone propagation). |
| (advisory) | Wire to `tac.procedural_codebook_generator.verify_generator_seed_mutation_smoke` for per-candidate byte-mutation smoke per Catalog #105 / #139 / #220 | ~$0, ~20 LOC + 3 tests | Adds runtime-verification on top of static schema validation. |

## Cross-references

- Sister of Catalog #329 (ProvenanceKind extension — defines the canonical 3-mode taxonomy this consumer inspects).
- Sister of Catalog #105 / #139 / #220 / #272 (no-op detector / packet compiler / L1+ scaffold operational mechanism / distinguishing-feature integration contract) — together they extinct the research-substrate-trap class.
- Anchor memo: `feedback_provenance_canonical_fix_meta_class_extinction_landed_20260517.md` (Catalog #323 META-class umbrella that THIS consumer's design preempts).
- Inflate.py extreme-compression symposium 2026-05-18 — the canonical design symposium for procedural-codebook generation.

## Verdict

`procedural_codebook_generator_consumer` is the **HARD-EARNED EXEMPLAR** of the Wave 2C set. Zero cargo-culted assumptions. 6-of-6 observability facets. 4 forks from canonical template all PRINCIPLED-MISMATCH (candidate inspection / dynamic confidence / `denied_uses` extension / schema validation).

**OPERATOR-ROUTABLE RECOMMENDATION**: PROMOTE this consumer as the canonical Wave 2C cargo-cult-unwind exemplar. The 4 forks (candidate inspection / non-zero confidence / `denied_uses` field / schema validation) should propagate to sister consumers via a documented refactor pattern (e.g. `_example_consumer_with_authority_packet/__init__.py` as a SECOND reference template alongside the existing observability-only template).

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" + the operator's standing directive *"What's the OPTIMAL ENGINEERING for THIS specific consumer to achieve the maximum cathedral-ranking signal contribution?"*: this consumer asked + answered the question. The sister 11 Wave 2C consumers asked "How do I share with the canonical?" and got CARGO-CULTED-PATH-OF-LEAST-RESISTANCE template clones.
