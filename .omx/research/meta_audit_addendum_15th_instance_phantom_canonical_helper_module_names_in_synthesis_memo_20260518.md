# 15th META-audit instance — phantom canonical-helper module names in unified-synthesis memo

**Date:** 2026-05-18
**Discovered by:** Main-Claude self-audit during synthesis-memo follow-on amendment
**Pattern class:** CONFLATE_DECLARATIVE_WITH_PHYSICAL (same as 14th instance G1 routing directive Provenance API)
**Source artifact:** `.omx/research/magic_codec_plus_water_filling_plus_lagrangian_redirection_unified_synthesis_cross_pollination_20260518.md` (commit `c29150445` — original body)
**Surface:** synthesis design memo prose-routing (not yet a STRICT-gated surface)
**Severity:** MEDIUM (memo is a NEXT-STEP routing directive; no GPU spend yet; would have produced ModuleNotFoundError on any subagent literal interpretation)

## What was claimed (phantom)

The synthesis memo body cited 5 canonical-helper module names as currently existing primitives:

1. `tac.magic_codec` (line 12 + 122)
2. `tac.water_filling` (line 14 + 122)
3. `tac.run_admm` (line 16 + 122)
4. `tac.meta_lagrangian_search` (line 16)
5. `tac.per_pair_optimal_treatment_plan_via_lagrangian_dual` (line 16 + 140) — as a top-level module

All 5 were stated as if they were importable Python modules. Each citation framed the helper as "the canonical X" with cross-references to specific Task #s as authority.

## What actually exists (physical)

Verified via `.venv/bin/python -c "import importlib; m = importlib.import_module('tac.X')"` on each name:

1. `tac.magic_codec` → **ModuleNotFoundError**. Actual canonical: `tac.codec_magic_registry` (6 public symbols). Sister: `tac.codec_op_admm_adapter` / `tac.codec_stack_planner` / `tac.codec_pipeline_joint_admm`.
2. `tac.water_filling` → **ModuleNotFoundError**. Actual canonical: `tac.water_filling_codec` (17 public symbols). Sister: `tac.water_filling_codec_v2`.
3. `tac.run_admm` → **ModuleNotFoundError**. Actual canonical: `tac.joint_admm_coordinator` (16 public symbols). Sister: `tac.joint_admm_proximal_water_filling_v2` (13) / `tac.joint_admm_proximal_pose_delta`.
4. `tac.meta_lagrangian_search` → **ModuleNotFoundError**. Actual canonical: `tac.meta_lagrangian_allocator` (9 public symbols, with `RATE_SCORE_PER_BYTE` / `CONTEST_ORIGINAL_BYTES` constants).
5. `tac.per_pair_optimal_treatment_plan_via_lagrangian_dual` (as a module) → **ModuleNotFoundError**. Actual canonical: the FUNCTION `per_pair_optimal_treatment_plan_via_lagrangian_dual` lives INSIDE `tac.master_gradient_consumers` (79 public symbols) alongside the `OptimalPerPairTreatmentPlan` dataclass + `OptimalPerPairTreatmentPlanError` exception. The synthesis memo's citation as a top-level module conflates function-vs-module surface.

Additionally, the body referenced `tac.uniward` (line 84) as the canonical UNIWARD helper. Verification: `tac.uniward` does NOT exist as a top-level module. The actual modules are `tac.uniward_delta` + `tac.uniward_texture` + `tac.symposium_impls.uniward_die_distortion_informed_embedding_map`. The memo's UNIWARD reference is functionally directionally correct (UNIWARD IS water-filling on inverse-local-variance) but the import path needs reader-side disambiguation.

The body also proposed TWO names as NEW helpers (`tac.water_filling_aware_quantize`, `tac.water_filling_plus_magic_codec_plus_admm`) — these are NOT phantom-existing claims (explicitly framed as proposed); however the proposed composite name `tac.water_filling_plus_magic_codec_plus_admm` is ugly and was SUPERSEDED in the amendment by pointing to the EXISTING canonical surface `tac.unified_action` (26 public symbols; see amendment §"CANONICAL UNIFIED-LAGRANGIAN SURFACE ALREADY EXISTS").

## Pattern: CONFLATE_DECLARATIVE_WITH_PHYSICAL (same as 14th instance)

The original 12-instance META-audit (commit `e86ca6d0c`) and the 14th instance (G1 routing directive's phantom Provenance API, commit `ecaa1c471`) both have the same structural cause:

**Declarative naming "by analogy" without grep-verifying actual import paths.**

In each case, the agent:
1. Knew the abstract concept ("there's a canonical helper for X")
2. Inferred a plausible canonical name from the concept name ("magic codec" → `tac.magic_codec`)
3. Wrote the prose-routing artifact citing the inferred name as a fact
4. Did NOT run `importlib.import_module()` or `grep "^def \|^class \|^from tac" src/tac/` to verify

The bug class CANNOT be caught by any STRICT preflight gate at the prose-design-memo surface today (no such gate exists; per CLAUDE.md "Gate consolidation discipline" non-negotiable, gate # approaching #400 limit makes adding one questionable). Instead the canonical mitigation is:

1. **Operator-readable correction in same artifact** — the amendment to the synthesis memo (appended 2026-05-18 per Catalog #110/#113 HISTORICAL_PROVENANCE) lists ALL 5 phantom names + actual canonical paths in a `## API CORRECTIONS` section. Latest-row-wins semantics make the corrections authoritative for any future reader.
2. **Standalone META-audit instance entry** — THIS file. The cumulative count (now 15 instances) is itself signal that the pattern is structural, not incidental, and warrants a discipline-level treatment per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable.

## Why this happened despite 14 prior instances

The 14th instance was just 1 hour earlier in the same session. I knew the bug class existed. I knew the canonical mitigation. I STILL wrote phantom-API names in the new synthesis memo.

Causal analysis:
- **Activation-failure of the "verify-import-paths-before-citing" reflex** — the reflex exists in declarative knowledge but does not fire automatically on every prose-routing-directive write. The 14 prior instances did NOT install a structural check that would have fired before the synthesis memo body landed.
- **Operator-routing-directive surface looks low-stakes** — prose synthesis memos feel like "just a design doc" not "a literal API reference." But ANY claim of "tac.X exists" IS a literal API reference because a downstream subagent literally tries `importlib.import_module('tac.X')`.
- **Time pressure / volume bias** — the body was written quickly under operator-saturate-the-queue cadence; the slow verification step felt like an interruption. This is exactly the CARGO-CULTED-fast-path that the META-audit pattern catches.

## Proposed structural mitigation

Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable, the canonical mitigation for the META-class is a STRICT preflight gate that:

1. Scans `.omx/research/*.md` for `tac.<module_path>` string references
2. Verifies each is importable via `importlib.util.find_spec(name)`
3. Refuses if any cited name is phantom-existing UNLESS:
   - Same-line waiver `# PHANTOM_NAME_INTENTIONAL_OK:<rationale>` (placeholder rejected; for design memos proposing NEW helpers explicitly)
   - File-level waiver `# PHANTOM_NAME_DESIGN_PROPOSAL_OK_FILE:<rationale>` (for design memos where every cited name is explicitly a proposal)

This would extinct the CONFLATE_DECLARATIVE_WITH_PHYSICAL META-class structurally at the prose-design-memo surface. Catalog # is approaching #400 limit per Catalog #299 (gate consolidation discipline) so this would consume one of the remaining slots — operator decision required.

**Alternative**: instead of a NEW catalog #, EXTEND the existing Catalog #287 (`check_no_docstring_overstatement_without_evidence_tag`) to scope-extend over `.omx/research/*.md` cited-module-names AND require either (a) the cited module is importable OR (b) the citation carries an explicit `# DESIGN_PROPOSAL_NOT_YET_IMPLEMENTED:<rationale>` tag. This is the META-meta consolidation pattern preferred per "Gate consolidation discipline."

## Cumulative META-audit instances chronology

1. Instances 1-12: original META-audit (commit `e86ca6d0c`; `meta_audit_conflate_declarative_with_physical_error_pattern_12_claim_self_audit_20260518.md`)
2. Instance 13: Codex metadata-bucket-vs-family epistemic catch (commit `f29d8a3a5`; `meta_audit_addendum_13th_instance_codex_metadata_bucket_vs_family_epistemic_catch_20260518.md`)
3. Instance 14: G1 routing directive phantom Provenance API self-catch (commit `ecaa1c471`; `meta_audit_addendum_14th_instance_self_caught_phantom_provenance_api_in_g1_routing_directive_20260518.md`)
4. **Instance 15 (THIS)**: synthesis memo phantom canonical-helper module names; 5 phantom imports + 1 uniward-as-top-level-module misclassification

Each instance happened within hours of the prior. The aggregation suggests an emergent structural failure mode that operator-readable per-instance correction does NOT extinct.

## Operator-routable decisions

1. **Approve STRICT preflight gate for phantom-API in design memos** (new Catalog # — approaching #400 limit; uses one of remaining slots) OR scope-extend Catalog #287 (preferred per META consolidation)
2. **Approve a session-wide reflex: every prose-routing-directive that cites `tac.X` MUST be preceded by an `importlib.import_module()` probe block in the same conversation** (operational discipline; not catalog-enforceable)
3. **Defer** — accept the 1-2 per-session rate of this bug class and rely on operator-readable corrections + addenda

## Cross-references

- Original META-audit: `meta_audit_conflate_declarative_with_physical_error_pattern_12_claim_self_audit_20260518.md`
- 13th instance: `meta_audit_addendum_13th_instance_codex_metadata_bucket_vs_family_epistemic_catch_20260518.md`
- 14th instance: `meta_audit_addendum_14th_instance_self_caught_phantom_provenance_api_in_g1_routing_directive_20260518.md`
- Cargo-cult burn-down supplement: `cargo_cult_burn_down_supplement_extending_meta_audit_across_session_20260518.md`
- Source artifact (corrected): `magic_codec_plus_water_filling_plus_lagrangian_redirection_unified_synthesis_cross_pollination_20260518.md` (now carries `## API CORRECTIONS` + `## CANONICAL UNIFIED-LAGRANGIAN SURFACE ALREADY EXISTS` appendix per Catalog #110/#113 HISTORICAL_PROVENANCE)
- Catalog #287 (`check_no_docstring_overstatement_without_evidence_tag`) — sister gate that the META-consolidation extension proposal would extend
- Catalog #299 (`check_catalog_quota_under_400`) — gate-quota brake making "add new gate" the more expensive option vs scope-extension
- CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable — canonical mitigation pattern

— Main-Claude 2026-05-18 (15th instance self-audit during synthesis-memo amendment)
