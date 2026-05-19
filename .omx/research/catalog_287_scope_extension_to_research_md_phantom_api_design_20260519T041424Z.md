# Catalog #287 scope-extension to research-memo phantom-API design

**Date:** 2026-05-18
**Author:** Main-Claude (META-PHANTOM-API-STRUCTURAL-EXTINCTION sister-subagent)
**Lane:** `lane_meta_phantom_api_structural_extinction_catalog_287_scope_extend_20260518`
**Authority:** operator decision E.1 2026-05-18 ("Approved proceed with all") + grand council T3 finding #5 PROCEED (commit `6606376eb`)

## Why scope-extend instead of new gate

Per CLAUDE.md "Gate consolidation discipline" (Catalog #299), catalog # is at 333 with a 400 cap. The mitigation proposal in the 15th-instance memo explicitly recommended:

> *Alternative*: instead of a NEW catalog #, EXTEND the existing Catalog #287 to scope-extend over `.omx/research/*.md` cited-module-names AND require either (a) the cited module is importable OR (b) the citation carries an explicit `# DESIGN_PROPOSAL_NOT_YET_IMPLEMENTED:<rationale>` tag.

Operator's mission directive explicitly says "scope-extend Catalog #287 ... over `.omx/research/*.md`". This memo documents the scope-extension design.

## What's being extended

Existing `check_no_docstring_overstatement_without_evidence_tag` (Catalog #287) currently:

- Scans `src/tac/**/*.py` for percentage / multiplier overstatement patterns
- Requires adjacent evidence tag `[empirical:...]` / `[contest-CUDA]` / etc. within ±5 lines
- Same-line waiver `# DOCSTRING_PERCENT_CLAIM_OK:<rationale>` (placeholder rejected)

Extension adds a second responsibility ("Catalog #287-B" sub-scope) to the SAME function:

- Scans `.omx/research/**/*.md` body text + memory files (`~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_*.md`) for `tac.X[.Y...]` module-name citations
- For each citation, the 2-component prefix `tac.X` must be importable via `importlib.util.find_spec()`
- Acceptance cascade:
  - (a) `tac.X` is importable → CLEAN
  - (b) same-line waiver `# PHANTOM_NAME_INTENTIONAL_OK:<rationale>` → CLEAN
  - (c) same-line waiver `# DESIGN_PROPOSAL_NOT_YET_IMPLEMENTED:<rationale>` → CLEAN (for design memos proposing NEW helpers)
  - (d) file-level waiver `# PHANTOM_NAME_DESIGN_PROPOSAL_OK_FILE:<rationale>` in first 30 lines → CLEAN (for design memos where every cited name is explicitly proposal-only)
  - all `<rationale>` / `<reason>` placeholder literals rejected per Catalog #287 family convention

## Why 2-component prefix only

A citation like `tac.atom.ledger.append_atom` is legitimate: `tac.atom.ledger` is importable; `append_atom` is a function inside it (not a module). Checking the full path with `find_spec` would falsely flag every function-inside-module citation. The 2-component prefix `tac.X` IS the actual module-name claim. Functions inside the module are pyflakes territory, not preflight scope.

A citation like `tac.magic_codec` has 2-component prefix `tac.magic_codec` which does NOT import → genuine phantom (canonical is `tac.codec_magic_registry`).

## Detection regex

```python
_CHECK_287B_TAC_MODULE_NAME_RE = re.compile(
    r"\btac\.[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*\b"
)
```

This matches dotted Python identifier paths starting with `tac.`. Boundaries via `\b` avoid partial matches like `mytac.foo`.

For each match, extract 2-component prefix and call `importlib.util.find_spec(prefix)`. Catch `ModuleNotFoundError`, `ImportError`, `ValueError` (some packages raise ValueError on certain malformed names).

## Excluded surfaces

- Markdown code-fenced blocks (between triple-backticks) — code examples may show proposed-not-yet-implemented APIs
- Inline-code spans (single-backtick) within prose are STILL scanned — these are the canonical "we have a tac.X helper" claims that the bug class targets
- HTML comments `<!-- ... -->` — used for waiver markers + paper-frozen banners
- Lines containing the canonical waiver tokens (`PHANTOM_NAME_INTENTIONAL_OK` / `DESIGN_PROPOSAL_NOT_YET_IMPLEMENTED`)

## Exempt path markers (sub-scope B)

- `.omx/research/MEMORY_CLUSTER_*` (cluster summary files are aggregations; defer to constituent memos)
- `.omx/research/_archive/` (historical archived research; closed for backfill)
- `.omx/research/_drafts/` (work-in-progress drafts; not yet authoritative)
- `_intake_` (vendored PR clones per Catalog #109)
- `.omx/oss_export/` (mirror)

## Self-exempt files (sub-scope B)

- `src/tac/preflight.py` (carries the regex literals + this docstring)
- `src/tac/tests/test_check_287_phantom_api_research_md_extension.py` (contains verbatim violation samples)
- `.omx/research/catalog_287_scope_extension_to_research_md_phantom_api_design_*.md` (THIS file — design memo describes phantom citations as examples)
- `.omx/research/meta_audit_addendum_15th_instance_phantom_canonical_helper_module_names_in_synthesis_memo_20260518.md` (the source memo documenting the phantom names that triggered this work)
- Memory files (memory dir is operator-private; scan only if `--scan-memory` opt-in; default OFF for OSS hermeticity per Catalog #290/#291/#292 sister design)

## Initial wire-in: WARN-ONLY

Per CLAUDE.md "Strict-flip atomicity rule" + the live-repo dry-run revealing 7+ phantom citations in a 10-file sample (estimated 50-200 total across `.omx/research/`), initial wire-in is `strict=False`. Existing violations require a backfill sister wave (operator-routed) where each phantom citation is either:

1. Corrected to the actual importable canonical name (e.g. `tac.magic_codec` → `tac.codec_magic_registry`)
2. Tagged `# DESIGN_PROPOSAL_NOT_YET_IMPLEMENTED:<rationale>` if the memo explicitly proposes a new helper that hasn't been built
3. Wrapped in file-level `# PHANTOM_NAME_DESIGN_PROPOSAL_OK_FILE:<rationale>` for design memos where every cited name is explicitly proposal-only

Strict-flip planned after live count = 0.

## Acceptance test plan (~20 tests)

1. **Live-repo regression guard** — current violation count <= a documented ceiling
2. **Phantom citation flagged** — `tac.magic_codec` in a research memo flagged
3. **Real citation accepted** — `tac.unified_action` accepted
4. **Function-inside-module accepted** — `tac.atom.ledger.append_atom` accepted (2-component `tac.atom.ledger` is importable)
5. **Top-level non-tac.X ignored** — `numpy.array` not flagged (not a `tac.X` claim)
6. **Same-line `PHANTOM_NAME_INTENTIONAL_OK` accepted with rationale**
7. **Same-line `PHANTOM_NAME_INTENTIONAL_OK` rejects placeholder `<rationale>`**
8. **Same-line `DESIGN_PROPOSAL_NOT_YET_IMPLEMENTED` accepted**
9. **Same-line `DESIGN_PROPOSAL_NOT_YET_IMPLEMENTED` rejects placeholder `<reason>`**
10. **File-level `PHANTOM_NAME_DESIGN_PROPOSAL_OK_FILE` in first 30 lines accepts whole file**
11. **File-level waiver outside first 30 lines NOT accepted**
12. **File-level waiver with placeholder rejected**
13. **Exempt path markers (`_archive/`, `_drafts/`, MEMORY_CLUSTER_) excluded**
14. **Code-fenced blocks excluded** (no flagging inside triple-backtick blocks)
15. **HTML comment lines excluded**
16. **Memory file scan OFF by default** (no `feedback_*.md` violations in default scope)
17. **Memory file scan ON when `scan_memory=True` opt-in**
18. **Existing `.py` docstring scope still works** (backwards compatibility — the percentage-overstatement check still fires)
19. **Self-exempt files (preflight.py, design memo, 15th-instance memo) not flagged**
20. **Strict mode raises with PreflightError when live count > 0**

## Cross-references

- CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable
- CLAUDE.md "Gate consolidation discipline" (Catalog #299) — scope-extension preferred over new gate
- 15th-instance memo `meta_audit_addendum_15th_instance_phantom_canonical_helper_module_names_in_synthesis_memo_20260518.md`
- Grand council T3 finding #5 `council_t3_finding_5_meta_audit_phantom_api_recurrence_20260518.md` PROCEED 8-of-8
- Catalog #185 sister META-meta drift gate (LIVE_COUNT zero claim discipline)
- Catalog #176 sister META-meta callsite-has-CLAUDE.md-row gate (this scope-extension MUST update the existing Catalog #287 CLAUDE.md row)
- Catalog #287 original landing memo `feedback_phase_2_land_5_gap_gates_omnibus_followon_landed_20260515.md`

## 6-hook wire-in declaration per Catalog #125

1. Sensitivity-map contribution — N/A: defensive validator gate, no signal contribution
2. Pareto constraint — N/A: no Pareto-relevant signal
3. Bit-allocator hook — N/A: no bit-allocator signal
4. Cathedral autopilot dispatch hook — ACTIVE: gate verdict is consumable by autopilot ranker as a canonical machine-readable verdict on memo claims (phantom-API-citing memos should NOT be promoted as design references)
5. Continual-learning posterior update — ACTIVE: every phantom-API audit emits structured violations that feed the META-audit posterior
6. Probe-disambiguator — N/A: no probe-disambiguator path; the gate IS the disambiguator between phantom-claim and real-citation

## Cargo-cult-vs-hard-earned per Catalog #292

- ASSUMPTION 1: "2-component prefix is the right granularity for `tac.X` claims" — HARD-EARNED (verified empirically: function-inside-module citations like `tac.atom.ledger.append_atom` are legitimate; over-specific path checks produce false positives)
- ASSUMPTION 2: "Scope-extension preferred over new gate per Catalog #299" — HARD-EARNED (operator directive + 15th-instance memo recommendation + catalog quota brake)
- ASSUMPTION 3: "Default to OFF for memory file scan to preserve OSS hermeticity" — HARD-EARNED (sister Catalog #290/#291/#292 pattern; clean clones + CI should not depend on operator's private `~/.claude/projects/` directory)



<!-- # FORMALIZATION_PENDING:pre_framework_memo_dated_2026-05-19_predates_canonical_equations_birthday_registry_population_in_progress_appended_by_strict_flip_enablers_per_operator_blanket_approval_per_claude_md_forbidden_premature_kill_without_research_exhaustion_this_is_DEFER_pending_canonical_equation_backfill_NOT_kill -->
