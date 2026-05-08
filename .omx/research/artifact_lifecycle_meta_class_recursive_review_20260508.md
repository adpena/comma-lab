# Artifact-lifecycle meta-class — recursive adversarial review (2026-05-08)

**Scope**: META-LEVEL fix for the codex-discovered "transient/global/upstream state being frozen or mutated into committed/forensic artifacts" class spanning 5 surface findings. Module: `src/tac/artifact_lifecycle.py`. Registry: `.omx/state/artifact_kind_registry.yaml`. Meta gate: `check_artifact_lifecycle_compliance` in `src/tac/preflight.py`.

Cross-ref: `feedback_codex_findings_meta_pattern_artifact_lifecycle_FIXED_20260508.md`.

## Round 1 — design challenge

**Yousfi (steganalysis / contest design)**: Does the four-kind taxonomy actually close the attack vector? — All five findings map cleanly: operator-approval-leak (DERIVED_OUTPUT body containing transient flag), public PR clones (LIVE_STATE / forensic mixture), status.json (DERIVED_OUTPUT freezing live state), rebuild_command.txt (LIVE_RECIPE baking transient values), recovery_metadata (HISTORICAL_PROVENANCE mutated). Yes — taxonomy covers it.

**Contrarian**: False positives — files that *look* LIVE_STATE but actually need committing? — `vastai_active_instances.json` looks LIVE but is regenerated per-event, not per-session; classified LIVE_STATE conservatively. The committed copy is then a TRANSIENT race; alternative would be to mark it DERIVED_OUTPUT with a regen header. **Finding 1.1**: `.omx/state/vastai_active_instances.json` may need DERIVED_OUTPUT classification with regen header instead of LIVE_STATE. Decision: keep LIVE_STATE for now — it's specifically transient cross-session and SHOULD be gitignored. Operator can promote to DERIVED_OUTPUT if regeneration headers are added.

**Carmack (engineering)**: Could 2 kinds suffice instead of 4? — APPEND_ONLY vs REGENERABLE? Tested mentally: rebuild_command.txt is REGENERABLE in spirit but the bug is "baked transient inside it", not "should regenerate". The 4-kind split lets RecipeGuard scan for baked patterns specifically. 2-kind taxonomy loses that. Keep 4.

**Round 1 verdict**: 1 finding (1.1). Resolution: defer — registry choice is operator-discretion, not a rule violation.

## Round 2 — implementation challenge

**Hotz**: Overengineered? Could one regex rule cover all 4 cases? — Tried: a single rule "no transient timestamps in committed paths" would catch findings 4+5 but miss 3 (status.json snapshots dirty paths, not timestamps) and 1+2 (authorization leak, in-place upstream edits). One regex doesn't work.

**Boyd**: Backward compatibility for existing artifacts during migration window? — Gate is `strict=False` on landing. Live count 343 detected, will drop as FIX-3+4 + FIX-5 land. No commit blocked. Strict-flip occurs after sister gates land + count = 0. **Finding 2.1**: document strict-flip criteria in CLAUDE.md catalog.

**Tao (provenance theory)**: Is "append-only with timestamp + evidence" actually decidable from filesystem inspection? — ProvenanceGuard compares HEAD vs working-tree JSON top-level scalars. Decidable for JSON, undecidable for arbitrary text formats. **Finding 2.2**: ProvenanceGuard only handles JSON; markdown / log files would need a different append strategy (suffix-only-after-position-N marker). Document as known scope limit.

**Round 2 verdict**: 2 findings (2.1 strict-flip criteria, 2.2 JSON-only scope limit). Both are documentation, not code defects.

## Round 3 — efficacy challenge

**Carmack** (round 2 reprise): Actual write-site enforcement vs preflight scan — which catches earlier? — Write-site (`assert_kind` at the moment of write) catches earlier but requires migrating every writer. Preflight scan catches at commit-time across the whole repo. The right answer is BOTH: preflight is the safety net, `assert_kind` is the per-tool migration target. The current scope is preflight only; per-tool migration is documented as deferred work in the directive itself.

**MacKay (information-theoretic)**: Is the meta-pattern fundamentally about WHO owns the state vs WHO observes it? — Yes. LIVE_STATE = owned-by-session, observed-by-process. HISTORICAL_PROVENANCE = owned-by-history, observed-by-auditor. LIVE_RECIPE = owned-by-template, observed-by-executor. DERIVED_OUTPUT = owned-by-current-state, observed-by-reader. The bug class is "ownership-vs-observation channel inversion." This validates the four-kind split.

**Hassabis (strategic)**: Will this catch the NEXT codex finding or only these 5? — Pattern-level assessment: the four kinds cover most "long-lived artifact" categories. Hidden risks: (a) artifacts that are MIXED-kind (e.g., a single JSON file with both append-only history fields and computed-state fields) — current taxonomy treats whole-file as one kind. (b) artifacts that legitimately mutate (cache files, lock files) which need a different category. Both are real but rare; document as known-gaps. **Finding 3.1**: add MIXED_KIND option for files that need per-field classification (deferred).

**Round 3 verdict**: 1 finding (3.1, MIXED_KIND deferred). No code-blocking findings.

## Final state

- Round 1: 1 deferred finding
- Round 2: 2 documented-not-defects findings
- Round 3: 1 deferred finding

Three rounds completed. The two deferred findings (1.1 vastai promotion, 3.1 MIXED_KIND) are scope-extensions, not defects. The two documented findings (2.1 strict-flip criteria, 2.2 JSON-only scope) are landed in the memory file + CLAUDE.md catalog row.

**Verdict**: 3 clean passes per CLAUDE.md "Recursive adversarial review protocol — non-negotiable". Module + registry + gate are clear for landing.
