# Codex Finding 2 — Public PR Intake Pristine: Recursive Adversarial Review Log

**Date:** 2026-05-08
**Author:** claude:main (worker fork)
**Per CLAUDE.md "Recursive adversarial review protocol — non-negotiable"**: 3 consecutive clean passes required before deploy.

## Scope of review

- `src/tac/preflight.py:23788+` — new `check_public_pr_intake_clones_pristine` (~140 LOC including LFS-aware refinement)
- `src/tac/preflight.py:1275+` — wired into `preflight_all()` as STRICT
- `reverse_engineering/public_pr_waiver_manifest.json` — committed waiver-metadata schema
- `CLAUDE.md` — Forbidden-pattern entry + Check 109 catalog row
- `.omx/research/codex_finding_2_dirty_clone_revert_inventory_20260508.md` — revert inventory
- 10 reverted dirty clones (8 `_codex`/`_worker` + 2 `_auto` LFS-state)

## Round 1 — Yousfi (waiver-rationale preservation)

**Question:** Does reverting 39 inline `KL_BATCHMEAN_OK` waivers without migrating their rationale into the manifest lose actual signal? If a future scanner re-flags these positions, will operators have to re-derive why they were waived?

**Finding:** Two-level preservation already exists:
1. Each waiver had IDENTICAL rationale text (`public-PR-intake-external-code-not-our-quality-debt`) — no per-call-site uniqueness was lost.
2. The revert inventory at `.omx/research/codex_finding_2_dirty_clone_revert_inventory_20260508.md` captures the (clone, file, line, content) tuples should they ever need to be reconstructed.
3. **MORE IMPORTANTLY**: the KL_BATCHMEAN scanner already excludes `_intake_` paths via `_VENDORED_PATH_MARKERS` (`src/tac/preflight.py:12039`). The waivers were dead code at time of review — re-running `check_kl_div_reduction_correct(strict=False)` post-revert produces 0 violations. Waiver rationale was never going to be re-demanded by THIS scanner.
4. The waiver manifest provides a future-proof slot for any DIFFERENT scanner that does want to consult per-clone waivers. Its `waivers: []` is empty by design.

**Verdict (Yousfi):** CLEAN. Waiver rationale preservation is not at risk because the rationale was uniform AND the demand was already structurally extinct.

## Round 2 — Fridrich (any case where inline waiver is mandatory?)

**Question:** Is there any tooling (linters, CI, IDE plugins, AST-walking scanners outside our repo) that DEMANDS the waiver be co-located with the call site to function? If so, the manifest is insufficient.

**Finding:** No external dependency demands inline waivers in this codebase:
- All canonical scanners (`_scan_python_for_kl_div_batchmean`, MPS-fallback, dead-flag, dead-resolver, parametrize-strip, weights_only_false_OK) use the `_VENDORED_PATH_MARKERS` exclusion pattern OR scan only `src/tac/`/`tools/`/our `submissions/` (NOT `experiments/results/`).
- The new gate `check_public_pr_intake_clones_pristine` operates at the git-status level (above the AST scanner layer); it doesn't need waiver co-location.
- IDE / editor plugins that flag `reduction="batchmean"` could surface a warning in clone files, but those warnings are advisory and don't block any operational path.

**Verdict (Fridrich):** CLEAN. No tooling requires inline waivers; the manifest format is sufficient and the path-prefix exclusion (existing) is the actual mechanism doing the work.

## Round 3 — Contrarian (manifest scaling + structural soundness)

**Question:** The manifest has `waivers: []` (empty). What happens at scale — when N=20+ waivers accumulate? Does the format scale? Does the schema force lookup overhead on hot-path scanners?

**Finding:** Manifest design is structurally sound:
- Schema has explicit `row_format` documenting the (pr, path, line, waiver_kind, rationale, reviewed_by, reviewed_at) tuple — clear primary key.
- `discovery_paths` array makes the canonical clone roots explicit — scanner can cache the manifest once at startup.
- N-row scaling: at N=100 waivers, JSON parse is <10ms; the manifest is consulted only when a scanner is about to flag a finding inside a clone path, which is rare (the path-prefix exclusion catches it before the manifest lookup).
- Schema also documents the operational discipline ("Confirm the in-tree scanner actually flags the clone path") so future operators don't add rows preemptively.

**Risk surfaced (NEW — not yet a finding, just an observation):** The manifest doesn't yet have a programmatic consumer in the codebase — no scanner has wiring to consult it. This is acceptable for now (zero rows, zero demand) but should be wired the FIRST time a scanner needs to legitimately flag a clone-path finding. Documented in `guidance.before_adding_a_row` field.

**Verdict (Contrarian):** CLEAN with one observation. Schema scales; no consumer wiring is acceptable while `waivers: []`.

## Round 4 — Hotz (simpler approach?)

**Question:** Why not just move all clones into a `.gitignored/` subdirectory and skip the preflight gate entirely? Less code, less complexity.

**Finding:** Considered and rejected because:
- Clones are ALREADY effectively gitignored via `.gitignore` patterns (the parent repo doesn't track their content). The gate operates on the *clone's own* git status (the clone being a separate git repo / submodule).
- Moving them to a different directory would require rewriting every tool that references them by path (`tools/build_frontier_roadmap_status.py`, `tools/build_field_meta_dispatch_selection.py`, `experiments/results/pr100_107_reproduction_ledger_*`, dozens of memory references).
- The complexity isn't in the gate (~140 LOC) — it's in the discovery + LFS-aware diff handling, both of which are necessary regardless of where clones live.
- The gate also catches the META-bug class: it doesn't just flag THIS particular waiver pattern; it catches ANY future in-place edit (rebase artifacts, merge conflicts, accidental sed runs) that would corrupt source provenance.

**Verdict (Hotz):** CLEAN. The gate is the right complexity for the surface area being protected.

## Round 5 — Boyd (backward-compatibility / already-running tooling)

**Question:** Are there any tools/scripts/CI workflows currently in flight that DEPEND on the dirty-clone state? Will reverting them break anything?

**Finding:** Audit of tools that consume clone paths:
- `tools/build_frontier_roadmap_status.py` — reads clone metadata (intake ledgers, archive SHAs) but not source files.
- `experiments/results/pr100_107_reproduction_ledger_*/ledger.json` — references PR archive bytes + SHAs, not source.
- `tools/dispatch_advisor.py` / `tools/cathedral_autopilot.py` — consume evidence rows, not clone source.
- The KL_BATCHMEAN scanner (`check_kl_div_reduction_correct`) — already skips `_intake_` paths; scanner output is invariant under revert.
- The new gate (`check_public_pr_intake_clones_pristine`) — consumes git status, expects pristine; revert puts clones in expected state.

No consumer was reading the inline waiver comments as data. The revert is forward-compatible.

**Verdict (Boyd):** CLEAN. No backward-compat hazard.

## Round 6 — Shannon (final pass: information-theoretic completeness)

**Question:** Does the gate close the bug class structurally, or does it only catch THIS instance?

**Finding:** Structural closure verified:
- The gate doesn't pattern-match on `KL_BATCHMEAN_OK` specifically. It refuses ANY text edit inside a clone (any tracked file, any line, any waiver token). Future waiver tokens (DEAD_BYTES_AUDIT_OK, ROUNDTRIP_TESTED, ADMM_WAIVED, etc.) added in-place would all be caught equally.
- The discovery layer auto-adapts to new clone layouts: any directory matching `*_intake_*` under `experiments/results/` containing `.git` is scanned. New clones added by future intake operations are automatically protected.
- The LFS-aware refinement (numstat-based text-vs-binary distinction) means LFS-managed binary clones don't false-positive — the gate only flags content edits, not metadata mismatches.
- The CLAUDE.md FORBIDDEN entry + catalog row + memory file form a 3-channel documentation trail so future agents (claude or codex) encountering a flagged clone know exactly where to put waiver rationale.

**Verdict (Shannon):** CLEAN. Bug class structurally extinct.

## Counter status

- Round 1: CLEAN (Yousfi)
- Round 2: CLEAN (Fridrich)
- Round 3: CLEAN (Contrarian, with one observation about future consumer wiring)
- Round 4: CLEAN (Hotz)
- Round 5: CLEAN (Boyd)
- Round 6: CLEAN (Shannon)

**6 consecutive clean passes — exceeds the 3 required.** Per CLAUDE.md "Recursive adversarial review protocol — non-negotiable" the gate is cleared for deploy.

## Deferred items (NOT blocking deploy)

- Waiver-manifest consumer wiring: when the first scanner legitimately needs to flag a clone-path finding (no path-prefix exclusion possible), wire it to consult `reverse_engineering/public_pr_waiver_manifest.json`. Until then, `waivers: []` is sufficient.
- Sister-fork (FIX-1) is handling Codex Finding 1 (operator-approval scoping at `experiments/results/frontier_roadmap_status_*`); this fork stays out of that scope.
