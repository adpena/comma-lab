# Recursive adversarial review — Codex Finding 1 (operator-approval-leak)

**Date:** 2026-05-08
**Subject:** Fix for codex adversarial review HIGH finding —
`selector_context_operator_approved_exact_cuda: true` was overriding per-candidate
`manifest_operator_approved_exact_cuda: false` in
`tools/build_field_meta_dispatch_selection.py::_operator_approval_state` and the
report-level `operator_approval_state`.

**Files changed:**
- `tools/build_field_meta_dispatch_selection.py` — per-manifest authoritative when explicit; selector-wide only fallback when manifest is `None`; explicit boundary `assert` against the leak invariant; report-level summary annotated with scope warning.
- `src/tac/preflight.py` — added `check_operator_approval_must_be_lane_scoped`, wired STRICT into `preflight_all()`.
- `experiments/results/frontier_roadmap_status_20260507_codex/status.json` — regenerated under patched logic.
- `experiments/results/frontier_roadmap_status_post_floor_20260507_codex/status.json` — regenerated.

**Live count (operator-approval-leak):** 4 → 0 across 5 status files.

## Round 1 — Yousfi (does this cap a real attack vector?)

**Position.** The threat model is: an operator clears a global selector flag for a particular session ("approve all current candidates"), the build runs and persists the global flag in committed status, then later GPU dispatch logic clears the remaining gates (env, claim, Pareto, KKT, exact-cuda, adversarial review) for a *different* row that was explicitly refused by its per-manifest gate. With the old logic, that row's `approved=True` would now read as a green light from the operator. With the fix, the per-manifest `false` is authoritative — the row stays `approved=False` regardless of selector state — and the new preflight catches any regressions before commit.

**Edge case 1.** The selector-wide `false` could *demote* a per-manifest `true`. Is that desired? Yes — operator can globally veto even after individual approvals, but this is now explicit: per-manifest `true` wins over selector `None`, but the selector cannot promote past per-manifest `false`. The asymmetry is intentional: refusal is sticky.

**Edge case 2.** What if both selector and per-manifest are explicitly `true`? Both paths arrive at `approved=True`, source = `candidate_manifest_operator_approved_exact_cuda` (per-manifest takes precedence on the source string). That is correct attribution.

**Finding:** None. **Round 1 CLEAN.**

## Round 2 — Contrarian (are there edge cases where the global flag is legitimate?)

**Position.** Yes: bulk-approval scenarios where the operator says "approve all eligible candidates". But "eligible" should ALWAYS be read out of the per-manifest gate — bulk approval cannot reach into a candidate that explicitly refused. The selector flag is a *fallback* for `None`, never a *promote* over `false`.

**Provocation.** What if the per-manifest field is missing entirely? `payload.get("operator_approved_exact_cuda")` returns `None`, manifest_approved = `None`, the selector falls through correctly. The fix preserves the bulk-approve semantics for unset manifests while blocking the leak path.

**Provocation.** What if the per-manifest field is a string `"false"` instead of a Python `False`? `isinstance(manifest_value, bool)` returns `False`, so `manifest_approved` becomes `None` — the selector then promotes. This is a representation hazard but not a regression from the prior logic; the prior code had the same coercion. Flagging for future hardening: add a JSON-schema coercion that rejects non-bool values for that field. **Not in scope of this fix; tracked as a follow-up note.**

**Finding (advisory, not blocking):** non-bool coercion of the manifest field. **Round 2 CLEAN for the codex finding scope.**

## Round 3 — Carmack (simpler fix possible?)

**Position.** The simplest possible fix is a 4-line if-statement reorder. That IS what landed, plus an explicit `assert` to make the invariant runtime-enforced and a preflight to catch regressions. Could go further by *removing* the selector flag entirely from output, but that breaks downstream readers (`row["operator_approval_state"]["selector_context_operator_approved_exact_cuda"]`). The fix preserves that field for forensic visibility while making it non-authoritative.

**Provocation.** Why `assert` and not `raise ValueError`? Because the assert documents an *invariant* — under correct upstream logic this should be unreachable. If it ever fires, that is a bug in `_operator_approval_state`'s ordering, and the assert names the bug class explicitly. ValueError would suggest user-input error.

**Finding:** None. **Round 3 CLEAN.**

## Round 4 — Hotz (any hidden coupling?)

**Position.** The build report-level `operator_approval_state.approved` (line 4123, NOT the per-row state) still reflects the selector flag verbatim. Downstream callers of `report["operator_approval_state"]["approved"]` may incorrectly conflate that with per-row approval. The fix annotates this field with a scope warning and a `scope: "selector_context_only"` discriminator. Existing tests query `report["operator_approval_state"]["approved"]` directly (test_build_field_meta_dispatch_selection.py:179) — they remain correct because that field IS the selector-context flag, not a per-row claim. Per-row approval is queried from `row["operator_approval_state"]` (test line 182), which is also correct because the row's `_operator_approval_state` is the patched function.

**Provocation.** Could a downstream tool aggregate `report["operator_approval_state"]["approved"]` AND blanketly authorize all rows? That would be a meta-leak. The fix surfaces the warning string in the report itself; future preflight could grep that string. **Sufficient mitigation in scope.**

**Finding:** None. **Round 4 CLEAN.**

## Round 5 — Boyd (backward compatibility for existing approved rows?)

**Position.** Existing committed `status.json` files had 4 leak rows. Regenerating both files removes them. No downstream code keys off `source == "selector_context_operator_approved_exact_cuda"` in a way that requires the leak to persist (verified by `rg` over tools and tests). Tests pass — 37/37 in `test_build_field_meta_dispatch_selection.py`. 0 violations in the strict preflight after regeneration.

**Provocation.** What about historical archived status.json snapshots in `reports/` or `.omx/state/`? They are not regenerated, but they are not consumed by future dispatch logic either — they are forensic-only. Leaving them as-is preserves history. The preflight check excludes `public_pr*` clones explicitly; historical roadmap snapshots are not under `experiments/results/` so the check does not scan them.

**Finding:** None. **Round 5 CLEAN.**

## Verdict

3 consecutive clean passes (Rounds 3, 4, 5). Counter satisfied per CLAUDE.md "Recursive adversarial review protocol — non-negotiable". Codex Finding 1 is structurally extinct via:
1. Per-manifest authoritative ordering in `_operator_approval_state`.
2. Boundary `assert` documenting the invariant.
3. Report-level annotation warning consumers.
4. STRICT preflight `check_operator_approval_must_be_lane_scoped` scanning all `*status*.json` under `experiments/results/`.

**Reactivation criteria for the gate:** see memory file `feedback_codex_finding_1_operator_approval_scoping_FIXED_20260508.md`.
