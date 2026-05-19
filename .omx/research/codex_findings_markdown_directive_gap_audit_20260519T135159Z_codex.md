# Codex findings - Markdown directive gap audit

**Date:** 2026-05-19T13:51:59Z
**Actor:** Codex
**Scope:** all Markdown control-plane files reachable from `/Users/adpena/Projects/pact`, with canonical emphasis on root docs plus `.omx/research`.
**Mode:** read-only audit except this findings memo.

## Executive verdict

There ARE Claude/operator Markdown directives that Codex could miss if it relies
only on `tools/canonical_task_status.py --list-pending`.

The live false-negative class is not "Markdown impossible to scan"; it is
specific and fixable:

- `tools/extract_canonical_tasks_from_directive.py` only recognizes headings of
  the form `ITEM N`.
- The newest Claude-to-Codex routing memos use `Wire-in #N`, `Build #N`, and
  `Item N`.
- Therefore `canonical_task_status --list-pending` can return `(no rows)` while
  unregistered actionable work is present in `.omx/research`.

## Post-fix status

Codex fixed the extractor false-negative class in this turn.

Changed:

- `tools/extract_canonical_tasks_from_directive.py` now parses:
  - `ITEM N` / `Item N` -> `ITEM_N`
  - `Wire-in #N` -> `WIRE_IN_N`
  - `Build #N` -> `BUILD_N`
  - `CLUSTER X` -> `CLUSTER_X`
  - bold `Sub-cluster X.Y` -> `CLUSTER_XY`
- Parent cluster headings that only bundle sub-clusters are suppressed, so
  `CLUSTER_F` does not shadow real `CLUSTER_F1` and `CLUSTER_F2` tasks.
- `src/tac/tests/test_extract_canonical_tasks_from_directive.py` covers the
  live 2026-05-19 directive files that exposed the miss.

State updated:

- Registered the missed live rows from the 07:20 comprehensive wire-in
  directive, 07:15 paid-dispatch directive, and BCEF max-score directive in
  `.omx/state/canonical_task_status.jsonl`.
- Marked canonical Wire-in #2 completed by linking it to the prior
  `WIRE_IN_2_VERIFY` evidence at commit `a3777ac05`.
- Marked post-slot-1 dispatch follow-ons as blocked only on the narrower
  post-fix verification gates, not on the stale "silent-no-spawn fix missing"
  premise. Slot 1's structural fix landed at commit `233fce252`.

Validation:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q src/tac/tests/test_extract_canonical_tasks_from_directive.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/ruff check tools/extract_canonical_tasks_from_directive.py src/tac/tests/test_extract_canonical_tasks_from_directive.py
.venv/bin/python tools/canonical_task_status.py --validate
.venv/bin/python tools/canonical_task_status.py --list-pending
```

## Audit counts

- Tracked/non-ignored Markdown files: 267.
- All local Markdown files under the repo, including hidden/generated trees:
  7485.
- Canonical `.omx/research` Markdown files: 2183.
- Recent `.omx/research` Markdown files dated/modified in the last two days:
  several hundred; high-signal subset reviewed by filename and heading pattern.

Generated or duplicate-heavy trees (`.claude/worktrees`, `.omx` snapshots,
`experiments/results`, some reverse-engineering result copies) should not be
treated as independent source-of-truth. Root docs plus `.omx/research` remain
the canonical control plane.

## Finding 1 - current extractor misses the newest directive shapes

**Severity:** HIGH

Evidence:

- `tools/extract_canonical_tasks_from_directive.py` has
  `ITEM_HEADER_RE = r"^#{2,4}\s+ITEM\s+(\d+)..."`.
- Running it on the two newest routing directives returns `[]` for both:
  - `.omx/research/codex_routing_directive_comprehensive_wire_in_and_integration_pass_20260519T072000Z.md`
  - `.omx/research/codex_routing_directive_session_20260519_paid_dispatch_batch_C6_plus_204_followon_20260519T071500Z.md`
- `tools/canonical_task_status.py --list-pending` currently returns `(no rows)`.

Impact:

Codex can falsely believe there is no pending Claude-routed work even when
newest operator-approved routing memos contain live items.

Recommended fix:

Completed in this turn. The extractor now parses at least:

- `### Wire-in #N: ...` -> `WIRE_IN_N`
- `### Build #N: ...` -> `BUILD_N`
- `#### Item N: ...` -> `ITEM_N`

and also cluster/sub-cluster rows. The missing rows were registered through the
canonical fcntl-locked task-status helper.

## Finding 2 - comprehensive wire-in directive has unregistered live items

**Source:** `.omx/research/codex_routing_directive_comprehensive_wire_in_and_integration_pass_20260519T072000Z.md`

Classification:

| Item | Status | Notes |
|---|---|---|
| Wire-in #1 `bit_allocator_per_pair_consumer` | LIKELY MISSED / PARTIAL-ABSORBED | Directive asks for exact package `src/tac/cathedral_consumers/bit_allocator_per_pair_consumer/__init__.py`; repo instead has `per_pair_coding_budget_allocation_consumer`. This may satisfy the spirit, but not the literal task/test surface. Needs explicit resolution memo or alias/consumer if desired. |
| Wire-in #2 per-byte sensitivity cathedral reweight | COMPLETED | Canonical task row exists and was completed at commit `a3777ac05`; current consumer advertises hooks #1/#3/#4/#6. |
| Wire-in #3 `per_pair_difficulty_atlas_consumer` -> posterior | MISSED / UNREGISTERED | Autopilot already consumes `per_pair_difficulty_atlas` sidecars, but directive asks for a NEW cathedral consumer with hook #5 and continual-learning posterior writes. No `src/tac/cathedral_consumers/per_pair_difficulty_atlas_consumer/` package exists. |
| Wire-in #4 master-gradient wire-in audit | COMPLETED | Canonical task row exists and was completed at commit `036a73a6c`. |
| Wire-in #5 HF Jobs dispatcher cathedral surface | PARTIAL / MISSING CONSUMER | HF Jobs ledger, dispatcher, operator-authorize route, and tests landed; no `src/tac/cathedral_consumers/hf_jobs_dispatcher_consumer/` package found. |
| Wire-in #6 HF Jobs cost ranker | PARTIAL / CHECK PRICES + FALLBACK | `src/tac/cost_band_calibration.py` includes `hf_jobs`, but local rates differ from the directive's quoted table (`t4-small` is `0.50` locally vs directive `0.40`). Also `_CHEAPER_ALTERNATIVE_FALLBACKS_PER_CLASS["eval"]` still lists only `("modal", "A10G")`, so the HF Jobs cheaper-eval routing part is not complete. Needs authority reconciliation, not silent change. |
| Build #1 HF Jobs SegNet surrogate | PARTIALLY UNBLOCKED | Dataset/jobs infrastructure and symposium memo exist; dispatch state should be checked through lane claims/operator-authorize before launching. |
| Build #2 Z6 Wave 2 4c refire | REGISTERED / BLOCKED ON POST-FIX VERIFICATION | Silent-no-spawn fix completed in commit `233fce252` per subagent progress at 2026-05-19T12:36:54Z. Remaining blocker is the directive-required fresh post-fix Z6 4c dispatch verification plus normal claim/preflight. |
| Build #3 STC v2 ratify/defer | REGISTERED / BLOCKED ON POST-FIX SMOKE | Same structural fix cleared; remaining blocker is successful post-fix STC smoke before RATIFY-or-DEFER. |

## Finding 3 - paid dispatch directive has unregistered live items

**Source:** `.omx/research/codex_routing_directive_session_20260519_paid_dispatch_batch_C6_plus_204_followon_20260519T071500Z.md`

Classification:

| Item | Status | Notes |
|---|---|---|
| Item 1 C6.1 lane_17_imp LTH reactivation | REGISTERED PAID DISPATCH | Needs lane claim, operator-authorize route, and per-substrate evidence before spend. |
| Item 2 C6.3 PR106 #05+#06 reformulated paired smoke | REGISTERED FOLLOW-ON | Wave 1 design memo landed; paid paired smoke follow-on is now mirrored in task status. |
| Item 3 C6.5 mae_v + saug operational fix | REGISTERED PAID DISPATCH | Needs exact trainer integration/claim path audit before spend. |
| Item 4 Catalog #204 A1 passthrough recovery | REGISTERED RECOVERY DISPATCH | Catalog #204 driver fix landed; follow-on recovery eval remains explicit in directive. |
| Item 5 Z6 Wave 2 4c refire | REGISTERED / BLOCKED ON POST-FIX VERIFICATION | Silent-no-spawn fix now complete at `233fce252`; remaining blocker is fresh post-fix dispatch verification. |
| Item 6 STC v2 ratify/defer | REGISTERED / BLOCKED ON POST-FIX SMOKE | Silent-no-spawn fix now complete at `233fce252`; remaining blocker is a successful post-fix STC smoke. |

Do not dispatch these from chat alone. Convert to canonical rows first, then
run the normal claim/preflight/operator-authorize path.

## Finding 4 - max-score batch directive has unregistered clusters

**Source:** `.omx/research/codex_routing_directive_session_20260519_max_score_lowering_batch_BCEF_20260519T051028Z.md`

The BCEF directive uses `## CLUSTER ...` headings, so it is another extractor
false-negative. E.1 and F.2 were already mirrored/completed in canonical task
status, but the following remain unregistered:

| Cluster | Status | Notes |
|---|---|---|
| Cluster B event-driven retroactive sweep gate | REGISTERED | Requires a new preflight gate that verifies retroactive-sweep evidence when a new gate lands. Scope touches `src/tac/preflight.py`, tests, CLAUDE.md row, memo. This is high-churn; wait for `preflight.py` stability. |
| Cluster C Phantom-API Wave 2-4 backfill drain | REGISTERED | Continue Wave 1's 418 -> 194 reduction down to 0 and strict-flip Catalog #287 sub-scope B. Scope owns many existing `.omx/research/*.md` memos plus manifest JSON. |
| Cluster F.1 sigma=15 per-substrate sweep design | REGISTERED | Produce design memo for actual grayscale-LUT codepath sigma grid across five consumers. F.2 600-pair-independence test is already completed separately. |

## Finding 5 - T3 findings Lagrangian memo is active partner-owned, not a Codex miss

**Source:** `.omx/research/grand_council_t3_findings_lagrangian_and_pp_integration_design_symposium_20260519.md`

The memo contains a concrete slot-21 build spec:

- `src/tac/findings_lagrangian/`
- `tools/cathedral_autopilot_autonomous_loop.py` uncertainty downweighting
- `src/tac/mps_diagnostic/drift_predictor.py` uncertainty extension
- Catalog #345 preflight gate in `src/tac/preflight.py`
- about 80 tests

However, `.omx/state/subagent_progress.jsonl` shows the T3 subagent completed
the council memo/anchor/lane updates and was ready for canonical serializer
commit at 2026-05-19T13:51:12Z. Treat this as active partner-owned until the
worktree stabilizes. Do not seize `src/tac/canonical_equations/`,
`src/tac/preflight.py`, or related Lagrangian surfaces while they are dirty.

The same memo has one real operator decision: Q6 asks whether to run a
hierarchical-Bayesian shrinkage measurement cycle. That is blocked on operator
choice, not a Codex execution miss.

## Finding 6 - current dirty worktree confirms realtime churn

At audit time `git status` showed local main ahead plus partner WIP:

- modified `.omx/state/lane_maturity_audit.log`
- modified `.omx/state/lane_registry.json`
- modified `src/tac/preflight.py`
- untracked 2026-05-19 routing/council memos and canonical-equation surfaces

This triggers the operator's "back off and monitor until stable" rule for
shared high-churn surfaces.

## Immediate next actions

1. Resolve literal-vs-spirit status for Wire-in #1:
   `bit_allocator_per_pair_consumer` vs existing
   `per_pair_coding_budget_allocation_consumer`.
2. Implement or explicitly defer Wire-in #3
   `per_pair_difficulty_atlas_consumer` hook #5 posterior update.
3. Implement or explicitly defer Wire-in #5
   `hf_jobs_dispatcher_consumer`.
4. Reconcile HF Jobs cost-table authority before editing rates.
5. Add HF Jobs to the cheaper-eval fallback list only after rate authority is
   reconciled.
6. Only after registration/claim/preflight, consider paid dispatch items from
   the 07:15 directive.

## Commands run

```bash
git status --branch --short
rg --files -g '*.md' | wc -l
find . -path ./.git -prune -o -name '*.md' -print | wc -l
find .omx/research -maxdepth 1 -name '*.md' -print | wc -l
.venv/bin/python tools/canonical_task_status.py --list-pending
.venv/bin/python tools/canonical_task_status.py --list-by-owner codex
.venv/bin/python tools/extract_canonical_tasks_from_directive.py --json --directive .omx/research/codex_routing_directive_comprehensive_wire_in_and_integration_pass_20260519T072000Z.md
.venv/bin/python tools/extract_canonical_tasks_from_directive.py --json --directive .omx/research/codex_routing_directive_session_20260519_paid_dispatch_batch_C6_plus_204_followon_20260519T071500Z.md
rg -n "bit_allocator_per_pair_consumer|per_pair_difficulty_atlas_consumer|hf_jobs_dispatcher_consumer|predicted_delta_uncertainty|findings_lagrangian" src/tac tools .omx/research
```
