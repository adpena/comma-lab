---
name: Check 82 + 83 STRICT — bug classes "fix-not-at-callsite" + "MPS-derived-decision" extinct
description: 2026-04-29 PM. Round 3 grand council prescribed 3 STRICT preflight checks. Landed 2 in commit 45d808ae; 3rd (empirical-claims-have-evidence) deferred pending legacy-doc tagging cleanup.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## What landed (commit 45d808ae)

**Check 82** — `check_callsite_contracts_satisfied` (preflight.py around line 5728+)
- AST-scans every caller of contract-registered helpers in `src/tac/`, `experiments/`, `scripts/`, `submissions/`
- Registry: `CALLSITE_CONTRACTS = {"reconstruct_poses": {"baseline_poses"}}`
- Exempt: `src/tac/tests/test_pose_gaussian_process.py` (intentionally tests no-baseline path)
- Detects bare-name (`reconstruct_poses(...)`), attribute-access (`gp.reconstruct_poses(...)`), and treats `**kwargs` splat as opaquely-satisfied
- Lands STRICT @ 0 violations
- Sister bug class to silent-default (Check 81): Check 81 catches `argparse default=X` overriding profile values; Check 82 catches `kwarg omitted at call site` defaulting to a stale baked-in value

**Check 83** — `check_no_proxy_metric_drives_decision` (preflight.py around line 5860+)
- Scans `docs/`, `reports/`, `scripts/`, `src/tac/`, `experiments/`, `submissions/`, `.ralph/`, `.omx/`, `BATTLE_PLAN.md`, `PROGRAM.md` (.md, .sh, .py)
- Decision verbs: `GREEN|RED|KILL|killed|promote|promoted|FALSIFIED|FALSIFICATION|dispatched|blessed`
- MPS tokens: `[MPS-PROXY]|MPS-PROXY|MPS-derived|MPS|CPU|advisory only`
- Window: ±10 lines must contain `[contest-CUDA]` tag OR a post-mortem exemption tag
- Post-mortem exempt tags: `[WITHDRAWN]|[POST-MORTEM]|[HISTORICAL]|[ARCHIVED]|[advisory only]|[MPS-PROXY]|WITHDRAWN|POST-MORTEM|FALSIFICATION WITHDRAWN`
- Path exempt: `.omx/context/` (frozen historical), `.omx/research/` (catalog not decisions), `reports/graphs/` (judging surface), `/.claude/projects/` (user-private memory), `/memory/`, `MEMORY.md`
- File exempt: `CLAUDE.md`, `src/tac/preflight.py`, regression test files
- Lands STRICT @ 0 violations after 1-line `docs/hardware_layout.md` cleanup (added `[contest-CUDA]` requirement to validation-rule line)

## What's deferred (Check 84 — task #241)

`check_empirical_claims_have_evidence` was the 3rd prescribed check but needs a legacy-doc tagging sweep first. Spec:
- Scan docs/reports/scripts for `\b(saves|improves|beats|verified|empirical|reduces|achieves)\b.*?(-?\d+(\.\d+)?%?)`
- Require adjacent `[prediction]` OR `[empirical:<path>]` OR `[contest-CUDA]` tag
- Lane PD docstring 49% vs empirical 18.5% is the canonical incident (commit ef8592d9)
- Land warn-only first, sweep legacy claims for tags, promote to STRICT

## Knowledge captured to CLAUDE.md (commit 45d808ae)

3 new FORBIDDEN PATTERNS entries:
1. **Forbidden empirical-claim-without-evidence-tag** (the docstring-overstatement trap) — Lane PD 49% → 18.5%
2. **Forbidden fix-lands-in-helper-but-not-callsite** (the dangling-helper trap) — Lane GP fit_pose_gp.py
3. **Forbidden MPS-derived strategic decision** (the MPS-falsification trap) — STC FALSIFICATION withdrawn

These appear right after the existing FORBIDDEN PATTERNS list so they load with every session.

## Live STRICT preflight check count: 84
(was 81 at session start, now +3 from this loop: 81 silent-default audit, 82 callsite contracts, 83 no-MPS-decision; 84 empirical-claims pending)

## Cross-refs

- Commit: 45d808ae "2 new STRICT preflight checks (82, 83) + CLAUDE.md FORBIDDEN extension"
- Memory: `feedback_three_active_bug_classes_needing_strict_checks_20260429.md` (the council prescription)
- Memory: `feedback_silent_default_bug_class_findings_20260429.md` (Check 81 sister)
- Memory: `feedback_no_local_mps_for_authoritative_kill_or_promote_20260429.md` (Check 83 motivation)
- Memory: `feedback_lane_pd_savings_overstated_in_docstring_20260429.md` (Check 84 motivation)
- Code: `src/tac/preflight.py` (check definitions + `_MPS_DECISION_*` exempt sets)
- Tests: `src/tac/tests/test_callsite_contracts_and_no_mps_decision.py` (13 regression tests)
- CLAUDE.md FORBIDDEN PATTERNS section (the canonical extinction list)
