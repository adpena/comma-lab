---
name: 3 active bug classes the Round 3 grand council prescribed STRICT preflight checks for
description: 2026-04-29 PM. Round 3 codex final-rigor grand council identified 3 bug classes still active in the codebase that need permanent extinction via STRICT preflight checks. Each has a documented incident from this loop.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## The 3 active bug classes

### 1. claim-without-empirical-verification
**Incident**: Lane PD docstring stated "49% savings vs raw fp16" but empirical regression test caught actual savings at 18.5% (commit ef8592d9). The 49% number was a derivation from theoretical bytes, not from running the actual encoder.

**Why**: docstrings/reports/scripts can claim "saves X / improves Y / beats Z / verified W / empirical V" without a tagged evidence trail. The reader assumes the number is measured; it's actually projected.

**How to apply (check spec)**: Scan docs/reports/scripts for keywords {saves, improves, beats, verified, empirical, reduces, achieves} attached to a numeric value (`-?\d+(\.\d+)?%?`) and require either:
- `[prediction]` tag in the same line — explicit projection
- `[empirical:<path>]` tag in the same line — points to a result artifact
- `[contest-CUDA]` tag — measured by canonical scorer

Without one of those tags adjacent to the claim, fail loud.

**Sister-pattern**: `[MPS-PROXY]` is a tag too but it's *forbidden* for promote/kill decisions (Check D `check_scores_have_lane_tag` already covers MPS contamination of scores).

### 2. fix-lands-in-helper-but-not-callsite
**Incident**: Lane GP Fix-A added `baseline_poses=` kwarg support to `reconstruct_poses()` — but never updated `experiments/fit_pose_gp.py:33` which was the actual call site. The "fix" sat unused for ~2 weeks (commit 8746793e finally landed it).

**Why**: AST-scanning function definitions is easy; AST-scanning every caller for kwarg presence is the harder discipline. Without that, "fix landed" is true at the lib level but false at the deploy level.

**How to apply (check spec)**: Maintain a registry of dangerous helper kwargs:
```python
CALLSITE_CONTRACTS = {
    "tac.pose_gp.reconstruct_poses": {"baseline_poses"},
    # add more as bugs are found
}
```
For each registered helper, AST-scan all callers (anywhere in `experiments/`, `src/tac/`, `scripts/`, `submissions/`) and require the listed kwargs to be present. Fail loud on any caller missing the contract.

**Sister-pattern of**: silent-default check (Check 81). That one catches `argparse default=X` overriding profile values; this one catches `kwarg omitted` defaulting to a stale value baked into the helper.

### 3. MPS-derived-strategic-decision
**Incident**: STC clean-source pipeline was FALSIFIED based on local MPS encoder argmax. User correctly objected: "MPS is trash and nowhere close to auth eval" → CLAUDE.md non-negotiable. Withdrawn (commit cc1ba193).

**Why**: MPS-vs-CUDA drift is documented at PoseNet 23×, SegNet 2×, score 2.5×. Any GREEN/RED/KILL/promote/falsify decision derived from MPS is invalid. The kill/promote tag has 0 epistemic weight without a contest-CUDA artifact.

**How to apply (check spec)**: Scan docs/reports/findings/run_log/memory for decision verbs `{GREEN, RED, KILL, killed, promote, promoted, falsified, FALSIFICATION, dispatched, blessed}` in the same paragraph as `{MPS, CPU, [MPS-PROXY], MPS-PROXY, advisory only}`. Require a nearby `[contest-CUDA]` tag (within ±10 lines or same section heading). Without it, fail loud.

**Edge case**: MPS results CAN be cited as "smoke test passed" / "compiles" / "doesn't crash" without contest-CUDA. The forbidden pattern is *strategic decision*, not *operational status*. The check needs a verb-list filter.

## Why all 3 land STRICT directly at 0 violations

Lane PD docstring: just fixed, claim is now `~18.5% empirical (regression test ef8592d9)`.
Lane GP fit_pose_gp.py:33: just fixed, callsite now passes `baseline_poses=baseline`.
STC clean-source FALSIFICATION: just withdrawn (commit cc1ba193).

So the live-violation count for all 3 should be 0 right after these fixes land — meeting the canonical "Lane A → strict directly at 0" pattern (CLAUDE.md preflight promotion path).

## Where they fit in preflight.py

Wire after `check_silent_default_audit_clean` (line 321) in `preflight_all()`:
```python
# 2026-04-29 Round 3 grand council prescription: 3 active bug classes.
check_empirical_claims_have_evidence(strict=True, verbose=verbose)
check_callsite_contracts_satisfied(strict=True, verbose=verbose)
check_no_proxy_metric_drives_decision(strict=True, verbose=verbose)
```

Each check function lands in `preflight.py` after `check_silent_default_audit_clean` (line 5725).

## Cross-refs

- /tmp/codex_runs/phase1_final_extreme_rigor.prompt.txt (Round 3 prescription)
- feedback_silent_default_bug_class_findings_20260429.md (sister bug class)
- feedback_lane_pd_savings_overstated_in_docstring_20260429.md (incident #1)
- project_lane_gp_v3_landed_runge_phenomenon_20260429.md (incident #2 sibling)
- project_lane_stc_clean_source_FALSIFIED_20260429.md (incident #3 — withdrawn)
- CLAUDE.md FORBIDDEN PATTERNS section (the canonical extinction list)
