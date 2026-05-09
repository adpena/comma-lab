# Operator Authorizations — committed audit trail

<!-- generated_at: 2026-05-09T00:00:00Z, from_state_hash: phase_b_option_c_landing_20260509 -->

This directory holds dated, **committed** operator-authorization memos that
gate Phase B (and similar) dispatches for Lane 12-v2 NeRV-as-renderer and
related learned-codec / representation lanes.

Per the **Option C compromise** (operator decision 2026-05-09 via
AskUserQuestion arbitration of codex round 8 HIGH 2 vs a6535b1ed):

- The dispatch-gate scanner in `src/tac/lane_12_v2_nerv_as_renderer.py`
  (`phase_b_preconditions_status`) accepts an explicit
  `auth_memo_path=` argument (or `--phase-b-auth-memo <path>` CLI flag).
- The argument MUST resolve to a path under the git repo root. Anchors
  outside the repo (`~/.claude`, `/tmp`, any non-repo absolute path) are
  REFUSED with `ValueError` per `_assert_auth_memo_path_repo_relative`.
- STRICT preflight Catalog #150
  (`check_phase_b_auth_memo_in_repo`) refuses any caller that passes a
  non-repo-relative `auth_memo_path=` literal (with same-line waiver
  `# PHASE_B_AUTH_MEMO_OK:<reason>` for the legacy `consult_session_state=True`
  fallback path).

## Why this exists

Codex round 8 HIGH 2 flagged that `consult_session_state=True` consulting
`~/.claude` was non-hermetic (machine-dependent) and spoofable (any
`feedback_*.md` body containing `operator_phase_b_authorization=true`
could pass the gate). Round 8 wanted `consult_session_state=False` as
the new default. a6535b1ed had intentionally LANDED `True` because the
gate would otherwise be a permanently-PENDING placeholder.

Operator decision: **Option C compromise** — keep `True` default
(preserves a6535b1ed's "real" gate behavior + back-compat) PLUS accept
explicit `--phase-b-auth-memo <committed_repo_path>` for deterministic +
reviewable authorization. The CLI flag path is the recommended
production pattern; the `~/.claude` scan stays as a back-compat fallback
only.

## Filename convention

```
phase_b_auth_<lane_id>_<YYYYMMDD>.md
```

Examples:
- `phase_b_auth_lane_12_v2_nerv_as_renderer_20260509.md`
- `phase_b_auth_lane_12_v2_phase_b_dispatch_t6_20260510.md`

## Required body token

The memo body MUST contain (line-level, NOT inside a fenced code block,
NOT inside a blockquote, NOT inside backticks):

```
operator_phase_b_authorization=true
```

The same parser `_phase_b_authorization_memo_is_explicit()` that gates
the legacy `~/.claude` scan also gates this path; the rule for what
counts as an "explicit" line-level token is identical.

Recommended additional metadata (operator-discretion, not enforced):

- Date + UTC timestamp of authorization
- Operator handle (e.g., `@adpena`)
- Lane id authorized
- Predicted GPU spend + cost ceiling
- Re-authorization expiry (if applicable)
- Reference to the parent decision context (memory file / commit)

## Commit discipline

The authorization memo MUST be committed to git in the same commit as
(or strictly before) the dispatch invocation that consumes it. Reviewers
of the dispatch PR see the authorization memo + the `--phase-b-auth-memo`
flag in the same diff.

Per CLAUDE.md "Bugs must be permanently fixed AND self-protected
against": Catalog #150 enforces this at preflight time (`check_phase_b_auth_memo_in_repo`).

## Cross-references

- `src/tac/lane_12_v2_nerv_as_renderer.py` — `phase_b_preconditions_status`
  (the consumer)
- `src/tac/preflight.py` — `check_phase_b_auth_memo_in_repo` (Catalog #150)
- Memory: `feedback_phase_b_option_c_landed_20260509.md`
- Directive: `.omx/research/codex_review_round7_round8_findings_directive_20260509.md`
- NOT YET pin (ITEM 4 — to be marked RESOLVED by this landing):
  `project_top_priority_revisit_NOT_YET_operator_decisions_20260509.md`
