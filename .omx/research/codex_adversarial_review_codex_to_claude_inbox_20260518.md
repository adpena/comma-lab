# Codex adversarial review: Codex to Claude inbox landing

Reviewer: INBOX-LANDING-ADVERSARIAL-REVIEW-20260518
Date: 2026-05-18
Scope: `src/tac/codex_to_claude_inbox.py`, `tools/codex_to_claude_inbox.py`, Catalog #333 preflight, operator briefing wire-in, tests, `.gitignore`, `CLAUDE.md`, and live inbox JSONL rows.

## Findings

### Blocking 1: `ack` can race with answer/default/withdraw and reopen a resolved question

`append_inbox_ack` reads `latest_status_by_event_id()` before acquiring `_inbox_lock`, then appends an ack row whose `status` is whatever it saw before the lock. Because `latest_status_by_event_id()` treats every row, including ack rows, as latest-row-wins for the target event, this interleaving is possible:

1. question is open;
2. ack reads `current=open`;
3. answer/default/withdraw acquires lock and appends terminal status;
4. ack appends last with `status=open`;
5. `latest_status_by_event_id()` reports the question open again.

Relevant code: `src/tac/codex_to_claude_inbox.py:601-622` and `src/tac/codex_to_claude_inbox.py:364-368`.

Block before commit. Fix by putting the ack status read and append under the same inbox lock, and/or by making status derivation ignore ack rows as non-transition events.

### Blocking 2: operator default rows can close a question with a post-hoc arbitrary default

`append_inbox_question` stores `codex_default_if_no_response`, but `append_inbox_operator_default_invoked` only checks that the target question exists, is open, has a deadline, and the deadline has passed. It does not require that the question declared a default, and it does not require `default_used == question["codex_default_if_no_response"]`. The CLI exposes `--default-used` as a free string, and preflight considers the question resolved once the status becomes `operator_default_invoked`.

Relevant code: `src/tac/codex_to_claude_inbox.py:457-463`, `src/tac/codex_to_claude_inbox.py:560-586`, `tools/codex_to_claude_inbox.py:116-122`, and `src/tac/preflight.py:27816-27837`.

Block before commit. The default path is only safe if it proves the exact predeclared default was invoked, or records an explicit non-default override as a separate operator-decision event that does not silently satisfy the normal deadline gate.

### Non-blocking before commit, but should fix before strict-flip: directive-specified waiver/all-lanes surfaces are incomplete

The routing directive specifies a same-line `INBOX_QUESTION_DEADLINE_EXPIRED_OK:<rationale>` waiver and says `tools/all_lanes_preflight.py` should add the inbox gate explicitly. The landed check has no waiver path, and `tools/all_lanes_preflight.py` currently has operator-briefing and canonical-task-status gates but no direct inbox deadline gate. The operator briefing JSON can contain `codex_inbox_summary={"_error": ...}`, but the all-lanes operator-briefing gate only checks dispatch/xray fields, not inbox health.

Relevant code/spec: `.omx/research/codex_routing_directive_codex_to_claude_inbox_bidirectional_channel_20260518.md:186-206`, `src/tac/preflight.py:27781-27854`, `tools/all_lanes_preflight.py:1475-1500`, and `tools/all_lanes_preflight.py:3940-3972`.

This does not block the initial warn-only landing if the two blocking issues above are fixed first, but it should be closed before claiming strict-flip readiness.

## Review Evidence

- `git diff --check` passed.
- Static AST parse passed for `src/tac/codex_to_claude_inbox.py`, `tools/codex_to_claude_inbox.py`, `src/tac/tests/test_codex_to_claude_inbox.py`, and `src/tac/tests/test_operator_briefing.py`.
- No tests were run because the requested scope was focused static/read review and the worktree contains active state/source WIP owned by others.
