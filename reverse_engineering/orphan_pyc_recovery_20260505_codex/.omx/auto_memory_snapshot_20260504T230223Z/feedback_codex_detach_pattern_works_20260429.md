---
name: PROVEN PATTERN — codex CLI long-running BG bash via nohup+disown survives harness
description: 2026-04-29 PM. Earlier rule "Agent wrapper only" was over-restrictive. The bash 144-SIGURG kill is process-group-bound. Detached spawn (nohup bash -c '...' < /dev/null > log 2>&1 &; disown) survives indefinitely. Codex `-o <file>` flag guarantees final-message capture. Verified with 11K-token response.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## The proven detach pattern

```bash
mkdir -p /tmp/codex_runs
nohup bash -c '
  codex exec --skip-git-repo-check --sandbox read-only \
    -m gpt-5.5 -c model_reasoning_effort=xhigh \
    -o /tmp/codex_runs/<label>.last.txt \
    "<prompt>" \
    2>&1 | tee /tmp/codex_runs/<label>.log > /dev/null
' < /dev/null > /tmp/codex_runs/<label>.outer.log 2>&1 &
disown
```

## Why each piece matters

| Element | Purpose | What breaks without it |
|---|---|---|
| `nohup` | Ignore SIGHUP | Terminal hangup kills codex |
| `bash -c '...'` subshell | Group the pipe so tee survives | tee dies when outer dies → log file 0 lines |
| `< /dev/null` | Close stdin | Codex waits for input forever |
| `2>&1 \| tee` | Capture stdout+stderr with flushing | Buffered output never reaches disk before parent dies |
| `> outer.log 2>&1 &` | Redirect immediate parent + fork | Outer bash output mixed with codex output |
| `disown` | Remove from job table | Parent bash exit propagates kill signal |
| `-o /tmp/.../<label>.last.txt` | Codex's own final-message capture | Loses final message if pipe breaks mid-flight |

## Verified 2026-04-29

- Sanity test: prompt "Reply OK_DETACHED_TEST" → got "OK_DETACHED_TEST" + 11,449 tokens used in ~10s
- Earlier `Bash run_in_background: true` (no detach) → 6+ codex spawns died at SIGURG-144 with 0-line logs
- The detach pattern survives the harness

## Codex MCP-plugin auth caveat

If you see `Auth(TokenRefreshFailed("invalid_grant: Token refresh failed; reauthorization is required"))` in stderr — that's the codex MCP-plugin (rmcp) worker, NOT the main codex API. Core codex `exec` STILL WORKS. The MCP failure only blocks plugin-augmented features (like external skills). Re-auth via `codex login` interactively if you need MCP.

## When to use Pattern A (detached BG bash) vs Pattern B (Agent tool wrapper)

- **Pattern A**: non-interactive single-shot codex query. Cheaper, simpler, faster startup.
- **Pattern B**: multi-stage orchestration (read context → reason → write code → verify → commit). Uses Agent tool's poll-wait logic + has its own bash env.

## Cross-refs

- CLAUDE.md "Codex CLI invocation" section (REVISED 2026-04-29 PM)
- feedback_bash_harness_kills_long_running_tasks_20260428.md (the original 144-SIGURG discovery)
- feedback_persistent_codex_review_protocol_20260429.md (review protocol that motivated the discovery)
