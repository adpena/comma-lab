# oh-my-codex + Ralph runbook

This repo is designed for a Ralph-style loop with oh-my-codex.

## Recommended path

1. Run `bash start.sh`.
2. Let the script install local tooling and bootstrap the upstream challenge repo.
3. When OMX opens, paste `PROMPT.md`.
4. Let the loop work from disk state, not from fragile chat memory.

## Why this layout exists

- `.omx/` holds durable state, plans, findings, logs, and next actions.
- `.ralph/` holds the file-based run log and any task-specific resumable state.
- `.agents/tasks/` is the seeded home for task/PRD-style JSON and related instructions.
- `AGENTS.md` defines the mutation frontier and required evidence discipline.
- `PROGRAM.md` defines the operating constitution.

## Default posture

- One owner first: prefer a single Ralph loop.
- Add teams only when the work clearly benefits from parallelism.
- Keep both submission tracks alive.
- Prefer measured scores over taste.
- Leave the repo resumable after every meaningful stop.
