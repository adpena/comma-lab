#!/bin/sh
# Post-commit hook: runs diff-scan on the last commit to update staleness.
# Install (opt-in): ln -sf ../../tools/post_commit_hook.sh .git/hooks/post-commit
REPO_ROOT="$(git rev-parse --show-toplevel)"
[ -x "$REPO_ROOT/.venv/bin/python" ] || exit 0
"$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/tools/review_tracker.py" diff-scan --since HEAD~1 2>/dev/null
