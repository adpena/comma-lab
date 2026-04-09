#!/usr/bin/env bash
# auto_commit.sh — Scoop up and commit all changed tracked files
#
# Run this periodically or after each work cycle to ensure nothing
# stays uncommitted. Groups changes by directory for readable history.
#
# Usage:
#   ./tools/auto_commit.sh              # commit everything changed
#   ./tools/auto_commit.sh --dry-run    # show what would be committed

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

timestamp=$(date +"%Y-%m-%d %H:%M")

# Check for changes
if git diff --quiet HEAD -- && [ -z "$(git ls-files --others --exclude-standard)" ]; then
    echo "Nothing to commit."
    exit 0
fi

# Group 1: Research state (.ralph, .omx)
changed_state=$(git diff --name-only HEAD -- .ralph/ .omx/state/ .omx/research/ .omx/notepad.md .omx/metrics.json 2>/dev/null || true)
if [ -n "$changed_state" ]; then
    if $DRY_RUN; then
        echo "[DRY RUN] Would commit research state: $changed_state"
    else
        git add .ralph/ .omx/state/ .omx/research/ .omx/notepad.md .omx/metrics.json 2>/dev/null || true
        git commit -m "state: auto-capture research state at $timestamp

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>" 2>/dev/null || true
    fi
fi

# Group 2: Docs and reports
changed_docs=$(git diff --name-only HEAD -- docs/ reports/ 2>/dev/null || true)
if [ -n "$changed_docs" ]; then
    if $DRY_RUN; then
        echo "[DRY RUN] Would commit docs/reports: $changed_docs"
    else
        git add docs/ reports/ 2>/dev/null || true
        git commit -m "docs: auto-capture documentation updates at $timestamp

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>" 2>/dev/null || true
    fi
fi

# Group 3: Experiments and training
changed_exp=$(git diff --name-only HEAD -- experiments/ src/ 2>/dev/null || true)
if [ -n "$changed_exp" ]; then
    if $DRY_RUN; then
        echo "[DRY RUN] Would commit experiments/src: $changed_exp"
    else
        git add experiments/ src/ 2>/dev/null || true
        git commit -m "experiments: auto-capture experiment updates at $timestamp

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>" 2>/dev/null || true
    fi
fi

# Group 4: Submission changes
changed_sub=$(git diff --name-only HEAD -- submissions/ 2>/dev/null || true)
if [ -n "$changed_sub" ]; then
    if $DRY_RUN; then
        echo "[DRY RUN] Would commit submissions: $changed_sub"
    else
        git add submissions/ 2>/dev/null || true
        git commit -m "submissions: auto-capture submission updates at $timestamp

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>" 2>/dev/null || true
    fi
fi

# Group 5: Tooling and config
changed_tools=$(git diff --name-only HEAD -- tools/ CLAUDE.md configs/ 2>/dev/null || true)
new_tools=$(git ls-files --others --exclude-standard -- tools/ CLAUDE.md configs/ 2>/dev/null || true)
if [ -n "$changed_tools" ] || [ -n "$new_tools" ]; then
    if $DRY_RUN; then
        echo "[DRY RUN] Would commit tools/config: $changed_tools $new_tools"
    else
        git add tools/ CLAUDE.md configs/ 2>/dev/null || true
        git commit -m "tools: auto-capture tooling updates at $timestamp

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>" 2>/dev/null || true
    fi
fi

# Group 6: Anything else that's tracked and changed
remaining=$(git diff --name-only HEAD 2>/dev/null || true)
new_remaining=$(git ls-files --others --exclude-standard 2>/dev/null | grep -v "^experiments/cache/" | grep -v "\.pt$" | head -20 || true)
if [ -n "$remaining" ] || [ -n "$new_remaining" ]; then
    if $DRY_RUN; then
        echo "[DRY RUN] Would commit remaining: $remaining $new_remaining"
    else
        git add -u 2>/dev/null || true
        # Add new files but skip large binaries
        echo "$new_remaining" | while read -r f; do
            [ -n "$f" ] && git add "$f" 2>/dev/null || true
        done
        git commit -m "misc: auto-capture remaining changes at $timestamp

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>" 2>/dev/null || true
    fi
fi

echo "Done. Recent commits:"
git log --oneline -5
