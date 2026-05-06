#!/usr/bin/env bash
# Draft `gh pr create` invocation for the comma video compression challenge
# "best write-up" prize submission.
#
# This script is the operator's one-liner; it is NOT auto-fired. Review
# WRITEUP.md, decide whether the contest organizers are still accepting
# write-up submissions post-deadline (the prize line in the contest README
# does not specify a separate deadline for the write-up prize beyond the
# main "submit by May, 3rd 2026 11:59pm AOE" line — verify before running),
# then execute manually:
#
#   bash submissions/apogee/PR_ONE_LINER.sh
#
# Pre-conditions:
#   - You have a fork of comma_video_compression_challenge on GitHub.
#   - You are on a branch in that fork dedicated to this PR.
#   - submissions/apogee/WRITEUP.md is the body content.

set -euo pipefail

# Adjust the head ref to your fork + branch as appropriate.
HEAD_REF="${HEAD_REF:-adpena:writeup-apogee}"

gh pr create \
    --repo commaai/comma_video_compression_challenge \
    --base master \
    --head "${HEAD_REF}" \
    --title "Apogee write-up: gradient-bug post-mortem + game-theoretic floor + closed-loop dispatch toolchain" \
    --body-file submissions/apogee/WRITEUP.md
