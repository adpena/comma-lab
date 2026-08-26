#!/bin/bash
# commit_autosha.sh — the least-hand-typing front door for the canonical commit serializer.
#
# Usage:  tools/commit_autosha.sh "<message>" <file> [<file> ...]
#
# Computes each file's POST-EDIT working-tree sha256 immediately before the
# serializer call and passes it as --expected-content-sha256, exactly the
# canonical contract (declare the content you intend to commit, verified at
# lock-acquire). This replaces the hand-typed SHA=$(shasum ...) dance
# (operator 2026-08-15: "We should be hand typing the least amount possible").
#
# CAVEAT (unchanged semantics, stated honestly): hashing at call time matches
# the long-standing inline practice, so the sister-collision window is the same
# milliseconds it always was; the FIX-1 concurrent-edit check and Catalog #340
# sister-checkpoint guard remain the real absorption protections. For a long
# gap between editing and committing, capture SHAs at edit time and call the
# serializer explicitly.
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "usage: $0 \"<commit message>\" <file> [<file> ...]" >&2
    exit 2
fi

MSG="$1"
shift

ARGS=()
for f in "$@"; do
    if [ ! -f "$f" ]; then
        echo "refuse: '$f' is not a regular file (deletions/renames need the explicit serializer call)" >&2
        exit 3
    fi
    SHA=$(shasum -a 256 "$f" | awk '{print $1}')
    ARGS+=(--expected-content-sha256 "$f=$SHA")
done

exec .venv/bin/python tools/subagent_commit_serializer.py \
    --message "$MSG" --files "$@" "${ARGS[@]}"  # ARGS carries per-file --expected-content-sha256 pairs built above
