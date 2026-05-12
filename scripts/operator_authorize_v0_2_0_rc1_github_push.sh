#!/bin/bash
# Thin shim: delegates to the canonical operator-authorize entry point.
#
# Recipe: ``.omx/operator_authorize_recipes/v0_2_0_rc1_github_push.yaml``
#
# Per simplification audit `a2a901c4f43d66a74` (operator approved 2026-05-12)
# + Catalog #162 ``check_operator_authorize_canonical_use``.
#
# The canonical entry point displays the cost-band banner, runs preflight, and
# manages the lane claim. The bespoke ``git push <remote> <tag>`` action is
# preserved below as the operator-facing sequence (recipe platform=none means
# the canonical entry point's _dispatch_noop only prints a notice).
#
# Lane: lane_fix_g_t1c_wrapper_unification_20260512
# Cross-ref: feedback_fix_g_t1c_wrapper_unification_landed_20260512.md

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Phase 1: canonical entry point — recipe banner only. This platform=none
# wrapper owns the real action prompt and must not create phantom claims.
.venv/bin/python tools/operator_authorize.py \
    --recipe v0_2_0_rc1_github_push \
    --agent "claude:operator_authorize_v0_2_0_rc1_github_push" \
    --no-claim \
    --dry-run \
    "$@" || exit $?

# Phase 2: bespoke action sequence preserved for back-compat.
TAG="v0.2.0-rc1"
REMOTE="${GIT_REMOTE:-origin}"

if ! git rev-parse --verify "refs/tags/${TAG}" >/dev/null 2>&1; then
    echo "[v020rc1-push] FATAL: local tag ${TAG} does not exist; nothing to push." >&2
    exit 1
fi
TAG_SHA=$(git rev-list -n 1 "refs/tags/${TAG}")

REMOTE_TAG_SHA=""
REMOTE_TAGS=$(git ls-remote --tags "$REMOTE" "refs/tags/${TAG}" 2>/dev/null || true)
if echo "$REMOTE_TAGS" | grep -q "${TAG}"; then
    REMOTE_TAG_SHA=$(echo "$REMOTE_TAGS" | awk '{print $1}')
fi

echo "[v020rc1-push] local tag sha:  ${TAG_SHA}"
echo "[v020rc1-push] remote tag sha: ${REMOTE_TAG_SHA:-<not present>}"

if [ -n "$REMOTE_TAG_SHA" ]; then
    if [ "$REMOTE_TAG_SHA" = "$TAG_SHA" ]; then
        echo "[v020rc1-push] local and remote tags match — push would be a no-op"
    else
        echo "[v020rc1-push] WARN: local and remote tags DIFFER — push would FAIL without --force"
    fi
fi

read -r -p "Proceed with 'git push ${REMOTE} ${TAG}'? [y/N] " confirm
case "$confirm" in
    [yY]|[yY][eE][sS])
        git push "$REMOTE" "$TAG"
        echo "[v020rc1-push] complete — tag ${TAG} pushed to ${REMOTE}"
        ;;
    *)
        echo "[v020rc1-push] aborted — no remote push"
        exit 0
        ;;
esac
