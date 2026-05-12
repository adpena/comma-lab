#!/bin/bash
# Operator authorize: push the v0.2.0-rc1 LOCAL git tag to the GitHub remote.
#
# Per operator F-2 in operator decision dashboard 2026-05-11: this is
# FROZEN-OPERATOR pending "don't submit PR yet" + Q5 council verdict on the
# tac-packet-compiler crate publication ecosystem timing.
#
# This script REQUIRES explicit operator confirmation; per CLAUDE.md "Public
# Disclosure Hygiene" + "Submission PR gate", it does NOT auto-push.
#
# Per CLAUDE.md "Operator gates must be wired and used": this script is the
# canonical wrapper for the F-2 transition from FROZEN-OPERATOR to
# OPERATOR-AUTHORIZED.
#
# Cost: $0 (free GitHub push).
# Risk: public-facing release; once pushed, the tag is permanent (force-delete
#       requires GitHub repo admin access + violates immutable-tag conventions).
#
# Usage: bash scripts/operator_authorize_v0_2_0_rc1_github_push.sh
#
# Lane: lane_operator_one_command_authorize_scripts L0
# Cross-ref: project_operator_decision_dashboard_20260511.md F-2

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TAG="v0.2.0-rc1"
REMOTE="${GIT_REMOTE:-origin}"

# Pre-flight: tag must exist locally.
if ! git rev-parse --verify "refs/tags/${TAG}" >/dev/null 2>&1; then
    echo "[v020rc1-push] FATAL: local tag ${TAG} does not exist; nothing to push." >&2
    exit 1
fi
TAG_SHA=$(git rev-list -n 1 "refs/tags/${TAG}")

# Pre-flight: remote tag must NOT exist (or operator wants to skip if it does).
REMOTE_TAG_SHA=""
if git ls-remote --tags "$REMOTE" "refs/tags/${TAG}" 2>/dev/null | grep -q "${TAG}"; then
    REMOTE_TAG_SHA=$(git ls-remote --tags "$REMOTE" "refs/tags/${TAG}" | awk '{print $1}')
fi

cat <<EOF

=== ${TAG} GitHub push operator confirmation ===

local tag sha:           ${TAG_SHA}
remote tag sha:          ${REMOTE_TAG_SHA:-<not present>}
remote:                  ${REMOTE}

This will run:
  git push ${REMOTE} ${TAG}

risk:                    public-facing release. Once pushed, the tag is
                         permanent (force-delete requires admin access).
                         Per CLAUDE.md "Public Disclosure Hygiene", verify:
  - No credentials in the tagged commit's diff
  - No private infrastructure URLs in tracked files
  - LICENSE + THIRD_PARTY_NOTICES.md unchanged from approved baseline
  - check_public_release_hygiene STRICT preflight has been run

cost:                    \$0 (free GitHub push)
envelope status:         FREE — this is an OSS publication act, not a GPU spend.

EOF
if [ -n "$REMOTE_TAG_SHA" ]; then
    echo "WARN: remote tag ${TAG} ALREADY EXISTS at sha ${REMOTE_TAG_SHA}."
    if [ "$REMOTE_TAG_SHA" = "$TAG_SHA" ]; then
        echo "      The local and remote tags match — push would be a no-op."
    else
        echo "      The local and remote tags DIFFER — push would FAIL without --force."
        echo "      Force-pushing tags violates immutable-tag conventions; do NOT use --force."
    fi
    echo ""
fi

read -r -p "Proceed with 'git push ${REMOTE} ${TAG}'? [y/N] " confirm
case "$confirm" in
    [yY]|[yY][eE][sS])
        ;;
    *)
        echo "[v020rc1-push] aborted — no remote push"
        exit 0
        ;;
esac

git push "$REMOTE" "$TAG"
echo "[v020rc1-push] complete — tag ${TAG} pushed to ${REMOTE}"
echo "  next: review the GitHub release page; if needed, draft a release note"
echo "        before the operator authorizes scripts/operator_authorize_crates_io_publish.sh"
