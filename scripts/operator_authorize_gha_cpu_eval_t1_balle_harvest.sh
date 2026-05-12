#!/bin/bash
# Thin shim: delegates to the canonical operator-authorize entry point.
#
# Recipe: ``.omx/operator_authorize_recipes/gha_cpu_eval_t1_balle_harvest.yaml``
#
# Per simplification audit `a2a901c4f43d66a74` (operator approved 2026-05-12)
# + Catalog #162 ``check_operator_authorize_canonical_use``.
#
# Env-vars passed through:
#   T1_BALLE_LABEL=<modal_harvest_label>   (required)
#   T1_BALLE_ARCHIVE_PATH=<path>           (optional; Plan B)
#   T1_BALLE_FORK_REPO=<repo>              (default adpena/comma_video_compression_challenge)
#   T1_BALLE_DRY_RUN=1                     (Plan C)
#
# Lane: lane_fix_g_t1c_wrapper_unification_20260512
# Cross-ref: feedback_fix_g_t1c_wrapper_unification_landed_20260512.md

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Phase 1: canonical entry point — banner + lane-claim handled below.
.venv/bin/python tools/operator_authorize.py \
    --recipe gha_cpu_eval_t1_balle_harvest \
    --agent "claude:operator_authorize_gha_cpu_eval_t1_balle_harvest" \
    --no-claim \
    --dry-run

# Phase 2: bespoke action sequence preserved for back-compat.
T1_BALLE_LABEL="${T1_BALLE_LABEL:-}"
if [ -z "$T1_BALLE_LABEL" ]; then
    echo "[gha-cpu-eval-t1-balle] FATAL: T1_BALLE_LABEL must be set" >&2
    echo "  Example: T1_BALLE_LABEL=t1_balle_cheap_config_20260512T171203Z bash $0" >&2
    exit 1
fi

T1_BALLE_DRY_RUN="${T1_BALLE_DRY_RUN:-0}"
T1_BALLE_ARCHIVE_PATH="${T1_BALLE_ARCHIVE_PATH:-}"
T1_BALLE_FORK_REPO="${T1_BALLE_FORK_REPO:-adpena/comma_video_compression_challenge}"
T1_BALLE_MODAL_POLL_DEADLINE_SEC="${T1_BALLE_MODAL_POLL_DEADLINE_SEC:-86400}"
T1_BALLE_MODAL_POLL_INTERVAL_SEC="${T1_BALLE_MODAL_POLL_INTERVAL_SEC:-300}"

echo "[gha-cpu-eval-t1-balle] T1_BALLE_LABEL: ${T1_BALLE_LABEL}"
echo "[gha-cpu-eval-t1-balle] T1_BALLE_FORK_REPO: ${T1_BALLE_FORK_REPO}"
echo "[gha-cpu-eval-t1-balle] T1_BALLE_DRY_RUN: ${T1_BALLE_DRY_RUN}"

read -r -p "Proceed with GHA [contest-CPU] eval queueing (cost \$0)? [y/N] " confirm
case "$confirm" in
    [yY]|[yY][eE][sS]) ;;
    *) echo "[gha-cpu-eval-t1-balle] aborted — no dispatch fired"; exit 0;;
esac

echo "[gha-cpu-eval-t1-balle] running preflight..."
.venv/bin/python -m tac.preflight --scope dev >/dev/null 2>&1 || {
    echo "[gha-cpu-eval-t1-balle] FATAL: preflight failed; resolve before dispatch" >&2
    exit 1
}

if [ -z "$T1_BALLE_ARCHIVE_PATH" ]; then
    EXPECTED_DIR="experiments/results/lane_${T1_BALLE_LABEL}_modal"
    EXPECTED_ARCHIVE="${EXPECTED_DIR}/harvested_archive.zip"
    DEADLINE=$(( $(date +%s) + T1_BALLE_MODAL_POLL_DEADLINE_SEC ))
    while [ ! -f "$EXPECTED_ARCHIVE" ]; do
        NOW=$(date +%s)
        if [ "$NOW" -ge "$DEADLINE" ]; then
            echo "[gha-cpu-eval-t1-balle] FATAL: Modal harvest deadline exceeded" >&2
            exit 6
        fi
        REMAINING=$(( DEADLINE - NOW ))
        echo "[gha-cpu-eval-t1-balle] waiting for ${EXPECTED_ARCHIVE} (deadline in ${REMAINING}s)"
        sleep "$T1_BALLE_MODAL_POLL_INTERVAL_SEC"
    done
    T1_BALLE_ARCHIVE_PATH="$EXPECTED_ARCHIVE"
fi

if [ ! -f "$T1_BALLE_ARCHIVE_PATH" ]; then
    echo "[gha-cpu-eval-t1-balle] FATAL: archive not found: ${T1_BALLE_ARCHIVE_PATH}" >&2
    exit 2
fi

ARCHIVE_SHA=$(.venv/bin/python -c "
import hashlib
h = hashlib.sha256()
with open('${T1_BALLE_ARCHIVE_PATH}', 'rb') as f:
    for chunk in iter(lambda: f.read(1 << 20), b''):
        h.update(chunk)
print(h.hexdigest())
")
ARCHIVE_SIZE=$(wc -c < "$T1_BALLE_ARCHIVE_PATH" | tr -d ' ')

echo "[gha-cpu-eval-t1-balle] sha256: ${ARCHIVE_SHA}"
echo "[gha-cpu-eval-t1-balle] size: ${ARCHIVE_SIZE}"

if [ "$T1_BALLE_DRY_RUN" = "1" ]; then
    echo "[gha-cpu-eval-t1-balle] Plan C (dry-run): skipping release upload + GHA dispatch"
    exit 0
fi

RELEASE_TAG="cpu-eval-${T1_BALLE_LABEL}-$(date -u +%Y%m%dT%H%M%SZ)"
RELEASE_NOTES="Auto-created by scripts/operator_authorize_gha_cpu_eval_t1_balle_harvest.sh
- archive_sha256: ${ARCHIVE_SHA}
- archive_size_bytes: ${ARCHIVE_SIZE}
- dispatched_at_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)
- purpose: T1 Balle [contest-CPU] eval on contest-compliant Linux x86_64 runner"

gh release create "$RELEASE_TAG" \
    -R "$T1_BALLE_FORK_REPO" \
    --title "T1 Balle CPU eval — ${T1_BALLE_LABEL}" \
    --notes "$RELEASE_NOTES" \
    "$T1_BALLE_ARCHIVE_PATH"

ASSET_URL=$(gh release view "$RELEASE_TAG" \
    -R "$T1_BALLE_FORK_REPO" \
    --json assets \
    --jq '.assets[] | select(.name | endswith(".zip")) | .browser_download_url')

if [ -z "$ASSET_URL" ]; then
    echo "[gha-cpu-eval-t1-balle] FATAL: could not read release asset URL" >&2
    exit 4
fi

SUBMISSION_NAME="gha_cpu_${T1_BALLE_LABEL}"
OUTPUT_DIR="experiments/results/gha_cpu_eval_${T1_BALLE_LABEL}_$(date -u +%Y%m%dT%H%M%SZ)"

.venv/bin/python tools/trigger_gha_cpu_eval.py \
    --archive-url "$ASSET_URL" \
    --archive-sha256 "$ARCHIVE_SHA" \
    --archive-size-bytes "$ARCHIVE_SIZE" \
    --label "$T1_BALLE_LABEL" \
    --submission-name "$SUBMISSION_NAME" \
    --lane-id "gha_cpu_eval_t1_balle_${T1_BALLE_LABEL}" \
    --repo "$T1_BALLE_FORK_REPO" \
    --runner ubuntu-latest \
    --output-dir "$OUTPUT_DIR" \
    --agent "claude:operator_authorize_gha_cpu_eval_t1_balle_harvest"

echo "[gha-cpu-eval-t1-balle] dispatch fired"
echo "  release tag: ${RELEASE_TAG}"
echo "  metadata:    ${OUTPUT_DIR}/dispatch_metadata.json"
echo "  next harvest: .venv/bin/python tools/harvest_gha_cpu_eval.py --dispatch-metadata ${OUTPUT_DIR}/dispatch_metadata.json"
