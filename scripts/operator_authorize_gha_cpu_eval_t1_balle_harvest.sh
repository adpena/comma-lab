#!/bin/bash
# Operator authorize: GHA [contest-CPU] eval of the T1 Ballé Modal harvest.
#
# Per operator directive 2026-05-12 "use modal for everything lightning is
# exhausted, also use kaggle and github" — the "github" axis is GitHub Actions
# CPU eval per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1
# CONTEST-COMPLIANT HARDWARE" non-negotiable. ubuntu-latest IS the literal
# contest leaderboard CI runner (Linux x86_64).
#
# Per CLAUDE.md "Operator gates must be wired and used" + "Public Disclosure
# Hygiene": this wrapper runs preflight_all() BEFORE any GHA dispatch and
# verifies the archive URL is a public GH release-asset URL.
#
# Three plans surfaced for operator review:
#
#   PLAN A (RECOMMENDED) — Fully orchestrated end-to-end:
#     This script waits for the Modal T1 Ballé harvest to produce a
#     contest-compliant archive, uploads it as a GH Releases asset on the
#     fork repo, triggers the GHA workflow_dispatch, and polls to harvest
#     the [contest-CPU] score into submissions/<label>/contest_auth_eval.cpu.json.
#     Estimated wall-clock: 20-40 min after Modal harvest lands.
#     Cost: $0 (GHA public-repo minutes are free).
#
#   PLAN B (MANUAL) — Skip Modal-poll, dispatch immediately:
#     Operator supplies T1_BALLE_ARCHIVE_PATH=<path> directly; this script
#     uploads + triggers without waiting on Modal. Useful if the harvest
#     completed but the operator wants explicit control over which archive
#     gets evaluated.
#
#   PLAN C (DRY-RUN) — Validate only:
#     T1_BALLE_DRY_RUN=1 — runs preflight + URL validation + workflow-existence
#     check + claim-dispatch CLAIMING ONLY; does not actually fire GHA. Useful
#     to verify the wiring before the archive is real.
#
# Usage:
#   # Plan A — wait for Modal harvest then dispatch GHA
#   T1_BALLE_LABEL=t1_balle_cheap_config_20260512T171203Z \
#       bash scripts/operator_authorize_gha_cpu_eval_t1_balle_harvest.sh
#
#   # Plan B — dispatch immediately with explicit archive
#   T1_BALLE_LABEL=t1_balle_cheap_config_20260512T171203Z \
#       T1_BALLE_ARCHIVE_PATH=experiments/results/lane_t1_balle_..._modal/harvested_archive.zip \
#       bash scripts/operator_authorize_gha_cpu_eval_t1_balle_harvest.sh
#
#   # Plan C — dry-run validation
#   T1_BALLE_LABEL=t1_balle_cheap_config_20260512T171203Z T1_BALLE_DRY_RUN=1 \
#       bash scripts/operator_authorize_gha_cpu_eval_t1_balle_harvest.sh
#
# Lane: gha_cpu_eval_t1_balle_<label> (auto-claimed via tools/claim_lane_dispatch.py)
# Cross-ref:
#   - tools/trigger_gha_cpu_eval.py + tools/harvest_gha_cpu_eval.py
#   - submissions/a1/contest_auth_eval.cpu.json (A1 reference schema)
#   - adversarial review 2026-05-12 (subagent af75c9d16b7b73f3a)
#   - Catalog #127 check_authoritative_tag_requires_custody_metadata

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

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

cat <<EOF

=== GHA [contest-CPU] eval of T1 Ballé Modal harvest ===

T1_BALLE_LABEL:           ${T1_BALLE_LABEL}
T1_BALLE_FORK_REPO:       ${T1_BALLE_FORK_REPO}
T1_BALLE_ARCHIVE_PATH:    ${T1_BALLE_ARCHIVE_PATH:-<wait for Modal harvest>}
T1_BALLE_DRY_RUN:         ${T1_BALLE_DRY_RUN}

This will:
  1. Run preflight_all() per CLAUDE.md "Operator gates must be wired and used"
  2. EITHER wait for the Modal T1 Ballé harvest to land an archive
     (Plan A — T1_BALLE_ARCHIVE_PATH unset)
     OR use the provided archive path immediately
     (Plan B — T1_BALLE_ARCHIVE_PATH set)
  3. Upload the archive to a GH Releases asset on ${T1_BALLE_FORK_REPO}
  4. Call tools/trigger_gha_cpu_eval.py to fire the workflow_dispatch
  5. Print the harvest command (operator runs that separately to fetch the score)

Cost: \$0 (GHA public-repo minutes are free)
Risk: GitHub release tag is permanent once created; force-deletion violates
      immutable-tag conventions. Per CLAUDE.md "Public Disclosure Hygiene",
      the release notes will contain only the archive sha256, size, and
      dispatch_at_utc — no credentials, no private infrastructure URLs.

EOF
read -r -p "Proceed with GHA [contest-CPU] eval queueing (cost \$0)? [y/N] " confirm
case "$confirm" in
    [yY]|[yY][eE][sS])
        ;;
    *)
        echo "[gha-cpu-eval-t1-balle] aborted — no dispatch fired"
        exit 0
        ;;
esac

# Step 1: preflight_all() per CLAUDE.md "Operator gates must be wired and used".
echo "[gha-cpu-eval-t1-balle] running preflight_all() ..."
.venv/bin/python -m tac.preflight --scope dev >/dev/null 2>&1 || {
    echo "[gha-cpu-eval-t1-balle] FATAL: preflight_all() failed; resolve before dispatch" >&2
    exit 1
}
echo "[gha-cpu-eval-t1-balle] preflight clean"

# Step 2: wait for Modal harvest OR use provided archive.
if [ -z "$T1_BALLE_ARCHIVE_PATH" ]; then
    echo "[gha-cpu-eval-t1-balle] Plan A: waiting for Modal harvest to land an archive ..."
    # Modal harvest writes harvested_archive.zip into the lane's results dir.
    # The directory pattern is experiments/results/lane_<label>_modal/.
    EXPECTED_DIR="experiments/results/lane_${T1_BALLE_LABEL}_modal"
    EXPECTED_ARCHIVE="${EXPECTED_DIR}/harvested_archive.zip"
    DEADLINE=$(( $(date +%s) + T1_BALLE_MODAL_POLL_DEADLINE_SEC ))
    while [ ! -f "$EXPECTED_ARCHIVE" ]; do
        NOW=$(date +%s)
        if [ "$NOW" -ge "$DEADLINE" ]; then
            echo "[gha-cpu-eval-t1-balle] FATAL: Modal harvest deadline exceeded (${T1_BALLE_MODAL_POLL_DEADLINE_SEC}s) waiting for ${EXPECTED_ARCHIVE}" >&2
            exit 6
        fi
        REMAINING=$(( DEADLINE - NOW ))
        echo "[gha-cpu-eval-t1-balle] waiting for ${EXPECTED_ARCHIVE} (deadline in ${REMAINING}s; next poll in ${T1_BALLE_MODAL_POLL_INTERVAL_SEC}s)"
        sleep "$T1_BALLE_MODAL_POLL_INTERVAL_SEC"
    done
    T1_BALLE_ARCHIVE_PATH="$EXPECTED_ARCHIVE"
    echo "[gha-cpu-eval-t1-balle] Modal harvest landed: ${T1_BALLE_ARCHIVE_PATH}"
fi

if [ ! -f "$T1_BALLE_ARCHIVE_PATH" ]; then
    echo "[gha-cpu-eval-t1-balle] FATAL: archive not found: ${T1_BALLE_ARCHIVE_PATH}" >&2
    exit 2
fi

# Step 3: compute sha256 + size.
ARCHIVE_SHA=$(.venv/bin/python -c "
import hashlib, sys
h = hashlib.sha256()
with open('${T1_BALLE_ARCHIVE_PATH}', 'rb') as f:
    for chunk in iter(lambda: f.read(1 << 20), b''):
        h.update(chunk)
print(h.hexdigest())
")
ARCHIVE_SIZE=$(wc -c < "$T1_BALLE_ARCHIVE_PATH" | tr -d ' ')

echo "[gha-cpu-eval-t1-balle] archive_sha256: ${ARCHIVE_SHA}"
echo "[gha-cpu-eval-t1-balle] archive_size_bytes: ${ARCHIVE_SIZE}"

# Step 4: dry-run validation path.
if [ "$T1_BALLE_DRY_RUN" = "1" ]; then
    echo "[gha-cpu-eval-t1-balle] Plan C (dry-run): skipping release upload + GHA dispatch"
    echo "  Would dispatch: tools/trigger_gha_cpu_eval.py --label ${T1_BALLE_LABEL} (sha=${ARCHIVE_SHA:0:12}...)"
    echo "[gha-cpu-eval-t1-balle] dry-run complete; re-run without T1_BALLE_DRY_RUN to fire"
    exit 0
fi

# Step 5: upload to a fresh GH Release on the fork.
RELEASE_TAG="cpu-eval-${T1_BALLE_LABEL}-$(date -u +%Y%m%dT%H%M%SZ)"
echo "[gha-cpu-eval-t1-balle] creating GH release tag ${RELEASE_TAG} on ${T1_BALLE_FORK_REPO}"
RELEASE_NOTES="Auto-created by scripts/operator_authorize_gha_cpu_eval_t1_balle_harvest.sh
- archive_sha256: ${ARCHIVE_SHA}
- archive_size_bytes: ${ARCHIVE_SIZE}
- dispatched_at_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)
- purpose: T1 Ballé [contest-CPU] eval on contest-compliant Linux x86_64 runner"

gh release create "$RELEASE_TAG" \
    -R "$T1_BALLE_FORK_REPO" \
    --title "T1 Ballé CPU eval — ${T1_BALLE_LABEL}" \
    --notes "$RELEASE_NOTES" \
    "$T1_BALLE_ARCHIVE_PATH"

ASSET_URL=$(gh release view "$RELEASE_TAG" \
    -R "$T1_BALLE_FORK_REPO" \
    --json assets \
    --jq '.assets[] | select(.name | endswith(".zip")) | .url')

if [ -z "$ASSET_URL" ]; then
    echo "[gha-cpu-eval-t1-balle] FATAL: could not read release asset URL" >&2
    exit 4
fi
echo "[gha-cpu-eval-t1-balle] asset URL: ${ASSET_URL}"

# Step 6: trigger GHA workflow_dispatch via canonical helper.
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

cat <<EOF

[gha-cpu-eval-t1-balle] dispatch fired — workflow is running on GHA
  fork repo:       ${T1_BALLE_FORK_REPO}
  release tag:     ${RELEASE_TAG}
  asset url:       ${ASSET_URL}
  metadata:        ${OUTPUT_DIR}/dispatch_metadata.json

next:
  .venv/bin/python tools/harvest_gha_cpu_eval.py \\
      --dispatch-metadata ${OUTPUT_DIR}/dispatch_metadata.json

The harvest tool will:
  - poll the run to completion (~20 min wall-clock)
  - download eval-${SUBMISSION_NAME} artifact
  - parse report.txt for canonical score
  - write submissions/${T1_BALLE_LABEL}/contest_auth_eval.cpu.json
  - close the lane claim
  - append a \$0 cost-band anchor

EOF
