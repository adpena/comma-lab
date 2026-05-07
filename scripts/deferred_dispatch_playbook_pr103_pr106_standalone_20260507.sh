#!/usr/bin/env bash
# Deferred dispatch playbook for the PR103-on-PR106 standalone candidate.
#
# This script is the gate-5 executor (contest-CUDA auth eval) for the
# 5-gate promotion ladder defined in
# `.omx/research/grand_council_pr106_substrate_findings_zig_default_20260507.md`:
#
#   1. runtime packet         OK  submissions/pr103_pr106_final_runtime/
#   2. brotli/constriction    OK  pyproject + uv.lock both confirmed
#   3. strict static compliance OK  pre_submission_compliance.static.json
#   4. lane claim             OK  pr103_pr106_standalone (ttl 168h)
#   5. contest-CUDA auth eval --  THIS SCRIPT IS THE EXECUTOR
#
# Usage:
#   bash scripts/deferred_dispatch_playbook_pr103_pr106_standalone_20260507.sh --dry-run
#   bash scripts/deferred_dispatch_playbook_pr103_pr106_standalone_20260507.sh --provider lightning
#   bash scripts/deferred_dispatch_playbook_pr103_pr106_standalone_20260507.sh --provider vastai
#
# Candidate evidence:
#   source bytes:    186,239 (current PR106 frontier baseline)
#   candidate bytes: 185,578 (-661 / -0.355%)
#   candidate path:  experiments/results/pr103_repack_pr106_standalone_20260507/archive.zip
#   candidate sha:   ec0890c2d2317dcad903ed37ffddb2794cd19c1df9effa057cb7f05af205e1ce
#   ac_fallback_set: [] (correctly inert on raw PR106)
#   score claim:     false (gate 5 will produce the contest-CUDA score)

set -euo pipefail

PROVIDER="${1:-}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "${PROVIDER}" == "--dry-run" ]]; then
    DRY_RUN=1
    PROVIDER=""
elif [[ "${PROVIDER}" == "--provider" ]]; then
    PROVIDER="${2:?--provider needs lightning|vastai|modal|azure}"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVE="${REPO_ROOT}/experiments/results/pr103_repack_pr106_standalone_20260507/archive.zip"
EXPECTED_SHA="ec0890c2d2317dcad903ed37ffddb2794cd19c1df9effa057cb7f05af205e1ce"
EXPECTED_BYTES="185578"
RUNTIME_DIR="${REPO_ROOT}/submissions/pr103_pr106_final_runtime"

echo "[playbook] PR103-on-PR106 standalone candidate dispatch"
echo "[playbook] repo_root=${REPO_ROOT}"
echo "[playbook] archive=${ARCHIVE}"
echo "[playbook] expected_sha=${EXPECTED_SHA}"
echo "[playbook] expected_bytes=${EXPECTED_BYTES}"
echo "[playbook] runtime_dir=${RUNTIME_DIR}"
echo "[playbook] provider=${PROVIDER:-<dry-run>}"
echo

# Pre-flight: candidate archive bytes + sha
if [[ ! -f "${ARCHIVE}" ]]; then
    echo "FATAL: candidate archive missing: ${ARCHIVE}" >&2
    echo "Re-run tools/prove_pr103_pr106_runtime_closure.py to regenerate." >&2
    exit 2
fi

ACTUAL_BYTES="$(stat -c '%s' "${ARCHIVE}" 2>/dev/null || stat -f '%z' "${ARCHIVE}")"
if [[ "${ACTUAL_BYTES}" != "${EXPECTED_BYTES}" ]]; then
    echo "FATAL: archive bytes drift -- expected ${EXPECTED_BYTES}, got ${ACTUAL_BYTES}" >&2
    exit 3
fi

if command -v sha256sum &>/dev/null; then
    ACTUAL_SHA="$(sha256sum "${ARCHIVE}" | awk '{print $1}')"
elif command -v shasum &>/dev/null; then
    ACTUAL_SHA="$(shasum -a 256 "${ARCHIVE}" | awk '{print $1}')"
else
    echo "FATAL: neither sha256sum nor shasum available" >&2
    exit 4
fi

if [[ "${ACTUAL_SHA}" != "${EXPECTED_SHA}" ]]; then
    echo "FATAL: archive sha drift -- expected ${EXPECTED_SHA}, got ${ACTUAL_SHA}" >&2
    exit 5
fi

echo "[playbook] OK candidate archive ${ACTUAL_BYTES} B / ${ACTUAL_SHA:0:16}"

# Pre-flight: lane claim presence
LANE_STATE=claim_unknown
if [[ -f "${REPO_ROOT}/.omx/state/active_lane_dispatch_claims.md" ]]; then
    if grep -q pr103_pr106_standalone "${REPO_ROOT}/.omx/state/active_lane_dispatch_claims.md"; then
        LANE_STATE=claim_present
    else
        LANE_STATE=claim_missing
    fi
fi
echo "[playbook] lane claim state: ${LANE_STATE}"
if [[ "${LANE_STATE}" == claim_missing ]]; then
    echo "WARNING: lane claim absent; re-claim before dispatch" >&2
fi
echo

if [[ "${DRY_RUN}" == "1" || -z "${PROVIDER}" ]]; then
    cat <<EOF
[playbook] DRY RUN -- to actually dispatch:

  Lightning T4:
    .venv/bin/python scripts/launch_lightning_batch_job.py exact-eval \\
        --archive "${ARCHIVE}" \\
        --label pr103_pr106_standalone_20260507 \\
        --runtime-tree "${RUNTIME_DIR}" \\
        --teamspace comma-lab --studio lossy-compression-challenge \\
        --platform lightning \\
        --allow-skip-remote-preflight-reason "macos-mixed-case-repo-path"

  Vast.ai 4090:
    .venv/bin/python scripts/launch_lane_on_vastai.py \\
        --archive "${ARCHIVE}" \\
        --label pr103_pr106_standalone_20260507 \\
        --runtime-tree "${RUNTIME_DIR}" \\
        --gpu RTX_4090 --max-cost 5.00 --disk 60

  Modal T4 (when billing reloaded):
    .venv/bin/python experiments/modal_train_lane.py \\
        --archive-path "${ARCHIVE}" \\
        --eval-only --gpu T4 --label pr103_pr106_standalone_20260507

After dispatch lands [contest-CUDA] score:
  1. tools/lane_maturity.py mark <lane> --gate contest_cuda --evidence "<score> <eval_artifact>"
  2. Update reports/latest.md with the new frontier
  3. Submit PR if the score beats current public leaderboard
EOF
    exit 0
fi

case "${PROVIDER}" in
    lightning)
        exec "${REPO_ROOT}/.venv/bin/python" \
             "${REPO_ROOT}/scripts/launch_lightning_batch_job.py" exact-eval \
             --archive "${ARCHIVE}" \
             --label pr103_pr106_standalone_20260507 \
             --runtime-tree "${RUNTIME_DIR}" \
             --teamspace comma-lab --studio lossy-compression-challenge \
             --platform lightning \
             --allow-skip-remote-preflight-reason "macos-mixed-case-repo-path"
        ;;
    vastai)
        exec "${REPO_ROOT}/.venv/bin/python" \
             "${REPO_ROOT}/scripts/launch_lane_on_vastai.py" \
             --archive "${ARCHIVE}" \
             --label pr103_pr106_standalone_20260507 \
             --runtime-tree "${RUNTIME_DIR}" \
             --gpu RTX_4090 --max-cost 5.00 --disk 60
        ;;
    modal)
        exec "${REPO_ROOT}/.venv/bin/python" \
             "${REPO_ROOT}/experiments/modal_train_lane.py" \
             --archive-path "${ARCHIVE}" \
             --eval-only --gpu T4 \
             --label pr103_pr106_standalone_20260507
        ;;
    azure)
        echo "Azure dispatch wiring lives in scripts/azure_*.sh; consult tools/azure_status.py" >&2
        exit 7
        ;;
    *)
        echo "FATAL: unknown provider '${PROVIDER}' (lightning|vastai|modal|azure)" >&2
        exit 8
        ;;
esac
