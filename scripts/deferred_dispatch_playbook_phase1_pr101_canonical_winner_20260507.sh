#!/usr/bin/env bash
# Deferred dispatch playbook — Phase 1 of the bilevel optimization.
#
# Per Grand Council 2026-05-07 verdict
# (`.omx/research/grand_council_optimal_path_to_shannon_floor_20260507.md`):
#
#   Phase 1: PR100 + Op1+Op2+Op2.5 stack
#   Predicted score: 0.190 [predicted-band only]
#   Estimated GPU $: 5
#   Wall-clock: 1 day
#
# This playbook is the gate-5 executor (contest-CUDA auth eval) for the
# canonical-winner replay. Use Lightning 4090 (preferred per operator
# update 2026-05-07: "you can use the lightning 4090 for training and
# try T4 for auth eval but use 4090 or others for auth eval just the
# time will be different but time hasn't really been an issue for us")
# OR Lightning T4 (canonical recipe per
# `reference_lightning_studio_canonical_dispatch_recipe_20260505`).
#
# Strategy: replay PR101 (gold) on PR101's NATIVE substrate to verify we
# reproduce 0.193 [contest-CUDA]. If we do, the Op1+Op2 cathedral stack
# is verified end-to-end on a medal-band substrate.
#
# Usage:
#   bash scripts/deferred_dispatch_playbook_phase1_pr101_canonical_winner_20260507.sh --dry-run
#   bash scripts/deferred_dispatch_playbook_phase1_pr101_canonical_winner_20260507.sh --provider lightning-4090
#   bash scripts/deferred_dispatch_playbook_phase1_pr101_canonical_winner_20260507.sh --provider lightning-t4
#   bash scripts/deferred_dispatch_playbook_phase1_pr101_canonical_winner_20260507.sh --provider vastai
#
# Phase 1 expected anchors (public PR #101 body):
#   archive bytes:  178,258
#   reported score: 0.193 [public-PR-claim]
#   substrate:      hnerv_ft_microcodec (PR100 base + schema-driven decoder)

set -euo pipefail

PROVIDER="${1:-}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "${PROVIDER}" == "--dry-run" ]]; then
    DRY_RUN=1
    PROVIDER=""
elif [[ "${PROVIDER}" == "--provider" ]]; then
    PROVIDER="${2:?--provider needs lightning-4090|lightning-t4|vastai|modal}"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# PR101 intake archive (the canonical-winner archive, already in repo)
PR101_INTAKE_DIR="${REPO_ROOT}/experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/source"
EXPECTED_PR101_SCORE_PUBLIC="0.193"
EXPECTED_PR101_BYTES="178258"

echo "[phase1-playbook] Phase 1 dispatch — PR101 canonical-winner replay"
echo "[phase1-playbook] repo_root=${REPO_ROOT}"
echo "[phase1-playbook] pr101_intake_dir=${PR101_INTAKE_DIR}"
echo "[phase1-playbook] expected_score=${EXPECTED_PR101_SCORE_PUBLIC} [public-PR-claim]"
echo "[phase1-playbook] expected_bytes=${EXPECTED_PR101_BYTES}"
echo "[phase1-playbook] provider=${PROVIDER:-<dry-run>}"
echo

# ── Pre-flight: PR101 intake present ─────────────────────────────────────────
if [[ ! -d "${PR101_INTAKE_DIR}" ]]; then
    echo "FATAL: PR101 intake directory missing: ${PR101_INTAKE_DIR}" >&2
    echo "  Run intake of public PR #101 source first." >&2
    exit 2
fi

PR101_ARCHIVE_CANDIDATES=(
    "${PR101_INTAKE_DIR}/submissions/hnerv_ft_microcodec/archive.zip"
    "${PR101_INTAKE_DIR}/archive.zip"
    "${REPO_ROOT}/experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip"
)
PR101_ARCHIVE=""
for cand in "${PR101_ARCHIVE_CANDIDATES[@]}"; do
    if [[ -f "${cand}" ]]; then
        PR101_ARCHIVE="${cand}"
        break
    fi
done

if [[ -z "${PR101_ARCHIVE}" ]]; then
    echo "WARNING: PR101 archive not found at any canonical path:" >&2
    for cand in "${PR101_ARCHIVE_CANDIDATES[@]}"; do
        echo "  - ${cand}" >&2
    done
    echo "[phase1-playbook] continuing in dry-run mode only; need archive build before real dispatch" >&2
    PR101_ARCHIVE="<NEEDS-BUILD>"
fi

if [[ "${PR101_ARCHIVE}" != "<NEEDS-BUILD>" ]]; then
    ACTUAL_BYTES="$(stat -c '%s' "${PR101_ARCHIVE}" 2>/dev/null || stat -f '%z' "${PR101_ARCHIVE}")"
    echo "[phase1-playbook] PR101 archive: ${PR101_ARCHIVE} (${ACTUAL_BYTES} B)"
    if [[ "${ACTUAL_BYTES}" != "${EXPECTED_PR101_BYTES}" ]]; then
        echo "  WARNING: bytes ${ACTUAL_BYTES} != expected ${EXPECTED_PR101_BYTES}" >&2
        echo "  This is OK if intake bundled differently; verify by SHA before scoring as PR101 anchor" >&2
    fi
fi

# ── Pre-flight: lane claim ──────────────────────────────────────────────────
LANE_STATE=claim_unknown
LANE_CLAIM_FILE="${REPO_ROOT}/.omx/state/active_lane_dispatch_claims.md"
if [[ -f "${LANE_CLAIM_FILE}" ]]; then
    if grep -q "phase1_pr101_canonical_winner" "${LANE_CLAIM_FILE}"; then
        LANE_STATE=claim_present
    else
        LANE_STATE=claim_missing
    fi
fi
echo "[phase1-playbook] lane claim state: ${LANE_STATE}"
if [[ "${LANE_STATE}" == claim_missing ]]; then
    echo "[phase1-playbook] claiming lane phase1_pr101_canonical_winner..."
    if [[ "${DRY_RUN}" != "1" ]]; then
        "${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/tools/claim_lane_dispatch.py" claim \
            --lane-id phase1_pr101_canonical_winner \
            --platform "${PROVIDER:-pending}" \
            --instance-job-id "phase1-pending-$(date -u +%Y%m%dT%H%M%SZ)" \
            --agent claude-cathedral-bilevel \
            --status active \
            --notes "Phase 1 PR101 canonical-winner replay — predicted 0.190 [predicted-band only]" \
            --ttl-hours 168 || echo "  (claim may have failed; check tools output)"
    else
        echo "  [dry-run] would claim phase1_pr101_canonical_winner with TTL 168h"
    fi
fi
echo

# ── Dispatch routing ────────────────────────────────────────────────────────
if [[ "${DRY_RUN}" == "1" || -z "${PROVIDER}" ]]; then
    cat <<EOF
[phase1-playbook] DRY RUN — to actually dispatch:

  Lightning 4090 (preferred per operator 2026-05-07; both training + auth eval):
    .venv/bin/python scripts/launch_lightning_batch_job.py exact-eval \\
        --archive "${PR101_ARCHIVE}" \\
        --label phase1_pr101_canonical_winner_20260507 \\
        --teamspace comma-lab --studio lossy-compression-challenge \\
        --gpu RTX_4090 \\
        --platform lightning \\
        --allow-skip-remote-preflight-reason "macos-mixed-case-repo-path"

  Lightning T4 (canonical recipe; faster auth eval but training is slower):
    .venv/bin/python scripts/launch_lightning_batch_job.py exact-eval \\
        --archive "${PR101_ARCHIVE}" \\
        --label phase1_pr101_canonical_winner_20260507 \\
        --teamspace comma-lab --studio lossy-compression-challenge \\
        --gpu T4 \\
        --platform lightning \\
        --allow-skip-remote-preflight-reason "macos-mixed-case-repo-path"

  Vast.ai 4090:
    .venv/bin/python scripts/launch_lane_on_vastai.py \\
        --archive "${PR101_ARCHIVE}" \\
        --label phase1_pr101_canonical_winner_20260507 \\
        --gpu RTX_4090 --max-cost 5.00 --disk 60

  Modal T4 (when billing reloaded):
    .venv/bin/python experiments/modal_train_lane.py \\
        --archive-path "${PR101_ARCHIVE}" \\
        --eval-only --gpu T4 \\
        --label phase1_pr101_canonical_winner_20260507

After dispatch lands [contest-CUDA] score:
  1. Run tools/auto_promote_contest_cuda.py --apply  (auto-mark gates + update reports)
  2. Append to bilevel_atom_ledger.jsonl  (already automatic via auto-promote)
  3. tools/run_bilevel_optimization.py --phase auto  (advance to Phase 2 if score < 0.193)

Predicted outcomes:
  - score ~0.193 [contest-CUDA]: PR101 reproduction successful; cathedral verified on
    canonical-winner substrate. Advance to Phase 2 (RAFT poses).
  - score >0.193: archive bytes drifted vs published; investigate intake integrity.
  - score <0.193: cathedral provided extra wins (auto_select beats hardcoded byte_maps).
EOF
    exit 0
fi

case "${PROVIDER}" in
    lightning-4090)
        echo "[phase1-playbook] launching Lightning 4090..."
        exec "${REPO_ROOT}/.venv/bin/python" \
             "${REPO_ROOT}/scripts/launch_lightning_batch_job.py" exact-eval \
             --archive "${PR101_ARCHIVE}" \
             --label phase1_pr101_canonical_winner_20260507 \
             --teamspace comma-lab --studio lossy-compression-challenge \
             --gpu RTX_4090 \
             --platform lightning \
             --allow-skip-remote-preflight-reason "macos-mixed-case-repo-path"
        ;;
    lightning-t4|lightning)
        echo "[phase1-playbook] launching Lightning T4..."
        exec "${REPO_ROOT}/.venv/bin/python" \
             "${REPO_ROOT}/scripts/launch_lightning_batch_job.py" exact-eval \
             --archive "${PR101_ARCHIVE}" \
             --label phase1_pr101_canonical_winner_20260507 \
             --teamspace comma-lab --studio lossy-compression-challenge \
             --gpu T4 \
             --platform lightning \
             --allow-skip-remote-preflight-reason "macos-mixed-case-repo-path"
        ;;
    vastai)
        echo "[phase1-playbook] launching Vast.ai 4090..."
        exec "${REPO_ROOT}/.venv/bin/python" \
             "${REPO_ROOT}/scripts/launch_lane_on_vastai.py" \
             --archive "${PR101_ARCHIVE}" \
             --label phase1_pr101_canonical_winner_20260507 \
             --gpu RTX_4090 --max-cost 5.00 --disk 60
        ;;
    modal)
        echo "[phase1-playbook] launching Modal T4..."
        exec "${REPO_ROOT}/.venv/bin/python" \
             "${REPO_ROOT}/experiments/modal_train_lane.py" \
             --archive-path "${PR101_ARCHIVE}" \
             --eval-only --gpu T4 \
             --label phase1_pr101_canonical_winner_20260507
        ;;
    *)
        echo "FATAL: unknown provider '${PROVIDER}' (lightning-4090|lightning-t4|vastai|modal)" >&2
        exit 8
        ;;
esac
