#!/usr/bin/env bash
# Deferred-dispatch playbook — fires the public top-3 1:1 replay batch when paid GPU returns.
#
# Council 5/5 ENDORSE (2026-05-07): replays FIRST as deterministic anchors,
# THEN stack our own engineering. Reasoning per Hotz: "ship the deterministic
# ones first; beat top-3 = match top-3 + 1ε."
#
# The three top-3 winners and their adapters (already committed):
#
#   PR #101 (gold, 0.19284):  experiments/public_runtime_adapters/pr101_hnerv_ft_microcodec_adapter/inflate.sh
#   PR #102 (bronze, 0.194987): experiments/public_runtime_adapters/pr102_hnerv_lc_v2_scale095_rplus1_adapter/inflate.sh
#   PR #103 (silver, 0.19487): experiments/public_runtime_adapters/pr103_hnerv_lc_ac_adapter/inflate.sh
#
# Total cost: ~$0.90 (3 × $0.30 each, parallel via tools/parallel_dispatch_top_k.py).
# Wall-clock: ~30-45 minutes (parallel dispatch).
#
# Outcome: three new [contest-CUDA] anchors at the public top-3 scores.
# This is byte-faithful 1:1 replay, NOT a prediction — the scores are
# what the public scoreboard already shows (re-verified independently
# by us on contest-faithful CUDA).
#
# Usage:
#   bash scripts/deferred_dispatch_playbook_top3_replay_20260507.sh <provider>
#   where <provider> ∈ {lightning, vastai}.

set -euo pipefail

PROVIDER="${1:-}"
if [[ -z "$PROVIDER" ]]; then
  echo "ERROR: provide provider arg: lightning | vastai" >&2
  exit 64
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Top-3 PR mapping (PR-number :: archive :: adapter :: expected score :: lane_id)
declare -a TOP3=(
  "101 experiments/results/public_pr_intake_full/public_pr101_intake_20260505_auto/archive.zip experiments/public_runtime_adapters/pr101_hnerv_ft_microcodec_adapter/inflate.sh 0.19284 lane_pr101_replay"
  "102 experiments/results/public_pr_intake_full/public_pr102_intake_20260505_auto/archive.zip experiments/public_runtime_adapters/pr102_hnerv_lc_v2_scale095_rplus1_adapter/inflate.sh 0.194987 lane_pr102_replay"
  "103 experiments/results/public_pr_intake_full/public_pr103_intake_20260505_auto/archive.zip experiments/public_runtime_adapters/pr103_hnerv_lc_ac_adapter/inflate.sh 0.19487 lane_pr103_replay"
)

# Pre-flight: every adapter + archive must exist
for entry in "${TOP3[@]}"; do
  read -r pr_num archive adapter expected_score lane_id <<< "$entry"
  [ -f "$archive" ] || { echo "FATAL: PR$pr_num archive missing: $archive" >&2; exit 1; }
  [ -x "$adapter" ] || { echo "FATAL: PR$pr_num adapter not executable: $adapter" >&2; exit 1; }
  echo "[pr${pr_num}] adapter+archive OK; expected score $expected_score [contest-CUDA]"
done

case "$PROVIDER" in
  lightning)
    echo "=== Lightning T4 parallel dispatch — top-3 replay ==="
    for entry in "${TOP3[@]}"; do
      read -r pr_num archive adapter expected_score lane_id <<< "$entry"
      JOB_NAME="claude_pr${pr_num}_replay_$(date -u +%Y%m%dT%H%M%SZ)"
      RUN_ID="claude_pr${pr_num}_replay_$(date -u +%Y%m%dT%H%M%SZ)"
      MANIFEST_DIR="experiments/results/lightning_batch/${RUN_ID}"

      .venv/bin/python tools/claim_lane_dispatch.py claim \
        --lane-id "$lane_id" --platform lightning \
        --instance-job-id "$JOB_NAME" --agent operator \
        --status "active_dispatch_pr${pr_num}_top3_replay" \
        --notes "1:1 PR #${pr_num} replay; expected $expected_score [contest-CUDA]"

      mkdir -p "$MANIFEST_DIR"
      .venv/bin/python scripts/lightning_repro_workspace.py \
        --remote "s_01knw7wnzbe79wfq5mqqbx1mbz@ssh.lightning.ai" \
        --remote-pact "/teamspace/studios/this_studio/pact" \
        --run-id "$RUN_ID" \
        --manifest-out "${MANIFEST_DIR}/source_manifest.json" \
        --source "src/" --source "experiments/public_runtime_adapters/" \
        --source "experiments/results/public_pr_intake_full/" \
        --source "upstream/" --source "tools/" --source "scripts/" \
        --source "pyproject.toml" --source "uv.lock" \
        --artifact "$archive" \
        --requirements-mode no-install --no-verify

      .venv/bin/python scripts/launch_lightning_batch_job.py exact-eval \
        --job-name "$JOB_NAME" \
        --archive "$archive" \
        --repo-dir "$PWD" --upstream-dir "$PWD/upstream" \
        --teamspace "comma-lab" --studio "lossy-compression-challenge" --user "adpena" \
        --inflate-sh "$PWD/$adapter" \
        --predicted-band $(awk -v s="$expected_score" 'BEGIN { printf "%.4f %.4f", s-0.001, s+0.001 }') \
        --baseline-score "$expected_score" \
        --baseline-archive-bytes "$(stat -c '%s' "$archive" 2>/dev/null || stat -f '%z' "$archive")" \
        --infer-expected-archive --adjudicate --regression-threshold 0.001 \
        --dispatch-lane-id "$lane_id" \
        --source-manifest "${MANIFEST_DIR}/source_manifest.json" \
        --allow-skip-remote-preflight-reason "1:1 PR #${pr_num} replay via deterministic adapter; council 5/5 ENDORSE first-batch top-3 anchors before apogee speculation" \
        --env "INFLATE_TORCH_SPEC=torch==2.5.1+cu124" \
        --env "UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124" \
        --env "UV_INDEX_STRATEGY=unsafe-best-match" &
    done
    wait
    echo "All three Lightning dispatches submitted in parallel."
    ;;

  vastai)
    echo "=== Vast.ai 4090 parallel dispatch — top-3 replay ==="
    echo "Note: requires Vast.ai credit (>= \$3 for safety margin)."
    for entry in "${TOP3[@]}"; do
      read -r pr_num archive adapter expected_score lane_id <<< "$entry"
      LABEL="pr${pr_num}_replay_$(date -u +%Y%m%dT%H%M%SZ)"

      # remote_archive_only_eval.sh handles archive-only inflate + auth eval
      ARCHIVE_PATH="$archive" \
      ARCHIVE_LABEL="pr${pr_num}_replay" \
      INFLATE_SH="$adapter" \
      PREDICTED_LOW="0.190" \
      PREDICTED_HIGH="0.200" \
      CONTROLLED_BASELINE="public_pr${pr_num} (claimed score $expected_score [contest-CUDA])" \
      .venv/bin/python scripts/launch_lane_on_vastai.py full \
        --lane-script scripts/remote_archive_only_eval.sh \
        --label "$LABEL" \
        --predicted-band 0.190 0.200 \
        --estimated-cost 0.30 \
        --council-priority 1 \
        --min-disk-gb 60 \
        --anchor-dirs "$(dirname "$archive")" "$(dirname "$adapter")" &
    done
    wait
    echo "All three Vast.ai dispatches submitted in parallel."
    ;;

  *)
    echo "ERROR: unknown provider '$PROVIDER'" >&2; exit 64
    ;;
esac

echo ""
echo "=== Dispatch fired. Next steps ==="
echo "  1. Monitor: tools/check_vastai.py / Lightning dashboard"
echo "  2. Harvest results: tools/harvest_modal_calls.py --execute / parallel-eval result.json"
echo "  3. Promote: each PR replay should land within ±0.001 of public claim"
echo "  4. After replays land: stack apogee_int6/int7 + Ω-1/Ω-3 on top of confirmed PR101 substrate"
