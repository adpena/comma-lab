#!/usr/bin/env bash
# Deferred-dispatch playbook — fires {int6, int7} parallel dispatch when paid GPU returns.
#
# Pre-conditions last investigated 2026-05-07T12:13Z:
#   - apogee_int6 archive: experiments/results/apogee_int6_repack_20260504_claude/apogee_int6_archive.zip
#     SHA: 0176a2691a4daf5991170404d30a304ae30389621c0fc54914628414aef39ff1
#     basin-parity PASS (parity_evidence.json present), but current
#     predispatch_sanity may still refuse the predicted score band when the
#     SHA-tied rate-distortion floor is violated. This playbook re-runs the
#     gate and stops before any claim/stage/submit when it fails.
#   - apogee_int7 archive: experiments/results/apogee_int7_repack_20260504_claude/apogee_int7_archive.zip
#     SHA: 44deb963508a7069c98538211e967c844c0126f79dfd1a0ccf69740ec1bae99b
#     basin-parity PASS (parity_evidence.json present)
#   - Older Lightning workspaces are forensic only. Fresh dispatch delegates to
#     tools/lightning_dispatch_pr106_stack.py so repo paths, runtime paths,
#     claims, source manifests, and g4dn/T4 settings stay canonical.
#
# Usage:
#   When ANY of these unblocks:
#     (a) Vast.ai credit reload (~$25)
#     (b) Lightning T4 AWS capacity returns
#     (c) Azure $200 credits enabled (`az login`)
#     (d) bat00 SSH unblocks
#   Run: bash scripts/deferred_dispatch_playbook_apogee_int6_int7_20260507.sh <provider>
#   where <provider> ∈ {lightning, vastai, azure, bat00}.
#
# Stops at first dispatch attempt; subsequent harvest is a manual `tools/harvest_and_reseed.py` step.

set -euo pipefail

PROVIDER="${1:-}"
if [[ -z "$PROVIDER" ]]; then
  echo "ERROR: provide provider arg: lightning | vastai | azure | bat00" >&2
  exit 64
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Re-verify predispatch sanity for both candidates (cheap, ~3s)
# ─────────────────────────────────────────────────────────────────────────────
for bits in 6 7; do
  archive="experiments/results/apogee_int${bits}_repack_20260504_claude/apogee_int${bits}_archive.zip"
  evidence="experiments/results/apogee_int${bits}_basin_parity_20260507_claude/parity_evidence.json"
  rel_err_pct=$(if [[ "$bits" == "6" ]]; then echo "1.55"; else echo "0.79"; fi)

  echo "[$bits] re-verifying predispatch sanity..."
  .venv/bin/python tools/predispatch_sanity.py \
    --archive "$archive" \
    --predicted-low 0.190 --predicted-high 0.215 \
    --rel-err-pct "$rel_err_pct" \
    --lane-class apogee_intN \
    --readiness-evidence-json "$evidence" \
    --distortion-proxy-ran || {
      echo "[$bits] predispatch sanity FAILED — investigate before dispatching" >&2
      exit 65
    }
  echo "[$bits] PASS"
done

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Dispatch per provider
# ─────────────────────────────────────────────────────────────────────────────
case "$PROVIDER" in
  lightning)
    echo "=== Lightning T4 dispatch (canonical recipe) ==="
    for bits in 6 7; do
      archive="experiments/results/apogee_int${bits}_repack_20260504_claude/apogee_int${bits}_archive.zip"
      evidence="experiments/results/apogee_int${bits}_basin_parity_20260507_claude/parity_evidence.json"
      job_name="claude_apogee_int${bits}_lightning_$(date -u +%Y%m%dT%H%M%SZ)"

      cmd=(
        .venv/bin/python tools/lightning_dispatch_pr106_stack.py
        --lane "apogee_int${bits}"
        --archive "$archive"
        --predicted-low 0.190
        --predicted-high 0.215
        --apogee-distortion-gate-json "$evidence"
        --machine g4dn.2xlarge
        --job-name "$job_name"
      )
      if [[ -n "${LIGHTNING_SSH_TARGET:-}" ]]; then
        cmd+=(--ssh-target "$LIGHTNING_SSH_TARGET")
      fi
      if [[ -n "${LIGHTNING_REMOTE_PACT:-}" ]]; then
        cmd+=(--remote-pact "$LIGHTNING_REMOTE_PACT")
      fi
      "${cmd[@]}"
    done
    ;;

  vastai)
    echo "=== Vast.ai 4090 dispatch (canonical recipe) ==="
    for bits in 6 7; do
      LABEL="apogee_int${bits}_post_basin_parity_$(date -u +%Y%m%dT%H%M%SZ)"
      .venv/bin/python scripts/launch_lane_on_vastai.py full \
        --lane-script scripts/remote_lane_apogee_intN.sh \
        --label "$LABEL" \
        --predicted-band 0.190 0.215 \
        --estimated-cost 0.40 \
        --council-priority 1 \
        --min-disk-gb 60 \
        --anchor-dirs experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex \
                     experiments/results/sensitivity_map_pr106_20260504_claude \
                     "experiments/results/apogee_int${bits}_basin_parity_20260507_claude"
    done
    ;;

  azure)
    echo "=== Azure pivot — ensure 'az login' completed before invoking ==="
    echo "Use task #312 wired infrastructure; specific commands depend on lane registry state."
    echo "Per CLAUDE.md memory: \$200 free credits available."
    exit 1
    ;;

  bat00)
    echo "=== bat00 local CUDA (RTX 2070S→3090) ==="
    echo "Per CLAUDE.md fleet: scripts/bat00.py wsl 'COMMAND' for WSL2 port 2222."
    echo "Need: rsync archive + scorers, run remote_archive_only_eval.sh."
    exit 1
    ;;

  *)
    echo "ERROR: unknown provider '$PROVIDER'" >&2
    exit 64
    ;;
esac

echo ""
echo "=== Dispatch fired. Next steps ==="
echo "  1. Monitor: tools/check_vastai.py  OR  Lightning dashboard"
echo "  2. Harvest: tools/harvest_modal_calls.py --execute / tools/harvest_and_reseed.py"
echo "  3. Update anchors: tools/harvest_and_reseed.py --harvested-jsonl <path> --anchors-path .omx/calibration/anchors_apogee_intN.json"
echo "  4. Update lane_maturity: tools/lane_maturity.py mark lane_apogee_int{6,7} --gate contest_cuda --evidence <eval_artifact>"
