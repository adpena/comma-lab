#!/usr/bin/env bash
# Deferred-dispatch playbook — fires {int6, int7} parallel dispatch when paid GPU returns.
#
# Pre-conditions verified 2026-05-07T12:13Z:
#   - apogee_int6 archive: experiments/results/apogee_int6_repack_20260504_claude/apogee_int6_archive.zip
#     SHA: 0176a2691a4daf5991170404d30a304ae30389621c0fc54914628414aef39ff1
#     basin-parity PASS (parity_evidence.json present)
#     predispatch_sanity ALL 5 GATES PASS exit 0 (with --readiness-evidence-json + --distortion-proxy-ran)
#   - apogee_int7 archive: experiments/results/apogee_int7_repack_20260504_claude/apogee_int7_archive.zip
#     SHA: 44deb963508a7069c98538211e967c844c0126f79dfd1a0ccf69740ec1bae99b
#     basin-parity PASS (parity_evidence.json present)
#   - Lightning workspace already staged at:
#     experiments/results/lightning_batch/claude_apogee_int6_override_20260507_101520Z/source_manifest.json
#     manifest_sha256: dea0c7a6c60feb3685aadaa8b509e50280a2f820be6fe4a9107f6d967612f46d
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
      JOB_NAME="claude_apogee_int${bits}_lightning_$(date -u +%Y%m%dT%H%M%SZ)"
      RUN_ID="claude_apogee_int${bits}_lightning_$(date -u +%Y%m%dT%H%M%SZ)"
      MANIFEST_DIR="experiments/results/lightning_batch/${RUN_ID}"

      .venv/bin/python tools/claim_lane_dispatch.py claim \
        --lane-id "lane_apogee_int${bits}" --platform lightning \
        --instance-job-id "$JOB_NAME" --agent operator \
        --status "active_dispatch_lightning_t4_post_basin_parity_pass" \
        --notes "Deferred-dispatch playbook fired post-billing-resolution"

      mkdir -p "$MANIFEST_DIR"
      .venv/bin/python scripts/lightning_repro_workspace.py \
        --remote "s_01knw7wnzbe79wfq5mqqbx1mbz@ssh.lightning.ai" \
        --remote-pact "/teamspace/studios/this_studio/pact" \
        --run-id "$RUN_ID" \
        --manifest-out "${MANIFEST_DIR}/source_manifest.json" \
        --source "src/" --source "submissions/apogee_intN/" --source "upstream/" \
        --source "tools/" --source "scripts/" --source "pyproject.toml" --source "uv.lock" \
        --artifact "$archive" \
        --requirements-mode no-install --no-verify

      .venv/bin/python scripts/launch_lightning_batch_job.py exact-eval \
        --job-name "$JOB_NAME" \
        --archive "$archive" \
        --repo-dir "$PWD" --upstream-dir "$PWD/upstream" \
        --teamspace "comma-lab" --studio "lossy-compression-challenge" --user "adpena" \
        --inflate-sh "$PWD/submissions/apogee_intN/inflate.sh" \
        --predicted-band 0.190 0.215 \
        --baseline-score 0.20945673 --baseline-archive-bytes 186239 \
        --infer-expected-archive --adjudicate --regression-threshold 0.05 \
        --dispatch-lane-id "lane_apogee_int${bits}" \
        --source-manifest "${MANIFEST_DIR}/source_manifest.json" \
        --allow-skip-remote-preflight-reason "Track C basin-parity gates all pass; deferred-dispatch playbook 2026-05-07" \
        --env "INFLATE_TORCH_SPEC=torch==2.5.1+cu124" \
        --env "UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124" \
        --env "UV_INDEX_STRATEGY=unsafe-best-match"
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
echo "  2. Harvest: tools/harvest_modal_calls.py / tools/harvest_and_reseed.py"
echo "  3. Update anchors: tools/harvest_and_reseed.py --harvested-jsonl <path> --anchors-path .omx/calibration/anchors_apogee_intN.json"
echo "  4. Update lane_maturity: tools/lane_maturity.py mark lane_apogee_int{6,7} --gate contest_cuda --evidence <eval_artifact>"
