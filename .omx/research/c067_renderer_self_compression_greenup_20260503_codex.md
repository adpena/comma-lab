# C067 Renderer Self-Compression Greenup

Date: 2026-05-03
Author: Codex
Scope: renderer self-compression burn/export/transplant/preflight support only.
No remote GPU dispatch was performed in this pass.

## Source And Target

- Source archive: `experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/archive.zip`
- Source archive bytes: `276214`
- Source archive SHA-256: `226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a`
- C067 renderer member bytes: `59288`
- C067 renderer member SHA-256: `5657593aec0bf380f0bd614578cc4a76da8589b6ef8ce0331c28b6a4d6658efb`
- Unchanged-component sub-0.300 byte ceiling: `252760`
- Required unchanged-component savings vs C067: `23454` archive bytes

The only valid post-burn transplant change is `renderer.bin`; `masks.mkv` and
`optimized_poses.bin` must remain byte-identical.

## Local Fix

The Q-FAITHFUL snapshot exporter had an export-stage gap: it wrote
Brotli-compressed QFAI bytes as `renderer.bin`, then handed that raw archive to
`experiments/repack_quantizr_faithful_qzs3_archive.py`, whose renderer loader
expects raw `QFAI`, `QZS3`, or Torch-FP4 magic. A fresh burn could therefore
train successfully and fail during snapshot export before transplant preflight.

Fixed in `scripts/q_faithful_snapshot_loop.py`:

- `renderer.bin` is now raw `QFAI` bytes.
- Brotli-compressed QFAI is preserved only as an unarchived sidecar for byte
  accounting.
- Export metadata now records `renderer_bin_wire_format="QFAI"` and
  `renderer_bin_brotli_compressed=false`.

Regression added in `src/tac/tests/test_q_faithful_snapshot_loop.py`:

- `test_qfai_export_writes_raw_renderer_bin_for_qzs3_repack`
- It proves exported `renderer.bin` starts with `QFAI` and can be consumed by
  the QZS3 repacker.

## Current Readiness

Prepared burn packet remains valid:

- `experiments/results/c067_fixed_renderer_burn_prep_20260503/c067_qfaithful_fixedmask_fixedpose_seed20260503/fixed_c067_renderer_burn_manifest.json`
- `experiments/results/c067_fixed_renderer_burn_prep_20260503/c067_qfaithful_fixedmask_fixedpose_seed20260503/run_fixed_renderer_burn.sh`

The manifest records correct C067 member custody and argparse dry-run success
for both `train_renderer.py` and `q_faithful_snapshot_loop.py`. The local
training dispatch gate still records missing Lane 12 clearance, but
`.omx/research/c067_fixed_renderer_burn_audited_override_20260503_codex.md`
documents the operator override for the already queued paid training burn.

Fresh export-unlock scan:

- Path: `experiments/results/c067_renderer_self_compression_greenup_20260503_worker/trained_renderer_export_unlock_plan.json`
- Candidate count: `131`
- Non-surrogate candidate count: `32`
- H100-ready preflight count: `0`
- Verdict: `blocked_no_h100_dispatch`
- Blocker: `no non-surrogate trained-renderer archive passed preflight`

This is the expected local state before the RTX PRO burn returns a new trained
renderer snapshot. No exact-eval dispatch command is ready locally.

## Verification

```bash
.venv/bin/python -m pytest src/tac/tests/test_q_faithful_snapshot_loop.py
```

Result: `18 passed`.

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_q_faithful_snapshot_loop.py \
  src/tac/tests/test_prepare_c067_fixed_renderer_burn.py \
  src/tac/tests/test_preflight_trained_renderer_transplant.py \
  src/tac/tests/test_preflight_renderer_transplant_pose_safety.py \
  src/tac/tests/test_plan_trained_renderer_export_unlock.py \
  src/tac/tests/test_qbf1_renderer_codec.py
```

Result: `45 passed`.

```bash
bash -n experiments/results/c067_fixed_renderer_burn_prep_20260503/c067_qfaithful_fixedmask_fixedpose_seed20260503/run_fixed_renderer_burn.sh
.venv/bin/python -m py_compile \
  scripts/q_faithful_snapshot_loop.py \
  experiments/prepare_c067_fixed_renderer_burn.py \
  experiments/preflight_trained_renderer_transplant.py \
  experiments/preflight_renderer_transplant_pose_safety.py \
  experiments/plan_trained_renderer_export_unlock.py
```

Result: both passed.

## Exact Next Local Commands

Use these after the queued RTX PRO burn has produced at least one snapshot
archive. Do not exact-eval from this sequence; it only builds local readiness.

```bash
set -euo pipefail

export C067_ARCHIVE=experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/archive.zip
export GREENUP_ROOT=experiments/results/c067_renderer_self_compression_greenup_20260503_worker
export CANDIDATE_ARCHIVE=<path-to-post-burn-snapshot-archive.zip>
export CANDIDATE_ID="$(basename "$(dirname "${CANDIDATE_ARCHIVE}")")"
export CANDIDATE_DIR="${GREENUP_ROOT}/post_burn_${CANDIDATE_ID}"
mkdir -p "${CANDIDATE_DIR}/unpacked" "${CANDIDATE_DIR}/logs"

.venv/bin/python - <<'PY'
from pathlib import Path
import os
from experiments.build_blockfp_c067_archive import extract_runtime_members

archive = Path(os.environ["CANDIDATE_ARCHIVE"])
out = Path(os.environ["CANDIDATE_DIR"]) / "unpacked"
members, _contract = extract_runtime_members(archive)
for name in ("renderer.bin", "masks.mkv", "optimized_poses.bin"):
    (out / name).write_bytes(members[name])
PY

.venv/bin/python -u experiments/preflight_trained_renderer_transplant.py \
  --source-archive "${C067_ARCHIVE}" \
  --renderer-export "${CANDIDATE_DIR}/unpacked/renderer.bin" \
  --output-dir "${CANDIDATE_DIR}/transplant" \
  --force \
  2>&1 | tee "${CANDIDATE_DIR}/logs/transplant_preflight.log"

export SELECTED_CANDIDATE_ARCHIVE="$(
  jq -r '.best_by_archive_bytes.archive_path' \
    "${CANDIDATE_DIR}/transplant/trained_renderer_blockfp_preflight.json"
)"

.venv/bin/python -u experiments/preflight_renderer_transplant_pose_safety.py \
  --source-archive "${C067_ARCHIVE}" \
  --candidate-archive "${SELECTED_CANDIDATE_ARCHIVE}" \
  --output-json "${CANDIDATE_DIR}/pose_safety_preflight.json" \
  --max-pairs 5 \
  2>&1 | tee "${CANDIDATE_DIR}/logs/pose_safety_preflight.log"

.venv/bin/python -u experiments/preflight_trained_renderer_transplant.py \
  --source-archive "${C067_ARCHIVE}" \
  --renderer-export "${CANDIDATE_DIR}/unpacked/renderer.bin" \
  --output-dir "${CANDIDATE_DIR}/transplant_posegate" \
  --pose-safety-json "${CANDIDATE_DIR}/pose_safety_preflight.json" \
  --force \
  2>&1 | tee "${CANDIDATE_DIR}/logs/transplant_posegate_preflight.log"

.venv/bin/python -u experiments/plan_trained_renderer_export_unlock.py \
  --scan-dir "${CANDIDATE_DIR}/transplant_posegate" \
  --output "${CANDIDATE_DIR}/trained_renderer_export_unlock_plan.json" \
  2>&1 | tee "${CANDIDATE_DIR}/logs/export_unlock_plan.log"
```

If the final plan reports `readiness.verdict="h100_ready_after_claim"`, the
next exact-eval dispatch must first claim
`c067_trained_renderer_self_compression_blockfp` with
`tools/claim_lane_dispatch.py claim ...`. The preflight-emitted Lightning
commands are the command source after that claim.
