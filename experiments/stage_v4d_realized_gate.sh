#!/usr/bin/env bash
# ddm_v4d STAGED realized gate — QA65 (dim0 offset) + QA66 (per-pair beta) on
# the v4c composed base.  DO NOT self-fire.  MAIN fires ONE candidate at a time
# when the n600 scorer slot is idle.
#
# WHAT IT MEASURES: the exact composed row (SegNet d_seg + PoseNet d_pose + rate)
# of the v4d byte-closed archive (cell_drop50 base tokens + RE-SOLVED static
# two-plane pose + per-pair photometric (a,b) + per-pair rolling-shutter beta +
# optional dim0 offset-coding + lossless quad), apples-to-apples with the
# MEASURED v4c gate S=0.992972 and the v4b gate S=1.534258 (same eval_root, same
# 0.mkv, same device).  Axis: [macOS-CPU advisory - real evaluator, real bytes];
# contest-CPU authority is a later Modal flight (operator-GO).  Pointer UNMOVED.
#
# MECHANISM (no upstream edits): surgical archive.zip swap into a COPY of the
# pfs1 D1 submission dir, with the v4d receiver (inflate_runner_v4d.py) +
# vendored decode deps, then the stock eval_root/evaluate.sh.
#
# PATH EXPORT is REQUIRED (the bare-python death): evaluate.sh + inflate.sh call
# bare `python`; without the venv on PATH they die "command not found".
#
# Usage:  bash experiments/stage_v4d_realized_gate.sh [device cpu|cuda] [tag]
set -euo pipefail

export PATH="/Users/adpena/Projects/pact/.venv/bin:$PATH"

DEVICE="${1:-cpu}"
TAG="${2:-qa66_celldrop50}"
V4D_DIR="/Volumes/VertigoDataTier/pact/ddm_v4d_20260731"
EVAL_ROOT="/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d1/eval_root"
TEMPLATE_SUB="${EVAL_ROOT}/submissions/pfs1"
ARCHIVE="${V4D_DIR}/v4d_composed_${TAG}_archive.zip"

# REPO IS THE SOURCE OF TRUTH FOR THE RECEIVER (ddm_cx1, 2026-08-03).  The SSD
# copy was a deployment snapshot and had gone STALE: it was byte-identical to the
# pre-ix2 repo receiver, i.e. it carried NO single-member-container path, so a
# container archive staged through this script would have silently taken the
# legacy 6-member branch and died on a missing manifest.json.  Prefer the repo
# file, fall back to the SSD snapshot, and SAY which one is being used.
REPO_ROOT="/Users/adpena/Projects/pact"
if [ -f "${REPO_ROOT}/experiments/inflate_runner_v4d.py" ]; then
  RECEIVER="${REPO_ROOT}/experiments/inflate_runner_v4d.py"
  RECEIVER_SRC="repo"
else
  RECEIVER="${V4D_DIR}/inflate_runner_v4d.py"
  RECEIVER_SRC="ssd-snapshot"
fi
# The ix2 container primitives are FREE generic decode code (rule-118): they are
# vendored into the runtime tree, never counted, and copied from the repo so the
# encoder and the decoder cannot drift.
IX2_MODULE="${REPO_ROOT}/src/tac/optimization/ddm_ix2_archive_container.py"

[ -f "$ARCHIVE" ] || { echo "missing byte-closed archive: $ARCHIVE" >&2; exit 1; }
[ -f "$RECEIVER" ] || { echo "missing v4d receiver: $RECEIVER" >&2; exit 1; }
[ -d "$TEMPLATE_SUB" ] || { echo "missing pfs1 template sub: $TEMPLATE_SUB" >&2; exit 1; }

RUN_SUB="${EVAL_ROOT}/submissions/v4d_${TAG}"
rm -rf "$RUN_SUB"
mkdir -p "$RUN_SUB"
for f in inflate.sh pfs1_warp_receiver.py ddm_r7_token_coder.py \
         ddm_tr1_runtime.py repair_entropy_coder_runtime_adapters.py; do
  cp "${TEMPLATE_SUB}/${f}" "${RUN_SUB}/${f}"
done
cp "$RECEIVER" "${RUN_SUB}/inflate_runner.py"
[ -f "$IX2_MODULE" ] && cp "$IX2_MODULE" "${RUN_SUB}/ddm_ix2_archive_container.py"
cp "$ARCHIVE" "${RUN_SUB}/archive.zip"

echo "[v4d gate] archive=$(basename "$ARCHIVE") device=${DEVICE}"
echo "[v4d gate] archive.zip bytes: $(stat -f%z "${RUN_SUB}/archive.zip" 2>/dev/null || stat -c%s "${RUN_SUB}/archive.zip")"
echo "[v4d gate] receiver=inflate_runner_v4d [src=${RECEIVER_SRC}] (frame0_policy=warp_two_plane_static_photo_beta_v4d)"
echo "[v4d gate] container form: $(unzip -Z1 "${RUN_SUB}/archive.zip" | tr '\n' ' ')"

time bash "${EVAL_ROOT}/evaluate.sh" \
  --submission-dir "$RUN_SUB" \
  --video-names-file "${EVAL_ROOT}/public_test_video_names.txt" \
  --device "$DEVICE"

echo "[v4d gate] report:"
cat "${RUN_SUB}/report.txt"
echo "[v4d gate] receipt schema: ddm_v4d_realized_gate.v1"
echo "  REF v4c MEASURED: d_seg 0.00431179 d_pose 0.01038450 S 0.992972"
echo "  v4d = v4c + QA66 per-pair beta (measured -0.0135 S) [+ QA65 dim0 offset]"
echo "  ACCEPT/verify: realized d_seg ~0.004312 (cell_drop50 tokens; UNCHANGED"
echo "    from v4c — same tokens) ; realized d_pose per the v4d build advisory ;"
echo "    realized S < v4c 0.992972."
