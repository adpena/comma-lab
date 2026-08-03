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

# THE REPO IS THE SOURCE OF TRUTH FOR THE WHOLE RUNTIME TREE (ddm_cx1 for the
# receiver, 2026-08-03; ddm_cu1 generalised it to every module, same day).
#
# ddm_cx1 found the receiver copy on the SSD had gone STALE (byte-identical to
# the pre-ix2 repo receiver -> no single-member-container path -> a container
# archive would silently take the legacy 6-member branch and die on a missing
# manifest.json) and made THAT ONE FILE come from the repo.  ddm_cu1 measured
# the rest of the tree and found the seam was wider and worse: the four modules
# still copied from ${TEMPLATE_SUB} were HAND-DE-`tac`-IFIED deployment
# snapshots.  The repo copies carried top-level `from tac....` imports
# (ddm_r7_token_coder, repair_entropy_coder_runtime_adapters) that raise
# ModuleNotFoundError in a contest runtime tree, where inflate.sh runs bare
# `python` and no `tac` package exists.  So the SSD snapshot was not a
# convenience -- it was the ONLY runnable copy, and the tracked source had
# silently stopped being the source of truth.  ddm_cu1 landed the dual-layout
# import (flat sibling first, `tac....` fallback) in those modules, so the repo
# files now serve BOTH layouts and the snapshot is no longer needed.
#
# Everything below is FREE generic decode code (rule-118): vendored into the
# runtime tree, never counted, and copied from the repo so encoder and decoder
# cannot drift.  If a repo file is missing we FAIL rather than fall back to a
# snapshot -- a silent fallback is what produced the stale-receiver defect.
REPO_ROOT="${PACT_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RECEIVER="${REPO_ROOT}/experiments/inflate_runner_v4d.py"
RECEIVER_SRC="repo"
IX2_MODULE="${REPO_ROOT}/src/tac/optimization/ddm_ix2_archive_container.py"
# vendored-name -> repo path (the receiver's full import closure)
RUNTIME_MODULES=(
  "ddm_r7_token_coder.py:${REPO_ROOT}/experiments/ddm_r7_token_coder.py"
  "ddm_tr1_runtime.py:${REPO_ROOT}/src/tac/optimization/ddm_tr1_runtime.py"
  "pfs1_warp_receiver.py:${REPO_ROOT}/src/tac/optimization/pfs1_warp_receiver.py"
  "repair_entropy_coder_runtime_adapters.py:${REPO_ROOT}/src/tac/optimization/repair_entropy_coder_runtime_adapters.py"
)

[ -f "$ARCHIVE" ] || { echo "missing byte-closed archive: $ARCHIVE" >&2; exit 1; }
[ -f "$RECEIVER" ] || { echo "missing v4d receiver in repo: $RECEIVER" >&2; exit 1; }
[ -f "$IX2_MODULE" ] || { echo "missing ix2 container in repo: $IX2_MODULE" >&2; exit 1; }
[ -d "$TEMPLATE_SUB" ] || { echo "missing pfs1 template sub: $TEMPLATE_SUB" >&2; exit 1; }

RUN_SUB="${EVAL_ROOT}/submissions/v4d_${TAG}"
rm -rf "$RUN_SUB"
mkdir -p "$RUN_SUB"
# inflate.sh is the stock 4-line bare-`python` wrapper and is the only file the
# template still supplies; it carries no decode logic.
cp "${TEMPLATE_SUB}/inflate.sh" "${RUN_SUB}/inflate.sh"
cp "$RECEIVER" "${RUN_SUB}/inflate_runner.py"
cp "$IX2_MODULE" "${RUN_SUB}/ddm_ix2_archive_container.py"
for entry in "${RUNTIME_MODULES[@]}"; do
  name="${entry%%:*}"; src="${entry#*:}"
  [ -f "$src" ] || { echo "missing runtime module in repo: $src" >&2; exit 1; }
  cp "$src" "${RUN_SUB}/${name}"
done
cp "$ARCHIVE" "${RUN_SUB}/archive.zip"

echo "[v4d gate] archive=$(basename "$ARCHIVE") device=${DEVICE}"
echo "[v4d gate] archive.zip bytes: $(stat -f%z "${RUN_SUB}/archive.zip" 2>/dev/null || stat -c%s "${RUN_SUB}/archive.zip")"
echo "[v4d gate] receiver=inflate_runner_v4d [src=${RECEIVER_SRC}] (frame0_policy=warp_two_plane_static_photo_beta_v4d)"
echo "[v4d gate] runtime tree provenance (repo=source of truth, ddm_cu1):"
for f in inflate_runner.py ddm_ix2_archive_container.py ddm_r7_token_coder.py \
         ddm_tr1_runtime.py pfs1_warp_receiver.py \
         repair_entropy_coder_runtime_adapters.py; do
  echo "    $(shasum -a 256 "${RUN_SUB}/${f}" | cut -c1-12)  ${f}"
done
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
