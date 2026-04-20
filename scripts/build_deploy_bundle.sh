#!/bin/bash
# Build a lean deployment bundle for Vast.ai
# Includes ALL necessary source files, excludes large/unnecessary files
#
# Usage:
#   scripts/build_deploy_bundle.sh [BUNDLE_DIR]
#
# Default bundle dir: /tmp/pact_deploy
set -euo pipefail

BUNDLE_DIR="${1:-/tmp/pact_deploy}"

echo "Building deploy bundle -> $BUNDLE_DIR"
rm -rf "$BUNDLE_DIR"
mkdir -p "$BUNDLE_DIR"

# Source code (complete, not cherry-picked)
echo "  syncing src/tac/ ..."
rsync -a --exclude='__pycache__' --exclude='*.pyc' --exclude='*.pyo' \
  src/tac/ "$BUNDLE_DIR/src/tac/"

# Experiment scripts (no large results directories)
echo "  syncing experiments/*.py ..."
mkdir -p "$BUNDLE_DIR/experiments"
rsync -a --exclude='__pycache__' --exclude='*.pyc' --exclude='*.pyo' \
  --exclude='results' --exclude='precomputed_local' \
  experiments/*.py "$BUNDLE_DIR/experiments/"

# Pipeline profiler script
echo "  copying scripts/profile_contest_pipeline.py ..."
mkdir -p "$BUNDLE_DIR/scripts"
cp scripts/profile_contest_pipeline.py "$BUNDLE_DIR/scripts/"

# Submission files (renderer path only, no archive backups, no large files)
echo "  copying submissions/robust_current/ ..."
mkdir -p "$BUNDLE_DIR/submissions/robust_current"
for f in inflate.sh inflate_renderer.py config.env compress_masks.py archive.zip; do
  if [ -f "submissions/robust_current/$f" ]; then
    cp "submissions/robust_current/$f" "$BUNDLE_DIR/submissions/robust_current/"
  else
    echo "  WARNING: submissions/robust_current/$f not found — skipping"
  fi
done

# Checkpoint (canonical, verified — MD5: cff8dca4)
echo "  copying renderer checkpoint ..."
if [ -f "experiments/results/v5_lagrangian_renderer/renderer_best.pt" ]; then
  cp "experiments/results/v5_lagrangian_renderer/renderer_best.pt" "$BUNDLE_DIR/renderer_best.pt"
else
  echo "  WARNING: renderer_best.pt not found at expected path"
fi

# GT poses (if available — pre-computed ego-motion targets)
if [ -f "experiments/results/gt_poses.pt" ]; then
  echo "  copying gt_poses.pt ..."
  cp "experiments/results/gt_poses.pt" "$BUNDLE_DIR/gt_poses.pt"
fi

echo ""
echo "Bundle: $(du -sh "$BUNDLE_DIR" | awk '{print $1}')"
echo "Files: $(find "$BUNDLE_DIR" -type f | wc -l | tr -d ' ')"
echo ""
echo "Key files:"
find "$BUNDLE_DIR" -name "*.pt" -o -name "*.zip" | while read -r f; do
  size=$(du -sh "$f" | awk '{print $1}')
  echo "  $size  ${f#$BUNDLE_DIR/}"
done
