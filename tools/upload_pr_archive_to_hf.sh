#!/usr/bin/env bash
# Upload the comma video compression challenge PR archive corpus to HuggingFace.
#
# Prerequisite (one-time, operator action — interactive token paste):
#   hf auth login
#
# Then this script:
#   1. Creates the dataset repo if it doesn't exist (private by default)
#   2. Materializes a canonical deduplicated release view from raw intake
#   3. Copies the dataset card + LICENSE into the release view
#   4. Uploads the release view with the resumable large-folder uploader
#
# After upload, the dataset is at:
#   https://huggingface.co/datasets/adpena/comma_video_compression_challenge_pr_archive
#
# Per CLAUDE.md "Strategic Secrecy Rule": this dataset is publicly downloadable
# but contains ONLY public PR data (the contest is closed; all PRs are merged
# or open on commaai/comma_video_compression_challenge). No private custody.
#
# Usage:
#   bash tools/upload_pr_archive_to_hf.sh
#   bash tools/upload_pr_archive_to_hf.sh --private    # default
#   bash tools/upload_pr_archive_to_hf.sh --public     # flip to public
set -euo pipefail

REPO_VISIBILITY="--private"
if [ "${1:-}" = "--public" ]; then
    REPO_VISIBILITY=""
fi

DATASET_SLUG="adpena/comma_video_compression_challenge_pr_archive"
RAW_SOURCE_DIR="experiments/results/public_pr_intake_full"
SOURCE_DIR="experiments/results/public_pr_archive_release_view"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! hf auth whoami &>/dev/null; then
    echo "FATAL: hf CLI not authenticated. Run 'hf auth login' first."
    exit 2
fi

echo "[hf-upload] Creating dataset repo (idempotent): $DATASET_SLUG"
hf repos create "$DATASET_SLUG" --type dataset $REPO_VISIBILITY 2>&1 | head -3 || true

echo "[hf-upload] Materializing canonical deduplicated release view..."
python "$HERE/tools/materialize_pr_archive_release_view.py" \
    --source-root "$HERE/$RAW_SOURCE_DIR" \
    --output-root "$HERE/$SOURCE_DIR" \
    --force

echo "[hf-upload] Copying dataset card + LICENSE into release view..."
cp "$HERE/docs/comma_pr_archive_dataset_card.md" "$HERE/$SOURCE_DIR/README.md"
cp "$HERE/LICENSE" "$HERE/$SOURCE_DIR/LICENSE" 2>/dev/null || true

echo "[hf-upload] Uploading $SOURCE_DIR -> datasets/$DATASET_SLUG"
echo "  (resumable large-folder upload; this corpus may be many GB)"
hf upload-large-folder "$DATASET_SLUG" "$HERE/$SOURCE_DIR" \
    --repo-type dataset \
    --num-workers "${HF_UPLOAD_WORKERS:-8}" \
    --exclude ".cache/**" \
    --exclude "**/.cache/**" \
    --exclude "**/.git/**" \
    --exclude "**/__pycache__/**" \
    --exclude "**/.DS_Store"

echo ""
echo "[hf-upload] DONE. Dataset live at:"
echo "  https://huggingface.co/datasets/$DATASET_SLUG"
echo ""
echo "Consumer one-liner:"
echo "  from datasets import load_dataset"
echo "  ds = load_dataset('$DATASET_SLUG')"
