#!/usr/bin/env bash
# Upload a Kaggle mirror of the comma video compression challenge PR archive.
#
# Hugging Face is the canonical host. This script mirrors the same deduplicated
# release view to Kaggle for discoverability and notebooks.
#
# Prerequisite:
#   kaggle auth login
#
# Usage:
#   bash tools/upload_pr_archive_to_kaggle.sh --create
#   bash tools/upload_pr_archive_to_kaggle.sh --version "refresh public PR corpus"
set -euo pipefail

MODE="${1:-}"
VERSION_MESSAGE="${2:-refresh public PR archive corpus mirror}"
DATASET_ID="${KAGGLE_DATASET_ID:-adpena/comma-video-compression-pr-archive}"
RAW_SOURCE_DIR="experiments/results/public_pr_intake_full"
RELEASE_VIEW_DIR="experiments/results/public_pr_archive_release_view"
KAGGLE_VIEW_DIR="experiments/results/public_pr_archive_kaggle_mirror"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-$HERE/.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="$(command -v python3)"
fi

if [ "$MODE" != "--create" ] && [ "$MODE" != "--version" ]; then
    echo "Usage: bash tools/upload_pr_archive_to_kaggle.sh --create"
    echo "   or: bash tools/upload_pr_archive_to_kaggle.sh --version \"message\""
    exit 2
fi

if ! command -v kaggle >/dev/null 2>&1; then
    echo "FATAL: kaggle CLI not found. Install/auth it before uploading."
    exit 2
fi

echo "[kaggle-upload] Materializing canonical HF release view..."
"$PYTHON_BIN" "$HERE/tools/materialize_pr_archive_release_view.py" \
    --source-root "$HERE/$RAW_SOURCE_DIR" \
    --output-root "$HERE/$RELEASE_VIEW_DIR" \
    --force
cp "$HERE/docs/comma_pr_archive_dataset_card.md" "$HERE/$RELEASE_VIEW_DIR/README.md"
cp "$HERE/LICENSE" "$HERE/$RELEASE_VIEW_DIR/LICENSE" 2>/dev/null || true

echo "[kaggle-upload] Materializing Kaggle mirror view..."
"$PYTHON_BIN" "$HERE/tools/materialize_pr_archive_kaggle_mirror.py" \
    --source-root "$HERE/$RELEASE_VIEW_DIR" \
    --output-root "$HERE/$KAGGLE_VIEW_DIR" \
    --dataset-id "$DATASET_ID" \
    --force

echo "[kaggle-upload] Running strict public-link hygiene scan on final Kaggle view..."
"$PYTHON_BIN" "$HERE/tools/audit_public_publish_links.py" \
    "$HERE/$KAGGLE_VIEW_DIR" \
    --repo-root "$HERE" \
    --strict

if [ "$MODE" = "--create" ]; then
    echo "[kaggle-upload] Creating Kaggle dataset: $DATASET_ID"
    kaggle datasets create -p "$HERE/$KAGGLE_VIEW_DIR"
else
    echo "[kaggle-upload] Creating Kaggle dataset version: $DATASET_ID"
    kaggle datasets version -p "$HERE/$KAGGLE_VIEW_DIR" -m "$VERSION_MESSAGE"
fi

echo "[kaggle-upload] DONE: https://www.kaggle.com/datasets/$DATASET_ID"
