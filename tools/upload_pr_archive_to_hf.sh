#!/usr/bin/env bash
# Upload the comma video compression challenge PR archive corpus to HuggingFace.
#
# Prerequisite (one-time, operator action - interactive token paste):
#   hf auth login
#
# Then this script:
#   1. Creates the dataset repo if it doesn't exist
#   2. Materializes a canonical deduplicated release view from raw intake
#   3. Copies the dataset card + LICENSE into the release view
#   4. Uploads the release view with the resumable large-folder uploader
#
# After upload, the dataset is at:
#   https://huggingface.co/datasets/adpena/comma_video_compression_challenge_pr_archive
#
# The release view contains only public PR data and sanitized provenance. Use
# --private for a staging upload; use --public after the strict hygiene scan is
# green and the operator intentionally wants the dataset visible to the community.
#
# Usage:
#   bash tools/upload_pr_archive_to_hf.sh
#   bash tools/upload_pr_archive_to_hf.sh --private    # default
#   bash tools/upload_pr_archive_to_hf.sh --public     # flip to public
set -euo pipefail

VISIBILITY_MODE="${1:---private}"
case "$VISIBILITY_MODE" in
    --private)
        REPO_VISIBILITY="--private"
        HF_PRIVATE="true"
        ;;
    --public)
        REPO_VISIBILITY=""
        HF_PRIVATE="false"
        ;;
    *)
        echo "Usage: bash tools/upload_pr_archive_to_hf.sh [--private|--public]" >&2
        exit 2
        ;;
esac

DATASET_SLUG="adpena/comma_video_compression_challenge_pr_archive"
RAW_SOURCE_DIR="experiments/results/public_pr_intake_full"
SOURCE_DIR="experiments/results/public_pr_archive_release_view"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-$HERE/.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="$(command -v python3)"
fi
export PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}"

if ! hf auth whoami &>/dev/null; then
    echo "FATAL: hf CLI not authenticated. Run 'hf auth login' first."
    exit 2
fi

echo "[hf-upload] Creating dataset repo (idempotent): $DATASET_SLUG"
hf repos create "$DATASET_SLUG" --type dataset $REPO_VISIBILITY --exist-ok

echo "[hf-upload] Materializing canonical deduplicated release view..."
"$PYTHON_BIN" "$HERE/tools/materialize_pr_archive_release_view.py" \
    --source-root "$HERE/$RAW_SOURCE_DIR" \
    --output-root "$HERE/$SOURCE_DIR" \
    --force

echo "[hf-upload] Copying dataset card + LICENSE into release view..."
cp "$HERE/docs/comma_pr_archive_dataset_card.md" "$HERE/$SOURCE_DIR/README.md"
cp "$HERE/LICENSE" "$HERE/$SOURCE_DIR/LICENSE" 2>/dev/null || true

echo "[hf-upload] Running strict public-release hygiene scan on final release view..."
"$PYTHON_BIN" - "$HERE/$SOURCE_DIR" <<'PY'
import sys
from pathlib import Path

from tac.preflight import check_public_release_hygiene

root = Path(sys.argv[1])
check_public_release_hygiene(
    repo_root=Path.cwd(),
    strict=True,
    verbose=True,
    scan_paths=[root],
)
PY

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

# `hf upload-large-folder` intentionally writes resumability metadata under
# LOCAL_PATH/.cache/huggingface. Keep that state out of the canonical release
# view after a completed upload so local audits and mirror jobs see a pure tree.
rm -rf "$HERE/$SOURCE_DIR/.cache"

echo "[hf-upload] Enforcing dataset visibility: private=$HF_PRIVATE"
"$PYTHON_BIN" - "$DATASET_SLUG" "$HF_PRIVATE" <<'PY'
import sys

from huggingface_hub import HfApi

repo_id = sys.argv[1]
private = sys.argv[2].lower() == "true"
api = HfApi()
api.update_repo_settings(repo_id=repo_id, repo_type="dataset", private=private)
info = api.repo_info(repo_id, repo_type="dataset")
if bool(info.private) != private:
    raise SystemExit(
        f"FATAL: HF visibility mismatch for {repo_id}: expected private={private}, observed {info.private}"
    )
print({"repo": repo_id, "private": bool(info.private), "sha": info.sha})
PY

echo ""
echo "[hf-upload] DONE. Dataset live at:"
echo "  https://huggingface.co/datasets/$DATASET_SLUG"
echo ""
echo "Consumer one-liner:"
echo "  from datasets import load_dataset"
echo "  ds = load_dataset('$DATASET_SLUG')"
