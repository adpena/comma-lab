#!/bin/bash
# Auth-quality scorer eval on Lightning T4 with DALI.
# Run from local machine:
#   LIGHTNING_SSH_TARGET=lightning-pact bash scripts/lightning_auth_eval.sh [archive.zip]
#
# Repeatable: upload archive → inflate on T4 → score with DALI → fetch report.

set -euo pipefail

LIGHTNING_SSH_TARGET="${LIGHTNING_SSH_TARGET:?Set LIGHTNING_SSH_TARGET to a ~/.ssh/config alias for the Lightning Studio}"
case "$LIGHTNING_SSH_TARGET" in
  ssh.lightning.ai)
    echo "FATAL: use an SSH config alias, not bare ssh.lightning.ai" >&2
    exit 2
    ;;
esac
# Inline SSH options so preflight check_ssh_commands_have_connect_timeout
# sees ConnectTimeout literally on the SSH= line (its 3-line lookahead
# from `ssh ...` doesn't backtrack to find SSH_OPTS array assignment).
SSH=(
  ssh
  -o BatchMode=yes
  -o PasswordAuthentication=no
  -o KbdInteractiveAuthentication=no
  -o ConnectTimeout=20
  -o ConnectionAttempts=3
  -o ServerAliveInterval=15
  -o ServerAliveCountMax=4
  -o TCPKeepAlive=yes
)
SCP=(
  scp
  -o BatchMode=yes
  -o PasswordAuthentication=no
  -o KbdInteractiveAuthentication=no
  -o ConnectTimeout=20
  -o ConnectionAttempts=3
  -o ServerAliveInterval=15
  -o ServerAliveCountMax=4
  -o TCPKeepAlive=yes
)

ARCHIVE="${1:-submissions/robust_current/archive.zip}"
EVAL_DIR="/tmp/auth_eval_$(date +%Y%m%dT%H%M%S)"

echo "============================================"
echo "  Lightning T4 Auth Scorer Eval"
echo "  $(date)"
echo "  Archive: $ARCHIVE ($(stat -f%z "$ARCHIVE" 2>/dev/null || stat --printf="%s" "$ARCHIVE") bytes)"
echo "  MD5: $(md5 -q "$ARCHIVE" 2>/dev/null || md5sum "$ARCHIVE" | awk '{print $1}')"
echo "============================================"

# 1. Upload
echo ""
echo "[1/5] Uploading archive + inflate scripts..."
"${SSH[@]}" "$LIGHTNING_SSH_TARGET" "mkdir -p $EVAL_DIR/submission/inflated"
"${SCP[@]}" "$ARCHIVE" "$LIGHTNING_SSH_TARGET:$EVAL_DIR/submission/archive.zip"
"${SCP[@]}" submissions/robust_current/inflate_postfilter.py "$LIGHTNING_SSH_TARGET:$EVAL_DIR/submission/"
"${SCP[@]}" submissions/robust_current/config.env "$LIGHTNING_SSH_TARGET:$EVAL_DIR/submission/"
echo "  Done"

# 2. Inflate + Score (single remote command)
echo ""
echo "[2/5] Running inflate + score on Lightning T4..."
"${SSH[@]}" "$LIGHTNING_SSH_TARGET" "
source /system/conda/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate cloudspace 2>/dev/null

cd $EVAL_DIR

# Unzip archive
unzip -o submission/archive.zip -d submission/archive

# Inflate
echo '=== Inflating ==='
export PYTHONPATH=/home/zeus/content/pact/src:/home/zeus/content/upstream
python submission/inflate_postfilter.py \
  submission/archive \
  submission/inflated \
  /home/zeus/content/upstream/public_test_video_names.txt \
  submission/archive/postfilter_int8.pt \
  --device cuda 2>&1

echo '=== Scoring with DALI ==='
cd /home/zeus/content/upstream
python evaluate.py \
  --submission-dir $EVAL_DIR/submission \
  --uncompressed-dir /home/zeus/content/upstream/videos \
  --video-names-file /home/zeus/content/upstream/public_test_video_names.txt \
  --device cuda \
  --report $EVAL_DIR/report.txt 2>&1
" 2>&1 | tee /tmp/lightning_auth_eval.log

# 3. Fetch report
echo ""
echo "[3/5] Fetching report..."
"${SCP[@]}" "$LIGHTNING_SSH_TARGET:$EVAL_DIR/report.txt" /tmp/lightning_auth_report.txt 2>/dev/null || true

# 4. Display results
echo ""
echo "[4/5] Results:"
if [ -f /tmp/lightning_auth_report.txt ]; then
    cat /tmp/lightning_auth_report.txt
    echo ""
    echo "[5/5] Report saved to /tmp/lightning_auth_report.txt"
else
    echo "  No report — check /tmp/lightning_auth_eval.log for errors"
    tail -10 /tmp/lightning_auth_eval.log
fi

echo ""
echo "============================================"
echo "  DONE — $(date)"
echo "============================================"
