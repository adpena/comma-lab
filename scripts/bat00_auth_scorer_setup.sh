#!/bin/bash
# Auth scorer setup for bat00 WSL2 (RTX 2070 Super, 8GB VRAM, 11GB RAM)
#
# Prerequisites: upload these to C:\Users\adpena\pact_eval\submission\
#   - archive.zip (contains 0.mkv + postfilter_int8.pt)
#   - inflate.sh (our custom inflate script)
#   - inflate_postfilter.py (learned postfilter inflate)
#   - config.env (PYTHON_INFLATE=postfilter)
#   - postfilter_int8.pt (alongside inflate script for fallback)
#
# Run from PowerShell:
#   ssh bat00 "wsl -e bash -c 'nohup bash /mnt/c/Users/adpena/pact_eval/setup.sh > /mnt/c/Users/adpena/pact_eval/setup.log 2>&1 &'"
#
# Check progress:
#   ssh bat00 "type C:\Users\adpena\pact_eval\setup.log"
#   ssh bat00 "type C:\Users\adpena\pact_eval\results\report.txt"

# Use WSL native filesystem for speed (NTFS /mnt/c/ causes IO errors)
WORK_DIR="${HOME}/pact_eval"
# Submission files stay on /mnt/c/ (where they were uploaded)
UPLOAD_DIR="${BAT00_UPLOAD_DIR:-/mnt/c/Users/${USER}/pact_eval}"
SUBMISSION_DIR="${WORK_DIR}/submission"
UPSTREAM_DIR="${WORK_DIR}/upstream"
RESULTS_DIR="${BAT00_RESULTS_DIR:-${UPLOAD_DIR}/results}"
LOG="${RESULTS_DIR}/setup.log"

mkdir -p "$RESULTS_DIR"

# Log everything from the start
exec > >(tee -a "$LOG") 2>&1

echo "========================================"
echo "  bat00 Auth Scorer Setup"
echo "  $(date)"
echo "  GPU: RTX 2070 Super (8GB VRAM)"
echo "========================================"

# ── Pre-flight ──────────────────────────────────────────────
echo ""
echo "=== [1/7] Pre-flight checks ==="

# GPU accessible?
if nvidia-smi > /dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader
else
    echo "FATAL: nvidia-smi not found. GPU not accessible from WSL."
    exit 1
fi

# Copy submission files from Windows upload dir to WSL native filesystem
mkdir -p "$SUBMISSION_DIR"
for f in archive.zip inflate.sh inflate_postfilter.py config.env postfilter_int8.pt; do
    if [ -f "$UPLOAD_DIR/submission/$f" ]; then
        cp "$UPLOAD_DIR/submission/$f" "$SUBMISSION_DIR/$f"
    fi
done

# Verify required files
for f in archive.zip inflate.sh inflate_postfilter.py config.env; do
    if [ ! -f "$SUBMISSION_DIR/$f" ]; then
        echo "FATAL: Missing $f (upload to $UPLOAD_DIR/submission/ first)"
        exit 1
    fi
done
echo "  All submission files copied to native WSL filesystem"

# Archive hash for replicability
ARCHIVE_MD5=$(md5sum "$SUBMISSION_DIR/archive.zip" | awk '{print $1}')
echo "  archive.zip md5: $ARCHIVE_MD5"

# git-lfs
if ! command -v git-lfs > /dev/null 2>&1 && ! git lfs version > /dev/null 2>&1; then
    echo "FATAL: git-lfs not installed. Run: sudo apt-get install git-lfs"
    exit 1
fi

# ffmpeg
if ! command -v ffmpeg > /dev/null 2>&1; then
    echo "FATAL: ffmpeg not installed. Run: sudo apt-get install ffmpeg"
    exit 1
fi

# ── Dependencies ────────────────────────────────────────────
echo ""
echo "=== [2/7] Installing Python dependencies ==="

# Fix DNS if broken (common WSL2 issue)
if ! ping -c 1 -W 2 github.com > /dev/null 2>&1; then
    echo "  Fixing DNS..."
    echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf > /dev/null
    echo "nameserver 8.8.4.4" | sudo tee -a /etc/resolv.conf > /dev/null
fi

# Use a venv (PEP 668 requires this on Ubuntu 24.04)
VENV_DIR="${WORK_DIR}/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "  Creating venv..."
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# Check if already installed
if python -c "import torch; import nvidia.dali; print('deps OK')" 2>/dev/null; then
    echo "  Dependencies already installed — skipping"
else
    echo "  Installing torch + DALI + scorer deps..."
    pip install --quiet \
        torch torchvision --index-url https://download.pytorch.org/whl/cu126
    pip install --quiet \
        nvidia-dali-cuda120 safetensors timm einops \
        segmentation-models-pytorch av numpy tqdm pydantic
fi

# ── Clone upstream ──────────────────────────────────────────
echo ""
echo "=== [3/7] Upstream scorer repo ==="

if [ -d "$UPSTREAM_DIR" ] && [ -f "$UPSTREAM_DIR/evaluate.py" ]; then
    echo "  Already cloned"
else
    echo "  Cloning..."
    rm -rf "$UPSTREAM_DIR"
    git clone --depth 1 \
        https://github.com/commaai/comma_video_compression_challenge.git \
        "$UPSTREAM_DIR"
    cd "$UPSTREAM_DIR" && git lfs pull && cd "$WORK_DIR"
fi

# Verify models are real files (not LFS stubs)
MODEL_SIZE=$(stat -c%s "$UPSTREAM_DIR/models/posenet.safetensors" 2>/dev/null || echo 0)
if [ "$MODEL_SIZE" -lt 1000000 ]; then
    echo "  Models are LFS stubs — pulling..."
    cd "$UPSTREAM_DIR" && git lfs pull && cd "$WORK_DIR"
fi
echo "  PoseNet: $(du -h "$UPSTREAM_DIR/models/posenet.safetensors" | awk '{print $1}')"
echo "  SegNet: $(du -h "$UPSTREAM_DIR/models/segnet.safetensors" | awk '{print $1}')"

# ── Verify CUDA + DALI ─────────────────────────────────────
echo ""
echo "=== [4/7] Verifying CUDA + DALI ==="

python3 << 'PYEOF'
import torch
print(f"  torch: {torch.__version__}")
print(f"  CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem // (1024*1024)} MB")
    print(f"  Compute cap: {torch.cuda.get_device_capability(0)}")
else:
    print("  FATAL: CUDA not available")
    exit(1)

try:
    import nvidia.dali
    print(f"  DALI: {nvidia.dali.__version__}")
except ImportError:
    print("  WARNING: DALI not available — will use PyAV (NOT auth-matching)")

import numpy, av
print(f"  numpy: {numpy.__version__}, av: {av.__version__}")
print("  ALL GOOD")
PYEOF

# ── Run evaluation ──────────────────────────────────────────
echo ""
echo "=== [5/7] Running evaluation (DALI + CUDA) ==="
echo "  This takes ~10-15 minutes on RTX 2070 Super"
echo "  submission: $SUBMISSION_DIR"
echo "  upstream: $UPSTREAM_DIR"

cd "$UPSTREAM_DIR"

# Use reduced batch size for 11GB WSL RAM
bash evaluate.sh \
    --submission-dir "$SUBMISSION_DIR" \
    --device cuda \
    2>&1 | tee "$RESULTS_DIR/eval_full_output.txt"

# ── Extract results ─────────────────────────────────────────
echo ""
echo "=== [6/7] Results ==="

if [ -f "$SUBMISSION_DIR/report.txt" ]; then
    cat "$SUBMISSION_DIR/report.txt"
    cp "$SUBMISSION_DIR/report.txt" "$RESULTS_DIR/report.txt"
else
    echo "  ERROR: report.txt not found"
fi

# ── Save replicability record ───────────────────────────────
echo ""
echo "=== [7/7] Replicability record ==="

python3 << PYEOF
import json, subprocess, platform, os
record = {
    "timestamp": "$(date -Iseconds)",
    "platform": "bat00_wsl2",
    "gpu": "RTX 2070 Super",
    "archive_md5": "$ARCHIVE_MD5",
    "archive_bytes": os.path.getsize("$SUBMISSION_DIR/archive.zip"),
    "device": "cuda",
    "hostname": platform.node(),
    "python": platform.python_version(),
}
try:
    import torch
    record["torch_version"] = torch.__version__
    record["cuda_available"] = torch.cuda.is_available()
except: pass
try:
    import nvidia.dali
    record["dali_version"] = nvidia.dali.__version__
    record["gt_decode"] = "DALI_NVDEC"
except:
    record["gt_decode"] = "PyAV"

with open("$RESULTS_DIR/replicability.json", "w") as f:
    json.dump(record, f, indent=2)
print(json.dumps(record, indent=2))
PYEOF

echo ""
echo "========================================"
echo "  DONE — $(date)"
echo "  Results: $RESULTS_DIR/"
echo "========================================"
