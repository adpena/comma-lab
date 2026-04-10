#!/bin/bash
# bat00 WSL permanent setup script
# Run this INSIDE WSL on bat00 (not PowerShell)
#
# From your Mac: ssh bat00 "wsl bash"
# Then paste this whole script.
set -euo pipefail

echo "=== bat00 WSL permanent setup ==="

# 1. Fix DNS permanently (the #1 WSL issue)
echo "[network]
generateResolvConf = false" | sudo tee /etc/wsl.conf > /dev/null

# Remove the auto-generated resolv.conf symlink
sudo rm -f /etc/resolv.conf
echo "nameserver 8.8.8.8
nameserver 1.1.1.1" | sudo tee /etc/resolv.conf > /dev/null
sudo chattr +i /etc/resolv.conf  # make it immutable so WSL can't overwrite it
echo "[OK] DNS fixed permanently"

# 2. Install uv
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
fi
echo "[OK] uv installed"

# 3. Install git-lfs
if ! command -v git-lfs &>/dev/null; then
    sudo apt-get update && sudo apt-get install -y git-lfs
    git lfs install
fi
echo "[OK] git-lfs installed"

# 4. Create workspace
mkdir -p ~/tac
cd ~/tac

# 5. Check CUDA
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')" 2>/dev/null || echo "PyTorch not installed yet — will install with uv"

echo ""
echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. Upload tac bundle: scp -r /tmp/tac_bundle bat00:~/tac/"
echo "  2. SSH in: ssh bat00 'wsl bash -c \"cd ~/tac && bash setup.sh && bash train.sh\"'"
echo "  3. Or use our build_bundle.py to create an optimized package"
