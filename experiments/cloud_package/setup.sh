#!/bin/bash
set -euo pipefail
echo "=== Setting up tac training environment ==="

# Install uv if not present
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Clone upstream for scorer models
if [ ! -d upstream ]; then
    git clone --depth 1 https://github.com/commaai/comma_video_compression_challenge.git upstream
    cd upstream && git lfs pull && cd ..
fi

# Install deps
uv pip install --system -r requirements.txt

echo "=== Setup complete ==="
