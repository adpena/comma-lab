#!/usr/bin/env python3
"""Build an optimized training bundle for any cloud platform.

Creates a self-contained directory with everything needed to train,
and nothing else. 243MB vs 6GB git clone + LFS.

Contents:
  src/tac/       — the library (~100KB)
  upstream/      — scorer models + GT video only (~126MB)
  archive.zip    — compressed submission (~864KB)
  saliency.npy   — precomputed saliency (~116MB, optional)
  train_renderer_fridrich.py — canonical training script from deploy_config
  setup.sh       — uv-based dependency install
  train.sh       — configurable training launcher

Usage:
    python -m tac.deploy.build_bundle                    # build to /tmp/tac_bundle
    python -m tac.deploy.build_bundle --output ./bundle  # custom output
    python -m tac.deploy.build_bundle --no-saliency      # skip saliency
    python -m tac.deploy.build_bundle --precomputed      # include precomputed tensors
"""
import argparse
import os
import shutil
from pathlib import Path

from tac.deploy.base import repo_root
from tac.deploy.deploy_config import EXPERIMENT_SCRIPT

REPO = repo_root()
TRAINING_SCRIPT = Path(EXPERIMENT_SCRIPT)


def build(output: str = "/tmp/tac_bundle", include_saliency: bool = True, include_precomputed: bool = False):
    out = Path(output)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    upstream = REPO / "upstream"

    # tac library
    shutil.copytree(REPO / "src" / "tac", out / "src" / "tac",
                    ignore=shutil.ignore_patterns("__pycache__"))

    # training script: keep the bundle wired to the canonical provider config.
    shutil.copy2(REPO / TRAINING_SCRIPT, out / TRAINING_SCRIPT.name)

    # upstream essentials only (not 6GB git repo)
    (out / "upstream" / "models").mkdir(parents=True)
    (out / "upstream" / "videos").mkdir(parents=True)
    for f in upstream.glob("models/*.safetensors"):
        shutil.copy2(f, out / "upstream" / "models" / f.name)
    shutil.copy2(upstream / "videos" / "0.mkv", out / "upstream" / "videos" / "0.mkv")
    for f in upstream.glob("*.py"):
        shutil.copy2(f, out / "upstream" / f.name)

    # archive
    shutil.copy2(REPO / "submissions" / "robust_current" / "archive.zip", out / "archive.zip")

    # saliency
    sal_path = REPO / "experiments" / "masks" / "posenet_saliency.npy"
    if include_saliency and sal_path.exists():
        shutil.copy2(sal_path, out / "saliency.npy")

    # precomputed tensors (optional, ~7.5GB)
    if include_precomputed:
        precomp = REPO / "experiments" / "precomputed"
        if precomp.exists():
            shutil.copytree(precomp, out / "precomputed",
                            ignore=shutil.ignore_patterns("__pycache__"))

    # setup script (uv-optimized)
    (out / "setup.sh").write_text("""#!/bin/bash
set -euo pipefail
echo "=== tac optimized bootstrap ==="
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv pip install --system torch torchvision av safetensors timm einops segmentation-models-pytorch numpy pydantic
[ -f saliency.npy ] || python -c "import numpy as np; np.save('saliency.npy', np.ones((30,874,1164), dtype=np.float32))"
echo "=== Ready ==="
""")
    os.chmod(out / "setup.sh", 0o755)

    # train launcher
    (out / "train.sh").write_text("""#!/bin/bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
export PATH="$HOME/.local/bin:$PATH"
TAG="${TAG:-cloud_h64_standard}"
VARIANT="${VARIANT:-base}"
RESULTS_DIR="${RESULTS_DIR:-./weights/${TAG}}"
mkdir -p "$RESULTS_DIR"
export TAC_UPSTREAM_DIR="${TAC_UPSTREAM_DIR:-./upstream}"
export TAC_MODELS_DIR="${TAC_MODELS_DIR:-./upstream/models}"
export TAC_RESULTS_DIR="$RESULTS_DIR"
RESUME=""
[ -f "${RESULTS_DIR}/renderer_best.pt" ] && RESUME="${RESULTS_DIR}/renderer_best.pt" && echo "Resuming from $RESUME"
echo "=== Training: $TAG variant=$VARIANT ==="
PYTHONPATH=src:upstream PYTHONUNBUFFERED=1 python - "$VARIANT" "$RESUME" <<'PY'
import json
import subprocess
import sys

from tac.deploy.deploy_config import build_flags

variant = sys.argv[1]
resume = sys.argv[2].strip()
cmd = build_flags(
    variant=variant,
    provider_script_path="./""" + TRAINING_SCRIPT.name + """",
    resume_from=resume or None,
)
print(json.dumps({"cmd": cmd}, indent=2))
raise SystemExit(subprocess.run(cmd, check=False).returncode)
PY
""")
    os.chmod(out / "train.sh", 0o755)

    # Print summary
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"Bundle: {out} ({total / 1e6:.0f} MB)")
    for item in sorted(out.iterdir()):
        if item.is_dir():
            size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
            print(f"  {item.name}/  {size / 1e6:.1f} MB")
        else:
            print(f"  {item.name}  {item.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build optimized cloud training bundle")
    parser.add_argument("--output", default="/tmp/tac_bundle")
    parser.add_argument("--no-saliency", action="store_true")
    parser.add_argument("--precomputed", action="store_true", help="Include 7.5GB precomputed tensors")
    args = parser.parse_args()
    build(args.output, include_saliency=not args.no_saliency, include_precomputed=args.precomputed)
