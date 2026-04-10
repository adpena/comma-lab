#!/usr/bin/env python3
from __future__ import annotations

import shutil
import os
import importlib
import subprocess
import sys
from pathlib import Path


READ_ONLY_ROOT = Path(__file__).resolve().parent
ACTIVE_ROOT = READ_ONLY_ROOT

PIP_DEPS = [
    "av",
    "safetensors",
    "timm",
    "einops",
    "segmentation-models-pytorch",
    "numpy",
]


def ensure_runtime_dependencies() -> None:
    for dep in PIP_DEPS:
        module_name = dep.replace("-", "_")
        try:
            importlib.import_module(module_name)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", dep])

    if subprocess.run(["bash", "-lc", "command -v git-lfs >/dev/null 2>&1"]).returncode != 0:
        subprocess.check_call(["bash", "-lc", "apt-get update && apt-get install -y git-lfs"])


def ensure_writable_root() -> Path:
    if not Path("/kaggle/working").exists():
        return READ_ONLY_ROOT

    writable_root = Path("/kaggle/working") / "pact_kernel" / "comma-lab-segnet-attack-fixed-h32"
    writable_root.mkdir(parents=True, exist_ok=True)

    for name in ("experiments", "submissions", "reports", "prompts", "docs"):
        source = READ_ONLY_ROOT / name
        if source.exists():
            shutil.copytree(source, writable_root / name, dirs_exist_ok=True)

    return writable_root


def ensure_upstream() -> Path:
    workspace = ACTIVE_ROOT / "workspace" / "upstream"
    upstream = workspace / "comma_video_compression_challenge"
    workspace.mkdir(parents=True, exist_ok=True)
    if not upstream.exists():
        subprocess.check_call([
            "git", "clone", "--depth", "1",
            "https://github.com/commaai/comma_video_compression_challenge.git",
            str(upstream),
        ])
        subprocess.check_call(["git", "lfs", "pull"], cwd=upstream)
    return upstream


def main() -> int:
    global ACTIVE_ROOT
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    ACTIVE_ROOT = ensure_writable_root()
    sys.path.insert(0, str(ACTIVE_ROOT))
    sys.path.insert(0, str(ACTIVE_ROOT / "experiments"))
    sys.path.insert(0, str(ACTIVE_ROOT / "submissions" / "robust_current"))
    ensure_runtime_dependencies()
    ensure_upstream()
    import train_postfilter_segnet_attack as target_module
    result = target_module.main(['--hidden', '32', '--alpha', '20', '--epochs', '1000', '--tag', 'segnet_attack_fixed_ste_h32_kaggle'])
    if isinstance(result, int):
        return result
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
