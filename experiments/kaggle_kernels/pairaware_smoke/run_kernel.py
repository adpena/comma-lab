#!/usr/bin/env python3
from __future__ import annotations

import os
import importlib
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "submissions" / "robust_current"))

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


def ensure_upstream() -> Path:
    workspace = ROOT / "workspace" / "upstream"
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
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    ensure_runtime_dependencies()
    ensure_upstream()
    import train_postfilter_pairaware as target_module
    result = target_module.main(['--hidden', '16', '--height', '64', '--width', '64', '--batch-size', '1', '--device', 'cuda'])
    if isinstance(result, int):
        return result
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
