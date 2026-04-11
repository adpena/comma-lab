from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class KaggleKernelSpec:
    slug: str
    title: str
    module_name: str | None = None
    code_source: Path | None = None
    code_file: str = "run_kernel.py"
    args: tuple[str, ...] = ()
    include_paths: tuple[Path, ...] = ()
    dataset_sources: tuple[str, ...] = ()
    bootstrap_preamble: str | None = None


def build_kernel_metadata(
    *,
    username: str,
    slug: str,
    title: str,
    code_file: str,
    dataset_sources: tuple[str, ...] = (),
) -> dict[str, object]:
    return {
        "id": f"{username}/{slug}",
        "title": title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": list(dataset_sources),
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def _launcher_source(spec: KaggleKernelSpec) -> str:
    if spec.module_name is None:
        raise ValueError("module_name is required for launcher-based kernels")
    args_literal = repr(list(spec.args))
    return f"""#!/usr/bin/env python3
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

    writable_root = Path("/kaggle/working") / "pact_kernel" / "{spec.slug}"
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
    import {spec.module_name} as target_module
    result = target_module.main({args_literal})
    if isinstance(result, int):
        return result
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


def write_bundle(
    *,
    bundle_dir: Path,
    username: str,
    spec: KaggleKernelSpec,
    repo_root: Path | None = None,
) -> None:
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    metadata = build_kernel_metadata(
        username=username,
        slug=spec.slug,
        title=spec.title,
        code_file=spec.code_file,
        dataset_sources=spec.dataset_sources,
    )
    (bundle_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    if spec.code_source is not None:
        code_source = Path(spec.code_source)
        code_destination = bundle_dir / spec.code_file
        code_destination.parent.mkdir(parents=True, exist_ok=True)
        if spec.bootstrap_preamble:
            code_text = code_source.read_text()
            code_destination.write_text(spec.bootstrap_preamble + "\n\n" + code_text)
        else:
            shutil.copy2(code_source, code_destination)
    else:
        (bundle_dir / spec.code_file).write_text(_launcher_source(spec))
    root = repo_root.resolve() if repo_root is not None else None
    for source in spec.include_paths:
        source_path = Path(source)
        if spec.code_source is not None and source_path.resolve() == Path(spec.code_source).resolve():
            continue
        if spec.code_source is not None and source_path.suffix not in {".py", ".pyi"}:
            destination = bundle_dir / source_path.name
        elif root is not None:
            try:
                rel = source_path.resolve().relative_to(root)
                destination = bundle_dir / rel
            except ValueError:
                destination = bundle_dir / source_path.name
        else:
            destination = bundle_dir / source_path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
