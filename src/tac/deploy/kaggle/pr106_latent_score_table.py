"""Kaggle bundle writer for the PR106 latent-sidecar score-table lane."""
from __future__ import annotations

import gzip
import json
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path

from tac.deploy.claims import active_claim_row
from tac.deploy.kaggle.kaggle_kernel_builder import build_kernel_metadata
from tac.deploy.kaggle.source_bundle import add_path_to_tar, dataset_sources
from tac.deploy.pr106_latent import (
    REMOTE_SCRIPT,
    Pr106LatentScoreTableSpec,
    score_table_env,
)

DEFAULT_DATASET_SLUG = "comma-lab-private-assets"
DEFAULT_SOURCE_DATASET_SLUG = "comma-lab-pr106-latent-source"
DEFAULT_KERNEL_SLUG = "comma-lab-pr106-latent-score-table"
DEFAULT_KERNEL_TITLE = "comma-lab PR106 latent score-table"
DEFAULT_JOB_NAME = "kaggle_pr106_latent_score_table"
DEFAULT_PR106_ARCHIVE_IN_BUNDLE = "inputs/pr106_archive.zip"
DEFAULT_SOURCE_BUNDLE_NAME = "pact_pr106_latent_source_bundle.tar.gz"
PYTORCH_P100_FALLBACK_DEPS = (
    "torch==2.4.1+cu121",
    "torchvision==0.19.1+cu121",
    "torchaudio==2.4.1+cu121",
)
PYTORCH_CU121_INDEX_URL = "https://download.pytorch.org/whl/cu121"

REQUIRED_REPO_PATHS: tuple[str, ...] = (
    "src/tac",
    "experiments/build_pr106_latent_score_table.py",
    "experiments/build_pr106_latent_sidecar.py",
    "experiments/contest_auth_eval.py",
    "scripts/bootstrap_dali_hash_pinned.py",
    "scripts/ensure_remote_pip.sh",
    "scripts/probe_nvdec.sh",
    "scripts/remote_lane_pr106_latent_sidecar.sh",
    "submissions/__init__.py",
    "submissions/pr106_latent_sidecar",
    "tools/tool_bootstrap.py",
)


@dataclass(frozen=True)
class KagglePr106LatentBundleSpec:
    """Inputs for a deterministic Kaggle PR106 latent score-table bundle."""

    username: str
    job_name: str = DEFAULT_JOB_NAME
    slug: str = DEFAULT_KERNEL_SLUG
    title: str = DEFAULT_KERNEL_TITLE
    dataset_ref: str | None = None
    source_dataset_ref: str | None = None
    delta_radius: int = 1
    latent_dim: int = 28
    n_pairs: int = 600
    batch_pairs: int = 2
    candidate_batch_size: int = 8
    sidecar_top_k: int = 600
    pr106_archive_in_bundle: str = DEFAULT_PR106_ARCHIVE_IN_BUNDLE
    source_bundle_name: str = DEFAULT_SOURCE_BUNDLE_NAME

    def score_table_spec(self) -> Pr106LatentScoreTableSpec:
        return Pr106LatentScoreTableSpec(
            job_name=self.job_name,
            pr106_archive=self.pr106_archive_in_bundle,
            delta_radius=self.delta_radius,
            latent_dim=self.latent_dim,
            n_pairs=self.n_pairs,
            batch_pairs=self.batch_pairs,
            candidate_batch_size=self.candidate_batch_size,
            sidecar_top_k=self.sidecar_top_k,
        )


def write_source_bundle(
    *,
    repo_root: Path,
    output_path: Path,
    spec: KagglePr106LatentBundleSpec,
    pr106_archive: Path,
    claims_path: Path,
) -> dict[str, object]:
    """Write the source/runtime/archive tarball consumed by the Kaggle kernel."""

    spec.score_table_spec().validate()
    if not claims_path.is_file():
        raise FileNotFoundError(f"active lane-claim ledger is required: {claims_path}")
    if not pr106_archive.is_file():
        raise FileNotFoundError(f"PR106 archive is required: {pr106_archive}")
    active_claim_row(
        claims_path,
        lane_id=spec.score_table_spec().lane_id,
        instance_job_id=spec.job_name,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    with gzip.GzipFile(filename=str(output_path), mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
            for rel_path in REQUIRED_REPO_PATHS:
                source = repo_root / rel_path
                if not source.exists():
                    raise FileNotFoundError(f"required Kaggle source-bundle path missing: {rel_path}")
                add_path_to_tar(tar, source, Path(rel_path))

            add_path_to_tar(tar, claims_path, Path(".omx/state/active_lane_dispatch_claims.md"))
            add_path_to_tar(tar, pr106_archive, Path(spec.pr106_archive_in_bundle))

    return {
        "schema": "kaggle_pr106_latent_source_bundle_v1",
        "source_bundle": output_path.name,
        "required_repo_paths": list(REQUIRED_REPO_PATHS),
        "claim_ledger": ".omx/state/active_lane_dispatch_claims.md",
        "pr106_archive": spec.pr106_archive_in_bundle,
        "job_name": spec.job_name,
        "lane_id": spec.score_table_spec().lane_id,
        "score_claim": False,
    }


def render_launcher(spec: KagglePr106LatentBundleSpec) -> str:
    """Render the Kaggle script-kernel launcher."""

    env = score_table_env(
        spec.score_table_spec(),
        output_dir="/kaggle/working/pr106_latent_score_table",
    )
    env_literal = repr(env)
    return f'''#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()

PIP_DEPS = (
    "av",
    "brotli",
    "einops",
    "safetensors",
    "segmentation-models-pytorch",
    "timm",
)
PYTORCH_P100_FALLBACK_DEPS = {PYTORCH_P100_FALLBACK_DEPS!r}
PYTORCH_CU121_INDEX_URL = {PYTORCH_CU121_INDEX_URL!r}
UPSTREAM_REPO = "https://github.com/commaai/comma_video_compression_challenge.git"
SOURCE_BUNDLE_NAME = {spec.source_bundle_name!r}
SOURCE_TREE_NAME = SOURCE_BUNDLE_NAME.removesuffix(".tar.gz")
WORKSPACE = Path("/kaggle/working/pact_pr106_latent_workspace")


def _install_missing_deps() -> None:
    missing = []
    import_names = {{
        "av": "av",
        "brotli": "brotli",
        "einops": "einops",
        "safetensors": "safetensors",
        "segmentation-models-pytorch": "segmentation_models_pytorch",
        "timm": "timm",
    }}
    for dep in PIP_DEPS:
        try:
            __import__(import_names[dep])
        except ImportError:
            missing.append(dep)
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


def _ensure_torch_supports_visible_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        return
    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{{major}}{{minor}}"
    arch_list = set(torch.cuda.get_arch_list())
    if arch in arch_list:
        return
    if (major, minor) != (6, 0):
        raise RuntimeError(
            f"Visible CUDA device requires {{arch}}, but torch {{torch.__version__}} "
            f"supports {{sorted(arch_list)}}"
        )
    if os.environ.get("PR106_LATENT_TORCH_FALLBACK_REEXEC") == "1":
        raise RuntimeError(
            f"P100 torch fallback installed but torch {{torch.__version__}} still "
            f"does not support {{arch}}; arch_list={{sorted(arch_list)}}"
        )
    print(
        f"[kaggle] Installing PyTorch P100 fallback for {{arch}}; "
        f"current torch={{torch.__version__}} arch_list={{sorted(arch_list)}}",
        flush=True,
    )
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--force-reinstall",
        "--no-cache-dir",
        *PYTORCH_P100_FALLBACK_DEPS,
        "--index-url",
        PYTORCH_CU121_INDEX_URL,
    ])
    env = os.environ.copy()
    env["PR106_LATENT_TORCH_FALLBACK_REEXEC"] = "1"
    os.execve(sys.executable, [sys.executable, str(SCRIPT_PATH)], env)


def _search_roots() -> list[Path]:
    input_root = Path(os.environ.get("CLOUD_INPUT_ROOT", "/kaggle/input"))
    return [SCRIPT_PATH.parent, input_root]


def _find_source_bundle() -> Path | None:
    for root in _search_roots():
        if not root.exists():
            continue
        hits = sorted(root.rglob(SOURCE_BUNDLE_NAME))
        if hits:
            return hits[0]
    return None


def _find_expanded_source_tree() -> Path | None:
    for root in _search_roots():
        if not root.exists():
            continue
        candidates = [p for p in root.rglob(SOURCE_TREE_NAME) if p.is_dir()]
        candidates.extend(p for p in root.rglob("src") if (p / "tac").is_dir())
        for candidate in sorted(candidates):
            source_tree = candidate if (candidate / "src" / "tac").is_dir() else candidate.parent
            if (source_tree / "src" / "tac").is_dir():
                return source_tree
    return None


def _safe_extract_tarball(bundle: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle, "r:gz") as tar:
        base = destination.resolve()
        for member in tar.getmembers():
            target = destination / member.name
            try:
                target.resolve().relative_to(base)
            except ValueError as exc:
                raise RuntimeError(f"unsafe source-bundle member: {{member.name}}") from exc
        tar.extractall(destination)


def _copy_expanded_source_tree(source_tree: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source_tree, destination)


def _ensure_pr106_archive_zip(workspace: Path) -> None:
    archive = workspace / {spec.pr106_archive_in_bundle!r}
    if archive.is_file():
        return
    extracted = archive.with_suffix("")
    payload = extracted / "0.bin"
    if not payload.is_file():
        raise FileNotFoundError(
            f"required PR106 archive missing: {{archive}}; also missing extracted payload {{payload}}"
        )
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as z:
        info = zipfile.ZipInfo("0.bin", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.external_attr = 0o644 << 16
        z.writestr(info, payload.read_bytes())


def _ensure_git_lfs() -> None:
    if subprocess.run(["bash", "-lc", "command -v git-lfs >/dev/null 2>&1"]).returncode == 0:
        return
    subprocess.check_call(["bash", "-lc", "apt-get update && apt-get install -y git-lfs"])


def _ensure_upstream(workspace: Path) -> Path:
    upstream = workspace / "upstream"
    if (upstream / "models").exists():
        return upstream
    _ensure_git_lfs()
    subprocess.check_call(["git", "clone", "--depth", "1", UPSTREAM_REPO, str(upstream)])
    for attempt in range(1, 4):
        try:
            subprocess.check_call(["git", "lfs", "pull"], cwd=upstream)
            return upstream
        except subprocess.CalledProcessError:
            if attempt == 3:
                raise
            time.sleep(5 * attempt)
    return upstream


def main() -> int:
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    bundle = _find_source_bundle()
    if bundle is not None:
        _safe_extract_tarball(bundle, WORKSPACE)
    else:
        source_tree = _find_expanded_source_tree()
        if source_tree is None:
            raise FileNotFoundError(
                f"required source bundle {{SOURCE_BUNDLE_NAME!r}} or expanded tree "
                f"{{SOURCE_TREE_NAME!r}} not found under {{[str(root) for root in _search_roots()]}}"
            )
        _copy_expanded_source_tree(source_tree, WORKSPACE)
    workspace = WORKSPACE
    _ensure_pr106_archive_zip(workspace)
    sys.path.insert(0, str(workspace / "src"))
    sys.path.insert(0, str(workspace))
    import tac.deploy.pr106_latent  # noqa: F401
    _install_missing_deps()
    _ensure_torch_supports_visible_cuda()
    upstream = _ensure_upstream(workspace)

    pythonpath_parts = [
        str(workspace / "src"),
        str(workspace),
        str(upstream),
        os.environ.get("PYTHONPATH", ""),
    ]
    env = os.environ.copy()
    env.update({env_literal})
    env.update({{
        "CLOUD_PLATFORM": "kaggle",
        "PYBIN": sys.executable,
        "PYTHONPATH": ":".join(part for part in pythonpath_parts if part),
        "TAC_UPSTREAM_DIR": str(upstream),
        "WORKSPACE": str(workspace),
    }})

    out_dir = Path(env["PR106_LATENT_LOG_DIR"]).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["bash", {REMOTE_SCRIPT!r}],
        cwd=workspace,
        env=env,
        text=True,
        check=False,
    )
    summary = {{
        "schema": "kaggle_pr106_latent_score_table_run_v1",
        "score_claim": False,
        "promotion_requires_adjudication": True,
        "source_bundle": str(bundle),
        "returncode": result.returncode,
        "job_name": {spec.job_name!r},
        "lane_id": env["PR106_LATENT_SCORE_TABLE_LANE_ID"],
        "log_dir": env["PR106_LATENT_LOG_DIR"],
        "contest_auth_eval_json": str(Path(env["PR106_LATENT_LOG_DIR"]) / "eval" / "contest_auth_eval.json"),
    }}
    (out_dir / "kaggle_pr106_latent_score_table_summary.json").write_text(
        json.dumps(summary, indent=2) + "\\n",
        encoding="utf-8",
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
'''


def write_bundle(
    *,
    repo_root: Path,
    bundle_dir: Path,
    spec: KagglePr106LatentBundleSpec,
    pr106_archive: Path,
    claims_path: Path,
) -> dict[str, object]:
    """Write a deterministic private Kaggle kernel bundle."""

    spec.score_table_spec().validate()
    if not claims_path.is_file():
        raise FileNotFoundError(f"active lane-claim ledger is required: {claims_path}")
    if not pr106_archive.is_file():
        raise FileNotFoundError(f"PR106 archive is required: {pr106_archive}")
    active_claim_row(
        claims_path,
        lane_id=spec.score_table_spec().lane_id,
        instance_job_id=spec.job_name,
    )

    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    metadata = build_kernel_metadata(
        username=spec.username,
        slug=spec.slug,
        title=spec.title,
        code_file="run_kernel.py",
        dataset_sources=dataset_sources(spec),
        launch_policy={
            "score_claim": False,
            "provider": "kaggle",
            "lane_id": spec.score_table_spec().lane_id,
            "job_name": spec.job_name,
            "promotion_requires": "contest_auth_eval_json_adjudication",
        },
    )
    (bundle_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (bundle_dir / "run_kernel.py").write_text(render_launcher(spec), encoding="utf-8")

    manifest = {
        "schema": "kaggle_pr106_latent_score_table_bundle_v1",
        "score_claim": False,
        "kernel_metadata": "kernel-metadata.json",
        "code_file": "run_kernel.py",
        "dataset_sources": list(dataset_sources(spec)),
        "source_bundle_name": spec.source_bundle_name,
        "job_name": spec.job_name,
        "lane_id": spec.score_table_spec().lane_id,
    }
    (bundle_dir / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


__all__ = [
    "DEFAULT_DATASET_SLUG",
    "DEFAULT_JOB_NAME",
    "DEFAULT_KERNEL_SLUG",
    "DEFAULT_KERNEL_TITLE",
    "DEFAULT_PR106_ARCHIVE_IN_BUNDLE",
    "DEFAULT_SOURCE_BUNDLE_NAME",
    "DEFAULT_SOURCE_DATASET_SLUG",
    "KagglePr106LatentBundleSpec",
    "REQUIRED_REPO_PATHS",
    "render_launcher",
    "write_bundle",
    "write_source_bundle",
]
