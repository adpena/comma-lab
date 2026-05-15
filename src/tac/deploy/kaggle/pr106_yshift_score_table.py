# SPDX-License-Identifier: MIT
"""Kaggle bundle writer for the PR106 y-shift score-table lane.

The lane itself is provider-neutral in :mod:`tac.deploy.pr106_yshift`.  This
module owns only the Kaggle script-kernel packaging: generated launcher source,
source-dataset tarball creation, private-GPU kernel metadata, and the active
dispatch-claim ledger that the CUDA score-table producer verifies before doing
scorer work.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from tac.deploy.claims import active_claim_row
from tac.deploy.kaggle.kaggle_kernel_builder import build_kernel_metadata
from tac.deploy.kaggle.source_bundle import (
    add_bytes_to_tar,
    add_path_to_tar,
    dataset_sources,
    open_deterministic_tar_gz,
)
from tac.deploy.pr106_yshift import (
    REMOTE_SCRIPT,
    Pr106YshiftScoreTableSpec,
    score_table_env,
)

DEFAULT_DATASET_SLUG = "comma-lab-private-assets"
DEFAULT_SOURCE_DATASET_SLUG = "comma-lab-pr106-yshift-source"
DEFAULT_KERNEL_SLUG = "comma-lab-pr106-yshift-score-table"
DEFAULT_KERNEL_TITLE = "comma-lab PR106 yshift score-table"
DEFAULT_JOB_NAME = "kaggle_pr106_yshift_score_table"
DEFAULT_PR106_ARCHIVE_IN_BUNDLE = "inputs/pr106_archive.zip"
DEFAULT_SOURCE_BUNDLE_NAME = "pact_pr106_yshift_source_bundle.tar.gz"
SOURCE_BUNDLE_MANIFEST = "source_bundle_manifest.json"

REQUIRED_REPO_PATHS: tuple[str, ...] = (
    "src/tac",
    "experiments/build_pr106_yshift_score_table.py",
    "experiments/build_pr106_yshift_sidechannel.py",
    "experiments/contest_auth_eval.py",
    "scripts/bootstrap_dali_hash_pinned.py",
    "scripts/ensure_remote_pip.sh",
    "scripts/probe_nvdec.sh",
    "scripts/remote_lane_pr106_yshift_sidechannel.sh",
    "submissions/__init__.py",
    "submissions/apogee_intN/src",
    "submissions/pr106_yshift_sidechannel",
    "tools/tool_bootstrap.py",
)


@dataclass(frozen=True)
class KagglePr106YshiftBundleSpec:
    """Inputs for a deterministic Kaggle PR106 y-shift score-table bundle."""

    username: str
    job_name: str = DEFAULT_JOB_NAME
    slug: str = DEFAULT_KERNEL_SLUG
    title: str = DEFAULT_KERNEL_TITLE
    dataset_ref: str | None = None
    source_dataset_ref: str | None = None
    candidate_radius: int = 3
    score_step: float = 1.0
    n_pairs: int = 600
    batch_pairs: int = 8
    candidate_batch_size: int = 32
    pr106_archive_in_bundle: str = DEFAULT_PR106_ARCHIVE_IN_BUNDLE
    source_bundle_name: str = DEFAULT_SOURCE_BUNDLE_NAME

    def score_table_spec(self) -> Pr106YshiftScoreTableSpec:
        return Pr106YshiftScoreTableSpec(
            job_name=self.job_name,
            pr106_archive=self.pr106_archive_in_bundle,
            candidate_radius=self.candidate_radius,
            score_step=self.score_step,
            n_pairs=self.n_pairs,
            batch_pairs=self.batch_pairs,
            candidate_batch_size=self.candidate_batch_size,
        )


def write_source_bundle(
    *,
    repo_root: Path,
    output_path: Path,
    spec: KagglePr106YshiftBundleSpec,
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

    pr106_archive_sha256 = hashlib.sha256(pr106_archive.read_bytes()).hexdigest()
    manifest = {
        "schema": "kaggle_pr106_yshift_source_bundle_v1",
        "source_bundle": spec.source_bundle_name,
        "required_repo_paths": list(REQUIRED_REPO_PATHS),
        "claim_ledger": ".omx/state/active_lane_dispatch_claims.md",
        "pr106_archive": spec.pr106_archive_in_bundle,
        "pr106_archive_sha256": pr106_archive_sha256,
        "job_name": spec.job_name,
        "lane_id": spec.score_table_spec().lane_id,
        "candidate_radius": spec.candidate_radius,
        "score_step": spec.score_step,
        "score_claim": False,
    }

    with open_deterministic_tar_gz(output_path) as tar:
        add_bytes_to_tar(
            tar,
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
            Path(SOURCE_BUNDLE_MANIFEST),
        )
        for rel_path in REQUIRED_REPO_PATHS:
            source = repo_root / rel_path
            if not source.exists():
                raise FileNotFoundError(f"required Kaggle source-bundle path missing: {rel_path}")
            add_path_to_tar(tar, source, Path(rel_path))

        add_path_to_tar(tar, claims_path, Path(".omx/state/active_lane_dispatch_claims.md"))
        add_path_to_tar(tar, pr106_archive, Path(spec.pr106_archive_in_bundle))

    return manifest


def render_launcher(spec: KagglePr106YshiftBundleSpec) -> str:
    """Render the Kaggle script-kernel launcher."""

    env = score_table_env(
        spec.score_table_spec(),
        output_dir="/kaggle/working/pr106_yshift_score_table",
    )
    env_literal = repr(env)
    return f'''#!/usr/bin/env python3
from __future__ import annotations

import json
import hashlib
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
PYTORCH_P100_FALLBACK_DEPS = (
    "torch==2.4.1+cu121",
    "torchvision==0.19.1+cu121",
    "torchaudio==2.4.1+cu121",
)
PYTORCH_CU121_INDEX_URL = "https://download.pytorch.org/whl/cu121"
UPSTREAM_REPO = "https://github.com/commaai/comma_video_compression_challenge.git"
SOURCE_BUNDLE_NAME = {spec.source_bundle_name!r}
SOURCE_BUNDLE_MANIFEST = {SOURCE_BUNDLE_MANIFEST!r}
SOURCE_TREE_NAME = SOURCE_BUNDLE_NAME.removesuffix(".tar.gz")
WORKSPACE = Path("/kaggle/working/pact_pr106_yshift_workspace")


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
    if os.environ.get("PR106_YSHIFT_TORCH_FALLBACK_REEXEC") == "1":
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
    env["PR106_YSHIFT_TORCH_FALLBACK_REEXEC"] = "1"
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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json_object(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{{path}} must contain a JSON object")
    return payload


def _is_verified_source_bundle_tree(source_tree: Path) -> bool:
    archive = source_tree / {spec.pr106_archive_in_bundle!r}
    extracted_payload = archive.with_suffix("") / "0.bin"
    required_dirs = (
        source_tree / "src" / "tac",
        source_tree / "submissions" / "pr106_yshift_sidechannel",
    )
    required_files = (
        source_tree / SOURCE_BUNDLE_MANIFEST,
        source_tree / ".omx" / "state" / "active_lane_dispatch_claims.md",
        source_tree / "experiments" / "build_pr106_yshift_score_table.py",
        source_tree / {REMOTE_SCRIPT!r},
    )
    if source_tree.name != SOURCE_TREE_NAME:
        return False
    if not all(path.is_dir() for path in required_dirs):
        return False
    if not all(path.is_file() for path in required_files):
        return False
    try:
        manifest = _read_json_object(source_tree / SOURCE_BUNDLE_MANIFEST)
    except Exception:
        return False
    expected_manifest = {{
        "schema": "kaggle_pr106_yshift_source_bundle_v1",
        "job_name": {spec.job_name!r},
        "lane_id": {spec.score_table_spec().lane_id!r},
        "candidate_radius": {spec.candidate_radius!r},
        "score_step": {spec.score_step!r},
        "score_claim": False,
    }}
    if any(manifest.get(key) != value for key, value in expected_manifest.items()):
        return False
    expected_archive_sha256 = manifest.get("pr106_archive_sha256")
    if not isinstance(expected_archive_sha256, str) or len(expected_archive_sha256) != 64:
        return False
    if archive.is_file() and extracted_payload.exists():
        return False
    if archive.is_file():
        return _sha256_file(archive) == expected_archive_sha256
    archive_dir = archive.with_suffix("")
    if not extracted_payload.is_file():
        return False
    extra_members = [p for p in archive_dir.iterdir() if p.name != "0.bin"]
    return not extra_members


def _find_verified_expanded_source_bundle_tree() -> Path | None:
    for root in _search_roots():
        if not root.exists():
            continue
        for candidate in sorted(root.rglob(SOURCE_TREE_NAME)):
            if candidate.is_dir() and _is_verified_source_bundle_tree(candidate):
                return candidate
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
    manifest_path = workspace / SOURCE_BUNDLE_MANIFEST
    expected_archive_sha256 = None
    if manifest_path.is_file():
        manifest = _read_json_object(manifest_path)
        value = manifest.get("pr106_archive_sha256")
        if isinstance(value, str):
            expected_archive_sha256 = value
    extracted = archive.with_suffix("")
    payload = extracted / "0.bin"
    if archive.is_file():
        if payload.exists():
            raise FileExistsError(f"ambiguous PR106 archive inputs: both {{archive}} and {{payload}} exist")
        if expected_archive_sha256 is not None:
            archive_sha = _sha256_file(archive)
            if archive_sha != expected_archive_sha256:
                raise RuntimeError(
                    f"PR106 archive SHA mismatch after source-bundle staging: "
                    f"got {{archive_sha}} expected {{expected_archive_sha256}}"
                )
        return
    if not payload.is_file():
        raise FileNotFoundError(
            f"required PR106 archive missing: {{archive}}; also missing extracted payload {{payload}}"
        )
    extra_members = [p for p in extracted.iterdir() if p.name != "0.bin"]
    if extra_members:
        raise RuntimeError(f"unexpected extracted PR106 archive members: {{[p.name for p in extra_members]}}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as z:
        info = zipfile.ZipInfo("0.bin", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.external_attr = 0o644 << 16
        z.writestr(info, payload.read_bytes())
    if expected_archive_sha256 is not None:
        archive_sha = _sha256_file(archive)
        if archive_sha != expected_archive_sha256:
            raise RuntimeError(
                f"reconstructed PR106 archive SHA mismatch: got {{archive_sha}} "
                f"expected {{expected_archive_sha256}}"
            )


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
    source_tree = None
    source_tree_verified = False
    if bundle is not None:
        _safe_extract_tarball(bundle, WORKSPACE)
    else:
        source_tree = _find_verified_expanded_source_bundle_tree()
        if source_tree is not None:
            source_tree_verified = True
            _copy_expanded_source_tree(source_tree, WORKSPACE)
        elif os.environ.get("PR106_YSHIFT_ALLOW_EXPANDED_SOURCE_TREE") != "1":
            raise FileNotFoundError(
                f"required source bundle {{SOURCE_BUNDLE_NAME!r}} or verified expanded "
                f"source-bundle tree {{SOURCE_TREE_NAME!r}} not found under "
                f"{{[str(root) for root in _search_roots()]}}; refusing expanded-source fallback "
                "for Kaggle custody. Set PR106_YSHIFT_ALLOW_EXPANDED_SOURCE_TREE=1 only for "
                "local developer smoke tests."
            )
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
    import tac.deploy.pr106_yshift  # noqa: F401
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
        "PR106_YSHIFT_ALLOW_PROVIDER_CLAIM_MIRROR": "1",
        "PYBIN": sys.executable,
        "PYTHONPATH": ":".join(part for part in pythonpath_parts if part),
        "TAC_UPSTREAM_DIR": str(upstream),
        "WORKSPACE": str(workspace),
    }})

    out_dir = Path(env["PR106_YSHIFT_LOG_DIR"]).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["bash", {REMOTE_SCRIPT!r}],
        cwd=workspace,
        env=env,
        text=True,
        check=False,
    )
    summary = {{
        "schema": "kaggle_pr106_yshift_score_table_run_v2",
        "score_claim": False,
        "promotion_requires_adjudication": True,
        "source_bundle": str(bundle) if bundle is not None else None,
        "source_tree": str(source_tree) if source_tree is not None else None,
        "source_tree_verified": source_tree_verified,
        "returncode": result.returncode,
        "job_name": {spec.job_name!r},
        "lane_id": env["PR106_YSHIFT_SCORE_TABLE_LANE_ID"],
        "log_dir": env["PR106_YSHIFT_LOG_DIR"],
        "contest_auth_eval_json": str(Path(env["PR106_YSHIFT_LOG_DIR"]) / "eval" / "contest_auth_eval.json"),
    }}
    (out_dir / "kaggle_pr106_yshift_score_table_summary.json").write_text(
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
    spec: KagglePr106YshiftBundleSpec,
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
    inline_source_manifest = write_source_bundle(
        repo_root=repo_root,
        output_path=bundle_dir / spec.source_bundle_name,
        spec=spec,
        pr106_archive=pr106_archive,
        claims_path=claims_path,
    )

    manifest = {
        "schema": "kaggle_pr106_yshift_score_table_bundle_v2",
        "score_claim": False,
        "kernel_metadata": "kernel-metadata.json",
        "code_file": "run_kernel.py",
        "dataset_sources": list(dataset_sources(spec)),
        "inline_source_bundle": spec.source_bundle_name,
        "inline_source_manifest": inline_source_manifest,
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
    "REQUIRED_REPO_PATHS",
    "KagglePr106YshiftBundleSpec",
    "render_launcher",
    "write_bundle",
    "write_source_bundle",
]
