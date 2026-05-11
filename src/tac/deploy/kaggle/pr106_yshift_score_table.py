"""Kaggle bundle writer for the PR106 y-shift score-table lane.

The lane itself is provider-neutral in :mod:`tac.deploy.pr106_yshift`.  This
module owns only the Kaggle script-kernel packaging: generated launcher source,
minimal runtime/source copies, private-GPU kernel metadata, and the active
dispatch-claim ledger that the CUDA score-table producer verifies before doing
scorer work.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from tac.deploy.cloud_bootstrap import BOOTSTRAP_STUB
from tac.deploy.claims import active_claim_row
from tac.deploy.kaggle.kaggle_kernel_builder import build_kernel_metadata
from tac.deploy.pr106_yshift import (
    REMOTE_SCRIPT,
    Pr106YshiftScoreTableSpec,
    score_table_env,
)

DEFAULT_DATASET_SLUG = "comma-lab-private-assets"
DEFAULT_KERNEL_SLUG = "comma-lab-pr106-yshift-score-table"
DEFAULT_KERNEL_TITLE = "comma-lab PR106 yshift score-table"
DEFAULT_JOB_NAME = "kaggle_pr106_yshift_score_table"
DEFAULT_PR106_ARCHIVE_IN_BUNDLE = "inputs/pr106_archive.zip"

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
    candidate_radius: int = 3
    score_step: float = 1.0
    n_pairs: int = 600
    batch_pairs: int = 8
    candidate_batch_size: int = 32
    pr106_archive_in_bundle: str = DEFAULT_PR106_ARCHIVE_IN_BUNDLE

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


def _copy_tree_filtered(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "*.pyc",
            ".pytest_cache",
            ".ruff_cache",
            ".mypy_cache",
        ),
    )


def _copy_repo_path(repo_root: Path, rel_path: str, bundle_dir: Path) -> None:
    source = repo_root / rel_path
    if not source.exists():
        raise FileNotFoundError(f"required Kaggle bundle path missing: {rel_path}")
    destination = bundle_dir / rel_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        _copy_tree_filtered(source, destination)
    else:
        shutil.copy2(source, destination)


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
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()

{BOOTSTRAP_STUB}

PIP_DEPS = (
    "av",
    "brotli",
    "einops",
    "safetensors",
    "segmentation-models-pytorch",
    "timm",
)
UPSTREAM_REPO = "https://github.com/commaai/comma_video_compression_challenge.git"


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
    workspace = SCRIPT_PATH.parent
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.path.insert(0, str(workspace / "src"))
    sys.path.insert(0, str(workspace))
    _tac_bootstrap(
        dataset_hint={DEFAULT_DATASET_SLUG!r},
        verify_submodule="tac.deploy.pr106_yshift",
        extra_search_dirs=(str(workspace),),
    )
    _install_missing_deps()
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
        "schema": "kaggle_pr106_yshift_score_table_run_v1",
        "score_claim": False,
        "promotion_requires_adjudication": True,
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
        dataset_sources=(spec.dataset_ref,) if spec.dataset_ref else (),
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

    for rel_path in REQUIRED_REPO_PATHS:
        _copy_repo_path(repo_root, rel_path, bundle_dir)

    claim_destination = bundle_dir / ".omx" / "state" / "active_lane_dispatch_claims.md"
    claim_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(claims_path, claim_destination)

    archive_destination = bundle_dir / spec.pr106_archive_in_bundle
    archive_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pr106_archive, archive_destination)

    manifest = {
        "schema": "kaggle_pr106_yshift_score_table_bundle_v1",
        "score_claim": False,
        "kernel_metadata": "kernel-metadata.json",
        "code_file": "run_kernel.py",
        "required_repo_paths": list(REQUIRED_REPO_PATHS),
        "claim_ledger": ".omx/state/active_lane_dispatch_claims.md",
        "pr106_archive": spec.pr106_archive_in_bundle,
        "job_name": spec.job_name,
        "lane_id": spec.score_table_spec().lane_id,
    }
    (bundle_dir / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


__all__ = [
    "DEFAULT_JOB_NAME",
    "DEFAULT_KERNEL_SLUG",
    "DEFAULT_KERNEL_TITLE",
    "DEFAULT_PR106_ARCHIVE_IN_BUNDLE",
    "KagglePr106YshiftBundleSpec",
    "REQUIRED_REPO_PATHS",
    "render_launcher",
    "write_bundle",
]
