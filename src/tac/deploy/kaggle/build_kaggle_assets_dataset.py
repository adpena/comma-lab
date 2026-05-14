#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
DATASET_DIR = REPO_ROOT / "experiments" / "kaggle_datasets" / "comma_lab_private_assets"
ARCHIVE_PATH = REPO_ROOT / "reports" / "raw" / "2026-04-06-av1-roi-experiments" / "decode_base_archive.zip"
SALIENCY_PATH = REPO_ROOT / "reports" / "masks" / "posenet_saliency.npy"
README_PATH = DATASET_DIR / "README.md"
METADATA_PATH = DATASET_DIR / "dataset-metadata.json"


def build_repo_wheel() -> Path:
    dist_dir = Path(tempfile.mkdtemp(prefix="comma_lab_wheel_"))
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            str(REPO_ROOT),
            "--no-deps",
            "-w",
            str(dist_dir),
        ]
    )
    wheels = sorted(dist_dir.glob("comma_video_lab_ball_pack-*.whl"))
    if not wheels:
        raise FileNotFoundError(f"No wheel built in {dist_dir}")
    return wheels[0]


def render_dataset_readme(*, wheel_name: str, archive_name: str, saliency_name: str) -> str:
    return (
        "# comma-lab private assets\n\n"
        "Private Kaggle dataset used only to stage small runtime assets for kernel runs.\n\n"
        "Current contents:\n"
        f"- `{wheel_name}`\n"
        f"- `{archive_name}`\n"
        f"- `{saliency_name}`\n"
    )


def stage_assets_dataset(
    *,
    dataset_dir: Path,
    wheel_path: Path,
    archive_path: Path,
    saliency_path: Path,
    metadata_path: Path,
) -> dict[str, object]:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for source in (metadata_path, wheel_path, archive_path, saliency_path):
        destination = dataset_dir / source.name
        if source.resolve() == destination.resolve():
            continue
        shutil.copy2(source, destination)
    (dataset_dir / "README.md").write_text(
        render_dataset_readme(
            wheel_name=wheel_path.name,
            archive_name=archive_path.name,
            saliency_name=saliency_path.name,
        )
    )
    return {
        "dataset_dir": str(dataset_dir),
        "wheel_name": wheel_path.name,
        "archive_name": archive_path.name,
        "saliency_name": saliency_path.name,
    }


def push_dataset_version(dataset_dir: Path, *, message: str) -> int:
    result = subprocess.run(
        [
            "uv",
            "run",
            "--with",
            "kaggle",
            "kaggle",
            "datasets",
            "version",
            "-p",
            str(dataset_dir),
            "-m",
            message,
            "-r",
            "zip",
        ],
        check=False,
    )
    return result.returncode


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage and optionally version the private Kaggle asset dataset.")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--archive", type=Path, default=ARCHIVE_PATH)
    parser.add_argument("--saliency", type=Path, default=SALIENCY_PATH)
    parser.add_argument("--metadata", type=Path, default=METADATA_PATH)
    parser.add_argument("--push", action="store_true", help="Create a new Kaggle dataset version after staging")
    parser.add_argument("--message", default="stage tac wheel + saliency for kaggle kernels")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    wheel_path = build_repo_wheel()
    staged = stage_assets_dataset(
        dataset_dir=args.dataset_dir,
        wheel_path=wheel_path,
        archive_path=args.archive,
        saliency_path=args.saliency,
        metadata_path=args.metadata,
    )
    print(f"staged {staged['wheel_name']} into {staged['dataset_dir']}")
    if not args.push:
        return 0
    return push_dataset_version(args.dataset_dir, message=args.message)


if __name__ == "__main__":
    raise SystemExit(main())
