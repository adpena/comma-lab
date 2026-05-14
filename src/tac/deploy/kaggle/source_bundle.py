# SPDX-License-Identifier: MIT
"""Shared Kaggle source-bundle helpers for score-table kernels."""
from __future__ import annotations

import gzip
import tarfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol


class DatasetSourceSpec(Protocol):
    dataset_ref: str | None
    source_dataset_ref: str | None


def dataset_sources(spec: DatasetSourceSpec) -> tuple[str, ...]:
    """Return Kaggle dataset refs in deterministic mount order."""
    sources: list[str] = []
    for ref in (spec.source_dataset_ref, spec.dataset_ref):
        if ref and ref not in sources:
            sources.append(ref)
    return tuple(sources)


def deterministic_tar_filter(info: tarfile.TarInfo) -> tarfile.TarInfo:
    """Normalize tar metadata so source bundles are reproducible."""
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    if info.isdir():
        info.mode = 0o755
    elif info.isfile():
        info.mode = 0o644
    return info


def include_in_source_bundle(path: Path) -> bool:
    """Reject local cache files from source/runtime Kaggle bundles."""
    ignored_dirs = {
        "__pycache__",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
    }
    if any(part in ignored_dirs for part in path.parts):
        return False
    return path.suffix not in {".pyc", ".pyo"}


def add_path_to_tar(
    tar: tarfile.TarFile,
    source: Path,
    arcname: Path,
) -> None:
    """Add a file or directory to a deterministic Kaggle source tarball."""
    if source.is_dir():
        for path in sorted(source.rglob("*")):
            rel = path.relative_to(source)
            if not include_in_source_bundle(rel):
                continue
            tar.add(path, arcname=str(arcname / rel), recursive=False, filter=deterministic_tar_filter)
    else:
        tar.add(source, arcname=str(arcname), recursive=False, filter=deterministic_tar_filter)


@contextmanager
def open_deterministic_tar_gz(output_path: Path) -> Iterator[tarfile.TarFile]:
    """Open a reproducible gzip-compressed tar writer.

    ``gzip.GzipFile(filename=...)`` writes the output file name into the gzip
    header.  Kaggle source bundles must be byte-reproducible across staging
    directories and file names, so the original-name header is deliberately
    empty while tar member metadata is normalized by ``add_path_to_tar``.
    """

    with (
        output_path.open("wb") as fh,
        gzip.GzipFile(filename="", mode="wb", fileobj=fh, mtime=0) as gz,
        tarfile.open(fileobj=gz, mode="w") as tar,
    ):
        yield tar


__all__ = [
    "DatasetSourceSpec",
    "add_path_to_tar",
    "dataset_sources",
    "deterministic_tar_filter",
    "include_in_source_bundle",
    "open_deterministic_tar_gz",
]
