from __future__ import annotations

import shutil
from pathlib import Path

from .paths import default_upstream_root, repo_root


INSTALL_PAYLOADS: dict[str, tuple[str, ...]] = {
    "exact_current": (
        "archive.zip",
        "inflate.sh",
        "inflate.py",
    ),
    "robust_current": (
        "archive.zip",
        "inflate.sh",
        "inflate.py",
        "inflate_postfilter.py",
        "inflate_grain_mask.py",
        "postfilter_int8.pt",
        "config.env",
        "analyze_roi.py",
    ),
}


def install_payload_paths(name: str, source_submission_dir: Path) -> list[Path]:
    rel_paths = INSTALL_PAYLOADS.get(name)
    if rel_paths is None:
        raise ValueError(f"Unsupported submission payload: {name}")

    payloads: list[Path] = []
    for rel in rel_paths:
        path = source_submission_dir / rel
        if not path.exists():
            raise FileNotFoundError(f"Required payload entry missing for {name}: {path}")
        if path.is_dir():
            payloads.extend(sorted(p for p in path.rglob("*") if p.is_file()))
        else:
            payloads.append(path)
    return payloads


def install_payload_manifest(name: str, source_submission_dir: Path) -> list[tuple[str, int]]:
    return [
        (path.relative_to(source_submission_dir).as_posix(), path.stat().st_size)
        for path in install_payload_paths(name, source_submission_dir)
    ]


def install_payload_bytes(name: str, source_submission_dir: Path) -> int:
    return sum(size for _, size in install_payload_manifest(name, source_submission_dir))


def install_submission(name: str, upstream_root: Path | None = None, force: bool = True) -> Path:
    upstream_root = upstream_root or default_upstream_root()
    src_dir = repo_root() / "submissions" / name
    dst_dir = upstream_root / "submissions" / name

    if not src_dir.exists():
        raise FileNotFoundError(f"starter-pack submission not found: {src_dir}")
    if not upstream_root.exists():
        raise FileNotFoundError(f"upstream repo not found: {upstream_root}")

    if dst_dir.exists() and force:
        shutil.rmtree(dst_dir)
    elif dst_dir.exists() and not force:
        raise FileExistsError(f"destination already exists: {dst_dir}")

    dst_dir.mkdir(parents=True, exist_ok=True)
    for src_path in install_payload_paths(name, src_dir):
        rel_path = src_path.relative_to(src_dir)
        dst_path = dst_dir / rel_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)
    return dst_dir
