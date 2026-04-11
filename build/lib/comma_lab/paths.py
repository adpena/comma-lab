from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def workspace_root() -> Path:
    return repo_root() / "workspace"


def default_upstream_root() -> Path:
    return workspace_root() / "upstream" / "comma_video_compression_challenge"


def upstream_snapshot_path() -> Path:
    return workspace_root() / "upstream_snapshot.json"
