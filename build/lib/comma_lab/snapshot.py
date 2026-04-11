from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PINNED_FILES = [
    "README.md",
    "evaluate.py",
    "evaluate.sh",
    "frame_utils.py",
    "modules.py",
    "public_test_video_names.txt",
    ".github/workflows/eval.yml",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_rev_parse(root: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return proc.stdout.strip() or None


def make_snapshot(upstream_root: Path) -> dict:
    files = {}
    for rel in PINNED_FILES:
        path = upstream_root / rel
        files[rel] = sha256_file(path) if path.exists() else None

    public_names = []
    names_file = upstream_root / "public_test_video_names.txt"
    if names_file.exists():
        public_names = [line.strip() for line in names_file.read_text().splitlines() if line.strip()]

    return {
        "repo_url": "https://github.com/commaai/comma_video_compression_challenge.git",
        "branch": "master",
        "last_verified_at": datetime.now(timezone.utc).isoformat(),
        "commit": git_rev_parse(upstream_root),
        "public_test_video_names": public_names,
        "files": files,
    }


def write_snapshot(upstream_root: Path, out_path: Path) -> dict:
    snapshot = make_snapshot(upstream_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(snapshot, indent=2) + "\n")
    return snapshot


def load_snapshot(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())
