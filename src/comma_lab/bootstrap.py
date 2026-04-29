from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .paths import default_upstream_root, upstream_snapshot_path
from .snapshot import write_snapshot


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, check=check, text=True)  # subprocess-no-check-OK: wrapper threads check= keyword (default True per signature)


def bootstrap_upstream(dest: Path | None = None, repo_url: str = "https://github.com/commaai/comma_video_compression_challenge.git", do_lfs: bool = True) -> dict:
    dest = dest or default_upstream_root()
    dest.parent.mkdir(parents=True, exist_ok=True)

    if (dest / ".git").exists():
        run(["git", "fetch", "origin", "master", "--depth=1"], cwd=dest)
        run(["git", "checkout", "master"], cwd=dest)
        run(["git", "pull", "--ff-only"], cwd=dest)
    else:
        run(["git", "clone", "--depth=1", repo_url, str(dest)])

    if do_lfs and shutil.which("git-lfs"):
        try:
            run(["git", "lfs", "install"], cwd=dest)
            run(["git", "lfs", "pull"], cwd=dest)
        except subprocess.CalledProcessError:
            pass

    return write_snapshot(dest, upstream_snapshot_path())
