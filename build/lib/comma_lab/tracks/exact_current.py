from __future__ import annotations

from pathlib import Path
import zipfile


def create_minimal_archive(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.writestr("README.txt", "starter-pack exact-current archive placeholder\n")
    return path
