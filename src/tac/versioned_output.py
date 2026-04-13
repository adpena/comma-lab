"""Versioned output utility: write timestamped files and maintain a latest symlink.

Every data visualization / report / experiment script should use
``versioned_write`` instead of writing directly to a fixed path.  This
guarantees that previous outputs are never silently overwritten.

Usage::

    from tac.versioned_output import versioned_write

    versioned_write(
        base_path=Path("reports/graphs/dashboard_data.json"),
        content=json.dumps(data, indent=2),
        config_tag="robust_current",
    )
"""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path


def versioned_write(
    base_path: Path,
    content: str | bytes,
    *,
    config_tag: str = "",
) -> Path:
    """Write *content* to a timestamped file and point *base_path* at it.

    Parameters
    ----------
    base_path:
        The canonical output path (e.g. ``reports/graphs/dashboard_data.json``).
        A symlink at this location will always point to the latest versioned
        file.
    content:
        File content -- ``str`` for text files, ``bytes`` for binary.
    config_tag:
        Optional short identifier (model name, submission name, config slug).

    Returns
    -------
    Path to the versioned file that was actually written.
    """
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = _sanitize_tag(config_tag)
    suffix = base_path.suffix
    stem = base_path.stem

    parts = [stem, timestamp]
    if tag:
        parts.append(tag)
    versioned_name = "_".join(parts) + suffix
    versioned_path = base_path.parent / versioned_name

    if isinstance(content, bytes):
        versioned_path.write_bytes(content)
    else:
        versioned_path.write_text(content)

    _update_latest_link(base_path, versioned_path)
    return versioned_path


def versioned_copy(
    base_path: Path,
    source_path: Path,
    *,
    config_tag: str = "",
) -> Path:
    """Copy *source_path* to a timestamped name and point *base_path* at it."""
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = _sanitize_tag(config_tag)
    suffix = base_path.suffix
    stem = base_path.stem

    parts = [stem, timestamp]
    if tag:
        parts.append(tag)
    versioned_name = "_".join(parts) + suffix
    versioned_path = base_path.parent / versioned_name

    shutil.copy2(source_path, versioned_path)
    _update_latest_link(base_path, versioned_path)
    return versioned_path


def _sanitize_tag(tag: str) -> str:
    """Remove characters that are unsafe in filenames."""
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in tag).strip("_")


def _update_latest_link(link_path: Path, target_path: Path) -> None:
    """Create or update a symlink at *link_path* -> *target_path*.

    Falls back to a plain copy on platforms where symlinks are unreliable.
    """
    try:
        if link_path.is_symlink() or link_path.exists():
            link_path.unlink()
        link_path.symlink_to(target_path.name)
    except OSError:
        shutil.copy2(target_path, link_path)
