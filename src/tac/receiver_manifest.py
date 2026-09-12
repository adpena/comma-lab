"""Canonical writer and transactional row rebind for receiver ``MANIFEST.sha256``.

The byte format and shipped-file census are the promoted-tree rule originally
proved by ``experiments.ddm_ntb2_public.regenerate_manifest``.  Keep the writer
here so archive-pin producers and public-proof tools share one implementation.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

from tac.candidate_seal import SealContractError, runtime_digest_skip_reason, sha256_file

MANIFEST_NAME = "MANIFEST.sha256"


def shipped_files(root: Path) -> tuple[Path, ...]:
    """Return the sorted files that can ship in a receiver runtime tree."""

    root = Path(root)
    return tuple(
        path
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and not path.is_symlink()
        and runtime_digest_skip_reason(path.relative_to(root).as_posix()) is None
    )


def canonical_receiver_manifest_bytes(
    root: Path, *, include_archive: bool = False
) -> bytes:
    """Render the canonical ``sha256  relative/path`` listing in memory."""

    root = Path(root)
    rows = "".join(
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n"
        for path in shipped_files(root)
        if path.name != MANIFEST_NAME
        and (include_archive or path.relative_to(root).as_posix() != "archive.zip")
    )
    return rows.encode("utf-8")


def _listing(root: Path) -> dict[str, str]:
    # Reuse the seal-inputs validator's parser rather than defining a second
    # MANIFEST.sha256 grammar here.
    from tac.decode_wall_clock import _manifest_listing

    listing = _manifest_listing(root)
    if listing is None:
        raise SealContractError(f"decode_wall_clock: receiver tree lacks {MANIFEST_NAME}")
    return listing


def require_receiver_manifest_rows(root: Path, relative_paths: Iterable[str]) -> tuple[str, ...]:
    """Refuse before a producer writes when the manifest cannot track every target."""

    root = Path(root)
    listing = _listing(root)
    targets = tuple(dict.fromkeys(str(Path(path).as_posix()) for path in relative_paths))
    if not targets:
        raise SealContractError("decode_wall_clock: no rewritten manifest rows declared")
    for rel in targets:
        if rel == MANIFEST_NAME or Path(rel).is_absolute() or ".." in Path(rel).parts:
            raise SealContractError(f"decode_wall_clock: invalid rewritten manifest row: {rel}")
        if rel not in listing:
            raise SealContractError(f"decode_wall_clock: manifest omits rewritten file: {rel}")
        if not (root / rel).is_file() or (root / rel).is_symlink():
            raise SealContractError(f"decode_wall_clock: rewritten file is absent: {rel}")
    return targets


def rebind_receiver_manifest(root: Path, relative_paths: Iterable[str]) -> dict[str, object]:
    """Rebind declared rows, preserving a byte-identical manifest on a no-op.

    Every non-target row must already match its file, so this operation cannot
    launder unrelated staleness.  The complete replacement is rendered before
    the manifest is touched and staged in the tree's parent directory.
    """

    from tac.decode_wall_clock import validate_receiver_manifest

    root = Path(root)
    manifest = root / MANIFEST_NAME
    targets = require_receiver_manifest_rows(root, relative_paths)
    listing = _listing(root)
    target_set = set(targets)
    present = {
        path.relative_to(root).as_posix()
        for path in shipped_files(root)
        if path.name != MANIFEST_NAME
    }
    missing_files = sorted(set(listing) - present)
    if missing_files:
        raise SealContractError(
            "decode_wall_clock: manifest lists absent file while rebinding: "
            + ", ".join(missing_files[:4])
        )
    required = present - {"archive.zip"}
    omitted = sorted(required - set(listing))
    if omitted:
        raise SealContractError(
            "decode_wall_clock: manifest omits unrelated shipped file while rebinding: "
            + ", ".join(omitted[:4])
        )
    for rel, digest in listing.items():
        path = root / rel
        if rel not in target_set and sha256_file(path) != digest:
            raise SealContractError(
                f"decode_wall_clock: manifest has unrelated stale row while rebinding: {rel}"
            )

    current = manifest.read_bytes()
    if all(sha256_file(root / rel) == listing[rel] for rel in targets):
        validation = validate_receiver_manifest(root)
        return {
            "changed": False,
            "relative_paths": list(targets),
            "manifest_sha256": sha256_file(manifest),
            "validation": validation,
        }

    include_archive = "archive.zip" in listing
    replacement = canonical_receiver_manifest_bytes(root, include_archive=include_archive)
    mode = manifest.stat().st_mode
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=root.parent, prefix=f".{root.name}.{MANIFEST_NAME}.", delete=False
        ) as stream:
            stream.write(replacement)
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        os.chmod(temporary, mode)
        os.replace(temporary, manifest)
        temporary = None
        validation = validate_receiver_manifest(root)
    except Exception:
        manifest.write_bytes(current)
        os.chmod(manifest, mode)
        raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    return {
        "changed": replacement != current,
        "relative_paths": list(targets),
        "manifest_sha256": sha256_file(manifest),
        "validation": validation,
    }
