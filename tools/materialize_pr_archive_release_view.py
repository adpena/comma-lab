#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize a deduplicated public PR archive corpus upload view.

The raw intake tree intentionally preserves contestant source checkouts and
forensic state. That is too large and too noisy for public dataset upload:
public source mirrors repeat fixed contest assets (`videos/0.mkv`,
PoseNet/SegNet weights), git/LFS object stores, caches, and vendored codec
binaries many times.

This tool builds a canonical release view that keeps byte-exact scored
`archive.zip` files, metadata, provenance, logs, README/LICENSE, and filtered
source code, while omitting reconstructable shared assets into an auditable
`OMITTED_SHARED_ASSETS.json` manifest.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO = repo_root_from_tool(__file__)
ensure_repo_imports(REPO)
DEFAULT_SOURCE = REPO / "experiments" / "results" / "public_pr_intake_full"
DEFAULT_OUTPUT = REPO / "experiments" / "results" / "public_pr_archive_release_view"
SANITIZED_TEXT_SUFFIXES = {".json", ".log", ".md", ".txt"}
LOCAL_OPERATOR_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_])/(?:Users|home)/[A-Za-z0-9._-]+(?:/[^\s)\"'<>`]*)?"
)
PRIVATE_COMMA_LAB_URL_RE = re.compile(r"https://github\.com/adpena/comma-lab(?:/[^\s)\"'<>]*)?")

from tac.repo_io import json_text, write_json  # noqa: E402
from tools.audit_public_publish_links import audit_public_publish_links  # noqa: E402


def display_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def omission_reason(rel: Path, size: int, source_size_limit: int) -> str | None:
    parts = rel.parts
    name = rel.name
    rel_s = rel.as_posix()

    if ".git" in parts:
        return "git_metadata_reconstructable_from_head_repo_sha"
    if ".cache" in parts:
        return "local_upload_cache"
    if "__pycache__" in parts or name.endswith(".pyc"):
        return "python_cache"
    if "/source/videos/" in f"/{rel_s}":
        return "fixed_contest_video_reconstructable_from_upstream"
    if rel_s.endswith("/source/models/posenet.safetensors"):
        return "fixed_contest_posenet_weight_reconstructable_from_upstream"
    if rel_s.endswith("/source/models/segnet.safetensors"):
        return "fixed_contest_segnet_weight_reconstructable_from_upstream"
    if name == "ffmpeg-new":
        return "vendored_ffmpeg_binary_not_needed_for_source_mirror"
    if name.startswith("libSvtAv1Enc.so"):
        return "vendored_svtav1_shared_library_not_needed_for_source_mirror"
    if name == ".DS_Store":
        return "platform_metadata"
    if "/source/" in f"/{rel_s}" and name != "archive.zip" and size > source_size_limit:
        return "large_source_asset_requires_manual_release_review"
    return None


def link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def sanitize_public_text(text: str) -> tuple[str, int]:
    sanitized, count = LOCAL_OPERATOR_PATH_RE.subn("${LOCAL_PATH}", text)
    sanitized, private_count = PRIVATE_COMMA_LAB_URL_RE.subn(
        "${PUBLIC_COMMA_LAB_REPO_URL}",
        sanitized,
    )
    return sanitized, count + private_count


def include_release_file(src: Path, dst: Path, rel: Path) -> tuple[str, int]:
    if rel.suffix.lower() in SANITIZED_TEXT_SUFFIXES:
        try:
            text = src.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return link_or_copy(src, dst), 0
        sanitized, count = sanitize_public_text(text)
        if count:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(sanitized, encoding="utf-8")
            return "sanitized_copy", count
    return link_or_copy(src, dst), 0


def materialize(
    source_root: Path,
    output_root: Path,
    *,
    force: bool,
    source_size_limit: int,
    strict_link_hygiene: bool = True,
) -> dict[str, object]:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"source root not found: {source_root}")
    if output_root.exists():
        if not force:
            raise FileExistsError(f"output root exists; pass --force: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    copied = 0
    hardlinked = 0
    omitted: list[dict[str, object]] = []
    omitted_by_reason: Counter[str] = Counter()
    sanitized_file_count = 0
    sanitized_replacement_count = 0
    included_bytes = 0
    omitted_bytes = 0

    for src in sorted(p for p in source_root.rglob("*") if p.is_file()):
        rel = src.relative_to(source_root)
        size = src.stat().st_size
        reason = omission_reason(rel, size, source_size_limit)
        if reason:
            omitted.append({"path": rel.as_posix(), "bytes": size, "reason": reason})
            omitted_by_reason[reason] += 1
            omitted_bytes += size
            continue

        mode, replacements = include_release_file(src, output_root / rel, rel)
        if replacements:
            sanitized_file_count += 1
            sanitized_replacement_count += replacements
        if mode == "hardlink":
            hardlinked += 1
        else:
            copied += 1
        included_bytes += size

    manifest = {
        "schema": "comma_pr_archive_release_view_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_root": display_path(source_root),
        "output_root": display_path(output_root),
        "source_size_limit_bytes": source_size_limit,
        "included_file_count": copied + hardlinked,
        "included_bytes": included_bytes,
        "hardlinked_file_count": hardlinked,
        "copied_file_count": copied,
        "omitted_file_count": len(omitted),
        "omitted_bytes": omitted_bytes,
        "omitted_by_reason": dict(sorted(omitted_by_reason.items())),
        "omitted": omitted,
        "sanitized_file_count": sanitized_file_count,
        "sanitized_replacement_count": sanitized_replacement_count,
    }
    write_json(output_root / "OMITTED_SHARED_ASSETS.json", manifest)
    link_payload = audit_public_publish_links([output_root], base_root=output_root, live=False)
    link_violations = [
        "{path}:{line}: {kind}: {url} ({detail})".format(**violation)
        for violation in link_payload["violations"]
    ]
    if link_violations and strict_link_hygiene:
        raise RuntimeError(
            "PUBLIC PR ARCHIVE LINK HYGIENE violations:\n"
            + "\n".join(f"  - {violation}" for violation in link_violations[:40])
        )
    manifest["public_link_count"] = int(link_payload["link_count"])
    manifest["public_link_violation_count"] = len(link_violations)
    write_json(output_root / "OMITTED_SHARED_ASSETS.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--source-size-limit",
        type=int,
        default=25_000_000,
        help="omit non-archive files under source/ above this size unless explicitly allowed",
    )
    parser.add_argument("--no-strict-link-hygiene", action="store_true")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)

    manifest = materialize(
        args.source_root,
        args.output_root,
        force=args.force,
        source_size_limit=args.source_size_limit,
        strict_link_hygiene=not args.no_strict_link_hygiene,
    )
    if args.format == "json":
        print(json_text(manifest), end="")
    else:
        print(
            "release view materialized: "
            f"{manifest['included_file_count']} included file(s), "
            f"{manifest['included_bytes']} included byte(s), "
            f"{manifest['omitted_file_count']} omitted file(s), "
            f"{manifest['omitted_bytes']} omitted byte(s)"
        )
        print(f"output: {manifest['output_root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
