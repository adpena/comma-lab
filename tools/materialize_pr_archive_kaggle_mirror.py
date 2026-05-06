#!/usr/bin/env python3
"""Materialize a Kaggle mirror view for the public PR archive corpus.

Hugging Face remains the canonical host for the file-heavy corpus. Kaggle is a
secondary discovery/notebook mirror, so this tool consumes the already
deduplicated release view and adds only Kaggle's required
`dataset-metadata.json`.
"""

from __future__ import annotations

import argparse
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO = repo_root_from_tool(__file__)
ensure_repo_imports(REPO)

from tac.preflight import check_public_release_hygiene  # noqa: E402
from tac.repo_io import json_text, write_json  # noqa: E402
from tools.audit_public_publish_links import audit_public_publish_links  # noqa: E402

DEFAULT_SOURCE = REPO / "experiments" / "results" / "public_pr_archive_release_view"
DEFAULT_OUTPUT = REPO / "experiments" / "results" / "public_pr_archive_kaggle_mirror"
DEFAULT_DATASET_ID = "adpena/comma-video-compression-pr-archive"


def display_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def should_skip_release_view_file(rel: Path) -> bool:
    parts = rel.parts
    name = rel.name
    return (
        ".cache" in parts
        or ".git" in parts
        or "__pycache__" in parts
        or name.endswith(".pyc")
        or name == ".DS_Store"
        or name == "dataset-metadata.json"
    )


def build_dataset_metadata(dataset_id: str, *, description: str | None = None) -> dict[str, object]:
    return {
        "title": "Comma Video Compression PR Archive",
        "subtitle": "Deduplicated public scored PR corpus",
        "description": description
        or (
            "Secondary Kaggle mirror of the canonical Hugging Face public PR "
            "archive corpus for commaai/comma_video_compression_challenge. "
            "Contains byte-exact scored archive.zip files, PR metadata, "
            "provenance logs, filtered source mirrors, and the omission ledger. "
            "The raw forensic intake tree is intentionally not mirrored."
        ),
        "id": dataset_id,
        "licenses": [{"name": "other"}],
        "keywords": [
            "video-compression",
            "neural-codec",
            "rate-distortion",
            "hnerv",
            "reproducibility",
        ],
        "expectedUpdateFrequency": "never",
        "userSpecifiedSources": (
            "Public Pull Requests from "
            "https://github.com/commaai/comma_video_compression_challenge; "
            "canonical corpus at "
            "https://huggingface.co/datasets/adpena/"
            "comma_video_compression_challenge_pr_archive."
        ),
        "resources": [
            {
                "path": "FETCH_SUMMARY.json",
                "description": "Machine-readable summary of public PR archive/source recovery.",
            },
            {
                "path": "OMITTED_SHARED_ASSETS.json",
                "description": "Audit ledger for reconstructable assets omitted from the mirror.",
            },
            {
                "path": "README.md",
                "description": "Dataset card and consumer notes.",
            },
        ],
    }


def materialize_kaggle_mirror(
    source_root: Path,
    output_root: Path,
    *,
    dataset_id: str,
    force: bool,
    strict_hygiene: bool = True,
) -> dict[str, object]:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"release view not found: {source_root}")
    if not (source_root / "FETCH_SUMMARY.json").is_file():
        raise FileNotFoundError(f"release view missing FETCH_SUMMARY.json: {source_root}")
    if not (source_root / "OMITTED_SHARED_ASSETS.json").is_file():
        raise FileNotFoundError(f"release view missing OMITTED_SHARED_ASSETS.json: {source_root}")

    if output_root.exists():
        if not force:
            raise FileExistsError(f"output root exists; pass --force: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    copied = 0
    hardlinked = 0
    skipped: list[str] = []
    included_bytes = 0

    for src in sorted(p for p in source_root.rglob("*") if p.is_file()):
        rel = src.relative_to(source_root)
        if should_skip_release_view_file(rel):
            skipped.append(rel.as_posix())
            continue
        mode = link_or_copy(src, output_root / rel)
        if mode == "hardlink":
            hardlinked += 1
        else:
            copied += 1
        included_bytes += src.stat().st_size

    metadata = build_dataset_metadata(dataset_id)
    metadata["resources"].append(
        {
            "path": "KAGGLE_MIRROR_MANIFEST.json",
            "description": "Mirror-generation manifest for this Kaggle upload view.",
        }
    )
    write_json(output_root / "dataset-metadata.json", metadata)

    manifest = {
        "schema": "comma_pr_archive_kaggle_mirror_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset_id": dataset_id,
        "source_root": display_path(source_root),
        "output_root": display_path(output_root),
        "included_file_count": copied + hardlinked,
        "included_bytes": included_bytes,
        "hardlinked_file_count": hardlinked,
        "copied_file_count": copied,
        "skipped_file_count": len(skipped),
        "skipped": skipped,
        "canonical_hf_dataset": "adpena/comma_video_compression_challenge_pr_archive",
    }
    write_json(output_root / "KAGGLE_MIRROR_MANIFEST.json", manifest)
    hygiene_violations = check_public_release_hygiene(
        repo_root=REPO,
        strict=strict_hygiene,
        verbose=False,
        scan_paths=[output_root],
    )
    link_payload = audit_public_publish_links([output_root], base_root=output_root, live=False)
    link_violations = [
        "{path}:{line}: {kind}: {url} ({detail})".format(**violation)
        for violation in link_payload["violations"]
    ]
    if link_violations and strict_hygiene:
        raise RuntimeError(
            "KAGGLE MIRROR LINK HYGIENE violations:\n"
            + "\n".join(f"  - {violation}" for violation in link_violations[:40])
        )
    manifest["hygiene_violation_count"] = len(hygiene_violations)
    manifest["public_link_count"] = int(link_payload["link_count"])
    manifest["public_link_violation_count"] = len(link_violations)
    write_json(output_root / "KAGGLE_MIRROR_MANIFEST.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-strict-hygiene", action="store_true")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)

    manifest = materialize_kaggle_mirror(
        args.source_root,
        args.output_root,
        dataset_id=args.dataset_id,
        force=args.force,
        strict_hygiene=not args.no_strict_hygiene,
    )
    if args.format == "json":
        print(json_text(manifest), end="")
    else:
        print(
            "kaggle mirror materialized: "
            f"{manifest['included_file_count']} included file(s), "
            f"{manifest['included_bytes']} included byte(s), "
            f"{manifest['skipped_file_count']} skipped file(s)"
        )
        print(f"output: {manifest['output_root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
