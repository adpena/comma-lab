#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a broad local archive.zip byte-surface inventory.

This diagnostic scans archive.zip files under one or more roots, records ZIP
member layout, duplicate archive SHA clusters, and a simple per-member entropy
headroom estimate. It is meant to choose the next existing-archive byte lane
after a specific substrate path is blocked. It does not make a score claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

RATE_DENOM_BYTES = 37_545_489
RATE_MULTIPLIER = 25.0


@dataclass(frozen=True)
class ArchiveMemberSurface:
    name: str
    compress_type: int
    compressed_bytes: int
    uncompressed_bytes: int
    entropy_bits_per_byte: float
    zip_bits_per_uncompressed_byte: float
    estimated_recoverable_zip_bytes: int


@dataclass(frozen=True)
class ArchiveSurfaceRow:
    archive_zip_path: str
    archive_zip_bytes: int
    archive_zip_sha256: str
    member_count: int
    member_names: list[str]
    total_member_compressed_bytes: int
    total_member_uncompressed_bytes: int
    largest_member_name: str | None
    largest_member_uncompressed_bytes: int
    estimated_recoverable_zip_bytes: int
    estimated_rate_delta_if_floor_reached: float
    duplicate_archive_sha_count: int
    submission_shape_hint: str
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False
    members: list[ArchiveMemberSurface] | None = None


@dataclass(frozen=True)
class ArchiveSurfaceInventory:
    schema: str
    generated_at_utc: str
    roots: list[str]
    archives_seen: int
    zip_errors: int
    zip_error_paths: list[str]
    rows: list[ArchiveSurfaceRow]
    top_recoverable: list[ArchiveSurfaceRow]
    duplicate_sha_clusters: list[dict[str, object]]
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _utc_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _iter_archive_zips(roots: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for root in roots:
        if root.is_file():
            if root.name == "archive.zip":
                paths.append(root)
            continue
        if root.is_dir():
            paths.extend(root.rglob("archive.zip"))
    return sorted(set(paths))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _entropy_bits_per_byte(payload: bytes) -> float:
    if not payload:
        return 0.0
    counts = Counter(payload)
    n = len(payload)
    entropy = 0.0
    for count in counts.values():
        p = count / n
        entropy -= p * math.log2(p)
    return entropy


def _recoverable_bytes(
    *,
    compressed_bytes: int,
    uncompressed_bytes: int,
    entropy_bits_per_byte: float,
) -> int:
    if compressed_bytes <= 0 or uncompressed_bytes <= 0:
        return 0
    current_bits = compressed_bytes * 8.0
    floor_bits = entropy_bits_per_byte * uncompressed_bytes
    if floor_bits >= current_bits:
        return 0
    return int((current_bits - floor_bits) // 8)


def _shape_hint(member_names: list[str]) -> str:
    names = set(member_names)
    if names == {"0.bin"}:
        return "single_0bin_member"
    if names == {"x"}:
        return "single_x_member"
    required = {"renderer.bin", "masks.mkv", "optimized_poses.pt"}
    if required.issubset(names):
        return "canonical_renderer_masks_pose_members"
    if len(names) == 1:
        return "single_unknown_member"
    return "multi_member_archive"


def _profile_archive(path: Path, *, include_members: bool) -> ArchiveSurfaceRow:
    archive_sha = _sha256_file(path)
    archive_bytes = path.stat().st_size
    members: list[ArchiveMemberSurface] = []
    with zipfile.ZipFile(path, "r") as zf:
        for info in zf.infolist():
            payload = zf.read(info.filename)
            entropy = _entropy_bits_per_byte(payload)
            zip_bpb = (
                8.0 * info.compress_size / info.file_size
                if info.file_size
                else 0.0
            )
            recoverable = _recoverable_bytes(
                compressed_bytes=int(info.compress_size),
                uncompressed_bytes=int(info.file_size),
                entropy_bits_per_byte=entropy,
            )
            members.append(
                ArchiveMemberSurface(
                    name=info.filename,
                    compress_type=int(info.compress_type),
                    compressed_bytes=int(info.compress_size),
                    uncompressed_bytes=int(info.file_size),
                    entropy_bits_per_byte=round(entropy, 5),
                    zip_bits_per_uncompressed_byte=round(zip_bpb, 5),
                    estimated_recoverable_zip_bytes=recoverable,
                )
            )

    largest = max(members, key=lambda row: row.uncompressed_bytes, default=None)
    total_compressed = sum(member.compressed_bytes for member in members)
    total_uncompressed = sum(member.uncompressed_bytes for member in members)
    recoverable = sum(member.estimated_recoverable_zip_bytes for member in members)
    rate_delta = -RATE_MULTIPLIER * recoverable / RATE_DENOM_BYTES
    member_names = [member.name for member in members]
    return ArchiveSurfaceRow(
        archive_zip_path=str(path),
        archive_zip_bytes=archive_bytes,
        archive_zip_sha256=archive_sha,
        member_count=len(members),
        member_names=member_names,
        total_member_compressed_bytes=total_compressed,
        total_member_uncompressed_bytes=total_uncompressed,
        largest_member_name=largest.name if largest else None,
        largest_member_uncompressed_bytes=(
            largest.uncompressed_bytes if largest else 0
        ),
        estimated_recoverable_zip_bytes=recoverable,
        estimated_rate_delta_if_floor_reached=rate_delta,
        duplicate_archive_sha_count=1,
        submission_shape_hint=_shape_hint(member_names),
        members=members if include_members else None,
    )


def build_inventory(
    roots: list[Path],
    *,
    top: int,
    max_archives: int | None,
    include_members: bool,
) -> ArchiveSurfaceInventory:
    archive_paths = _iter_archive_zips(roots)
    if max_archives is not None:
        archive_paths = archive_paths[: max(0, max_archives)]

    rows: list[ArchiveSurfaceRow] = []
    zip_error_paths: list[str] = []
    for path in archive_paths:
        try:
            rows.append(_profile_archive(path, include_members=include_members))
        except (OSError, zipfile.BadZipFile, RuntimeError, ValueError):
            zip_error_paths.append(str(path))

    sha_to_paths: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        sha_to_paths[row.archive_zip_sha256].append(row.archive_zip_path)

    duplicate_counts = {
        sha: len(paths) for sha, paths in sha_to_paths.items() if len(paths) > 1
    }
    rows = [
        replace(
            row,
            duplicate_archive_sha_count=duplicate_counts.get(
                row.archive_zip_sha256,
                1,
            ),
        )
        for row in rows
    ]
    ranked = sorted(
        rows,
        key=lambda row: (
            -row.estimated_recoverable_zip_bytes,
            -row.archive_zip_bytes,
            row.archive_zip_path,
        ),
    )
    duplicate_clusters = [
        {
            "archive_zip_sha256": sha,
            "count": len(paths),
            "paths": sorted(paths)[:25],
        }
        for sha, paths in sorted(
            sha_to_paths.items(),
            key=lambda item: (-len(item[1]), item[0]),
        )
        if len(paths) > 1
    ]
    return ArchiveSurfaceInventory(
        schema="archive_surface_inventory_v1",
        generated_at_utc=_utc_iso(),
        roots=[str(root) for root in roots],
        archives_seen=len(archive_paths),
        zip_errors=len(zip_error_paths),
        zip_error_paths=zip_error_paths,
        rows=ranked,
        top_recoverable=ranked[:top],
        duplicate_sha_clusters=duplicate_clusters[:top],
    )


def render_markdown(inventory: ArchiveSurfaceInventory) -> str:
    lines = [
        "# Archive Surface Inventory",
        "",
        f"- Generated UTC: {inventory.generated_at_utc}",
        f"- Roots: {', '.join(f'`{root}`' for root in inventory.roots)}",
        f"- Archives seen: {inventory.archives_seen}",
        f"- ZIP errors: {inventory.zip_errors}",
        "- Score claim: false",
        "- Promotion eligible: false",
        "- Ready for exact eval dispatch: false",
        "",
        "## Top Recoverable ZIP Surfaces",
        "",
        "| rank | archive | bytes | members | shape | recoverable bytes | rate delta | duplicate count |",
        "|---:|---|---:|---:|---|---:|---:|---:|",
    ]
    for rank, row in enumerate(inventory.top_recoverable, start=1):
        lines.append(
            "| "
            f"{rank} | "
            f"`{row.archive_zip_path}` | "
            f"{row.archive_zip_bytes} | "
            f"{row.member_count} | "
            f"{row.submission_shape_hint} | "
            f"{row.estimated_recoverable_zip_bytes} | "
            f"{row.estimated_rate_delta_if_floor_reached:.12g} | "
            f"{row.duplicate_archive_sha_count} |"
        )
    lines.extend(["", "## Duplicate SHA Clusters", ""])
    if not inventory.duplicate_sha_clusters:
        lines.append("No duplicate archive SHA clusters found.")
    else:
        lines.append("| sha256 | count | first paths |")
        lines.append("|---|---:|---|")
        for cluster in inventory.duplicate_sha_clusters:
            paths = ", ".join(f"`{path}`" for path in cluster["paths"][:5])
            lines.append(
                f"| `{cluster['archive_zip_sha256']}` | {cluster['count']} | {paths} |"
            )
    if inventory.zip_error_paths:
        lines.extend(["", "## ZIP Errors", ""])
        for path in inventory.zip_error_paths:
            lines.append(f"- `{path}`")
    lines.append("")
    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        nargs="*",
        type=Path,
        default=[Path("experiments/results"), Path("submissions")],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results")
        / f"archive_surface_inventory_{_utc_stamp()}",
    )
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--max-archives", type=int)
    parser.add_argument("--include-members", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    inventory = build_inventory(
        args.roots,
        top=args.top,
        max_archives=args.max_archives,
        include_members=args.include_members,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(inventory.to_dict(), indent=2, sort_keys=True) + "\n"
    (args.output_dir / "archive_surface_inventory.json").write_text(
        payload,
        encoding="utf-8",
    )
    (args.output_dir / "archive_surface_inventory.md").write_text(
        render_markdown(inventory),
        encoding="utf-8",
    )
    sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
