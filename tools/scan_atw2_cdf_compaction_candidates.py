#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Scan archive.zip artifacts for ATW2 CDF-compaction candidates.

The scanner is structural rather than name-based: it opens each archive.zip,
tries configured member names with the ATW2 parser, and records only parseable
ATW2 payloads. It does not claim score movement or promotion eligibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.substrates.atw_codec_v2.cdf_dead_section import (  # noqa: E402
    analyze_atw2_cdf_section,
)
from tac.substrates.atw_codec_v2.archive import parse_archive  # noqa: E402

FULL_CANDIDATE_MIN_PAIRS = 600


@dataclass(frozen=True)
class Atw2CdfCandidateRow:
    archive_zip_path: str
    archive_zip_bytes: int
    archive_zip_sha256: str
    member_name: str
    member_bytes: int
    member_compress_type: int
    num_pairs: int
    candidate_class: str
    full_candidate: bool
    cdf_offset: int
    cdf_bytes: int
    cdf_classes: int
    cdf_symbols: int
    conservative_bytes_saved: int
    conservative_delta_s_rate_only: float
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False


@dataclass(frozen=True)
class Atw2CdfScanReport:
    roots: list[str]
    member_names: list[str]
    archives_seen: int
    zip_errors: int
    zip_error_paths: list[str]
    candidates_found: int
    skipped_after_limit: bool
    candidates: list[Atw2CdfCandidateRow]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["candidates"] = [asdict(row) for row in self.candidates]
        return payload


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_archive_zips(roots: list[Path]) -> list[Path]:
    archive_paths: list[Path] = []
    for root in roots:
        if root.is_file():
            if root.name == "archive.zip":
                archive_paths.append(root)
            continue
        if root.is_dir():
            archive_paths.extend(root.rglob("archive.zip"))
    return sorted(set(archive_paths))


def scan_atw2_cdf_candidates(
    roots: list[Path],
    *,
    member_names: list[str],
    max_archives: int | None,
) -> Atw2CdfScanReport:
    archive_paths = _iter_archive_zips(roots)
    if max_archives is not None:
        selected_paths = archive_paths[: max(0, max_archives)]
        skipped_after_limit = len(archive_paths) > len(selected_paths)
    else:
        selected_paths = archive_paths
        skipped_after_limit = False

    candidates: list[Atw2CdfCandidateRow] = []
    zip_error_paths: list[str] = []
    for archive_path in selected_paths:
        try:
            with zipfile.ZipFile(archive_path, "r") as zf:
                infos = {info.filename: info for info in zf.infolist()}
                for member_name in member_names:
                    info = infos.get(member_name)
                    if info is None:
                        continue
                    member_bytes = zf.read(member_name)
                    try:
                        analysis = analyze_atw2_cdf_section(member_bytes)
                        parsed = parse_archive(member_bytes)
                    except (ValueError, struct.error, json.JSONDecodeError, UnicodeError):
                        continue
                    num_pairs = int(parsed.latent_residual.shape[0])
                    full_candidate = num_pairs >= FULL_CANDIDATE_MIN_PAIRS
                    candidate_class = (
                        "full_candidate"
                        if full_candidate
                        else "smoke_or_small_candidate"
                    )
                    archive_size = archive_path.stat().st_size
                    candidates.append(
                        Atw2CdfCandidateRow(
                            archive_zip_path=str(archive_path),
                            archive_zip_bytes=archive_size,
                            archive_zip_sha256=_sha256_file(archive_path),
                            member_name=member_name,
                            member_bytes=len(member_bytes),
                            member_compress_type=int(info.compress_type),
                            num_pairs=num_pairs,
                            candidate_class=candidate_class,
                            full_candidate=full_candidate,
                            cdf_offset=analysis.cdf_offset,
                            cdf_bytes=analysis.cdf_bytes,
                            cdf_classes=analysis.cdf_classes,
                            cdf_symbols=analysis.cdf_symbols,
                            conservative_bytes_saved=(
                                analysis.conservative_bytes_saved
                            ),
                            conservative_delta_s_rate_only=(
                                analysis.conservative_delta_s_rate_only
                            ),
                        )
                    )
        except (zipfile.BadZipFile, OSError):
            zip_error_paths.append(str(archive_path))

    return Atw2CdfScanReport(
        roots=[str(root) for root in roots],
        member_names=member_names,
        archives_seen=len(selected_paths),
        zip_errors=len(zip_error_paths),
        zip_error_paths=zip_error_paths,
        candidates_found=len(candidates),
        skipped_after_limit=skipped_after_limit,
        candidates=candidates,
    )


def render_markdown(report: Atw2CdfScanReport) -> str:
    lines = [
        "# ATW2 CDF Compaction Candidate Scan",
        "",
        f"- Archives scanned: {report.archives_seen}",
        f"- ZIP errors: {report.zip_errors}",
        f"- Candidates found: {report.candidates_found}",
        f"- Skipped after limit: {str(report.skipped_after_limit).lower()}",
        "- Score claim: false",
        "- Promotion eligible: false",
        "- Ready for exact eval dispatch: false",
        "",
    ]
    if report.zip_error_paths:
        lines.extend(["## ZIP Errors", ""])
        for path in report.zip_error_paths:
            lines.append(f"- `{path}`")
        lines.append("")
    if not report.candidates:
        lines.extend(
            [
                "## Result",
                "",
                "No parseable ATW2 `archive.zip` members were found in the scanned roots.",
                "This blocks real-archive CDF compaction until an ATW2 candidate archive is built or harvested.",
                "",
            ]
        )
        return "\n".join(lines)

    lines.extend(
        [
            "## Candidates",
            "",
            "| archive_zip_path | member | class | pairs | zip bytes | member bytes | cdf bytes | conservative bytes saved | rate-only delta |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report.candidates:
        lines.append(
            "| "
            f"`{row.archive_zip_path}` | "
            f"`{row.member_name}` | "
            f"{row.candidate_class} | "
            f"{row.num_pairs} | "
            f"{row.archive_zip_bytes} | "
            f"{row.member_bytes} | "
            f"{row.cdf_bytes} | "
            f"{row.conservative_bytes_saved} | "
            f"{row.conservative_delta_s_rate_only:.12g} |"
        )
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
    parser.add_argument("--member-name", action="append", dest="member_names")
    parser.add_argument("--max-archives", type=int)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--md-output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    member_names = args.member_names or ["0.bin", "x"]
    report = scan_atw2_cdf_candidates(
        args.roots,
        member_names=member_names,
        max_archives=args.max_archives,
    )
    payload = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
    if args.md_output is not None:
        args.md_output.parent.mkdir(parents=True, exist_ok=True)
        args.md_output.write_text(render_markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
