#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Scan roots for ATW2 archive.zip candidates and compact their CDF sections.

This is the full-candidate actuator for the ATW2 CDF removal lane. It composes
the structural scanner with the ZIP compactor, writes each compacted archive
under a deterministic output directory, and emits a custody report. It does not
make a score claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from scan_atw2_cdf_compaction_candidates import (  # noqa: E402
    Atw2CdfScanReport,
    scan_atw2_cdf_candidates,
)
from tac.substrates.atw_codec_v2.cdf_dead_section import (  # noqa: E402
    compact_atw2_cdf_table_in_archive_zip,
)


@dataclass(frozen=True)
class Atw2CdfBatchCompactionRow:
    index: int
    source_archive_zip_path: str
    output_archive_zip_path: str
    member_name: str
    source_archive_zip_bytes: int
    output_archive_zip_bytes: int
    archive_zip_bytes_saved: int
    archive_zip_delta_s_rate_only: float
    source_archive_zip_sha256: str
    output_archive_zip_sha256: str
    inner_archive_bytes_saved: int
    raw_equal: bool
    max_abs_raw_byte_delta: int
    source_raw_sha256: str
    compact_raw_sha256: str
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False


@dataclass(frozen=True)
class Atw2CdfBatchFailureRow:
    index: int
    source_archive_zip_path: str
    member_name: str
    error: str


@dataclass(frozen=True)
class Atw2CdfBatchCompactionReport:
    schema: str
    generated_at_utc: str
    roots: list[str]
    output_dir: str
    scan_report: dict[str, object]
    candidates_seen: int
    compacted_count: int
    failure_count: int
    total_archive_zip_bytes_saved: int
    compacted: list[Atw2CdfBatchCompactionRow]
    failures: list[Atw2CdfBatchFailureRow]
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["compacted"] = [asdict(row) for row in self.compacted]
        payload["failures"] = [asdict(row) for row in self.failures]
        return payload


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _utc_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _candidate_output_path(output_dir: Path, *, index: int, source_sha256: str) -> Path:
    return output_dir / f"candidate_{index:04d}_{source_sha256[:16]}" / "archive.zip"


def compact_scan_candidates(
    roots: list[Path],
    *,
    output_dir: Path,
    member_names: list[str],
    max_archives: int | None,
    max_candidates: int | None,
    device: str,
    keep_going: bool,
) -> Atw2CdfBatchCompactionReport:
    scan = scan_atw2_cdf_candidates(
        roots,
        member_names=member_names,
        max_archives=max_archives,
    )
    candidates = scan.candidates
    if max_candidates is not None:
        candidates = candidates[: max(0, max_candidates)]

    compacted: list[Atw2CdfBatchCompactionRow] = []
    failures: list[Atw2CdfBatchFailureRow] = []
    for index, candidate in enumerate(candidates):
        source_zip = Path(candidate.archive_zip_path)
        out_zip = _candidate_output_path(
            output_dir,
            index=index,
            source_sha256=candidate.archive_zip_sha256,
        )
        try:
            proof = compact_atw2_cdf_table_in_archive_zip(
                source_zip,
                out_zip,
                member_name=candidate.member_name,
                device=device,
            )
        except Exception as exc:
            failures.append(
                Atw2CdfBatchFailureRow(
                    index=index,
                    source_archive_zip_path=str(source_zip),
                    member_name=candidate.member_name,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            if not keep_going:
                break
            continue

        compacted.append(
            Atw2CdfBatchCompactionRow(
                index=index,
                source_archive_zip_path=str(source_zip),
                output_archive_zip_path=str(out_zip),
                member_name=proof.member_name,
                source_archive_zip_bytes=proof.source_archive_zip_bytes,
                output_archive_zip_bytes=proof.compact_archive_zip_bytes,
                archive_zip_bytes_saved=proof.archive_zip_bytes_saved,
                archive_zip_delta_s_rate_only=proof.archive_zip_delta_s_rate_only,
                source_archive_zip_sha256=proof.source_archive_zip_sha256,
                output_archive_zip_sha256=_sha256_file(out_zip),
                inner_archive_bytes_saved=proof.inner_proof.archive_bytes_saved,
                raw_equal=proof.inner_proof.raw_equal,
                max_abs_raw_byte_delta=proof.inner_proof.max_abs_raw_byte_delta,
                source_raw_sha256=proof.inner_proof.source_raw_sha256,
                compact_raw_sha256=proof.inner_proof.compact_raw_sha256,
            )
        )

    total_saved = sum(row.archive_zip_bytes_saved for row in compacted)
    return Atw2CdfBatchCompactionReport(
        schema="atw2_cdf_batch_compaction_report_v1",
        generated_at_utc=_utc_iso(),
        roots=[str(root) for root in roots],
        output_dir=str(output_dir),
        scan_report=scan.to_dict(),
        candidates_seen=len(candidates),
        compacted_count=len(compacted),
        failure_count=len(failures),
        total_archive_zip_bytes_saved=total_saved,
        compacted=compacted,
        failures=failures,
    )


def render_markdown(report: Atw2CdfBatchCompactionReport) -> str:
    lines = [
        "# ATW2 CDF Batch Compaction Report",
        "",
        f"- Generated UTC: {report.generated_at_utc}",
        f"- Roots: {', '.join(f'`{root}`' for root in report.roots)}",
        f"- Candidates seen: {report.candidates_seen}",
        f"- Compacted: {report.compacted_count}",
        f"- Failures: {report.failure_count}",
        f"- Total ZIP bytes saved: {report.total_archive_zip_bytes_saved}",
        "- Score claim: false",
        "- Promotion eligible: false",
        "- Ready for exact eval dispatch: false",
        "",
    ]
    if report.compacted:
        lines.extend(
            [
                "## Compacted",
                "",
                "| index | source | output | saved bytes | raw equal | max raw delta |",
                "|---:|---|---|---:|---:|---:|",
            ]
        )
        for row in report.compacted:
            lines.append(
                "| "
                f"{row.index} | "
                f"`{row.source_archive_zip_path}` | "
                f"`{row.output_archive_zip_path}` | "
                f"{row.archive_zip_bytes_saved} | "
                f"{str(row.raw_equal).lower()} | "
                f"{row.max_abs_raw_byte_delta} |"
            )
        lines.append("")
    if report.failures:
        lines.extend(["## Failures", ""])
        for row in report.failures:
            lines.append(
                f"- `{row.source_archive_zip_path}` member `{row.member_name}`: {row.error}"
            )
        lines.append("")
    if not report.compacted and not report.failures:
        lines.extend(
            [
                "## Result",
                "",
                "No parseable ATW2 compaction candidates were found.",
                "",
            ]
        )
    return "\n".join(lines)


def _write_report(report: Atw2CdfBatchCompactionReport, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "batch_compaction_report.json").write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "batch_compaction_report.md").write_text(
        render_markdown(report),
        encoding="utf-8",
    )


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
        default=Path("experiments/results") / f"atw2_cdf_batch_compaction_{_utc_stamp()}",
    )
    parser.add_argument("--member-name", action="append", dest="member_names")
    parser.add_argument("--max-archives", type=int)
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove an existing output directory before running.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output_dir = args.output_dir
    if output_dir.exists():
        if not args.force:
            raise SystemExit(f"output directory exists; pass --force: {output_dir}")
        shutil.rmtree(output_dir)
    member_names = args.member_names or ["0.bin", "x"]
    report = compact_scan_candidates(
        args.roots,
        output_dir=output_dir,
        member_names=member_names,
        max_archives=args.max_archives,
        max_candidates=args.max_candidates,
        device=args.device,
        keep_going=args.keep_going,
    )
    _write_report(report, output_dir)
    sys.stdout.write(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    return 1 if report.failure_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
