#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize an ATW2 smoke archive.zip and compact its dead CDF section.

This is a one-command reproducibility actuator for the ATW2 CDF removal lane.
It emits a tiny CPU smoke archive, rewrites its parser-visible/current-runtime-
dead ``cdf_table_blob`` to the compact sentinel, proves raw-output parity, and
writes a JSON/Markdown custody report. It does not make a score claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.substrates.atw_codec_v2.cdf_dead_section import (  # noqa: E402
    compact_atw2_cdf_table_in_archive_zip,
)


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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_trainer(
    *,
    output_dir: Path,
    epochs: int,
    device: str,
    variant: str,
    timeout_seconds: int,
) -> subprocess.CompletedProcess[str]:
    trainer = REPO_ROOT / "experiments" / "train_substrate_atw_codec_v2.py"
    cmd = [
        sys.executable,
        str(trainer),
        "--output-dir",
        str(output_dir),
        "--epochs",
        str(epochs),
        "--device",
        device,
        "--smoke",
        "--variant",
        variant,
    ]
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=True,
    )


def _archive_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _build_report(
    *,
    output_dir: Path,
    source_dir: Path,
    compact_dir: Path,
    trainer_proc: subprocess.CompletedProcess[str],
    proof: dict[str, Any],
    smoke_stats: dict[str, Any],
) -> dict[str, Any]:
    source_zip = source_dir / "archive.zip"
    compact_zip = compact_dir / "archive.zip"
    source_payload = source_dir / "0.bin"
    return {
        "schema": "atw2_cdf_materialized_smoke_compaction_report_v1",
        "generated_at_utc": _utc_iso(),
        "output_dir": str(output_dir),
        "source_dir": str(source_dir),
        "compact_dir": str(compact_dir),
        "source_payload_0bin": _archive_record(source_payload),
        "source_archive_zip": _archive_record(source_zip),
        "compact_archive_zip": _archive_record(compact_zip),
        "trainer": {
            "returncode": trainer_proc.returncode,
            "stdout": trainer_proc.stdout,
            "stderr": trainer_proc.stderr,
        },
        "smoke_stats": smoke_stats,
        "proof": proof,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def render_markdown(report: dict[str, Any]) -> str:
    proof = report["proof"]
    inner = proof["inner_proof"]
    lines = [
        "# ATW2 CDF Materialized Smoke Compaction Report",
        "",
        f"- Generated UTC: {report['generated_at_utc']}",
        "- Score claim: false",
        "- Promotion eligible: false",
        "- Ready for exact eval dispatch: false",
        "",
        "## Inputs",
        "",
        f"- Source payload: `{report['source_payload_0bin']['path']}`",
        f"- Source payload bytes: {report['source_payload_0bin']['bytes']}",
        f"- Source payload SHA-256: `{report['source_payload_0bin']['sha256']}`",
        f"- Source archive.zip: `{report['source_archive_zip']['path']}`",
        f"- Source archive.zip bytes: {report['source_archive_zip']['bytes']}",
        f"- Source archive.zip SHA-256: `{report['source_archive_zip']['sha256']}`",
        "",
        "## Compaction",
        "",
        f"- Compact archive.zip: `{report['compact_archive_zip']['path']}`",
        f"- Compact archive.zip bytes: {report['compact_archive_zip']['bytes']}",
        f"- Compact archive.zip SHA-256: `{report['compact_archive_zip']['sha256']}`",
        f"- ZIP bytes saved: {proof['archive_zip_bytes_saved']}",
        f"- ZIP rate-only delta: {proof['archive_zip_delta_s_rate_only']}",
        f"- Inner ATW2 bytes saved: {inner['archive_bytes_saved']}",
        f"- Compact CDF bytes: {inner['compact_cdf_bytes']}",
        "",
        "## Raw Parity",
        "",
        f"- Raw equal: {str(inner['raw_equal']).lower()}",
        f"- Max absolute raw byte delta: {inner['max_abs_raw_byte_delta']}",
        f"- Raw byte count: {inner['raw_byte_count']}",
        f"- Source raw SHA-256: `{inner['source_raw_sha256']}`",
        f"- Compact raw SHA-256: `{inner['compact_raw_sha256']}`",
        "",
    ]
    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results") / f"atw2_cdf_materialized_smoke_{_utc_stamp()}",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--variant", choices=("A", "B"), default="B")
    parser.add_argument("--member-name", default="0.bin")
    parser.add_argument("--timeout-seconds", type=int, default=120)
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
    source_dir = output_dir / "source"
    compact_dir = output_dir / "compact"
    source_zip = source_dir / "archive.zip"
    compact_zip = compact_dir / "archive.zip"

    trainer_proc = _run_trainer(
        output_dir=source_dir,
        epochs=args.epochs,
        device=args.device,
        variant=args.variant,
        timeout_seconds=args.timeout_seconds,
    )
    if not source_zip.is_file():
        raise SystemExit(f"ATW2 smoke trainer did not produce archive.zip: {source_zip}")

    proof = compact_atw2_cdf_table_in_archive_zip(
        source_zip,
        compact_zip,
        member_name=args.member_name,
        device=args.device,
    ).to_dict()
    smoke_stats = _read_json(source_dir / "smoke_stats.json")
    report = _build_report(
        output_dir=output_dir,
        source_dir=source_dir,
        compact_dir=compact_dir,
        trainer_proc=trainer_proc,
        proof=proof,
        smoke_stats=smoke_stats,
    )
    report_json = output_dir / "materialized_compaction_report.json"
    report_md = output_dir / "materialized_compaction_report.md"
    report_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_md.write_text(render_markdown(report), encoding="utf-8")
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
