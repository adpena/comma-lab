#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare a proven HiNeRV target-region sidecar action with backend fit."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.action_effect import ActionEffect  # noqa: E402
from tac.analysis.hinerv_target_region_action_comparison import (  # noqa: E402
    HI_NERV_TARGET_REGION_ACTION_COMPARISON_SCHEMA,
    build_hinerv_target_region_action_comparison_from_archive,
    write_hinerv_target_region_action_comparison,
)
from tac.repo_io import sha256_file  # noqa: E402
from tac.substrates.hi_nerv.archive_candidate import (  # noqa: E402
    _read_hiv1_payload_from_archive_zip,
    build_hi_nerv_target_region_action_parseback_survival,
)
from tac.substrates.hi_nerv.inflate import inflate_one_video  # noqa: E402

DEFAULT_OUTPUT_ROOT = Path("/Volumes/VertigoDataTier/pact/experiments/results")


def _default_output_dir() -> Path:
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_OUTPUT_ROOT / f"hinerv_target_region_action_comparison_{stamp}"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object expected: {path}")
    return payload


def _read_action_effect_rows(paths: list[Path]) -> list[ActionEffect]:
    rows: list[ActionEffect] = []
    for path in paths:
        text = path.expanduser().resolve(strict=True).read_text(encoding="utf-8")
        for line in text.splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL object row expected: {path}")
            rows.append(ActionEffect.from_dict(payload))
    return rows


def _materialize_inflate_survival_receipt(
    *,
    archive: Path,
    survival_receipt: Path,
    output_dir: Path,
) -> Path:
    source = _read_json(survival_receipt.expanduser().resolve(strict=True))
    raw_path = output_dir / "inflated_from_hinerv_interpreter.raw"
    payload = _read_hiv1_payload_from_archive_zip(archive)
    inflate_one_video(payload, raw_path, device="cpu")
    proof = build_hi_nerv_target_region_action_parseback_survival(
        archive,
        expected_support_sha256=(
            source.get("expected_support_sha256") or source.get("support_sha256")
        ),
        expected_payload_bytes=source.get("expected_payload_bytes"),
        inflated_raw_path=raw_path,
    )
    if source.get("action_id"):
        proof["action_id"] = source["action_id"]
    proof["producer"] = "hinerv_target_region_action_comparison_cli_inflate_materializer"
    proof["source_parseback_receipt_path"] = survival_receipt.as_posix()
    proof_path = output_dir / "target_region_action_parseback_inflate_survival.json"
    proof["artifact_path"] = proof_path.as_posix()
    proof["inflated_raw_custody"] = {
        "path": raw_path.as_posix(),
        "bytes": int(raw_path.stat().st_size),
        "sha256": sha256_file(raw_path),
        "command": (
            "tac.substrates.hi_nerv.inflate.inflate_one_video("
            "payload_from_archive_zip, inflated_from_hinerv_interpreter.raw, device=cpu)"
        ),
        "archive_path": archive.as_posix(),
        "archive_sha256": sha256_file(archive),
        "archive_payload_sha256": hashlib.sha256(payload).hexdigest(),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    proof_path.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema": "hinerv_target_region_action_inflate_materialization.v1",
        "archive_path": archive.as_posix(),
        "archive_sha256": sha256_file(archive),
        "source_parseback_receipt_path": survival_receipt.as_posix(),
        "survival_receipt_path": proof_path.as_posix(),
        "inflated_raw_path": raw_path.as_posix(),
        "inflated_raw_sha256": sha256_file(raw_path),
        "inflated_raw_bytes": int(raw_path.stat().st_size),
        "survived": proof.get("survived"),
        "parseback_survived": proof.get("parseback_survived"),
        "inflate_survived": proof.get("inflate_survived"),
        "blockers": list(proof.get("blockers") or []),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    (output_dir / "inflate_materialization_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return proof_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="HIV1 archive.zip with target-region action")
    parser.add_argument(
        "--survival-receipt",
        type=Path,
        required=True,
        help="hi_nerv_target_region_action_parseback_survival/inflate_survival JSON",
    )
    parser.add_argument(
        "--runner-report",
        type=Path,
        default=None,
        help="compact runner report containing target_region_wall_normal_lift and sidecar candidate details",
    )
    parser.add_argument(
        "--materialize-inflate-raw",
        action="store_true",
        help=(
            "Run the HiNeRV inflate interpreter for --archive, retain raw bytes "
            "under --output-dir, rebuild the survival receipt with inflate "
            "proof, and compare that same row."
        ),
    )
    parser.add_argument(
        "--action-effect-rows",
        type=Path,
        action="append",
        default=[],
        help="JSONL ActionEffect rows to include in the lowering race. Repeatable.",
    )
    parser.add_argument("--action-id", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = (args.output_dir or _default_output_dir()).expanduser().resolve(strict=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = args.archive.expanduser().resolve(strict=True)
    survival_receipt = args.survival_receipt.expanduser().resolve(strict=True)
    runner_report = (
        None
        if args.runner_report is None
        else args.runner_report.expanduser().resolve(strict=True)
    )
    if args.materialize_inflate_raw:
        survival_receipt = _materialize_inflate_survival_receipt(
            archive=archive,
            survival_receipt=survival_receipt,
            output_dir=out_dir,
        )
    report = build_hinerv_target_region_action_comparison_from_archive(
        archive,
        survival_receipt=survival_receipt,
        runner_report=runner_report,
        action_id=args.action_id,
        action_effect_sources=_read_action_effect_rows(args.action_effect_rows),
    )
    written = write_hinerv_target_region_action_comparison(report, out_dir)
    comparison = report.get("comparison") if isinstance(report.get("comparison"), dict) else {}
    lowering_race = report.get("lowering_race") if isinstance(report.get("lowering_race"), dict) else {}
    verdict = lowering_race.get("verdict") if isinstance(lowering_race.get("verdict"), dict) else {}
    summary = {
        "schema": HI_NERV_TARGET_REGION_ACTION_COMPARISON_SCHEMA,
        "output_dir": out_dir.as_posix(),
        **written,
        "action_id": report.get("action_id"),
        "support_sha256": report.get("support_sha256"),
        "decoded_support_sha256": report.get("decoded_support_sha256"),
        "decoded_action_sha256": report.get("decoded_action_sha256"),
        "encoded_program_sha256": report.get("encoded_program_sha256"),
        "best_lowering": (
            comparison.get("best_lowering")
            or verdict.get("best_lowering")
            or lowering_race.get("best_lowering")
        ),
        "first_failing_surface": (
            comparison.get("first_failing_surface")
            or verdict.get("first_failing_surface")
            or lowering_race.get("first_failing_surface")
        ),
        "next_blocker": comparison.get("next_blocker"),
        "sidecar_current_inflate_survived": comparison.get("sidecar_current_inflate_survived"),
        "promotion_eligible": False,
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
