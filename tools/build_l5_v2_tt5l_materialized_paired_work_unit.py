#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the L5 v2 TT5L materialized paired Modal work-unit plan."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.optimization.l5_staircase_v2 import (  # noqa: E402
    TT5L_MATERIALIZED_PAIRED_WORK_UNIT_PLAN_ARTIFACT_PATH,
)
from tac.optimization.l5_v2_measurement_schedule import (  # noqa: E402
    L5V2_TT5L_SIDEINFO_VARIANT_PACKET_ARTIFACT_PATH,
    L5V2_TT5L_SIDEINFO_VARIANT_PACKET_SUBMISSION_DIR,
)
from tac.optimization.l5_v2_tt5l_materialized_work_unit import (  # noqa: E402
    TT5L_MATERIALIZED_PAIRED_WORK_UNIT_DEFAULT_VARIANT,
    TT5L_MATERIALIZED_PAIRED_WORK_UNIT_REPORT_PATH,
    build_tt5l_materialized_paired_work_unit_plan,
    default_tt5l_variant_archive_path,
    render_tt5l_materialized_paired_work_unit_markdown,
    select_tt5l_variant_archive,
    tt5l_materialized_paired_work_unit_json,
)


def _refuse_tmp(path: Path) -> None:
    text = str(path)
    if text.startswith("/tmp/") or "/private/tmp/" in text or "/var/tmp/" in text:
        raise ValueError(f"refusing to write TT5L materialized artifact to tmp: {text!r}")


def _write(path: Path, text: str) -> None:
    _refuse_tmp(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help=(
            "Archive to materialize. If omitted, selects --variant from "
            "--variant-manifest."
        ),
    )
    parser.add_argument(
        "--variant",
        default=TT5L_MATERIALIZED_PAIRED_WORK_UNIT_DEFAULT_VARIANT,
        help="Side-info variant to select when --archive is omitted.",
    )
    parser.add_argument(
        "--variant-manifest",
        type=Path,
        default=Path(L5V2_TT5L_SIDEINFO_VARIANT_PACKET_ARTIFACT_PATH),
        help="TT5L side-info variant packet manifest.",
    )
    parser.add_argument(
        "--submission-dir",
        type=Path,
        default=Path(L5V2_TT5L_SIDEINFO_VARIANT_PACKET_SUBMISSION_DIR),
        help="Runtime submission directory containing inflate.sh.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(TT5L_MATERIALIZED_PAIRED_WORK_UNIT_PLAN_ARTIFACT_PATH),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(TT5L_MATERIALIZED_PAIRED_WORK_UNIT_REPORT_PATH),
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--modal-bin", default=".venv/bin/modal")
    parser.add_argument("--gpu", default="T4")
    parser.add_argument("--inflate-sh", default="inflate.sh")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        for path in (args.output_json, args.output_md):
            _refuse_tmp(path)
        materialized_from = {}
        archive = args.archive
        if archive is None:
            materialized_from = select_tt5l_variant_archive(
                variant_manifest=args.variant_manifest,
                variant=args.variant,
                repo_root=args.repo_root,
            )
            archive = Path(str(materialized_from["archive_path"]))
        else:
            materialized_from = {
                "variant": args.variant,
                "variant_manifest_path": str(args.variant_manifest),
                "default_variant_archive_path": str(
                    default_tt5l_variant_archive_path(variant=args.variant)
                ),
                "selection": "explicit_archive_argument",
            }
        payload = build_tt5l_materialized_paired_work_unit_plan(
            archive=archive,
            submission_dir=args.submission_dir,
            repo_root=args.repo_root,
            materialized_from=materialized_from,
            modal_bin=args.modal_bin,
            gpu=args.gpu,
            inflate_sh=args.inflate_sh,
        )
        _write(args.output_json, tt5l_materialized_paired_work_unit_json(payload))
        _write(args.output_md, render_tt5l_materialized_paired_work_unit_markdown(payload))
    except (OSError, ValueError) as exc:
        print(f"[l5-v2-tt5l-materialize] FATAL: {exc}", file=sys.stderr)
        return 2
    print(
        "[l5-v2-tt5l-materialize] "
        f"archive={payload['archive']['path']} "
        f"sha256={payload['archive']['sha256']} "
        f"variant={payload.get('materialized_from', {}).get('variant', '')} "
        f"output_json={args.output_json} "
        f"output_md={args.output_md} "
        "score_claim=false promotion_eligible=false dispatch_attempted=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
