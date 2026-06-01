#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Rescore only changed HPRC pair windows and patch onto a full-video baseline."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.local_acceleration.mlx_scorer_response import (  # noqa: E402
    MANIFEST_CACHE_INTEGRITY_MODE,
    build_mlx_scorer_response_payload,
    write_mlx_scorer_response_payload,
)
from tac.substrates.hprc.incremental_pair_response import (  # noqa: E402
    build_hprc_incremental_pair_response_report,
    write_hprc_incremental_pair_response_report,
)

OWNED_MARKER = ".hprc_incremental_pair_response_owned.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--candidate-variant-id", required=True)
    parser.add_argument("--pair-ranges", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-root", default=REPO_ROOT, type=Path)
    parser.add_argument("--candidate-archive", type=Path)
    parser.add_argument("--submission-dir", type=Path)
    parser.add_argument("--reference-cache-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--scorer-batch-pairs", type=int, default=1)
    parser.add_argument("--cache-render-batch-pairs", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--allow-batch-shape-research-signal", action="store_true")
    parser.add_argument("--allow-large-tensor-cache", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve(strict=False)
    profile_path = _resolve(args.profile, base=repo_root)
    output_dir = _resolve(args.output_dir, base=repo_root)
    _prepare_owned_dir(output_dir, force=bool(args.force))
    profile = _load_json_object(profile_path)
    candidate_row = _variant_row(profile, str(args.candidate_variant_id))
    candidate_archive = _resolve(
        args.candidate_archive or Path(str(candidate_row["archive_zip_path"])),
        base=repo_root,
    )
    submission_dir = _resolve(
        args.submission_dir or (candidate_archive.parent / "submission"),
        base=repo_root,
    )
    reference_cache_dir = _resolve(
        args.reference_cache_dir or Path(str(profile["reference_cache_dir"])),
        base=repo_root,
    )
    if not candidate_archive.is_file():
        raise SystemExit(f"candidate archive missing: {candidate_archive}")
    if not submission_dir.is_dir():
        raise SystemExit(f"submission dir missing: {submission_dir}")
    if int(args.scorer_batch_pairs) != 1 and not args.allow_batch_shape_research_signal:
        raise SystemExit(
            "--scorer-batch-pairs > 1 requires --allow-batch-shape-research-signal"
        )

    cache_dir = output_dir / "mlx_incremental_cache" / str(args.candidate_variant_id)
    work_dir = output_dir / "mlx_incremental_work" / str(args.candidate_variant_id)
    cache_report_path = output_dir / "mlx_incremental_cache_report.json"
    materialize_argv = [
        sys.executable,
        str(repo_root / "tools/materialize_mlx_scorer_cache_from_submission.py"),
        "--archive",
        str(candidate_archive),
        "--submission-dir",
        str(submission_dir),
        "--output-cache-dir",
        str(cache_dir),
        "--work-dir",
        str(work_dir),
        "--report-output",
        str(cache_report_path),
        "--hprc-direct-cache",
        "--pair-ranges",
        str(args.pair_ranges),
        "--batch-pairs",
        str(int(args.cache_render_batch_pairs)),
        "--allow-large-tensor-cache",
        "--force",
    ]
    if not args.allow_large_tensor_cache:
        materialize_argv.remove("--allow-large-tensor-cache")
    completed = subprocess.run(
        materialize_argv,
        cwd=repo_root,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise SystemExit(
            "incremental HPRC cache materialization failed:\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )

    response_path = output_dir / "mlx_incremental_response.json"
    components_dir = output_dir / "mlx_incremental_components" / str(
        args.candidate_variant_id
    )
    response = build_mlx_scorer_response_payload(
        reference_cache_dir=reference_cache_dir,
        candidate_cache_dir=cache_dir,
        archive_size_bytes=int(candidate_row["archive_zip_bytes"]),
        repo_root=repo_root,
        batch_pairs=int(args.scorer_batch_pairs),
        device_type=str(args.device),
        components_dir=components_dir,
        progress_every=int(args.progress_every),
        allow_gpu_research_signal=str(args.device) == "gpu",
        allow_batch_shape_research_signal=bool(args.allow_batch_shape_research_signal),
        allow_unaudited_candidate_cache_debug=True,
        cache_integrity_mode=MANIFEST_CACHE_INTEGRITY_MODE,
        response_family=f"hprc_incremental_pair_response_{args.candidate_variant_id}",
    )
    write_mlx_scorer_response_payload(response, response_path)
    report = build_hprc_incremental_pair_response_report(
        profile_path=profile_path,
        candidate_variant_id=str(args.candidate_variant_id),
        candidate_response_path=response_path,
        candidate_cache_dir=cache_dir,
        materialization_report_path=cache_report_path,
        repo_root=repo_root,
    )
    report["materialization_command"] = {
        "argv": materialize_argv,
        "stdout": completed.stdout.strip(),
        "stderr_tail": completed.stderr.strip()[-4000:],
    }
    report["scorer_batch_pairs"] = int(args.scorer_batch_pairs)
    report["batch_shape_research_signal"] = int(args.scorer_batch_pairs) != 1
    report_path = output_dir / "hprc_incremental_pair_response_report.json"
    write_hprc_incremental_pair_response_report(
        output_path=report_path,
        report=report,
        allow_overwrite=True,
    )
    print(
        json.dumps(
            {
                "report": report_path.as_posix(),
                "changed_pair_count": len(report["changed_pair_rows"]),
                "delta_total_mlx_score_advisory": report[
                    "delta_total_mlx_score_advisory"
                ],
                "score_claim": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _prepare_owned_dir(path: Path, *, force: bool) -> None:
    if path.exists():
        marker = path / OWNED_MARKER
        if force:
            if not marker.exists() and any(path.iterdir()):
                raise SystemExit(f"refusing --force on non-owned output dir: {path}")
            shutil.rmtree(path)
        elif any(path.iterdir()):
            raise SystemExit(f"output dir exists; pass --force: {path}")
    path.mkdir(parents=True, exist_ok=True)
    (path / OWNED_MARKER).write_text(
        json.dumps({"schema": "owned_directory_marker.v1", "tool": Path(__file__).name})
        + "\n",
        encoding="utf-8",
    )


def _variant_row(profile: dict[str, Any], variant_id: str) -> dict[str, Any]:
    rows = profile.get("variant_rows")
    if not isinstance(rows, list):
        raise SystemExit("profile missing variant_rows")
    matches = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("variant_id") == variant_id
    ]
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one variant row for {variant_id!r}")
    return matches[0]


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise SystemExit(f"JSON root must be object: {path}")
    return payload


def _resolve(path: Path, *, base: Path) -> Path:
    expanded = path.expanduser()
    return expanded if expanded.is_absolute() else (base / expanded).resolve(strict=False)


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
