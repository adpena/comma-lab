#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run deterministic raw postprocess probes and optional advisory scoring."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.inflate_postprocess_surface import (  # noqa: E402
    PostprocessSpec,
    RawVideoShape,
    apply_postprocess,
    get_builtin_spec,
    plan_payload,
    postprocess_spec_from_dict,
    sha256_file,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_advisory(
    *,
    raw: Path,
    archive: Path,
    output_dir: Path,
    axis_label: str,
    batch_size: int,
    num_threads: int,
    timeout: int,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "run_raw_advisory_eval.py"),
        "--raw",
        str(raw),
        "--archive",
        str(archive),
        "--output-dir",
        str(output_dir),
        "--axis-label",
        axis_label,
        "--batch-size",
        str(batch_size),
        "--num-threads",
        str(num_threads),
        "--timeout",
        str(timeout),
    ]
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "advisory_wrapper.stdout.log").write_text(proc.stdout, encoding="utf-8")
    (output_dir / "advisory_wrapper.stderr.log").write_text(proc.stderr, encoding="utf-8")
    payload_path = output_dir / "raw_advisory_eval.json"
    if payload_path.is_file():
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    else:
        payload = {
            "returncode": proc.returncode,
            "blockers": ["raw_advisory_eval_json_missing"],
        }
    payload["wrapper_returncode"] = proc.returncode
    payload["wrapper_cmd"] = cmd
    payload["wrapper_stdout_log"] = str(output_dir / "advisory_wrapper.stdout.log")
    payload["wrapper_stderr_log"] = str(output_dir / "advisory_wrapper.stderr.log")
    return payload


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    shape = RawVideoShape(
        frames=args.frames,
        height=args.height,
        width=args.width,
        channels=args.channels,
    )
    baseline_raw = args.baseline_raw.resolve()
    archive = args.archive.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    specs = [get_builtin_spec(spec_id) for spec_id in args.spec]
    specs.extend(_load_custom_specs(args.custom_spec_json))
    if not specs:
        raise ValueError("at least one --spec or --custom-spec-json is required")
    baseline_sha = sha256_file(baseline_raw)
    archive_sha = sha256_file(archive)

    rows = []
    for spec in specs:
        candidate_dir = output_root / "candidates" / spec.spec_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        candidate_raw = candidate_dir / "0.raw"
        result = apply_postprocess(
            input_raw=baseline_raw,
            output_raw=candidate_raw,
            spec=spec,
            shape=shape,
        )
        row: dict[str, Any] = {
            "spec_id": spec.spec_id,
            "postprocess": result.as_dict(),
            "archive": {
                "path": str(archive),
                "bytes": archive.stat().st_size,
                "sha256": archive_sha,
            },
            "blockers": [
                "raw_postprocess_advisory_not_stock_inflate_runtime",
                "transform_parameters_not_charged_archive_bytes",
                "exact_cuda_auth_eval_missing",
            ],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
        if args.run_advisory and result.passed_visible_change:
            advisory = _run_advisory(
                raw=candidate_raw,
                archive=archive,
                output_dir=candidate_dir / "advisory_eval",
                axis_label=args.axis_label,
                batch_size=args.batch_size,
                num_threads=args.num_threads,
                timeout=args.timeout,
            )
            row["advisory_eval"] = advisory
            if args.baseline_score is not None and advisory.get("canonical_score") is not None:
                row["delta_vs_baseline_score"] = (
                    float(advisory["canonical_score"]) - float(args.baseline_score)
                )
        elif args.run_advisory:
            row["blockers"].append("no_visible_raw_change")

        if args.cleanup_candidate_raw and candidate_raw.is_file() and (
            not args.run_advisory
            or isinstance(row.get("advisory_eval"), dict)
            and row["advisory_eval"].get("returncode") == 0
        ):
            row["cleanup"] = {
                "candidate_raw_deleted": True,
                "candidate_raw_sha256_before_delete": sha256_file(candidate_raw),
                "candidate_raw_bytes_before_delete": candidate_raw.stat().st_size,
            }
            candidate_raw.unlink()
        else:
            row["cleanup"] = {"candidate_raw_deleted": False}

        _write_json(candidate_dir / "postprocess_candidate_manifest.json", row)
        rows.append(row)

    successful = [
        row
        for row in rows
        if isinstance(row.get("advisory_eval"), dict)
        and row["advisory_eval"].get("returncode") == 0
    ]
    improved = [
        row
        for row in successful
        if row.get("delta_vs_baseline_score") is not None
        and float(row["delta_vs_baseline_score"]) < 0.0
    ]
    best = None
    if successful:
        best = min(successful, key=lambda row: float(row["advisory_eval"]["canonical_score"]))
    payload = {
        "schema": "inflate_postprocess_smoke.v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "producer": "tools/run_inflate_postprocess_smoke.py",
        "inputs": {
            "baseline_raw": str(baseline_raw),
            "baseline_raw_sha256": baseline_sha,
            "archive": str(archive),
            "archive_sha256": archive_sha,
            "shape": shape.as_dict(),
            "spec_ids": args.spec,
            "custom_spec_json": [str(path) for path in args.custom_spec_json],
            "run_advisory": bool(args.run_advisory),
            "baseline_score": args.baseline_score,
        },
        "plan": plan_payload(),
        "summary": {
            "candidate_count": len(rows),
            "visible_change_count": sum(
                1 for row in rows if row["postprocess"].get("passed_visible_change")
            ),
            "advisory_success_count": len(successful),
            "improved_count": len(improved),
            "best_spec_id": best.get("spec_id") if best else None,
            "best_score": best.get("advisory_eval", {}).get("canonical_score") if best else None,
            "best_delta_vs_baseline_score": best.get("delta_vs_baseline_score") if best else None,
        },
        "candidates": rows,
        "authority": {
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "notes": (
                "Raw-level postprocess smoke only. Positive rows must be "
                "converted to a stock inflate runtime candidate with charged "
                "parameters before exact eval."
            ),
        },
    }
    _write_json(args.output.resolve(), payload)
    return payload


def _load_custom_specs(paths: list[Path]) -> list[PostprocessSpec]:
    specs: list[PostprocessSpec] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("specs"), list):
            rows = payload["specs"]
        elif isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            rows = [payload]
        else:
            raise ValueError(f"{path}: expected spec object, specs[] object, or list")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"{path}: custom spec row is not an object")
            specs.append(postprocess_spec_from_dict(row))
    return specs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-raw", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--spec", action="append", default=[])
    parser.add_argument("--custom-spec-json", action="append", type=Path, default=[])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=1200)
    parser.add_argument("--height", type=int, default=874)
    parser.add_argument("--width", type=int, default=1164)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--run-advisory", action="store_true")
    parser.add_argument("--baseline-score", type=float)
    parser.add_argument("--axis-label", default="[macOS-CPU advisory inflate-postprocess]")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--cleanup-candidate-raw", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_smoke(args)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "summary": payload["summary"],
                "score_claim": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
