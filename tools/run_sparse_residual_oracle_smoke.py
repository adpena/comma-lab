#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run charged sparse-residual oracle probes against an inflated raw output."""

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

from tac.optimization.inflate_postprocess_surface import RawVideoShape  # noqa: E402
from tac.optimization.sparse_residual_oracle import (  # noqa: E402
    SparseResidualOracleConfig,
    authority_payload,
    plan_payload,
    select_sparse_residual_plan,
    sha256_file,
    write_charge_proxy_archive,
    write_sparse_residual_candidate,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _candidate_slug(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    return safe[:96] if safe else "unnamed"


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
    config = SparseResidualOracleConfig(
        top_k_pixels=args.top_k_pixels,
        max_abs_delta=args.max_abs_delta,
        frame_selector=args.frame_selector,
        gain_metric=args.gain_metric,
        chunk_frames=args.chunk_frames,
        quantize_bits=args.quantize_bits,
        compression=args.compression,
        rate_cap_bytes=args.rate_cap_bytes,
    )
    output_root = args.output_root.resolve()
    baseline_raw = args.baseline_raw.resolve()
    target_raw = args.target_raw.resolve()
    baseline_archive = args.archive.resolve()
    target_hash = sha256_file(target_raw)
    candidate_id = _candidate_slug(args.candidate_id or f"target_{target_hash[:12]}")
    candidate_dir = output_root / f"k{args.top_k_pixels}_d{args.max_abs_delta}_{args.frame_selector}_{candidate_id}"
    if candidate_dir.exists() and any(candidate_dir.iterdir()) and not args.overwrite_candidate:
        raise SystemExit(
            f"candidate directory already exists and is non-empty: {candidate_dir}. "
            "Pass --overwrite-candidate only for an intentional rerun."
        )
    candidate_dir.mkdir(parents=True, exist_ok=True)
    correction_bin = candidate_dir / "sparse_residual_corrections.bin"
    candidate_raw = candidate_dir / "0.raw"
    charge_proxy_archive = candidate_dir / "archive_charge_proxy.zip"

    plan = select_sparse_residual_plan(
        baseline_raw=baseline_raw,
        target_raw=target_raw,
        shape=shape,
        config=config,
    )
    apply_result = write_sparse_residual_candidate(
        baseline_raw=baseline_raw,
        output_raw=candidate_raw,
        correction_bin=correction_bin,
        plan=plan,
        shape=shape,
    )
    archive_charge = write_charge_proxy_archive(
        baseline_archive=baseline_archive,
        correction_payload=plan.packed,
        output_archive=charge_proxy_archive,
    )

    row: dict[str, Any] = {
        "schema": "sparse_residual_oracle_candidate.v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "producer": "tools/run_sparse_residual_oracle_smoke.py",
        "inputs": {
            "baseline_raw": str(baseline_raw),
            "baseline_raw_sha256": sha256_file(baseline_raw),
            "target_raw": str(target_raw),
            "target_raw_sha256": target_hash,
            "baseline_archive": str(baseline_archive),
            "baseline_archive_sha256": sha256_file(baseline_archive),
            "shape": shape.as_dict(),
            "candidate_id": candidate_id,
        },
        "plan": plan_payload(config, plan),
        "candidate": apply_result.as_dict(),
        "archive_charge": archive_charge,
        "authority": authority_payload(),
    }
    if args.run_advisory and apply_result.passed_visible_change:
        advisory = _run_advisory(
            raw=candidate_raw,
            archive=charge_proxy_archive,
            output_dir=candidate_dir / "advisory_eval",
            axis_label=args.axis_label,
            batch_size=args.batch_size,
            num_threads=args.num_threads,
            timeout=args.timeout,
        )
        row["advisory_eval"] = advisory
        if args.baseline_score is not None and advisory.get("canonical_score") is not None:
            row["delta_vs_baseline_score"] = float(advisory["canonical_score"]) - float(args.baseline_score)
    elif args.run_advisory:
        row["advisory_eval"] = {"skipped": True, "reason": "no_visible_raw_change"}

    advisory_payload = row.get("advisory_eval")
    advisory_terminal = (
        not args.run_advisory
        or isinstance(advisory_payload, dict)
        and (advisory_payload.get("returncode") == 0 or advisory_payload.get("skipped") is True)
    )
    if args.cleanup_candidate_raw and candidate_raw.is_file() and advisory_terminal:
        row["cleanup"] = {
            "candidate_raw_deleted": True,
            "candidate_raw_sha256_before_delete": sha256_file(candidate_raw),
            "candidate_raw_bytes_before_delete": candidate_raw.stat().st_size,
        }
        candidate_raw.unlink()
    else:
        row["cleanup"] = {"candidate_raw_deleted": False}

    _write_json(candidate_dir / "sparse_residual_candidate_manifest.json", row)

    summary = {
        "schema": "sparse_residual_oracle_smoke.v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "producer": "tools/run_sparse_residual_oracle_smoke.py",
        "candidate_manifest": str(candidate_dir / "sparse_residual_candidate_manifest.json"),
        "summary": {
            "n_kept": plan.sparse["n_kept"],
            "packed_bytes": plan.packed_bytes,
            "changed_pixel_count": apply_result.changed_pixel_count,
            "changed_byte_count": apply_result.changed_byte_count,
            "changed_frame_count": apply_result.changed_frame_count,
            "advisory_score": row.get("advisory_eval", {}).get("canonical_score")
            if isinstance(row.get("advisory_eval"), dict)
            else None,
            "delta_vs_baseline_score": row.get("delta_vs_baseline_score"),
            "score_claim": False,
        },
        "candidate": row,
        "authority": authority_payload(),
    }
    _write_json(args.output.resolve(), summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-raw", type=Path, required=True)
    parser.add_argument("--target-raw", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--candidate-id",
        help=(
            "Optional stable slug appended to the candidate directory. "
            "Defaults to a target-raw SHA prefix so runs with different decode "
            "semantics cannot overwrite each other."
        ),
    )
    parser.add_argument(
        "--overwrite-candidate",
        action="store_true",
        help="Allow overwriting a non-empty candidate directory for intentional reruns.",
    )
    parser.add_argument("--top-k-pixels", type=int, required=True)
    parser.add_argument("--max-abs-delta", type=int, default=1)
    parser.add_argument("--frame-selector", choices=["all", "even", "odd"], default="all")
    parser.add_argument("--gain-metric", choices=["l1", "linf"], default="l1")
    parser.add_argument("--chunk-frames", type=int, default=8)
    parser.add_argument("--quantize-bits", type=int, choices=[4, 8, 16], default=8)
    parser.add_argument("--compression", choices=["zlib", "none"], default="zlib")
    parser.add_argument("--rate-cap-bytes", type=int)
    parser.add_argument("--frames", type=int, default=1200)
    parser.add_argument("--height", type=int, default=874)
    parser.add_argument("--width", type=int, default=1164)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--run-advisory", action="store_true")
    parser.add_argument("--baseline-score", type=float)
    parser.add_argument("--axis-label", default="[macOS-CPU advisory sparse-residual-oracle]")
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
