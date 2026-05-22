#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run official inflate/raw controls for FEC6 decoder q-mutation candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "rank_or_kill_eligible": False,
    "promotable": False,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _extract_single_member(archive_zip: Path, output_dir: Path, *, expected_member: str = "x") -> dict[str, Any]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    with zipfile.ZipFile(archive_zip) as zf:
        infos = zf.infolist()
        names = [info.filename for info in infos]
        if names != [expected_member]:
            raise SystemExit(f"{archive_zip} must contain exactly {expected_member!r}; got {names!r}")
        info = infos[0]
        if info.compress_type != zipfile.ZIP_STORED:
            raise SystemExit(
                f"{archive_zip} member {expected_member!r} must be ZIP_STORED; "
                f"got compress_type={info.compress_type}"
            )
        data = zf.read(expected_member)
        if len(data) != info.file_size:
            raise SystemExit(
                f"{archive_zip} member {expected_member!r} size mismatch: "
                f"info.file_size={info.file_size}, read={len(data)}"
            )
    out = output_dir / expected_member
    out.write_bytes(data)
    return {
        "archive_zip": str(archive_zip.resolve()),
        "archive_zip_bytes": archive_zip.stat().st_size,
        "archive_zip_sha256": _sha256_file(archive_zip),
        "member": expected_member,
        "member_bytes": len(data),
        "member_sha256": _sha256_bytes(data),
        "compress_type": int(info.compress_type),
        "crc": int(info.CRC),
        "extracted_path": str(out.resolve()),
    }


def _run_inflate(
    *,
    runtime_dir: Path,
    data_dir: Path,
    output_dir: Path,
    file_list: Path,
    python_bin: str,
    timeout: int,
) -> dict[str, Any]:
    inflate_sh = runtime_dir / "inflate.sh"
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PACT_PYTHON_BIN"] = python_bin
    cmd = ["bash", str(inflate_sh), str(data_dir), str(output_dir), str(file_list)]
    start = time.monotonic()
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "elapsed_seconds": time.monotonic() - start,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _compare_raws(
    *,
    baseline_raw: Path,
    candidate_raw: Path,
    frame_bytes: int,
    sample_limit: int,
) -> dict[str, Any]:
    baseline_size = baseline_raw.stat().st_size
    candidate_size = candidate_raw.stat().st_size
    if baseline_size != candidate_size:
        raise SystemExit(
            f"raw size mismatch: baseline={baseline_size}, candidate={candidate_size}"
        )
    if baseline_size % frame_bytes:
        raise SystemExit(f"baseline size {baseline_size} not divisible by frame bytes {frame_bytes}")
    baseline_digest = hashlib.sha256()
    candidate_digest = hashlib.sha256()
    changed: list[int] = []
    samples: list[dict[str, Any]] = []
    byte_l1_total = 0
    byte_linf = 0
    changed_byte_count = 0
    changed_frame_l1: list[tuple[int, int]] = []
    with baseline_raw.open("rb") as left, candidate_raw.open("rb") as right:
        frame_index = 0
        while True:
            left_frame = left.read(frame_bytes)
            right_frame = right.read(frame_bytes)
            if not left_frame and not right_frame:
                break
            if len(left_frame) != frame_bytes or len(right_frame) != frame_bytes:
                raise SystemExit("truncated raw frame while comparing")
            baseline_digest.update(left_frame)
            candidate_digest.update(right_frame)
            if left_frame != right_frame:
                changed.append(frame_index)
                left_arr = np.frombuffer(left_frame, dtype=np.uint8).astype(np.int16)
                right_arr = np.frombuffer(right_frame, dtype=np.uint8).astype(np.int16)
                diff = np.abs(right_arr - left_arr)
                frame_l1 = int(diff.sum(dtype=np.int64))
                frame_linf = int(diff.max(initial=0))
                frame_changed_bytes = int(np.count_nonzero(diff))
                byte_l1_total += frame_l1
                byte_linf = max(byte_linf, frame_linf)
                changed_byte_count += frame_changed_bytes
                changed_frame_l1.append((frame_index, frame_l1))
                if len(samples) < sample_limit:
                    samples.append(
                        {
                            "frame_index": frame_index,
                            "baseline_sha256": _sha256_bytes(left_frame),
                            "candidate_sha256": _sha256_bytes(right_frame),
                            "byte_l1": frame_l1,
                            "changed_byte_count": frame_changed_bytes,
                            "byte_linf": frame_linf,
                        }
                    )
            frame_index += 1
    changed_frame_l1.sort(key=lambda item: (-item[1], item[0]))
    return {
        "frame_count": frame_index,
        "frame_bytes": frame_bytes,
        "baseline_raw_bytes": baseline_size,
        "candidate_raw_bytes": candidate_size,
        "baseline_raw_sha256": baseline_digest.hexdigest(),
        "candidate_raw_sha256": candidate_digest.hexdigest(),
        "changed_frame_count": len(changed),
        "changed_frame_indices": changed,
        "changed_frame_hashes_sample": samples,
        "byte_delta_summary": {
            "byte_l1_total": byte_l1_total,
            "byte_linf": byte_linf,
            "changed_byte_count": changed_byte_count,
            "changed_byte_fraction": (
                changed_byte_count / float(candidate_size) if candidate_size else 0.0
            ),
            "top_changed_frames_by_l1": [
                {"frame_index": frame_index, "byte_l1": frame_l1}
                for frame_index, frame_l1 in changed_frame_l1[:sample_limit]
            ],
        },
        "passed_visible_change": len(changed) > 0,
    }


def _candidate_dirs(candidate_root: Path, candidate_ids: list[str], max_candidates: int | None) -> list[Path]:
    if candidate_ids:
        rows = [candidate_root / candidate_id for candidate_id in candidate_ids]
    else:
        rows = [
            path
            for path in sorted(candidate_root.iterdir())
            if path.is_dir() and (path / "archive.zip").is_file()
        ]
    missing = [str(path) for path in rows if not (path / "archive.zip").is_file()]
    if missing:
        raise SystemExit(f"candidate archive missing for selected ids: {missing}")
    if max_candidates is not None:
        rows = rows[: int(max_candidates)]
    return rows


def run_controls(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    runtime_dir = args.runtime_dir.resolve()
    file_list = output_root / "file_list.txt"
    output_root.mkdir(parents=True, exist_ok=True)
    file_list.write_text(args.file_list_name + "\n", encoding="utf-8")

    frame_bytes = int(args.frame_height) * int(args.frame_width) * 3
    baseline_dir = output_root / "baseline"
    baseline_raw = args.baseline_raw.resolve() if args.baseline_raw else baseline_dir / "inflated" / Path(args.file_list_name).with_suffix(".raw")
    baseline_extract = None
    baseline_inflate = None
    if args.baseline_raw is None or not baseline_raw.is_file():
        if args.source_archive_zip is None:
            raise SystemExit("provide --source-archive-zip when --baseline-raw is absent")
        baseline_extract = _extract_single_member(
            args.source_archive_zip.resolve(),
            baseline_dir / "data_dir",
        )
        baseline_inflate = _run_inflate(
            runtime_dir=runtime_dir,
            data_dir=baseline_dir / "data_dir",
            output_dir=baseline_dir / "inflated",
            file_list=file_list,
            python_bin=args.python_bin,
            timeout=args.inflate_timeout,
        )
        (baseline_dir / "inflate.stdout.log").write_text(baseline_inflate["stdout"], encoding="utf-8")
        (baseline_dir / "inflate.stderr.log").write_text(baseline_inflate["stderr"], encoding="utf-8")
        if baseline_inflate["returncode"] != 0:
            raise SystemExit(f"baseline inflate failed: {baseline_inflate['returncode']}")
        if not baseline_raw.is_file():
            raise SystemExit(f"baseline raw not produced: {baseline_raw}")

    baseline_record = {
        "raw_path": str(baseline_raw.resolve()),
        "raw_bytes": baseline_raw.stat().st_size,
        "raw_sha256": _sha256_file(baseline_raw),
        "archive_extract": baseline_extract,
        "inflate": (
            {
                "cmd": baseline_inflate["cmd"],
                "returncode": baseline_inflate["returncode"],
                "elapsed_seconds": baseline_inflate["elapsed_seconds"],
                "stdout_log": str((baseline_dir / "inflate.stdout.log").resolve()),
                "stderr_log": str((baseline_dir / "inflate.stderr.log").resolve()),
            }
            if baseline_inflate is not None
            else None
        ),
    }

    rows = []
    for candidate_dir in _candidate_dirs(
        args.candidate_root.resolve(),
        args.candidate_id,
        args.max_candidates,
    ):
        out_dir = output_root / "candidates" / candidate_dir.name
        archive_zip = candidate_dir / "archive.zip"
        extract = _extract_single_member(archive_zip, out_dir / "data_dir")
        inflate = _run_inflate(
            runtime_dir=runtime_dir,
            data_dir=out_dir / "data_dir",
            output_dir=out_dir / "inflated",
            file_list=file_list,
            python_bin=args.python_bin,
            timeout=args.inflate_timeout,
        )
        (out_dir / "inflate.stdout.log").write_text(inflate["stdout"], encoding="utf-8")
        (out_dir / "inflate.stderr.log").write_text(inflate["stderr"], encoding="utf-8")
        raw_path = out_dir / "inflated" / Path(args.file_list_name).with_suffix(".raw")
        comparison = None
        blockers: list[str] = [
            "advisory_component_response_not_measured",
            "exact_cuda_auth_eval_missing",
        ]
        if inflate["returncode"] != 0 or not raw_path.is_file():
            blockers.append("official_inflate_failed")
        else:
            comparison = _compare_raws(
                baseline_raw=baseline_raw,
                candidate_raw=raw_path,
                frame_bytes=frame_bytes,
                sample_limit=args.sample_limit,
            )
            if not comparison["passed_visible_change"]:
                blockers.append("no_visible_raw_change")
            if args.cleanup_candidate_raw:
                raw_sha = comparison["candidate_raw_sha256"]
                raw_path.unlink()
                cleanup = {
                    "candidate_raw_deleted": True,
                    "candidate_raw_sha256_before_delete": raw_sha,
                }
            else:
                cleanup = {"candidate_raw_deleted": False}
        manifest_path = candidate_dir / "mutation_manifest.json"
        mutation_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
        row = {
            "candidate_id": candidate_dir.name,
            "source_candidate_dir": str(candidate_dir.resolve()),
            "mutation_manifest": mutation_manifest,
            "archive_extract": extract,
            "inflate": {
                "cmd": inflate["cmd"],
                "returncode": inflate["returncode"],
                "elapsed_seconds": inflate["elapsed_seconds"],
                "stdout_log": str((out_dir / "inflate.stdout.log").resolve()),
                "stderr_log": str((out_dir / "inflate.stderr.log").resolve()),
                "output_raw_path": str(raw_path.resolve()),
                "output_raw_exists_after_cleanup": raw_path.is_file(),
            },
            "raw_comparison": comparison,
            "cleanup": cleanup if inflate["returncode"] == 0 and comparison is not None else None,
            "blockers": blockers,
            **FALSE_AUTHORITY,
        }
        _write_json(out_dir / "inflate_control_manifest.json", row)
        rows.append(row)

    visible = sum(
        1
        for row in rows
        if isinstance(row.get("raw_comparison"), dict)
        and row["raw_comparison"].get("passed_visible_change")
    )
    return {
        "schema": "fec6_decoder_q_candidate_inflate_controls_v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "producer": "tools/run_decoder_q_candidate_inflate_controls.py",
        "inputs": {
            "runtime_dir": str(runtime_dir),
            "candidate_root": str(args.candidate_root.resolve()),
            "source_archive_zip": str(args.source_archive_zip.resolve()) if args.source_archive_zip else None,
            "file_list_name": args.file_list_name,
            "frame_height": int(args.frame_height),
            "frame_width": int(args.frame_width),
            "frame_bytes": frame_bytes,
            "max_candidates": args.max_candidates,
            "candidate_id": args.candidate_id,
            "cleanup_candidate_raw": bool(args.cleanup_candidate_raw),
        },
        "baseline": baseline_record,
        "summary": {
            "candidate_count": len(rows),
            "visible_change_count": visible,
            "no_visible_change_count": len(rows) - visible,
        },
        "candidates": rows,
        "authority": {
            **FALSE_AUTHORITY,
            "notes": "Official inflate/raw visibility control only; component response and exact eval still required.",
        },
        **FALSE_AUTHORITY,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-archive-zip", type=Path)
    parser.add_argument("--baseline-raw", type=Path)
    parser.add_argument("--candidate-id", action="append", default=[])
    parser.add_argument("--file-list-name", default="0.hevc")
    parser.add_argument("--frame-height", type=int, default=874)
    parser.add_argument("--frame-width", type=int, default=1164)
    parser.add_argument("--python-bin", default=".venv/bin/python")
    parser.add_argument("--inflate-timeout", type=int, default=900)
    parser.add_argument("--sample-limit", type=int, default=16)
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--cleanup-candidate-raw", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_controls(args)
    _write_json(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "candidate_count": payload["summary"]["candidate_count"],
                "visible_change_count": payload["summary"]["visible_change_count"],
                "no_visible_change_count": payload["summary"]["no_visible_change_count"],
                "baseline_raw_sha256": payload["baseline"]["raw_sha256"],
                "score_claim": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
