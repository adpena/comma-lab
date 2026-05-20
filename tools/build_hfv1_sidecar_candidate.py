#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build deterministic HFV1 sidecar candidates for PR101/PR110-style runtimes.

This tool is intentionally scorer-free. It can build a byte-closed archive,
stage an official ``inflate.sh <archive_dir> <output_dir> <file_list>`` run,
and compare changed raw frames against a baseline raw output. It never makes a
score claim and never marks a candidate exact-eval ready.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

LFV1_HEADER = "<4sHHHH"
LFV1_ROW = "<BHHHHHH"
HFV1_HEADER = "<4sIII"
HFV1_ROW = "<fffff"
LFV1_ALPHA_MAX_BY_VERSION = {1: 8.0, 2: 0.02}
LFV1_POWER_MAX_BY_VERSION = {1: 8.0, 2: 8.0}
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
DEFAULT_FRAME_HEIGHT = 874
DEFAULT_FRAME_WIDTH = 1164
DEFAULT_FRAME_COUNT = 1200
DEFAULT_FILE_LIST_NAME = "0.mkv"
OUTPUT_RAW_NAME = "0.raw"


class Hfv1CandidateError(ValueError):
    """Raised when an HFV1 sidecar candidate cannot be built safely."""


SAFE_CANDIDATE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_pair_list(raw: str) -> list[int]:
    pairs: list[int] = []
    for token in raw.split(","):
        item = token.strip()
        if not item:
            continue
        value = int(item)
        if value < 0:
            raise Hfv1CandidateError(f"pair indices must be non-negative: {value}")
        pairs.append(value)
    if not pairs:
        raise Hfv1CandidateError("--pair-list did not contain any pair indices")
    return pairs


def _validate_candidate_id(candidate_id: str) -> str:
    if not SAFE_CANDIDATE_ID.fullmatch(candidate_id) or candidate_id in {".", ".."}:
        raise Hfv1CandidateError(f"unsafe candidate id: {candidate_id!r}")
    return candidate_id


def _validate_selected_pairs(selected_pairs: list[int], *, frame_count: int) -> list[int]:
    if frame_count <= 0:
        raise Hfv1CandidateError("--frame-count must be positive")
    max_pair_exclusive = frame_count // 2
    if len(set(selected_pairs)) != len(selected_pairs):
        raise Hfv1CandidateError(f"duplicate selected pairs are not LFV1-safe: {selected_pairs}")
    for pair in selected_pairs:
        if pair < 0 or pair >= max_pair_exclusive:
            raise Hfv1CandidateError(
                f"pair index {pair} outside emitted pair range [0, {max_pair_exclusive})"
            )
    return selected_pairs


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise Hfv1CandidateError(f"{path}:{line_no}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise Hfv1CandidateError(f"{path}:{line_no}: row must be an object")
            rows.append(row)
    return rows


def _select_pairs_from_rows(
    *,
    pair_rows: Path,
    top_k: int,
    selection_key: str,
    mode_id: str | None,
    descending: bool,
) -> tuple[list[int], list[dict[str, Any]]]:
    if top_k <= 0:
        raise Hfv1CandidateError("--top-k must be positive")
    rows = _load_jsonl_rows(pair_rows)
    if mode_id is not None:
        rows = [row for row in rows if row.get("mode_id") == mode_id]
    scored: list[dict[str, Any]] = []
    seen_pairs: set[int] = set()
    for row in rows:
        pair = row.get("pair")
        metric = row.get(selection_key)
        if isinstance(pair, bool) or not isinstance(pair, int):
            continue
        if isinstance(metric, bool) or not isinstance(metric, (int, float)):
            continue
        if pair in seen_pairs:
            continue
        seen_pairs.add(int(pair))
        item = dict(row)
        item["_selection_value"] = float(metric)
        scored.append(item)
    if not scored:
        raise Hfv1CandidateError(
            f"no selectable rows in {pair_rows} for key={selection_key!r}, mode_id={mode_id!r}"
        )
    scored.sort(key=lambda row: float(row["_selection_value"]), reverse=descending)
    selected = scored[:top_k]
    return [int(row["pair"]) for row in selected], selected


def _read_x_member(source_archive: Path) -> bytes:
    if not source_archive.is_file():
        raise Hfv1CandidateError(f"source archive does not exist: {source_archive}")
    with zipfile.ZipFile(source_archive, "r") as archive:
        names = archive.namelist()
        if "x" not in names:
            raise Hfv1CandidateError(f"{source_archive} does not contain required member 'x'")
        return archive.read("x")


def _rate_denominator_bytes(upstream_dir: Path | None) -> int | None:
    if upstream_dir is None:
        return None
    videos = upstream_dir / "videos"
    if not videos.is_dir():
        return None
    return sum(path.stat().st_size for path in videos.rglob("*") if path.is_file())


def _quantize_u16(value: float, low: float, high: float) -> int:
    if high <= low:
        raise Hfv1CandidateError("quantization range must have high > low")
    scaled = round((float(value) - float(low)) / (float(high) - float(low)) * 65_535.0)
    return max(0, min(65_535, int(scaled)))


def _active_geometry_values(
    *,
    frame_height: int,
    frame_width: int,
    alpha: float,
    radius_scale: float,
    power: float,
    origin_x_frac: float,
    origin_y_frac: float,
) -> tuple[tuple[float, float, float, float, float], tuple[float, float, float, float, float]]:
    radius = math.hypot(float(frame_width), float(frame_height))
    identity_row = (0.0, radius, 1.0, (frame_width - 1) * 0.5, (frame_height - 1) * 0.5)
    active_row = (
        float(alpha),
        max(float(radius_scale), 1e-9) * radius,
        float(power),
        float(origin_x_frac) * float(frame_width - 1),
        float(origin_y_frac) * float(frame_height - 1),
    )
    return identity_row, active_row


def _pack_hfv1(
    *,
    active_frames: set[int],
    frame_count: int,
    frame_height: int,
    frame_width: int,
    alpha: float,
    radius_scale: float,
    power: float,
    origin_x_frac: float,
    origin_y_frac: float,
) -> tuple[bytes, dict[str, Any]]:
    if frame_count <= 0 or frame_height <= 0 or frame_width <= 0:
        raise Hfv1CandidateError("frame count, height, and width must be positive")
    for frame in active_frames:
        if frame < 0 or frame >= frame_count:
            raise Hfv1CandidateError(f"active frame {frame} outside [0, {frame_count})")
    identity_row, active_row = _active_geometry_values(
        frame_height=frame_height,
        frame_width=frame_width,
        alpha=alpha,
        radius_scale=radius_scale,
        power=power,
        origin_x_frac=origin_x_frac,
        origin_y_frac=origin_y_frac,
    )
    import struct

    header = struct.Struct(HFV1_HEADER)
    row = struct.Struct(HFV1_ROW)
    body = b"".join(
        row.pack(*(active_row if index in active_frames else identity_row))
        for index in range(frame_count)
    )
    raw = header.pack(b"HFV1", frame_count, frame_height, frame_width) + body
    return raw, {
        "sidecar_format": "HFV1_full",
        "format": "HFV1 <4sIII header + float32 alpha/radius/power/origin_x/origin_y rows",
        "frame_count": frame_count,
        "image_size": {"height": frame_height, "width": frame_width},
        "identity_alpha_fast_path_required": True,
        "identity_row": {
            "alpha": identity_row[0],
            "radius": identity_row[1],
            "power": identity_row[2],
            "origin_x": identity_row[3],
            "origin_y": identity_row[4],
        },
        "active_row": {
            "alpha": active_row[0],
            "radius": active_row[1],
            "power": active_row[2],
            "origin_x": active_row[3],
            "origin_y": active_row[4],
        },
    }


def _pack_lfv1(
    *,
    version: int,
    active_pairs: list[int],
    frame_height: int,
    frame_width: int,
    alpha: float,
    radius_scale: float,
    power: float,
    origin_x_frac: float,
    origin_y_frac: float,
) -> tuple[bytes, dict[str, Any]]:
    if frame_height <= 0 or frame_width <= 0:
        raise Hfv1CandidateError("height and width must be positive")
    alpha_max = LFV1_ALPHA_MAX_BY_VERSION.get(int(version))
    power_max = LFV1_POWER_MAX_BY_VERSION.get(int(version))
    if alpha_max is None or power_max is None:
        raise Hfv1CandidateError(f"unsupported LFV1 version: {version}")
    for pair in active_pairs:
        if pair < 0 or pair > 65_535:
            raise Hfv1CandidateError(f"pair index outside LFV1 uint16 range: {pair}")
    _identity_row, active_row = _active_geometry_values(
        frame_height=frame_height,
        frame_width=frame_width,
        alpha=alpha,
        radius_scale=radius_scale,
        power=power,
        origin_x_frac=origin_x_frac,
        origin_y_frac=origin_y_frac,
    )
    import struct

    header = struct.Struct(LFV1_HEADER)
    row = struct.Struct(LFV1_ROW)
    diag = math.hypot(float(frame_width), float(frame_height))
    rows = [
        row.pack(
            1,
            int(pair),
            _quantize_u16(active_row[0], 0.0, alpha_max),
            _quantize_u16(active_row[1], 0.0, diag),
            _quantize_u16(active_row[2], 0.0, power_max),
            _quantize_u16(active_row[3], 0.0, float(frame_width - 1)),
            _quantize_u16(active_row[4], 0.0, float(frame_height - 1)),
        )
        for pair in active_pairs
    ]
    raw = header.pack(b"LFV1", int(version), len(rows), frame_width, frame_height) + b"".join(rows)
    return raw, {
        "sidecar_format": "LFV1_sparse",
        "format": "LFV1 <4sHHHH header + uint8 opcode/uint16 pair and quantized geometry rows",
        "version": int(version),
        "row_bytes": struct.Struct(LFV1_ROW).size,
        "header_bytes": struct.Struct(LFV1_HEADER).size,
        "row_count": len(rows),
        "quantization_ranges": {
            "alpha": [0.0, alpha_max],
            "radius": [0.0, diag],
            "power": [0.0, power_max],
            "origin_x": [0.0, float(frame_width - 1)],
            "origin_y": [0.0, float(frame_height - 1)],
        },
        "image_size": {"height": frame_height, "width": frame_width},
        "active_row_float_source": {
            "alpha": active_row[0],
            "radius": active_row[1],
            "power": active_row[2],
            "origin_x": active_row[3],
            "origin_y": active_row[4],
        },
    }


def _write_stored_zip(path: Path, members: dict[str, bytes]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(members):
            info = zipfile.ZipInfo(name, FIXED_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            info.create_system = 3
            payload = members[name]
            archive.writestr(info, payload, compress_type=zipfile.ZIP_STORED)
            records.append(
                {
                    "name": name,
                    "bytes": len(payload),
                    "sha256": _sha256_bytes(payload),
                    "compression_method": "ZIP_STORED",
                }
            )
    return records


def _extract_stored_zip(path: Path, output_dir: Path) -> list[dict[str, Any]]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with zipfile.ZipFile(path, "r") as archive:
        for info in archive.infolist():
            name = info.filename
            if name in seen:
                raise Hfv1CandidateError(f"duplicate ZIP member in candidate archive: {name!r}")
            seen.add(name)
            member_path = Path(name)
            if member_path.is_absolute() or ".." in member_path.parts or len(member_path.parts) != 1:
                raise Hfv1CandidateError(f"unsafe ZIP member in candidate archive: {name!r}")
            if info.compress_type != zipfile.ZIP_STORED:
                raise Hfv1CandidateError(f"candidate archive member is not ZIP_STORED: {name!r}")
            payload = archive.read(info)
            out_path = output_dir / name
            out_path.write_bytes(payload)
            records.append(
                {
                    "name": name,
                    "path": str(out_path),
                    "bytes": len(payload),
                    "sha256": _sha256_bytes(payload),
                    "compression_method": "ZIP_STORED",
                }
            )
    required = {"x"}
    missing = sorted(required - seen)
    if missing:
        raise Hfv1CandidateError(f"candidate archive missing required members: {missing}")
    return records


def _stream_changed_frames(
    *,
    baseline_raw: Path,
    candidate_raw: Path,
    frame_bytes: int,
) -> dict[str, Any]:
    if not baseline_raw.is_file():
        raise Hfv1CandidateError(f"baseline raw does not exist: {baseline_raw}")
    if not candidate_raw.is_file():
        raise Hfv1CandidateError(f"candidate raw does not exist: {candidate_raw}")
    changed: list[int] = []
    sample_hashes: list[dict[str, str | int]] = []
    baseline_sha = hashlib.sha256()
    candidate_sha = hashlib.sha256()
    frame_index = 0
    with baseline_raw.open("rb") as baseline, candidate_raw.open("rb") as candidate:
        while True:
            left = baseline.read(frame_bytes)
            right = candidate.read(frame_bytes)
            if not left and not right:
                break
            if len(left) != frame_bytes or len(right) != frame_bytes:
                raise Hfv1CandidateError(
                    f"raw frame alignment failure at frame {frame_index}: "
                    f"{len(left)} vs {len(right)} bytes"
                )
            baseline_sha.update(left)
            candidate_sha.update(right)
            if left != right:
                changed.append(frame_index)
                if len(sample_hashes) < 64:
                    sample_hashes.append(
                        {
                            "frame_index": frame_index,
                            "baseline_sha256": _sha256_bytes(left),
                            "candidate_sha256": _sha256_bytes(right),
                        }
                    )
            frame_index += 1
    return {
        "frame_count": frame_index,
        "baseline_raw_bytes": baseline_raw.stat().st_size,
        "candidate_raw_bytes": candidate_raw.stat().st_size,
        "baseline_raw_sha256": baseline_sha.hexdigest(),
        "candidate_raw_sha256": candidate_sha.hexdigest(),
        "changed_frame_indices": changed,
        "changed_frame_hashes_sample": sample_hashes,
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
    if not inflate_sh.is_file():
        raise Hfv1CandidateError(f"inflate.sh not found: {inflate_sh}")
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
    elapsed = time.monotonic() - start
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "elapsed_seconds": elapsed,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def build_candidate(args: argparse.Namespace) -> dict[str, Any]:
    out_root = args.output_root.resolve()
    candidate_id = _validate_candidate_id(args.candidate_id)
    candidate_dir = (out_root / candidate_id).resolve()
    if candidate_dir.parent != out_root:
        raise Hfv1CandidateError(f"candidate directory escaped output root: {candidate_dir}")
    candidate_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir = args.runtime_dir.resolve() if args.runtime_dir is not None else None
    source_archive = (
        args.source_archive.resolve()
        if args.source_archive is not None
        else (runtime_dir / "archive.zip" if runtime_dir is not None else None)
    )
    if source_archive is None:
        raise Hfv1CandidateError("provide --source-archive or --runtime-dir with archive.zip")

    if args.pair_list:
        selected_pairs = _parse_pair_list(args.pair_list)
        selected_rows: list[dict[str, Any]] = []
        selection_rule = "explicit_pair_list"
    else:
        if args.pair_rows is None:
            raise Hfv1CandidateError("provide --pair-rows unless --pair-list is used")
        selected_pairs, selected_rows = _select_pairs_from_rows(
            pair_rows=args.pair_rows.resolve(),
            top_k=args.top_k,
            selection_key=args.selection_key,
            mode_id=args.mode_id,
            descending=not args.ascending,
        )
        selection_rule = (
            f"top{args.top_k}_{args.selection_key}"
            + (f"_mode_id_{args.mode_id}" if args.mode_id is not None else "")
        )

    if args.sidecar_format == "lfv1" and args.max_sidecar_bytes is not None:
        import struct

        max_rows = (
            int(args.max_sidecar_bytes)
            - struct.calcsize(LFV1_HEADER)
        ) // struct.calcsize(LFV1_ROW)
        if max_rows <= 0:
            raise Hfv1CandidateError("--max-sidecar-bytes is too small for any LFV1 row")
        selected_pairs = selected_pairs[:max_rows]
        selected_rows = selected_rows[:max_rows]
    selected_pairs = _validate_selected_pairs(selected_pairs, frame_count=args.frame_count)
    selected_frames = sorted({frame for pair in selected_pairs for frame in (2 * pair, 2 * pair + 1)})
    if args.sidecar_format == "hfv1":
        sidecar_raw, sidecar_contract = _pack_hfv1(
            active_frames=set(selected_frames),
            frame_count=args.frame_count,
            frame_height=args.frame_height,
            frame_width=args.frame_width,
            alpha=args.alpha,
            radius_scale=args.radius_scale,
            power=args.power,
            origin_x_frac=args.origin_x_frac,
            origin_y_frac=args.origin_y_frac,
        )
        sidecar_member = "foveation_params.bin"
    else:
        sidecar_raw, sidecar_contract = _pack_lfv1(
            version=args.lfv1_version,
            active_pairs=selected_pairs,
            frame_height=args.frame_height,
            frame_width=args.frame_width,
            alpha=args.alpha,
            radius_scale=args.radius_scale,
            power=args.power,
            origin_x_frac=args.origin_x_frac,
            origin_y_frac=args.origin_y_frac,
        )
        sidecar_member = "lapose_foveation_tuples.lfv1"
    x_payload = _read_x_member(source_archive)

    data_dir = candidate_dir / "data_dir"
    archive_extract_dir = candidate_dir / "archive_extracted_data_dir"
    archive_dir = candidate_dir / "archive"
    output_dir = candidate_dir / "inflated"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "x").write_bytes(x_payload)
    (data_dir / sidecar_member).write_bytes(sidecar_raw)
    file_list = candidate_dir / "file_list.txt"
    file_list.write_text(args.file_list_name + "\n", encoding="utf-8")
    archive_path = archive_dir / "archive.zip"
    member_records = _write_stored_zip(
        archive_path,
        {
            sidecar_member: sidecar_raw,
            "x": x_payload,
        },
    )
    extracted_member_records = _extract_stored_zip(archive_path, archive_extract_dir)
    rate_denominator = _rate_denominator_bytes(args.upstream_dir.resolve() if args.upstream_dir is not None else None)
    archive_delta_bytes = archive_path.stat().st_size - source_archive.stat().st_size

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "hfv1_sidecar_candidate_v1",
        "producer": "tools/build_hfv1_sidecar_candidate.py",
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "candidate_id": candidate_id,
        "candidate_dir": str(candidate_dir),
        "source_archive": {
            "path": str(source_archive),
            "bytes": source_archive.stat().st_size,
            "sha256": _sha256_file(source_archive),
        },
        "runtime_dir": str(runtime_dir) if runtime_dir is not None else None,
        "selection": {
            "selection_rule": selection_rule,
            "pair_rows": str(args.pair_rows.resolve()) if args.pair_rows is not None else None,
            "selection_key": args.selection_key,
            "mode_id": args.mode_id,
            "selected_pairs": selected_pairs,
            "selected_frames": selected_frames,
            "selected_pair_records": selected_rows,
        },
        "sidecar": {
            **sidecar_contract,
            "member": sidecar_member,
            "path": str(data_dir / sidecar_member),
            "bytes": len(sidecar_raw),
            "sha256": _sha256_bytes(sidecar_raw),
            "max_sidecar_bytes": args.max_sidecar_bytes,
        },
        "archive": {
            "path": str(archive_path),
            "bytes": archive_path.stat().st_size,
            "sha256": _sha256_file(archive_path),
            "delta_bytes_vs_source_archive": archive_delta_bytes,
            "rate_denominator_bytes": rate_denominator,
            "estimated_score_rate_penalty": (
                25.0 * float(archive_delta_bytes) / float(rate_denominator)
                if rate_denominator
                else None
            ),
            "members": member_records,
            "extracted_members": extracted_member_records,
            "official_inflate_input": str(archive_extract_dir),
        },
        "sensitivity_contract": {
            "posenet_segnet_sensitivity_collapsed": False,
            "pose_route": "pair_index -> frames 2k and 2k+1",
            "seg_route": "frame_index after selected-pair expansion",
            "orthogonalization_status": "candidate_build_only_no_joint_training",
            "inflate_time_scorer_or_training_dependency": False,
        },
        "blockers": [
            "component_response_not_measured_for_hfv1_candidate",
            "exact_cuda_auth_eval_missing",
        ],
    }

    if args.run_inflate:
        if runtime_dir is None:
            raise Hfv1CandidateError("--run-inflate requires --runtime-dir")
        run = _run_inflate(
            runtime_dir=runtime_dir,
            data_dir=archive_extract_dir,
            output_dir=output_dir,
            file_list=file_list,
            python_bin=args.python_bin,
            timeout=args.inflate_timeout,
        )
        logs_dir = candidate_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        (logs_dir / "inflate.stdout.log").write_text(run["stdout"], encoding="utf-8")
        (logs_dir / "inflate.stderr.log").write_text(run["stderr"], encoding="utf-8")
        raw_path = output_dir / Path(args.file_list_name).with_suffix(".raw")
        inflate_report: dict[str, Any] = {
            "cmd": run["cmd"],
            "returncode": run["returncode"],
            "elapsed_seconds": run["elapsed_seconds"],
            "stdout_log": str(logs_dir / "inflate.stdout.log"),
            "stderr_log": str(logs_dir / "inflate.stderr.log"),
            "output_raw": {
                "path": str(raw_path),
                "exists": raw_path.is_file(),
                "bytes": raw_path.stat().st_size if raw_path.is_file() else None,
                "sha256": _sha256_file(raw_path) if raw_path.is_file() else None,
            },
        }
        if run["returncode"] != 0:
            manifest["blockers"].append("official_inflate_run_failed")
        elif args.baseline_raw is not None:
            comparison = _stream_changed_frames(
                baseline_raw=args.baseline_raw.resolve(),
                candidate_raw=raw_path,
                frame_bytes=args.frame_height * args.frame_width * 3,
            )
            comparison["expected_changed_frame_indices"] = selected_frames
            comparison["changed_frames_match_selection"] = (
                comparison["changed_frame_indices"] == selected_frames
            )
            expected_set = set(selected_frames)
            changed_set = set(comparison["changed_frame_indices"])
            comparison["changed_frames_within_selection"] = changed_set.issubset(expected_set)
            comparison["selected_frame_change_count"] = len(changed_set & expected_set)
            comparison["unexpected_changed_frame_indices"] = sorted(changed_set - expected_set)
            comparison["passed"] = bool(
                comparison["changed_frames_within_selection"]
                and comparison["selected_frame_change_count"] > 0
                and comparison["baseline_raw_bytes"] == comparison["candidate_raw_bytes"]
            )
            inflate_report["raw_comparison"] = comparison
            if not comparison["passed"]:
                if comparison["baseline_raw_sha256"] == comparison["candidate_raw_sha256"]:
                    manifest["blockers"].append("raw_noop_control_failed")
                else:
                    manifest["blockers"].append("raw_locality_control_failed")
        manifest["official_inflate_control"] = inflate_report

    manifest_path = candidate_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    summary = {
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "archive": manifest["archive"],
        "selected_pairs": selected_pairs,
        "selected_frames": selected_frames,
        "official_inflate_control_passed": (
            manifest.get("official_inflate_control", {})
            .get("raw_comparison", {})
            .get("passed")
        ),
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }
    _write_json(candidate_dir / "summary.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, help="Submission runtime directory containing inflate.sh and optionally archive.zip.")
    parser.add_argument("--source-archive", type=Path, help="Source archive.zip containing member x. Defaults to --runtime-dir/archive.zip.")
    parser.add_argument("--pair-rows", type=Path, help="JSONL rows with pair/component metrics.")
    parser.add_argument("--pair-list", default="", help="Comma-separated explicit pair indices. Overrides --pair-rows selection.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--selection-key", default="component_score_no_rate")
    parser.add_argument("--mode-id", default="none", help="Filter pair rows by mode_id; use empty string to disable.")
    parser.add_argument("--ascending", action="store_true")
    parser.add_argument("--frame-height", type=int, default=DEFAULT_FRAME_HEIGHT)
    parser.add_argument("--frame-width", type=int, default=DEFAULT_FRAME_WIDTH)
    parser.add_argument("--frame-count", type=int, default=DEFAULT_FRAME_COUNT)
    parser.add_argument("--alpha", type=float, default=0.00055)
    parser.add_argument("--radius-scale", type=float, default=0.78)
    parser.add_argument("--power", type=float, default=1.4)
    parser.add_argument("--origin-x-frac", type=float, default=0.5)
    parser.add_argument("--origin-y-frac", type=float, default=0.45)
    parser.add_argument("--sidecar-format", choices=["hfv1", "lfv1"], default="hfv1")
    parser.add_argument("--lfv1-version", type=int, choices=sorted(LFV1_ALPHA_MAX_BY_VERSION), default=2)
    parser.add_argument("--max-sidecar-bytes", type=int)
    parser.add_argument("--upstream-dir", type=Path, default=REPO_ROOT / "upstream")
    parser.add_argument("--run-inflate", action="store_true")
    parser.add_argument("--baseline-raw", type=Path)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--inflate-timeout", type=int, default=1800)
    parser.add_argument("--file-list-name", default=DEFAULT_FILE_LIST_NAME)
    args = parser.parse_args(argv)
    if args.mode_id == "":
        args.mode_id = None
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        summary = build_candidate(parse_args(argv))
    except Hfv1CandidateError as exc:
        raise SystemExit(f"FATAL: {exc}") from exc
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
