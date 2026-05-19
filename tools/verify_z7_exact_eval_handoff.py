#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""No-spend exact-eval handoff doctor for Z7 recurrent-vs-static packets.

This tool classifies whether the latest Z7 score-aware packet can enter paired
CPU/CUDA exact eval. It does not open lane claims, dispatch provider work, or
make score/promotion claims. Its job is to preserve the same-byte
recurrent-vs-static signal while failing closed on the current one-pair smoke.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import stat
import struct
import zipfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STATS_JSON = (
    "experiments/results/z7_gru_score_aware_smoke_codex_20260518T1255Z/"
    "z7_gru_prebuild_full_main_export_stats.json"
)
DEFAULT_OUTPUT_DIR = ".omx/state/z7_exact_eval_handoff"
LANE_ID = "lane_per_substrate_symposium_z7_lstm_predictive_coding_20260517"
SUBSTRATE_ID = "time_traveler_l5_z7_lstm_predictive_coding"
PAIR_GROUP_ID = "z7_temporal_coherence_vs_static_capacity_same_bytes"
KNOWN_SUBSTRATE_PAIR_GROUP_IDS = {
    "time_traveler_l5_z7_lstm_predictive_coding": PAIR_GROUP_ID,
    "time_traveler_l5_z7_mamba2": (
        "z7_mamba2_temporal_coherence_vs_static_capacity_same_bytes"
    ),
}
REQUIRED_PAIR_COUNT = 600
HEX_DIGITS = set("0123456789abcdefABCDEF")
Z7_MAMBA2_SUBSTRATE_ID = "time_traveler_l5_z7_mamba2"
Z7MCM2_MAGIC = b"Z7M2"
Z7MCM2_SCHEMA_VERSION = 1
Z7MCM2_HEADER_FMT = "<4sBHHHHBBBBIIIIIII"
Z7MCM2_HEADER_SIZE = struct.calcsize(Z7MCM2_HEADER_FMT)
CAMERA_HW = (874, 1164)
RUNTIME_GEOMETRY_POSITIVE_CONTROL_SCHEMA = (
    "z7_mamba2_runtime_geometry_positive_control_v1"
)


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_now_compact() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _safe_rel(repo_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def _resolve_repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _short_sha(value: str, *, label: str) -> str:
    raw = str(value or "").strip()
    if len(raw) >= 12 and all(char in HEX_DIGITS for char in raw[:12]):
        return raw[:12].lower()
    return f"<{label}_archive_sha256_prefix>"


def _safe_token(value: str, *, fallback: str) -> str:
    token = "".join(
        char if char.isalnum() or char in ("_", "-") else "_"
        for char in str(value or "").strip()
    ).strip("_-")
    return token or fallback


def _stats_identity(stats: Mapping[str, Any]) -> dict[str, str]:
    lane_id = str(stats.get("lane_id") or LANE_ID).strip() or LANE_ID
    substrate_id = str(stats.get("substrate_id") or SUBSTRATE_ID).strip() or SUBSTRATE_ID
    explicit_pair_group = (
        stats.get("pair_group_id")
        or stats.get("exact_eval_pair_group_id")
        or stats.get("same_bytes_pair_group_id")
    )
    pair_group_id = str(
        explicit_pair_group
        or KNOWN_SUBSTRATE_PAIR_GROUP_IDS.get(substrate_id)
        or PAIR_GROUP_ID
    ).strip() or PAIR_GROUP_ID
    return {
        "lane_id": lane_id,
        "substrate_id": substrate_id,
        "pair_group_id": pair_group_id,
    }


def _zip_member_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    try:
        with zipfile.ZipFile(path) as zf:
            for info in [row for row in zf.infolist() if not row.is_dir()]:
                blob = zf.read(info.filename)
                rows.append(
                    {
                        "name": info.filename,
                        "file_size": info.file_size,
                        "compress_size": info.compress_size,
                        "crc": f"{info.CRC:08x}",
                        "sha256": hashlib.sha256(blob).hexdigest(),
                        "compression_method": info.compress_type,
                    }
                )
    except (OSError, zipfile.BadZipFile) as exc:
        blockers.append(f"zip_unreadable:{type(exc).__name__}")
    return rows, blockers


def _parse_z7mcm2_header_meta(archive_bytes: bytes) -> dict[str, int]:
    """Parse geometry fields from a raw Z7MCM2 ``0.bin`` member."""
    if len(archive_bytes) < Z7MCM2_HEADER_SIZE:
        raise ValueError(
            f"z7mcm2 archive too short: {len(archive_bytes)} < {Z7MCM2_HEADER_SIZE}"
        )
    header = struct.unpack(Z7MCM2_HEADER_FMT, archive_bytes[:Z7MCM2_HEADER_SIZE])
    magic = header[0]
    version = int(header[1])
    latent_dim = int(header[2])
    ego_dim = int(header[3])
    num_pairs = int(header[4])
    (
        encoder_len,
        decoder_len,
        predictor_len,
        latent_init_len,
        residuals_len,
        ego_len,
        meta_len,
    ) = (int(value) for value in header[10:17])
    if magic != Z7MCM2_MAGIC:
        raise ValueError(f"z7mcm2 bad magic: {magic!r}")
    if version != Z7MCM2_SCHEMA_VERSION:
        raise ValueError(f"z7mcm2 unsupported schema version: {version}")
    if latent_init_len != latent_dim:
        raise ValueError("z7mcm2 latent_init_len != latent_dim")
    if residuals_len != num_pairs * latent_dim:
        raise ValueError("z7mcm2 residuals_len != num_pairs*latent_dim")
    if ego_len != num_pairs * ego_dim:
        raise ValueError("z7mcm2 ego_motion_len != num_pairs*ego_motion_dim")
    meta_start = (
        Z7MCM2_HEADER_SIZE
        + encoder_len
        + decoder_len
        + predictor_len
        + latent_init_len
        + residuals_len
        + ego_len
    )
    meta_end = meta_start + meta_len
    if meta_end != len(archive_bytes):
        raise ValueError(
            f"z7mcm2 archive size {len(archive_bytes)} != expected {meta_end}"
        )
    meta = json.loads(archive_bytes[meta_start:meta_end].decode("utf-8"))
    return {
        "num_pairs": num_pairs,
        "output_height": int(meta["output_height"]),
        "output_width": int(meta["output_width"]),
    }


def _read_single_z7mcm2_header_from_zip(path: Path) -> dict[str, int]:
    """Read and parse the single ``0.bin`` Z7MCM2 member from an archive zip."""
    with zipfile.ZipFile(path) as zf:
        members = [info for info in zf.infolist() if not info.is_dir()]
        if len(members) != 1:
            raise ValueError(f"expected one zip member, got {len(members)}")
        if members[0].filename != "0.bin":
            raise ValueError(f"expected 0.bin zip member, got {members[0].filename!r}")
        return _parse_z7mcm2_header_meta(zf.read(members[0].filename))


def _archive_status(
    *,
    repo_root: Path,
    mode: str,
    path_value: str,
    expected_bytes: Any,
    expected_sha256: Any,
    parse_z7mcm2_header: bool = False,
) -> dict[str, Any]:
    path = _resolve_repo_path(repo_root, path_value)
    blockers: list[str] = []
    exists = path.is_file()
    zip_rows: list[dict[str, Any]] = []
    actual_bytes: int | None = None
    actual_sha: str | None = None
    z7mcm2_header: dict[str, int] | None = None
    if not exists:
        blockers.append(f"z7_exact_handoff_{mode}_archive_zip_missing")
    else:
        actual_bytes = path.stat().st_size
        actual_sha = _sha256(path)
        zip_rows, zip_blockers = _zip_member_rows(path)
        blockers.extend(f"z7_exact_handoff_{mode}_{b}" for b in zip_blockers)
        if len(zip_rows) != 1:
            blockers.append(f"z7_exact_handoff_{mode}_archive_zip_not_single_member")
        elif zip_rows[0].get("name") != "0.bin":
            blockers.append(f"z7_exact_handoff_{mode}_archive_zip_member_not_0_bin")
        if isinstance(expected_bytes, int) and actual_bytes != expected_bytes:
            blockers.append(f"z7_exact_handoff_{mode}_archive_bytes_mismatch")
        if expected_sha256 and str(expected_sha256).lower() != str(actual_sha).lower():
            blockers.append(f"z7_exact_handoff_{mode}_archive_sha256_mismatch")
        if parse_z7mcm2_header:
            try:
                z7mcm2_header = _read_single_z7mcm2_header_from_zip(path)
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                blockers.append(
                    "z7_exact_handoff_"
                    f"{mode}_z7mcm2_header_meta_invalid:{type(exc).__name__}"
                )
    return {
        "mode": mode,
        "zip_path": path_value,
        "zip_exists": exists,
        "zip_bytes": actual_bytes,
        "zip_sha256": actual_sha or expected_sha256,
        "expected_zip_bytes": expected_bytes,
        "expected_zip_sha256": expected_sha256,
        "zip_member_rows": zip_rows,
        "z7mcm2_header": z7mcm2_header,
        "blockers": blockers,
    }


def _runtime_custody(repo_root: Path, runtime_dir_value: str) -> dict[str, Any]:
    path = _resolve_repo_path(repo_root, runtime_dir_value)
    blockers: list[str] = []
    records: list[dict[str, Any]] = []
    if not path.is_dir():
        return {
            "path": runtime_dir_value,
            "exists": False,
            "file_count": 0,
            "total_bytes": 0,
            "aggregate_sha256": None,
            "inflate_sh_executable": False,
            "blockers": ["z7_exact_handoff_submission_runtime_dir_missing"],
        }
    inflate_sh = path / "inflate.sh"
    if not inflate_sh.is_file():
        blockers.append("z7_exact_handoff_submission_runtime_inflate_sh_missing")
    elif not (inflate_sh.stat().st_mode & stat.S_IXUSR):
        blockers.append("z7_exact_handoff_submission_runtime_inflate_sh_not_executable")
    for cur in sorted(p for p in path.rglob("*") if p.is_file()):
        if "__pycache__" in cur.parts:
            continue
        rel = cur.relative_to(path).as_posix()
        records.append(
            {
                "path": rel,
                "bytes": cur.stat().st_size,
                "sha256": _sha256(cur),
            }
        )
    agg = hashlib.sha256()
    total_bytes = 0
    for record in records:
        total_bytes += int(record["bytes"])
        agg.update(record["path"].encode("utf-8"))
        agg.update(b"\0")
        agg.update(str(record["sha256"]).encode("ascii"))
        agg.update(b"\0")
    return {
        "path": runtime_dir_value,
        "exists": True,
        "file_count": len(records),
        "total_bytes": total_bytes,
        "aggregate_sha256": agg.hexdigest(),
        "inflate_sh_executable": inflate_sh.is_file()
        and bool(inflate_sh.stat().st_mode & stat.S_IXUSR),
        "records": records,
        "blockers": blockers,
    }


def _is_z7_mamba2_stats(stats: Mapping[str, Any], substrate_id: str) -> bool:
    return (
        substrate_id == Z7_MAMBA2_SUBSTRATE_ID
        or stats.get("name") == "z7_mamba2_full_main_export_stats"
    )


def _int_pair(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    if not all(isinstance(item, int) for item in value):
        return None
    return [int(value[0]), int(value[1])]


def _is_hex_sha256(value: Any) -> bool:
    raw = str(value or "")
    return len(raw) == 64 and all(char in HEX_DIGITS for char in raw)


def _runtime_geometry_positive_control_blockers(
    *,
    stats: Mapping[str, Any],
    static: Mapping[str, Any],
    config: Mapping[str, Any],
    inflate_verify: Any,
    recurrent: Mapping[str, Any],
    static_row: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate Z7-Mamba2 geometry positive-control against real archive headers."""
    blockers: list[str] = []
    block = stats.get("runtime_geometry_positive_control")
    if not isinstance(block, Mapping):
        return None, ["z7_exact_handoff_runtime_geometry_positive_control_missing"]

    normalized_block = dict(block)
    if block.get("schema") != RUNTIME_GEOMETRY_POSITIVE_CONTROL_SCHEMA:
        blockers.append("z7_exact_handoff_runtime_geometry_positive_control_bad_schema")

    recurrent_header = recurrent.get("z7mcm2_header")
    static_header = static_row.get("z7mcm2_header")
    if not isinstance(recurrent_header, Mapping):
        blockers.append("z7_exact_handoff_recurrent_z7mcm2_header_missing")
    if not isinstance(static_header, Mapping):
        blockers.append("z7_exact_handoff_static_control_z7mcm2_header_missing")
    if (
        isinstance(recurrent_header, Mapping)
        and isinstance(static_header, Mapping)
        and dict(recurrent_header) != dict(static_header)
    ):
        blockers.append(
            "z7_exact_handoff_recurrent_static_z7mcm2_header_mismatch"
        )

    archive_header = block.get("archive_header")
    if not isinstance(archive_header, Mapping):
        blockers.append(
            "z7_exact_handoff_runtime_geometry_archive_header_missing"
        )
        archive_header = {}
    if isinstance(recurrent_header, Mapping):
        for key in ("num_pairs", "output_height", "output_width"):
            if archive_header.get(key) != recurrent_header.get(key):
                blockers.append(
                    "z7_exact_handoff_runtime_geometry_"
                    f"archive_header_{key}_mismatch"
                )

    num_pairs = block.get("num_pairs")
    if not isinstance(num_pairs, int) or num_pairs <= 0:
        blockers.append("z7_exact_handoff_runtime_geometry_num_pairs_invalid")
        num_pairs = recurrent_header.get("num_pairs") if isinstance(
            recurrent_header, Mapping
        ) else config.get("num_pairs")
    if isinstance(num_pairs, int):
        if config.get("num_pairs") != num_pairs:
            blockers.append(
                "z7_exact_handoff_runtime_geometry_config_num_pairs_mismatch"
            )
        if isinstance(recurrent_header, Mapping) and recurrent_header.get(
            "num_pairs"
        ) != num_pairs:
            blockers.append(
                "z7_exact_handoff_runtime_geometry_header_num_pairs_mismatch"
            )

    render_hw = _int_pair(block.get("render_hw"))
    if render_hw is None:
        blockers.append("z7_exact_handoff_runtime_geometry_render_hw_invalid")
    elif isinstance(recurrent_header, Mapping) and render_hw != [
        recurrent_header.get("output_height"),
        recurrent_header.get("output_width"),
    ]:
        blockers.append("z7_exact_handoff_runtime_geometry_render_hw_mismatch")

    camera_hw = _int_pair(block.get("camera_hw"))
    if camera_hw is None:
        blockers.append("z7_exact_handoff_runtime_geometry_camera_hw_invalid")
        camera_hw = [CAMERA_HW[0], CAMERA_HW[1]]
    elif camera_hw != [CAMERA_HW[0], CAMERA_HW[1]]:
        blockers.append("z7_exact_handoff_runtime_geometry_camera_hw_mismatch")

    if isinstance(num_pairs, int) and camera_hw is not None:
        expected_frames = int(num_pairs) * 2
        expected_raw_bytes = (
            expected_frames * 3 * int(camera_hw[0]) * int(camera_hw[1])
        )
        if block.get("expected_frames_written") != expected_frames:
            blockers.append(
                "z7_exact_handoff_runtime_geometry_expected_frames_mismatch"
            )
        if block.get("expected_raw_bytes") != expected_raw_bytes:
            blockers.append(
                "z7_exact_handoff_runtime_geometry_expected_raw_bytes_mismatch"
            )
        if isinstance(inflate_verify, Mapping) and not inflate_verify.get(
            "verify_failed"
        ):
            if inflate_verify.get("frames_written") != expected_frames:
                blockers.append(
                    "z7_exact_handoff_runtime_geometry_recurrent_frames_mismatch"
                )
            if inflate_verify.get("raw_bytes") != expected_raw_bytes:
                blockers.append(
                    "z7_exact_handoff_runtime_geometry_recurrent_raw_bytes_mismatch"
                )
        if static.get("inflate_verify_frames_written") != expected_frames:
            blockers.append(
                "z7_exact_handoff_runtime_geometry_static_frames_mismatch"
            )
        if static.get("inflate_verify_raw_bytes") != expected_raw_bytes:
            blockers.append(
                "z7_exact_handoff_runtime_geometry_static_raw_bytes_mismatch"
            )

        sample_pair_indices = block.get("sample_pair_indices")
        if not isinstance(sample_pair_indices, list) or not sample_pair_indices or any(
            not isinstance(idx, int) or idx < 0 or idx >= int(num_pairs)
            for idx in sample_pair_indices
        ):
            blockers.append(
                "z7_exact_handoff_runtime_geometry_sample_pair_indices_invalid"
            )

    recurrent_sample_sha = block.get("recurrent_sampled_raw_sha256")
    static_sample_sha = block.get("static_sampled_raw_sha256")
    if not _is_hex_sha256(recurrent_sample_sha):
        blockers.append(
            "z7_exact_handoff_runtime_geometry_recurrent_sample_sha_invalid"
        )
    if not _is_hex_sha256(static_sample_sha):
        blockers.append(
            "z7_exact_handoff_runtime_geometry_static_sample_sha_invalid"
        )
    if block.get("recurrent_static_sample_changed") is not True:
        blockers.append(
            "z7_exact_handoff_runtime_geometry_recurrent_static_sample_not_changed"
        )
    if (
        _is_hex_sha256(recurrent_sample_sha)
        and _is_hex_sha256(static_sample_sha)
        and str(recurrent_sample_sha).lower() == str(static_sample_sha).lower()
    ):
        blockers.append(
            "z7_exact_handoff_runtime_geometry_recurrent_static_sample_sha_equal"
        )

    return normalized_block, blockers


def _paired_dispatch_command(
    *,
    archive_zip: str,
    archive_sha: str,
    mode: str,
    submission_dir: str,
    lane_id: str,
    substrate_id: str,
    pair_group_id: str,
    execute: bool,
) -> str:
    short = _short_sha(archive_sha, label=mode)
    safe_mode = mode.replace("_", "-")
    lane_base = f"{_safe_token(lane_id, fallback='lane_z7')}_exact_eval_{safe_mode}"
    label_prefix = _safe_token(substrate_id, fallback="z7")
    cmd = [
        ".venv/bin/python",
        "tools/dispatch_modal_paired_auth_eval.py",
        "--archive",
        archive_zip,
        "--expected-archive-sha256",
        archive_sha,
        "--submission-dir",
        submission_dir,
        "--inflate-sh",
        "inflate.sh",
        "--label",
        f"{label_prefix}_{safe_mode}",
        "--run-id",
        f"{label_prefix}_{safe_mode}_{short}",
        "--pair-group-id",
        pair_group_id,
        "--lane-id-base",
        lane_base,
        "--output-root",
        "experiments/results",
        "--modal-bin",
        ".venv/bin/modal",
        "--gpu",
        "T4",
        "--claim-agent",
        "codex:z7_exact_eval_handoff",
        "--claim-notes",
        (
            "Z7 recurrent-vs-static paired exact-eval handoff; "
            f"mode={mode}; substrate_id={substrate_id}; "
            f"source_lane_id={lane_id}; archive_sha={archive_sha}; "
            f"pair_group_id={pair_group_id}; score_claim=false"
        ),
        "--expected-runtime-tree-sha256",
        "auto",
        "--skip-axis-if-promotable-anchor-exists",
    ]
    if execute:
        cmd.append("--execute")
    return shlex.join(cmd)


def _command_templates(
    *,
    lane_id: str,
    substrate_id: str,
    pair_group_id: str,
    execute: bool,
) -> dict[str, str]:
    return {
        "recurrent_paired_contest_cpu_cuda": _paired_dispatch_command(
            archive_zip="<ratified_z7_run_dir>/archive.zip",
            archive_sha="<recurrent_archive_zip_sha256>",
            mode="recurrent",
            submission_dir="<ratified_z7_run_dir>/submission_runtime",
            lane_id=lane_id,
            substrate_id=substrate_id,
            pair_group_id=pair_group_id,
            execute=execute,
        ),
        "static_control_paired_contest_cpu_cuda": _paired_dispatch_command(
            archive_zip="<ratified_z7_run_dir>/static_capacity_control/archive.zip",
            archive_sha="<static_control_archive_zip_sha256>",
            mode="static_control",
            submission_dir="<ratified_z7_run_dir>/submission_runtime",
            lane_id=lane_id,
            substrate_id=substrate_id,
            pair_group_id=pair_group_id,
            execute=execute,
        ),
    }


def build_packet(
    *,
    repo_root: Path = REPO_ROOT,
    stats_json: Path | None = None,
    required_pair_count: int = REQUIRED_PAIR_COUNT,
) -> dict[str, Any]:
    stats_path = stats_json or repo_root / DEFAULT_STATS_JSON
    if not stats_path.is_absolute():
        stats_path = repo_root / stats_path
    blockers: list[str] = []
    try:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        stats = {}
        blockers.append(f"z7_exact_handoff_stats_json_unreadable:{type(exc).__name__}")
    if not isinstance(stats, Mapping):
        stats = {}
        blockers.append("z7_exact_handoff_stats_json_not_object")
    identity = _stats_identity(stats)
    lane_id = identity["lane_id"]
    substrate_id = identity["substrate_id"]
    pair_group_id = identity["pair_group_id"]
    requires_z7mcm2_geometry = _is_z7_mamba2_stats(stats, substrate_id)

    static = stats.get("static_capacity_control")
    if not isinstance(static, Mapping):
        static = {}
        blockers.append("z7_exact_handoff_static_capacity_control_missing")

    config = stats.get("config")
    if not isinstance(config, Mapping):
        config = {}
        blockers.append("z7_exact_handoff_config_missing")

    for field in (
        "score_claim",
        "promotion_eligible",
        "ready_for_paid_dispatch",
        "rank_or_kill_eligible",
    ):
        if stats.get(field) is not False:
            blockers.append(f"z7_exact_handoff_stats_{field}_not_false")
        if static and static.get(field) not in (None, False):
            blockers.append(f"z7_exact_handoff_static_control_{field}_not_false")
    if stats.get("ready_for_exact_eval_dispatch") is not False:
        blockers.append(
            "z7_exact_handoff_stats_ready_for_exact_eval_dispatch_not_false"
        )
    if static and static.get("ready_for_exact_eval_dispatch") not in (None, False):
        blockers.append(
            "z7_exact_handoff_static_control_ready_for_exact_eval_dispatch_not_false"
        )

    if stats.get("loss_mode") != "score_aware":
        blockers.append("z7_exact_handoff_loss_mode_not_score_aware")
    if stats.get("score_aware_scorer_loss_used") is not True:
        blockers.append("z7_exact_handoff_score_aware_scorer_loss_not_used")
    inflate_verify = stats.get("inflate_verify")
    if isinstance(inflate_verify, Mapping) and inflate_verify.get("verify_failed"):
        blockers.append("z7_exact_handoff_inflate_verify_failed")
    elif isinstance(inflate_verify, Mapping):
        verify_device = str(inflate_verify.get("device") or "").strip().lower()
        if verify_device not in {"cpu", "cuda"}:
            blockers.append("z7_exact_handoff_inflate_verify_device_not_cpu_or_cuda")
        device_contract = stats.get("device_runtime_contract")
        if (
            isinstance(device_contract, Mapping)
            and device_contract.get("device_type") == "mps"
            and verify_device != "cpu"
        ):
            blockers.append("z7_exact_handoff_mps_training_must_cpu_inflate_verify")
    else:
        blockers.append("z7_exact_handoff_inflate_verify_missing")

    recurrent = _archive_status(
        repo_root=repo_root,
        mode="recurrent",
        path_value=str(stats.get("archive_zip_path") or ""),
        expected_bytes=stats.get("archive_zip_bytes"),
        expected_sha256=stats.get("archive_zip_sha256"),
        parse_z7mcm2_header=requires_z7mcm2_geometry,
    )
    static_row = _archive_status(
        repo_root=repo_root,
        mode="static_control",
        path_value=str(static.get("archive_zip_path") or ""),
        expected_bytes=static.get("archive_zip_bytes"),
        expected_sha256=static.get("archive_zip_sha256"),
        parse_z7mcm2_header=requires_z7mcm2_geometry,
    )
    for row in (recurrent, static_row):
        blockers.extend(row.get("blockers") or [])

    runtime_geometry_positive_control: dict[str, Any] | None = None
    if requires_z7mcm2_geometry:
        (
            runtime_geometry_positive_control,
            runtime_geometry_blockers,
        ) = _runtime_geometry_positive_control_blockers(
            stats=stats,
            static=static,
            config=config,
            inflate_verify=inflate_verify,
            recurrent=recurrent,
            static_row=static_row,
        )
        blockers.extend(runtime_geometry_blockers)

    if recurrent.get("zip_bytes") != static_row.get("zip_bytes"):
        blockers.append("z7_exact_handoff_recurrent_static_archive_byte_count_mismatch")
    if static.get("same_archive_zip_bytes_as_recurrent") is not True:
        blockers.append("z7_exact_handoff_static_control_same_archive_bytes_flag_not_true")
    runtime_output_changed = static.get("runtime_output_changed_vs_recurrent")
    byte_diff = static.get("runtime_output_byte_differences_vs_recurrent")
    missing_static_output_evidence = (
        runtime_output_changed is None and byte_diff is None
    )
    if missing_static_output_evidence:
        blockers.append(
            "z7_exact_handoff_static_control_inflate_output_evidence_missing"
        )
    elif runtime_output_changed is not True:
        blockers.append("z7_exact_handoff_static_control_runtime_output_not_changed")
    if isinstance(byte_diff, int):
        if byte_diff <= 0:
            blockers.append(
                "z7_exact_handoff_static_control_byte_differences_not_positive"
            )
    elif not missing_static_output_evidence:
        blockers.append("z7_exact_handoff_static_control_byte_differences_not_positive")

    num_pairs = config.get("num_pairs")
    if not isinstance(num_pairs, int):
        blockers.append("z7_exact_handoff_num_pairs_missing_or_not_int")
    elif num_pairs != int(required_pair_count):
        blockers.append(f"z7_exact_handoff_current_packet_not_{required_pair_count}_pairs")

    runtime = _runtime_custody(
        repo_root,
        str(stats.get("submission_runtime_dir") or ""),
    )
    blockers.extend(runtime.get("blockers") or [])

    submission_dir = str(stats.get("submission_runtime_dir") or "")
    plan_command_allowed_blockers = {
        f"z7_exact_handoff_current_packet_not_{required_pair_count}_pairs",
    }
    valid_archive_pair = all(
        blocker in plan_command_allowed_blockers for blocker in blockers
    )
    plan_commands: dict[str, str] = {}
    execute_commands: dict[str, str] = {}
    if valid_archive_pair and submission_dir:
        plan_commands = {
            "recurrent_paired_contest_cpu_cuda": _paired_dispatch_command(
                archive_zip=str(stats.get("archive_zip_path") or ""),
                archive_sha=str(stats.get("archive_zip_sha256") or ""),
                mode="recurrent",
                submission_dir=submission_dir,
                lane_id=lane_id,
                substrate_id=substrate_id,
                pair_group_id=pair_group_id,
                execute=False,
            ),
            "static_control_paired_contest_cpu_cuda": _paired_dispatch_command(
                archive_zip=str(static.get("archive_zip_path") or ""),
                archive_sha=str(static.get("archive_zip_sha256") or ""),
                mode="static_control",
                submission_dir=submission_dir,
                lane_id=lane_id,
                substrate_id=substrate_id,
                pair_group_id=pair_group_id,
                execute=False,
            ),
        }
    ready = not blockers
    if ready:
        execute_commands = {
            key: command + " --execute" for key, command in plan_commands.items()
        }

    return {
        "schema": "z7_exact_eval_handoff_v1",
        "generated_utc": _utc_now(),
        "lane_id": lane_id,
        "substrate_id": substrate_id,
        "stats_json": _safe_rel(repo_root, stats_path),
        "required_pair_count": int(required_pair_count),
        "current_pair_count": num_pairs if isinstance(num_pairs, int) else None,
        "ready_for_exact_eval_handoff": ready,
        "provider_dispatch_attempted": False,
        "lane_claim_opened": False,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_paid_dispatch": False,
        "rank_or_kill_eligible": False,
        "evidence_grade": "z7_exact_eval_handoff_no_spend",
        "pair_group_id": pair_group_id,
        "axis_plan": [
            {
                "axis": "[contest-CUDA]",
                "mode": "recurrent",
                "inflate_device_policy": "auto",
                "evaluate_device": "cuda",
                "required_hardware": "linux_x86_64_t4",
                "authority_precondition": "modal_paired_auth_eval_cuda_harvested",
            },
            {
                "axis": "[contest-CPU]",
                "mode": "recurrent",
                "inflate_device_policy": "auto",
                "evaluate_device": "cpu",
                "required_hardware": "linux_x86_64_cpu",
                "authority_precondition": "modal_paired_auth_eval_cpu_harvested",
            },
            {
                "axis": "[contest-CUDA]",
                "mode": "static_control",
                "inflate_device_policy": "auto",
                "evaluate_device": "cuda",
                "required_hardware": "linux_x86_64_t4",
                "authority_precondition": "modal_paired_auth_eval_cuda_harvested",
            },
            {
                "axis": "[contest-CPU]",
                "mode": "static_control",
                "inflate_device_policy": "auto",
                "evaluate_device": "cpu",
                "required_hardware": "linux_x86_64_cpu",
                "authority_precondition": "modal_paired_auth_eval_cpu_harvested",
            },
        ],
        "source_archive_rows": [recurrent, static_row],
        "same_archive_zip_bytes": recurrent.get("zip_bytes") == static_row.get("zip_bytes"),
        "runtime_output_changed_vs_recurrent": static.get(
            "runtime_output_changed_vs_recurrent"
        ),
        "runtime_output_byte_differences_vs_recurrent": byte_diff,
        "runtime_geometry_positive_control": runtime_geometry_positive_control,
        "runtime_custody": runtime,
        "modal_plan_commands_for_current_packet": plan_commands,
        "modal_execute_commands_after_ratified_full_packet": execute_commands,
        "modal_command_templates_after_ratified_full_packet": _command_templates(
            lane_id=lane_id,
            substrate_id=substrate_id,
            pair_group_id=pair_group_id,
            execute=True
        ),
        "result_review_blockers": [] if ready else blockers,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats-json", type=Path, default=Path(DEFAULT_STATS_JSON))
    parser.add_argument("--required-pair-count", type=int, default=REQUIRED_PAIR_COUNT)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--write-artifact", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args(argv)

    payload = build_packet(
        repo_root=REPO_ROOT,
        stats_json=args.stats_json,
        required_pair_count=args.required_pair_count,
    )
    if args.write_artifact:
        out_dir = args.output_dir
        if not out_dir.is_absolute():
            out_dir = REPO_ROOT / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"z7_exact_eval_handoff_{_utc_now_compact()}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        payload["artifact_path"] = _safe_rel(REPO_ROOT, path)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.json:
        print(text)
    else:
        status = "READY" if payload["ready_for_exact_eval_handoff"] else "BLOCKED"
        print(f"Z7 exact-eval handoff: {status}")
        for blocker in payload.get("result_review_blockers", []):
            print(f"- {blocker}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
