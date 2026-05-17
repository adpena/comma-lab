# SPDX-License-Identifier: MIT
"""Materialize byte-closed TT5L side-info variant packets.

This module intentionally stops at archive packet custody. It does not claim a
score and does not mark packets ready for exact eval dispatch; lane claims and
paired CPU/CUDA exact-eval evidence are separate operator actions.
"""

from __future__ import annotations

import hashlib
import json
import struct
import zipfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import brotli  # type: ignore[import-not-found]
import numpy as np

from tac.optimization.l5_v2_measurement_schedule import (
    L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS,
    L5V2_TT5L_SIDEINFO_VARIANT_PACKET_ARTIFACT_PATH,
    L5V2_TT5L_SIDEINFO_VARIANT_PACKET_OUTPUT_ROOT,
    L5V2_TT5L_SIDEINFO_VARIANT_PACKET_REPORT_PATH,
    L5V2_TT5L_SIDEINFO_VARIANT_PACKET_SCHEMA,
    L5V2_TT5L_SIDEINFO_VARIANT_PACKET_SOURCE_ARCHIVE_PATH,
    L5V2_TT5L_SIDEINFO_VARIANT_PACKET_SUBMISSION_DIR,
    L5V2_TT5L_SIDEINFO_VARIANT_PACKET_TOOL_PATH,
)
from tac.optimizer.exact_readiness import runtime_dependency_manifest
from tac.substrates.time_traveler_l5_autonomy.archive import (
    TT5L_HEADER_FMT,
    TT5L_HEADER_SIZE,
    TT5L_MAGIC,
    TT5L_SCHEMA_VERSION,
    TT5L_SIDE_INFO_SECTION_WIDTHS,
    TimeTravelerArchive,
    parse_archive,
    parse_tt5l_archive_bytes,
    side_info_liveness_stats,
)

TT5L_SIDEINFO_VARIANT_PACKET_SCHEMA = L5V2_TT5L_SIDEINFO_VARIANT_PACKET_SCHEMA
TT5L_SIDEINFO_VARIANT_PACKET_TOOL_PATH = L5V2_TT5L_SIDEINFO_VARIANT_PACKET_TOOL_PATH
TT5L_CONTEST_NUM_PAIRS = 600
DEFAULT_TT5L_SOURCE_ARCHIVE_PATH = (
    L5V2_TT5L_SIDEINFO_VARIANT_PACKET_SOURCE_ARCHIVE_PATH
)
DEFAULT_TT5L_SIDEINFO_VARIANT_OUTPUT_ROOT = (
    L5V2_TT5L_SIDEINFO_VARIANT_PACKET_OUTPUT_ROOT
)
DEFAULT_TT5L_SIDEINFO_VARIANT_JSON_PATH = (
    L5V2_TT5L_SIDEINFO_VARIANT_PACKET_ARTIFACT_PATH
)
DEFAULT_TT5L_SIDEINFO_VARIANT_REPORT_PATH = (
    L5V2_TT5L_SIDEINFO_VARIANT_PACKET_REPORT_PATH
)
DEFAULT_TT5L_SIDEINFO_VARIANT_SUBMISSION_DIR = (
    L5V2_TT5L_SIDEINFO_VARIANT_PACKET_SUBMISSION_DIR
)
TT5L_SIDEINFO_VARIANT_SEED = 20260517
TT5L_VARIANTS_EXPECTED_TO_TRANSFORM_SOURCE = frozenset(
    {"random_lsb", "shuffled", "ablated"}
)


def _sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def _select_archive_member(zf: zipfile.ZipFile) -> zipfile.ZipInfo:
    members = [
        info
        for info in zf.infolist()
        if not info.is_dir() and "__MACOSX/" not in info.filename
    ]
    if not members:
        raise ValueError("TT5L archive.zip has no file members")
    by_name = {info.filename: info for info in members}
    if "0.bin" in by_name:
        return by_name["0.bin"]
    if "x" in by_name:
        return by_name["x"]
    if len(members) == 1:
        return members[0]
    names = ", ".join(info.filename for info in members)
    raise ValueError(f"TT5L archive.zip member ambiguous: {names}")


def read_tt5l_archive_zip(path: str | Path) -> tuple[zipfile.ZipInfo, bytes]:
    """Return the TT5L member metadata and bytes from an archive.zip."""

    archive_path = Path(path)
    with zipfile.ZipFile(archive_path, "r") as zf:
        info = _select_archive_member(zf)
        return info, zf.read(info.filename)


def _write_archive_zip(
    path: Path,
    *,
    member_name: str,
    member_bytes: bytes,
    compress_type: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compression = (
        zipfile.ZIP_DEFLATED
        if compress_type == zipfile.ZIP_DEFLATED
        else zipfile.ZIP_STORED
    )
    info = zipfile.ZipInfo(member_name, date_time=(2026, 1, 1, 0, 0, 0))
    info.compress_type = compression
    info.external_attr = 0o644 << 16
    kwargs: dict[str, Any] = {"compression": compression}
    if compression == zipfile.ZIP_DEFLATED:
        kwargs["compresslevel"] = 9
    with zipfile.ZipFile(path, "w", **kwargs) as zf:
        zf.writestr(info, member_bytes)


def _archive_member_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            data = zf.read(info.filename)
            rows.append(
                {
                    "name": info.filename,
                    "file_size": int(info.file_size),
                    "compress_size": int(info.compress_size),
                    "compress_type": int(info.compress_type),
                    "crc32": f"{int(info.CRC):08x}",
                    "sha256": _sha256_bytes(data),
                }
            )
    return rows


def _write_variant_archive_manifest(
    *,
    archive_path: Path,
    repo_root: Path,
    variant: str,
) -> tuple[Path, dict[str, Any]]:
    archive_sha = _sha256_file(archive_path)
    payload = {
        "schema": "tt5l_sideinfo_variant_archive_manifest_v1",
        "variant": variant,
        "archive_path": _repo_relative(archive_path, repo_root),
        "archive_sha256": archive_sha,
        "candidate_archive_sha256": archive_sha,
        "archive_bytes": archive_path.stat().st_size,
        "candidate_archive_bytes": archive_path.stat().st_size,
        "members": _archive_member_rows(archive_path),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "dispatch_attempted": False,
    }
    manifest_path = archive_path.parent / "archive_manifest.json"
    _write_json(manifest_path, payload)
    return manifest_path, payload


def _write_runtime_report(
    *,
    submission_dir: str | Path | None,
    repo_root: Path,
    source_archive_path: Path,
    source_archive_sha256: str,
    source_archive_bytes: int,
    source_archive_member: str,
    variant_rows: list[Mapping[str, Any]],
) -> str:
    if submission_dir is None:
        return ""
    runtime_dir = Path(submission_dir)
    if not runtime_dir.is_absolute():
        runtime_dir = repo_root / runtime_dir
    if not runtime_dir.is_dir():
        return ""
    report_path = runtime_dir / "report.txt"
    lines = [
        "L5 v2 TT5L side-info effect-curve runtime custody",
        "",
        "score_claim: false",
        "promotion_eligible: false",
        "ready_for_exact_eval_dispatch: false",
        "dispatch_attempted: false",
        "classification: runtime custody report only; paired contest CPU/CUDA exact-eval cells required before score claims",
        "",
        f"source_archive_path: {_repo_relative(source_archive_path, repo_root)}",
        f"source_archive_sha256: {source_archive_sha256}",
        f"source_archive_bytes: {source_archive_bytes}",
        f"source_archive_member: {source_archive_member}",
        "",
        "variant_archives:",
    ]
    for row in variant_rows:
        lines.append(
            "- "
            f"variant={row.get('variant')} "
            f"archive_path={row.get('archive_path')} "
            f"archive_sha256={row.get('archive_sha256')} "
            f"archive_bytes={row.get('archive_bytes')} "
            f"archive_manifest_path={row.get('archive_manifest_path')}"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return _repo_relative(report_path, repo_root)


def _replace_sideinfo_blob(
    source_member_bytes: bytes,
    parsed: TimeTravelerArchive,
    side_info: np.ndarray,
) -> bytes:
    if side_info.shape != (parsed.num_pairs, parsed.per_pair_bytes):
        raise ValueError(
            "side-info variant shape mismatch: "
            f"{side_info.shape} != {(parsed.num_pairs, parsed.per_pair_bytes)}"
        )
    sections = parse_tt5l_archive_bytes(source_member_bytes)
    world_start, world_len = sections["world_model_blob"]
    _side_start, _side_len = sections["per_pair_side_info_blob"]
    ac_start, ac_len = sections["ac_state_blob"]
    meta_start, meta_len = sections["meta_blob"]
    world_blob = source_member_bytes[world_start : world_start + world_len]
    ac_blob = source_member_bytes[ac_start : ac_start + ac_len]
    meta_blob = source_member_bytes[meta_start : meta_start + meta_len]
    side_blob = bytes(
        brotli.compress(
            np.ascontiguousarray(side_info, dtype=np.int8).tobytes(),
            quality=9,
        )
    )
    header = struct.pack(
        TT5L_HEADER_FMT,
        TT5L_MAGIC,
        TT5L_SCHEMA_VERSION,
        parsed.num_pairs,
        parsed.hidden_dim,
        parsed.num_hidden_layers,
        parsed.output_height,
        parsed.output_width,
        parsed.foveation_grid_h,
        parsed.foveation_grid_w,
        parsed.pose_dim,
        parsed.per_pair_bytes,
        len(world_blob),
        len(side_blob),
        len(ac_blob),
        len(meta_blob),
    )
    if len(header) != TT5L_HEADER_SIZE:
        raise AssertionError("TT5L header size drift while replacing side-info")
    return header + world_blob + side_blob + ac_blob + meta_blob


def _predict_residual_slice(per_pair_bytes: int) -> slice:
    offset = 0
    for name, width in TT5L_SIDE_INFO_SECTION_WIDTHS:
        end = min(offset + width, per_pair_bytes)
        if name == "predict_residual":
            return slice(offset, end)
        offset = end
    return slice(per_pair_bytes, per_pair_bytes)


def tt5l_sideinfo_variant_arrays(
    source_sideinfo: np.ndarray,
    *,
    seed: int = TT5L_SIDEINFO_VARIANT_SEED,
) -> dict[str, np.ndarray]:
    """Return deterministic side-info arrays for the L5-v2 effect curve."""

    source = np.ascontiguousarray(source_sideinfo, dtype=np.int8)
    rng = np.random.default_rng(seed)
    shuffled = source.copy()
    if shuffled.shape[0] > 1:
        shuffled = shuffled[rng.permutation(shuffled.shape[0])].copy()
    random_lsb = rng.choice(
        np.array([-1, 1], dtype=np.int8),
        size=source.shape,
        replace=True,
    ).astype(np.int8, copy=False)
    ablated = source.copy()
    ablated[:, _predict_residual_slice(int(source.shape[1]))] = 0
    variants = {
        "zero": np.zeros_like(source, dtype=np.int8),
        "random_lsb": random_lsb,
        "shuffled": shuffled,
        "trained": source.copy(),
        "ablated": ablated,
    }
    return {
        variant: variants[variant]
        for variant in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS
    }


def _variant_semantics(variant: str) -> str:
    return {
        "zero": "all per-pair side-info bytes set to zero",
        "random_lsb": "deterministic +/-1 int8 side-info control",
        "shuffled": "trained side-info row order deterministically permuted",
        "trained": "trained side-info from the source TT5L packet",
        "ablated": "trained side-info with predict_residual section zeroed",
    }[variant]


def _variant_generation_rule(variant: str, *, seed: int) -> str:
    return {
        "zero": "np.zeros_like(source_sideinfo)",
        "random_lsb": f"default_rng({seed}).choice([-1, 1], size=source_shape)",
        "shuffled": f"default_rng({seed}).permutation(source_rows)",
        "trained": "source_sideinfo.copy()",
        "ablated": "source_sideinfo with predict_residual slice zeroed",
    }[variant]


def _variant_blockers(
    *,
    variant: str,
    source_liveness: Mapping[str, Any],
    variant_liveness: Mapping[str, Any],
    same_as_zero: bool,
    same_as_trained: bool,
    sideinfo_changed_from_source: bool,
    archive_sha_changed_from_source: bool,
    archive_member_sha_changed_from_source: bool,
    sideinfo_section_sha_changed_from_source: bool,
) -> list[str]:
    blockers = ["requires_paired_cpu_cuda_exact_eval_before_score_claim"]
    source_nonzero = int(source_liveness.get("nonzero_values") or 0)
    variant_nonzero = int(variant_liveness.get("nonzero_values") or 0)
    if source_nonzero == 0:
        blockers.append("source_trained_sideinfo_all_zero")
        if variant in {"trained", "shuffled", "ablated"}:
            blockers.append(f"{variant}_variant_degenerate_from_zero_source")
    if variant in {"random_lsb", "shuffled", "trained"} and variant_nonzero <= 0:
        blockers.append(f"{variant}_sideinfo_nonzero_missing")
    if variant not in {"zero", "trained"} and same_as_zero:
        blockers.append(f"{variant}_matches_zero_control")
    if variant not in {"trained"} and same_as_trained:
        blockers.append(f"{variant}_matches_trained_control")
    if (
        variant in TT5L_VARIANTS_EXPECTED_TO_TRANSFORM_SOURCE
        and not sideinfo_changed_from_source
    ):
        blockers.append(f"{variant}_expected_sideinfo_change_missing")
    if sideinfo_changed_from_source:
        if not archive_sha_changed_from_source:
            blockers.append(f"{variant}_archive_sha_noop_against_source")
        if not archive_member_sha_changed_from_source:
            blockers.append(f"{variant}_archive_member_noop_against_source")
        if not sideinfo_section_sha_changed_from_source:
            blockers.append(f"{variant}_sideinfo_section_noop_against_source")
    return list(dict.fromkeys(blockers))


def _section_manifest(blob: bytes) -> dict[str, dict[str, int | str]]:
    return {
        name: {
            "offset": int(offset),
            "length": int(length),
            "sha256": _sha256_bytes(blob[offset : offset + length]),
        }
        for name, (offset, length) in parse_tt5l_archive_bytes(blob).items()
    }


def _section_identity_against_source(
    source_sections: Mapping[str, Mapping[str, int | str]],
    variant_sections: Mapping[str, Mapping[str, int | str]],
) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for name in ("world_model_blob", "ac_state_blob", "meta_blob"):
        source_sha = source_sections.get(name, {}).get("sha256")
        variant_sha = variant_sections.get(name, {}).get("sha256")
        out[name] = bool(source_sha and source_sha == variant_sha)
    return out


def _runtime_custody(
    *,
    submission_dir: str | Path | None,
    repo_root: Path,
) -> dict[str, Any]:
    if submission_dir is None:
        return {
            "available": False,
            "submission_dir": "",
            "blockers": ["submission_runtime_not_recorded"],
        }
    path = Path(submission_dir)
    if not path.is_absolute():
        path = repo_root / path
    if not path.exists():
        return {
            "available": False,
            "submission_dir": _repo_relative(path, repo_root),
            "blockers": ["submission_runtime_dir_missing"],
        }
    try:
        manifest = runtime_dependency_manifest(path, repo_root)
    except (OSError, ValueError, RuntimeError, SyntaxError) as exc:
        return {
            "available": False,
            "submission_dir": _repo_relative(path, repo_root),
            "blockers": [f"runtime_manifest_error:{type(exc).__name__}"],
            "error": str(exc),
        }
    return {
        "available": True,
        "submission_dir": _repo_relative(path, repo_root),
        "runtime_tree_sha256": manifest.get("runtime_tree_sha256"),
        "runtime_content_tree_sha256": manifest.get("runtime_content_tree_sha256"),
        "runtime_file_count": manifest.get("runtime_file_count"),
        "manifest": manifest,
        "blockers": [],
    }


def build_tt5l_sideinfo_variant_packets(
    *,
    source_archive: str | Path,
    output_root: str | Path,
    repo_root: str | Path,
    seed: int = TT5L_SIDEINFO_VARIANT_SEED,
    submission_dir: str | Path | None = None,
    command_argv: list[str] | None = None,
) -> dict[str, Any]:
    """Write TT5L side-info variant archives and return a custody manifest."""

    root = Path(repo_root).resolve()
    source_path = Path(source_archive)
    if not source_path.is_absolute():
        source_path = root / source_path
    out_root = Path(output_root)
    if not out_root.is_absolute():
        out_root = root / out_root
    if not source_path.exists():
        raise FileNotFoundError(f"TT5L source archive missing: {source_path}")

    member_info, member_bytes = read_tt5l_archive_zip(source_path)
    parsed = parse_archive(member_bytes)
    source_sections = _section_manifest(member_bytes)
    source_sideinfo = np.ascontiguousarray(parsed.per_pair_side_info, dtype=np.int8)
    source_liveness = side_info_liveness_stats(source_sideinfo)
    source_archive_sha = _sha256_file(source_path)
    source_archive_member_sha = _sha256_bytes(member_bytes)
    source_sideinfo_section_sha = str(
        source_sections["per_pair_side_info_blob"]["sha256"]
    )
    source_sideinfo_sha = _sha256_bytes(source_sideinfo.tobytes())
    variants = tt5l_sideinfo_variant_arrays(source_sideinfo, seed=seed)
    zero_sideinfo = variants["zero"]
    trained_sideinfo = variants["trained"]

    variant_rows: list[dict[str, Any]] = []
    for variant, sideinfo in variants.items():
        sideinfo = np.ascontiguousarray(sideinfo, dtype=np.int8)
        variant_blob = (
            member_bytes
            if variant == "trained" and np.array_equal(sideinfo, source_sideinfo)
            else _replace_sideinfo_blob(member_bytes, parsed, sideinfo)
        )
        variant_sections = _section_manifest(variant_blob)
        side_section = variant_sections["per_pair_side_info_blob"]
        archive_path = out_root / variant / "archive.zip"
        _write_archive_zip(
            archive_path,
            member_name=member_info.filename,
            member_bytes=variant_blob,
            compress_type=member_info.compress_type,
        )
        archive_bytes = archive_path.stat().st_size
        archive_sha = _sha256_file(archive_path)
        archive_member_sha = _sha256_bytes(variant_blob)
        archive_manifest_path, archive_manifest = _write_variant_archive_manifest(
            archive_path=archive_path,
            repo_root=root,
            variant=variant,
        )
        sideinfo_section_sha = str(side_section["sha256"])
        sideinfo_liveness = side_info_liveness_stats(sideinfo)
        same_as_zero = bool(np.array_equal(sideinfo, zero_sideinfo))
        same_as_trained = bool(np.array_equal(sideinfo, trained_sideinfo))
        sideinfo_changed_from_source = not bool(np.array_equal(sideinfo, source_sideinfo))
        archive_sha_changed_from_source = archive_sha != source_archive_sha
        archive_member_sha_changed_from_source = (
            archive_member_sha != source_archive_member_sha
        )
        sideinfo_section_sha_changed_from_source = (
            sideinfo_section_sha != source_sideinfo_section_sha
        )
        row = {
            "variant": variant,
            "semantics": _variant_semantics(variant),
            "generation_rule": _variant_generation_rule(variant, seed=seed),
            "variant_seed": int(seed),
            "variant_source": "source_trained_sideinfo",
            "archive_path": _repo_relative(archive_path, root),
            "archive_sha256": archive_sha,
            "archive_bytes": archive_bytes,
            "archive_manifest_path": _repo_relative(archive_manifest_path, root),
            "archive_manifest_sha256": _sha256_file(archive_manifest_path),
            "archive_member": member_info.filename,
            "archive_member_sha256": archive_member_sha,
            "archive_member_bytes": len(variant_blob),
            "archive_manifest": archive_manifest,
            "source_archive_sha256": source_archive_sha,
            "source_archive_member_sha256": source_archive_member_sha,
            "source_sideinfo_section_sha256": source_sideinfo_section_sha,
            "archive_sha_changed_from_source": archive_sha_changed_from_source,
            "archive_member_sha_changed_from_source": (
                archive_member_sha_changed_from_source
            ),
            "sideinfo_section_sha_changed_from_source": (
                sideinfo_section_sha_changed_from_source
            ),
            "parsed_sections": variant_sections,
            "mutated_sections": ["tt5l_header", "per_pair_side_info_blob"],
            "mutated_byte_ranges": [
                {
                    "section": "tt5l_header",
                    "offset": 0,
                    "length": TT5L_HEADER_SIZE,
                    "reason": "side-info compressed length may update header length fields",
                },
                {
                    "section": "per_pair_side_info_blob",
                    "offset": side_section["offset"],
                    "length": side_section["length"],
                    "reason": "variant-specific compressed side-info payload",
                },
            ],
            "non_target_sections_identical_to_source": (
                _section_identity_against_source(source_sections, variant_sections)
            ),
            "allowed_header_delta": (
                "WORLD_MODEL_LEN/AC_STATE_LEN/META_LEN preserved; "
                "PER_PAIR_SIDE_INFO_LEN may differ by variant"
            ),
            "zip_compress_type": (
                "deflated"
                if member_info.compress_type == zipfile.ZIP_DEFLATED
                else "stored"
            ),
            "sideinfo_sha256": _sha256_bytes(sideinfo.tobytes()),
            "sideinfo_liveness": sideinfo_liveness,
            "sideinfo_equal_source": same_as_trained,
            "sideinfo_equal_zero": same_as_zero,
            "sideinfo_changed_from_source": sideinfo_changed_from_source,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "rank_or_kill_eligible": False,
            "dispatch_attempted": False,
            "blockers": _variant_blockers(
                variant=variant,
                source_liveness=source_liveness,
                variant_liveness=sideinfo_liveness,
                same_as_zero=same_as_zero,
                same_as_trained=same_as_trained,
                sideinfo_changed_from_source=sideinfo_changed_from_source,
                archive_sha_changed_from_source=archive_sha_changed_from_source,
                archive_member_sha_changed_from_source=(
                    archive_member_sha_changed_from_source
                ),
                sideinfo_section_sha_changed_from_source=(
                    sideinfo_section_sha_changed_from_source
                ),
            ),
        }
        variant_rows.append(row)

    manifest_blockers = [
        "requires_paired_cpu_cuda_exact_eval_for_sideinfo_effect_curve",
        "requires_dispatch_lane_claim_before_auth_eval",
        "score_claim_forbidden_until_effect_curve_artifact_passes",
    ]
    if int(source_liveness.get("nonzero_values") or 0) <= 0:
        manifest_blockers.append("tt5l_source_trained_sideinfo_all_zero")
    if int(parsed.num_pairs) != TT5L_CONTEST_NUM_PAIRS:
        manifest_blockers.append(
            "tt5l_source_num_pairs_not_full_contest:"
            f"{int(parsed.num_pairs)}_expected_{TT5L_CONTEST_NUM_PAIRS}"
        )
    runtime_report_path = _write_runtime_report(
        submission_dir=submission_dir,
        repo_root=root,
        source_archive_path=source_path,
        source_archive_sha256=source_archive_sha,
        source_archive_bytes=source_path.stat().st_size,
        source_archive_member=member_info.filename,
        variant_rows=variant_rows,
    )
    runtime = _runtime_custody(
        submission_dir=submission_dir,
        repo_root=root,
    )
    if runtime_report_path:
        runtime["report_path"] = runtime_report_path
    manifest_blockers.extend(str(item) for item in runtime.get("blockers", []))

    return {
        "schema": TT5L_SIDEINFO_VARIANT_PACKET_SCHEMA,
        "tool": TT5L_SIDEINFO_VARIANT_PACKET_TOOL_PATH,
        "generated_at_utc": (
            datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        ),
        "command_argv": command_argv or [],
        "source_archive_path": _repo_relative(source_path, root),
        "source_archive_sha256": source_archive_sha,
        "source_archive_bytes": source_path.stat().st_size,
        "source_archive_member": member_info.filename,
        "source_archive_member_sha256": source_archive_member_sha,
        "source_archive_member_bytes": len(member_bytes),
        "source_parsed_sections": source_sections,
        "source_sideinfo_sha256": source_sideinfo_sha,
        "source_sideinfo_liveness": source_liveness,
        "runtime": runtime,
        "source_header": {
            "schema_version": parsed.schema_version,
            "num_pairs": parsed.num_pairs,
            "hidden_dim": parsed.hidden_dim,
            "num_hidden_layers": parsed.num_hidden_layers,
            "output_height": parsed.output_height,
            "output_width": parsed.output_width,
            "foveation_grid_h": parsed.foveation_grid_h,
            "foveation_grid_w": parsed.foveation_grid_w,
            "pose_dim": parsed.pose_dim,
            "per_pair_bytes": parsed.per_pair_bytes,
        },
        "variant_seed": int(seed),
        "required_effect_curve_variants": list(
            L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS
        ),
        "variant_count": len(variant_rows),
        "variants": variant_rows,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "dispatch_attempted": False,
        "blockers": list(dict.fromkeys(manifest_blockers)),
    }


def tt5l_sideinfo_variant_packets_json(payload: Mapping[str, Any]) -> str:
    """Return sorted-key JSON for a TT5L side-info packet manifest."""

    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def render_tt5l_sideinfo_variant_packets_markdown(
    payload: Mapping[str, Any],
) -> str:
    """Render a compact operator-facing packet custody report."""

    runtime = payload.get("runtime")
    runtime_summary = {}
    if isinstance(runtime, Mapping):
        runtime_summary = {
            "available": runtime.get("available"),
            "submission_dir": runtime.get("submission_dir"),
            "runtime_tree_sha256": runtime.get("runtime_tree_sha256"),
            "runtime_content_tree_sha256": runtime.get(
                "runtime_content_tree_sha256"
            ),
            "runtime_file_count": runtime.get("runtime_file_count"),
            "blockers": runtime.get("blockers"),
        }
    lines = [
        "# L5 v2 TT5L side-info variant packets",
        "",
        f"- schema: `{payload.get('schema')}`",
        f"- source_archive_path: `{payload.get('source_archive_path')}`",
        f"- source_archive_sha256: `{payload.get('source_archive_sha256')}`",
        f"- source_archive_bytes: `{payload.get('source_archive_bytes')}`",
        f"- source_archive_member: `{payload.get('source_archive_member')}`",
        f"- source_sideinfo_liveness: `{payload.get('source_sideinfo_liveness')}`",
        f"- runtime: `{runtime_summary}`",
        f"- score_claim: `{str(payload.get('score_claim')).lower()}`",
        f"- promotion_eligible: `{str(payload.get('promotion_eligible')).lower()}`",
        (
            "- ready_for_exact_eval_dispatch: "
            f"`{str(payload.get('ready_for_exact_eval_dispatch')).lower()}`"
        ),
        f"- blockers: `{payload.get('blockers')}`",
        "",
        "## Variant archives",
        "",
        (
            "| Variant | Archive bytes | Nonzero side-info values | "
            "Member changed | Side-section changed | Seed | SHA-256 | Blockers |"
        ),
        "| --- | ---: | ---: | --- | --- | ---: | --- | --- |",
    ]
    rows = payload.get("variants")
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            liveness = row.get("sideinfo_liveness")
            nonzero = (
                liveness.get("nonzero_values")
                if isinstance(liveness, Mapping)
                else ""
            )
            lines.append(
                "| "
                f"`{row.get('variant')}` | "
                f"{row.get('archive_bytes')} | "
                f"{nonzero} | "
                f"{row.get('archive_member_sha_changed_from_source')} | "
                f"{row.get('sideinfo_section_sha_changed_from_source')} | "
                f"{row.get('variant_seed')} | "
                f"`{row.get('archive_sha256')}` | "
                f"`{row.get('blockers')}` |"
            )
        generation_rows = [row for row in rows if isinstance(row, Mapping)]
        if generation_rows:
            lines.extend(["", "## Variant generation rules", ""])
            for row in generation_rows:
                lines.append(
                    f"- `{row.get('variant')}`: seed=`{row.get('variant_seed')}`; "
                    f"rule=`{row.get('generation_rule')}`"
                )
    lines.extend(
        [
            "",
            "## Classification",
            "",
            "These archives are packet controls for the TT5L side-info effect curve. "
            "They are not score claims and are not dispatch-ready until a lane claim "
            "and paired CPU/CUDA exact-eval cells exist for each variant.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "DEFAULT_TT5L_SIDEINFO_VARIANT_JSON_PATH",
    "DEFAULT_TT5L_SIDEINFO_VARIANT_OUTPUT_ROOT",
    "DEFAULT_TT5L_SIDEINFO_VARIANT_REPORT_PATH",
    "DEFAULT_TT5L_SIDEINFO_VARIANT_SUBMISSION_DIR",
    "DEFAULT_TT5L_SOURCE_ARCHIVE_PATH",
    "TT5L_CONTEST_NUM_PAIRS",
    "TT5L_SIDEINFO_VARIANT_PACKET_SCHEMA",
    "TT5L_SIDEINFO_VARIANT_PACKET_TOOL_PATH",
    "build_tt5l_sideinfo_variant_packets",
    "read_tt5l_archive_zip",
    "render_tt5l_sideinfo_variant_packets_markdown",
    "tt5l_sideinfo_variant_arrays",
    "tt5l_sideinfo_variant_packets_json",
]
