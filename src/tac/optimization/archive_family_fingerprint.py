# SPDX-License-Identifier: MIT
"""Archive-family fingerprints for repair and final-rate adapter routing."""

from __future__ import annotations

import struct
import zipfile
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from tac.repo_io import sha256_file

ARCHIVE_FAMILY_FINGERPRINT_SCHEMA = "archive_family_fingerprint.v1"
ARCHIVE_FAMILY_COVERAGE_REPORT_SCHEMA = "archive_family_coverage_report.v1"
REPAIR_BYTE_TRANSFORM_ARCHIVE_ADAPTER_REGISTRY_SCHEMA = (
    "repair_family_byte_transform_archive_adapter_registry.v2"
)

SCORE_AFFECTING_ARCHIVE_ADAPTERS: tuple[str, ...] = (
    "fec3_compact_selector",
    "fec5_fixed_huffman_k8_selector",
    "fec6_fixed_huffman_k16_selector",
    "fec8_static_second_order_k16_selector",
    "fes1_all_none_selector",
    "pact_nerv_selector_v4_packet",
)
CODER_BOUNDARY_ARCHIVE_ADAPTERS: tuple[str, ...] = (
    "zip_packet_member_recompress",
)
POST_CONTAINER_ARCHIVE_ADAPTERS: tuple[str, ...] = (
    "zip_archive_repack",
)
NEXT_SCORE_AFFECTING_ADAPTER_CLASSES: tuple[str, ...] = (
    "pact_nerv_selector_v3_packet",
    "hnerv_latent_sidecar_hdm",
    "renderer_dfl1_payload",
    "renderer_rpk1_payload",
    "renderer_asym_payload",
    "raw_hnerv_payload",
    "stc_raw_payload",
    "tensor_factorize_payload",
)

_FAMILY_AUTOMATION_SURFACES: Mapping[str, tuple[str, tuple[str, ...], str]] = {
    "fec3_compact_selector": (
        "fp11_selector_payload_mutation",
        ("bit", "byte", "pair", "frame", "entropy_coder"),
        "before_entropy_coder_distribution_shaping",
    ),
    "fec5_fixed_huffman_k8_selector": (
        "fp11_selector_payload_mutation",
        ("bit", "byte", "pair", "frame", "entropy_coder"),
        "before_entropy_coder_distribution_shaping",
    ),
    "fec6_fixed_huffman_k16_selector": (
        "fec6_selector_payload_mutation",
        ("bit", "byte", "pair", "frame", "entropy_coder"),
        "before_entropy_coder_distribution_shaping",
    ),
    "fec8_static_second_order_k16_selector": (
        "fp11_selector_payload_mutation",
        ("bit", "byte", "pair", "pair_transition", "frame", "entropy_coder"),
        "before_entropy_coder_distribution_shaping",
    ),
    "fes1_all_none_selector": (
        "fp11_selector_payload_mutation",
        ("bit", "byte", "pair", "frame", "palette"),
        "before_entropy_coder_distribution_shaping",
    ),
    "pact_nerv_selector_v3_packet": (
        "pact_nerv_packet_parser_mutator",
        ("byte", "tensor", "frame", "pair", "batch", "full_video"),
        "before_entropy_coder_distribution_shaping",
    ),
    "pact_nerv_selector_v4_packet": (
        "psv4_selector_payload_mutation",
        ("byte", "selector", "frame", "pair", "batch", "full_video"),
        "before_entropy_coder_distribution_shaping",
    ),
    "hnerv_latent_sidecar_hdm": (
        "hnerv_sidecar_latent_mutator",
        ("byte", "latent", "channel", "pair", "frame"),
        "before_entropy_coder_distribution_shaping",
    ),
    "renderer_dfl1_payload": (
        "renderer_payload_parser_mutator",
        ("byte", "payload", "region", "boundary", "frame", "batch"),
        "before_entropy_coder_distribution_shaping",
    ),
    "renderer_rpk1_payload": (
        "renderer_payload_parser_mutator",
        ("byte", "payload", "region", "boundary", "frame", "batch"),
        "before_entropy_coder_distribution_shaping",
    ),
    "renderer_asym_payload": (
        "renderer_payload_parser_mutator",
        ("byte", "payload", "region", "boundary", "frame", "batch"),
        "before_entropy_coder_distribution_shaping",
    ),
    "raw_hnerv_payload": (
        "raw_hnerv_payload_parser_mutator",
        ("byte", "tensor", "latent", "frame", "pair"),
        "before_entropy_coder_distribution_shaping",
    ),
}

_PACT_NERV_RUNTIME_FAMILIES: tuple[str, ...] = (
    "pact_nerv_selector_v3_packet",
    "pact_nerv_selector_v4_packet",
)


def _repo_rel(path: str | Path, repo_root: str | Path | None) -> str:
    value = Path(path)
    if repo_root is None:
        return value.as_posix()
    try:
        return value.resolve(strict=False).relative_to(
            Path(repo_root).resolve(strict=False)
        ).as_posix()
    except ValueError:
        return value.as_posix()


def ordered_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def repair_archive_adapter_registry() -> dict[str, Any]:
    """Return the current archive adapter capability table.

    The registry is intentionally conservative: score-affecting payload adapters
    are listed only after a packet-aware parser/mutator and receiver proof exist.
    Generic ZIP transforms remain coder-boundary or post-container only.
    """

    return {
        "schema": REPAIR_BYTE_TRANSFORM_ARCHIVE_ADAPTER_REGISTRY_SCHEMA,
        "score_affecting_adapters": list(SCORE_AFFECTING_ARCHIVE_ADAPTERS),
        "coder_boundary_adapters": list(CODER_BOUNDARY_ARCHIVE_ADAPTERS),
        "post_container_adapters": list(POST_CONTAINER_ARCHIVE_ADAPTERS),
        "unsupported_detected_families_fail_closed": True,
        "next_score_affecting_adapter_classes": list(NEXT_SCORE_AFFECTING_ADAPTER_CLASSES),
        "next_adapter_classes": list(NEXT_SCORE_AFFECTING_ADAPTER_CLASSES),
    }


def _primary_payload_member(infos: list[zipfile.ZipInfo]) -> zipfile.ZipInfo:
    by_name = {info.filename: info for info in infos}
    for preferred in ("x", "0.bin", "archive.bin", "payload.bin", "renderer.bin", "p"):
        if preferred in by_name and not by_name[preferred].is_dir():
            return by_name[preferred]
    payload_like = [
        info
        for info in infos
        if not info.is_dir()
        and not info.filename.endswith((".py", ".sh", ".txt", ".json", ".toml", ".md"))
        and "__pycache__" not in info.filename
    ]
    candidates = payload_like or [info for info in infos if not info.is_dir()]
    if not candidates:
        raise ValueError("archive has no file members")
    return max(candidates, key=lambda info: (info.file_size, -len(info.filename), info.filename))


def _selector_magic_from_fp11(payload: bytes) -> tuple[str, int, int] | None:
    if len(payload) < 12 or not payload.startswith(b"FP11"):
        return None
    source_len = struct.unpack_from("<I", payload, 4)[0]
    selector_len_offset = 8 + source_len
    if selector_len_offset + 6 > len(payload):
        return None
    selector_len = struct.unpack_from("<H", payload, selector_len_offset)[0]
    selector_start = selector_len_offset + 2
    selector_end = selector_start + selector_len
    if selector_len <= 0 or selector_start + 4 > len(payload) or selector_end > len(payload):
        return None
    magic = payload[selector_start : selector_start + 4].decode("ascii", errors="replace")
    return magic, selector_len, source_len


def _families_from_payload(
    *,
    member_name: str,
    member_count: int,
    payload: bytes,
    member_names: list[str],
) -> tuple[list[str], dict[str, Any]]:
    families: list[str] = []
    details: dict[str, Any] = {}
    head = payload[:16]

    if member_count == 1:
        families.append("single_file_payload_archive")
    if member_count > 1:
        families.append("multi_member_runtime_archive")
    if any("sidecar" in name.lower() for name in member_names):
        families.append("sidecar_named_archive")

    if head.startswith(b"FP11"):
        families.append("fp11_frame_selector_wrapper")
        selector = _selector_magic_from_fp11(payload)
        if selector is not None:
            magic, selector_len, source_len = selector
            details["fp11_selector_magic"] = magic
            details["fp11_selector_bytes"] = selector_len
            details["fp11_source_bytes"] = source_len
            if magic == "FEC3":
                families.append("fec3_compact_selector")
            elif magic == "FEC5":
                families.append("fec5_fixed_huffman_k8_selector")
            elif magic == "FEC6":
                families.append("fec6_fixed_huffman_k16_selector")
            elif magic == "FEC8":
                families.append("fec8_static_second_order_k16_selector")
            elif magic == "FES1":
                families.append("fes1_all_none_selector")
            elif magic.startswith("FEC"):
                families.append("fec_selector_wrapper_other")
            else:
                families.append("fp11_selector_wrapper_other")
    elif head.startswith(b"PSV3"):
        families.append("pact_nerv_selector_v3_packet")
    elif head.startswith(b"PSV4"):
        families.append("pact_nerv_selector_v4_packet")
    elif head.startswith(b"DFL1"):
        families.append("renderer_dfl1_payload")
    elif b"RPK1" in payload[:12]:
        families.append("renderer_rpk1_payload")
    elif head.startswith(b"ASYM"):
        families.append("renderer_asym_payload")
    elif payload[:2] == b"\xfe\r" and b"HDM" in payload[:32]:
        families.append("hnerv_latent_sidecar_hdm")
    elif member_name == "x" and payload[:4] not in (b"FP11", b"PSV3", b"PSV4"):
        families.append("raw_hnerv_payload")

    if not families or families == ["single_file_payload_archive"]:
        families.append("generic_zip_payload")
    return ordered_unique(families), details


def _runtime_portability_blockers_from_archive(
    archive: zipfile.ZipFile,
    *,
    families: list[str],
    member_names: list[str],
) -> list[str]:
    """Detect runtime portability blockers separately from score adapters."""
    if not any(family in families for family in _PACT_NERV_RUNTIME_FAMILIES):
        return []

    runtime_candidates = [
        name
        for name in member_names
        if name == "inflate.py" or name.endswith("/inflate.py")
    ]
    blockers: list[str] = []
    for name in runtime_candidates:
        try:
            text = archive.read(name).decode("utf-8", errors="ignore")
        except (OSError, KeyError, zipfile.BadZipFile):
            continue
        if "import torch" in text or "from torch" in text:
            blockers.append("pact_nerv_inflate_torch_dependency")
            break
    return ordered_unique(blockers)


def fingerprint_archive_family(
    archive_path: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Fingerprint one archive without granting score or dispatch authority."""

    path = Path(archive_path)
    blockers: list[str] = []
    member_names: list[str] = []
    member_records: list[dict[str, Any]] = []
    families: list[str] = []
    details: dict[str, Any] = {}
    primary_name = None
    primary_head_hex = ""
    primary_bytes = 0
    primary_compressed_bytes = 0
    primary_compression_method = None
    runtime_portability_blockers: list[str] = []

    if not path.is_file():
        blockers.append("archive_missing")
    else:
        try:
            with zipfile.ZipFile(path) as archive:
                infos = [info for info in archive.infolist() if not info.is_dir()]
                member_names = [info.filename for info in infos]
                member_records = [
                    {
                        "name": info.filename,
                        "bytes": info.file_size,
                        "zip_compressed_bytes": info.compress_size,
                        "zip_compression_method": info.compress_type,
                    }
                    for info in infos
                ]
                primary = _primary_payload_member(infos)
                payload = archive.read(primary.filename)
                primary_name = primary.filename
                primary_bytes = primary.file_size
                primary_compressed_bytes = primary.compress_size
                primary_compression_method = primary.compress_type
                primary_head_hex = payload[:16].hex()
                families, details = _families_from_payload(
                    member_name=primary.filename,
                    member_count=len(infos),
                    payload=payload,
                    member_names=member_names,
                )
                runtime_portability_blockers = _runtime_portability_blockers_from_archive(
                    archive,
                    families=families,
                    member_names=member_names,
                )
        except (OSError, ValueError, zipfile.BadZipFile, struct.error, KeyError) as exc:
            blockers.append(f"archive_family_probe_failed:{exc}")

    implemented_score = [
        family for family in families if family in SCORE_AFFECTING_ARCHIVE_ADAPTERS
    ]
    unsupported_score = [
        family
        for family in families
        if family in NEXT_SCORE_AFFECTING_ADAPTER_CLASSES
        and family not in SCORE_AFFECTING_ARCHIVE_ADAPTERS
    ]
    return {
        "schema": ARCHIVE_FAMILY_FINGERPRINT_SCHEMA,
        "archive_path": _repo_rel(path, repo_root),
        "archive_sha256": sha256_file(path) if path.is_file() else None,
        "archive_bytes": path.stat().st_size if path.is_file() else 0,
        "zip_member_count": len(member_names),
        "zip_members": member_records,
        "primary_payload_member": primary_name,
        "primary_payload_bytes": primary_bytes,
        "primary_payload_zip_compressed_bytes": primary_compressed_bytes,
        "primary_payload_zip_compression_method": primary_compression_method,
        "primary_payload_head_hex": primary_head_hex,
        "detected_archive_families": ordered_unique(families),
        "fingerprint_details": details,
        "adapter_registry": repair_archive_adapter_registry(),
        "score_affecting_adapter_implemented": bool(implemented_score),
        "implemented_score_affecting_families": implemented_score,
        "unsupported_score_affecting_families": unsupported_score,
        "coder_boundary_adapter_available": path.is_file() and not blockers,
        "post_container_adapter_available": path.is_file() and not blockers,
        "score_affecting_unsupported_families_fail_closed": bool(unsupported_score),
        "numpy_portable_inflate": not bool(runtime_portability_blockers),
        "runtime_portability_blockers": runtime_portability_blockers,
        "blockers": ordered_unique(blockers),
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _adapter_automation_row(
    family: str,
    *,
    count: int,
    implemented: bool,
    priority: int,
) -> dict[str, Any]:
    adapter_id, scopes, entropy_stage = _FAMILY_AUTOMATION_SURFACES.get(
        family,
        (
            "archive_family_parser_mutator",
            ("byte", "payload"),
            "before_entropy_coder_distribution_shaping",
        ),
    )
    return {
        "schema": "archive_family_adapter_automation_row.v1",
        "family": family,
        "archive_count": int(count),
        "priority": int(priority),
        "adapter_id": adapter_id,
        "implemented": bool(implemented),
        "required_executor_surface": adapter_id,
        "entropy_position": entropy_stage,
        "optimization_scopes": list(scopes),
        "next_action": (
            "route_to_existing_score_affecting_adapter"
            if implemented
            else "build_parser_mutator_receiver_proof_and_exact_handoff_gate"
        ),
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def build_archive_family_coverage_report(
    archive_paths: Iterable[str | Path],
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    rows = [
        fingerprint_archive_family(path, repo_root=repo_root)
        for path in sorted({str(path) for path in archive_paths})
    ]
    counts: Counter[str] = Counter()
    unsupported_counts: Counter[str] = Counter()
    implemented_counts: Counter[str] = Counter()
    for row in rows:
        counts.update(str(item) for item in row.get("detected_archive_families") or [])
        unsupported_counts.update(
            str(item) for item in row.get("unsupported_score_affecting_families") or []
        )
        implemented_counts.update(
            str(item) for item in row.get("implemented_score_affecting_families") or []
        )
    implemented_rows = [
        _adapter_automation_row(
            family,
            count=count,
            implemented=True,
            priority=index,
        )
        for index, (family, count) in enumerate(sorted(implemented_counts.items()), start=1)
    ]
    gap_rows = [
        _adapter_automation_row(
            family,
            count=count,
            implemented=False,
            priority=index,
        )
        for index, (family, count) in enumerate(
            sorted(unsupported_counts.items(), key=lambda item: (-item[1], item[0])),
            start=1,
        )
    ]
    return {
        "schema": ARCHIVE_FAMILY_COVERAGE_REPORT_SCHEMA,
        "archive_count": len(rows),
        "family_counts": dict(sorted(counts.items())),
        "implemented_score_affecting_family_counts": dict(sorted(implemented_counts.items())),
        "unsupported_score_affecting_family_counts": dict(sorted(unsupported_counts.items())),
        "implemented_score_affecting_adapter_rows": implemented_rows,
        "unsupported_score_affecting_adapter_gap_queue": gap_rows,
        "score_affecting_adapter_implemented_archive_count": sum(
            1 for row in rows if row.get("score_affecting_adapter_implemented") is True
        ),
        "score_affecting_unsupported_archive_count": sum(
            1 for row in rows if row.get("score_affecting_unsupported_families_fail_closed") is True
        ),
        "adapter_registry": repair_archive_adapter_registry(),
        "rows": rows,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


__all__ = [
    "ARCHIVE_FAMILY_COVERAGE_REPORT_SCHEMA",
    "ARCHIVE_FAMILY_FINGERPRINT_SCHEMA",
    "NEXT_SCORE_AFFECTING_ADAPTER_CLASSES",
    "REPAIR_BYTE_TRANSFORM_ARCHIVE_ADAPTER_REGISTRY_SCHEMA",
    "build_archive_family_coverage_report",
    "fingerprint_archive_family",
    "repair_archive_adapter_registry",
]
