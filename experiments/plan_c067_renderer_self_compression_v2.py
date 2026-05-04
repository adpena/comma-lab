#!/usr/bin/env python3
"""Plan C067 renderer self-compression v2 byte probes.

This tool is planning/profiling only. It reads an existing C067-compatible
archive, extracts the logical ``renderer.bin`` with the reviewed runtime helper,
decodes the renderer state, and emits deterministic JSON describing concrete
next options for shrinking renderer bytes. It does not write candidate
archives, dispatch GPU work, run scorers, or make score claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import zipfile
import zlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "src", REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from experiments import build_blockfp_c067_archive as c067_archive  # noqa: E402
from tac.quantizr_faithful_renderer import build_quantizr_faithful_renderer  # noqa: E402
from tac.quantizr_qzs3_codec import (  # noqa: E402
    MIXED_QZS_MAGIC,
    QZS3_MAGIC,
    decode_mixed_qzs_block_state_dict,
    decode_qzs3_state_dict,
    encode_qzs3_state_dict,
    encode_qzs4_block_search_state_dict,
)


SCHEMA = "c067_renderer_self_compression_v2_plan_v1"
TOOL = "experiments/plan_c067_renderer_self_compression_v2.py"
EVIDENCE_GRADE = "empirical_planning_only"
ORIGINAL_VIDEO_BYTES = 37_545_489
LAMBDA_RATE = 25.0 / ORIGINAL_VIDEO_BYTES
CURRENT_C067_RENDERER_STREAM_BYTES = 55_965
CURRENT_C067_RENDERER_RAW_BYTES = 59_288
DEFAULT_MIN_DISPATCH_RENDERER_BYTE_SAVINGS = 1_024
DEFAULT_C067_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/lightning_batch/"
    "exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/archive.zip"
)
DEFAULT_QZS3_BLOCK_SIZES = (16, 24, 32, 48, 64, 96, 128, 256, 512)
DEFAULT_QBF1_BLOCK_SIZES = (32, 64, 128, 256, 512, 1024)
DEFAULT_MIXED_POLICY_SPECS = (
    "component-aware-v1:frame2_pre64",
    "component-aware-v1:frame2_block2_pre64",
    "component-aware-v1:frame2_all64",
)
DEFAULT_IMP_CYCLE_COUNTS = (1, 2, 5, 10)
RENDERER_MEMBER = "renderer.bin"
MASK_MEMBER = "masks.mkv"
POSE_MEMBER = "optimized_poses.bin"
POSE_COLLAPSE_DIST_THRESHOLD = 0.05
SANE_SCORE_COLLAPSE_THRESHOLD = 1.0


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _load_json_file(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"required JSON evidence path is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"could not parse JSON evidence path {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON evidence path must contain an object: {path}")
    return payload


def _sibling_queue_metadata(path: Path) -> dict[str, Any] | None:
    queue_path = path.parent / "lightning_queue_metadata.json"
    if not queue_path.exists():
        return None
    return _load_json_file(queue_path)


def _exact_negative_evidence_contract(paths: tuple[Path, ...]) -> dict[str, Any]:
    """Summarise exact CUDA negatives that should fail-close planning families."""

    evidence: list[dict[str, Any]] = []
    closes_global_qzs3_reblock = False
    for raw_path in paths:
        path = raw_path.resolve()
        payload = _load_json_file(path)
        queue = _sibling_queue_metadata(path)
        queue_meta = (queue or {}).get("queue_metadata", {})
        candidate_label = (
            str(queue_meta.get("candidate") or queue_meta.get("lane") or path.parent.name)
        )
        pose_dist = payload.get("avg_posenet_dist")
        seg_dist = payload.get("avg_segnet_dist")
        score = payload.get("score_recomputed_from_components", payload.get("final_score"))
        pose_collapse = pose_dist is not None and float(pose_dist) > POSE_COLLAPSE_DIST_THRESHOLD
        score_collapse = score is not None and float(score) > SANE_SCORE_COLLAPSE_THRESHOLD
        qzs3_global_reblock = (
            "qzs3" in candidate_label.lower()
            or str(queue_meta.get("purpose", "")).find("qzs3_reblock") >= 0
            or "global_b" in candidate_label.lower()
        )
        if qzs3_global_reblock and (pose_collapse or score_collapse):
            closes_global_qzs3_reblock = True
        evidence.append(
            {
                "path": str(path),
                "sha256": _sha256_file(path),
                "candidate_label": candidate_label,
                "queue_metadata": queue_meta,
                "archive_bytes": payload.get("archive_size_bytes"),
                "archive_sha256": (payload.get("provenance") or {}).get("archive_sha256"),
                "score_recomputed_from_components": score,
                "avg_posenet_dist": pose_dist,
                "avg_segnet_dist": seg_dist,
                "n_samples": payload.get("n_samples"),
                "device": (payload.get("provenance") or {}).get("device"),
                "gpu_model": (payload.get("provenance") or {}).get("gpu_model"),
                "pose_collapse_signal": pose_collapse,
                "score_collapse_signal": score_collapse,
                "closes_global_qzs3_reblock": bool(
                    qzs3_global_reblock and (pose_collapse or score_collapse)
                ),
                "score_claim": False,
                "promotion_eligible": False,
            }
        )
    return {
        "schema": "c067_renderer_self_compression_exact_negative_contract_v1",
        "evidence_count": len(evidence),
        "score_claim": False,
        "promotion_eligible": False,
        "pose_collapse_dist_threshold": POSE_COLLAPSE_DIST_THRESHOLD,
        "sane_score_collapse_threshold": SANE_SCORE_COLLAPSE_THRESHOLD,
        "closed_families": {
            "global_qzs3_reblock_above_source_block_size": closes_global_qzs3_reblock,
        },
        "evidence": evidence,
    }


def _archive_screen_contract(
    paths: tuple[Path, ...],
    *,
    exact_negative_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Load deterministic local archive screens from builder summaries."""

    records: list[dict[str, Any]] = []
    lookup: dict[str, dict[str, Any]] = {}
    global_qzs3_closed = (
        exact_negative_contract.get("closed_families", {}).get(
            "global_qzs3_reblock_above_source_block_size"
        )
        is True
    )
    for raw_path in paths:
        path = raw_path.resolve()
        payload = _load_json_file(path)
        for item in payload.get("candidates", []):
            if not isinstance(item, dict):
                continue
            policy = item.get("policy") or {}
            policy_name = str(policy.get("name") or item.get("candidate_id") or "")
            policy_spec = str(policy.get("spec") or "")
            archive_delta = item.get("archive_byte_delta")
            fail_closed = bool(global_qzs3_closed and policy_spec.startswith("global:"))
            record = {
                "source_summary": str(path),
                "policy_name": policy_name,
                "policy_spec": policy_spec,
                "output_archive": item.get("output_archive"),
                "output_archive_bytes": item.get("output_archive_bytes"),
                "output_archive_sha256": item.get("output_archive_sha256"),
                "archive_byte_delta": archive_delta,
                "exact_evaluable_archive": item.get("exact_evaluable_archive"),
                "score_claim": item.get("score_claim", False),
                "promotion_eligible": item.get("promotion_eligible", False),
                "fail_closed_by_contract": fail_closed,
                "fail_closed_reason": (
                    "global QZS3 archive-byte win belongs to an exact-negative "
                    "PoseNet-collapse reblock family"
                    if fail_closed
                    else None
                ),
                "archive_byte_win": archive_delta is not None and int(archive_delta) < 0,
            }
            records.append(record)
            keys = {policy_name}
            if policy_name.startswith("global_b"):
                keys.add(policy_name.replace("global_", "qzs3_", 1))
            if policy_name:
                keys.add(f"mixed_local_{policy_name}")
            if policy_spec.startswith("global:"):
                keys.add(f"qzs3_b{int(policy_spec.split(':', 1)[1]):04d}")
            for key in keys:
                if key:
                    lookup[key] = record
    dispatchable = [
        item
        for item in records
        if item["archive_byte_win"]
        and item.get("exact_evaluable_archive")
        and not item["fail_closed_by_contract"]
    ]
    best = min(
        records,
        key=lambda item: (
            int(item["output_archive_bytes"])
            if item.get("output_archive_bytes") is not None
            else 10**18,
            str(item["policy_name"]),
        ),
        default=None,
    )
    return {
        "schema": "c067_renderer_self_compression_archive_screen_contract_v1",
        "screen_count": len(paths),
        "candidate_count": len(records),
        "score_claim": False,
        "promotion_eligible": False,
        "best_by_archive_bytes": best,
        "dispatchable_archive_byte_win_count": len(dispatchable),
        "dispatchable_archive_byte_wins": dispatchable,
        "records": records,
        "candidate_lookup": lookup,
    }


def parse_int_tuple(value: str) -> tuple[int, ...]:
    """Parse a comma-separated positive integer list for argparse/tests."""

    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("integer list must not be empty")
    out: list[int] = []
    for item in items:
        try:
            parsed = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid integer {item!r}") from exc
        if parsed <= 0 or parsed > 4096:
            raise argparse.ArgumentTypeError(
                f"integer entries must be in [1, 4096], got {parsed}"
            )
        if parsed not in out:
            out.append(parsed)
    return tuple(out)


def _normalise_positive_ints(values: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    if not values:
        raise ValueError(f"{label} must not be empty")
    out: list[int] = []
    for raw in values:
        value = int(raw)
        if value <= 0 or value > 4096:
            raise ValueError(f"{label} entries must be in [1, 4096], got {value}")
        if value not in out:
            out.append(value)
    return tuple(out)


def _renderer_wire_format(renderer: bytes) -> str:
    magic = renderer[:4]
    if magic == QZS3_MAGIC:
        return "QZS3"
    if magic == MIXED_QZS_MAGIC:
        return "MQZ1"
    if magic == b"QBF1":
        return "QBF1"
    return f"unsupported_magic_{magic!r}"


def _decode_renderer_state(renderer: bytes) -> dict[str, Any]:
    magic = renderer[:4]
    if magic == QZS3_MAGIC:
        return decode_qzs3_state_dict(renderer, device="cpu")
    if magic == MIXED_QZS_MAGIC:
        return decode_mixed_qzs_block_state_dict(renderer, device="cpu")
    if magic == b"QBF1":
        from tac.qbf1_renderer_codec import decode_qbf1_state_dict

        return decode_qbf1_state_dict(renderer, device="cpu")
    raise ValueError(
        "renderer self-compression v2 requires a JointFrameGenerator renderer "
        f"with QZS3, MQZ1, or QBF1 magic; got {magic!r}"
    )


def _load_renderer_model(state: Mapping[str, Any]) -> Any:
    model = build_quantizr_faithful_renderer().eval()
    model.load_state_dict(dict(state), strict=True)
    return model


def _maybe_brotli_q11_size(payload: bytes) -> int | None:
    try:
        import brotli
    except ImportError:
        return None
    return int(len(brotli.compress(payload, quality=11)))


def _payload_byte_probe(
    payload: bytes,
    *,
    source_stream_codec: str | None,
    source_renderer_stream_bytes: int,
    source_renderer_raw_bytes: int,
    source_renderer_sha256: str,
) -> dict[str, Any]:
    brotli_q11 = _maybe_brotli_q11_size(payload)
    raw_bytes = len(payload)
    zlib9_bytes = len(zlib.compress(payload, level=9))
    if source_stream_codec is not None and "brotli" in source_stream_codec and brotli_q11 is not None:
        projected_stream_bytes = brotli_q11
        projection_basis = "brotli_q11_to_match_source_brotli_stream"
    else:
        projected_stream_bytes = raw_bytes
        projection_basis = "raw_renderer_bytes_to_match_direct_source_stream"
    return {
        "raw_bytes": int(raw_bytes),
        "sha256": _sha256_bytes(payload),
        "zlib9_bytes": int(zlib9_bytes),
        "brotli_q11_bytes": brotli_q11,
        "projected_renderer_stream_bytes": int(projected_stream_bytes),
        "projected_stream_basis": projection_basis,
        "delta_vs_current_stream_bytes": int(
            projected_stream_bytes - source_renderer_stream_bytes
        ),
        "formula_only_rate_delta_vs_current_stream": (
            LAMBDA_RATE * float(projected_stream_bytes - source_renderer_stream_bytes)
        ),
        "delta_vs_source_raw_renderer_bytes": int(raw_bytes - source_renderer_raw_bytes),
        "state_change_vs_source_renderer": _sha256_bytes(payload)
        != source_renderer_sha256,
    }


def _packed_renderer_stream_record(source_archive: Path) -> dict[str, Any] | None:
    """Return encoded renderer stream metadata from packed archives if present."""

    try:
        with zipfile.ZipFile(source_archive, "r") as zf:
            names = [info.filename for info in zf.infolist() if not info.is_dir()]
            if len(names) != 1:
                return None
            payload_name = names[0]
            if payload_name not in {
                c067_archive.UNPACKER.PAYLOAD_BIN,
                c067_archive.UNPACKER.PAYLOAD_BR,
                c067_archive.UNPACKER.PAYLOAD_SHORT_BR,
            }:
                return None
            payload_member = zf.read(payload_name)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"not a valid zip archive: {source_archive}") from exc

    header, members = c067_archive._parse_packed_payload_member(  # noqa: SLF001
        payload_name,
        payload_member,
    )
    if RENDERER_MEMBER not in members:
        return None
    renderer = members[RENDERER_MEMBER]
    renderer_meta = None
    for item in header.get("members", []):
        if isinstance(item, dict) and item.get("name") == RENDERER_MEMBER:
            renderer_meta = item
            break
    if renderer_meta is None:
        return None
    decoded_sha = renderer_meta.get("decoded_sha256")
    if decoded_sha is not None and str(decoded_sha) != _sha256_bytes(renderer):
        raise ValueError("packed renderer decoded SHA does not match header metadata")
    return {
        "source_stream_packaging": "packed_payload_member",
        "payload_member": payload_name,
        "payload_format": header.get("payload_format"),
        "payload_schema": header.get("schema"),
        "encoded_bytes": int(renderer_meta.get("bytes", len(renderer))),
        "encoded_sha256": renderer_meta.get("sha256"),
        "codec": renderer_meta.get("codec", "raw"),
        "decoded_bytes": int(renderer_meta.get("decoded_bytes", len(renderer))),
        "decoded_sha256": decoded_sha,
    }


def _inspect_source_archive(source_archive: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    source_archive = source_archive.resolve()
    runtime_members, source_packaging = c067_archive.extract_runtime_members(source_archive)
    if RENDERER_MEMBER not in runtime_members:
        raise ValueError(f"source archive does not contain logical {RENDERER_MEMBER}")
    required_missing = [
        name for name in (RENDERER_MEMBER, MASK_MEMBER, POSE_MEMBER) if name not in runtime_members
    ]
    if required_missing:
        raise ValueError(
            "renderer self-compression planning expects C067 runtime members; "
            f"missing={required_missing}"
        )

    renderer = runtime_members[RENDERER_MEMBER]
    packed_stream = _packed_renderer_stream_record(source_archive)
    if packed_stream is None:
        packed_stream = {
            "source_stream_packaging": "direct_runtime_member",
            "payload_member": None,
            "payload_format": None,
            "payload_schema": None,
            "encoded_bytes": len(renderer),
            "encoded_sha256": _sha256_bytes(renderer),
            "codec": "raw",
            "decoded_bytes": len(renderer),
            "decoded_sha256": _sha256_bytes(renderer),
        }

    source_meta = {
        "path": str(source_archive),
        "bytes": int(source_archive.stat().st_size),
        "sha256": _sha256_file(source_archive),
        "runtime_packaging": source_packaging,
        "logical_members": {
            name: {
                "bytes": len(data),
                "sha256": _sha256_bytes(data),
            }
            for name, data in sorted(runtime_members.items())
        },
        "renderer_stream": packed_stream,
    }
    return runtime_members, source_meta


def _qzs3_block_search_options(
    state: Mapping[str, Any],
    *,
    block_sizes: tuple[int, ...],
    source_renderer_stream_bytes: int,
    source_renderer_raw_bytes: int,
    source_renderer_sha256: str,
    source_stream_codec: str | None,
) -> dict[str, Any]:
    block_sizes = _normalise_positive_ints(block_sizes, label="qzs3 block sizes")
    _, helper_meta = encode_qzs4_block_search_state_dict(
        dict(state),
        block_sizes=block_sizes,
    )
    candidates: list[dict[str, Any]] = []
    template_keys = tuple(build_quantizr_faithful_renderer().state_dict())
    for block_size in block_sizes:
        payload = encode_qzs3_state_dict(dict(state), block_size=block_size)
        decoded = decode_qzs3_state_dict(payload, device="cpu")
        if tuple(decoded) != template_keys:
            raise ValueError(f"QZS3 b{block_size} decode did not preserve state keys")
        probe = _payload_byte_probe(
            payload,
            source_stream_codec=source_stream_codec,
            source_renderer_stream_bytes=source_renderer_stream_bytes,
            source_renderer_raw_bytes=source_renderer_raw_bytes,
            source_renderer_sha256=source_renderer_sha256,
        )
        candidates.append(
            {
                "candidate_id": f"qzs3_b{block_size:04d}",
                "family": "qzs3_block_search",
                "wire_format": "QZS3",
                "qzs3_block_size": int(block_size),
                "exact_evaluable_archive": True,
                "runtime_loader_ready": True,
                "archive_builder_hint": "experiments/repack_quantizr_faithful_qzs3_archive.py",
                "score_claim": False,
                "promotion_eligible": False,
                **probe,
            }
        )

    candidates.sort(
        key=lambda item: (
            int(item["projected_renderer_stream_bytes"]),
            int(item["raw_bytes"]),
            int(item["qzs3_block_size"]),
        )
    )
    return {
        "available": True,
        "helper": "tac.quantizr_qzs3_codec.encode_qzs4_block_search_state_dict",
        "qzs4_helper_meta": helper_meta,
        "candidate_count": len(candidates),
        "best": candidates[0],
        "candidates": candidates,
    }


def _qzs3_candidate_fail_closed_reason(
    candidate: Mapping[str, Any],
    *,
    source_qzs3_block_size: int | None,
    exact_negative_contract: Mapping[str, Any],
) -> str | None:
    closed = (
        exact_negative_contract.get("closed_families", {}).get(
            "global_qzs3_reblock_above_source_block_size"
        )
        is True
    )
    if not closed:
        return None
    if candidate.get("family") != "qzs3_block_search":
        return None
    block_size = int(candidate["qzs3_block_size"])
    if source_qzs3_block_size is not None and block_size == int(source_qzs3_block_size):
        return None
    return (
        "fail-closed: exact CUDA negative evidence shows naive global QZS3 "
        "block-size reblocking can collapse PoseNet for this frontier; this "
        "candidate may be studied as cliff mapping but must not be recommended "
        "for dispatch as a byte-only win"
    )


def _annotate_fail_closed_candidates(
    qzs3: dict[str, Any],
    mixed_local: dict[str, Any],
    *,
    source_qzs3_block_size: int | None,
    exact_negative_contract: Mapping[str, Any],
) -> None:
    """Attach fail-closed contract flags to planner candidates in-place."""

    for item in qzs3.get("candidates", []):
        reason = _qzs3_candidate_fail_closed_reason(
            item,
            source_qzs3_block_size=source_qzs3_block_size,
            exact_negative_contract=exact_negative_contract,
        )
        item["fail_closed_by_contract"] = reason is not None
        item["fail_closed_reason"] = reason
    if qzs3.get("best") is not None:
        reason = _qzs3_candidate_fail_closed_reason(
            qzs3["best"],
            source_qzs3_block_size=source_qzs3_block_size,
            exact_negative_contract=exact_negative_contract,
        )
        qzs3["best"]["fail_closed_by_contract"] = reason is not None
        qzs3["best"]["fail_closed_reason"] = reason
    for item in mixed_local.get("candidates", []):
        item["fail_closed_by_contract"] = False
        item["fail_closed_reason"] = None
    if mixed_local.get("best") is not None:
        mixed_local["best"]["fail_closed_by_contract"] = False
        mixed_local["best"]["fail_closed_reason"] = None


def _annotate_archive_screen_candidates(
    qzs3: dict[str, Any],
    mixed_local: dict[str, Any],
    *,
    archive_screen_contract: Mapping[str, Any],
) -> None:
    lookup = archive_screen_contract.get("candidate_lookup", {})
    if not isinstance(lookup, Mapping):
        lookup = {}
    for group in (qzs3, mixed_local):
        for item in group.get("candidates", []):
            record = lookup.get(item.get("candidate_id"))
            if not isinstance(record, Mapping):
                item["archive_screen"] = None
                continue
            item["archive_screen"] = {
                key: record.get(key)
                for key in (
                    "policy_name",
                    "policy_spec",
                    "output_archive",
                    "output_archive_bytes",
                    "output_archive_sha256",
                    "archive_byte_delta",
                    "archive_byte_win",
                    "fail_closed_by_contract",
                    "fail_closed_reason",
                )
            }
            if record.get("fail_closed_by_contract"):
                item["fail_closed_by_contract"] = True
                item["fail_closed_reason"] = record.get("fail_closed_reason")
            elif record.get("archive_byte_win") is False:
                item["fail_closed_by_contract"] = True
                item["fail_closed_reason"] = (
                    "fail-closed: deterministic archive-byte screen is not a "
                    "local archive byte win after charged headers/packing"
                )
        if group.get("best") is not None:
            best_id = group["best"].get("candidate_id")
            record = lookup.get(best_id)
            group["best"]["archive_screen"] = record if isinstance(record, Mapping) else None
            if isinstance(record, Mapping) and record.get("fail_closed_by_contract"):
                group["best"]["fail_closed_by_contract"] = True
                group["best"]["fail_closed_reason"] = record.get("fail_closed_reason")
            elif isinstance(record, Mapping) and record.get("archive_byte_win") is False:
                group["best"]["fail_closed_by_contract"] = True
                group["best"]["fail_closed_reason"] = (
                    "fail-closed: deterministic archive-byte screen is not a "
                    "local archive byte win after charged headers/packing"
                )


def _mixed_local_block_options(
    state: Mapping[str, Any],
    *,
    policy_specs: tuple[str, ...],
    source_renderer_stream_bytes: int,
    source_renderer_raw_bytes: int,
    source_renderer_sha256: str,
    source_stream_codec: str | None,
) -> dict[str, Any]:
    try:
        from experiments import build_mixed_qzs_block_candidate as mixed_qzs
    except Exception as exc:
        return {
            "available": False,
            "helper": "experiments/build_mixed_qzs_block_candidate.py",
            "reason": f"helper import failed: {exc}",
            "candidate_count": 0,
            "candidates": [],
        }

    candidates: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    template_keys = tuple(build_quantizr_faithful_renderer().state_dict())
    for spec in policy_specs:
        try:
            policy = mixed_qzs.parse_block_policy(spec)
            if policy.prefix_overrides:
                payload, block_meta = mixed_qzs.encode_mixed_qzs_block_payload(
                    dict(state),
                    policy,
                )
                decoded = decode_mixed_qzs_block_state_dict(payload, device="cpu")
                wire_format = "MQZ1"
                runtime_loader_ready = bool(block_meta.get("runtime_decoder_available"))
            else:
                payload = encode_qzs3_state_dict(
                    dict(state),
                    block_size=policy.default_block_size,
                )
                decoded = decode_qzs3_state_dict(payload, device="cpu")
                block_meta = {
                    "wire_format": "QZS3",
                    "policy": policy.as_json(),
                    "runtime_decoder_available": True,
                }
                wire_format = "QZS3"
                runtime_loader_ready = True
            if tuple(decoded) != template_keys:
                raise ValueError(f"policy {spec!r} decode did not preserve state keys")
            probe = _payload_byte_probe(
                payload,
                source_stream_codec=source_stream_codec,
                source_renderer_stream_bytes=source_renderer_stream_bytes,
                source_renderer_raw_bytes=source_renderer_raw_bytes,
                source_renderer_sha256=source_renderer_sha256,
            )
            candidates.append(
                {
                    "candidate_id": f"mixed_local_{policy.name}",
                    "family": "mixed_local_qzs_blocks",
                    "wire_format": wire_format,
                    "policy": policy.as_json(),
                    "exact_evaluable_archive": True,
                    "runtime_loader_ready": runtime_loader_ready,
                    "archive_builder_hint": "experiments/build_mixed_qzs_block_candidate.py",
                    "score_claim": False,
                    "promotion_eligible": False,
                    "block_meta": block_meta,
                    **probe,
                }
            )
        except Exception as exc:
            errors.append({"policy_spec": spec, "error": str(exc)})

    candidates.sort(
        key=lambda item: (
            int(item["projected_renderer_stream_bytes"]),
            int(item["raw_bytes"]),
            str(item["candidate_id"]),
        )
    )
    return {
        "available": True,
        "helper": "experiments/build_mixed_qzs_block_candidate.py",
        "policy_specs": list(policy_specs),
        "candidate_count": len(candidates),
        "best": candidates[0] if candidates else None,
        "candidates": candidates,
        "policy_errors": errors,
    }


def _qbf1_negative_evidence(
    state: Mapping[str, Any],
    *,
    block_sizes: tuple[int, ...],
    source_renderer_raw_bytes: int,
    source_renderer_stream_bytes: int,
) -> dict[str, Any]:
    from tac.qbf1_renderer_codec import profile_qbf1_v2_renderer_bytes

    block_sizes = _normalise_positive_ints(block_sizes, label="QBF1 block sizes")
    profile = profile_qbf1_v2_renderer_bytes(
        dict(state),
        block_sizes=block_sizes,
        reference_qzs3_nbytes=source_renderer_raw_bytes,
    )
    best = profile["best"]
    best_with_stream_deltas: dict[str, Any] = {}
    for name, record in best.items():
        item = dict(record)
        item["raw_delta_vs_current_encoded_stream_bytes"] = (
            int(item["raw_nbytes"]) - int(source_renderer_stream_bytes)
        )
        item["beats_current_encoded_stream_by_raw_bytes"] = (
            int(item["raw_nbytes"]) < int(source_renderer_stream_bytes)
        )
        best_with_stream_deltas[name] = item
    return {
        "available": True,
        "helper": "tac.qbf1_renderer_codec.profile_qbf1_v2_renderer_bytes",
        "schema": profile["schema"],
        "score_claim": False,
        "promotion_eligible": False,
        "reference": profile["reference"],
        "block_sizes": list(block_sizes),
        "best": best_with_stream_deltas,
        "readiness": profile["readiness"],
        "candidate_families": profile["candidate_families"],
        "decision": {
            "qbf1_dispatch_warranted": False,
            "reason": (
                "QBF1 remains negative evidence unless a concrete loader-ready "
                "layout beats both the decoded QZS3 renderer bytes and the "
                "current encoded renderer stream before exact CUDA is queued."
            ),
        },
    }


def _imp_sparsity_prior(
    model: Any,
    *,
    cycle_counts: tuple[int, ...],
    sparsity_increment: float,
) -> dict[str, Any]:
    from tac.iterative_magnitude_pruning import iter_prunable_parameters

    if not (0.0 < sparsity_increment < 1.0):
        raise ValueError(
            f"sparsity_increment must be in (0, 1), got {sparsity_increment}"
        )
    cycle_counts = _normalise_positive_ints(cycle_counts, label="IMP cycle counts")
    prunable = iter_prunable_parameters(model)
    prunable_value_count = sum(int(param.numel()) for _name, param in prunable)
    total_param_count = sum(int(param.numel()) for param in model.parameters())
    non_prunable_value_count = total_param_count - prunable_value_count
    dense_fp4_prunable_proxy_bytes = int(math.ceil(prunable_value_count / 2.0))
    cycles: list[dict[str, Any]] = []
    for cycle in cycle_counts:
        expected_sparsity = 1.0 - (1.0 - sparsity_increment) ** cycle
        expected_nnz = int(math.ceil(prunable_value_count * (1.0 - expected_sparsity)))
        sparse_csr_fp4_proxy_bytes = int(math.ceil(expected_nnz * 2.5))
        cycles.append(
            {
                "cycle": int(cycle),
                "expected_sparsity": expected_sparsity,
                "expected_prunable_nnz": expected_nnz,
                "sparse_csr_fp4_proxy_bytes": sparse_csr_fp4_proxy_bytes,
                "dense_fp4_prunable_proxy_bytes": dense_fp4_prunable_proxy_bytes,
                "proxy_delta_vs_dense_fp4_prunable_bytes": int(
                    sparse_csr_fp4_proxy_bytes - dense_fp4_prunable_proxy_bytes
                ),
                "beats_dense_fp4_prunable_proxy": (
                    sparse_csr_fp4_proxy_bytes < dense_fp4_prunable_proxy_bytes
                ),
            }
        )
    return {
        "available": True,
        "helper": "tac.iterative_magnitude_pruning.iter_prunable_parameters",
        "score_claim": False,
        "promotion_eligible": False,
        "dispatchable_candidate": False,
        "bridge_builder_invoked": False,
        "use": (
            "sparsity prior only: rank tensor groups or future atoms; do not "
            "treat no-train pruning as scorer evidence"
        ),
        "sparsity_increment": sparsity_increment,
        "prunable_tensor_count": len(prunable),
        "prunable_value_count": prunable_value_count,
        "non_prunable_value_count": non_prunable_value_count,
        "total_param_count": total_param_count,
        "cycle_priors": cycles,
        "first_cycle_beating_dense_fp4_proxy": next(
            (
                item["cycle"]
                for item in cycles
                if item["beats_dense_fp4_prunable_proxy"]
            ),
            None,
        ),
        "risk": (
            "Sparse bytes are not enough; trained IMP or sparse export still "
            "needs a charged decoder contract and exact CUDA component gates."
        ),
    }


def _select_dispatch_decision(
    *,
    qzs3: dict[str, Any],
    mixed_local: dict[str, Any],
    qbf1: dict[str, Any],
    source_renderer_stream_bytes: int,
    min_dispatch_renderer_byte_savings: int = DEFAULT_MIN_DISPATCH_RENDERER_BYTE_SAVINGS,
) -> dict[str, Any]:
    if min_dispatch_renderer_byte_savings < 0:
        raise ValueError(
            "min_dispatch_renderer_byte_savings must be >= 0, got "
            f"{min_dispatch_renderer_byte_savings}"
        )
    concrete: list[dict[str, Any]] = []
    for group in (qzs3, mixed_local):
        for item in group.get("candidates", []):
            if item.get("fail_closed_by_contract"):
                continue
            if not item.get("exact_evaluable_archive") or not item.get("runtime_loader_ready"):
                continue
            concrete.append(item)
    concrete.sort(
        key=lambda item: (
            int(item["projected_renderer_stream_bytes"]),
            int(item["raw_bytes"]),
            str(item["candidate_id"]),
        )
    )
    best = concrete[0] if concrete else None
    byte_positive = bool(
        best is not None
        and int(best["projected_renderer_stream_bytes"]) < int(source_renderer_stream_bytes)
        and bool(best["state_change_vs_source_renderer"])
    )
    best_byte_savings = (
        -int(best["delta_vs_current_stream_bytes"])
        if best is not None
        and int(best.get("delta_vs_current_stream_bytes", 0)) < 0
        else 0
    )
    material_byte_positive = bool(
        byte_positive and best_byte_savings >= int(min_dispatch_renderer_byte_savings)
    )
    qbf1_dispatch = bool((qbf1.get("decision") or {}).get("qbf1_dispatch_warranted"))
    warranted = bool(material_byte_positive and not qbf1_dispatch)
    fail_closed_candidates = [
        item
        for group in (qzs3, mixed_local)
        for item in group.get("candidates", [])
        if item.get("fail_closed_by_contract")
    ]
    if best is None and fail_closed_candidates:
        reason = (
            "all local byte-positive QZS3 candidates are fail-closed by exact "
            "negative PoseNet-collapse evidence; no MQZ1/QBF1 archive-byte win "
            "is available from this planner"
        )
    elif best is None:
        reason = "no exact-evaluable QZS3/MQZ1 renderer candidate was produced"
    elif not best["state_change_vs_source_renderer"]:
        reason = (
            f"best candidate {best['candidate_id']} is a no-op against the "
            "source renderer bytes"
        )
    elif int(best["projected_renderer_stream_bytes"]) >= int(source_renderer_stream_bytes):
        reason = (
            f"best concrete candidate {best['candidate_id']} projects to "
            f"{best['projected_renderer_stream_bytes']} bytes, not below the "
            f"current {source_renderer_stream_bytes}-byte renderer stream"
        )
    elif best_byte_savings < int(min_dispatch_renderer_byte_savings):
        reason = (
            f"best concrete candidate {best['candidate_id']} locally saves only "
            f"{best_byte_savings} renderer-stream bytes, below the "
            f"{min_dispatch_renderer_byte_savings}-byte dispatch gate; keep as "
            "polish-only and prioritize learned/trained renderer compression"
        )
    else:
        reason = (
            f"best concrete candidate {best['candidate_id']} locally beats the "
            f"current renderer stream by {-int(best['delta_vs_current_stream_bytes'])} "
            "bytes; exact CUDA is warranted only after building a deterministic "
            "archive and claiming the lane"
        )
    return {
        "exact_cuda_dispatch_warranted": warranted,
        "score_claim": False,
        "promotion_eligible": False,
        "min_dispatch_renderer_byte_savings": int(min_dispatch_renderer_byte_savings),
        "best_renderer_byte_savings": int(best_byte_savings),
        "best_concrete_candidate": None
        if best is None
        else {
            "candidate_id": best["candidate_id"],
            "family": best["family"],
            "wire_format": best["wire_format"],
            "projected_renderer_stream_bytes": best["projected_renderer_stream_bytes"],
            "delta_vs_current_stream_bytes": best["delta_vs_current_stream_bytes"],
            "raw_bytes": best["raw_bytes"],
            "sha256": best["sha256"],
            "archive_builder_hint": best["archive_builder_hint"],
        },
        "reason": reason,
        "dispatch_prerequisites": [
            "claim lane with tools/claim_lane_dispatch.py before any exact eval",
            "build a deterministic archive with all score-affecting bytes charged",
            "record archive bytes/SHA, runtime manifest, source archive custody, and build provenance",
            "run experiments/contest_auth_eval.py --device cuda on archive.zip -> inflate.sh -> upstream/evaluate.py",
            "treat the result as non-promotable until component gates and score recomputation pass",
        ],
        "qbf1_dispatch_warranted": qbf1_dispatch,
        "fail_closed_candidate_count": len(fail_closed_candidates),
        "best_fail_closed_candidate": None
        if not fail_closed_candidates
        else {
            "candidate_id": fail_closed_candidates[0]["candidate_id"],
            "family": fail_closed_candidates[0]["family"],
            "wire_format": fail_closed_candidates[0]["wire_format"],
            "projected_renderer_stream_bytes": fail_closed_candidates[0][
                "projected_renderer_stream_bytes"
            ],
            "delta_vs_current_stream_bytes": fail_closed_candidates[0][
                "delta_vs_current_stream_bytes"
            ],
            "raw_bytes": fail_closed_candidates[0]["raw_bytes"],
            "sha256": fail_closed_candidates[0]["sha256"],
            "fail_closed_reason": fail_closed_candidates[0]["fail_closed_reason"],
        },
    }


def build_plan(
    *,
    source_archive: Path = DEFAULT_C067_ARCHIVE,
    qzs3_block_sizes: tuple[int, ...] = DEFAULT_QZS3_BLOCK_SIZES,
    mixed_policy_specs: tuple[str, ...] = DEFAULT_MIXED_POLICY_SPECS,
    qbf1_block_sizes: tuple[int, ...] = DEFAULT_QBF1_BLOCK_SIZES,
    imp_cycle_counts: tuple[int, ...] = DEFAULT_IMP_CYCLE_COUNTS,
    imp_sparsity_increment: float = 0.20,
    reference_renderer_stream_bytes: int | None = None,
    exact_negative_evidence_paths: tuple[Path, ...] = (),
    archive_screen_summary_paths: tuple[Path, ...] = (),
    min_dispatch_renderer_byte_savings: int = DEFAULT_MIN_DISPATCH_RENDERER_BYTE_SAVINGS,
) -> dict[str, Any]:
    """Build the deterministic planning JSON payload."""

    runtime_members, source = _inspect_source_archive(source_archive)
    renderer = runtime_members[RENDERER_MEMBER]
    state = _decode_renderer_state(renderer)
    model = _load_renderer_model(state)
    source_stream = source["renderer_stream"]
    source_renderer_stream_bytes = (
        int(reference_renderer_stream_bytes)
        if reference_renderer_stream_bytes is not None
        else int(source_stream["encoded_bytes"])
    )
    if source_renderer_stream_bytes <= 0:
        raise ValueError(
            "reference renderer stream bytes must be positive; got "
            f"{source_renderer_stream_bytes}"
        )
    source_renderer_raw_bytes = len(renderer)
    source_renderer_sha256 = _sha256_bytes(renderer)
    source_stream_codec = source_stream.get("codec")
    source_qzs3_block_size = (
        int.from_bytes(renderer[4:6], "little")
        if renderer.startswith(QZS3_MAGIC) and len(renderer) >= 6
        else None
    )

    qzs3 = _qzs3_block_search_options(
        state,
        block_sizes=qzs3_block_sizes,
        source_renderer_stream_bytes=source_renderer_stream_bytes,
        source_renderer_raw_bytes=source_renderer_raw_bytes,
        source_renderer_sha256=source_renderer_sha256,
        source_stream_codec=source_stream_codec,
    )
    mixed_local = _mixed_local_block_options(
        state,
        policy_specs=mixed_policy_specs,
        source_renderer_stream_bytes=source_renderer_stream_bytes,
        source_renderer_raw_bytes=source_renderer_raw_bytes,
        source_renderer_sha256=source_renderer_sha256,
        source_stream_codec=source_stream_codec,
    )
    qbf1 = _qbf1_negative_evidence(
        state,
        block_sizes=qbf1_block_sizes,
        source_renderer_raw_bytes=source_renderer_raw_bytes,
        source_renderer_stream_bytes=source_renderer_stream_bytes,
    )
    imp_prior = _imp_sparsity_prior(
        model,
        cycle_counts=imp_cycle_counts,
        sparsity_increment=imp_sparsity_increment,
    )
    exact_negative_contract = _exact_negative_evidence_contract(
        exact_negative_evidence_paths
    )
    archive_screen = _archive_screen_contract(
        archive_screen_summary_paths,
        exact_negative_contract=exact_negative_contract,
    )
    _annotate_fail_closed_candidates(
        qzs3,
        mixed_local,
        source_qzs3_block_size=source_qzs3_block_size,
        exact_negative_contract=exact_negative_contract,
    )
    _annotate_archive_screen_candidates(
        qzs3,
        mixed_local,
        archive_screen_contract=archive_screen,
    )
    dispatch = _select_dispatch_decision(
        qzs3=qzs3,
        mixed_local=mixed_local,
        qbf1=qbf1,
        source_renderer_stream_bytes=source_renderer_stream_bytes,
        min_dispatch_renderer_byte_savings=min_dispatch_renderer_byte_savings,
    )
    return {
        "schema": SCHEMA,
        "tool": TOOL,
        "score_claim": False,
        "promotion_eligible": False,
        "remote_or_gpu_dispatch": False,
        "evidence_grade": EVIDENCE_GRADE,
        "canonical_score_source_required": (
            "archive.zip -> inflate.sh -> upstream/evaluate.py via "
            "experiments/contest_auth_eval.py --device cuda"
        ),
        "source_archive": source,
        "source_renderer": {
            "member_name": RENDERER_MEMBER,
            "wire_format": _renderer_wire_format(renderer),
            "raw_bytes": source_renderer_raw_bytes,
            "sha256": source_renderer_sha256,
            "qzs3_block_size": source_qzs3_block_size,
            "state_tensor_count": len(state),
            "encoded_stream_bytes_used_for_gate": source_renderer_stream_bytes,
            "encoded_stream_reference_source": (
                "caller_override"
                if reference_renderer_stream_bytes is not None
                else "archive_metadata"
            ),
            "current_c067_reference_stream_bytes": CURRENT_C067_RENDERER_STREAM_BYTES,
            "current_c067_reference_raw_bytes": CURRENT_C067_RENDERER_RAW_BYTES,
            "matches_current_c067_stream_reference": (
                source_renderer_stream_bytes == CURRENT_C067_RENDERER_STREAM_BYTES
            ),
            "matches_current_c067_raw_reference": (
                source_renderer_raw_bytes == CURRENT_C067_RENDERER_RAW_BYTES
            ),
        },
        "planning_constraints": {
            "build_planning_only": True,
            "deterministic_json": True,
            "candidate_archives_written": False,
            "scorers_loaded": False,
            "gpu_required": False,
            "imp_bridge_builder_invoked": False,
            "no_score_claim": True,
            "dispatch_gate_renderer_stream_bytes": source_renderer_stream_bytes,
            "min_dispatch_renderer_byte_savings": int(min_dispatch_renderer_byte_savings),
        },
        "qzs3_block_search": qzs3,
        "mixed_local_block_candidates": mixed_local,
        "qbf1_negative_evidence": qbf1,
        "imp_sparsity_prior": imp_prior,
        "exact_negative_contract": exact_negative_contract,
        "archive_screen_contract": {
            key: value
            for key, value in archive_screen.items()
            if key != "candidate_lookup"
        },
        "dispatch_recommendation": dispatch,
    }


def write_plan(output_json: Path, **kwargs: Any) -> dict[str, Any]:
    plan = build_plan(**kwargs)
    output_json = output_json.resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_bytes(_json_bytes(plan))
    return plan


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-archive",
        type=Path,
        default=DEFAULT_C067_ARCHIVE,
        help="C067-compatible source archive to inspect",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="optional JSON path; stdout is always written",
    )
    parser.add_argument(
        "--qzs3-block-sizes",
        type=parse_int_tuple,
        default=DEFAULT_QZS3_BLOCK_SIZES,
    )
    parser.add_argument(
        "--mixed-policy",
        action="append",
        default=None,
        help=(
            "repeatable mixed/local QZS policy spec; defaults to component-aware "
            "policies from build_mixed_qzs_block_candidate.py"
        ),
    )
    parser.add_argument(
        "--qbf1-block-sizes",
        type=parse_int_tuple,
        default=DEFAULT_QBF1_BLOCK_SIZES,
    )
    parser.add_argument(
        "--imp-cycle-counts",
        type=parse_int_tuple,
        default=DEFAULT_IMP_CYCLE_COUNTS,
    )
    parser.add_argument("--imp-sparsity-increment", type=float, default=0.20)
    parser.add_argument(
        "--reference-renderer-stream-bytes",
        type=int,
        default=None,
        help=(
            "override the renderer stream byte gate; by default this is read "
            "from archive metadata when available"
        ),
    )
    parser.add_argument(
        "--exact-negative-evidence-json",
        type=Path,
        action="append",
        default=None,
        help=(
            "Repeatable exact CUDA negative JSON. Pose/score collapse in a "
            "global QZS3 reblock closes that family for dispatch planning."
        ),
    )
    parser.add_argument(
        "--archive-screen-summary-json",
        type=Path,
        action="append",
        default=None,
        help=(
            "Repeatable deterministic local archive-screen summary JSON used "
            "to reject stream-only byte wins that are not archive-byte wins."
        ),
    )
    parser.add_argument(
        "--min-dispatch-renderer-byte-savings",
        type=int,
        default=DEFAULT_MIN_DISPATCH_RENDERER_BYTE_SAVINGS,
        help=(
            "minimum local renderer-stream byte saving before this planner "
            "recommends exact CUDA dispatch; smaller wins are polish-only"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan = build_plan(
        source_archive=args.source_archive,
        qzs3_block_sizes=args.qzs3_block_sizes,
        mixed_policy_specs=tuple(args.mixed_policy)
        if args.mixed_policy
        else DEFAULT_MIXED_POLICY_SPECS,
        qbf1_block_sizes=args.qbf1_block_sizes,
        imp_cycle_counts=args.imp_cycle_counts,
        imp_sparsity_increment=args.imp_sparsity_increment,
        reference_renderer_stream_bytes=args.reference_renderer_stream_bytes,
        exact_negative_evidence_paths=tuple(args.exact_negative_evidence_json or ()),
        archive_screen_summary_paths=tuple(args.archive_screen_summary_json or ()),
        min_dispatch_renderer_byte_savings=args.min_dispatch_renderer_byte_savings,
    )
    text = _json_bytes(plan).decode("utf-8")
    if args.output_json is not None:
        args.output_json.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
