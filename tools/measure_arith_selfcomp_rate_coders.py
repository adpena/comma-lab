#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure exact rate-coder frames without launching or scoring anything.

The only written artifact is the explicit ``--output`` JSON.  Source caches,
residuals, and donor checkpoints are opened read-only.  PDW1 is always rebuilt
from the supplied n600 cache and frozen SegNet head; PDW2 never receives a fake
coded row.  This tool has no score, contest-axis, dispatch, or promotion
authority.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from tac.boundary_math.power_diagram_witness import (
    decode_pdw1,
    encode_pdw1,
    initialize_video_fed_target,
)
from tac.optimization.arith_selfcomp_rate_coders import (
    RateCoderError,
    authority_labels,
    measure_block_fp,
    measure_byte_ladder,
    measure_iid_signed_array_ladder,
    measure_signed_array_ladder,
    serialize_signed_array,
)

SCHEMA = "arith_selfcomp_rate_coders.measurement.v2"
EXPECTED_PDW1_BYTES = 338
EXPECTED_PDW1_SHA256 = "84a49d802dc5bd9c416013fd71bc6f08655a2f3c23c249374469a4dc4d8ee275"
HISTORICAL_SELFCOMP_ROW = {
    "authority_label": "HISTORICAL_MEASURED_NOT_CURRENT_DONOR_NOT_SAME_CHECKPOINT_EQUIVALENCE",
    "reported_size": "52.6KB",
    "reported_bits_per_parameter": 6.54,
    "source_task": "#496",
    "comparison_rule": "display separately; never subtract from or equate to current donor sections",
}
PRIOR_DONOR_SECTION_ROWS = {
    "authority_label": "PRIOR_MEASURED_RECEIPT_ROWS_NOT_REMEASURED_BY_THIS_INVOCATION",
    "base_weight_int8_brotli_bytes": 61_842,
    "code_weight_int8_brotli_bytes": 20_355,
    "section_identity_required": True,
}
SECTION_CODE_TOKENS = ("code", "latent", "embedding", "embed", "dxi", "pair_code")


class MeasurementError(ValueError):
    """Raised when a measurement would otherwise overclaim authority."""


def _sha256_file(path: Path, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_receipt(path: Path, *, hash_file: bool = True) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise MeasurementError(f"input is not a file: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved) if hash_file else None,
        "read_only_use": True,
    }


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MeasurementError(f"cannot read JSON {path}: {exc}") from exc


def _write_output(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise MeasurementError(f"refusing to overwrite existing --output: {path}")
    if not path.parent.is_dir():
        raise MeasurementError(f"--output parent does not exist: {path.parent}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _git_value(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "UNAVAILABLE"


def _parse_named_path(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, path = raw.split("=", 1)
    if not name or not path or any(character.isspace() for character in name):
        raise argparse.ArgumentTypeError("NAME=PATH requires a nonempty whitespace-free NAME")
    return name, Path(path)


def _numeric_scalar(value: Any) -> float | int | None:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "iuf":
        return None
    scalar = array.item()
    if isinstance(scalar, float) and not np.isfinite(scalar):
        raise MeasurementError("non-finite scalar metadata")
    return scalar


def _load_residual_op(path: Path) -> tuple[dict[str, np.ndarray], dict[str, float | int], dict[str, Any]]:
    receipt = _path_receipt(path)
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, float | int] = {}
    suffix = path.suffix.lower()
    if suffix == ".npy":
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        arrays[path.stem] = np.asarray(value)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            for member in sorted(archive.files):
                value = archive[member]
                scalar = _numeric_scalar(value)
                if scalar is not None and member in {
                    "d_seg",
                    "d_pose",
                    "n_pairs",
                    "archive_bytes",
                    "realized_dseg",
                }:
                    metadata[member] = scalar
                else:
                    arrays[member] = np.asarray(value)
    else:
        raise MeasurementError(f"residual op point must be .npy or .npz: {path}")
    if not arrays:
        raise MeasurementError(f"residual op point has no array payload: {path}")
    for member, array in arrays.items():
        if array.dtype.kind != "i" or array.dtype.itemsize not in (1, 2, 4, 8):
            raise MeasurementError(f"residual {path}:{member} must be signed int8/int16/int32/int64")
        if array.ndim < 3:
            raise MeasurementError(f"residual {path}:{member} must have explicit [...,H,W,C] shape")
    return arrays, metadata, receipt


def _aggregate_ladders(ladders: Sequence[Mapping[str, Mapping[str, Any]]]) -> dict[str, Any]:
    names = sorted(set.intersection(*(set(ladder) for ladder in ladders)))
    aggregate: dict[str, Any] = {}
    for name in names:
        rows = [ladder[name] for ladder in ladders]
        available = all(bool(row.get("available", True)) for row in rows)
        aggregate[name] = {
            "available": available,
            "framed_bytes": sum(int(row["framed_bytes"]) for row in rows) if available else None,
            "all_parseback_exact": available and all(bool(row["parseback_exact"]) for row in rows),
            "tensor_count": len(rows),
            "unavailable_reasons": [
                str(row.get("unavailable_reason")) for row in rows if not row.get("available", True)
            ],
        }
    return aggregate


def _measure_residual_op(name: str, path: Path) -> dict[str, Any]:
    arrays, metadata, source = _load_residual_op(path)
    per_member: list[dict[str, Any]] = []
    ladders: list[dict[str, dict[str, Any]]] = []
    for member, array in arrays.items():
        ladder = measure_signed_array_ladder(array)
        ladders.append(ladder)
        per_member.append(
            {
                "member": member,
                "shape": list(array.shape),
                "dtype": array.dtype.str,
                "elements": int(array.size),
                "canonical_array_sha256": hashlib.sha256(serialize_signed_array(array)).hexdigest(),
                "coders": ladder,
            }
        )
    aggregate = _aggregate_ladders(ladders)
    brotli = aggregate.get("brotli_q11", {})
    brotli_bytes = brotli.get("framed_bytes") if brotli.get("available") else None
    available_sizes = {
        coder: row["framed_bytes"]
        for coder, row in aggregate.items()
        if row.get("available") and isinstance(row.get("framed_bytes"), int)
    }
    ratios = {
        coder: (float(size) / float(brotli_bytes) if brotli_bytes else None) for coder, size in available_sizes.items()
    }
    best_name = min(available_sizes, key=available_sizes.get) if available_sizes else None
    return {
        "op_point": name,
        "source": source,
        "input_metadata": metadata,
        "members": per_member,
        "aggregate": aggregate,
        "coder_over_brotli_ratio": ratios,
        "best_available_coder": best_name,
        "best_over_brotli_ratio": ratios.get(best_name) if best_name else None,
        "authority_label": "MEASURED_EXACT_INPUT_ARRAY_LOCAL_PARSEBACK_BYTES",
        "score_authority": False,
    }


def _settled_seg_secant_curve(path: Path | None) -> dict[str, Any] | None:
    """Import settled n24 Brotli/zstd rows without inventing missing payload bytes."""

    if path is None:
        return None
    document = _load_json(path)
    if not isinstance(document, Mapping) or document.get("schema") != "seg_secant_rd_curve_composed.v1":
        raise MeasurementError("--seg-secant-curve must use the composed v1 schema")
    if int(document.get("unique_pair_count", 0)) < 24:
        raise MeasurementError("--seg-secant-curve must contain at least 24 unique real pairs")
    source_receipts: list[dict[str, Any]] = []
    for reference in document.get("source_receipts", []):
        if not isinstance(reference, Mapping):
            raise MeasurementError("Seg-secant source receipt reference is malformed")
        receipt_path = Path(str(reference.get("path", "")))
        actual = _path_receipt(receipt_path)
        if actual["sha256"] != reference.get("sha256"):
            raise MeasurementError(f"Seg-secant source receipt hash mismatch: {receipt_path}")
        source_receipts.append(actual)
    if len(source_receipts) < 2:
        raise MeasurementError("Seg-secant curve must bind at least two immutable source receipts")
    points: list[dict[str, Any]] = []
    for point in document.get("measured_points", []):
        if not isinstance(point, Mapping):
            raise MeasurementError("Seg-secant point is malformed")
        pair_count = int(point.get("pair_count", 0))
        brotli_mean = float(point["brotli_q11_bytes_per_pair"])
        zstd_mean = float(point["zstd_19_bytes_per_pair"])
        if pair_count < 24 or brotli_mean <= 0 or zstd_mean <= 0:
            raise MeasurementError("Seg-secant point lacks positive n>=24 byte custody")
        exact_payload_blocker = (
            "SETTLED_RECEIPTS_PRESERVE_STREAM_HASHES_AND_BROTLI_ZSTD_TOTALS_NOT_RESIDUAL_BYTES; "
            "LZMA_CONSTRICTION_ZIGZAG_RLE_REQUIRE_EXACT_REDERIVATION"
        )
        points.append(
            {
                "point_id": str(point["point_id"]),
                "family": str(point["family"]),
                "pair_count": pair_count,
                "stream_count": pair_count * 2,
                "d_seg": float(point["d_seg"]),
                "d_pose": float(point["d_pose"]),
                "measured_existing_coders": {
                    "brotli_q11": {
                        "mean_bytes_per_pair": brotli_mean,
                        "total_bytes_all_streams": round(brotli_mean * pair_count),
                        "parseback_exact": True,
                    },
                    "zstd_19": {
                        "mean_bytes_per_pair": zstd_mean,
                        "total_bytes_all_streams": round(zstd_mean * pair_count),
                        "parseback_exact": True,
                    },
                },
                "zstd_over_brotli_ratio": zstd_mean / brotli_mean,
                "unmeasured_requested_coders": {
                    "lzma_xz_preset9": exact_payload_blocker,
                    "constriction_spatial_context_arithmetic": exact_payload_blocker,
                    "zigzag_rle_arithmetic": exact_payload_blocker,
                },
                "best_complete_measured_coder": "brotli_q11" if brotli_mean <= zstd_mean else "zstd_19",
                "authority_label": "MEASURED_SETTLED_N24_ROWS_EXISTING_CODERS_ONLY",
            }
        )
    return {
        "curve": _path_receipt(path),
        "source_receipts": source_receipts,
        "points": points,
        "requested_coder_measurement_status": "BLOCKED_EXACT_PAYLOAD_BYTES_NOT_PRESERVED",
        "waterfill_after_complete_measured_coder_choice": document.get("waterfill"),
        "waterfill_coder": "brotli_q11",
        "waterfill_reuse_basis": (
            "Brotli is smaller than zstd at every settled point; imported waterfill is the already "
            "measured Brotli n600-equivalent result. Novel-coder waterfill is forbidden until rederivation."
        ),
        "verdict_scope": (
            "n24 settled range-payload receipts only; missing exact residual bytes block novel-coder ratios, "
            "not the existing Brotli/zstd rows"
        ),
    }


def _pdw1_measurement(gt_cache: Path, segnet_head: Path, prior_receipt: Path | None) -> dict[str, Any]:
    cache_receipt = _path_receipt(gt_cache)
    head_receipt = _path_receipt(segnet_head)
    receipt = initialize_video_fed_target(gt_cache, segnet_head)
    payload = encode_pdw1(receipt.target)
    decoded = decode_pdw1(payload)
    if encode_pdw1(decoded) != payload:
        raise MeasurementError("PDW1 strict decode/re-encode identity failed")
    digest = hashlib.sha256(payload).hexdigest()
    if len(payload) != EXPECTED_PDW1_BYTES:
        raise MeasurementError(f"authoritative n600 PDW1 must be exactly 338 bytes, got {len(payload)}")
    if digest != EXPECTED_PDW1_SHA256:
        raise MeasurementError(f"authoritative n600 PDW1 SHA-256 mismatch: {digest}")
    prior_check: dict[str, Any] | None = None
    if prior_receipt is not None:
        prior = _load_json(prior_receipt)
        prior_hex = prior.get("generator", {}).get("pdw1_hex") if isinstance(prior, dict) else None
        prior_digest = None
        if isinstance(prior_hex, str):
            try:
                prior_digest = hashlib.sha256(bytes.fromhex(prior_hex)).hexdigest()
            except ValueError as exc:
                raise MeasurementError("prior PDW1 receipt has invalid generator.pdw1_hex") from exc
        prior_check = {
            "receipt": _path_receipt(prior_receipt),
            "receipt_payload_sha256": prior_digest,
            "matches_rederived": prior_digest == digest,
        }
        if prior_digest is not None and prior_digest != digest:
            raise MeasurementError("prior PDW1 receipt bytes do not match authoritative rederivation")
    return {
        "authority_label": "MEASURED_REDERIVED_PDW1_EXACTLY_338_BYTES",
        "bytes": len(payload),
        "sha256": digest,
        "strict_decode_reencode_identical": True,
        "cache": cache_receipt,
        "segnet_head": head_receipt,
        "selected_partition_sha256": receipt.selected_partition_sha256,
        "selected_shape": list(receipt.selected_shape),
        "selected_dtype": receipt.selected_dtype,
        "selected_partitions": receipt.selected_partitions,
        "active_classes": list(receipt.active_classes),
        "class_counts": list(receipt.class_counts),
        "adjacency": [list(edge) for edge in receipt.adjacency],
        "frozen_head_sha256": receipt.frozen_head_sha256,
        "coders": measure_byte_ladder(payload),
        "prior_receipt_check": prior_check,
        "score_authority": False,
    }


def _flatten_mapping(value: Mapping[str, Any], prefix: str = "") -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for raw_name, raw_value in value.items():
        name = f"{prefix}.{raw_name}" if prefix else str(raw_name)
        if isinstance(raw_value, Mapping):
            output.update(_flatten_mapping(raw_value, name))
            continue
        if hasattr(raw_value, "detach") and hasattr(raw_value, "cpu"):
            raw_value = raw_value.detach().cpu().numpy()
        if isinstance(raw_value, np.ndarray):
            output[name] = np.asarray(raw_value)
    return output


def _load_donor(path: Path, ema_prefix: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    source = _path_receipt(path)
    suffix = path.suffix.lower()
    ema_identity: str
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            tensors = {name: np.asarray(archive[name]) for name in sorted(archive.files)}
        ema_identity = "filename_explicit_ema" if "ema" in path.name.lower() else "not_identified"
    elif suffix == ".npy":
        tensors = {path.stem: np.asarray(np.load(path, mmap_mode="r", allow_pickle=False))}
        ema_identity = "filename_explicit_ema" if "ema" in path.name.lower() else "not_identified"
    elif suffix in {".pt", ".pth", ".ckpt"}:
        try:
            import torch
        except ImportError as exc:
            raise MeasurementError("Torch donor checkpoint requires torch") from exc
        try:
            loaded = torch.load(path, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, ValueError) as exc:
            raise MeasurementError(f"cannot load donor checkpoint read-only: {exc}") from exc
        if not isinstance(loaded, Mapping):
            raise MeasurementError("donor checkpoint root must be a mapping")
        exact_ema_keys = ("ema_state_dict", "model_ema", "ema", "ema_shadow")
        selected_key = next((key for key in exact_ema_keys if isinstance(loaded.get(key), Mapping)), None)
        if selected_key is not None:
            tensors = _flatten_mapping(loaded[selected_key])
            ema_identity = f"explicit_mapping:{selected_key}"
        else:
            flattened = _flatten_mapping(loaded)
            prefix = ema_prefix.rstrip(".") + "."
            selected = {name[len(prefix) :]: value for name, value in flattened.items() if name.startswith(prefix)}
            tensors = selected or flattened
            ema_identity = f"explicit_prefix:{ema_prefix}" if selected else "not_identified"
    else:
        raise MeasurementError("donor checkpoint must be .npz, .npy, .pt, .pth, or .ckpt")
    tensors = {
        name: value
        for name, value in tensors.items()
        if not name.startswith("__") and value.dtype.kind in "ifu" and value.ndim > 0
    }
    if not tensors:
        raise MeasurementError("donor contains no numeric non-scalar tensors")
    if ema_identity == "not_identified":
        raise MeasurementError(
            "donor EMA section is not identifiable; use an EMA-named NPZ or an explicit EMA mapping/prefix"
        )
    return tensors, {**source, "ema_identity": ema_identity, "tensor_count": len(tensors)}


def _quantize_int8(value: np.ndarray) -> tuple[np.ndarray, dict[str, Any], int]:
    array = np.asarray(value)
    if array.dtype == np.dtype("int8"):
        return np.ascontiguousarray(array), {"mode": "already_int8_exact", "scale": None}, 1
    if array.dtype.kind != "f" or not np.isfinite(array).all():
        raise MeasurementError("donor tensor must be finite floating point or exact int8")
    maximum = float(np.max(np.abs(array), initial=0.0))
    scale = np.float32(maximum / 127.0 if maximum else 1.0)
    quantized = np.rint(array.astype(np.float64) / float(scale)).clip(-127, 127).astype(np.int8)
    metadata = {
        "mode": "deterministic_symmetric_per_tensor_int8",
        "scale": float(scale),
        "zero_point": 0,
        "reconstruction_mse": float(np.mean(np.square(quantized.astype(np.float64) * float(scale) - array))),
        "matched_realized_dseg": "UNMEASURED_WEIGHT_DOMAIN_ONLY",
    }
    return np.ascontiguousarray(quantized), metadata, 5  # mode byte + float32 scale


def _section_for_name(name: str) -> str:
    lowered = name.lower()
    return "code" if any(token in lowered for token in SECTION_CODE_TOKENS) else "base_weight"


def _spatial_view(array: np.ndarray, layout: str) -> tuple[np.ndarray | None, str | None, int]:
    if layout == "oihw-to-hwio" and array.ndim == 4:
        return np.ascontiguousarray(array.transpose(2, 3, 1, 0)), "OIHW_TO_HWIO", 1
    if layout == "oihw-to-hwio" and array.ndim == 2:
        return np.ascontiguousarray(array[:, :, None]), "OI_MATRIX_TO_HW1", 1
    if layout == "oihw-to-hwio" and array.ndim == 1:
        return np.ascontiguousarray(array[None, :, None]), "VECTOR_TO_1xW1", 1
    if layout == "trailing-hwc" and array.ndim >= 3:
        return array, "TRAILING_HWC_IDENTITY", 1
    if layout == "trailing-hwc" and array.ndim == 2:
        return np.ascontiguousarray(array[:, :, None]), "MATRIX_TO_HW1", 1
    if layout == "trailing-hwc" and array.ndim == 1:
        return np.ascontiguousarray(array[None, :, None]), "VECTOR_TO_1xW1", 1
    return None, None, 0


def _measure_donor_section(
    section: str,
    tensors: Sequence[tuple[str, np.ndarray]],
    *,
    spatial_layout: str,
    block_sizes: Sequence[int],
    block_thresholds: Sequence[float],
) -> dict[str, Any]:
    per_tensor: list[dict[str, Any]] = []
    iid_ladders: list[dict[str, dict[str, Any]]] = []
    spatial_ladders: list[dict[str, dict[str, Any]]] = []
    metadata_bytes_total = 0
    block_totals: dict[str, dict[str, Any]] = {}
    for name, original in tensors:
        qint, quantization, quant_metadata_bytes = _quantize_int8(original)
        name_bytes = name.encode("utf-8")
        if len(name_bytes) > 0xFFFF:
            raise MeasurementError(f"donor tensor name is too long: {name[:80]}")
        tensor_metadata_bytes = 2 + len(name_bytes) + quant_metadata_bytes
        metadata_bytes_total += tensor_metadata_bytes
        iid_ladder = measure_iid_signed_array_ladder(qint)
        iid_ladders.append(iid_ladder)
        spatial, transform, transform_bytes = _spatial_view(qint, spatial_layout)
        spatial_ladder = measure_signed_array_ladder(spatial) if spatial is not None else None
        if spatial_ladder is not None:
            spatial_ladders.append(spatial_ladder)
        block_rows = []
        if original.dtype.kind == "f":
            for block_size in block_sizes:
                for threshold in block_thresholds:
                    row = measure_block_fp(original, block_size=block_size, clip_threshold=threshold)
                    key = f"block{block_size}_threshold{threshold:g}"
                    accumulator = block_totals.setdefault(
                        key,
                        {
                            "framed_bytes": 0,
                            "qint_bytes": 0,
                            "exponent_bytes": 0,
                            "header_bytes": 0,
                            "tensor_count": 0,
                            "packed_byte_coders": {},
                        },
                    )
                    accumulator["framed_bytes"] += row["framed_bytes"] + 2 + len(name_bytes)
                    accumulator["qint_bytes"] += row["byte_accounting"]["qint_bytes"]
                    accumulator["exponent_bytes"] += row["byte_accounting"]["exponent_bytes"]
                    accumulator["header_bytes"] += row["byte_accounting"]["header_bytes"] + 2 + len(name_bytes)
                    accumulator["tensor_count"] += 1
                    for coder, coder_row in row["packed_byte_coders"].items():
                        coder_total = accumulator["packed_byte_coders"].setdefault(
                            coder,
                            {
                                "available": True,
                                "framed_plus_tensor_name_bytes": 0,
                                "all_parseback_exact": True,
                                "unavailable_reasons": [],
                            },
                        )
                        if not coder_row.get("available", True):
                            coder_total["available"] = False
                            coder_total["unavailable_reasons"].append(coder_row.get("unavailable_reason"))
                            continue
                        coder_total["framed_plus_tensor_name_bytes"] += (
                            int(coder_row["framed_bytes"]) + 2 + len(name_bytes)
                        )
                        coder_total["all_parseback_exact"] = bool(
                            coder_total["all_parseback_exact"] and coder_row["parseback_exact"]
                        )
                    block_rows.append({"config": key, "measurement": row})
        per_tensor.append(
            {
                "name": name,
                "source_shape": list(original.shape),
                "source_dtype": original.dtype.str,
                "int8_shape": list(qint.shape),
                "int8_sha256": hashlib.sha256(qint.tobytes()).hexdigest(),
                "quantization": quantization,
                "per_tensor_metadata_bytes": tensor_metadata_bytes,
                "iid_and_byte_coders": iid_ladder,
                "spatial_transform": transform,
                "spatial_transform_metadata_bytes": transform_bytes,
                "spatial_context_coders": spatial_ladder,
                "spatial_context_blocker": None
                if spatial_ladder is not None
                else f"unsupported tensor/layout for true [...,H,W,C] context: ndim={qint.ndim} layout={spatial_layout}",
                "block_fp": block_rows,
            }
        )
    aggregate = _aggregate_ladders(iid_ladders)
    for row in aggregate.values():
        if row["available"]:
            row["framed_plus_section_metadata_bytes"] = row["framed_bytes"] + metadata_bytes_total
    context_complete = len(spatial_ladders) == len(tensors)
    context_aggregate = _aggregate_ladders(spatial_ladders) if spatial_ladders else {}
    for row in context_aggregate.values():
        if row["available"]:
            row["eligible_framed_plus_metadata_bytes"] = row["framed_bytes"] + metadata_bytes_total
        row["full_section_total_authority"] = context_complete
    element_count = sum(int(value.size) for _, value in tensors)
    for accumulator in block_totals.values():
        accumulator["uncompressed_bits_per_parameter"] = 8.0 * accumulator["framed_bytes"] / element_count
        for coder_total in accumulator["packed_byte_coders"].values():
            coder_total["bits_per_parameter"] = (
                8.0 * coder_total["framed_plus_tensor_name_bytes"] / element_count
                if coder_total["available"]
                else None
            )
    return {
        "section": section,
        "tensor_count": len(tensors),
        "elements": element_count,
        "section_metadata_bytes": metadata_bytes_total,
        "iid_and_byte_coders": aggregate,
        "spatial_context": {
            "complete_tensor_coverage": context_complete,
            "eligible_tensor_count": len(spatial_ladders),
            "layout_contract": spatial_layout,
            "coders": context_aggregate,
            "blocker": None
            if context_complete
            else "FULL_SECTION_CONTEXT_TOTAL_BLOCKED_UNSUPPORTED_SHAPES_NO_FLATTENED_SEQUENCE_AS_SPATIAL",
        },
        "block_fp": {
            "configs": block_totals,
            "matched_realized_dseg": "OWED_N_GE_24_UNLESS_ACTUAL_RESULTS_SUPPLIED",
        },
        "per_tensor": per_tensor,
    }


def _actual_block_fp_results(paths: Sequence[Path], donor_sha256: str | None) -> dict[str, Any]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for path in paths:
        payload = _load_json(path)
        source = _path_receipt(path)
        if not isinstance(payload, Mapping):
            rejected.append({"source": source, "reason": "JSON root is not an object"})
            continue
        n_pairs = payload.get("n_pairs")
        dseg = payload.get("matched_realized_dseg")
        supplied_donor_sha = payload.get("donor_checkpoint_sha256")
        if (
            not isinstance(n_pairs, int)
            or n_pairs < 24
            or not isinstance(dseg, (int, float))
            or not np.isfinite(dseg)
            or donor_sha256 is None
            or supplied_donor_sha != donor_sha256
        ):
            rejected.append(
                {
                    "source": source,
                    "reason": (
                        "requires integer n_pairs>=24, finite matched_realized_dseg, and exact "
                        "donor_checkpoint_sha256 matching this invocation"
                    ),
                }
            )
            continue
        accepted.append(
            {"source": source, "n_pairs": n_pairs, "matched_realized_dseg": float(dseg), "payload": payload}
        )
    return {
        "accepted": accepted,
        "rejected": rejected,
        "owed": not bool(accepted),
        "status": "MEASURED_INPUTS_SUPPLIED" if accepted else "OWED_MATCHED_REALIZED_DSEG_N_GE_24",
    }


def _measure_donor(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.donor_checkpoint is None:
        return None
    try:
        tensors, source = _load_donor(args.donor_checkpoint, args.donor_ema_prefix)
    except MeasurementError as exc:
        return {
            "available": False,
            "blocker": str(exc),
            "source": _path_receipt(args.donor_checkpoint),
            "historical_comparison": HISTORICAL_SELFCOMP_ROW,
        }
    sections = {
        section: [(name, value) for name, value in tensors.items() if _section_for_name(name) == section]
        for section in ("base_weight", "code")
    }
    measured = {
        section: _measure_donor_section(
            section,
            values,
            spatial_layout=args.donor_spatial_layout,
            block_sizes=args.block_fp_block_size,
            block_thresholds=args.block_fp_threshold,
        )
        for section, values in sections.items()
        if values
    }
    combined_block_fp: dict[str, Any] | None = None
    if set(measured) == {"base_weight", "code"}:
        total_elements = sum(int(section["elements"]) for section in measured.values())
        config_names = set.intersection(
            *(set(section["block_fp"]["configs"]) for section in measured.values())
        )
        combined_configs: dict[str, Any] = {}
        candidates: list[tuple[float, str, str, float]] = []
        for config_name in sorted(config_names):
            coder_names = set.intersection(
                *(
                    set(section["block_fp"]["configs"][config_name]["packed_byte_coders"])
                    for section in measured.values()
                )
            )
            coder_rows: dict[str, Any] = {}
            for coder in sorted(coder_names):
                rows = [section["block_fp"]["configs"][config_name]["packed_byte_coders"][coder] for section in measured.values()]
                available = all(bool(row["available"]) for row in rows)
                total_bytes = (
                    sum(int(row["framed_plus_tensor_name_bytes"]) for row in rows) if available else None
                )
                bits_per_parameter = 8.0 * total_bytes / total_elements if total_bytes is not None else None
                coder_rows[coder] = {
                    "available": available,
                    "framed_plus_tensor_name_bytes": total_bytes,
                    "bits_per_parameter": bits_per_parameter,
                    "all_parseback_exact": available and all(bool(row["all_parseback_exact"]) for row in rows),
                }
                if bits_per_parameter is not None and coder != "raw":
                    candidates.append((abs(bits_per_parameter - 1.017), config_name, coder, bits_per_parameter))
            combined_configs[config_name] = {"packed_byte_coders": coder_rows}
        closest = min(candidates) if candidates else None
        combined_block_fp = {
            "authority_label": "MEASURED_CLASSICAL_BLOCK_FP_RATE_DOMAIN_ONLY",
            "total_elements": total_elements,
            "target_bits_per_parameter": 1.017,
            "configs": combined_configs,
            "closest_to_target": None
            if closest is None
            else {
                "config": closest[1],
                "coder": closest[2],
                "bits_per_parameter": closest[3],
                "absolute_target_gap": closest[0],
            },
            "matched_realized_dseg": "OWED_MATCHED_REALIZED_DSEG_N_GE_24",
            "learned_method_authority": False,
            "method_scope": "classical ternary shared-exponent block-FP plus general-purpose byte coder",
        }
    return {
        "available": True,
        "authority_label": "MEASURED_CURRENT_READ_ONLY_DONOR_LOCAL_CODER_BYTES_NOT_SCORE",
        "source": source,
        "section_classifier": {"code_tokens": list(SECTION_CODE_TOKENS), "fallback": "base_weight"},
        "sections": measured,
        "classical_block_fp_combined": combined_block_fp,
        "missing_sections": [section for section, values in sections.items() if not values],
        "historical_comparison": HISTORICAL_SELFCOMP_ROW,
        "prior_donor_section_rows": PRIOR_DONOR_SECTION_ROWS,
        "same_checkpoint_equivalence": False,
        "historical_current_delta": "FORBIDDEN_NON_EQUIVALENT_CHECKPOINTS",
    }


def _load_measured_curve(path: Path, family: str) -> list[dict[str, float]]:
    payload = _load_json(path)
    if not isinstance(payload, Mapping) or "MEASURED" not in str(payload.get("authority_label", "")):
        raise MeasurementError(f"{family} curve must be an object with a MEASURED authority_label")
    points = payload.get("points")
    if not isinstance(points, list) or len(points) < 2:
        raise MeasurementError(f"{family} curve requires at least two measured points")
    output: list[dict[str, float]] = []
    for index, point in enumerate(points):
        if not isinstance(point, Mapping) or "bytes" not in point or "distortion" not in point:
            raise MeasurementError(f"{family} curve point {index} lacks bytes/distortion")
        byte_value = float(point["bytes"])
        distortion = float(point["distortion"])
        if not np.isfinite(byte_value) or not np.isfinite(distortion) or byte_value < 0 or distortion < 0:
            raise MeasurementError(f"{family} curve point {index} is non-finite or negative")
        output.append({"bytes": byte_value, "distortion": distortion})
    return output


def _waterfill(seg_path: Path | None, pose_path: Path | None) -> dict[str, Any]:
    if seg_path is None or pose_path is None:
        return {
            "status": "BLOCKED_MISSING_REQUIRED_MEASURED_INPUTS",
            "blocker": "solve_measured_waterfill requires both --seg-curve-json and --pose-curve-json",
            "synthetic_scores_used": False,
        }
    try:
        from tac.optimization.joint_seg_pose_rate import solve_measured_waterfill

        signature = inspect.signature(solve_measured_waterfill)
        if tuple(signature.parameters) != ("seg_curve", "pose_curve"):
            raise MeasurementError(f"unexpected solve_measured_waterfill API: {signature}")
        seg_curve = _load_measured_curve(seg_path, "Seg")
        pose_curve = _load_measured_curve(pose_path, "Pose")
        result = solve_measured_waterfill(seg_curve, pose_curve)
    except (ImportError, MeasurementError, TypeError, ValueError) as exc:
        return {
            "status": "BLOCKED_EXACT_API_OR_MEASURED_INPUT_VALIDATION",
            "blocker": str(exc),
            "synthetic_scores_used": False,
        }
    return {
        "status": "INVOKED_EXACT_API_WITH_REQUIRED_MEASURED_INPUTS",
        "seg_curve_source": _path_receipt(seg_path),
        "pose_curve_source": _path_receipt(pose_path),
        "result": result,
        "synthetic_scores_used": False,
    }


def _routing_table() -> dict[str, Any]:
    return {
        "#539": {
            "route": "PDW1 frozen-head channel-target and active-facet diagnostics",
            "authority": "338-byte PDW1 measured only after gt_n600+SegNet strict rederivation",
            "blocker": "spatial pullback/receiver remains outside target bytes",
        },
        "#553": {
            "route": "PDW2 gauge-fixed margin-preserving format probe",
            "authority": "DERIVED_ONLY_NO_STRICT_ENCODER",
            "bytes": 138,
            "coded_row": None,
        },
        "#386": {
            "route": "existing per-class carrier/residual and head-offset kit",
            "authority": "separate governed surface; no PDW or checkpoint equivalence inferred",
        },
        "proposed_#553_accounting_amendment": {
            "text": (
                "Any future PDW2 entropy row must count strict PDW2 framing, gauge/reference metadata, "
                "all model tables, termination, and decoder/source dependency bytes; it must prove "
                "encode->fresh decode->encode identity and the five-condition near-tie gate before "
                "changing DERIVED_ONLY_NO_STRICT_ENCODER."
            ),
            "current_138B_status": "construction layout only; no fake entropy-coded byte claim",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-cache", type=Path, required=True, help="exact gt_n600.npz")
    parser.add_argument("--segnet-head", type=Path, required=True, help="frozen SegNet safetensors file")
    parser.add_argument("--pdw1-receipt", type=Path, help="optional prior receipt cross-check; never byte authority")
    parser.add_argument(
        "--residual-op-point",
        action="append",
        type=_parse_named_path,
        default=[],
        metavar="NAME=PATH",
        help="exact signed residual .npy/.npz; repeat for operating points",
    )
    parser.add_argument(
        "--seg-secant-curve",
        type=Path,
        help="settled composed n24 curve; imports Brotli/zstd rows and fails closed on absent payload bytes",
    )
    parser.add_argument("--donor-checkpoint", type=Path, help="read-only EMA checkpoint")
    parser.add_argument("--donor-ema-prefix", default="ema", help="EMA prefix fallback for mapping checkpoints")
    parser.add_argument(
        "--donor-spatial-layout",
        choices=("oihw-to-hwio", "trailing-hwc"),
        default="oihw-to-hwio",
        help="explicit true-spatial interpretation; unsupported tensors block full context totals",
    )
    parser.add_argument("--block-fp-block-size", type=int, action="append", default=[])
    parser.add_argument("--block-fp-threshold", type=float, action="append", default=[])
    parser.add_argument(
        "--block-fp-actual-result",
        type=Path,
        action="append",
        default=[],
        help="JSON with actual n_pairs>=24 and matched_realized_dseg",
    )
    parser.add_argument("--seg-curve-json", type=Path, help="MEASURED Seg curve for exact waterfill API")
    parser.add_argument("--pose-curve-json", type=Path, help="MEASURED Pose curve for exact waterfill API")
    parser.add_argument("--output", type=Path, required=True, help="write-once machine-readable JSON receipt")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.block_fp_block_size:
        args.block_fp_block_size = [16]
    if not args.block_fp_threshold:
        args.block_fp_threshold = [0.5]
    if any(value <= 0 for value in args.block_fp_block_size):
        parser.error("--block-fp-block-size values must be > 0")
    if any(not np.isfinite(value) or value <= 0 for value in args.block_fp_threshold):
        parser.error("--block-fp-threshold values must be finite and > 0")
    op_names = [name for name, _ in args.residual_op_point]
    if len(op_names) != len(set(op_names)):
        parser.error("--residual-op-point names must be unique")
    try:
        pdw1 = _pdw1_measurement(args.gt_cache, args.segnet_head, args.pdw1_receipt)
        residuals = [_measure_residual_op(name, path) for name, path in args.residual_op_point]
        settled_residual_curve = _settled_seg_secant_curve(args.seg_secant_curve)
        donor = _measure_donor(args)
        donor_sha256 = (
            str(donor["source"]["sha256"])
            if donor is not None
            and donor.get("available")
            and isinstance(donor.get("source"), Mapping)
            and donor["source"].get("sha256")
            else None
        )
        block_actual = _actual_block_fp_results(args.block_fp_actual_result, donor_sha256)
        if donor is not None and donor.get("available"):
            for section in donor["sections"].values():
                section["block_fp"]["matched_realized_dseg"] = block_actual["status"]
        manifest = {
            "schema": SCHEMA,
            "authority": authority_labels(),
            "argv": [sys.executable, *sys.argv] if argv is None else [sys.executable, *argv],
            "argv_shell_escaped": shlex.join([sys.executable, *(sys.argv if argv is None else argv)]),
            "writes": {"explicit_output_json_only": str(args.output.resolve()), "bulky_outputs": False},
            "implementation_custody": {
                "git_head_at_measurement": _git_value("rev-parse", "HEAD"),
                "git_branch_at_measurement": _git_value("branch", "--show-current"),
                "git_worktree_dirty_at_measurement": bool(_git_value("status", "--porcelain")),
                "tool_sha256": _sha256_file(Path(__file__).resolve()),
                "coder_module_sha256": _sha256_file(
                    Path(__file__).resolve().parents[1]
                    / "src/tac/optimization/arith_selfcomp_rate_coders.py"
                ),
                "block_fp_module_sha256": _sha256_file(
                    Path(__file__).resolve().parents[1] / "src/tac/block_fp_codec.py"
                ),
                "python": sys.version,
                "platform": platform.platform(),
                "rng_seed": None,
                "rng_reason": "deterministic codecs and fixed inputs use no random sampling",
            },
            "pdw1": pdw1,
            "pdw2": {
                "authority_label": "DERIVED_ONLY_NO_STRICT_ENCODER",
                "construction_bytes": 138,
                "strict_encoder_present": False,
                "coded_row": None,
                "blocker": "NO_STRICT_PDW2_ENCODER_DECODER_PARSEBACK; ENTROPY BYTE CLAIM FORBIDDEN",
            },
            "residual_operating_points": residuals,
            "settled_seg_secant_curve": settled_residual_curve,
            "donor": donor,
            "block_fp_matched_realized_dseg": block_actual,
            "waterfill": _waterfill(args.seg_curve_json, args.pose_curve_json),
            "routing": _routing_table(),
            "claims": {
                "score": False,
                "contest_axis": None,
                "dispatch_attempted": False,
                "pointer_mutated": False,
                "donor_checkpoint_mutated": False,
                "synthetic_scores_used": False,
            },
        }
        _write_output(args.output, manifest)
    except (MeasurementError, RateCoderError, OSError, ValueError) as exc:
        parser.exit(2, f"measurement refused: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
