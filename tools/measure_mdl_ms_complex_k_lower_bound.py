#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Audit concrete code lengths for the frozen n600 digital cell complex.

This is a deterministic, cached-input, local-CPU measurement.  It does not
train, render, call a scorer, dispatch work, or inspect a live run.  Concrete
description lengths are reported only as declared-code-family upper bounds on
individual Kolmogorov complexity.  The theorem orientation is fail-closed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
import zipfile
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tac.boundary_math.context_partition_codec import (  # noqa: E402
    decode_partition_stack,
    encode_partition_stack,
)

EXPECTED_CACHE_SHA256 = "cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6"
EXPECTED_LSTARS_SEMANTIC_U8_SHA256 = (
    "f2c8be94774780bda718adf337900403a8533b6ffa1352b5aae19e200a005557"
)
CLASS_ORDER = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
EXPECTED_LSTARS_SHAPE = (600, 384, 512)
EXPECTED_GT_POSES_SHAPE = (600, 6)
RATE_DENOMINATOR_BYTES = 37_545_489
RATE_NUMERATOR_MULTIPLIER = 25
STRICT_RATE_CEILING = Decimal("0.15")
PREFIX_ROUNDTRIP_FRAMES = 2

EXPECTED_CONTOUR_STREAM_BYTES = 457_528
EXPECTED_SHARED_EDGE_ESTIMATE_BYTES = 228_764
INHERITED_PALETTE_CHARGE_BYTES = 15
EXPECTED_XI_PAYLOAD_BYTES = 6_634
EXPECTED_XI_SECTION_BYTES = 7_195
EXPECTED_XI_Q_LEVELS = 4_096
EXPECTED_XI_D_POSE = Decimal("0.0016095471538913576")

getcontext().prec = 40


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of a file without materializing it."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one small, deterministic JSON receipt by sibling-file replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def stored_npy_memmap(npz_path: Path, key: str) -> np.memmap:
    """Memory-map one ZIP_STORED ``.npy`` member without inflating the NPZ."""
    member = key if key.endswith(".npy") else f"{key}.npy"
    with zipfile.ZipFile(npz_path) as archive:
        info = archive.getinfo(member)
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(
                f"{npz_path}:{member} is compressed; read-only memmap unavailable"
            )
        local_header = int(info.header_offset)
    with npz_path.open("rb") as handle:
        handle.seek(local_header)
        header = handle.read(30)
        if len(header) != 30:
            raise ValueError(f"truncated local ZIP header for {npz_path}:{member}")
        fields = struct.unpack("<IHHHHHIIIHH", header)
        if fields[0] != 0x04034B50:
            raise ValueError(f"bad local ZIP header for {npz_path}:{member}")
        npy_start = local_header + 30 + int(fields[-2]) + int(fields[-1])
        handle.seek(npy_start)
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            shape, fortran, dtype = np.lib.format._read_array_header(handle, version)
        data_offset = handle.tell()
    return np.memmap(
        npz_path,
        dtype=dtype,
        mode="r",
        offset=data_offset,
        shape=shape,
        order="F" if fortran else "C",
    )


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def canonical_little_endian_dtype(dtype: np.dtype[Any]) -> np.dtype[Any]:
    """Preserve dtype width/kind while fixing canonical little-endian order."""
    normalized = np.dtype(dtype)
    if normalized.byteorder == "|":
        return normalized
    return normalized.newbyteorder("<")


def audit_frozen_targets(
    lstars: np.memmap, gt_poses: np.memmap
) -> tuple[dict[str, Any], list[np.ndarray]]:
    """Validate and hash the complete frozen targets in bounded frame chunks."""
    if tuple(lstars.shape) != EXPECTED_LSTARS_SHAPE:
        raise ValueError(f"lstars shape {lstars.shape} != {EXPECTED_LSTARS_SHAPE}")
    if tuple(gt_poses.shape) != EXPECTED_GT_POSES_SHAPE:
        raise ValueError(f"gt_poses shape {gt_poses.shape} != {EXPECTED_GT_POSES_SHAPE}")
    if not np.issubdtype(lstars.dtype, np.integer):
        raise TypeError(f"lstars dtype {lstars.dtype} is not an integer dtype")
    if not np.issubdtype(gt_poses.dtype, np.floating):
        raise TypeError(f"gt_poses dtype {gt_poses.dtype} is not a floating dtype")
    if CLASS_ORDER != ("Road", "Lane", "Undrivable", "Movable", "MyCar"):
        raise RuntimeError("canonical class-order constant changed")

    lstars_dtype = canonical_little_endian_dtype(lstars.dtype)
    gt_poses_dtype = canonical_little_endian_dtype(gt_poses.dtype)
    lstars_storage_hash = hashlib.sha256()
    lstars_semantic_hash = hashlib.sha256()
    poses_hash = hashlib.sha256()
    counts = np.zeros(len(CLASS_ORDER), dtype=np.int64)
    minimum = len(CLASS_ORDER)
    maximum = -1
    frames: list[np.ndarray] = []

    for index in range(EXPECTED_LSTARS_SHAPE[0]):
        frame = np.asarray(lstars[index])
        frame_min = int(frame.min())
        frame_max = int(frame.max())
        minimum = min(minimum, frame_min)
        maximum = max(maximum, frame_max)
        if frame_min < 0 or frame_max >= len(CLASS_ORDER):
            raise ValueError(
                f"lstars[{index}] label range [{frame_min},{frame_max}] is outside [0,4]"
            )
        counts += np.bincount(frame.reshape(-1), minlength=len(CLASS_ORDER))[
            : len(CLASS_ORDER)
        ]
        canonical_frame = np.ascontiguousarray(frame, dtype=lstars_dtype)
        lstars_storage_hash.update(canonical_frame.tobytes(order="C"))
        lstars_semantic_hash.update(
            np.ascontiguousarray(frame, dtype=np.uint8).tobytes(order="C")
        )
        frames.append(frame)

    canonical_poses = np.ascontiguousarray(gt_poses, dtype=gt_poses_dtype)
    if not np.isfinite(canonical_poses).all():
        raise ValueError("gt_poses contains non-finite values")
    poses_hash.update(canonical_poses.tobytes(order="C"))
    expected_pixels = int(np.prod(EXPECTED_LSTARS_SHAPE, dtype=np.int64))
    if int(counts.sum()) != expected_pixels:
        raise RuntimeError(f"class-count sum {counts.sum()} != {expected_pixels}")
    if minimum != 0 or maximum != 4:
        raise ValueError(f"complete lstars range [{minimum},{maximum}] != [0,4]")
    if np.any(counts == 0):
        raise ValueError(f"one or more canonical classes are absent: {counts.tolist()}")
    semantic_sha256 = lstars_semantic_hash.hexdigest()
    if semantic_sha256 != EXPECTED_LSTARS_SEMANTIC_U8_SHA256:
        raise ValueError(
            "lstars semantic uint8 SHA-256 "
            f"{semantic_sha256} != expected {EXPECTED_LSTARS_SEMANTIC_U8_SHA256}"
        )

    class_counts = [
        {"class_id": index, "class_name": name, "pixels": int(counts[index])}
        for index, name in enumerate(CLASS_ORDER)
    ]
    return (
        {
            "evidence_label": "MEASURED",
            "canonical_class_order": list(CLASS_ORDER),
            "canonical_class_order_evidence_label": (
                "INPUT_CONTRACT_AND_SEMANTIC_HASH_CUSTODY"
            ),
            "class_order_source": "frozen tool contract; ids are cache values 0..4",
            "lstars": {
                "shape": list(lstars.shape),
                "storage_dtype": str(lstars.dtype),
                "storage_preserving_canonical_serialization": (
                    f"raw C-order values, dtype {lstars_dtype.str}, no NPY header"
                ),
                "storage_preserving_canonical_serialization_sha256": (
                    lstars_storage_hash.hexdigest()
                ),
                "semantic_class_label_serialization": (
                    "raw C-order class ids, dtype |u1, no NPY header"
                ),
                "semantic_class_label_serialization_sha256": semantic_sha256,
                "expected_semantic_class_label_serialization_sha256": (
                    EXPECTED_LSTARS_SEMANTIC_U8_SHA256
                ),
                "semantic_sha256_matches": True,
                "minimum": minimum,
                "maximum": maximum,
                "total_pixels": expected_pixels,
                "per_class_pixel_counts": class_counts,
            },
            "gt_poses": {
                "shape": list(gt_poses.shape),
                "storage_dtype": str(gt_poses.dtype),
                "canonical_serialization": (
                    f"raw C-order values, dtype {gt_poses_dtype.str}, no NPY header"
                ),
                "canonical_serialization_sha256": poses_hash.hexdigest(),
                "semantic_custody": (
                    "cached frozen-PoseNet target coordinates; diagnostic only; not temporal xi"
                ),
            },
        },
        frames,
    )


def rate_term(byte_count: int) -> dict[str, Any]:
    """Return exact and decimal forms of the rate term for an integer byte count."""
    if byte_count < 0:
        raise ValueError("byte_count must be nonnegative")
    numerator = RATE_NUMERATOR_MULTIPLIER * int(byte_count)
    decimal = Decimal(numerator) / Decimal(RATE_DENOMINATOR_BYTES)
    return {
        "byte_count": int(byte_count),
        "exact_fraction": f"{numerator}/{RATE_DENOMINATOR_BYTES}",
        "decimal": format(decimal, ".18f"),
    }


def audit_strict_rate_ceiling() -> dict[str, Any]:
    """Derive the strict integer byte ceiling without floating-point rounding."""
    # 0.15 * D / 25 = 3D/500.  For integer B, 500B < 3D.
    continuous_numerator = 3 * RATE_DENOMINATOR_BYTES
    continuous_denominator = 500
    maximum_integer = (continuous_numerator - 1) // continuous_denominator
    if maximum_integer != 225_272:
        raise RuntimeError(f"unexpected strict byte ceiling {maximum_integer}")
    return {
        "evidence_label": "DERIVED",
        "formula": "B < (3 * 37545489) / 500",
        "continuous_ceiling_exact_fraction": (
            f"{continuous_numerator}/{continuous_denominator}"
        ),
        "continuous_ceiling_decimal": format(
            Decimal(continuous_numerator) / Decimal(continuous_denominator), ".3f"
        ),
        "largest_integer_bytes_strictly_below": maximum_integer,
        "rate_at_225272": rate_term(225_272),
        "rate_at_225273": rate_term(225_273),
    }


def audit_necessity_summary(
    path: Path, summary: dict[str, Any]
) -> dict[str, Any]:
    """Extract the exact-geometry row and fail closed on the inherited estimates."""
    try:
        row = summary["eps"]["0.0"]
        seed = row["seed"]
        contour_bytes = int(seed["brotli_q11_bytes"])
        adjusted_value = Decimal(str(seed["brotli_q11_bytes_shared_edge_adjusted"]))
        dseg_geo = Decimal(str(row["dseg_geo"]))
        dseg_real = Decimal(str(row["dseg_real"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"malformed necessity summary {path}: {exc}") from exc
    if contour_bytes != EXPECTED_CONTOUR_STREAM_BYTES:
        raise ValueError(f"contour stream {contour_bytes} != {EXPECTED_CONTOUR_STREAM_BYTES}")
    if adjusted_value != Decimal(EXPECTED_SHARED_EDGE_ESTIMATE_BYTES):
        raise ValueError(
            f"shared-edge estimate {adjusted_value} != {EXPECTED_SHARED_EDGE_ESTIMATE_BYTES}"
        )
    if dseg_geo != 0:
        raise ValueError(f"eps=0.0 dseg_geo {dseg_geo} != 0")
    if dseg_real <= 0:
        raise ValueError("eps=0.0 palette witness unexpectedly has zero realized d_seg")
    return {
        "evidence_label": "MEASURED",
        "path": str(path.resolve()),
        "file_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "eps_key": "0.0",
        "inner_contour_stream_bytes": {
            "evidence_label": "MEASURED",
            "bytes": contour_bytes,
            "description": "Brotli-q11 inner coordinate stream",
        },
        "post_brotli_shared_edge_estimate_bytes": {
            "evidence_label": "DERIVED",
            "bytes": int(adjusted_value),
            "description": "arithmetic /2 estimate; not an emitted decoder-closed code",
        },
        "dseg_geo": {"evidence_label": "MEASURED", "value": str(dseg_geo)},
        "dseg_real": {"evidence_label": "MEASURED", "value": str(dseg_real)},
    }


def audit_xi_receipt(path: Path, receipt: dict[str, Any]) -> dict[str, Any]:
    """Extract measured temporal-xi custody without conflating cached target poses."""
    carrier = receipt.get("pose_carrier")
    if not isinstance(carrier, dict):
        carrier = receipt.get("byte_close", {}).get("pose_carrier")
    confirmation = receipt.get("pose_carrier_confirmation")
    if not isinstance(carrier, dict) or not isinstance(confirmation, dict):
        raise ValueError(f"malformed xi receipt {path}: pose carrier fields missing")
    try:
        n_pairs = int(carrier["n_pairs"])
        xi_bytes = int(carrier["xi_bytes"])
        section_bytes = int(carrier["pose_carrier_section_bytes"])
        q_levels = int(carrier["xi_q_levels"])
        d_pose = Decimal(str(confirmation["d_pose_carrier_warp_f0_witness_f1"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"malformed xi receipt {path}: {exc}") from exc
    expected = (
        n_pairs == 600
        and carrier.get("active") is True
        and xi_bytes == EXPECTED_XI_PAYLOAD_BYTES
        and section_bytes == EXPECTED_XI_SECTION_BYTES
        and q_levels == EXPECTED_XI_Q_LEVELS
        and d_pose == EXPECTED_XI_D_POSE
    )
    if not expected:
        raise ValueError(
            "xi custody mismatch: "
            f"n={n_pairs}, active={carrier.get('active')}, xi={xi_bytes}, "
            f"section={section_bytes}, q={q_levels}, d_pose={d_pose}"
        )
    return {
        "evidence_label": "MEASURED",
        "path": str(path.resolve()),
        "file_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "n_pairs": n_pairs,
        "xi_payload_bytes": xi_bytes,
        "self_contained_xi_section_bytes": section_bytes,
        "xi_q_levels": q_levels,
        "realized_d_pose": str(d_pose),
        "semantic_custody": (
            "quantized temporal xi section; lossy and distinct from cached gt_poses"
        ),
    }


def measure_context_code(frames: list[np.ndarray]) -> dict[str, Any]:
    """Emit the full temporal code and prove a deterministic prefix roundtrip."""
    prefix_source = frames[:PREFIX_ROUNDTRIP_FRAMES]
    prefix_code = encode_partition_stack(
        prefix_source, n_classes=len(CLASS_ORDER), template="temporal"
    )
    prefix_decoded = decode_partition_stack(prefix_code.payload)
    prefix_ok = len(prefix_decoded) == len(prefix_source) and all(
        np.array_equal(source, decoded)
        for source, decoded in zip(prefix_source, prefix_decoded, strict=True)
    )
    if not prefix_ok:
        raise RuntimeError("deterministic prefix context-codec roundtrip failed")

    code = encode_partition_stack(frames, n_classes=len(CLASS_ORDER), template="temporal")
    header_bytes = code.total_bytes - code.model_bytes - code.stream_bytes
    if header_bytes <= 0:
        raise RuntimeError("context-codec byte split is inconsistent")
    return {
        "evidence_label": "MEASURED",
        "claim_class": (
            "DECLARED-CODE-FAMILY UPPER BOUND; exact partition payload; non-promotable"
        ),
        "codec": "context_partition_codec CPC1 temporal",
        "n_frames": code.n_frames,
        "shape": list(code.shape),
        "payload_bytes": code.total_bytes,
        "header_bytes": header_bytes,
        "model_bytes": code.model_bytes,
        "stream_bytes": code.stream_bytes,
        "bytes_per_frame": code.bytes_per_frame,
        "payload_sha256": hashlib.sha256(code.payload).hexdigest(),
        "full_payload_decode_run": False,
        "full_payload_exactness_basis": (
            "emitted lossless codec contract plus existing whole-stack codec tests; "
            "this invocation decoded only the separately encoded deterministic prefix"
        ),
        "prefix_roundtrip": {
            "evidence_label": "MEASURED",
            "frames": PREFIX_ROUNDTRIP_FRAMES,
            "payload_bytes": prefix_code.total_bytes,
            "payload_sha256": hashlib.sha256(prefix_code.payload).hexdigest(),
            "bit_exact": True,
        },
        "rate_term": rate_term(code.total_bytes),
    }


def current_git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git rev-parse HEAD failed: {result.stderr.strip()}")
    return result.stdout.strip()


def build_result(
    *,
    cache_path: Path,
    necessity_path: Path,
    xi_path: Path,
    expected_cache_sha256: str,
) -> dict[str, Any]:
    """Run the complete deterministic audit and return its JSON-ready receipt."""
    for path in (cache_path, necessity_path, xi_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    actual_cache_sha256 = sha256_file(cache_path)
    if actual_cache_sha256 != expected_cache_sha256:
        raise ValueError(
            f"cache SHA-256 {actual_cache_sha256} != expected {expected_cache_sha256}"
        )

    lstars = stored_npy_memmap(cache_path, "lstars")
    gt_poses = stored_npy_memmap(cache_path, "gt_poses")
    target_audit, frames = audit_frozen_targets(lstars, gt_poses)
    necessity = audit_necessity_summary(necessity_path, load_json(necessity_path))
    xi = audit_xi_receipt(xi_path, load_json(xi_path))
    context_code = measure_context_code(frames)
    ceiling = audit_strict_rate_ceiling()

    intended_seg_bytes = (
        EXPECTED_SHARED_EDGE_ESTIMATE_BYTES + INHERITED_PALETTE_CHARGE_BYTES
    )
    intended_total_bytes = intended_seg_bytes + EXPECTED_XI_SECTION_BYTES
    expected_intended_total = 235_974
    if intended_total_bytes != expected_intended_total:
        raise RuntimeError(f"intended model total {intended_total_bytes} != 235974")
    exact_family_total_bytes = context_code["payload_bytes"] + EXPECTED_XI_SECTION_BYTES
    strict_maximum = ceiling["largest_integer_bytes_strictly_below"]
    continuous_ceiling = Decimal(3 * RATE_DENOMINATOR_BYTES) / Decimal(500)

    return {
        "schema": "mdl_digital_cell_complex_code_length_audit_v1",
        "authority": "[macOS-CPU advisory] NON-PROMOTABLE",
        "research_only": True,
        "lane_id": "lane_mdl_ms_complex_k_lower_bound_20260718",
        "requested_claim_verdict": "FALSIFIED_AT_CLAIM_LEVEL",
        "universal_k_numeric_lower_bound": "TRIVIAL_ONLY",
        "universal_k_threshold_verdict": "INCONCLUSIVE",
        "exact_zero_distortion_receiver_closed": False,
        "score_claim": False,
        "promotion_claim": False,
        "frontier_pointer_delta": "NONE",
        "execution_allowed_or_used": False,
        "top_level_outcome_evidence_label": "DERIVED",
        "ait_contract": {
            "evidence_label": "DERIVED",
            "conditional_relation": "K_U(T | D) <= |c(T)| + O_U(1)",
            "unconditional_relation": "K_U(T) <= K_U(D) + |c(T)| + O_U(1)",
            "measurement_orientation": (
                "an emitted code length is a declared-family upper bound, never a "
                "numeric universal-complexity lower bound"
            ),
            "evaluator_target_custody": (
                "T_E=(S,P), where S is the frozen SegNet argmax partition and P is "
                "the cached frozen-PoseNet output"
            ),
            "carrier_description_object_custody": (
                "carrier_description_object=(S,xi_quantized); xi_quantized is a "
                "carrier-side description object and is not an output of evaluator E"
            ),
            "measured_object": "digital frozen-scorer argmax cell complex",
            "classical_morse_smale_claim": False,
        },
        "provenance": {
            "evidence_label": "MEASURED",
            "git_head_before_landing": current_git_head(),
            "tool_path": str(Path(__file__).resolve()),
            "tool_sha256": sha256_file(Path(__file__).resolve()),
            "execution": "cached local CPU only; no scorer/render/training/provider/live-run access",
        },
        "inputs": {
            "evidence_label": "MEASURED",
            "cache": {
                "path": str(cache_path.resolve()),
                "file_bytes": cache_path.stat().st_size,
                "sha256": actual_cache_sha256,
                "expected_sha256": expected_cache_sha256,
                "sha256_matches": True,
                "zip_member_access": "ZIP_STORED read-only memmap",
            },
            "necessity_summary": necessity,
            "xi_receipt": xi,
        },
        "frozen_target_audit": target_audit,
        "lossless_seg_declared_family": context_code,
        "intended_digital_complex_model": {
            "evidence_label": "DERIVED",
            "segmentation_split": {
                "inner_contour_stream_bytes": {
                    "evidence_label": "MEASURED",
                    "bytes": EXPECTED_CONTOUR_STREAM_BYTES,
                },
                "post_brotli_shared_edge_estimate_bytes": {
                    "evidence_label": "DERIVED",
                    "bytes": EXPECTED_SHARED_EDGE_ESTIMATE_BYTES,
                },
                "inherited_palette_cell_label_charge_bytes": {
                    "evidence_label": "INFERRED",
                    "bytes": INHERITED_PALETTE_CHARGE_BYTES,
                },
                "optimistic_seg_bytes": {
                    "evidence_label": "DERIVED",
                    "bytes": intended_seg_bytes,
                },
                "adjacency_charge": {
                    "evidence_label": "INFERRED",
                    "bytes": 0,
                    "assumption": (
                        "adjacency is derived from a fully parseable edge-incidence graph"
                    ),
                    "validated_on_current_stream": False,
                },
                "digital_junction_tie_locus_charge": {
                    "evidence_label": "INFERRED",
                    "bytes": 0,
                    "assumption": (
                        "digital junctions and tie loci are derived as curve intersections; "
                        "their precision is charged to edge seeds"
                    ),
                    "validated_on_current_stream": False,
                },
                "generic_decoder_generator_charge": {
                    "evidence_label": "INFERRED",
                    "assumption_authority": "RULE_118_CONTRACT",
                    "counted_bytes": 0,
                    "assumption": (
                        "the generic decoder and generator are uncounted rule-118 code"
                    ),
                    "validated_on_current_stream": False,
                },
                "zero_marginal_assumption_custody": (
                    "NOT VALIDATED: the current contour inner stream is neither framed nor "
                    "parse-back closed, so adjacency, junction/tie-locus, and generic-decoder "
                    "zero-marginal assumptions do not have receiver-closure evidence"
                ),
            },
            "temporal_xi_split": {
                "entropy_payload_bytes": {
                    "evidence_label": "MEASURED",
                    "bytes": EXPECTED_XI_PAYLOAD_BYTES,
                },
                "self_contained_section_bytes": {
                    "evidence_label": "MEASURED",
                    "bytes": EXPECTED_XI_SECTION_BYTES,
                },
                "realized_d_pose": {
                    "evidence_label": "MEASURED",
                    "value": str(EXPECTED_XI_D_POSE),
                },
            },
            "optimistic_total": {
                "evidence_label": "DERIVED",
                "bytes": intended_total_bytes,
                "formula": "228764 + 15 + 7195",
                "rate_term": rate_term(intended_total_bytes),
                "continuous_ceiling_gap_bytes": format(
                    Decimal(intended_total_bytes) - continuous_ceiling, ".3f"
                ),
                "integer_ceiling_gap_bytes": intended_total_bytes - strict_maximum,
                "interpretation": (
                    "above-threshold inherited code family; does not prove universal "
                    "complexity is above the threshold"
                ),
            },
            "blockers": [
                "no complete self-delimiting contour decoder or framing",
                "division after nonlinear Brotli is not an emitted shared-edge codec",
                "cached argmax cells are not proven classical Morse-Smale cells",
                "palette witness has measured nonzero realized Seg distortion",
                "temporal-xi section has measured nonzero realized Pose distortion",
            ],
            "is_archive": False,
            "is_lossless_zero_distortion_custody": False,
            "is_universal_complexity_numeric_lower_bound": False,
        },
        "declared_family_combined_with_xi": {
            "evidence_label": "DERIVED",
            "seg_payload_bytes": context_code["payload_bytes"],
            "xi_section_bytes": EXPECTED_XI_SECTION_BYTES,
            "combined_bytes": exact_family_total_bytes,
            "rate_term": rate_term(exact_family_total_bytes),
            "seg_partition_exact": True,
            "pose_zero_distortion": False,
            "receiver_closed_rgb_witness": False,
            "interpretation": (
                "declared-code-family upper bound on exact cached partition plus the "
                "existing lossy xi section; not an exact-witness archive"
            ),
        },
        "strict_rate_ceiling": ceiling,
        "target_pose_lossless_diagnostic": {
            "evidence_label": "MEASURED",
            "status": "NOT_RUN_OPTIONAL",
            "reason": "cached gt_poses are not temporal xi and are not a receiver",
        },
        "execution_guardrails": {
            "evidence_label": "MEASURED",
            "training": False,
            "scorer_call": False,
            "render": False,
            "provider_or_gpu_dispatch": False,
            "live_run_access": False,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit concrete code lengths for the cached n600 digital argmax cell complex "
            "without promoting them to universal-complexity lower bounds."
        )
    )
    parser.add_argument("--cache-npz", required=True, type=Path)
    parser.add_argument("--necessity-summary", required=True, type=Path)
    parser.add_argument("--xi-receipt", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--expected-cache-sha256",
        default=EXPECTED_CACHE_SHA256,
        help="expected frozen cache SHA-256 (fails closed on mismatch)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_path = args.out.resolve(strict=False)
    temporary_root = Path("/tmp").resolve(strict=False)
    if output_path == temporary_root or output_path.is_relative_to(temporary_root):
        raise ValueError(
            f"--out must be durable and may not resolve under {temporary_root}: {output_path}"
        )
    if len(args.expected_cache_sha256) != 64:
        raise ValueError("--expected-cache-sha256 must be 64 hexadecimal characters")
    try:
        int(args.expected_cache_sha256, 16)
    except ValueError as exc:
        raise ValueError("--expected-cache-sha256 must be hexadecimal") from exc
    result = build_result(
        cache_path=args.cache_npz,
        necessity_path=args.necessity_summary,
        xi_path=args.xi_receipt,
        expected_cache_sha256=args.expected_cache_sha256.lower(),
    )
    atomic_json(args.out, result)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "requested_claim_verdict": result["requested_claim_verdict"],
                "universal_k_threshold_verdict": result["universal_k_threshold_verdict"],
                "score_claim": result["score_claim"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
