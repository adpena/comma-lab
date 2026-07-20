#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure full shared-resize kernel closure on a SHA-pinned #49 fixture.

This is a bounded local CPU measurement. It decodes a caller-pinned real video
fixture, derives exact structural coverage, measures canonical primitive-basis
uint8 reachability, and compares the legacy #49 mask fill with coder-admitted
full-kernel affine-cell candidates. It makes no score or promotion claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
for _path in (REPO, SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.optimization.resize_full_kernel import (  # noqa: E402
    FULL_RESIZE_KERNEL_SCHEMA,
    UINT8_REACHABILITY_SEMANTICS,
    FullResizeKernel,
)

RECEIPT_SCHEMA = "resize_null_preimage_full_kernel_measurement.v1"
POINTER = "0.19108 [contest-CPU] UNMOVED"
VALID_PREFERENCES = ("constant", "horizontal", "vertical", "neighbor_mean")


class FullKernelMeasurementError(RuntimeError):
    """Fail-closed fixture, decode, or receipt-custody error."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    resolved = path.expanduser().resolve()
    if any(
        resolved == root or root in resolved.parents
        for root in (Path("/tmp"), Path("/private/tmp"), Path("/var/tmp"))
    ):
        raise FullKernelMeasurementError("receipt path must be durable, not temporary")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, resolved)
    finally:
        temporary.unlink(missing_ok=True)


def _decode_fixture(path: Path, n_frames: int) -> np.ndarray:
    try:
        import av
    except ImportError as exc:  # pragma: no cover - environment-specific fail-close
        raise FullKernelMeasurementError("PyAV is required to decode the #49 fixture") from exc

    frames: list[np.ndarray] = []
    container = av.open(str(path))
    try:
        for decoded in container.decode(video=0):
            frame = np.ascontiguousarray(decoded.to_ndarray(format="rgb24"), dtype=np.uint8)
            if frame.shape != (874, 1164, 3):
                raise FullKernelMeasurementError(
                    f"decoded fixture frame shape {frame.shape} != (874,1164,3)"
                )
            frames.append(frame)
            if len(frames) == n_frames:
                break
    finally:
        container.close()
    if len(frames) != n_frames:
        raise FullKernelMeasurementError(
            f"fixture yielded {len(frames)} frames, expected {n_frames}"
        )
    return np.stack(frames)


def _parse_preferences(value: str) -> tuple[str, ...]:
    preferences = tuple(part.strip() for part in value.split(",") if part.strip())
    if not preferences or any(part not in VALID_PREFERENCES for part in preferences):
        raise argparse.ArgumentTypeError(
            f"preferences must be comma-separated members of {VALID_PREFERENCES}"
        )
    return preferences


def measure(
    fixture: Path,
    *,
    expected_sha256: str,
    n_frames: int,
    preferences: tuple[str, ...],
    max_nodes_per_block: int,
) -> dict[str, Any]:
    fixture = fixture.expanduser().resolve()
    if not fixture.is_file():
        raise FullKernelMeasurementError(f"fixture is absent: {fixture}")
    actual_sha256 = _sha256_file(fixture)
    if actual_sha256 != expected_sha256.lower():
        raise FullKernelMeasurementError(
            f"fixture SHA-256 {actual_sha256} != expected {expected_sha256.lower()}"
        )
    frames = _decode_fixture(fixture, n_frames)
    compiler = FullResizeKernel.build()
    started = time.monotonic()
    frame_rows: list[dict[str, Any]] = []
    reachability_rows: list[dict[str, Any]] = []
    for frame_index, frame in enumerate(frames):
        reachability = compiler.uint8_reachability(frame)
        fill = compiler.compile_min_description_preimage(
            frame,
            preferences=preferences,
            max_nodes_per_block=max_nodes_per_block,
        )
        source_numerators, denominator = compiler.operator.apply_numerators(frame)
        selected_numerators, selected_denominator = compiler.operator.apply_numerators(
            fill.frame
        )
        exact = bool(
            denominator == selected_denominator
            and np.array_equal(source_numerators, selected_numerators)
        )
        if not exact:
            raise FullKernelMeasurementError(
                f"frame {frame_index} selected candidate failed numerator equality"
            )
        row = fill.to_dict()
        row.update(
            {
                "frame_index": frame_index,
                "source_frame_sha256": _sha256_array(frame),
                "selected_frame_sha256": _sha256_array(fill.frame),
                "exact_resize_numerator_equal": True,
                "resize_numerator_denominator": denominator,
                "uint8_reachability": reachability.to_dict(),
            }
        )
        frame_rows.append(row)
        reachability_rows.append(reachability.to_dict())

    def sum_bytes(field: str, coder: str) -> int:
        return sum(int(row[field][coder]) for row in frame_rows)

    byte_summary: dict[str, Any] = {}
    for coder in ("brotli", "lzma"):
        original = sum_bytes("original_bytes", coder)
        old = sum_bytes("old_mask_bytes", coder)
        selected = sum_bytes("selected_bytes", coder)
        byte_summary[coder] = {
            "original_bytes": original,
            "old_mask_bytes": old,
            "full_kernel_admitted_bytes": selected,
            "old_mask_delta_vs_original": old - original,
            "full_kernel_delta_vs_old_mask": selected - old,
            "full_kernel_delta_vs_original": selected - original,
            "full_kernel_reduction_vs_old_mask_percent": (
                100.0 * (old - selected) / old if old else 0.0
            ),
            "full_kernel_reduction_vs_original_percent": (
                100.0 * (original - selected) / original if original else 0.0
            ),
        }

    reachability_total = {
        key: sum(int(row[key]) for row in reachability_rows)
        for key in (
            "zero_weight_coordinate_directions",
            "active_tensor_directions",
            "feasible_height_col0_directions",
            "feasible_height_col1_directions",
            "feasible_width_tensor_directions",
            "feasible_active_directions",
            "full_basis_directions",
            "feasible_basis_directions_lower_bound",
        )
    }
    reachability_total["feasible_basis_fraction_lower_bound"] = (
        reachability_total["feasible_basis_directions_lower_bound"]
        / reachability_total["full_basis_directions"]
    )
    reachability_total["feasible_basis_percent_lower_bound"] = (
        100.0 * reachability_total["feasible_basis_fraction_lower_bound"]
    )
    reachability_total.update(
        {
            "semantics": UINT8_REACHABILITY_SEMANTICS,
            "is_lower_bound_on_full_bounded_lattice_intersection": True,
            "decomposition_axis": (
                "zero-weight coordinates, height-null col0, height-null col1, "
                "row-space x width-null tensor"
            ),
            "class_or_margin_stratum": (
                "not measured: this fixture-only compiler pass loads no frozen scorer "
                "labels/margins; kernel-family decomposition is supplied instead"
            ),
        }
    )

    candidate_totals: dict[str, dict[str, int | float | bool]] = {}
    for preference in preferences:
        rows = [
            candidate
            for frame in frame_rows
            for candidate in frame["candidates"]
            if candidate["preference"] == preference
        ]
        candidate_totals[preference] = {
            "exact_blocks": sum(int(row["exact_blocks"]) for row in rows),
            "fallback_blocks": sum(int(row["fallback_blocks"]) for row in rows),
            "budget_blocks": sum(int(row["budget_blocks"]) for row in rows),
            "proven_infeasible_blocks": sum(
                int(row["proven_infeasible_blocks"]) for row in rows
            ),
            "nodes_visited": sum(int(row["nodes_visited"]) for row in rows),
            "all_exact_numerator_equal": all(
                bool(row["exact_numerator_equal"]) for row in rows
            ),
            "max_float_projection_residual": max(
                float(row["max_float_projection_residual"]) for row in rows
            ),
        }

    return {
        "schema": RECEIPT_SCHEMA,
        "compiler_schema": FULL_RESIZE_KERNEL_SCHEMA,
        "captured_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "measurement_axis": f"[{platform.system()}-{platform.machine()} CPU advisory]",
        "hardware_substrate": platform.platform(),
        "fixture": {
            "path": str(fixture),
            "sha256": actual_sha256,
            "expected_sha256": expected_sha256.lower(),
            "sha256_matches": True,
            "n_frames": n_frames,
            "decoded_frames_sha256": _sha256_array(frames),
        },
        "geometry": {
            "camera_hw": [compiler.camera_h, compiler.camera_w],
            "scorer_hw": [compiler.scorer_h, compiler.scorer_w],
            "resize": "bilinear_align_corners_false_disjoint_two_tap",
        },
        "coverage": compiler.coverage().to_dict(),
        "uint8_reachability": reachability_total,
        "minimum_description": {
            "semantics": "bounded deterministic coder-admitted heuristic, not global MDL optimum",
            "preferences": list(preferences),
            "max_nodes_per_block": max_nodes_per_block,
            "selected_names": [row["selected_name"] for row in frame_rows],
            "selected_full_kernel_frames": sum(
                bool(row["selected_uses_full_kernel"]) for row in frame_rows
            ),
            "bytes": byte_summary,
            "candidate_solver_totals": candidate_totals,
        },
        "frame_rows": frame_rows,
        "provenance": {
            "tool_path": str(Path(__file__).resolve()),
            "tool_sha256": _sha256_file(Path(__file__).resolve()),
            "compiler_path": str(
                REPO / "src/tac/optimization/resize_full_kernel.py"
            ),
            "compiler_sha256": _sha256_file(
                REPO / "src/tac/optimization/resize_full_kernel.py"
            ),
            "git_head": __import__("subprocess").check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
            ).strip(),
        },
        "elapsed_seconds": time.monotonic() - started,
        "pointer": POINTER,
        "pointer_moved": False,
        "score_claim": False,
        "promotion_eligible": False,
        "research_only": True,
        "verdict_scope": "one SHA-pinned real fixture subset plus exact structural law",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--n-frames", type=int, default=1)
    parser.add_argument("--preferences", type=_parse_preferences, default=("constant",))
    parser.add_argument("--max-nodes-per-block", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.n_frames < 1 or args.n_frames > 16:
        parser.error("--n-frames must be in [1,16]")
    if args.max_nodes_per_block < 1:
        parser.error("--max-nodes-per-block must be positive")
    expected = args.expected_sha256.lower()
    if len(expected) != 64:
        parser.error("--expected-sha256 must contain 64 hex characters")
    try:
        int(expected, 16)
    except ValueError as exc:
        parser.error(f"--expected-sha256 must be hexadecimal: {exc}")
    receipt = measure(
        args.fixture,
        expected_sha256=expected,
        n_frames=args.n_frames,
        preferences=args.preferences,
        max_nodes_per_block=args.max_nodes_per_block,
    )
    _atomic_json(args.output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
