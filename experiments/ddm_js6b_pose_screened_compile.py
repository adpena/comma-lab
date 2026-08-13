#!/usr/bin/env python3
"""Pose-screen the retained JS6 proposal bank before any HP3/RC64 compile.

This arm is scorer-free.  It adapts the canonical Q3 block projector as a
diagnostic and applies the measured two-instance semantic-cell pose envelope as
the fail-closed admission bound.  Every projected array is retained.  A bank
with no survivor produces an explicit non-fire receipt and no candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
from pathlib import Path
from typing import Any, Final

import numpy as np

REPO: Final = Path(__file__).resolve().parents[1]
RUN_ID: Final = "ddm_js6b_pose_screened_compile_20260813"
DEFAULT_OUTPUT: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_js6b_pose_screened_compile_20260813"
)
DEFAULT_BANK: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_js6_seg_representation_join_20260813/"
    "proposal_bank"
)
PROPOSAL_COUNT: Final = 200
N_PAIRS: Final = 600
HEIGHT: Final = 384
WIDTH: Final = 512
DENOMINATOR_PIXELS: Final = N_PAIRS * HEIGHT * WIDTH
S_PER_FLIP: Final = 100.0 / DENOMINATOR_PIXELS
POSE_DAMAGE_LOWER_S_PER_CELL: Final = 5.7e-6
POSE_DAMAGE_UPPER_S_PER_CELL: Final = 3.4e-5
POSE_MARGINAL_S_PER_D_POSE: Final = 603.0
STORAGE_RESERVE_BYTES: Final = 4 * 1024**3
EXPECTED_Q3_ARRAY_BYTES: Final = PROPOSAL_COUNT * 2 * HEIGHT * WIDTH * 3 * 4
AXIS: Final = (
    "[macOS-CPU scorer-free retained JS6 tensors; measured contest-CUDA T4 n600 "
    "two-instance pose-damage envelope] COMPONENT-ONLY"
)

PROPOSAL_INDEX_SHA256: Final = (
    "09fe909fa087a39e925ab39322022c9cfb88de612987512718b38a37a98734f8"
)
FAMILY_LAW: Final = REPO / ".omx/research/ddm_re1_round1_full_auth_row_20260813.md"
FAMILY_LAW_SHA256: Final = (
    "dfd68512fb7a8d23eb551841b56ed43bc9e81cbe0e0e92ed84a35560db8f83c4"
)
Q3_SOURCE: Final = REPO / "experiments/train_tr1_partition_renderer_mlx.py"
Q3_SOURCE_SHA256: Final = (
    "d0c9b5ef94ac3ddcc315345696114f312969808c53acf837a76e16a74ae50899"
)
JS6_SOURCE: Final = REPO / "experiments/ddm_js6_seg_representation_join.py"
JS6_SOURCE_SHA256: Final = (
    "8213c189723f398c6be59f12997f2cd2a4fc9c9109c423944242029b66bb103f"
)
Q3_LUMA_WEIGHTS: Final = (0.299, 0.587, 0.114)


class JS6BError(RuntimeError):
    """A source pin, retained payload, screen, or resume invariant failed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def require_record(record: dict[str, Any], *, beneath: Path | None = None) -> Path:
    path = Path(str(record.get("path", ""))).resolve()
    if beneath is not None:
        try:
            path.relative_to(beneath.resolve())
        except ValueError as exc:
            raise JS6BError(f"retained payload escapes {beneath}: {path}") from exc
    if not path.is_file() or file_record(path) != record:
        raise JS6BError(f"retained payload differs: {path}")
    return path


def _atomic_replace(path: Path, payload: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with partial.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)
    return file_record(path)


def retain_bytes(path: Path, payload: bytes) -> dict[str, Any]:
    expected = {
        "path": str(path.resolve()),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    if path.is_file():
        if file_record(path) != expected:
            raise JS6BError(f"refusing to replace a different retained payload: {path}")
        return expected
    return _atomic_replace(path, payload)


def retain_json(path: Path, value: Any) -> dict[str, Any]:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    return retain_bytes(path, payload)


def replace_json(path: Path, value: Any) -> dict[str, Any]:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    return _atomic_replace(path, payload)


def retain_npy(path: Path, value: np.ndarray) -> dict[str, Any]:
    stream = io.BytesIO()
    np.save(stream, np.asarray(value), allow_pickle=False)
    return retain_bytes(path, stream.getvalue())


def q3_constraint_and_projector() -> tuple[np.ndarray, np.ndarray]:
    """Adapt #837's exact 6x12 float yuv6 constraint and Euclidean projector."""
    constraint = np.zeros((6, 12), dtype=np.float64)
    for pixel in range(4):
        constraint[pixel, 3 * pixel : 3 * pixel + 3] = Q3_LUMA_WEIGHTS
        constraint[4, 3 * pixel] = 0.25
        constraint[5, 3 * pixel + 2] = 0.25
    projector = np.eye(12, dtype=np.float64) - np.linalg.pinv(constraint) @ constraint
    if not np.allclose(projector @ projector, projector):
        raise JS6BError("Q3 projector is not idempotent")
    if float(np.max(np.abs(constraint @ projector))) >= 1e-10:
        raise JS6BError("Q3 projector is not in the float yuv6 kernel")
    if int(np.linalg.matrix_rank(projector)) != 6:
        raise JS6BError("Q3 projector rank differs from six")
    return constraint, projector


def q3_components(delta: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Return retained-worthy Q3-null and pose-visible components of one RGB delta."""
    value = np.asarray(delta)
    if value.ndim != 3 or value.shape[-1] != 3:
        raise JS6BError(f"expected HWC RGB delta, got {value.shape}")
    height, width, channels = value.shape
    if height % 2 or width % 2:
        raise JS6BError("Q3 requires an even scorer lattice")
    if value.dtype != np.float32 or not np.all(np.isfinite(value)):
        raise JS6BError("Q3 input must be finite float32")
    constraint, projector = q3_constraint_and_projector()
    blocks = value.reshape(height // 2, 2, width // 2, 2, channels)
    blocks = blocks.transpose(0, 2, 1, 3, 4).reshape(-1, 12)
    # The contraction is deliberately explicit: some macOS Accelerate builds
    # route the extremely skinny ``(N, 12) @ (12, 12)`` GEMM through an
    # unstable threaded path.  ``einsum`` has the same algebra and is fully
    # deterministic for these fixed-size blocks.
    null_blocks = np.einsum("bi,ji->bj", blocks, projector, optimize=False)
    null = null_blocks.reshape(height // 2, width // 2, 2, 2, channels)
    null = null.transpose(0, 2, 1, 3, 4).reshape(value.shape).astype(np.float32)
    visible = np.subtract(value, null, dtype=np.float32)
    total_energy = float(np.sum(np.square(value, dtype=np.float64), dtype=np.float64))
    null_energy = float(np.sum(np.square(null, dtype=np.float64), dtype=np.float64))
    visible_energy = float(np.sum(np.square(visible, dtype=np.float64), dtype=np.float64))
    projected_blocks = null.reshape(height // 2, 2, width // 2, 2, channels)
    projected_blocks = projected_blocks.transpose(0, 2, 1, 3, 4).reshape(-1, 12)
    projected_constraints = np.einsum(
        "bi,ji->bj", projected_blocks, constraint, optimize=False
    )
    residual = float(np.max(np.abs(projected_constraints)))
    visible_fraction = 0.0 if total_energy == 0.0 else visible_energy / total_energy
    return null, visible, {
        "total_rgb_l2_energy": total_energy,
        "q3_null_rgb_l2_energy": null_energy,
        "q3_pose_visible_rgb_l2_energy": visible_energy,
        "q3_pose_visible_energy_fraction": visible_fraction,
        "q3_float_constraint_max_abs": residual,
    }


def screen_arithmetic(*, target_mass: int, semantic_cells: int) -> dict[str, Any]:
    """Apply the measured family envelope with an optimistic zero-byte rate floor."""
    if target_mass < 0 or semantic_cells <= 0:
        raise JS6BError("screen counts must be nonnegative with at least one semantic cell")
    seg_value = target_mass * S_PER_FLIP
    lower_risk = semantic_cells * POSE_DAMAGE_LOWER_S_PER_CELL
    upper_risk = semantic_cells * POSE_DAMAGE_UPPER_S_PER_CELL
    lower_net = -seg_value + lower_risk
    upper_net = -seg_value + upper_risk
    return {
        "optimistic_seg_value_s": seg_value,
        "optimistic_seg_delta_s": -seg_value,
        "measured_pose_risk_lower_s": lower_risk,
        "measured_pose_risk_upper_s": upper_risk,
        "optimistic_rate_delta_s": 0.0,
        "screened_net_delta_s_lower_zero_rate": lower_net,
        "screened_net_delta_s_upper_zero_rate": upper_net,
        "screen_margin_seg_value_minus_upper_pose_risk_s": seg_value - upper_risk,
        "admitted": upper_net < 0.0,
    }


def storage_preflight(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output)
    retained = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    remaining = max(0, EXPECTED_Q3_ARRAY_BYTES - retained)
    required = remaining + STORAGE_RESERVE_BYTES
    result = {
        "schema": "ddm_js6b_storage_preflight.v1",
        "tier": str(output.resolve()),
        "free_bytes": usage.free,
        "already_retained_bytes": retained,
        "expected_q3_array_payload_bytes": EXPECTED_Q3_ARRAY_BYTES,
        "remaining_expected_payload_bytes": remaining,
        "reserve_bytes": STORAGE_RESERVE_BYTES,
        "required_free_bytes": required,
        "passed": usage.free >= required,
        "cleanup_policy": "block and retain; no generated payload is deleted",
    }
    replace_json(output / "STORAGE_PREFLIGHT.json", result)
    if not result["passed"]:
        raise JS6BError(
            f"storage preflight failed: free={usage.free}, required={required}"
        )
    return result


def load_bank(bank: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    index = bank.resolve() / "proposal_index.jsonl"
    if sha256_file(index) != PROPOSAL_INDEX_SHA256:
        raise JS6BError("JS6 proposal index pin differs")
    rows = [json.loads(line) for line in index.read_text().splitlines() if line]
    ids = [str(row.get("proposal_id")) for row in rows]
    if len(rows) != PROPOSAL_COUNT or len(set(ids)) != PROPOSAL_COUNT:
        raise JS6BError("JS6 proposal bank is not the sealed 200-row unique census")
    if any(row.get("producer_source_sha256") != JS6_SOURCE_SHA256 for row in rows):
        raise JS6BError("JS6 proposal producer pin differs")
    return rows, file_record(index)


def verify_static_pins() -> dict[str, dict[str, Any]]:
    pins = {
        "family_law": (FAMILY_LAW, FAMILY_LAW_SHA256),
        "q3_source": (Q3_SOURCE, Q3_SOURCE_SHA256),
        "js6_source": (JS6_SOURCE, JS6_SOURCE_SHA256),
    }
    records: dict[str, dict[str, Any]] = {}
    for name, (path, expected) in pins.items():
        if sha256_file(path) != expected:
            raise JS6BError(f"consumed source pin differs: {name}")
        records[name] = file_record(path)
    return records


def screen_one(row: dict[str, Any], output: Path, bank: Path) -> dict[str, Any]:
    proposal_id = str(row["proposal_id"])
    proposal_root = output / "retained/screens" / proposal_id
    result_path = proposal_root / "SCREEN_ROW.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        if result.get("proposal_id") != proposal_id:
            raise JS6BError("resumed screen row belongs to another proposal")
        if result.get("input_delta") is not None:
            require_record(result["input_delta"], beneath=bank)
            require_record(result["q3_null_delta"], beneath=output)
            require_record(result["q3_pose_visible_delta"], beneath=output)
        return result

    target_mass = int(
        row["receiver_surface"]["exact_field_target_edge_mass_on_support"]
    )
    semantic_cells = int(row["token_site_count"])
    arithmetic = screen_arithmetic(
        target_mass=target_mass,
        semantic_cells=semantic_cells,
    )
    # The lower calibration is already measured on the same family.  A row
    # that loses even there cannot be rescued by the stricter upper-bound
    # admission test, so reading and projecting its multi-megabyte tensor
    # would create irrelevant payload.  Q3 remains mandatory for any row that
    # crosses this first gate.
    q3_required = arithmetic["screened_net_delta_s_lower_zero_rate"] < 0.0
    retained = row.get("retained_payloads", {})
    delta_record = retained.get("scorer_delta.float32.npy")
    if not isinstance(delta_record, dict):
        raise JS6BError(f"proposal lacks retained float32 scorer delta: {proposal_id}")
    if q3_required:
        delta_path = require_record(delta_record, beneath=bank)
        delta = np.load(delta_path, mmap_mode="r", allow_pickle=False)
        if delta.shape == (3, HEIGHT, WIDTH):
            source_geometry = "CHW"
            delta_hwc = np.moveaxis(np.asarray(delta), 0, -1)
        elif delta.shape == (HEIGHT, WIDTH, 3):
            source_geometry = "HWC"
            delta_hwc = np.asarray(delta)
        else:
            raise JS6BError(f"proposal scorer delta geometry differs: {proposal_id}")
        null_hwc, visible_hwc, diagnostics = q3_components(delta_hwc)
        if source_geometry == "CHW":
            null = np.moveaxis(null_hwc, -1, 0)
            visible = np.moveaxis(visible_hwc, -1, 0)
        else:
            null, visible = null_hwc, visible_hwc
        null_record: dict[str, Any] | None = retain_npy(
            proposal_root / "q3_null_delta.float32.npy", null
        )
        visible_record: dict[str, Any] | None = retain_npy(
            proposal_root / "q3_pose_visible_delta.float32.npy", visible
        )
        consumed_delta: dict[str, Any] | None = delta_record
    else:
        source_geometry = None
        consumed_delta = None
        null_record = None
        visible_record = None
        diagnostics = {
            "status": "NOT_MATERIALIZED_LOSES_LOWER_MEASURED_CALIBRATION",
            "reason": (
                "the measured lower pose-risk calibration already makes the optimistic "
                "zero-rate net delta nonnegative"
            ),
        }
    result = {
        "schema": "ddm_js6b_pose_screen_row.v1",
        "proposal_id": proposal_id,
        "ordinal": int(row["ordinal"]),
        "pair": int(row["pair"]),
        "directed_edge": row["directed_edge"],
        "target_mass_upper_bound_flips": target_mass,
        "semantic_cell_count": semantic_cells,
        "source_delta_geometry": source_geometry,
        "input_delta": consumed_delta,
        "q3_null_delta": null_record,
        "q3_pose_visible_delta": visible_record,
        "q3_diagnostics": diagnostics,
        "q3_required_after_lower_calibration": q3_required,
        "screen": arithmetic,
        "disposition": "SURVIVOR" if arithmetic["admitted"] else "HELD",
        "verdict_scope": (
            "INSTANCE: this retained JS6 proposal on CP135 under the measured two-instance "
            "per-cell pose envelope"
        ),
        "score_claim": False,
        "acceptance_tested": False,
        "honesty_limits": (
            "Q3 is float-lattice diagnostic only; the semantic-token proposal is not Q3-"
            "realized, integer/receiver nonlinear leakage is unmeasured, target mass is an "
            "optimistic upper bound, archive rate is optimistically zero, and only the "
            "same-dispatch T4 SegNet/PoseNet measurement can adjudicate a compiled survivor"
        ),
    }
    retain_json(result_path, result)
    return result


def run(*, output: Path, bank: Path, resume_from: str) -> dict[str, Any]:
    if resume_from != RUN_ID:
        raise JS6BError("resume token differs from the sealed run identity")
    output = output.resolve()
    bank = bank.resolve()
    storage_preflight(output)
    sources = verify_static_pins()
    rows, index_record = load_bank(bank)
    inputs = {
        "schema": "ddm_js6b_inputs.v1",
        "run_id": RUN_ID,
        "resume_from": resume_from,
        "proposal_index": index_record,
        "consumed_sources": sources,
        "proposal_count": len(rows),
        "axis": AXIS,
        "scorer_run": False,
    }
    retain_json(output / "checkpoints/stage_00_inputs.json", inputs)
    results: list[dict[str, Any]] = []
    for ordinal, row in enumerate(rows, start=1):
        results.append(screen_one(row, output, bank))
        replace_json(
            output / "STATE.json",
            {
                "schema": "ddm_js6b_state.v1",
                "run_id": RUN_ID,
                "stage": "screen",
                "completed_proposals": ordinal,
                "proposal_count": len(rows),
                "resume_from": resume_from,
            },
        )
    ranked = sorted(
        results,
        key=lambda item: (
            float(item["screen"]["screened_net_delta_s_upper_zero_rate"]),
            float(item["q3_diagnostics"].get("q3_pose_visible_energy_fraction", 1.0)),
            str(item["proposal_id"]),
        ),
    )
    jsonl = b"".join(
        (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode()
        for row in ranked
    )
    screen_record = retain_bytes(output / "POSE_SCREEN.jsonl", jsonl)
    survivors = [row for row in ranked if bool(row["screen"]["admitted"])]
    held = [row for row in ranked if not bool(row["screen"]["admitted"])]
    q3_materialized = [row for row in ranked if row["q3_null_delta"] is not None]
    if survivors:
        raise JS6BError(
            "the sealed bank unexpectedly has a survivor; fail closed before HP3/RC64 compile"
        )
    stale_fire_order = output / "SEALED_FIRE_ORDER.json"
    if stale_fire_order.exists():
        raise JS6BError("a stale fire order exists although the screen has no survivor")
    nonfire = {
        "schema": "ddm_js6b_nonfire_order.v1",
        "run_id": RUN_ID,
        "disposition": "FOLDED",
        "candidate_created": False,
        "sealed_fire_order_created": False,
        "survivors": 0,
        "held": len(held),
        "fire_trigger": (
            "none for this bank; reopen only after a receiver-realizable Q3 integer actuator "
            "or retained per-candidate Pose vectors supplies a tighter candidate-specific bound"
        ),
        "reason": (
            "all 200 proposals lose to the zero-edit control even under the lower measured "
            "pose-damage calibration and an optimistic zero-byte rate delta"
        ),
    }
    nonfire_record = retain_json(output / "NO_FIRE_ORDER.json", nonfire)
    summary = {
        "schema": "ddm_js6b_pose_screen_result.v1",
        "run_id": RUN_ID,
        "execution_status": "COMPLETE",
        "disposition": "FOLDED",
        "axis": AXIS,
        "selection_mode": "full sealed census, 200 of 200 proposals, no sampling",
        "proposal_count": len(ranked),
        "survivor_count": 0,
        "held_count": len(held),
        "best_upper_bound_row": ranked[0],
        "best_lower_calibration_row": min(
            ranked,
            key=lambda item: float(
                item["screen"]["screened_net_delta_s_lower_zero_rate"]
            ),
        ),
        "pose_damage_calibration": {
            "lower_s_per_semantic_cell": POSE_DAMAGE_LOWER_S_PER_CELL,
            "upper_s_per_semantic_cell": POSE_DAMAGE_UPPER_S_PER_CELL,
            "marginal_s_per_unit_d_pose": POSE_MARGINAL_S_PER_D_POSE,
            "source": sources["family_law"],
        },
        "q3": {
            "source": sources["q3_source"],
            "use": "diagnostic ranking only; not a realized integer semantic-token actuator",
            "materialized_proposal_count": len(q3_materialized),
            "materialization_gate": (
                "only rows that beat the lower measured pose calibration require Q3; "
                "pre-existing resumed diagnostics are retained"
            ),
            "all_materialized_null_and_visible_arrays_retained": True,
        },
        "screen_rows": screen_record,
        "nonfire_order": nonfire_record,
        "candidate_archive": None,
        "sealed_fire_order": None,
        "scorer_run": False,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "compile_suppressed_by_screen": True,
        "all_materialized_payloads_retained": True,
        "verdict_scope": (
            "FORMULATION: the sealed 200-row JS6 unprojected semantic-cell proposal bank on "
            "CP135 under the measured family envelope; not the Q3 actuator family"
        ),
    }
    retain_json(output / "FINAL_RESULT.json", summary)
    retain_json(output / "checkpoints/stage_20_nonfire_final.json", summary)
    replace_json(
        output / "STATE.json",
        {
            "schema": "ddm_js6b_state.v1",
            "run_id": RUN_ID,
            "stage": "complete",
            "completed_proposals": len(ranked),
            "proposal_count": len(ranked),
            "resume_from": resume_from,
        },
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--proposal-bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--resume-from", required=True)
    args = parser.parse_args(argv)
    result = run(
        output=args.output,
        bank=args.proposal_bank,
        resume_from=args.resume_from,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
