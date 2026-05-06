"""LA-POSE-guided foveation transport atoms for planning ledgers.

The rows emitted here are byte-bearing planning atoms, not archive evidence.
They let the meta-Lagrangian and field-equation planners compare a concrete
foveation-parameter byte budget against expected pair-local component deltas
without authorizing dispatch. A row becomes dispatchable only after a later
builder proves the exact archive bytes and inflate runtime consumes them.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from tac.analysis.lapose_motion_atoms import (
    LaposeMotionAtomError,
    records_from_json_payload,
)
from tac.analysis.lapose_paper_contract import LAPOSE_PAPER_REFERENCE
from tac.optimization.meta_lagrangian_allocator import build_atom_ledger
from tac.optimization.research_basis import research_basis_manifest

SCHEMA_VERSION = 1
SCHEMA = "lapose_foveation_transport_atom_manifest_v1"
TOOL = "tac.analysis.lapose_foveation_atoms.build_foveation_transport_atom_manifest"
DEFAULT_FRAME_WIDTH = 512
DEFAULT_FRAME_HEIGHT = 384
DEFAULT_FOVEAL_CENTER = (256.0, 174.0)
DEFAULT_CENTER_GAIN = (18.0, 10.0)
DEFAULT_RADIUS = 96.0
DEFAULT_RADIUS_GAIN = 0.25
DEFAULT_ALPHA = 1.6
DEFAULT_ALPHA_GAIN = 0.35
DEFAULT_POWER = 2.0
DEFAULT_POWER_GAIN = 0.2
DEFAULT_SCALAR_BYTES = 2
DEFAULT_PAIR_INDEX_BYTES = 2
DEFAULT_OPCODE_BYTES = 1
FOVEATION_SCALAR_COUNT = 5
RESEARCH_BASIS_IDS = [
    "lapose_2026",
    "telescope_2026",
    "foveated_diffusion_2026",
    "geometric_visual_servo_ot_2026",
]


def build_foveation_transport_atom_manifest(
    records: Iterable[Mapping[str, Any]],
    *,
    base_pose_dist: float,
    source: str,
    frame_width: int = DEFAULT_FRAME_WIDTH,
    frame_height: int = DEFAULT_FRAME_HEIGHT,
    foveal_center: tuple[float, float] = DEFAULT_FOVEAL_CENTER,
    center_gain: tuple[float, float] = DEFAULT_CENTER_GAIN,
    scalar_bytes: int = DEFAULT_SCALAR_BYTES,
    pair_index_bytes: int = DEFAULT_PAIR_INDEX_BYTES,
    opcode_bytes: int = DEFAULT_OPCODE_BYTES,
    max_atoms: int | None = None,
) -> dict[str, Any]:
    """Build planning-only foveation/geometry transport atom rows.

    ``records`` use the same pair-local shape as
    :func:`tac.analysis.lapose_motion_atoms.build_motion_atom_manifest`.
    ``byte_delta`` in each emitted atom is an estimated charged payload budget
    for one pair-local foveation parameter tuple:
    ``opcode + pair_index + five quantized scalars``. It is deliberately
    treated as proxy evidence until archive construction measures exact bytes.
    """

    if not source:
        raise LaposeMotionAtomError("source is required")
    if frame_width <= 0 or frame_height <= 0:
        raise LaposeMotionAtomError("frame dimensions must be positive")
    center_x, center_y = _finite_center(foveal_center, frame_width=frame_width, frame_height=frame_height)
    gain_x, gain_y = _finite_gain(center_gain)
    if scalar_bytes <= 0 or pair_index_bytes <= 0 or opcode_bytes < 0:
        raise LaposeMotionAtomError("byte model widths must be non-negative and scalar/pair widths positive")
    if max_atoms is not None and max_atoms <= 0:
        raise LaposeMotionAtomError("max_atoms must be positive when provided")

    normalized = sorted(
        (_normalize_record(record) for record in records),
        key=lambda record: int(record["pair_index"]),
    )
    if not normalized:
        raise LaposeMotionAtomError("at least one foveation record is required")
    _reject_duplicate_pair_indices(normalized)

    byte_delta = int(opcode_bytes + pair_index_bytes + FOVEATION_SCALAR_COUNT * scalar_bytes)
    byte_model = {
        "schema": "foveation_transport_quantized_tuple_byte_model_v1",
        "byte_delta_is_estimate": True,
        "opcode_bytes": int(opcode_bytes),
        "pair_index_bytes": int(pair_index_bytes),
        "scalar_count": FOVEATION_SCALAR_COUNT,
        "scalar_bytes": int(scalar_bytes),
        "estimated_bytes_per_pair_atom": byte_delta,
        "scalar_encoding": "planner_quantized_scalar_width_not_archive_codec",
        "exact_archive_bytes_required": True,
    }
    atoms = [
        _record_to_foveation_atom(
            record,
            byte_delta=byte_delta,
            frame_width=frame_width,
            frame_height=frame_height,
            foveal_center=(center_x, center_y),
            center_gain=(gain_x, gain_y),
            byte_model=byte_model,
        )
        for record in normalized
    ]
    ledger = build_atom_ledger(atoms, base_pose_dist=base_pose_dist, source=source)
    atoms = _ranked_atoms_from_ledger(atoms, ledger)
    if max_atoms is not None:
        atoms = atoms[:max_atoms]
        ledger = _truncate_ledger(ledger, max_atoms=max_atoms)
    return {
        "schema_version": SCHEMA_VERSION,
        "schema": SCHEMA,
        "tool": TOOL,
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "promotion_eligible": False,
        "source": source,
        "paper_reference": LAPOSE_PAPER_REFERENCE,
        "research_basis": research_basis_manifest(RESEARCH_BASIS_IDS),
        "base_pose_dist": float(base_pose_dist),
        "record_count": len(normalized),
        "record_sha256": _sha256_json(normalized),
        "frame_contract": {
            "width": int(frame_width),
            "height": int(frame_height),
            "base_foveal_center": [round(center_x, 12), round(center_y, 12)],
            "center_gain": [round(gain_x, 12), round(gain_y, 12)],
        },
        "charged_byte_model": byte_model,
        "source_atom_count": len(normalized),
        "atom_count": len(atoms),
        "atoms": atoms,
        "atom_ledger": ledger,
        "dispatch_blockers": [
            "planning_only_lapose_foveation_transport_atoms",
            "byte_delta_is_estimated_not_measured_archive_bytes",
            "requires_foveation_archive_builder",
            "requires_runtime_consumption_proof",
            "requires_noop_controls",
            "requires_exact_cuda_auth_eval",
        ],
    }


def _normalize_record(record: Mapping[str, Any]) -> dict[str, Any]:
    pair_index = _int_field(record, "pair_index")
    latent = _float_sequence(record.get("latent_action"), "latent_action")
    if not latent:
        raise LaposeMotionAtomError(f"pair {pair_index}: latent_action must be nonempty")
    confidence = float(record.get("confidence", 1.0))
    if not 0.0 <= confidence <= 1.0:
        raise LaposeMotionAtomError(f"pair {pair_index}: confidence must be in [0, 1]")
    return {
        "pair_index": pair_index,
        "hard_pair_rank": _optional_int_field(record, "hard_pair_rank"),
        "latent_action": latent,
        "latent_norm": math.sqrt(sum(value * value for value in latent)),
        "hard_pair_score": float(record.get("hard_pair_score", 0.0)),
        "expected_seg_dist_delta": float(record.get("expected_seg_dist_delta", 0.0)),
        "expected_pose_dist_delta": float(record.get("expected_pose_dist_delta", 0.0)),
        "confidence": confidence,
        "class_support": _int_list(record.get("class_support") or []),
        "pair_support": _int_list(record.get("pair_support") or [pair_index]),
        "hard_pair_support": _int_list(record.get("hard_pair_support") or []),
        "geometry_priors": _str_list(record.get("geometry_priors") or []),
        "openpilot_priors": _str_list(record.get("openpilot_priors") or []),
        "evidence_grade": str(record.get("evidence_grade") or "planning_lapose_foveation_transport"),
        "allocation_inference": bool(record.get("allocation_inference", False)),
        "evidence_source_path": str(record.get("evidence_source_path") or ""),
        "evidence_source_sha256": str(record.get("evidence_source_sha256") or ""),
        "source_archive_sha256": str(record.get("source_archive_sha256") or ""),
    }


def _record_to_foveation_atom(
    record: Mapping[str, Any],
    *,
    byte_delta: int,
    frame_width: int,
    frame_height: int,
    foveal_center: tuple[float, float],
    center_gain: tuple[float, float],
    byte_model: Mapping[str, Any],
) -> dict[str, Any]:
    pair_index = int(record["pair_index"])
    params = _foveation_parameters(
        record,
        frame_width=frame_width,
        frame_height=frame_height,
        foveal_center=foveal_center,
        center_gain=center_gain,
    )
    geometry_priors = _unique_strings(
        [
            *record["geometry_priors"],
            "lapose_latent_action",
            "hyperbolic_foveation_field",
            "mixed_resolution_token_allocation",
            "se3_geometric_transport_prior",
        ]
    )
    return {
        "atom_id": f"lapose_foveation_transport_pair:{pair_index}",
        "family": "lapose_foveation_transport_atom",
        "family_group": "pose_foveation",
        "pareto_scope": "lapose_foveation_transport",
        "score_claim": False,
        "dispatch_attempted": False,
        "dispatchable": False,
        "ready_for_exact_eval_dispatch": False,
        "byte_delta": byte_delta,
        "model_byte_delta": int(byte_model["opcode_bytes"]) + int(byte_model["pair_index_bytes"]),
        "data_byte_delta": int(byte_model["scalar_count"]) * int(byte_model["scalar_bytes"]),
        "byte_delta_is_estimate": True,
        "expected_seg_dist_delta": float(record["expected_seg_dist_delta"]),
        "expected_pose_dist_delta": float(record["expected_pose_dist_delta"]),
        "confidence": float(record["confidence"]),
        "evidence_grade": str(record["evidence_grade"]),
        "proxy_row": True,
        "pair_support": list(record["pair_support"]),
        "hard_pair_support": list(record["hard_pair_support"]),
        "class_support": list(record["class_support"]),
        "geometry_priors": geometry_priors,
        "openpilot_priors": list(record["openpilot_priors"]),
        "research_basis_ids": list(RESEARCH_BASIS_IDS),
        "interaction_assumptions": [
            "first_order_pair_local_foveation_proxy",
            "nonadditive_stack_interactions_unmeasured",
        ],
        "conflicts_with_families": [],
        "conflicts_with_atoms": [],
        "hard_pair_rank": record["hard_pair_rank"],
        "hard_pair_score": float(record["hard_pair_score"]),
        "latent_norm": float(record["latent_norm"]),
        "allocation_inference": bool(record["allocation_inference"]),
        "foveation_parameters": params,
        "charged_byte_model": dict(byte_model),
        "charged_byte_contract": (
            "The five foveation scalars are planning bytes only here. A later archive "
            "builder must encode them inside archive.zip and prove inflate consumes "
            "them before dispatch."
        ),
        "evidence_source_path": record["evidence_source_path"],
        "evidence_source_sha256": record["evidence_source_sha256"],
        "source_archive_sha256": record["source_archive_sha256"],
        "archive_manifest_path": "",
        "archive_manifest_sha256": "",
        "dispatch_blockers": [
            "planning_only_lapose_foveation_transport_atom",
            "byte_delta_is_estimated_not_measured_archive_bytes",
            "requires_byte_closed_archive",
            "requires_runtime_consumption_proof",
            "requires_exact_cuda_auth_eval",
        ],
    }


def _foveation_parameters(
    record: Mapping[str, Any],
    *,
    frame_width: int,
    frame_height: int,
    foveal_center: tuple[float, float],
    center_gain: tuple[float, float],
) -> dict[str, Any]:
    latent = list(record["latent_action"])
    latent_norm = float(record["latent_norm"])
    pose_signal = _latent_at(latent, 3)
    seg_signal = _latent_at(latent, 4)
    contrib_signal = _latent_at(latent, 5)
    pose_delta_signal = _latent_at(latent, 6)
    seg_delta_signal = _latent_at(latent, 7)
    center_x = _clip(
        foveal_center[0] + center_gain[0] * math.tanh(pose_signal + 0.5 * pose_delta_signal),
        0.0,
        float(frame_width - 1),
    )
    center_y = _clip(
        foveal_center[1] + center_gain[1] * math.tanh(seg_signal + 0.5 * seg_delta_signal),
        0.0,
        float(frame_height - 1),
    )
    alpha = max(0.0, DEFAULT_ALPHA + DEFAULT_ALPHA_GAIN * math.tanh(latent_norm))
    radius = max(
        1.0,
        DEFAULT_RADIUS
        * (1.0 + DEFAULT_RADIUS_GAIN * math.tanh(abs(contrib_signal) + float(record["hard_pair_score"]))),
    )
    power = max(0.0, DEFAULT_POWER + DEFAULT_POWER_GAIN * math.tanh(abs(seg_signal)))
    return {
        "schema": "lapose_guided_hyperbolic_foveation_tuple_v1",
        "pair_index": int(record["pair_index"]),
        "frame_indices": [2 * int(record["pair_index"]), 2 * int(record["pair_index"]) + 1],
        "alpha": round(alpha, 12),
        "radius": round(radius, 12),
        "power": round(power, 12),
        "origin_x": round(center_x, 12),
        "origin_y": round(center_y, 12),
        "derivation": (
            "deterministic LA-POSE-lite latent proxy mapped to Telescope-style "
            "hyperbolic foveation parameters"
        ),
    }


def _ranked_atoms_from_ledger(
    atoms: Sequence[Mapping[str, Any]],
    ledger: Mapping[str, Any],
) -> list[dict[str, Any]]:
    by_id = {str(atom["atom_id"]): dict(atom) for atom in atoms}
    ranked = []
    for row in ledger.get("rows") or []:
        if not isinstance(row, Mapping):
            continue
        atom_id = str(row.get("atom_id") or "")
        atom = by_id.get(atom_id)
        if atom is not None:
            ranked.append(atom)
    return ranked


def _truncate_ledger(ledger: Mapping[str, Any], *, max_atoms: int) -> dict[str, Any]:
    out = dict(ledger)
    rows = list(ledger.get("rows") or [])
    out["source_atom_count"] = len(rows)
    out["rows"] = rows[:max_atoms]
    out["atom_count"] = len(out["rows"])
    out["truncation"] = {
        "method": "rank_full_ledger_then_truncate",
        "max_atoms": max_atoms,
        "dropped_atom_count": max(len(rows) - max_atoms, 0),
    }
    return out


def _finite_center(
    value: tuple[float, float],
    *,
    frame_width: int,
    frame_height: int,
) -> tuple[float, float]:
    if len(value) != 2:
        raise LaposeMotionAtomError("foveal_center must contain x,y")
    x = float(value[0])
    y = float(value[1])
    if not math.isfinite(x) or not math.isfinite(y):
        raise LaposeMotionAtomError("foveal_center must be finite")
    if not (0.0 <= x <= frame_width - 1 and 0.0 <= y <= frame_height - 1):
        raise LaposeMotionAtomError("foveal_center must lie inside frame")
    return x, y


def _finite_gain(value: tuple[float, float]) -> tuple[float, float]:
    if len(value) != 2:
        raise LaposeMotionAtomError("center_gain must contain x,y")
    x = float(value[0])
    y = float(value[1])
    if not math.isfinite(x) or not math.isfinite(y):
        raise LaposeMotionAtomError("center_gain must be finite")
    return x, y


def _reject_duplicate_pair_indices(records: Sequence[Mapping[str, Any]]) -> None:
    seen: set[int] = set()
    duplicates: list[int] = []
    for record in records:
        pair_index = int(record["pair_index"])
        if pair_index in seen:
            duplicates.append(pair_index)
        seen.add(pair_index)
    if duplicates:
        joined = ", ".join(str(pair_index) for pair_index in duplicates)
        raise LaposeMotionAtomError(f"duplicate pair_index values: {joined}")


def _int_field(record: Mapping[str, Any], key: str) -> int:
    value = record.get(key)
    if not isinstance(value, int):
        raise LaposeMotionAtomError(f"{key} must be an integer")
    return value


def _optional_int_field(record: Mapping[str, Any], key: str) -> int | None:
    value = record.get(key)
    if value is None:
        return None
    if not isinstance(value, int):
        pair_index = record.get("pair_index", "<unknown>")
        raise LaposeMotionAtomError(f"pair {pair_index}: {key} must be an integer when provided")
    return value


def _float_sequence(value: Any, key: str) -> list[float]:
    if not isinstance(value, list | tuple):
        raise LaposeMotionAtomError(f"{key} must be a list")
    out = [float(item) for item in value]
    if any(not math.isfinite(item) for item in out):
        raise LaposeMotionAtomError(f"{key} contains non-finite values")
    return out


def _int_list(value: Any) -> list[int]:
    if not isinstance(value, list | tuple):
        raise LaposeMotionAtomError("expected list")
    out = []
    for item in value:
        if not isinstance(item, int):
            raise LaposeMotionAtomError("expected integer list")
        out.append(item)
    return out


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, list | tuple):
        raise LaposeMotionAtomError("expected list")
    return [str(item) for item in value]


def _unique_strings(values: Iterable[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


def _latent_at(latent: Sequence[float], index: int) -> float:
    return float(latent[index]) if len(latent) > index else 0.0


def _clip(value: float, low: float, high: float) -> float:
    return min(max(float(value), low), high)


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


__all__ = [
    "SCHEMA",
    "SCHEMA_VERSION",
    "build_foveation_transport_atom_manifest",
    "records_from_json_payload",
]
