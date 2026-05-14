# SPDX-License-Identifier: MIT
"""Water-fill planning for deterministic postdecode selector atoms.

This module turns scorer-proxy sweep rows into canonical meta-Lagrangian atoms.
It does not load scorers, mutate archives, or claim scores.  The output is the
bridge from local/MPS/CUDA-proxy postdecode perturbation sweeps into the shared
Pareto, water-fill, and field-equation planners.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from tac.optimization.meta_lagrangian_allocator import build_atom_ledger

SCHEMA_VERSION = "postdecode_selector_waterfill_plan_v1"
TRACK_ID = "frame0_postdecode_selector"
FAMILY = "frame0_postdecode_selector"
DEFAULT_RESEARCH_BASIS_IDS = (
    "FGS1",
    "FES1",
    "cooperative_receiver",
    "segnet_last_frame_only",
    "posenet_first_frame_sensitive",
)
CUDA_RANKABLE_AXES = frozenset(
    {
        "contest-cuda",
        "[contest-cuda]",
        "modal-t4-cuda-proxy-prefix",
        "modal-t4-cuda-proxy",
        "modal-a100-cuda-proxy-prefix",
        "modal-a100-cuda-proxy",
    }
)


class PostdecodeSelectorWaterfillError(ValueError):
    """Raised when a postdecode selector sweep cannot become planning atoms."""


def _require_modes(sweep: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    modes = sweep.get("modes")
    if not isinstance(modes, list) or not modes:
        raise PostdecodeSelectorWaterfillError("sweep JSON must contain non-empty 'modes' list")
    return [mode for mode in modes if isinstance(mode, Mapping)]


def _baseline_mode(modes: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    for row in modes:
        if str(row.get("mode")) == "none":
            return row
    raise PostdecodeSelectorWaterfillError("sweep modes must include mode='none' baseline")


def _float_field(row: Mapping[str, Any], *names: str) -> float:
    for name in names:
        if name in row:
            return float(row[name])
    raise PostdecodeSelectorWaterfillError(
        f"mode {row.get('mode')!r} missing required numeric field {names}"
    )


def _pair_array(row: Mapping[str, Any], key: str, n_pairs: int) -> list[float]:
    values = row.get(key)
    if not isinstance(values, list):
        raise PostdecodeSelectorWaterfillError(f"mode {row.get('mode')!r} missing {key}")
    if len(values) != n_pairs:
        raise PostdecodeSelectorWaterfillError(
            f"mode {row.get('mode')!r} {key} length {len(values)} != {n_pairs}"
        )
    return [float(value) for value in values]


def _score(avg_pose: float, avg_seg: float, archive_bytes: int) -> float:
    return 100.0 * float(avg_seg) + math.sqrt(10.0 * float(avg_pose)) + 25.0 * int(
        archive_bytes
    ) / 37_545_489


def _axis_evidence_grade(axis: str) -> str:
    axis_text = axis.strip() or "unknown-proxy"
    return f"[{axis_text} proxy]"


def _axis_transfer_contract(axis: str) -> dict[str, Any]:
    """Return dispatch/ranking trust metadata for a selector evidence axis."""

    lowered = axis.strip().lower()
    normalized = lowered.replace("_", "-").replace(" ", "-")
    if normalized in CUDA_RANKABLE_AXES:
        return {
            "rankable": True,
            "axis_transfer_status": "cuda_proxy_source",
            "axis_transfer_blockers": [],
        }
    if "mps" in lowered or "macos" in lowered:
        return {
            "rankable": False,
            "axis_transfer_status": "mps_to_cuda_uncalibrated",
            "axis_transfer_blockers": [
                "mps_to_cuda_transfer_uncalibrated",
                "fes1_mps_selector_exact_cuda_pose_regression_observed",
                "fes1_cpu_mps_to_cuda_scorer_device_split_confirmed",
            ],
        }
    return {
        "rankable": False,
        "axis_transfer_status": "unknown_axis_uncalibrated",
        "axis_transfer_blockers": ["unknown_proxy_axis_requires_cuda_confirmation"],
    }


def _common_atom_fields(
    *,
    sweep: Mapping[str, Any],
    evidence_source_path: str,
    confidence: float,
) -> dict[str, Any]:
    transfer = _axis_transfer_contract(str(sweep.get("axis", "unknown-proxy")))
    return {
        "family": FAMILY,
        "family_group": FAMILY,
        "pareto_scope": "postdecode_selector",
        "confidence": float(confidence),
        "evidence_grade": _axis_evidence_grade(str(sweep.get("axis", "unknown-proxy"))),
        "proxy_row": True,
        "rankable": bool(transfer["rankable"]),
        "axis_transfer_status": transfer["axis_transfer_status"],
        "score_claim": False,
        "source_archive_sha256": str(sweep.get("archive_sha256", "")),
        "evidence_source_path": evidence_source_path,
        "research_basis_ids": list(DEFAULT_RESEARCH_BASIS_IDS),
        "interaction_assumptions": [
            "postdecode_after_rgb_decoder",
            "scorer_free_inflate_runtime",
            "selector_bytes_charged_before_exact_eval",
            "must_revalidate_per_base_archive_and_runtime",
        ],
        "dispatch_blockers": [
            "proxy_or_planning_only",
            "requires_byte_closed_selector_archive",
            "requires_positive_contest_cuda_auth_eval",
            *transfer["axis_transfer_blockers"],
        ],
    }


def build_mode_atoms_from_sweep(
    sweep: Mapping[str, Any],
    *,
    mode_byte_delta: int = 0,
    confidence: float = 1.0,
    evidence_source_path: str = "",
) -> list[dict[str, Any]]:
    """Return one planning atom per non-baseline postdecode mode."""

    modes = _require_modes(sweep)
    baseline = _baseline_mode(modes)
    base_pose = _float_field(baseline, "avg_posenet_dist")
    base_seg = _float_field(baseline, "avg_segnet_dist")
    common = _common_atom_fields(
        sweep=sweep,
        evidence_source_path=evidence_source_path,
        confidence=confidence,
    )
    atoms: list[dict[str, Any]] = []
    for row in modes:
        mode = str(row.get("mode"))
        if mode == "none":
            continue
        atoms.append(
            {
                **common,
                "atom_id": f"postdecode_mode:{mode}",
                "byte_delta": int(mode_byte_delta),
                "expected_pose_dist_delta": _float_field(row, "avg_posenet_dist") - base_pose,
                "expected_seg_dist_delta": _float_field(row, "avg_segnet_dist") - base_seg,
                "candidate_mode": mode,
                "pair_support": [],
                "geometry_priors": ["first_frame_postdecode_perturbation"],
                "openpilot_priors": ["ego_motion_pairwise_pose_sensitivity"],
            }
        )
    return atoms


def build_oracle_selector_atom_from_sweep(
    sweep: Mapping[str, Any],
    *,
    selector_byte_delta: int,
    confidence: float = 1.0,
    evidence_source_path: str = "",
) -> dict[str, Any]:
    """Return a charged per-pair selector atom from sweep pair arrays.

    The selector chooses, for each pair, the mode that minimizes a local linear
    approximation of the official score around the baseline pose distance:
    ``100 * seg_pair + d(sqrt(10p))/dp * pose_pair``.  This is a planning
    oracle only; exact promotion requires the selector bytes to be packed into
    an archive and scored through contest-CUDA.
    """

    modes = _require_modes(sweep)
    baseline = _baseline_mode(modes)
    n_pairs = int(sweep.get("n_pairs") or 0)
    if n_pairs <= 0:
        pose_values = baseline.get("pair_posenet_dist")
        if not isinstance(pose_values, list) or not pose_values:
            raise PostdecodeSelectorWaterfillError("cannot infer n_pairs from sweep")
        n_pairs = len(pose_values)
    base_pose = _float_field(baseline, "avg_posenet_dist")
    base_seg = _float_field(baseline, "avg_segnet_dist")
    archive_bytes = int(sweep.get("archive_bytes", 0))
    if archive_bytes <= 0:
        raise PostdecodeSelectorWaterfillError("sweep must include positive archive_bytes")

    pose_weight = math.sqrt(10.0) / (2.0 * math.sqrt(max(base_pose, 1e-12)))
    palette = [str(row.get("mode")) for row in modes]
    mode_pose = [_pair_array(row, "pair_posenet_dist", n_pairs) for row in modes]
    mode_seg = [_pair_array(row, "pair_segnet_dist", n_pairs) for row in modes]

    indices: list[int] = []
    selected_pose: list[float] = []
    selected_seg: list[float] = []
    for pair_idx in range(n_pairs):
        scores = [
            100.0 * mode_seg[mode_idx][pair_idx] + pose_weight * mode_pose[mode_idx][pair_idx]
            for mode_idx in range(len(modes))
        ]
        best_idx = min(range(len(scores)), key=scores.__getitem__)
        indices.append(best_idx)
        selected_pose.append(mode_pose[best_idx][pair_idx])
        selected_seg.append(mode_seg[best_idx][pair_idx])

    avg_pose = sum(selected_pose) / n_pairs
    avg_seg = sum(selected_seg) / n_pairs
    selector_score = _score(avg_pose, avg_seg, archive_bytes + int(selector_byte_delta))
    baseline_score = _score(base_pose, base_seg, archive_bytes)
    counts = Counter(palette[idx] for idx in indices)
    common = _common_atom_fields(
        sweep=sweep,
        evidence_source_path=evidence_source_path,
        confidence=confidence,
    )
    return {
        **common,
        "atom_id": "postdecode_selector:oracle_pairwise_waterfill",
        "byte_delta": int(selector_byte_delta),
        "expected_pose_dist_delta": avg_pose - base_pose,
        "expected_seg_dist_delta": avg_seg - base_seg,
        "selector_policy": "pairwise_local_score_waterfill",
        "selector_palette": palette,
        "selector_indices": indices,
        "selected_mode_counts": dict(sorted(counts.items())),
        "selector_score_proxy_charged": selector_score,
        "selector_delta_vs_none_charged": selector_score - baseline_score,
        "pair_support": list(range(n_pairs)),
        "hard_pair_support": [idx for idx, mode_idx in enumerate(indices) if mode_idx != 0],
        "geometry_priors": [
            "first_frame_postdecode_perturbation",
            "per_pair_pose_sensitivity",
        ],
        "openpilot_priors": ["ego_motion_pairwise_pose_sensitivity"],
    }


def build_postdecode_selector_waterfill_plan(
    sweep: Mapping[str, Any],
    *,
    selector_byte_delta: int,
    mode_byte_delta: int = 0,
    confidence: float = 1.0,
    evidence_source_path: str = "",
) -> dict[str, Any]:
    """Build a postdecode selector plan plus canonical atom ledger."""

    modes = _require_modes(sweep)
    baseline = _baseline_mode(modes)
    atoms = build_mode_atoms_from_sweep(
        sweep,
        mode_byte_delta=mode_byte_delta,
        confidence=confidence,
        evidence_source_path=evidence_source_path,
    )
    selector_atom = build_oracle_selector_atom_from_sweep(
        sweep,
        selector_byte_delta=selector_byte_delta,
        confidence=confidence,
        evidence_source_path=evidence_source_path,
    )
    atoms.append(selector_atom)
    ledger = build_atom_ledger(
        atoms,
        base_pose_dist=_float_field(baseline, "avg_posenet_dist"),
        source=evidence_source_path or "postdecode_selector_sweep",
    )
    return {
        "schema": SCHEMA_VERSION,
        "track_id": TRACK_ID,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "axis": str(sweep.get("axis", "unknown-proxy")),
        "archive_bytes": int(sweep.get("archive_bytes", 0)),
        "archive_sha256": str(sweep.get("archive_sha256", "")),
        "mode_count": len(modes),
        "selector_byte_delta": int(selector_byte_delta),
        "mode_byte_delta": int(mode_byte_delta),
        "atom_count": len(atoms),
        "atoms": atoms,
        "atom_ledger": ledger,
    }


__all__ = [
    "PostdecodeSelectorWaterfillError",
    "build_mode_atoms_from_sweep",
    "build_oracle_selector_atom_from_sweep",
    "build_postdecode_selector_waterfill_plan",
]
