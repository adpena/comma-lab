#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Plan C067 hotspot-preserving mask-geometry atom policies.

This is a deterministic, planning-only compiler over existing CMG/PMG/Yousfi-
Fridrich row-run atom ledgers and exact-negative traces.  It ranks local mask
geometry atoms by measured hard-pair, component-trace, foveal, boundary, and
confusion signals per charged byte, then emits CMG3A field-policy records that
can be consumed by ``build_cmg3_adaptive_runs_candidate.py``.

No scorer is loaded, no GPU job is launched, and no score claim is made.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "yousfi_fridrich_atom_field_allocator_v1"
COMPILER_SCHEMA = "c067_hotspot_mask_geometry_compiler_v1"
TOOL = "experiments/plan_c067_hotspot_mask_geometry_compiler.py"
ORIGINAL_VIDEO_BYTES = 37_545_489
LAMBDA_RATE = 25.0 / ORIGINAL_VIDEO_BYTES
DEFAULT_CANDIDATE_SIZES = (16, 32, 64, 128)
CUDA_AUTH_EVAL_PATH = (
    "archive.zip -> inflate.sh -> upstream/evaluate.py via "
    "experiments/contest_auth_eval.py --device cuda"
)
CMG3A_NONZERO_CLASS_MIN = 1
CMG3A_NONZERO_CLASS_MAX_EXCLUSIVE = 5


class CompilerError(ValueError):
    """Raised when C067 hotspot compiler inputs are invalid."""


@dataclass(frozen=True)
class Atom:
    source_ledger: str
    atom_id: str
    family: str
    identity: dict[str, Any]
    pair_indices: tuple[int, ...]
    frame_indices: tuple[int, ...]
    class_ids: tuple[int, ...]
    residual_pixels: int
    charged_bytes: int
    marginal_score_saved_proxy: float
    rate_score_cost: float
    lagrangian_net_proxy: float
    weighted_residual_pixel_proxy: float
    boundary_pixel_fraction: float
    component_pair_weight: float
    hard_pair_weight: float
    foveal_weight: float
    confusion_fraction: float
    expected_base_runs_per_row: int | None

    @property
    def row_run_key(self) -> tuple[int, int, int, int, int]:
        try:
            return (
                int(self.identity["frame_index"]),
                int(self.identity["y"]),
                int(self.identity["x0"]),
                int(self.identity["x1_exclusive"]),
                int(self.identity["class_id"]),
            )
        except KeyError as exc:
            raise CompilerError(f"row_run atom {self.atom_id} missing {exc.args[0]!r}") from exc

    @property
    def span_pixels(self) -> int:
        _frame, _y, x0, x1, _class_id = self.row_run_key
        return max(0, int(x1) - int(x0))

    @property
    def primary_pair(self) -> int | None:
        return self.pair_indices[0] if self.pair_indices else None

    def row_run_policy_atom(self) -> dict[str, int]:
        frame, y, x0, x1, class_id = self.row_run_key
        return {
            "frame_index": frame,
            "y": y,
            "x0": x0,
            "x1_exclusive": x1,
            "class_id": class_id,
        }


@dataclass(frozen=True)
class RankedAtom:
    atom: Atom
    rank_score: float
    hotspot_signal: float
    break_even_bytes: float
    reject_reasons: tuple[str, ...]


@dataclass(frozen=True)
class CompilerConfig:
    candidate_sizes: tuple[int, ...] = DEFAULT_CANDIDATE_SIZES
    max_source_atoms: int = 2048
    policy_prefix: str = "c067_hotspot_geometry"
    hotspot_signal_weight: float = 2.0e-6
    min_atom_hotspot_signal: float = 1.0
    max_arbitrary_span_pixels: int = 384
    min_policy_hotspot_fraction: float = 0.65
    max_policy_pair_fraction: float = 0.35
    max_policy_frame_fraction: float = 0.35
    min_policy_spread_atoms: int = 16
    max_policy_proxy_bytes: int = 64_000
    foveation_center_x: float = 256.0
    foveation_center_y: float = 174.0
    horizon_y: float = 174.0
    low_rank_modes: int = 8
    foveation_param_bytes: int = 32


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CompilerError(f"{path} is not valid JSON") from exc


def _finite_float(value: Any, *, field: str, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CompilerError(f"{field} must be numeric")
    out = float(value)
    if not math.isfinite(out):
        raise CompilerError(f"{field} must be finite")
    return out


def _int_tuple(values: Any, *, field: str) -> tuple[int, ...]:
    if values is None:
        return ()
    if not isinstance(values, list):
        raise CompilerError(f"{field} must be a list")
    out: list[int] = []
    for index, value in enumerate(values):
        if not isinstance(value, int):
            raise CompilerError(f"{field}[{index}] must be int")
        out.append(int(value))
    return tuple(out)


def _identity_int(identity: dict[str, Any], key: str) -> tuple[int, ...]:
    value = identity.get(key)
    if isinstance(value, int):
        return (int(value),)
    if isinstance(value, list):
        return tuple(int(item) for item in value if isinstance(item, int))
    return ()


def _hist_total(histogram: Any) -> int:
    if not isinstance(histogram, dict):
        return 0
    total = 0
    for value in histogram.values():
        if isinstance(value, int):
            total += int(value)
    return total


def _confusion_fraction(raw: dict[str, Any], *, residual_pixels: int) -> float:
    source = raw.get("source_class_histogram_pixels")
    candidate = raw.get("candidate_class_histogram_pixels")
    if not isinstance(source, dict) or not isinstance(candidate, dict):
        return 0.0
    source_total = _hist_total(source)
    candidate_total = _hist_total(candidate)
    if source_total <= 0 or candidate_total <= 0:
        return 0.0
    overlap = 0
    for key, source_count in source.items():
        candidate_count = candidate.get(key, 0)
        if isinstance(source_count, int) and isinstance(candidate_count, int):
            overlap += min(int(source_count), int(candidate_count))
    denominator = max(int(residual_pixels), source_total, candidate_total, 1)
    return max(0.0, min(1.0, 1.0 - overlap / denominator))


def _candidate_expected_base_runs_per_row(payload: dict[str, Any]) -> int | None:
    inputs = payload.get("inputs")
    candidate = inputs.get("candidate", {}) if isinstance(inputs, dict) else {}
    if not isinstance(candidate, dict):
        return None
    if candidate.get("mode") == "reconstructed_from_cmg3_nonzero_row_runs_manifest":
        value = candidate.get("max_runs_per_row")
        return int(value) if isinstance(value, int) else None
    if candidate.get("mode") == "reconstructed_from_cmg3a_adaptive_manifest":
        value = candidate.get("base_runs_per_row")
        return int(value) if isinstance(value, int) else None
    return None


def _atom_from_json(raw: dict[str, Any], *, source_ledger: str, expected_base: int | None) -> Atom:
    if raw.get("score_claim") is True:
        raise CompilerError(f"atom {raw.get('atom_id')} has score_claim=true")
    identity = raw.get("identity")
    if not isinstance(identity, dict):
        raise CompilerError(f"atom {raw.get('atom_id')} has no identity object")
    cost_model = raw.get("cost_model")
    lagrangian = raw.get("lagrangian")
    weights = raw.get("weights", {})
    if not isinstance(cost_model, dict) or not isinstance(lagrangian, dict):
        raise CompilerError(f"atom {raw.get('atom_id')} lacks cost_model/lagrangian")
    charged_bytes = int(cost_model.get("estimated_charged_bytes", 0))
    if charged_bytes <= 0:
        raise CompilerError(f"atom {raw.get('atom_id')} has nonpositive charged bytes")
    residual_pixels = int(raw.get("residual_pixels", 0))
    marginal = _finite_float(
        lagrangian.get("estimated_marginal_score_saved_proxy"),
        field="estimated_marginal_score_saved_proxy",
    )
    rate_cost = _finite_float(
        lagrangian.get("estimated_rate_score_cost"),
        field="estimated_rate_score_cost",
        default=LAMBDA_RATE * charged_bytes,
    )
    net = _finite_float(
        lagrangian.get("estimated_lagrangian_net_proxy"),
        field="estimated_lagrangian_net_proxy",
        default=marginal - rate_cost,
    )
    if not isinstance(weights, dict):
        weights = {}
    pair_indices = _int_tuple(raw.get("pair_indices"), field="pair_indices") or _identity_int(identity, "pair_index")
    frame_indices = _int_tuple(raw.get("frame_indices"), field="frame_indices") or _identity_int(identity, "frame_index")
    class_ids = _int_tuple(raw.get("class_ids"), field="class_ids") or _identity_int(identity, "class_id")
    return Atom(
        source_ledger=source_ledger,
        atom_id=str(raw.get("atom_id")),
        family=str(raw.get("atom_family")),
        identity=identity,
        pair_indices=tuple(sorted(set(pair_indices))),
        frame_indices=tuple(sorted(set(frame_indices))),
        class_ids=tuple(sorted(set(class_ids))),
        residual_pixels=residual_pixels,
        charged_bytes=charged_bytes,
        marginal_score_saved_proxy=marginal,
        rate_score_cost=rate_cost,
        lagrangian_net_proxy=net,
        weighted_residual_pixel_proxy=_finite_float(
            weights.get("weighted_residual_pixel_proxy"),
            field="weighted_residual_pixel_proxy",
            default=float(residual_pixels),
        ),
        boundary_pixel_fraction=_finite_float(raw.get("boundary_pixel_fraction"), field="boundary_pixel_fraction"),
        component_pair_weight=_finite_float(
            weights.get("component_pair_weight_pixel_mean"),
            field="component_pair_weight_pixel_mean",
            default=1.0,
        ),
        hard_pair_weight=_finite_float(
            weights.get("hard_pair_weight_pixel_mean"),
            field="hard_pair_weight_pixel_mean",
            default=1.0,
        ),
        foveal_weight=_finite_float(weights.get("foveal_weight_pixel_mean"), field="foveal_weight_pixel_mean", default=1.0),
        confusion_fraction=_confusion_fraction(raw, residual_pixels=residual_pixels),
        expected_base_runs_per_row=expected_base,
    )


def load_atoms(ledger_jsons: Iterable[Path], *, max_source_atoms: int) -> tuple[list[Atom], list[dict[str, Any]]]:
    atoms: list[Atom] = []
    inputs: list[dict[str, Any]] = []
    for path in ledger_jsons:
        resolved = path.resolve()
        payload = _read_json(resolved)
        if not isinstance(payload, dict):
            raise CompilerError(f"{resolved} must contain a JSON object")
        if payload.get("score_claim") is True:
            raise CompilerError(f"{resolved} has score_claim=true; expected planning-only ledger")
        raw_atoms = payload.get("top_atoms")
        if not isinstance(raw_atoms, list):
            raise CompilerError(f"{resolved} must contain top_atoms")
        expected_base = _candidate_expected_base_runs_per_row(payload)
        for raw in raw_atoms[:max_source_atoms]:
            if not isinstance(raw, dict):
                raise CompilerError(f"{resolved}: top_atoms entries must be objects")
            atoms.append(_atom_from_json(raw, source_ledger=resolved.stem, expected_base=expected_base))
        inputs.append(
            {
                "kind": "atom_ledger",
                "path": str(resolved),
                "sha256": _sha256_file(resolved),
                "schema": payload.get("schema"),
                "score_claim": payload.get("score_claim"),
                "top_atoms_available": len(raw_atoms),
                "top_atoms_read": min(len(raw_atoms), max_source_atoms),
                "expected_builder_base_runs_per_row": expected_base,
            }
        )
    return atoms, inputs


def _extract_score(payload: dict[str, Any]) -> tuple[float | None, float | None, float | None, int | None]:
    score = payload.get("score_recomputed_from_components", payload.get("final_score"))
    pose = payload.get("avg_posenet_dist")
    seg = payload.get("avg_segnet_dist")
    bytes_value = payload.get("archive_size_bytes", payload.get("archive_bytes"))
    return (
        float(score) if isinstance(score, (int, float)) else None,
        float(pose) if isinstance(pose, (int, float)) else None,
        float(seg) if isinstance(seg, (int, float)) else None,
        int(bytes_value) if isinstance(bytes_value, int) else None,
    )


def load_negative_traces(
    *,
    exact_negative_jsons: Iterable[Path],
    negative_manifests: Iterable[Path],
) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    active_filters = {
        "reject_global_row_span_shape",
        "reject_full_residual_like_policy_spread",
        "reject_low_hotspot_density_policy",
    }
    for path in exact_negative_jsons:
        resolved = path.resolve()
        payload = _read_json(resolved)
        if not isinstance(payload, dict):
            raise CompilerError(f"{resolved} must contain a JSON object")
        score, pose, seg, archive_bytes = _extract_score(payload)
        failure_modes: list[str] = []
        if pose is not None and pose > 0.02:
            failure_modes.append("posenet_component_collapse")
        if seg is not None and seg > 0.01:
            failure_modes.append("segnet_component_collapse")
        if score is not None and score > 1.0:
            failure_modes.append("large_exact_negative_score")
        records.append(
            {
                "kind": "exact_negative_eval",
                "path": str(resolved),
                "sha256": _sha256_file(resolved),
                "score_recomputed_from_components": score,
                "avg_posenet_dist": pose,
                "avg_segnet_dist": seg,
                "archive_size_bytes": archive_bytes,
                "failure_modes": failure_modes or ["negative_trace_supplied"],
                "score_claim": False,
            }
        )
    for path in negative_manifests:
        resolved = path.resolve()
        payload = _read_json(resolved)
        if not isinstance(payload, dict):
            raise CompilerError(f"{resolved} must contain a JSON object")
        mode = None
        for key in ("cmg3", "pmg_hotspot_cmg3"):
            if isinstance(payload.get(key), dict):
                mode = payload[key].get("mode")
        if isinstance(mode, str) and "row_span" in mode:
            active_filters.add("reject_row_span_manifest_shape")
        records.append(
            {
                "kind": "negative_candidate_manifest",
                "path": str(resolved),
                "sha256": _sha256_file(resolved),
                "schema": payload.get("schema"),
                "mask_geometry_mode": mode,
                "score_claim": payload.get("score_claim", False),
            }
        )
    return records, sorted(active_filters)


def hotspot_signal(atom: Atom) -> float:
    component = max(0.0, atom.component_pair_weight - 1.0)
    hard = max(0.0, atom.hard_pair_weight - 1.0)
    foveal = max(0.0, atom.foveal_weight - 1.0)
    boundary = max(0.0, atom.boundary_pixel_fraction)
    return (
        0.45 * component
        + 0.55 * hard
        + 0.20 * foveal
        + 1.50 * boundary
        + 0.70 * atom.confusion_fraction
    )


def rank_atom(atom: Atom, *, config: CompilerConfig) -> RankedAtom:
    signal = hotspot_signal(atom)
    break_even_bytes = atom.marginal_score_saved_proxy / LAMBDA_RATE if LAMBDA_RATE > 0.0 else 0.0
    rank_score = (
        atom.lagrangian_net_proxy
        + config.hotspot_signal_weight * signal
        + 1.0e-9 * math.log1p(max(atom.weighted_residual_pixel_proxy, 0.0))
    )
    reasons: list[str] = []
    if atom.family != "row_run":
        reasons.append("unsupported_non_row_run_atom")
    else:
        class_id = atom.row_run_key[4]
        if not (CMG3A_NONZERO_CLASS_MIN <= class_id < CMG3A_NONZERO_CLASS_MAX_EXCLUSIVE):
            reasons.append("builder_incompatible_class_id_for_cmg3a_nonzero_row_run")
    if atom.lagrangian_net_proxy <= 0.0:
        reasons.append("nonpositive_lagrangian_net_proxy")
    if (
        atom.family == "row_run"
        and atom.span_pixels >= config.max_arbitrary_span_pixels
        and signal < config.min_atom_hotspot_signal
    ):
        reasons.append("known_negative_arbitrary_row_span_shape")
    if atom.confusion_fraction <= 0.0 and signal < 0.25:
        reasons.append("low_confusion_low_hotspot_signal")
    return RankedAtom(
        atom=atom,
        rank_score=rank_score,
        hotspot_signal=signal,
        break_even_bytes=break_even_bytes,
        reject_reasons=tuple(reasons),
    )


def _ranked_atom_record(item: RankedAtom, *, include_identity: bool = True) -> dict[str, Any]:
    atom = item.atom
    span_pixels = atom.span_pixels if atom.family == "row_run" else int(atom.residual_pixels)
    record: dict[str, Any] = {
        "atom_id": atom.atom_id,
        "source_ledger": atom.source_ledger,
        "atom_family": atom.family,
        "score_claim": False,
        "rank_score": round(item.rank_score, 12),
        "hotspot_signal": round(item.hotspot_signal, 12),
        "break_even_bytes": round(item.break_even_bytes, 6),
        "charged_bytes_proxy": int(atom.charged_bytes),
        "rate_score_cost": round(atom.rate_score_cost, 12),
        "marginal_score_saved_proxy": round(atom.marginal_score_saved_proxy, 12),
        "lagrangian_net_proxy": round(atom.lagrangian_net_proxy, 12),
        "residual_pixels": int(atom.residual_pixels),
        "span_pixels": int(span_pixels),
        "pair_indices": list(atom.pair_indices),
        "frame_indices": list(atom.frame_indices),
        "class_ids": list(atom.class_ids),
        "signals": {
            "component_pair_weight": round(atom.component_pair_weight, 12),
            "hard_pair_weight": round(atom.hard_pair_weight, 12),
            "foveal_weight": round(atom.foveal_weight, 12),
            "boundary_pixel_fraction": round(atom.boundary_pixel_fraction, 12),
            "confusion_fraction": round(atom.confusion_fraction, 12),
        },
    }
    if atom.family == "row_run":
        record["field_coordinates"] = _field_coordinates(atom)
    if include_identity:
        record["identity"] = dict(atom.identity)
    if item.reject_reasons:
        record["reject_reasons"] = list(item.reject_reasons)
    return record


def _field_coordinates(atom: Atom) -> dict[str, Any]:
    _frame, y, x0, x1, _class_id = atom.row_run_key
    mid_x = 0.5 * (float(x0) + float(x1))
    mid_y = float(y) + 0.5
    center_x = 256.0
    center_y = 174.0
    dx = (mid_x - center_x) / 256.0
    dy = (mid_y - center_y) / 192.0
    return {
        "scorer_space": "mask_512x384",
        "mid_x": round(mid_x, 6),
        "mid_y": round(mid_y, 6),
        "dx_from_default_foe": round(dx, 12),
        "dy_from_default_horizon": round(dy, 12),
        "radial_from_default_foe": round(math.sqrt(dx * dx + dy * dy), 12),
    }


def _planning_field_atoms(config: CompilerConfig) -> list[dict[str, Any]]:
    atoms = [
        {
            "atom_id": "c067_field_center_xy_q12",
            "atom_family": "learnable_foveation_center",
            "parameters": {
                "center_x": float(config.foveation_center_x),
                "center_y": float(config.foveation_center_y),
                "quantization": "signed_q12_delta_from_default_foe",
            },
            "estimated_charged_bytes": 4,
            "scorer_target": "PoseNet/SegNet hard-pair geometry near road horizon and ego-motion focus",
        },
        {
            "atom_id": "c067_field_anisotropic_hyperbolic_shape_q10",
            "atom_family": "learnable_anisotropic_hyperbolic_foveation",
            "parameters": {
                "radial_log_scale": "learned_q10",
                "tangential_log_scale": "learned_q10",
                "curvature": "learned_q10",
                "orientation": "learned_q10",
            },
            "estimated_charged_bytes": int(config.foveation_param_bytes),
            "scorer_target": "Preserve distant small-object/lane-boundary atoms without global full-residual spend",
        },
        {
            "atom_id": "c067_field_horizon_band_q10",
            "atom_family": "learnable_horizon_band",
            "parameters": {
                "horizon_y": float(config.horizon_y),
                "bandwidth": "learned_q10",
                "falloff": "learned_q10",
            },
            "estimated_charged_bytes": 6,
            "scorer_target": "Road/horizon row-run trust-region weighting for hard pairs",
        },
        {
            "atom_id": "c067_low_rank_hotspot_basis_selectors",
            "atom_family": "low_dimensional_subspace_basis",
            "parameters": {
                "basis": [
                    "constant",
                    "dx_from_foe",
                    "dy_from_horizon",
                    "radial_from_foe",
                    "anisotropic_radial",
                    "boundary_fraction",
                    "component_trace_weight",
                    "confusion_fraction",
                ][: max(1, int(config.low_rank_modes))],
                "selection": "learned_or_waterfilled_at_compress_time_only",
            },
            "estimated_charged_bytes": max(2, 2 * int(config.low_rank_modes)),
            "scorer_target": "Non-arbitrary atom ranking over a small learnable field basis",
        },
    ]
    for atom in atoms:
        bytes_value = int(atom["estimated_charged_bytes"])
        atom["score_claim"] = False
        atom["evidence_grade"] = "external_design_motivation_plus_planning_only"
        atom["rate_score_cost"] = round(LAMBDA_RATE * bytes_value, 12)
        atom["break_even_component_score_improvement_required"] = round(LAMBDA_RATE * bytes_value, 12)
        atom["exact_eval_branch_rule"] = {
            "score_claim": False,
            "charged_payload_required": True,
            "canonical_score_source_required": CUDA_AUTH_EVAL_PATH,
        }
    return atoms


def _dedupe_ranked_atoms(ranked: list[RankedAtom]) -> list[RankedAtom]:
    by_key: dict[tuple[int, int, int, int, int], RankedAtom] = {}
    for item in ranked:
        if item.atom.family != "row_run":
            continue
        key = item.atom.row_run_key
        current = by_key.get(key)
        if current is None or (
            item.rank_score,
            item.hotspot_signal,
            item.atom.atom_id,
        ) > (
            current.rank_score,
            current.hotspot_signal,
            current.atom.atom_id,
        ):
            by_key[key] = item
    return sorted(
        by_key.values(),
        key=lambda item: (
            -item.rank_score,
            -item.hotspot_signal,
            item.atom.source_ledger,
            item.atom.atom_id,
        ),
    )


def _policy_filter_reasons(selected: list[RankedAtom], *, config: CompilerConfig) -> list[str]:
    if not selected:
        return ["empty_policy"]
    selected_atoms = [item.atom for item in selected]
    selected_bytes = sum(atom.charged_bytes for atom in selected_atoms)
    hotspot_count = sum(1 for item in selected if item.hotspot_signal >= config.min_atom_hotspot_signal)
    hotspot_fraction = hotspot_count / max(len(selected), 1)
    pair_count = len({pair for atom in selected_atoms for pair in atom.pair_indices})
    frame_count = len({frame for atom in selected_atoms for frame in atom.frame_indices})
    pair_fraction = pair_count / max(len(selected), 1)
    frame_fraction = frame_count / max(len(selected), 1)
    reasons: list[str] = []
    if selected_bytes > config.max_policy_proxy_bytes:
        reasons.append("known_negative_full_residual_like_proxy_bytes")
    if hotspot_fraction < config.min_policy_hotspot_fraction:
        reasons.append("known_negative_low_hotspot_density_policy")
    if len(selected) >= config.min_policy_spread_atoms and pair_fraction > config.max_policy_pair_fraction:
        reasons.append("known_negative_full_residual_like_pair_spread")
    if len(selected) >= config.min_policy_spread_atoms and frame_fraction > config.max_policy_frame_fraction:
        reasons.append("known_negative_full_residual_like_frame_spread")
    return reasons


def _policy_from_selection(
    selected: list[RankedAtom],
    *,
    config: CompilerConfig,
    selection_size: int,
    output_json: Path,
) -> dict[str, Any]:
    atoms = [item.atom for item in selected]
    selected_bytes = sum(atom.charged_bytes for atom in atoms)
    selected_pixels = sum(atom.residual_pixels for atom in atoms)
    selected_score = sum(item.rank_score for item in selected)
    score_saved = sum(atom.marginal_score_saved_proxy for atom in atoms)
    rate_cost = sum(atom.rate_score_cost for atom in atoms)
    hotspot_fraction = sum(1 for item in selected if item.hotspot_signal >= config.min_atom_hotspot_signal) / max(
        len(selected), 1
    )
    pair_load = Counter(pair for atom in atoms for pair in atom.pair_indices)
    frame_load = Counter(frame for atom in atoms for frame in atom.frame_indices)
    class_load = Counter(class_id for atom in atoms for class_id in atom.class_ids)
    expected_bases = sorted(
        {int(atom.expected_base_runs_per_row) for atom in atoms if atom.expected_base_runs_per_row is not None}
    )
    expected_base = expected_bases[0] if len(expected_bases) == 1 else None
    policy_id = f"{config.policy_prefix}_top{selection_size:04d}"
    builder_tail = (
        f"--field-policy-json {output_json} --field-policy-id {policy_id}"
    )
    if expected_base is not None:
        builder_tail = f"--base-runs-per-row {expected_base} " + builder_tail
    return {
        "policy_id": policy_id,
        "mode": "c067_hotspot_mask_geometry_compiler",
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": "planning_only",
        "builder": "experiments/build_cmg3_adaptive_runs_candidate.py " + builder_tail,
        "selected_atom_count": len(atoms),
        "selected_row_run_atoms": [atom.row_run_policy_atom() for atom in atoms],
        "selected_atom_ids": [atom.atom_id for atom in atoms],
        "source_ledgers": sorted({atom.source_ledger for atom in atoms}),
        "required_base_runs_per_row": expected_base,
        "expected_base_runs_per_row": expected_base,
        "expected_base_runs_per_row_set": expected_bases,
        "estimated_proxy": {
            "selected_uncompressed_proxy_bytes": int(selected_bytes),
            "selected_residual_pixels": int(selected_pixels),
            "first_order_score_saved_proxy": round(score_saved, 12),
            "rate_score_cost": round(rate_cost, 12),
            "hotspot_rank_score": round(selected_score, 12),
            "hotspot_density_proxy": round(selected_score / max(selected_bytes, 1), 12),
            "hotspot_atom_fraction": round(hotspot_fraction, 12),
            "break_even_bytes": round(score_saved / LAMBDA_RATE, 6),
        },
        "support": {
            "top_pair_indices_by_selected_atom_count": [pair for pair, _count in pair_load.most_common(24)],
            "top_frame_indices_by_selected_atom_count": [frame for frame, _count in frame_load.most_common(24)],
            "class_atom_counts": {str(k): int(class_load[k]) for k in sorted(class_load)},
            "low_dimensional_field_basis": [
                "constant",
                "dx_from_foe",
                "dy_from_horizon",
                "radial_from_foe",
                "anisotropic_radial",
                "boundary_fraction",
                "component_trace_weight",
                "confusion_fraction",
            ][: max(1, int(config.low_rank_modes))],
        },
        "required_exact_eval_branch_rule": {
            "score_claim": False,
            "no_gpu_dispatch_from_this_plan": True,
            "branch": "build-byte-closed-archive -> byte/provenance screen -> exact CUDA diagnostic -> T4/equivalent replay only if component gates survive",
            "dispatch_claim_required_before_remote_job": True,
            "canonical_score_source_required": CUDA_AUTH_EVAL_PATH,
        },
    }


def _safe_builder_command(
    *,
    policy: dict[str, Any] | None,
    output_json: Path,
    frontier_archive: Path | None,
    decoded_mask_array: Path | None,
    output_dir: Path | None,
    target_body_bytes: int | None,
) -> list[str] | None:
    if policy is None or frontier_archive is None or decoded_mask_array is None or output_dir is None:
        return None
    command = [
        "python",
        "experiments/build_cmg3_adaptive_runs_candidate.py",
        "--frontier-archive",
        str(frontier_archive),
        "--decoded-mask-array",
        str(decoded_mask_array),
        "--output-dir",
        str(output_dir),
        "--field-policy-json",
        str(output_json),
        "--field-policy-id",
        str(policy["policy_id"]),
        "--body-search-mode",
        "auto",
        "--force",
    ]
    if policy.get("required_base_runs_per_row") is not None:
        command.extend(["--base-runs-per-row", str(policy["required_base_runs_per_row"])])
    if target_body_bytes is not None:
        command.extend(["--target-body-bytes", str(int(target_body_bytes))])
    else:
        command.extend(["--target-extra-runs", str(int(policy["selected_atom_count"]))])
    return command


def build_plan(
    *,
    ledger_jsons: list[Path],
    output_json: Path,
    exact_negative_jsons: list[Path] | None = None,
    negative_manifests: list[Path] | None = None,
    candidate_sizes: tuple[int, ...] = DEFAULT_CANDIDATE_SIZES,
    max_source_atoms: int = 2048,
    policy_prefix: str = "c067_hotspot_geometry",
    min_policy_hotspot_fraction: float = 0.65,
    min_atom_hotspot_signal: float = 1.0,
    max_arbitrary_span_pixels: int = 384,
    max_policy_proxy_bytes: int = 64_000,
    frontier_archive: Path | None = None,
    decoded_mask_array: Path | None = None,
    builder_output_dir: Path | None = None,
    target_body_bytes: int | None = None,
    foveation_center_x: float = 256.0,
    foveation_center_y: float = 174.0,
    horizon_y: float = 174.0,
    low_rank_modes: int = 8,
    foveation_param_bytes: int = 32,
) -> dict[str, Any]:
    if not ledger_jsons:
        raise CompilerError("at least one --ledger-json is required")
    if max_source_atoms <= 0:
        raise CompilerError("max_source_atoms must be positive")
    if not candidate_sizes or any(size <= 0 for size in candidate_sizes):
        raise CompilerError("candidate_sizes must contain positive integers")
    if not (0.0 <= min_policy_hotspot_fraction <= 1.0):
        raise CompilerError("min_policy_hotspot_fraction must be in [0,1]")
    if low_rank_modes <= 0:
        raise CompilerError("low_rank_modes must be positive")
    if foveation_param_bytes <= 0:
        raise CompilerError("foveation_param_bytes must be positive")
    config = CompilerConfig(
        candidate_sizes=tuple(sorted(set(int(size) for size in candidate_sizes))),
        max_source_atoms=max_source_atoms,
        policy_prefix=policy_prefix,
        min_policy_hotspot_fraction=min_policy_hotspot_fraction,
        min_atom_hotspot_signal=min_atom_hotspot_signal,
        max_arbitrary_span_pixels=max_arbitrary_span_pixels,
        max_policy_proxy_bytes=max_policy_proxy_bytes,
        foveation_center_x=foveation_center_x,
        foveation_center_y=foveation_center_y,
        horizon_y=horizon_y,
        low_rank_modes=low_rank_modes,
        foveation_param_bytes=foveation_param_bytes,
    )
    atoms, input_records = load_atoms(ledger_jsons, max_source_atoms=max_source_atoms)
    negative_records, active_negative_filters = load_negative_traces(
        exact_negative_jsons=exact_negative_jsons or [],
        negative_manifests=negative_manifests or [],
    )
    ranked_all = _dedupe_ranked_atoms([rank_atom(atom, config=config) for atom in atoms])
    selected_pool = [item for item in ranked_all if not item.reject_reasons]
    rejected_atoms = [item for item in ranked_all if item.reject_reasons]

    policies: list[dict[str, Any]] = []
    filtered_policies: list[dict[str, Any]] = []
    for size in config.candidate_sizes:
        if len(selected_pool) < size:
            continue
        selected = selected_pool[:size]
        policy = _policy_from_selection(selected, config=config, selection_size=size, output_json=output_json)
        reasons = _policy_filter_reasons(selected, config=config)
        if reasons:
            filtered_policies.append(
                {
                    "policy_id": policy["policy_id"],
                    "selected_atom_count": policy["selected_atom_count"],
                    "estimated_proxy": policy["estimated_proxy"],
                    "filter_reasons": reasons,
                    "score_claim": False,
                }
            )
        else:
            policies.append(policy)

    best_policy = policies[0] if policies else None
    safe_command = _safe_builder_command(
        policy=best_policy,
        output_json=output_json,
        frontier_archive=frontier_archive,
        decoded_mask_array=decoded_mask_array,
        output_dir=builder_output_dir,
        target_body_bytes=target_body_bytes,
    )
    family_counts = Counter(atom.family for atom in atoms)
    reject_reason_counts = Counter(reason for item in rejected_atoms for reason in item.reject_reasons)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "compiler_schema": COMPILER_SCHEMA,
        "producer": TOOL,
        "score_claim": False,
        "no_score_claim": True,
        "promotion_eligible": False,
        "evidence_grade": "planning_only",
        "cuda_jobs_launched": False,
        "remote_jobs_dispatched": False,
        "canonical_score_source_required": CUDA_AUTH_EVAL_PATH,
        "configuration": {
            "candidate_sizes": list(config.candidate_sizes),
            "max_source_atoms": int(max_source_atoms),
            "policy_prefix": policy_prefix,
            "min_atom_hotspot_signal": min_atom_hotspot_signal,
            "min_policy_hotspot_fraction": min_policy_hotspot_fraction,
            "max_arbitrary_span_pixels": int(max_arbitrary_span_pixels),
            "max_policy_proxy_bytes": int(max_policy_proxy_bytes),
            "foveation_center_x": float(foveation_center_x),
            "foveation_center_y": float(foveation_center_y),
            "horizon_y": float(horizon_y),
            "low_rank_modes": int(low_rank_modes),
            "foveation_param_bytes": int(foveation_param_bytes),
            "lambda_rate": LAMBDA_RATE,
            "break_even_bytes_per_score": ORIGINAL_VIDEO_BYTES / 25.0,
            "builder_class_contract": {
                "policy_target": "build_cmg3_adaptive_runs_candidate.py",
                "valid_row_run_class_id_min_inclusive": CMG3A_NONZERO_CLASS_MIN,
                "valid_row_run_class_id_max_exclusive": CMG3A_NONZERO_CLASS_MAX_EXCLUSIVE,
                "reason": "CMG3A adaptive policies select nonzero row-run atoms; class 0 is background/default and must not be emitted as a selected row-run repair atom.",
            },
        },
        "external_design_motivation": {
            "name": "Telescope: Learnable Hyperbolic Foveation for Ultra-Long-Range Object Detection",
            "arxiv": "2604.06332",
            "submitted": "2026-04-07",
            "authors": ["Ewen", "Rivkin", "Bijelic", "Heide"],
            "evidence_grade": "external",
            "score_claim": False,
            "contest_use": (
                "Design prior only: motivates learnable anisotropic/hyperbolic/foveated atom fields around "
                "road, horizon, ego-motion, hard-pair, and confusion geometry. It is not contest evidence."
            ),
        },
        "planning_field_atoms": _planning_field_atoms(config),
        "inputs": input_records,
        "exact_negative_trace_inputs": negative_records,
        "active_negative_shape_filters": active_negative_filters,
        "atom_summary": {
            "source_atom_count": len(atoms),
            "deduped_row_run_atom_count": len(ranked_all),
            "selectable_atom_count": len(selected_pool),
            "rejected_atom_count": len(rejected_atoms),
            "atom_family_counts": {str(k): int(family_counts[k]) for k in sorted(family_counts)},
            "rejected_atom_reason_counts": {
                str(k): int(reject_reason_counts[k]) for k in sorted(reject_reason_counts)
            },
        },
        "selected_atoms": [_ranked_atom_record(item) for item in selected_pool[: min(len(selected_pool), 128)]],
        "rejected_atoms": [_ranked_atom_record(item) for item in rejected_atoms[: min(len(rejected_atoms), 128)]],
        "candidate_policies": policies,
        "filtered_candidate_policies": {
            "known_negative_shape": filtered_policies,
            "filter_policy": (
                "Policies resembling exact-negative global row-span/full-residual failures are filtered by "
                "hotspot fraction, pair/frame spread, and proxy-byte caps before any archive build."
            ),
        },
        "required_exact_eval_branch_rule": {
            "score_claim": False,
            "no_gpu_dispatch_from_this_plan": True,
            "dispatch_claim_required_before_remote_job": True,
            "canonical_score_source_required": CUDA_AUTH_EVAL_PATH,
            "branch": [
                "build the candidate archive with the exact emitted field policy",
                "record archive bytes, SHA-256, payload closure, and runtime tree hash",
                "run exact CUDA diagnostic auth eval only after byte/provenance screen",
                "promote or rank only after T4/equivalent replay of identical archive bytes survives component gates",
            ],
        },
        "concrete_builder_command_if_safe": safe_command,
        "required_next_steps": [
            "Use concrete_builder_command_if_safe only after reviewing the selected atoms and existing dispatch claims.",
            "Do not launch Lightning/Vast/Modal evals from this planner output; claim a lane before any remote dispatch.",
            "Treat every output here as planning-only until exact CUDA auth eval on exact archive bytes exists.",
        ],
    }
    _write_json(output_json, payload)
    return payload


def _parse_positive_ints(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers") from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-json", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--exact-negative-json", type=Path, action="append", default=[])
    parser.add_argument("--negative-manifest", type=Path, action="append", default=[])
    parser.add_argument("--candidate-sizes", type=_parse_positive_ints, default=DEFAULT_CANDIDATE_SIZES)
    parser.add_argument("--max-source-atoms", type=int, default=2048)
    parser.add_argument("--policy-prefix", default="c067_hotspot_geometry")
    parser.add_argument("--min-policy-hotspot-fraction", type=float, default=0.65)
    parser.add_argument("--min-atom-hotspot-signal", type=float, default=1.0)
    parser.add_argument("--max-arbitrary-span-pixels", type=int, default=384)
    parser.add_argument("--max-policy-proxy-bytes", type=int, default=64_000)
    parser.add_argument("--frontier-archive", type=Path, default=None)
    parser.add_argument("--decoded-mask-array", type=Path, default=None)
    parser.add_argument("--builder-output-dir", type=Path, default=None)
    parser.add_argument("--target-body-bytes", type=int, default=None)
    parser.add_argument("--foveation-center-x", type=float, default=256.0)
    parser.add_argument("--foveation-center-y", type=float, default=174.0)
    parser.add_argument("--horizon-y", type=float, default=174.0)
    parser.add_argument("--low-rank-modes", type=int, default=8)
    parser.add_argument("--foveation-param-bytes", type=int, default=32)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_plan(
        ledger_jsons=args.ledger_json,
        output_json=args.output_json,
        exact_negative_jsons=args.exact_negative_json,
        negative_manifests=args.negative_manifest,
        candidate_sizes=args.candidate_sizes,
        max_source_atoms=args.max_source_atoms,
        policy_prefix=args.policy_prefix,
        min_policy_hotspot_fraction=args.min_policy_hotspot_fraction,
        min_atom_hotspot_signal=args.min_atom_hotspot_signal,
        max_arbitrary_span_pixels=args.max_arbitrary_span_pixels,
        max_policy_proxy_bytes=args.max_policy_proxy_bytes,
        frontier_archive=args.frontier_archive,
        decoded_mask_array=args.decoded_mask_array,
        builder_output_dir=args.builder_output_dir,
        target_body_bytes=args.target_body_bytes,
        foveation_center_x=args.foveation_center_x,
        foveation_center_y=args.foveation_center_y,
        horizon_y=args.horizon_y,
        low_rank_modes=args.low_rank_modes,
        foveation_param_bytes=args.foveation_param_bytes,
    )
    print(
        json.dumps(
            {
                "output_json": str(args.output_json.resolve()),
                "candidate_policy_count": len(payload["candidate_policies"]),
                "filtered_policy_count": len(payload["filtered_candidate_policies"]["known_negative_shape"]),
                "score_claim": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
