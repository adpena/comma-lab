#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded real-n600 joint seg/pose interval solve and rate telemetry.

Each invocation is capped at twelve real cache pairs.  Per-pair NPZ stages and
an atomic JSON state make the solve resumable; multiple receipts compose into
the n600-capable evidence surface.  This is advisory measurement only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
for path in (REPO, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tac.optimization.joint_seg_pose_rate import (  # noqa: E402
    JointSolveError,
    MarginBandConfig,
    derive_hyperplane_channel_band,
    derive_margin_rgb_band,
    generated_fill_predictor,
    range_payload_bytes_and_tiles,
    solve_interval_frame,
    solve_measured_waterfill,
)
from tac.optimization.uint8_lattice_feasibility import DisjointResizeOperator  # noqa: E402
from tac.optimization.vjp_custody import (  # noqa: E402
    CustodiedPair,
    VJPCustodyError,
    largest_feasible_pose_step,
    linearized_pose_delta6,
    load_vjp_manifest,
)
from tools.measure_uint8_lattice_feasibility import (  # noqa: E402
    _sha256_file,
    _stat_tree_snapshot,
    stored_npy_memmap,
)


def _resolve_m_safe() -> float:
    """m_safe = headroom * delta_R through the canonical law (n600 artifact, ddm_dr1
    2026-09-04); the exact n600 fallback keeps the tool runnable without ``tac``."""
    try:
        from tac.canonical_equations.margin_band_satisficing_threshold_20260712 import (
            resolve_margin_band_threshold,
        )

        return float(resolve_margin_band_threshold().m_safe)
    except Exception:  # tool must stay runnable; the value is the same law's own fallback
        return 0.04376363754272461


_M_SAFE = _resolve_m_safe()

SCHEMA = "joint_seg_pose_inverse_rate_receipt.v1"
STATE_SCHEMA = "joint_seg_pose_inverse_rate_state.v1"
BINDINGNESS_SCHEMA = "joint_seg_pose_pair_bindingness.v1"
MAX_SUBSET = 12
CAMERA_HW = (874, 1164)
SCORER_HW = (384, 512)
DEFAULT_CACHE = Path("/Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
DEFAULT_UPSTREAM = Path("/Users/adpena/Projects/pact/upstream")
SACRED = Path("/Users/adpena/Projects/pact/experiments/results/levelset_n600_witness_20260717T113932Z")
POINTER = "0.1910828242 [contest-CPU] UNMOVED"
SEED = 20260719
VJP_RIVAL_DISAGREEMENT_SCOPE = (
    "gradient-enabled VJP proposal arrangement versus inference-mode hard-oracle "
    "recomputation; rival identity is proposal-only and is not winner/Seg authority"
)
POSE_SEG_CROSSOVER_D_POSE = 2.5e-4
RUNG_E_PROVENANCE = {
    "rung": "E",
    "point": "joint_chosen_yhat_residual_vs_generated_free_predictor",
    "decision_coordinates": (
        "exact reachable scorer-plane integer numerators for frame0 and frame1"
    ),
    "free_predictor": (
        "declared generated piecewise-constant fill of the counted scorer-plane "
        "description"
    ),
    "counted_payload": "chosen_yhat minus generated-free-predictor range residual",
    "authority_scope": (
        "first non-toy payload-coordinate point; advisory only; not a receiver/archive "
        "or contest-score claim"
    ),
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(_canonical(value) + b"\n")
    os.replace(tmp, path)


def _array_manifest(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": _sha256_array(array),
    }


def _bindingness_arrays(
    *,
    binding0: np.ndarray,
    binding1: np.ndarray,
    positive_seg_radius: np.ndarray,
    fallback0: np.ndarray | None,
    fallback1: np.ndarray | None,
) -> tuple[dict[str, np.ndarray], dict[str, bool]]:
    binding0_array = np.asarray(binding0)
    binding1_array = np.asarray(binding1)
    if binding0_array.shape != binding1_array.shape:
        raise JointSolveError("bindingness frame maps have different geometry")
    positive_array = np.asarray(positive_seg_radius, dtype=np.bool_)
    if positive_array.shape != binding1_array.shape:
        try:
            positive_array = np.broadcast_to(
                positive_array[..., None], binding1_array.shape
            )
        except ValueError as exc:
            raise JointSolveError(
                "positive Seg-radius map differs from binding-map geometry"
            ) from exc

    fallback_present = {
        "frame0_exact_source_fallback": fallback0 is not None,
        "frame1_exact_source_fallback": fallback1 is not None,
    }
    fallback0_array = (
        np.zeros(binding0_array.shape, dtype=np.bool_)
        if fallback0 is None
        else np.asarray(fallback0, dtype=np.bool_)
    )
    fallback1_array = (
        np.zeros(binding1_array.shape, dtype=np.bool_)
        if fallback1 is None
        else np.asarray(fallback1, dtype=np.bool_)
    )
    if fallback0_array.shape != binding0_array.shape or fallback1_array.shape != binding1_array.shape:
        raise JointSolveError("exact-source fallback map differs from binding-map geometry")
    return {
        "frame0_binding_map": np.ascontiguousarray(binding0_array),
        "frame1_binding_map": np.ascontiguousarray(binding1_array),
        "positive_seg_radius_map": np.ascontiguousarray(positive_array),
        "frame0_exact_source_fallback_map": np.ascontiguousarray(fallback0_array),
        "frame1_exact_source_fallback_map": np.ascontiguousarray(fallback1_array),
    }, fallback_present


def _bindingness_reference(path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": BINDINGNESS_SCHEMA,
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "metadata_sha256": hashlib.sha256(_canonical(metadata)).hexdigest(),
        "arrays": metadata["arrays"],
    }


def _validate_bindingness_sidecar(
    reference: dict[str, Any],
    *,
    pair_id: int,
    config_sha256: str,
    expected_arrays: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Cross-validate the compressed bytes, embedded metadata, and full maps."""

    if reference.get("schema") != BINDINGNESS_SCHEMA:
        raise JointSolveError("bindingness reference schema mismatch")
    path = Path(str(reference.get("path", ""))).resolve()
    if not path.is_file():
        raise JointSolveError("bindingness sidecar is missing")
    if path.stat().st_size != reference.get("bytes") or _sha256_file(path) != reference.get("sha256"):
        raise JointSolveError("bindingness sidecar byte custody mismatch")
    try:
        with np.load(path, allow_pickle=False) as data:
            expected_keys = {
                "metadata_json",
                "frame0_binding_map",
                "frame1_binding_map",
                "positive_seg_radius_map",
                "frame0_exact_source_fallback_map",
                "frame1_exact_source_fallback_map",
            }
            if set(data.files) != expected_keys:
                raise JointSolveError("bindingness sidecar key set mismatch")
            metadata = json.loads(np.asarray(data["metadata_json"], dtype=np.uint8).tobytes())
            arrays = {key: np.asarray(data[key]) for key in expected_keys - {"metadata_json"}}
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise JointSolveError("bindingness sidecar is unreadable") from exc

    if (
        metadata.get("schema") != BINDINGNESS_SCHEMA
        or metadata.get("pair_id") != pair_id
        or metadata.get("config_sha256") != config_sha256
    ):
        raise JointSolveError("bindingness embedded metadata mismatch")
    if hashlib.sha256(_canonical(metadata)).hexdigest() != reference.get("metadata_sha256"):
        raise JointSolveError("bindingness metadata hash mismatch")
    manifests = {key: _array_manifest(value) for key, value in arrays.items()}
    if manifests != metadata.get("arrays") or manifests != reference.get("arrays"):
        raise JointSolveError("bindingness array custody mismatch")
    if expected_arrays is not None:
        expected_manifests = {
            key: _array_manifest(value) for key, value in expected_arrays.items()
        }
        if expected_manifests != manifests:
            raise JointSolveError("bindingness maps differ from deterministic replay")
    return metadata


def _publish_bindingness_sidecar(
    path: Path,
    *,
    pair_id: int,
    config_sha256: str,
    arrays: dict[str, np.ndarray],
    fallback_present: dict[str, bool],
) -> dict[str, Any]:
    """Atomically publish once, or validate an identical orphan on resume."""

    path = path.resolve()
    manifests = {key: _array_manifest(value) for key, value in arrays.items()}
    metadata = {
        "schema": BINDINGNESS_SCHEMA,
        "pair_id": pair_id,
        "config_sha256": config_sha256,
        "map_semantics": {
            "frame_binding_maps": "0 slack, 1 lower, 2 upper interval binding",
            "positive_seg_radius_map": "true exactly where accepted frame1 Seg radius is positive",
            "exact_source_fallback_maps": (
                "true exactly where bounded search used the conservative exact-source numerator"
            ),
        },
        "fallback_map_present_in_solver_result": fallback_present,
        "arrays": manifests,
    }
    if path.exists():
        reference = _bindingness_reference(path, metadata)
        _validate_bindingness_sidecar(
            reference,
            pair_id=pair_id,
            config_sha256=config_sha256,
            expected_arrays=arrays,
        )
        return reference

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("xb") as handle:
            np.savez_compressed(
                handle,
                metadata_json=np.frombuffer(_canonical(metadata), dtype=np.uint8),
                **arrays,
            )
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError as exc:
            raise JointSolveError(f"immutable bindingness sidecar already exists: {path}") from exc
    finally:
        tmp.unlink(missing_ok=True)
    reference = _bindingness_reference(path, metadata)
    _validate_bindingness_sidecar(
        reference,
        pair_id=pair_id,
        config_sha256=config_sha256,
        expected_arrays=arrays,
    )
    return reference


def _sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def _load_cache(path: Path) -> dict[str, np.memmap]:
    fields = {key: stored_npy_memmap(path, key) for key in ("n_pairs", "gt_f0", "gt_f1", "lstars", "margins", "gt_poses")}
    if int(np.asarray(fields["n_pairs"]).reshape(())) != 600:
        raise JointSolveError("only the real n600 cache is admissible")
    if fields["gt_f0"].shape != (600, *CAMERA_HW, 3) or fields["gt_f1"].shape != (600, *CAMERA_HW, 3):
        raise JointSolveError("real cache camera geometry mismatch")
    if fields["lstars"].shape != (600, *SCORER_HW) or fields["margins"].shape != (600, *SCORER_HW):
        raise JointSolveError("real cache scorer geometry mismatch")
    if fields["gt_poses"].shape != (600, 6):
        raise JointSolveError("real cache pose geometry mismatch")
    return fields


def _load_seg_pullback_sidecar(path: Path, pairs: Sequence[int]) -> dict[int, dict[str, np.ndarray]]:
    """Load a bounded, native-arithmetic winner/rival VJP sidecar fail-closed."""

    with np.load(path, allow_pickle=False) as data:
        required = {"pair_ids", "winner", "rival", "unit_head_normal_pullback_rgb", "pair_norms", "receiver_arithmetic"}
        missing = required.difference(data.files)
        if missing:
            raise JointSolveError(f"Seg pullback sidecar lacks keys: {sorted(missing)}")
        arithmetic = str(np.asarray(data["receiver_arithmetic"]).reshape(()))
        if arithmetic != "native_float32_cpu_torch":
            raise JointSolveError("Seg pullback sidecar receiver arithmetic is not native_float32_cpu_torch")
        ids = [int(x) for x in np.asarray(data["pair_ids"]).tolist()]
        if ids != list(pairs):
            raise JointSolveError("Seg pullback sidecar pair ids/order differ from invocation")
        winner = np.asarray(data["winner"])
        rival = np.asarray(data["rival"])
        pullback = np.asarray(data["unit_head_normal_pullback_rgb"])
        norms = np.asarray(data["pair_norms"])
    expected = (len(ids), *SCORER_HW)
    if winner.shape != expected or rival.shape != expected or norms.shape != expected:
        raise JointSolveError("Seg pullback sidecar arrangement geometry mismatch")
    if pullback.shape != (*expected, 3):
        raise JointSolveError("Seg pullback sidecar RGB-VJP geometry mismatch")
    return {
        pair_id: {"winner": winner[i], "rival": rival[i], "pullback": pullback[i], "pair_norms": norms[i]}
        for i, pair_id in enumerate(ids)
    }


def _is_vjp_manifest(path: Path) -> bool:
    return path.is_dir() or path.suffix.lower() == ".json"


def _custody_document_path(path: Path) -> Path:
    return path / "manifest.json" if path.is_dir() else path


def _load_vjp_rows(path: Path, pairs: Sequence[int]) -> dict[int, CustodiedPair]:
    try:
        return load_vjp_manifest(path.resolve(), list(pairs))
    except VJPCustodyError as exc:
        raise JointSolveError(str(exc)) from exc


def _radius_summary(radius: np.ndarray) -> dict[str, Any]:
    values = np.asarray(radius, dtype=np.float64)
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "p50": float(np.quantile(values, 0.5)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values, initial=0.0)),
    }


def _hard_oracle_band_proposal_summary(attempts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = len(attempts)
    evaluated = [attempt for attempt in attempts if attempt.get("hard_oracle") is not None]
    admitted = sum(bool(attempt.get("PASS")) for attempt in evaluated)
    rejected = len(evaluated) - admitted
    not_evaluated = total - len(evaluated)
    return {
        "attempt_count": total,
        "hard_oracle_evaluated_count": len(evaluated),
        "hard_oracle_admit_count": admitted,
        "hard_oracle_reject_count": rejected,
        "hard_oracle_not_evaluated_count": not_evaluated,
        "hard_oracle_admit_rate_over_attempts": None if total == 0 else admitted / total,
        "hard_oracle_reject_rate_over_attempts": None if total == 0 else rejected / total,
        "hard_oracle_not_evaluated_rate_over_attempts": None if total == 0 else not_evaluated / total,
        "hard_oracle_admit_rate_over_evaluations": (
            None if not evaluated else admitted / len(evaluated)
        ),
    }


def _conservative_exact_source_numerator_fallback_summary(
    result: Any,
) -> dict[str, Any] | None:
    fallback_map = result.conservative_exact_source_numerator_fallback_map
    if fallback_map is None:
        return None
    return {
        "conservative_exact_source_numerator_fallback_count": int(
            np.count_nonzero(fallback_map)
        ),
        "conservative_exact_source_numerator_fallback_map_sha256": _sha256_array(
            fallback_map
        ),
    }


def _vjp_field_summary(custody: CustodiedPair) -> dict[str, Any]:
    local_lipschitz = np.asarray(custody.seg_local_lipschitz, dtype=np.float64)
    q_norm = np.linalg.norm(np.asarray(custody.seg_q, dtype=np.float64), axis=-1)
    positive = local_lipschitz > 0
    unit_errors = np.abs(q_norm[positive] - 1.0)
    zero_support = q_norm[~positive]
    return {
        "measured_lip_local": {
            **_radius_summary(local_lipschitz),
            "zero_count": int(np.count_nonzero(~positive)),
            "zero_fraction": float(np.mean(~positive)),
        },
        "q_unit_norm_error_on_positive_lip": {
            "count": int(unit_errors.size),
            "mean": None if unit_errors.size == 0 else float(np.mean(unit_errors)),
            "p95": None if unit_errors.size == 0 else float(np.quantile(unit_errors, 0.95)),
            "max": None if unit_errors.size == 0 else float(np.max(unit_errors, initial=0.0)),
        },
        "q_norm_max_on_zero_lip": (
            None if zero_support.size == 0 else float(np.max(zero_support, initial=0.0))
        ),
    }


def _load_scorers(upstream: Path, threads: int) -> tuple[Any, Any, Any]:
    if not (upstream / "modules.py").is_file():
        raise JointSolveError(f"frozen upstream missing: {upstream}")
    sys.path.insert(0, str(upstream))
    import torch
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path

    torch.set_num_threads(threads)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    dn = DistortionNet().eval().to("cpu")
    dn.load_state_dicts(posenet_sd_path, segnet_sd_path, torch.device("cpu"))
    for parameter in dn.parameters():
        parameter.requires_grad_(False)
    return dn.segnet, dn.posenet, torch


def _repair_pose_linear_proposal(
    pose_j_y: np.ndarray,
    frame1_delta: np.ndarray,
    frame0_predictor_direction: np.ndarray,
    source_pose_delta6: np.ndarray,
    full_tau_pose: float,
    shrink: float,
) -> dict[str, Any]:
    """Propose against repair-shrunk linear debt, retaining the full hard limit."""

    effective_linear_pose_budget = full_tau_pose * shrink
    proposal = largest_feasible_pose_step(
        pose_j_y,
        frame1_delta,
        frame0_predictor_direction,
        source_pose_delta6,
        effective_linear_pose_budget,
        max_step=shrink,
    )
    return {
        **proposal,
        "effective_linear_pose_budget": effective_linear_pose_budget,
        "full_hard_oracle_pose_limit": full_tau_pose,
    }


def _pose_constraint_activity(
    *,
    pose_proposal: dict[str, Any],
    hard_d_pose: float | None,
    tau_pose: float,
    repair_shrink: float,
) -> dict[str, Any]:
    """Classify proposal-linear and frozen-hard Pose constraints explicitly."""

    feasible = bool(pose_proposal.get("feasible"))
    selected_step = pose_proposal.get("selected_step")
    planned_mse = pose_proposal.get("planned_predictor_step_pose_mse")
    linear_budget = float(pose_proposal["effective_linear_pose_budget"])
    step_tolerance = max(1e-12, abs(repair_shrink) * 1e-10)
    budget_tolerance = max(1e-15, abs(linear_budget) * 1e-10)
    if not feasible or selected_step is None or planned_mse is None:
        linear = {
            "classification": "violated_no_feasible_step",
            "active": None,
            "selected_step": None,
            "unconstrained_repair_step_limit": repair_shrink,
            "step_slack": None,
            "effective_budget": linear_budget,
            "planned_mse": None,
            "budget_slack": None,
        }
    else:
        step_slack = max(0.0, float(repair_shrink) - float(selected_step))
        budget_slack = linear_budget - float(planned_mse)
        active = step_slack > step_tolerance
        linear = {
            "classification": "active" if active else "inactive_slack",
            "active": active,
            "selected_step": float(selected_step),
            "unconstrained_repair_step_limit": float(repair_shrink),
            "step_slack": step_slack,
            "effective_budget": linear_budget,
            "planned_mse": float(planned_mse),
            "budget_slack": budget_slack,
            "at_budget_within_tolerance": abs(budget_slack) <= budget_tolerance,
        }

    hard_tolerance = max(1e-15, abs(tau_pose) * 1e-10)
    if hard_d_pose is None:
        hard = {
            "classification": "not_evaluated",
            "active": None,
            "tau_pose": float(tau_pose),
            "d_pose": None,
            "d_pose_over_tau_pose": None,
            "budget_slack": None,
            "zero_tau": tau_pose == 0.0,
        }
    else:
        hard_slack = float(tau_pose) - float(hard_d_pose)
        violated = hard_slack < -hard_tolerance
        active = not violated and hard_slack <= hard_tolerance
        if violated:
            classification = "violated"
        elif active:
            classification = "active_zero_tau_equality" if tau_pose == 0.0 else "active"
        else:
            classification = "inactive_slack"
        hard = {
            "classification": classification,
            "active": active if not violated else None,
            "tau_pose": float(tau_pose),
            "d_pose": float(hard_d_pose),
            "d_pose_over_tau_pose": (
                None if tau_pose == 0.0 else float(hard_d_pose) / float(tau_pose)
            ),
            "budget_slack": hard_slack,
            "zero_tau": tau_pose == 0.0,
        }

    hypothesis_applicable = tau_pose <= POSE_SEG_CROSSOVER_D_POSE
    hard_hypothesis_confirmed = hard.get("active") is False
    linear_model_inactive = linear.get("active") is False
    return {
        "criterion": {
            "linear": (
                "active iff the Pose model limits selected_step below the current "
                "repair shrink; otherwise inactive with step slack"
            ),
            "hard": (
                "active iff accepted frozen-hard d_pose is at tau_pose within tolerance; "
                "zero tau is an explicit equality case; otherwise inactive with budget slack"
            ),
        },
        "linear": linear,
        "hard": hard,
        "preregistered_hypothesis": (
            "the frozen-hard Pose constraint is inactive/slack at accepted "
            "source-centered Seg-band solutions at or below the approximately "
            "2.5e-4 crossover"
        ),
        "preregistered_hypothesis_at_or_below_crossover": hypothesis_applicable,
        "preregistered_hypothesis_confirmed_for_pair": (
            bool(hard_hypothesis_confirmed) if hypothesis_applicable else None
        ),
        "linear_proposer_inactive_for_pair": linear_model_inactive,
        "linear_and_hard_inactive_for_pair": (
            linear_model_inactive and hard_hypothesis_confirmed
        ),
        "verdict_scope": "this accepted pair and operating point only",
    }


def _rung_e_rate_point(
    *,
    chosen0: np.ndarray,
    chosen1: np.ndarray,
    predictor0: np.ndarray,
    predictor1: np.ndarray,
    rate0: dict[str, Any],
    rate1: dict[str, Any],
) -> dict[str, Any]:
    frames = {}
    for name, chosen, predictor, rate in (
        ("frame0", chosen0, predictor0, rate0),
        ("frame1", chosen1, predictor1, rate1),
    ):
        chosen_array = np.asarray(chosen)
        predictor_array = np.asarray(predictor)
        if chosen_array.shape != predictor_array.shape:
            raise JointSolveError("rung-E chosen/predictor geometry mismatch")
        residual = chosen_array.astype(np.int64) - predictor_array.astype(np.int64)
        frames[name] = {
            "chosen_yhat_numerators_sha256": _sha256_array(chosen_array),
            "generated_free_predictor_numerators_sha256": _sha256_array(predictor_array),
            "residual_numerators_sha256": _sha256_array(residual),
            "residual_nonzero_count": int(np.count_nonzero(residual)),
            "measured_brotli_q11_bytes": int(rate["brotli_q11_bytes"]),
            "measured_zstd_19_bytes": int(rate["zstd_19_bytes"]),
        }
    return {
        "shared_provenance": RUNG_E_PROVENANCE,
        "frames": frames,
    }


def _hard_verdict(segnet: Any, posenet: Any, torch: Any, f0: np.ndarray, f1: np.ndarray, labels: np.ndarray, target_pose: np.ndarray) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    import einops

    pair = torch.from_numpy(np.stack((f0, f1), axis=0)[None]).float()
    x = einops.rearrange(pair, "b t h w c -> b t c h w")
    with torch.inference_mode():
        batch_logits = segnet(segnet.preprocess_input(x))
        if batch_logits.ndim != 4 or batch_logits.shape[0] != x.shape[0]:
            raise JointSolveError(
                "SegNet hard oracle must return 4D BxCxHxW logits with batch "
                "dimension equal to the input batch (B=1); last-frame selection "
                "belongs to SegNet.preprocess_input"
            )
        logits = batch_logits[0]
        argmax = logits.argmax(dim=0).cpu().numpy().astype(np.int64)
        logits_np = logits.cpu().numpy()
        masked = logits_np.copy()
        np.put_along_axis(masked, argmax[None], -np.inf, axis=0)
        rival = masked.argmax(axis=0).astype(np.int64)
        pose_out = posenet(posenet.preprocess_input(x))
        pose = pose_out["pose"] if isinstance(pose_out, dict) else pose_out
        pose6 = pose[0, :6].cpu().numpy().astype(np.float64)
    mismatch = argmax != labels
    return {
        "d_seg": float(np.mean(mismatch)),
        "seg_mismatched_pixels": int(np.count_nonzero(mismatch)),
        "d_pose": float(np.mean((pose6 - target_pose) ** 2)),
        "pose6": pose6.tolist(),
    }, argmax, rival


def _gate_vjp_arrangement(
    *,
    stage_dir: Path,
    pair_id: int,
    config_sha256: str,
    source_control: dict[str, Any],
    custodied_winner: np.ndarray,
    custodied_rival: np.ndarray,
    hard_oracle_winner: np.ndarray,
    hard_oracle_rival: np.ndarray,
) -> dict[str, Any]:
    """Admit proposal rivals, but fail closed on Seg-authority winner drift."""

    winner_disagreement_pixels = int(
        np.count_nonzero(custodied_winner != hard_oracle_winner)
    )
    rival_disagreement_pixels = int(
        np.count_nonzero(custodied_rival != hard_oracle_rival)
    )
    telemetry = {
        "custodied_vs_hard_oracle_winner_disagreement_pixels": (
            winner_disagreement_pixels
        ),
        "inference_vs_vjp_rival_disagreement_pixels": rival_disagreement_pixels,
        "inference_vs_vjp_rival_disagreement_scope": VJP_RIVAL_DISAGREEMENT_SCOPE,
    }
    if winner_disagreement_pixels == 0:
        return telemetry

    failure_path = stage_dir / f"pair_{pair_id:04d}.hard_oracle_refusal.json"
    _atomic_json(
        failure_path,
        {
            "schema": "joint_seg_pose_hard_oracle_refusal.v1",
            "pair_id": pair_id,
            "config_sha256": config_sha256,
            "source_positive_control": source_control,
            **telemetry,
            "refusal": (
                "custodied gradient-enabled VJP winner differs from the "
                "inference-mode hard-oracle winner"
            ),
            "verdict_scope": (
                "custodied winner versus inference-mode hard-oracle winner for "
                "this pair only; rival disagreement is proposal-only and does "
                "not constitute winner/Seg-authority drift"
            ),
        },
    )
    raise JointSolveError(
        f"pair {pair_id} VJP winner differs from inference hard oracle at "
        f"{winner_disagreement_pixels} pixels; durable refusal={failure_path}"
    )


def _attribution(rate: dict[str, Any], labels: np.ndarray | None, rival: np.ndarray | None, margins: np.ndarray | None) -> dict[str, Any]:
    sums: dict[str, dict[str, int]] = defaultdict(lambda: {"brotli_q11_bytes": 0, "zstd_19_bytes": 0, "tiles": 0})
    for tile in rate["tiles"]:
        if labels is None:
            key = "pose_global/frame0"
        else:
            cy = min(SCORER_HW[0] - 1, int(tile["y"] + tile["h"] / 2))
            cx = min(SCORER_HW[1] - 1, int(tile["x"] + tile["w"] / 2))
            margin = float(margins[cy, cx])
            codim = "boundary_codim1" if margin < _M_SAFE else "cell_interior"
            winner_id, rival_id = int(labels[cy, cx]), int(rival[cy, cx])
            key = f"cell_winner_{winner_id}/hyperplane_{winner_id}-{rival_id}/{codim}/frame1"
        row = sums[key]
        row["brotli_q11_bytes"] += int(tile["brotli_q11_bytes"])
        row["zstd_19_bytes"] += int(tile["zstd_19_bytes"])
        row["tiles"] += 1
    return dict(sorted(sums.items()))


def _pair_ids(explicit: Sequence[int] | None, count: int) -> list[int]:
    result = [int(x) for x in explicit] if explicit else [int(x) for x in np.linspace(0, 599, count, dtype=np.int64)]
    if not result or len(result) > MAX_SUBSET or len(set(result)) != len(result) or any(x < 0 or x >= 600 for x in result):
        raise JointSolveError(f"pair selection must contain 1..{MAX_SUBSET} unique ids in [0,600)")
    return result


def run_measurement(args: argparse.Namespace) -> dict[str, Any]:
    cache_path, upstream = args.cache.resolve(), args.upstream.resolve()
    output, state, stage_dir = args.output.resolve(), args.state.resolve(), args.stage_dir.resolve()
    if output.exists():
        raise JointSolveError(f"receipt already exists: {output}")
    if any(str(path).startswith(("/tmp/", "/private/tmp/", "/var/tmp/")) for path in (output, state, stage_dir)):
        raise JointSolveError("durable evidence paths may not use tmp")
    pairs = _pair_ids(args.pair_indices, args.sample_pairs)
    if args.seg_band_scale != 0.0 and args.seg_pullback_sidecar is None:
        raise JointSolveError("positive Seg band requires --seg-pullback-sidecar with native winner/rival VJP custody")
    if args.pose_rgb_band != 0.0:
        raise JointSolveError("positive pose bands require a custodied real PoseNet-6 Jacobian sidecar; zero-band control only")
    if args.pose_jacobian_sidecar is None and args.tau_pose is not None:
        raise JointSolveError("--tau-pose requires --pose-jacobian-sidecar")
    if args.pose_jacobian_sidecar is not None and args.tau_pose is None:
        raise JointSolveError("--pose-jacobian-sidecar requires --tau-pose")
    if args.tau_pose is not None and (not np.isfinite(args.tau_pose) or args.tau_pose < 0):
        raise JointSolveError("--tau-pose must be finite and nonnegative")
    sacred_before = _stat_tree_snapshot(SACRED)
    fields = _load_cache(cache_path)
    config = {
        "schema": STATE_SCHEMA, "pairs": pairs, "seg_band_scale": args.seg_band_scale,
        "local_lipschitz": args.local_lipschitz, "max_seg_rgb_radius": args.max_seg_rgb_radius,
        "pose_rgb_band": args.pose_rgb_band, "pose_tolerance": args.pose_tolerance,
        "repair_steps": args.repair_steps, "max_nodes_per_block": args.max_nodes_per_block,
        "rung_e": RUNG_E_PROVENANCE,
        "seg_pullback_sidecar": None if args.seg_pullback_sidecar is None else str(args.seg_pullback_sidecar.resolve()),
        "seg_pullback_sidecar_sha256": None if args.seg_pullback_sidecar is None else _sha256_file(_custody_document_path(args.seg_pullback_sidecar.resolve())),
        "cache_sha256": _sha256_file(cache_path), "solver_sha256": _sha256_file(SRC / "tac/optimization/joint_seg_pose_rate.py"),
        "tool_sha256": _sha256_file(Path(__file__).resolve()), "predictor": "generated piecewise-constant fill of counted scorer-plane description",
    }
    if args.pose_jacobian_sidecar is not None:
        pose_sidecar = args.pose_jacobian_sidecar.resolve()
        config.update({
            "pose_jacobian_sidecar": str(pose_sidecar),
            "pose_jacobian_sidecar_sha256": _sha256_file(
                pose_sidecar / "manifest.json" if pose_sidecar.is_dir() else pose_sidecar
            ),
            "tau_pose": args.tau_pose,
            "derivative_representation": "J_y on shared scorer planes; J_x retained for A^T custody only",
        })
    config_sha = hashlib.sha256(_canonical(config)).hexdigest()
    rows: list[dict[str, Any]] = []
    if args.resume:
        loaded = json.loads(state.read_text())
        if loaded.get("config_sha256") != config_sha:
            raise JointSolveError("resume config/custody mismatch")
        rows = list(loaded.get("rows", []))
        for row in rows:
            stage = Path(row["stage"]["path"])
            if not stage.is_file() or _sha256_file(stage) != row["stage"]["sha256"]:
                raise JointSolveError("resume stage custody mismatch")
            stage_payload = json.loads(stage.read_text())
            if (
                stage_payload.get("pair_id") != row.get("pair_id")
                or stage_payload.get("config_sha256") != config_sha
                or stage_payload.get("bindingness") != row.get("bindingness")
            ):
                raise JointSolveError("resume stage/bindingness metadata mismatch")
            bindingness_metadata = _validate_bindingness_sidecar(
                row["bindingness"],
                pair_id=int(row["pair_id"]),
                config_sha256=config_sha,
            )
            if (
                bindingness_metadata["arrays"]["frame0_binding_map"]["sha256"]
                != stage_payload.get("binding0_sha256")
                or bindingness_metadata["arrays"]["frame1_binding_map"]["sha256"]
                != stage_payload.get("binding1_sha256")
            ):
                raise JointSolveError("resume bindingness/stage map hash mismatch")
    elif state.exists() or (stage_dir.exists() and any(stage_dir.iterdir())):
        raise JointSolveError("preserved state/stages exist; use --resume or new paths")
    else:
        _atomic_json(state, {"schema": STATE_SCHEMA, "config_sha256": config_sha, "config": config, "rows": []})

    operator = DisjointResizeOperator.build(camera_h=CAMERA_HW[0], camera_w=CAMERA_HW[1], scorer_h=SCORER_HW[0], scorer_w=SCORER_HW[1])
    segnet, posenet, torch = _load_scorers(upstream, args.cpu_threads)
    manifest_rows: dict[int, CustodiedPair] | None = None
    pullback_rows: dict[int, dict[str, np.ndarray]] | None = None
    if args.seg_pullback_sidecar is not None:
        seg_path = args.seg_pullback_sidecar.resolve()
        if _is_vjp_manifest(seg_path):
            manifest_rows = _load_vjp_rows(seg_path, pairs)
        else:
            pullback_rows = _load_seg_pullback_sidecar(seg_path, pairs)
    if args.pose_jacobian_sidecar is not None:
        pose_path = args.pose_jacobian_sidecar.resolve()
        if manifest_rows is None:
            manifest_rows = _load_vjp_rows(pose_path, pairs)
        elif (
            args.seg_pullback_sidecar is None
            or (pose_path / "manifest.json" if pose_path.is_dir() else pose_path)
            != (
                args.seg_pullback_sidecar.resolve() / "manifest.json"
                if args.seg_pullback_sidecar.resolve().is_dir()
                else args.seg_pullback_sidecar.resolve()
            )
        ):
            pose_rows = _load_vjp_rows(pose_path, pairs)
            if any(
                pose_rows[pair_id].metadata["tensors"]["pose_j_y"]["sha256"]
                != manifest_rows[pair_id].metadata["tensors"]["pose_j_y"]["sha256"]
                for pair_id in pairs
            ):
                raise JointSolveError("Seg and Pose custody manifests disagree")
    completed = {int(row["pair_id"]) for row in rows}
    for pair_id in pairs:
        if pair_id in completed:
            continue
        started = time.monotonic()
        source0 = np.asarray(fields["gt_f0"][pair_id], dtype=np.uint8).copy()
        source1 = np.asarray(fields["gt_f1"][pair_id], dtype=np.uint8).copy()
        labels = np.asarray(fields["lstars"][pair_id], dtype=np.int64).copy()
        margins = np.asarray(fields["margins"][pair_id], dtype=np.float64).copy()
        target_pose = np.asarray(fields["gt_poses"][pair_id], dtype=np.float64).copy()
        source_control, native_winner, native_rival = _hard_verdict(
            segnet, posenet, torch, source0, source1, labels, target_pose
        )
        source_pose_delta6 = np.asarray(source_control["pose6"], dtype=np.float64) - target_pose
        cache_disagreement = int(np.count_nonzero(native_winner != labels))
        n0, den0 = operator.apply_numerators(source0)
        n1, den1 = operator.apply_numerators(source1)
        y0, y1 = n0.astype(np.float64) / den0, n1.astype(np.float64) / den1
        predictor0, predictor1 = generated_fill_predictor(operator, y0), generated_fill_predictor(operator, y1)
        del source0, source1

        custody = None if manifest_rows is None else manifest_rows[pair_id]
        vjp_arrangement = {
            "custodied_vs_hard_oracle_winner_disagreement_pixels": None,
            "inference_vs_vjp_rival_disagreement_pixels": None,
            "inference_vs_vjp_rival_disagreement_scope": VJP_RIVAL_DISAGREEMENT_SCOPE,
        }
        if custody is not None:
            vjp_arrangement = _gate_vjp_arrangement(
                stage_dir=stage_dir,
                pair_id=pair_id,
                config_sha256=config_sha,
                source_control=source_control,
                custodied_winner=custody.winner,
                custodied_rival=custody.rival,
                hard_oracle_winner=native_winner,
                hard_oracle_rival=native_rival,
            )
            if not np.allclose(custody.cached_margin, margins, rtol=0.0, atol=0.0):
                raise JointSolveError(f"pair {pair_id} VJP cached margin differs from active cache")
        elif pullback_rows is not None:
            pullback_arrangement = pullback_rows[pair_id]
            vjp_arrangement = _gate_vjp_arrangement(
                stage_dir=stage_dir,
                pair_id=pair_id,
                config_sha256=config_sha,
                source_control=source_control,
                custodied_winner=pullback_arrangement["winner"],
                custodied_rival=pullback_arrangement["rival"],
                hard_oracle_winner=native_winner,
                hard_oracle_rival=native_rival,
            )
        using_pose_vjp = args.pose_jacobian_sidecar is not None
        predictor_num0, predictor_den0 = operator.apply_numerators(predictor0)
        predictor_num1, predictor_den1 = operator.apply_numerators(predictor1)
        if predictor_den0 != den0 or predictor_den1 != den1:
            raise JointSolveError("predictor range-coordinate denominator mismatch")
        frame0_predictor_direction = predictor_num0.astype(np.float64) / den0 - y0

        attempt_rows = []
        accepted = None
        solve_seconds = 0.0
        verify_seconds = 0.0
        attempt_count = 1 if args.seg_band_scale == 0.0 and args.pose_rgb_band == 0.0 and not using_pose_vjp else args.repair_steps + 1
        for repair in range(attempt_count):
            shrink = 0.5 ** repair
            bound_started = time.monotonic()
            band_config = MarginBandConfig(
                scale=args.seg_band_scale * shrink, local_lipschitz=args.local_lipschitz,
                max_rgb_radius=args.max_seg_rgb_radius,
            )
            if args.seg_band_scale == 0.0:
                band1 = derive_margin_rgb_band(margins, band_config)
            else:
                if custody is not None and pullback_rows is None:
                    pullback = {
                        "winner": custody.winner,
                        "rival": custody.rival,
                        "pullback": custody.seg_q,
                        "pair_norms": custody.pair_norms,
                    }
                    local_lipschitz_field = custody.seg_local_lipschitz
                else:
                    assert pullback_rows is not None
                    pullback = pullback_rows[pair_id]
                    local_lipschitz_field = None
                band1 = derive_hyperplane_channel_band(
                    margins, native_winner, native_rival, pullback["pullback"], pullback["pair_norms"], band_config,
                    local_lipschitz_field=local_lipschitz_field,
                ).channel_radii
            bound_seconds = time.monotonic() - bound_started
            solve_started = time.monotonic()
            pose_proposal = None
            if custody is None and not using_pose_vjp:
                band0 = np.full(SCORER_HW, args.pose_rgb_band * shrink, dtype=np.float64)
                solved0 = solve_interval_frame(operator, n0, den0, band0, predictor=predictor0, max_nodes_per_block=args.max_nodes_per_block)
                solved1 = solve_interval_frame(operator, n1, den1, band1, predictor=predictor1, max_nodes_per_block=args.max_nodes_per_block)
            else:
                solved1 = solve_interval_frame(
                    operator,
                    n1,
                    den1,
                    band1,
                    predictor=predictor1,
                    max_nodes_per_block=args.max_nodes_per_block,
                    conservative_exact_source_numerator_fallback=True,
                )
                if using_pose_vjp:
                    assert custody is not None and args.tau_pose is not None
                    frame1_delta = solved1.chosen_numerators.astype(np.float64) / den1 - y1
                    pose_proposal = _repair_pose_linear_proposal(
                        custody.pose_j_y,
                        frame1_delta,
                        frame0_predictor_direction,
                        source_pose_delta6,
                        args.tau_pose,
                        shrink,
                    )
                    if not pose_proposal["feasible"]:
                        solve_seconds += time.monotonic() - solve_started
                        pose_activity = _pose_constraint_activity(
                            pose_proposal=pose_proposal,
                            hard_d_pose=None,
                            tau_pose=args.tau_pose,
                            repair_shrink=shrink,
                        )
                        attempt_rows.append({
                            "repair": repair,
                            "shrink": shrink,
                            "effective_linear_pose_budget": pose_proposal[
                                "effective_linear_pose_budget"
                            ],
                            "bound_seconds": bound_seconds,
                            "pose_linear_proposal": pose_proposal,
                            "pose_constraint_activity": pose_activity,
                            "hard_oracle": None,
                            "PASS": False,
                            "refusal": "no frame0 predictor step is feasible under this instance linear model",
                        })
                        continue
                    band0 = np.abs(frame0_predictor_direction) * float(pose_proposal["selected_step"])
                else:
                    band0 = np.zeros((*SCORER_HW, 3), dtype=np.float64)
                solved0 = solve_interval_frame(
                    operator,
                    n0,
                    den0,
                    band0,
                    predictor=predictor0,
                    max_nodes_per_block=args.max_nodes_per_block,
                    conservative_exact_source_numerator_fallback=True,
                )
            solve_seconds += time.monotonic() - solve_started
            actual_linear_delta6 = None
            actual_linear_pose6 = None
            if pose_proposal is not None:
                actual_frame0_delta = solved0.chosen_numerators.astype(np.float64) / den0 - y0
                actual_frame1_delta = solved1.chosen_numerators.astype(np.float64) / den1 - y1
                actual_linear_delta6 = linearized_pose_delta6(
                    custody.pose_j_y,
                    actual_frame0_delta,
                    actual_frame1_delta,
                    source_pose_delta6,
                )
                actual_linear_pose6 = target_pose + actual_linear_delta6
            verify_started = time.monotonic()
            verdict, _candidate_winner, _candidate_rival = _hard_verdict(
                segnet, posenet, torch, solved0.frame, solved1.frame, native_winner, target_pose
            )
            verify_seconds += time.monotonic() - verify_started
            pose_limit = args.pose_tolerance if args.tau_pose is None else args.tau_pose
            passed = verdict["d_seg"] == 0.0 and verdict["d_pose"] <= pose_limit
            attempt = {"repair": repair, "shrink": shrink, "bound_seconds": bound_seconds, "hard_oracle": verdict, "PASS": passed}
            if pose_proposal is not None:
                assert actual_linear_delta6 is not None and actual_linear_pose6 is not None
                planned_delta6 = np.asarray(
                    pose_proposal["planned_predictor_step_pose_delta6"], dtype=np.float64
                )
                real_pose6 = np.asarray(verdict["pose6"], dtype=np.float64)
                attempt["pose_linear_proposal"] = pose_proposal
                attempt["effective_linear_pose_budget"] = pose_proposal[
                    "effective_linear_pose_budget"
                ]
                attempt["planned_predictor_step_linear_pose6"] = (target_pose + planned_delta6).tolist()
                attempt["actual_lattice_linear_pose6"] = actual_linear_pose6.tolist()
                attempt["actual_lattice_linear_pose_delta6_vs_cached_target"] = (
                    actual_linear_delta6.tolist()
                )
                attempt["actual_lattice_linear_pose_mse_vs_cached_target"] = float(
                    np.mean(actual_linear_delta6 * actual_linear_delta6)
                )
                residual = real_pose6 - actual_linear_pose6
                attempt["actual_lattice_linear_vs_hard_oracle_pose6_residual"] = residual.tolist()
                attempt["actual_lattice_linear_vs_hard_oracle_pose6_residual_mse"] = float(
                    np.mean(residual * residual)
                )
                attempt["predicted_vs_real_pose6_residual_basis"] = (
                    "actual_solved_lattice_deltas_through_J_y_first_order"
                )
                attempt["predicted_vs_real_pose6_residual"] = residual.tolist()
                attempt["predicted_vs_real_pose6_residual_mse"] = float(
                    np.mean(residual * residual)
                )
                attempt["pose_constraint_activity"] = _pose_constraint_activity(
                    pose_proposal=pose_proposal,
                    hard_d_pose=float(verdict["d_pose"]),
                    tau_pose=args.tau_pose,
                    repair_shrink=shrink,
                )
            attempt_rows.append(attempt)
            if passed:
                accepted = (solved0, solved1, verdict, repair, band0, band1, pose_proposal)
                break
        if accepted is None:
            failure_path = stage_dir / f"pair_{pair_id:04d}.hard_oracle_refusal.json"
            failure_payload = {
                "schema": "joint_seg_pose_hard_oracle_refusal.v1", "pair_id": pair_id,
                "config_sha256": config_sha, "source_positive_control": source_control,
                "cached_vs_native_winner_disagreement_pixels": cache_disagreement,
                **vjp_arrangement,
                "attempts": attempt_rows,
                "hard_oracle_band_proposals": _hard_oracle_band_proposal_summary(attempt_rows),
                "verdict_scope": "this pair and operating point only; not a formulation-family verdict",
            }
            if custody is not None:
                failure_payload["seg_vjp_field"] = _vjp_field_summary(custody)
            _atomic_json(failure_path, failure_payload)
            raise JointSolveError(
                f"pair {pair_id} exhausted hard-oracle repair; durable refusal={failure_path}; "
                f"last={attempt_rows[-1]}"
            )
        solved0, solved1, verdict, repair_count, accepted_band0, accepted_band1, pose_proposal = accepted
        rate_started = time.monotonic()
        rate0 = range_payload_bytes_and_tiles(solved0.chosen_numerators, predictor_num0)
        rate1 = range_payload_bytes_and_tiles(solved1.chosen_numerators, predictor_num1)
        rate_seconds = time.monotonic() - rate_started
        rung_e_rate_point = _rung_e_rate_point(
            chosen0=solved0.chosen_numerators,
            chosen1=solved1.chosen_numerators,
            predictor0=predictor_num0,
            predictor1=predictor_num1,
            rate0=rate0,
            rate1=rate1,
        )
        operating_point = {
            "seg_band_scale": args.seg_band_scale,
            "pose_rgb_band": args.pose_rgb_band,
            "pose_tolerance": args.pose_tolerance,
        }
        positive_telemetry = None
        if custody is not None or using_pose_vjp:
            operating_point["tau_pose"] = args.tau_pose
            positive = np.asarray(accepted_band1) > 0
            positive_telemetry = {
                "seg_positive_radius_channel_count": int(np.count_nonzero(positive)),
                "seg_zero_radius_channel_count": int(np.count_nonzero(~positive)),
                "seg_positive_radius_channel_fraction": float(np.mean(positive)),
                "seg_zero_radius_channel_fraction": float(np.mean(~positive)),
                "seg_channel_radius": _radius_summary(accepted_band1),
                "pose_selected_step": None if pose_proposal is None else pose_proposal["selected_step"],
                "effective_linear_pose_budget": (
                    None
                    if pose_proposal is None
                    else pose_proposal["effective_linear_pose_budget"]
                ),
                "full_hard_oracle_pose_limit": (
                    None
                    if pose_proposal is None
                    else pose_proposal["full_hard_oracle_pose_limit"]
                ),
                "planned_predictor_step_linear_pose_mse_vs_cached_target": (
                    None
                    if pose_proposal is None
                    else pose_proposal["planned_predictor_step_pose_mse"]
                ),
                "actual_lattice_linear_pose_mse_vs_cached_target": (
                    None
                    if pose_proposal is None
                    else attempt_rows[-1]["actual_lattice_linear_pose_mse_vs_cached_target"]
                ),
                "actual_lattice_linear_vs_hard_oracle_pose6_residual": (
                    None
                    if pose_proposal is None
                    else attempt_rows[-1]["actual_lattice_linear_vs_hard_oracle_pose6_residual"]
                ),
                "actual_lattice_linear_vs_hard_oracle_pose6_residual_mse": (
                    None
                    if pose_proposal is None
                    else attempt_rows[-1][
                        "actual_lattice_linear_vs_hard_oracle_pose6_residual_mse"
                    ]
                ),
                "predicted_vs_real_pose6_residual_basis": (
                    None
                    if pose_proposal is None
                    else attempt_rows[-1]["predicted_vs_real_pose6_residual_basis"]
                ),
                "predicted_vs_real_pose6_residual": (
                    None
                    if pose_proposal is None
                    else attempt_rows[-1]["predicted_vs_real_pose6_residual"]
                ),
                "predicted_vs_real_pose6_residual_mse": (
                    None
                    if pose_proposal is None
                    else attempt_rows[-1]["predicted_vs_real_pose6_residual_mse"]
                ),
                "pose_constraint_activity": (
                    None
                    if pose_proposal is None
                    else attempt_rows[-1]["pose_constraint_activity"]
                ),
                "frame0_channel_radius": _radius_summary(accepted_band0),
                "hard_oracle_band_proposals": _hard_oracle_band_proposal_summary(attempt_rows),
            }
            if custody is not None:
                positive_telemetry["seg_vjp_field"] = _vjp_field_summary(custody)
        frame0_fallback = _conservative_exact_source_numerator_fallback_summary(solved0)
        frame1_fallback = _conservative_exact_source_numerator_fallback_summary(solved1)
        bindingness_arrays, fallback_present = _bindingness_arrays(
            binding0=solved0.binding_map,
            binding1=solved1.binding_map,
            positive_seg_radius=np.asarray(accepted_band1) > 0,
            fallback0=solved0.conservative_exact_source_numerator_fallback_map,
            fallback1=solved1.conservative_exact_source_numerator_fallback_map,
        )
        bindingness = _publish_bindingness_sidecar(
            stage_dir / f"pair_{pair_id:04d}.bindingness.npz",
            pair_id=pair_id,
            config_sha256=config_sha,
            arrays=bindingness_arrays,
            fallback_present=fallback_present,
        )
        stage_path = stage_dir / f"pair_{pair_id:04d}.json"
        stage_payload = {
            "schema": "joint_seg_pose_pair_stage.v1", "pair_id": pair_id,
            "config_sha256": config_sha, "hard_oracle": verdict,
            **vjp_arrangement,
            "frame0_sha256": _sha256_array(solved0.frame), "frame1_sha256": _sha256_array(solved1.frame),
            "binding0_sha256": _sha256_array(solved0.binding_map), "binding1_sha256": _sha256_array(solved1.binding_map),
            "chosen_numerators0_sha256": _sha256_array(solved0.chosen_numerators),
            "chosen_numerators1_sha256": _sha256_array(solved1.chosen_numerators),
            "winner_sha256": _sha256_array(native_winner), "rival_sha256": _sha256_array(native_rival),
            "bindingness": bindingness,
            "rung_e_rate_point": rung_e_rate_point,
            "reconstruction": "deterministic from frozen cache scorer numerators + config + generated-fill predictor; camera frames are rebuildable and not persisted locally",
        }
        if frame0_fallback is not None and frame1_fallback is not None:
            stage_payload["conservative_exact_source_numerator_fallback"] = {
                "frame0": frame0_fallback,
                "frame1": frame1_fallback,
            }
        _atomic_json(stage_path, stage_payload)
        stage_sha = _sha256_file(stage_path)
        row = {
            "pair_id": pair_id, "operating_point": operating_point,
            "source_positive_control": source_control,
            "cached_vs_native_winner_disagreement_pixels": cache_disagreement,
            **vjp_arrangement,
            "hard_oracle": verdict, "hard_oracle_repair_count": repair_count, "attempts": attempt_rows,
            "frame0": {"telemetry": solved0.telemetry.__dict__, "binding_map_sha256": _sha256_array(solved0.binding_map), "rate": {k: v for k, v in rate0.items() if k != "tiles"}, "byte_attribution": _attribution(rate0, None, None, None)},
            "frame1": {"telemetry": solved1.telemetry.__dict__, "binding_map_sha256": _sha256_array(solved1.binding_map), "rate": {k: v for k, v in rate1.items() if k != "tiles"}, "byte_attribution": _attribution(rate1, native_winner, native_rival, margins)},
            "profile_seconds": {"integer_search_and_repair": solve_seconds, "hard_oracle_verify": verify_seconds, "rate_and_tile_compression": rate_seconds, "total": time.monotonic() - started},
            "bindingness": bindingness,
            "rung_e_rate_point": rung_e_rate_point,
            "stage": {"path": str(stage_path), "sha256": stage_sha},
        }
        if positive_telemetry is not None:
            assert frame0_fallback is not None and frame1_fallback is not None
            row["frame0"].update(frame0_fallback)
            row["frame1"].update(frame1_fallback)
            row["positive_band_telemetry"] = positive_telemetry
            row["vjp_custody"] = {
                "representation": "J_y consumed on shared scorer planes; J_x retained for A^T custody only",
                "exact_block_infeasibility_policy": (
                    "conservative exact source-numerator fallback"
                ),
                "seg_sidecar_tensor_hash": (
                    None if custody is None else custody.metadata["tensors"]["seg_g_y"]["sha256"]
                ),
                "pose_sidecar_tensor_hash": (
                    None if custody is None else custody.metadata["tensors"]["pose_j_y"]["sha256"]
                ),
            }
        rows.append(row)
        _atomic_json(state, {"schema": STATE_SCHEMA, "config_sha256": config_sha, "config": config, "rows": rows})

    if _stat_tree_snapshot(SACRED) != sacred_before:
        raise JointSolveError("sacred result tree changed during measurement")
    pose_activities = [
        row["positive_band_telemetry"]["pose_constraint_activity"]
        for row in rows
        if row.get("positive_band_telemetry", {}).get("pose_constraint_activity")
        is not None
    ]
    pose_hypothesis_activities = [
        activity
        for activity in pose_activities
        if activity["preregistered_hypothesis_at_or_below_crossover"]
    ]
    pose_hypothesis_confirm_count = sum(
        activity["hard"]["active"] is False
        for activity in pose_hypothesis_activities
    )
    receipt = {
        "schema": SCHEMA, "written_at_utc": datetime.now(UTC).isoformat(),
        "axis": f"[{platform.system()}-{platform.machine()} CPU advisory subset] NON-PROMOTABLE",
        "authority": {"score_claim": False, "promotion_eligible": False, "pointer": POINTER, "pointer_moved": False,
                      "verdict_scope": "selected real n600-cache pairs; frozen CPU-torch SegNet/PoseNet; no contest CPU/CUDA or receiver archive claim"},
        "labels": {"MEASURED": ["actual Brotli-Q11 residual bytes", "actual zstd-19 residual bytes", "frozen CPU scorer d_seg/d_pose", "stage timings"],
                   "DERIVED": ["rank-4 winner/rival hyperplane pullback channel bands", "pose derivative 5/sqrt(10d)", "crossover d_pose=2.5e-4"],
                   "INFERRED": []},
        "config": config, "config_sha256": config_sha, "pairs": rows,
        "shared_rung_e_provenance": RUNG_E_PROVENANCE,
        "receiver_arithmetic_declaration": {"dtype": "native float32", "semantics": "CPU-Torch conv/eval", "tie_policy": "authority scorer native argmax; generic-f64 is not substituted", "declared": True},
        "range_kernel_rate_law": {"counted_coordinates": "range(A) scorer numerator residual only", "ker_A_payload_bytes": 0, "ker_A_fill": "generated deterministically from declared decoder predictor and lattice solve", "camera_residual_is_not_serialized": True},
        "aggregate": {"pair_count": len(rows), "unique_pair_ids": sorted(int(r["pair_id"]) for r in rows),
                      "mean_d_seg": float(np.mean([r["hard_oracle"]["d_seg"] for r in rows])),
                      "mean_d_pose": float(np.mean([r["hard_oracle"]["d_pose"] for r in rows])),
                      "pose_constraint_activity_observation_count": len(pose_activities),
                      "pose_constraint_hypothesis_observation_count": len(pose_hypothesis_activities),
                      "pose_constraint_hypothesis_confirm_count": pose_hypothesis_confirm_count,
                      "pose_constraint_hypothesis_refute_count": len(pose_hypothesis_activities) - pose_hypothesis_confirm_count,
                      "total_brotli_q11_bytes": int(sum(r[f]["rate"]["brotli_q11_bytes"] for r in rows for f in ("frame0", "frame1"))),
                      "total_zstd_19_bytes": int(sum(r[f]["rate"]["zstd_19_bytes"] for r in rows for f in ("frame0", "frame1")))},
        "resumability": {"state": str(state), "stage_dir": str(stage_dir), "all_stage_checkpoints_preserved": True,
                         "checkpoint_form": "small write-once JSON manifests plus immutable compressed full-map bindingness NPZ; deterministic candidates rebuild from frozen cache and config",
                         "bindingness_resume_cross_validation": True},
        "sacred_tree_unchanged": True,
    }
    if manifest_rows is not None:
        receipt["vjp_custody"] = {
            "representation": "J_y consumed on shared scorer planes; J_x retained for A^T custody only",
            "exact_block_infeasibility_policy": (
                "conservative exact source-numerator fallback"
            ),
            "positive_band_linear_model_is_proposal_only": True,
            "frozen_hard_oracle_is_admission_authority": True,
        }
    _atomic_json(output, receipt)
    return receipt


def compose(receipts: Sequence[Path], output: Path) -> dict[str, Any]:
    if output.exists():
        raise JointSolveError(f"composed receipt already exists: {output}")
    docs = [json.loads(path.read_text()) for path in receipts]
    if not docs or any(doc.get("schema") != SCHEMA for doc in docs):
        raise JointSolveError("all inputs must be joint receipt v1")
    rows = [row for doc in docs for row in doc["pairs"]]
    pair_obs = {(int(row["pair_id"]), json.dumps(row["operating_point"], sort_keys=True)) for row in rows}
    if len(pair_obs) != len(rows):
        raise JointSolveError("duplicate pair/operating-point observation in composition")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    positive_composition = any("tau_pose" in row["operating_point"] for row in rows)
    for row in rows:
        operating_point = row["operating_point"]
        grouping_point = (
            {
                "seg_band_scale": operating_point["seg_band_scale"],
                "tau_pose": operating_point.get("tau_pose"),
            }
            if positive_composition
            else operating_point
        )
        grouped[json.dumps(grouping_point, sort_keys=True)].append(row)
    curves = []
    for key, group in sorted(grouped.items()):
        curves.append({"operating_point": json.loads(key), "pair_count": len(group),
                       "bytes": float(np.mean([sum(row[f]["rate"]["brotli_q11_bytes"] for f in ("frame0", "frame1")) for row in group])),
                       "d_seg": float(np.mean([row["hard_oracle"]["d_seg"] for row in group])),
                       "d_pose": float(np.mean([row["hard_oracle"]["d_pose"] for row in group]))})
    seg_curve = [{"bytes": x["bytes"], "distortion": x["d_seg"]} for x in curves]
    pose_curve = [{"bytes": x["bytes"], "distortion": x["d_pose"]} for x in curves]
    pose_activities = [
        row["positive_band_telemetry"]["pose_constraint_activity"]
        for row in rows
        if row.get("positive_band_telemetry", {}).get("pose_constraint_activity")
        is not None
    ]
    pose_hypothesis_activities = [
        activity
        for activity in pose_activities
        if activity["preregistered_hypothesis_at_or_below_crossover"]
    ]
    pose_hypothesis_confirm_count = sum(
        activity["hard"]["active"] is False
        for activity in pose_hypothesis_activities
    )
    result = {"schema": "joint_seg_pose_inverse_rate_composed.v1", "written_at_utc": datetime.now(UTC).isoformat(),
              "authority": {"score_claim": False, "pointer": POINTER, "verdict_scope": "composed real-cache advisory chunks only"},
              "source_receipts": [{"path": str(p.resolve()), "sha256": _sha256_file(p)} for p in receipts],
              "observation_count": len(rows), "unique_pair_count": len({int(r["pair_id"]) for r in rows}),
              "shared_rung_e_provenance": RUNG_E_PROVENANCE,
              "pose_constraint_hypothesis": {
                  "activity_observation_count": len(pose_activities),
                  "observation_count": len(pose_hypothesis_activities),
                  "confirm_count": pose_hypothesis_confirm_count,
                  "refute_count": len(pose_hypothesis_activities) - pose_hypothesis_confirm_count,
                  "verdict_scope": "composed accepted pair/operating-point observations only",
              },
              "measured_curves": curves, "waterfill": solve_measured_waterfill(seg_curve, pose_curve)}
    _atomic_json(output, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compose", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--stage-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sample-pairs", type=int, default=12)
    parser.add_argument("--pair-indices", nargs="+", type=int)
    parser.add_argument("--seg-pullback-sidecar", type=Path)
    parser.add_argument("--seg-band-scale", type=float, default=0.0)
    parser.add_argument("--local-lipschitz", type=float, default=1.0)
    parser.add_argument("--max-seg-rgb-radius", type=float, default=8.0)
    parser.add_argument("--pose-rgb-band", type=float, default=0.0)
    parser.add_argument("--pose-jacobian-sidecar", type=Path)
    parser.add_argument("--tau-pose", type=float)
    parser.add_argument("--pose-tolerance", type=float, default=1e-8)
    parser.add_argument("--repair-steps", type=int, default=4)
    parser.add_argument("--max-nodes-per-block", type=int, default=4096)
    parser.add_argument("--cpu-threads", type=int, default=4)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.compose:
        result = compose(args.compose, args.output.resolve())
    else:
        if args.state is None or args.stage_dir is None:
            raise SystemExit("measurement requires --state and --stage-dir")
        result = run_measurement(args)
    print(json.dumps({"output": str(args.output.resolve()), "schema": result["schema"]}, sort_keys=True))


if __name__ == "__main__":
    main()
