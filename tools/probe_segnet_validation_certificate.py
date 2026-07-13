#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Atomic, resumable local probe for a frozen-SegNet cheap validation gate.

The real path reuses the landed YOPO split and settled renderer.  It performs
exact frozen-CPU-SegNet comparisons for every candidate; it never imports the
contest evaluator and never upgrades empirical ``PROXY_ACCEPT`` to a proof.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for path in (REPO, REPO / "src", REPO / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from probe_yopo_first_layer_costate import (  # noqa: E402
    CHECKPOINT_DIR,
    GT_CACHE,
    REGIMES,
    SEGNET,
    _load_renderer,
    _render_chart,
    _renderer_parity_canary,
)

from experiments.train_witness_realized_through_R_mlx import (  # noqa: E402
    _torch_R_to_camera_uint8,
    cpu_verdict_d_pose,
)
from tac.boundary_math.segnet_gradient_replacement import (  # noqa: E402
    _yopo_first_layer_prefix_torch,
    yopo_first_layer_split_identity,
)
from tac.boundary_math.segnet_validation_certificate import (  # noqa: E402
    ProxyConfusionAccumulator,
    cadence_speedup,
    calibrate_empirical_pairwise_bounds,
    check_feature_trust_region,
    confusion_meter_canaries,
    derive_feature_trust_region,
)

SCHEMA = "segnet_validation_certificate_probe.v1"
SEED = 20260712
FRACTION_START = 1e-2
FRACTION_DECAY = 0.5
CALIBRATION_COUNT = 2
MAX_CANDIDATES = 24
SOURCE_PATHS = (
    "tools/probe_segnet_validation_certificate.py",
    "tools/probe_yopo_first_layer_costate.py",
    "src/tac/boundary_math/segnet_validation_certificate.py",
    "src/tac/boundary_math/segnet_gradient_replacement.py",
    "src/tac/cuda_levelset_training.py",
)
YOPO_RECEIPT = REPO / "experiments/results/yopo_first_layer_costate_probe_20260713T003635Z/receipt.json"
YOPO_RECEIPT_SHA256 = "a89585cd70b9630c90468f3a502e1efc778836cffc56ca7fb71e997fff2e6fa3"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _array_custody(value: Any) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(value))
    return {"sha256": _array_sha256(array), "dtype": array.dtype.str, "shape": list(array.shape)}


def _calibration_custody_payload(
    *,
    regime: str,
    anchor_margins: Any,
    candidate_margin_arrays: Any,
    feature_displacements_linf: Any,
    derived_bounds: Any,
    candidate_indices: list[int],
    fractions: list[float],
    anchor_feature: Any,
    anchor_split_identity_sha256: str,
) -> dict[str, Any]:
    """Bind every numeric calibration input/output and the exact feature cut."""

    candidates = np.asarray(candidate_margin_arrays)
    return {
        "regime": regime,
        "anchor_margins": _array_custody(anchor_margins),
        "candidate_margin_arrays": [
            _array_custody(candidates[index]) for index in range(candidates.shape[0])
        ],
        "feature_displacements_linf": _array_custody(feature_displacements_linf),
        "derived_bounds": _array_custody(derived_bounds),
        "candidate_indices": list(candidate_indices),
        "fractions": list(fractions),
        "anchor_feature": _array_custody(anchor_feature),
        "anchor_split_identity_sha256": anchor_split_identity_sha256,
    }


def _calibration_mutation_canary() -> dict[str, Any]:
    kwargs = {
        "regime": "canary",
        "anchor_margins": np.array([2.0, 3.0]),
        "candidate_margin_arrays": np.array([[1.5, 2.5], [1.0, 2.0]]),
        "feature_displacements_linf": np.array([0.5, 1.0]),
        "derived_bounds": np.array([1.0, 1.0]),
        "candidate_indices": [0, 1],
        "fractions": [0.01, 0.005],
        "anchor_feature": np.array([[[[1.0]]]], dtype=np.float32),
        "anchor_split_identity_sha256": "a" * 64,
    }
    base = _json_sha256(_calibration_custody_payload(**kwargs))
    mutated = dict(kwargs)
    mutated_candidates = np.array(kwargs["candidate_margin_arrays"], copy=True)
    mutated_candidates[0, 0] = np.nextafter(mutated_candidates[0, 0], np.inf)
    mutated["candidate_margin_arrays"] = mutated_candidates
    changed = _json_sha256(_calibration_custody_payload(**mutated))
    return {"status": "PASS" if base != changed else "FAIL", "base_sha256": base, "mutated_sha256": changed}


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)


def _linear_radius_canary() -> dict[str, Any]:
    region = derive_feature_trust_region(
        anchor_margins=[2.0], anchor_correct_mask=[True], pairwise_logit_change_bounds=[2.0],
        authority="rigorous_upper_bound", anchor_feature_sha256="a" * 64, bound_artifact_sha256="b" * 64,
    )
    inside = check_feature_trust_region(anchor_feature=[0.0], current_feature=[0.5], region=region,
                                        current_anchor_feature_sha256="a" * 64)
    outside = check_feature_trust_region(anchor_feature=[0.0], current_feature=[1.01], region=region,
                                         current_anchor_feature_sha256="a" * 64)
    passed = inside.status == "ACCEPT" and outside.status == "REFRESH"
    return {"status": "PASS" if passed else "FAIL", "inside": inside.__dict__, "outside": outside.__dict__}


def build_canary_receipt(argv: list[str]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "CANARIES_ONLY",
        "authority": {
            "axis": "[local mechanism only]",
            "score_claim": False,
            "promotion_eligible": False,
            "review_status": "unreviewed_fix_round_2",
        },
        "seed": SEED,
        "argv": argv,
        "controls": {
            "known_linear_map_inside_and_outside": _linear_radius_canary(),
            "confusion_meter_positive_and_negative": confusion_meter_canaries(),
            "calibration_content_mutation": _calibration_mutation_canary(),
        },
        "rigorous_mechanism": {
            "verdict": "NO-GO",
            "reason": "no actual downstream suffix pairwise-logit upper-bound artifact is supplied",
            "verdict_scope": "this certificate formulation only; trust regions remain intact as a family",
        },
    }


def _storage_free_bytes() -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for label, path in (
        ("local", REPO),
        ("vertigo_ssd", Path("/Volumes/VertigoDataTier/pact")),
        ("apdatastore_ssd", Path("/Volumes/APDataStore/pact")),
    ):
        if path.exists():
            usage = os.statvfs(path)
            rows[label] = {"path": str(path), "free_bytes": int(usage.f_bavail * usage.f_frsize)}
        else:
            rows[label] = {"path": str(path), "status": "NOT_MOUNTED"}
    return rows


def _runtime_environment() -> dict[str, Any]:
    tracked = ("PYTHONHASHSEED", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()
    return {"git_head": git, "environment": {name: os.environ.get(name) for name in tracked}}


def _feature_and_logits_camera(segnet: Any, frame_camera_uint8: np.ndarray) -> tuple[Any, Any]:
    import torch

    captured: list[Any] = []
    handle = segnet.encoder.model.blocks[0].register_forward_hook(lambda _m, _i, output: captured.append(output))
    try:
        frame = torch.from_numpy(np.stack([frame_camera_uint8, frame_camera_uint8], axis=0)[None]).float()
        scorer_input = segnet.preprocess_input(frame.permute(0, 1, 4, 2, 3).contiguous())
        logits = segnet(scorer_input)
    finally:
        handle.remove()
    if len(captured) != 1 or not bool(torch.isfinite(captured[0]).all()) or not bool(torch.isfinite(logits).all()):
        raise RuntimeError("frozen SegNet did not produce one finite YOPO split feature and logits")
    return captured[0], logits


def _segnet_input(segnet: Any, frame_camera_uint8: np.ndarray) -> Any:
    import torch

    frame = torch.from_numpy(np.stack([frame_camera_uint8, frame_camera_uint8], axis=0)[None]).float()
    return segnet.preprocess_input(frame.permute(0, 1, 4, 2, 3).contiguous())


def _cheap_yopo_feature(segnet: Any, frame_camera_uint8: np.ndarray) -> Any:
    """Run only the exact conv_stem -> bn1 -> blocks[0] prefix."""

    return _yopo_first_layer_prefix_torch(segnet, _segnet_input(segnet, frame_camera_uint8))


def _fixed_anchor_margin(logits: Any, anchor_prediction: Any) -> Any:
    import torch

    selected = logits.gather(1, anchor_prediction[:, None]).squeeze(1)
    competitors = logits.masked_fill(
        torch.nn.functional.one_hot(anchor_prediction, num_classes=logits.shape[1]).permute(0, 3, 1, 2).bool(),
        -torch.inf,
    ).max(1).values
    return selected - competitors


def _camera_frame(renderer: Any, theta: Any) -> np.ndarray:
    renderer.code[1] = theta.detach()
    render_grid, _lane = renderer.render_pair(0)
    if isinstance(render_grid, np.ndarray):
        render_grid_array = render_grid
    elif hasattr(render_grid, "detach"):
        render_grid_array = render_grid.detach().cpu().numpy()
    else:
        raise TypeError("renderer.render_pair must return a NumPy array or torch tensor")
    return _torch_R_to_camera_uint8(np.asarray(render_grid_array, dtype=np.float32))


def _exact_ce_dseg(logits: Any, labels: Any) -> tuple[float, float]:
    import torch.nn.functional as functional

    ce = float(functional.cross_entropy(logits, labels).item())
    prediction = logits.argmax(1)
    dseg = float((prediction != labels).count_nonzero().item()) / labels.numel()
    return ce, dseg


def _feature_parity(cheap_feature: Any, hook_feature: Any) -> dict[str, Any]:
    import torch

    cheap = cheap_feature.detach().cpu().numpy()
    hooked = hook_feature.detach().cpu().numpy()
    bitwise = bool(torch.equal(cheap_feature, hook_feature))
    return {
        "status": "PASS" if bitwise and _array_sha256(cheap) == _array_sha256(hooked) else "FAIL",
        "bitwise_equal": bitwise,
        "cheap_sha256": _array_sha256(cheap),
        "full_forward_hook_sha256": _array_sha256(hooked),
    }


def _measure_regime(
    *, regime: str, segnet: Any, posenet: Any, labels: Any, gt_f0: np.ndarray, gt_pose: np.ndarray
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    checkpoint = CHECKPOINT_DIR / REGIMES[regime]
    renderer, code, _model, _dash = _load_renderer(checkpoint)
    theta = torch.as_tensor(code[1], dtype=torch.float32).clone().requires_grad_(True)
    parity = _renderer_parity_canary(renderer, theta)
    if parity["max_abs"] != 0.0:
        raise RuntimeError("settled renderer parity canary failed")
    anchor_frame = _render_chart(renderer, theta)
    started = time.perf_counter()
    anchor_camera = _camera_frame(renderer, theta)
    anchor_camera_seconds = time.perf_counter() - started
    started = time.perf_counter()
    with torch.inference_mode():
        anchor_cheap_feature = _cheap_yopo_feature(segnet, anchor_camera)
    anchor_cheap_prefix_seconds = time.perf_counter() - started
    started = time.perf_counter()
    with torch.inference_mode():
        anchor_feature, anchor_logits = _feature_and_logits_camera(segnet, anchor_camera)
        anchor_ce, anchor_dseg = _exact_ce_dseg(anchor_logits, labels)
    anchor_exact_segnet_ce_dseg_seconds = time.perf_counter() - started
    anchor_feature_canary = _feature_parity(anchor_cheap_feature, anchor_feature)
    if anchor_feature_canary["status"] != "PASS":
        raise RuntimeError("cheap YOPO prefix differs from full-forward blocks[0] hook")
    anchor_prediction = anchor_logits.argmax(1)
    started = time.perf_counter()
    anchor_dpose = cpu_verdict_d_pose(posenet, gt_f0, anchor_camera, gt_pose)
    anchor_posenet_seconds = time.perf_counter() - started
    anchor_margin = _fixed_anchor_margin(anchor_logits, anchor_prediction).detach()
    # Candidate ordering reuses the existing differentiable settled chart.  It
    # is only a direction generator; exact d_seg/d_pose labels below always use
    # the camera uint8 frames from the authority path.
    differentiable_logits = segnet(anchor_frame.permute(0, 3, 1, 2).contiguous())
    loss = functional.cross_entropy(differentiable_logits, labels)
    direction = torch.autograd.grad(loss, theta)[0].detach()
    norm = float(torch.linalg.vector_norm(direction).item())
    if not np.isfinite(norm) or norm == 0.0:
        raise RuntimeError("exact frozen-SegNet renderer direction is zero or nonfinite")
    base_norm = max(float(torch.linalg.vector_norm(theta.detach()).item()), 1.0)
    rows: list[dict[str, Any]] = []
    for index in range(MAX_CANDIDATES):
        fraction = FRACTION_START * FRACTION_DECAY**index
        candidate_theta = theta.detach() - (fraction * base_norm / norm) * direction
        if torch.equal(candidate_theta, theta.detach()):
            rows.append({"candidate_index": index, "fraction": fraction, "status": "BIT_IDENTICAL_TERMINATION"})
            break
        started = time.perf_counter()
        with torch.inference_mode():
            camera = _camera_frame(renderer, candidate_theta)
        camera_seconds = time.perf_counter() - started
        started = time.perf_counter()
        with torch.inference_mode():
            cheap_feature = _cheap_yopo_feature(segnet, camera)
        cheap_prefix_seconds = time.perf_counter() - started
        started = time.perf_counter()
        with torch.inference_mode():
            feature, logits = _feature_and_logits_camera(segnet, camera)
            ce, dseg = _exact_ce_dseg(logits, labels)
            candidate_margin = _fixed_anchor_margin(logits, anchor_prediction)
        exact_segnet_seconds = time.perf_counter() - started
        feature_canary = _feature_parity(cheap_feature, feature)
        if feature_canary["status"] != "PASS":
            raise RuntimeError(f"candidate {index} cheap YOPO prefix differs from full-forward hook")
        started = time.perf_counter()
        dpose = cpu_verdict_d_pose(posenet, gt_f0, camera, gt_pose)
        posenet_seconds = time.perf_counter() - started
        rows.append({
            "candidate_index": index, "fraction": fraction, "status": "MEASURED",
            "cheap_feature": cheap_feature.cpu().numpy(),
            "feature_displacement_linf": float((cheap_feature - anchor_cheap_feature).abs().max().item()),
            "fixed_anchor_margin": candidate_margin.cpu().numpy(),
            "exact_ce": ce, "exact_ce_worsens": ce > anchor_ce,
            "exact_dseg": dseg, "exact_dseg_worsens": dseg > anchor_dseg,
            "exact_dpose": dpose, "exact_dpose_worsens": dpose > anchor_dpose,
            "feature_prefix_parity": feature_canary,
            "timing_measured_seconds": {
                "camera_render_R": camera_seconds,
                "cheap_prefix_only": cheap_prefix_seconds,
                "exact_segnet_ce_dseg": exact_segnet_seconds,
                "posenet": posenet_seconds,
            },
        })
    measured = [row for row in rows if row["status"] == "MEASURED"]
    if len(measured) <= CALIBRATION_COUNT:
        raise RuntimeError("bit-identical termination left no disjoint holdout after calibration")
    calibration = measured[:CALIBRATION_COUNT]
    anchor_feature_array = anchor_cheap_feature.cpu().numpy()
    anchor_margin_array = anchor_margin.cpu().numpy()
    calibration_margins = np.stack([row["fixed_anchor_margin"] for row in calibration])
    calibration_displacements = np.asarray([row["feature_displacement_linf"] for row in calibration])
    bounds = calibrate_empirical_pairwise_bounds(
        anchor_pairwise_margins=anchor_margin_array,
        candidate_pairwise_margins=calibration_margins,
        feature_displacements_linf=calibration_displacements,
    )
    split_identity = yopo_first_layer_split_identity(segnet)
    calibration_payload = _calibration_custody_payload(
        regime=regime,
        anchor_margins=anchor_margin_array,
        candidate_margin_arrays=calibration_margins,
        feature_displacements_linf=calibration_displacements,
        derived_bounds=bounds,
        candidate_indices=[row["candidate_index"] for row in calibration],
        fractions=[row["fraction"] for row in calibration],
        anchor_feature=anchor_feature_array,
        anchor_split_identity_sha256=split_identity,
    )
    calibration_sha = _json_sha256(calibration_payload)
    correct = (anchor_prediction == labels).cpu().numpy()
    region = derive_feature_trust_region(
        anchor_margins=anchor_margin.cpu().numpy(), anchor_correct_mask=correct,
        pairwise_logit_change_bounds=bounds, authority="empirical_local_estimate",
        anchor_feature_sha256=calibration_payload["anchor_feature"]["sha256"],
        calibration_receipt_sha256=calibration_sha,
    )
    meter = ProxyConfusionAccumulator()
    holdout_rows: list[dict[str, Any]] = []
    for row in measured[CALIBRATION_COUNT:]:
        started = time.perf_counter()
        decision = check_feature_trust_region(
            anchor_feature=anchor_feature_array,
            current_feature=row["cheap_feature"],
            region=region,
            current_anchor_feature_sha256=calibration_payload["anchor_feature"]["sha256"],
        )
        gate_seconds = time.perf_counter() - started
        accepts = decision.status == "PROXY_ACCEPT"
        meter.update(
            proxy_accepts=accepts,
            exact_ce_worsens=bool(row["exact_ce_worsens"]),
            exact_dseg_worsens=bool(row["exact_dseg_worsens"]),
            exact_dpose_worsens=bool(row["exact_dpose_worsens"]),
        )
        row["timing_measured_seconds"]["array_gate_only"] = gate_seconds
        holdout_rows.append(
            {key: value for key, value in row.items() if key not in {"fixed_anchor_margin", "cheap_feature"}}
            | {
                "proxy_decision": decision.__dict__,
                "unsafe_accept_ce": accepts and bool(row["exact_ce_worsens"]),
                "unsafe_accept_dseg": accepts and bool(row["exact_dseg_worsens"]),
                "unsafe_accept_dpose": accepts and bool(row["exact_dpose_worsens"]),
                "unsafe_accept_any": accepts
                and bool(row["exact_ce_worsens"] or row["exact_dseg_worsens"] or row["exact_dpose_worsens"]),
            }
        )
    return {
        "status": "MEASURED", "regime": regime,
        "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
        "renderer_parity": parity,
        "anchor": {
            "exact_ce": anchor_ce,
            "exact_dseg": anchor_dseg,
            "exact_dpose": anchor_dpose,
            "feature_prefix_parity": anchor_feature_canary,
            "timing_measured_seconds": {
                "camera_render_R": anchor_camera_seconds,
                "cheap_prefix_only": anchor_cheap_prefix_seconds,
                "exact_segnet_ce_dseg": anchor_exact_segnet_ce_dseg_seconds,
                "posenet": anchor_posenet_seconds,
            },
        },
        "candidate_order": [row["candidate_index"] for row in rows],
        "calibration": calibration_payload | {"sha256": calibration_sha},
        "feature_radius": region.feature_radius, "protected_pixels": region.protected_pixels,
        "candidates": [
            {key: value for key, value in row.items() if key not in {"fixed_anchor_margin", "cheap_feature"}}
            for row in rows
        ],
        "holdout": holdout_rows, "confusion": meter.to_dict(),
        "exact_posenet_pair_comparison": {
            "status": "MEASURED",
            "frame0": "gt_n6.npz:gt_f0[0]",
            "frame1": "identical candidate camera uint8 from settled render -> bicubic 874x1164 -> round/clamp",
            "target": "gt_n6.npz:gt_poses[0]",
            "helper": "experiments.train_witness_realized_through_R_mlx.cpu_verdict_d_pose",
        },
    }


def _inherited_yopo_component_rows() -> dict[int, dict[str, Any]]:
    """Read, but never rerun, the content-bound final YOPO timing receipt."""

    if _sha256(YOPO_RECEIPT) != YOPO_RECEIPT_SHA256:
        raise RuntimeError("final YOPO receipt custody changed")
    payload = json.loads(YOPO_RECEIPT.read_text())
    result: dict[int, dict[str, Any]] = {}
    for cadence in (2, 4):
        exact_rows: list[dict[str, Any]] = []
        approx_rows: list[dict[str, Any]] = []
        for regime, regime_row in payload["regimes"].items():
            arm = next(row for row in regime_row["arms"] if row["K"] == cadence)
            for list_index, step in enumerate(arm["steps"]):
                identity = {
                    "receipt_sha256": YOPO_RECEIPT_SHA256,
                    "regime": regime,
                    "K": cadence,
                    "step_list_index": list_index,
                    "step": step["step"],
                    "refresh": bool(step["refresh"]),
                }
                exact_rows.append(identity | {"seconds": step["timing_measured_seconds"]["ordinary_exact_teacher_forward_backward"]})
                if not step["refresh"]:
                    approx_rows.append(identity | {"seconds": step["timing_measured_seconds"]["yopo_provider"]})
        result[cadence] = {"t_exact_rows": exact_rows, "t_approx_nonrefresh_rows": approx_rows}
    return result


def _component_economics(regimes: list[dict[str, Any]]) -> dict[str, Any]:
    inherited = _inherited_yopo_component_rows()
    holdout = [candidate for regime in regimes for candidate in regime.get("holdout", [])]
    cheap_rows = [
        candidate["timing_measured_seconds"]["cheap_prefix_only"]
        + candidate["timing_measured_seconds"]["array_gate_only"]
        for candidate in holdout
    ]
    safety_segnet_rows = [
        candidate["timing_measured_seconds"]["exact_segnet_ce_dseg"] for candidate in holdout
    ]
    safety_posenet_rows = [candidate["timing_measured_seconds"]["posenet"] for candidate in holdout]
    rejection_rate = (
        sum(candidate["proxy_decision"]["status"] != "PROXY_ACCEPT" for candidate in holdout) / len(holdout)
        if holdout else 0.0
    )
    measured_cheap = float(np.median(cheap_rows)) if cheap_rows else None
    safety_measurement = {
        "purpose": "held-descent evidence only; excluded from component speedup action cost",
        "exact_segnet_ce_dseg_seconds_current_measured_median": (
            float(np.median(safety_segnet_rows)) if safety_segnet_rows else None
        ),
        "posenet_seconds_current_measured_median": (
            float(np.median(safety_posenet_rows)) if safety_posenet_rows else None
        ),
    }
    rows: dict[str, Any] = {}
    for cadence, custody in inherited.items():
        exact = float(np.median([row["seconds"] for row in custody["t_exact_rows"]]))
        approx_values = [row["seconds"] for row in custody["t_approx_nonrefresh_rows"]]
        approx = float(np.median(approx_values)) if approx_values else None
        fallback = exact * rejection_rate
        speedup = (
            cadence_speedup(
                cadence=cadence,
                t_exact=exact,
                t_approx=approx,
                t_validate_cheap=measured_cheap,
                t_fallback=fallback,
            )
            if approx is not None and measured_cheap is not None
            else None
        )
        rows[str(cadence)] = {
            "status": "DERIVED_COMPONENT_FORMULA" if speedup is not None else "BLOCKED_MISSING_COMPONENT_ROW",
            "sequence_integration_status": "UNINTEGRATED_COMPONENT_ECONOMICS_ONLY",
            "t_exact_seconds_inherited_measured_median": exact,
            "t_approx_seconds_inherited_measured_median": approx,
            "t_validate_cheap_seconds_current_measured_median": measured_cheap,
            "t_fallback_seconds_derived_rejection_weighted_inherited_full_teacher_forward_backward": fallback,
            "fallback_action": "full_teacher_and_refresh",
            "fallback_derivation": "t_exact_seconds_inherited_measured_median * rejection_rate",
            "rejection_rate": rejection_rate,
            "derived_speedup": speedup,
            **custody,
        }
    return {
        "source_receipt": {"path": str(YOPO_RECEIPT), "sha256": YOPO_RECEIPT_SHA256},
        "master_gate_rerun": False,
        "current_exact_scorer_safety_measurement": safety_measurement,
        "cadences": rows,
    }


def _terminal_verdict(
    regimes: list[dict[str, Any]],
    *,
    economics: dict[str, Any] | None = None,
    sequence_integrated_whole_step: dict[str, Any] | None = None,
) -> dict[str, Any]:
    joint_unsafe = sum(row.get("confusion", {}).get("unsafe_accepts_any", 0) for row in regimes)
    dseg_unsafe = sum(row.get("confusion", {}).get("unsafe_accepts_dseg", 0) for row in regimes)
    proxy_accepts = sum(
        row.get("confusion", {}).get("unsafe_accepts_any", 0)
        + row.get("confusion", {}).get("exact_safe_accepts", 0)
        for row in regimes
    )
    sequence_path = (
        Path(sequence_integrated_whole_step["receipt_path"]).resolve()
        if sequence_integrated_whole_step and isinstance(sequence_integrated_whole_step.get("receipt_path"), str)
        else None
    )
    sequence_path_in_results = False
    if sequence_path is not None:
        with contextlib.suppress(ValueError):
            sequence_path.relative_to((REPO / "experiments/results").resolve())
            sequence_path_in_results = True
    sequence_pass = bool(
        sequence_integrated_whole_step
        and sequence_integrated_whole_step.get("authority") == "MEASURED_SEQUENCE_INTEGRATED_WHOLE_STEP"
        and isinstance(sequence_integrated_whole_step.get("receipt_sha256"), str)
        and len(sequence_integrated_whole_step["receipt_sha256"]) == 64
        and all(character in "0123456789abcdef" for character in sequence_integrated_whole_step["receipt_sha256"])
        and sequence_path_in_results
        and sequence_path is not None
        and sequence_path.is_file()
        and _sha256(sequence_path) == sequence_integrated_whole_step["receipt_sha256"]
        and sequence_integrated_whole_step.get("measured_speedup", 0.0) >= 1.3
    )
    return {
        "rigorous": "NO-GO",
        "rigorous_reason": "no actual suffix upper bound exists in this probe",
        "segnet_dseg_proxy": "NO-GO" if dseg_unsafe else "ADVISORY-HOLDOUT-PASS",
        "segnet_dseg_proxy_reason": (
            f"{dseg_unsafe} d_seg-unsafe proxy accepts among {proxy_accepts} proxy accepts"
            if dseg_unsafe
            else f"zero d_seg-unsafe proxy accepts among {proxy_accepts} proxy accepts; advisory holdout evidence only"
        ),
        "joint_held_descent": "NO-GO" if joint_unsafe else "ADVISORY-HOLDOUT-PASS",
        "joint_held_descent_reason": (
            f"{joint_unsafe} CE/d_seg/d_pose-unsafe proxy accepts among {proxy_accepts} proxy accepts"
            if joint_unsafe
            else f"zero CE/d_seg/d_pose-unsafe proxy accepts among {proxy_accepts} proxy accepts; advisory holdout evidence only"
        ),
        "empirical": "NO-GO" if joint_unsafe else "ADVISORY-HOLDOUT-PASS",
        "empirical_reason": "joint CE/d_seg/d_pose holdout unsafe accept occurred" if joint_unsafe else "joint advisory holdout passed",
        "throughput": "GO" if sequence_pass else "NEEDS-MORE",
        "throughput_reason": (
            "sequence-integrated measured whole-step speedup is >=1.3x"
            if sequence_pass
            else "component economics are not a sequential whole-step measurement; sequence-integrated >=1.3x evidence remains required"
        ),
        "component_economics": economics,
        "sequence_integrated_whole_step": sequence_integrated_whole_step,
        "verdict_scope": "pair0, sealed early/boundary/late regimes, blocks[0], registered ladder, macOS-CPU advisory, this formulation only",
    }


def run_real(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from tac.boundary_math.seg_core import load_real_segnet

    receipt_path = args.output_dir / "receipt.json"
    if args.resume and not receipt_path.is_file():
        raise RuntimeError("--resume requires an existing receipt.json")
    if not args.resume and receipt_path.exists():
        raise RuntimeError("fresh probe refuses to overwrite an existing receipt.json")
    immutable = {
        "schema": SCHEMA, "seed": args.seed, "candidate_ladder": {"start": FRACTION_START, "decay": FRACTION_DECAY, "max": MAX_CANDIDATES},
        "authority": {
            "axis": "[macOS-CPU advisory]",
            "score_claim": False,
            "promotion_eligible": False,
            "review_status": "unreviewed_fix_round_2",
        },
        "inputs": {
            "segnet": {"path": str(SEGNET), "sha256": _sha256(SEGNET)},
            "gt_cache": {"path": str(GT_CACHE), "sha256": _sha256(GT_CACHE)},
            "final_yopo_receipt": {"path": str(YOPO_RECEIPT), "sha256": _sha256(YOPO_RECEIPT)},
        },
        "source_custody": {path: _sha256(REPO / path) for path in SOURCE_PATHS},
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            **_runtime_environment(),
        },
        "controls": build_canary_receipt(sys.argv)["controls"],
    }
    receipt = (
        json.loads(receipt_path.read_text())
        if args.resume and receipt_path.is_file()
        else immutable
        | {
            "status": "RUNNING",
            "regimes": [],
            "argv_history": [list(sys.argv)],
            "storage_free_bytes_at_start": _storage_free_bytes(),
        }
    )
    for key, value in immutable.items():
        if receipt.get(key) != value:
            raise RuntimeError(f"resume immutable field changed: {key}")
    if args.resume:
        receipt.setdefault("argv_history", []).append(list(sys.argv))
    atomic_write_json(receipt_path, receipt)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    segnet = load_real_segnet("cpu").eval()
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path
    distortion_net = DistortionNet().eval()
    distortion_net.load_state_dicts(posenet_sd_path, segnet_sd_path, torch.device("cpu"))
    posenet = distortion_net.posenet
    for parameter in posenet.parameters():
        parameter.requires_grad_(False)
    with np.load(GT_CACHE, allow_pickle=False) as cache:
        labels = torch.as_tensor(np.asarray(cache["lstars"])[0], dtype=torch.long).unsqueeze(0)
        gt_f0 = np.asarray(cache["gt_f0"])[0].astype(np.uint8, copy=True)
        gt_pose = np.asarray(cache["gt_poses"])[0].astype(np.float64, copy=True)
    completed = {row["regime"] for row in receipt["regimes"]}
    for regime in REGIMES:
        if regime in completed:
            continue
        receipt["regimes"].append(
            _measure_regime(
                regime=regime, segnet=segnet, posenet=posenet, labels=labels,
                gt_f0=gt_f0, gt_pose=gt_pose,
            )
        )
        atomic_write_json(receipt_path, receipt)
    receipt["economics"] = _component_economics(receipt["regimes"])
    receipt["verdict"] = _terminal_verdict(receipt["regimes"], economics=receipt["economics"])
    receipt["status"] = "MEASURED"
    receipt["storage_free_bytes_at_end"] = _storage_free_bytes()
    receipt["completed_at_utc"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    atomic_write_json(receipt_path, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--canaries-only", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.output_dir is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        args.output_dir = REPO / "experiments/results" / f"segnet_validation_certificate_{stamp}"
    args.output_dir = args.output_dir.resolve()
    args.output_dir.relative_to((REPO / "experiments/results").resolve())
    if args.canaries_only:
        atomic_write_json(args.output_dir / "receipt.json", build_canary_receipt(sys.argv))
    else:
        run_real(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
