#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Resumable local probe for anchor-only costate validation economics.

Operational decisions use one exact anchor validation, renderer/preprocess
work, two prefix-only calibration probes, and an O(pixels) margin/Fisher gate.
Fresh exact scorer forwards on accepted proposals are shadow controls and are
reported separately rather than hidden from the accounting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

import probe_segnet_validation_certificate as sealed_probe  # noqa: E402

from tac.scorer_surrogate.costate_trust_region import (  # noqa: E402
    array_sha256,
    check_costate_trust_region,
    derive_costate_trust_region,
    fit_empirical_jacobian_envelope,
    margin_fisher_proxy,
    validation_economics,
)

SCHEMA = "costate_trust_region_economics_probe.v1"
BASELINE_YOPO_RECEIPT = (
    REPO / "experiments/results/yopo_first_layer_costate_probe_20260713T003635Z/receipt.json"
)
BASELINE_YOPO_SHA256 = "a89585cd70b9630c90468f3a502e1efc778836cffc56ca7fb71e997fff2e6fa3"
FEATURE_REGION_RECEIPT = (
    REPO / "experiments/results/segnet_validation_certificate_20260713T015633Z/receipt.json"
)
FEATURE_REGION_SHA256 = "60fe88fa1a5058d018170005890ef0720f01b31762b5e7ef0b5c7d6dc19a7d60"
CALIBRATION_COEFFICIENT_COUNT = 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_bound_receipt(path: Path, expected_sha256: str) -> dict[str, Any]:
    if _sha256(path) != expected_sha256:
        raise RuntimeError(f"receipt custody changed: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"receipt is not a mapping: {path}")
    return payload


def _baseline_counts(payload: dict[str, Any]) -> dict[str, int]:
    rows = [
        step
        for regime in payload["regimes"].values()
        for arm in regime["arms"]
        for step in arm["steps"]
    ]
    return {
        "step_rows": len(rows),
        "operational_validation_forwards": sum(
            int(row["teacher_work_counts"]["operational_validation_forwards_including_labels"])
            for row in rows
        ),
        "operational_teacher_forward_backward": sum(
            int(row["teacher_work_counts"]["operational_teacher_forward_backward_including_labels"])
            for row in rows
        ),
        "measurement_teacher_forward_backward": sum(
            int(row["teacher_work_counts"]["measurement_only_teacher_forward_backward_including_labels"])
            for row in rows
        ),
        "total_teacher_forward_backward": sum(
            int(row["teacher_work_counts"]["actual_probe_teacher_forward_backward_including_labels"])
            for row in rows
        ),
    }


def _minimum_inherited_cosines(payload: dict[str, Any]) -> dict[str, float]:
    steps = [
        step
        for regime in payload["regimes"].values()
        for arm in regime["arms"]
        for step in arm["steps"]
        if not step["refresh"] and step["costate_metrics_global"]["valid"]
    ]
    return {
        "global_costate": min(float(step["costate_metrics_global"]["cosine_similarity"]) for step in steps),
        "boundary_annulus": min(
            float(step["costate_metrics_gt_boundary_annulus_bottom_k_0p05"]["cosine_similarity"])
            for step in steps
        ),
        "renderer_gradient": min(float(step["renderer_gradient_cosine"]) for step in steps),
    }


def _fisher_rms_displacement(
    *, anchor_input: np.ndarray, current_input: np.ndarray, margins: np.ndarray, protected: np.ndarray
) -> float:
    delta = np.max(np.abs(current_input - anchor_input), axis=1)
    exponential = np.exp(-np.abs(margins))
    weights = exponential / np.square(1.0 + exponential)
    weights *= protected
    mass = float(np.sum(weights, dtype=np.float64))
    if not math.isfinite(mass) or mass <= 0.0:
        raise RuntimeError("anchor Fisher field has zero or nonfinite protected mass")
    return math.sqrt(float(np.sum(weights * np.square(delta), dtype=np.float64)) / mass)


def _select_prefix_calibration(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    positive = sorted(
        (row for row in rows if float(row["input_displacement_margin_fisher_rms"]) > 0.0),
        key=lambda row: (float(row["input_displacement_margin_fisher_rms"]), int(row["candidate_index"])),
    )
    selected: list[dict[str, Any]] = []
    observed: set[float] = set()
    for row in positive:
        displacement = float(row["input_displacement_margin_fisher_rms"])
        if displacement in observed:
            continue
        observed.add(displacement)
        selected.append(row)
        if len(selected) == CALIBRATION_COEFFICIENT_COUNT:
            break
    if not selected:
        raise RuntimeError("candidate ladder contains no non-identical scorer input")
    return selected


def _fresh_exact_candidate(segnet: Any, labels: Any, camera: np.ndarray) -> dict[str, float]:
    import torch

    with torch.inference_mode():
        _feature, logits = sealed_probe._feature_and_logits_camera(segnet, camera)
        ce, dseg = sealed_probe._exact_ce_dseg(logits, labels)
    return {"ce": ce, "dseg": dseg}


def _measure_regime(
    *,
    regime_receipt: dict[str, Any],
    segnet: Any,
    labels: Any,
    ladder: dict[str, Any],
    calibration_parent_sha256: str,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    regime = str(regime_receipt["regime"])
    checkpoint = sealed_probe.CHECKPOINT_DIR / sealed_probe.REGIMES[regime]
    renderer, code, _model, _dash = sealed_probe._load_renderer(checkpoint)
    theta = torch.as_tensor(code[1], dtype=torch.float32).clone().requires_grad_(True)
    parity = sealed_probe._renderer_parity_canary(renderer, theta)
    if parity["max_abs"] != 0.0:
        raise RuntimeError("settled renderer parity canary failed")

    anchor_camera_started = time.perf_counter()
    anchor_camera = sealed_probe._camera_frame(renderer, theta)
    anchor_camera_seconds = time.perf_counter() - anchor_camera_started
    anchor_input = sealed_probe._segnet_input(segnet, anchor_camera).detach().cpu().numpy()

    anchor_validation_started = time.perf_counter()
    with torch.inference_mode():
        anchor_feature, anchor_logits = sealed_probe._feature_and_logits_camera(segnet, anchor_camera)
        anchor_ce, anchor_dseg = sealed_probe._exact_ce_dseg(anchor_logits, labels)
    anchor_validation_seconds = time.perf_counter() - anchor_validation_started
    anchor_prediction = anchor_logits.argmax(1)
    margins = sealed_probe._fixed_anchor_margin(anchor_logits, anchor_prediction).detach().cpu().numpy()
    protected = (anchor_prediction == labels).detach().cpu().numpy()
    if not bool(protected.any()) or bool((margins[protected] <= 0.0).any()):
        raise RuntimeError("anchor-correct margin field is invalid")

    gradient_anchor_started = time.perf_counter()
    differentiable_frame = sealed_probe._render_chart(renderer, theta)
    differentiable_logits = segnet(differentiable_frame.permute(0, 3, 1, 2).contiguous())
    teacher_loss = functional.cross_entropy(differentiable_logits, labels)
    direction = torch.autograd.grad(teacher_loss, theta)[0].detach()
    gradient_anchor_seconds = time.perf_counter() - gradient_anchor_started
    norm = float(torch.linalg.vector_norm(direction).item())
    if not math.isfinite(norm) or norm == 0.0:
        raise RuntimeError("anchor teacher renderer gradient is zero or nonfinite")
    base_norm = max(float(torch.linalg.vector_norm(theta.detach()).item()), 1.0)

    source_rows = {
        int(row["candidate_index"]): row
        for row in regime_receipt["candidates"]
        if row["status"] == "MEASURED"
    }
    candidate_rows: list[dict[str, Any]] = []
    candidate_cameras: dict[int, np.ndarray] = {}
    for candidate_index in sorted(source_rows):
        fraction = float(ladder["start"]) * float(ladder["decay"]) ** candidate_index
        source = source_rows[candidate_index]
        if fraction != float(source["fraction"]):
            raise RuntimeError("candidate ladder differs from the sealed exact-control receipt")
        candidate_theta = theta.detach() - (fraction * base_norm / norm) * direction
        render_started = time.perf_counter()
        camera = sealed_probe._camera_frame(renderer, candidate_theta)
        render_seconds = time.perf_counter() - render_started
        scorer_input = sealed_probe._segnet_input(segnet, camera).detach().cpu().numpy()
        displacement = _fisher_rms_displacement(
            anchor_input=anchor_input,
            current_input=scorer_input,
            margins=margins,
            protected=protected,
        )
        candidate_cameras[candidate_index] = camera
        candidate_rows.append(
            {
                "candidate_index": candidate_index,
                "fraction": fraction,
                "input_displacement_margin_fisher_rms": displacement,
                "camera_sha256": array_sha256(camera),
                "scorer_input_sha256": array_sha256(scorer_input),
                "scorer_input": scorer_input,
                "render_and_preprocess_seconds": render_seconds,
                "inherited_exact_control": {
                    "ce": float(source["exact_ce"]),
                    "dseg": float(source["exact_dseg"]),
                    "dpose": float(source["exact_dpose"]),
                    "ce_worsens": bool(source["exact_ce_worsens"]),
                    "dseg_worsens": bool(source["exact_dseg_worsens"]),
                    "dpose_worsens": bool(source["exact_dpose_worsens"]),
                },
            }
        )

    calibration_rows = _select_prefix_calibration(candidate_rows)
    calibration_input: list[float] = []
    calibration_feature: list[float] = []
    calibration_records: list[dict[str, Any]] = []
    for row in calibration_rows:
        index = int(row["candidate_index"])
        prefix_started = time.perf_counter()
        with torch.inference_mode():
            candidate_feature = sealed_probe._cheap_yopo_feature(segnet, candidate_cameras[index])
        prefix_seconds = time.perf_counter() - prefix_started
        feature_delta = float((candidate_feature - anchor_feature).abs().max().item())
        calibration_input.append(float(row["input_displacement_margin_fisher_rms"]))
        calibration_feature.append(feature_delta)
        calibration_records.append(
            {
                "candidate_index": index,
                "input_displacement_margin_fisher_rms": calibration_input[-1],
                "first_block_feature_displacement_linf": feature_delta,
                "prefix_only_seconds": prefix_seconds,
                "uses_exact_teacher_outcome": False,
            }
        )
    jacobian, beta = fit_empirical_jacobian_envelope(
        input_displacements=calibration_input,
        feature_displacements=calibration_feature,
    )
    feature_radius = float(regime_receipt["feature_radius"])
    reconstructed_suffix_bounds = np.zeros_like(margins, dtype=np.float64)
    reconstructed_suffix_bounds[protected] = margins[protected] / feature_radius
    calibration_payload = {
        "schema": "costate_trust_region_prefix_calibration.v1",
        "regime": regime,
        "parent_feature_region_receipt_sha256": calibration_parent_sha256,
        "anchor_frame_sha256": array_sha256(anchor_input),
        "anchor_margin_sha256": array_sha256(margins),
        "input_metric": "margin_fisher_rms",
        "calibration": calibration_records,
        "anchor_jacobian_norm_empirical": jacobian,
        "jacobian_lipschitz_empirical": beta,
        "authority": "empirical_local_estimate",
        "exact_teacher_outcome_blind": True,
    }
    calibration_sha = _json_sha256(calibration_payload)
    region = derive_costate_trust_region(
        anchor_margins=margins,
        anchor_correct_mask=protected,
        pairwise_suffix_lipschitz_upper=reconstructed_suffix_bounds,
        anchor_jacobian_norm_upper=jacobian,
        jacobian_lipschitz_upper=beta,
        authority="empirical_local_estimate",
        anchor_frame_sha256=array_sha256(anchor_input),
        calibration_receipt_sha256=calibration_sha,
        input_metric="margin_fisher_rms",
        provider_mode="current_prefix_vjp_banked_suffix",
    )

    accepted: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    gate_seconds: list[float] = []
    for row in candidate_rows:
        started = time.perf_counter()
        decision = check_costate_trust_region(
            anchor_frame=anchor_input,
            current_frame=row["scorer_input"],
            region=region,
            current_anchor_frame_sha256=array_sha256(anchor_input),
        )
        elapsed = time.perf_counter() - started
        gate_seconds.append(elapsed)
        compact = {key: value for key, value in row.items() if key != "scorer_input"}
        compact["decision"] = decision.__dict__
        compact["gate_seconds"] = elapsed
        if decision.reuses_costate:
            exact_started = time.perf_counter()
            fresh = _fresh_exact_candidate(
                segnet, labels, candidate_cameras[int(row["candidate_index"])]
            )
            fresh_seconds = time.perf_counter() - exact_started
            inherited = row["inherited_exact_control"]
            dseg_match = fresh["dseg"] == inherited["dseg"]
            ce_float32_match = np.float32(fresh["ce"]) == np.float32(inherited["ce"])
            compact["fresh_shadow_exact_control"] = {
                **fresh,
                "seconds": fresh_seconds,
                "ce_descends": fresh["ce"] < anchor_ce,
                "dseg_nonworsens": fresh["dseg"] <= anchor_dseg,
                "matches_inherited_dseg": dseg_match,
                "matches_inherited_ce_float32": bool(ce_float32_match),
                "operational_cost": False,
            }
            if not dseg_match or not ce_float32_match:
                raise RuntimeError("fresh accepted-candidate exact control differs from sealed receipt")
            accepted.append(compact)
        decisions.append(compact)

    accepted_descent = sum(
        int(row["fresh_shadow_exact_control"]["ce_descends"]) for row in accepted
    )
    accepted_dseg_safe = sum(
        int(row["fresh_shadow_exact_control"]["dseg_nonworsens"]) for row in accepted
    )
    return {
        "status": "MEASURED",
        "regime": regime,
        "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
        "renderer_parity": parity,
        "anchor": {
            "camera_sha256": array_sha256(anchor_camera),
            "scorer_input_sha256": array_sha256(anchor_input),
            "margin_sha256": array_sha256(margins),
            "exact_ce": anchor_ce,
            "exact_dseg": anchor_dseg,
            "protected_pixels": int(np.count_nonzero(protected)),
            "margin_fisher_proxy": margin_fisher_proxy(margins, mask=protected),
            "timing_seconds": {
                "camera_render": anchor_camera_seconds,
                "one_exact_anchor_validation": anchor_validation_seconds,
                "exact_teacher_gradient_anchor": gradient_anchor_seconds,
            },
        },
        "calibration": calibration_payload | {"sha256": calibration_sha},
        "region": {
            "authority": region.authority,
            "input_metric": region.input_metric,
            "input_radius": region.input_radius,
            "margin_radius": region.margin_radius,
            "descent_radius": None if math.isinf(region.descent_radius) else region.descent_radius,
            "descent_bound_available": region.descent_bound_available,
            "feature_radius": region.feature_radius,
            "anchor_fisher_proxy": region.anchor_fisher_proxy,
            "boundary_fisher_proxy_upper": region.boundary_fisher_proxy_upper,
            "rigorous_blocker": (
                "no rigorous first-block neighborhood bound, suffix pairwise-logit/costate bound, "
                "renderer-VJP upper bound, or projected-gradient floor artifact"
            ),
        },
        "decision_counts": {
            "candidates": len(decisions),
            "proxy_reuses": len(accepted),
            "refreshes": len(decisions) - len(accepted),
            "fresh_shadow_exact_forwards": len(accepted),
            "proxy_reuses_with_exact_ce_descent": accepted_descent,
            "proxy_reuses_with_exact_dseg_nonworsening": accepted_dseg_safe,
        },
        "gate_timing_seconds": {
            "median": float(np.median(np.asarray(gate_seconds, dtype=np.float64))),
            "minimum": min(gate_seconds),
            "maximum": max(gate_seconds),
        },
        "decisions": decisions,
        "empirical_descent_verdict": (
            "PASS"
            if accepted and accepted_descent == len(accepted) and accepted_dseg_safe == len(accepted)
            else "NO-GO_UNSAFE_REUSE"
            if accepted
            else "NO-GO_NO_REUSE"
        ),
        "verdict_scope": (
            "pair0; one sealed saved regime; exact anchor direction; registered fractional ladder; "
            "margin-Fisher RMS empirical first-block envelope; macOS-CPU advisory"
        ),
    }


def _canaries() -> dict[str, Any]:
    margins = np.array([[[2.0, 3.0]]], dtype=np.float64)
    protected = np.ones_like(margins, dtype=bool)
    frame = np.zeros((1, 3, 1, 2), dtype=np.float64)
    region = derive_costate_trust_region(
        anchor_margins=margins,
        anchor_correct_mask=protected,
        pairwise_suffix_lipschitz_upper=np.ones_like(margins),
        anchor_jacobian_norm_upper=1.0,
        jacobian_lipschitz_upper=0.0,
        suffix_costate_lipschitz_upper=0.25,
        renderer_vjp_norm_upper=1.0,
        projected_gradient_floor=0.5,
        provider_mode="current_prefix_vjp_banked_suffix",
        authority="rigorous_upper_bound",
        anchor_frame_sha256=array_sha256(frame),
        bound_artifact_sha256=hashlib.sha256(b"known-linear-bound").hexdigest(),
    )
    positive = check_costate_trust_region(
        anchor_frame=frame,
        current_frame=frame,
        region=region,
        current_anchor_frame_sha256=array_sha256(frame),
    )
    negative = check_costate_trust_region(
        anchor_frame=frame,
        current_frame=np.full_like(frame, region.input_radius),
        region=region,
        current_anchor_frame_sha256=array_sha256(frame),
    )
    return {
        "known_bound_positive": positive.__dict__,
        "known_bound_negative": negative.__dict__,
        "status": (
            "PASS"
            if positive.status == "CERTIFIED_REUSE" and negative.status == "REFRESH"
            else "FAIL"
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from tac.boundary_math.seg_core import load_real_segnet

    output_root = (REPO / "experiments/results").resolve()
    output_dir = args.output_dir.resolve()
    output_dir.relative_to(output_root)
    receipt_path = output_dir / "measurement_receipt.json"
    if receipt_path.exists() and not args.resume:
        raise RuntimeError("fresh run refuses to overwrite an existing receipt")
    if args.resume and not receipt_path.is_file():
        raise RuntimeError("--resume requires an existing measurement_receipt.json")

    yopo = _load_bound_receipt(BASELINE_YOPO_RECEIPT, BASELINE_YOPO_SHA256)
    feature = _load_bound_receipt(FEATURE_REGION_RECEIPT, FEATURE_REGION_SHA256)
    source_paths = (
        "src/tac/scorer_surrogate/costate_trust_region.py",
        "src/tac/boundary_math/segnet_validation_certificate.py",
        "src/tac/boundary_math/segnet_gradient_replacement.py",
        "tools/probe_costate_trust_region_economics.py",
        "tools/probe_segnet_validation_certificate.py",
        "tools/probe_yopo_first_layer_costate.py",
    )
    immutable = {
        "schema": SCHEMA,
        "authority": {
            "axis": "[macOS-CPU advisory; training-signal economics]",
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
        },
        "inputs": {
            "baseline_yopo": {"path": str(BASELINE_YOPO_RECEIPT), "sha256": BASELINE_YOPO_SHA256},
            "feature_region": {"path": str(FEATURE_REGION_RECEIPT), "sha256": FEATURE_REGION_SHA256},
            "segnet": {"path": str(sealed_probe.SEGNET), "sha256": _sha256(sealed_probe.SEGNET)},
            "gt_cache": {"path": str(sealed_probe.GT_CACHE), "sha256": _sha256(sealed_probe.GT_CACHE)},
        },
        "source_custody": {path: _sha256(REPO / path) for path in source_paths},
        "baseline_counts": _baseline_counts(yopo),
        "inherited_costate_fidelity": _minimum_inherited_cosines(yopo),
        "candidate_ladder": feature["candidate_ladder"],
        "controls": _canaries(),
    }
    receipt = (
        json.loads(receipt_path.read_text())
        if args.resume
        else immutable
        | {
            "status": "RUNNING",
            "created_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "runtime": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "torch": torch.__version__,
                "torch_num_threads": torch.get_num_threads(),
                "git_head": os.popen("git rev-parse HEAD").read().strip(),
            },
            "argv_history": [sys.argv],
            "regimes": [],
        }
    )
    for key, expected in immutable.items():
        if receipt.get(key) != expected:
            raise RuntimeError(f"resume immutable field changed: {key}")
    # A completed receipt is immutable evidence.  Re-check its bound inputs and
    # source custody above, then make terminal resume a byte-stable no-op.
    if args.resume and receipt.get("status") == "MEASURED":
        return receipt
    if args.resume:
        receipt["argv_history"].append(sys.argv)
    _atomic_write_json(receipt_path, receipt)
    if receipt["controls"]["status"] != "PASS":
        raise RuntimeError("known-bound trust-region canaries failed")

    torch.manual_seed(int(feature["seed"]))
    np.random.seed(int(feature["seed"]))
    torch.use_deterministic_algorithms(True)
    segnet = load_real_segnet("cpu").eval()
    for parameter in segnet.parameters():
        parameter.requires_grad_(False)
    with np.load(sealed_probe.GT_CACHE, allow_pickle=False) as cache:
        labels = torch.as_tensor(np.asarray(cache["lstars"])[0], dtype=torch.long).unsqueeze(0)
    completed = {str(row["regime"]) for row in receipt["regimes"]}
    for regime_receipt in feature["regimes"]:
        if regime_receipt["regime"] in completed:
            continue
        receipt["regimes"].append(
            _measure_regime(
                regime_receipt=regime_receipt,
                segnet=segnet,
                labels=labels,
                ladder=feature["candidate_ladder"],
                calibration_parent_sha256=FEATURE_REGION_SHA256,
            )
        )
        _atomic_write_json(receipt_path, receipt)

    anchors = len(receipt["regimes"])
    shadow_forwards = sum(
        int(row["decision_counts"]["fresh_shadow_exact_forwards"]) for row in receipt["regimes"]
    )
    baseline = receipt["baseline_counts"]
    receipt["economics"] = validation_economics(
        baseline_validation_forwards=int(baseline["operational_validation_forwards"]),
        baseline_teacher_calls=int(baseline["total_teacher_forward_backward"]),
        new_anchor_validations=anchors,
        new_anchors=anchors,
        shadow_control_forwards=shadow_forwards,
    )
    receipt["economics"]["baseline_validations_per_operational_teacher_anchor"] = (
        int(baseline["operational_validation_forwards"])
        / int(baseline["operational_teacher_forward_backward"])
    )
    receipt["economics"]["normalized_reduction_factor_vs_operational_teacher_anchors"] = (
        receipt["economics"]["baseline_validations_per_operational_teacher_anchor"]
        / receipt["economics"]["new_operational_validations_per_anchor"]
    )
    all_accepted_safe = all(
        row["fresh_shadow_exact_control"]["ce_descends"]
        and row["fresh_shadow_exact_control"]["dseg_nonworsens"]
        for regime in receipt["regimes"]
        for row in regime["decisions"]
        if row["decision"]["status"] == "PROXY_REUSE"
    )
    coverage = all(int(row["decision_counts"]["proxy_reuses"]) > 0 for row in receipt["regimes"])
    proxy_reuses = sum(int(row["decision_counts"]["proxy_reuses"]) for row in receipt["regimes"])
    receipt["verdict"] = {
        "rigorous_costate_certificate": "BLOCKED",
        "rigorous_blocker": (
            "first-block plus margin is insufficient for exact-descent proof; real rigorous suffix pairwise-logit "
            "and costate bounds, renderer-VJP bound, and projected-gradient floor are absent"
        ),
        "empirical_margin_fisher_region": "PASS" if all_accepted_safe and coverage else "NO-GO",
        "proxy_reuses": proxy_reuses,
        "all_proxy_reuses_preserve_fresh_exact_ce_descent_and_dseg": bool(proxy_reuses and all_accepted_safe),
        "reuse_coverage_all_three_regimes": coverage,
        "operational_validation_once_per_anchor": True,
        "sequence_integrated_live_trainer": "UNMEASURED_NOT_WIRED",
        "verdict_scope": (
            "pair0; three sealed early/boundary/late saved regimes; exact anchor direction; registered candidate "
            "ladder; empirical margin-Fisher RMS first-block envelope; macOS-CPU advisory; no live trainer"
        ),
    }
    receipt["status"] = "MEASURED"
    receipt["completed_at_utc"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    _atomic_write_json(receipt_path, receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--canaries-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.output_dir is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        args.output_dir = REPO / "experiments/results" / f"costate_trust_region_economics_{stamp}"
    if args.canaries_only:
        _atomic_write_json(args.output_dir / "measurement_receipt.json", {"schema": SCHEMA, "controls": _canaries()})
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
