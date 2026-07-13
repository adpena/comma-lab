#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Resumable local Jacobian-drift probe for direct full-costate reuse.

The real path uses the sealed task-454 regimes and differentiable
``_render_chart -> contest R -> frozen CPU SegNet`` graph.  All HVP-derived
quantities are empirical training signal.  They never close ``Lip(DJ)`` or
full-SegNet activation-cell custody and therefore never become certificates.
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
for path in (REPO, REPO / "src", REPO / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import probe_segnet_validation_certificate as sealed_probe  # noqa: E402

from tac.scorer_surrogate.costate_trust_region import (  # noqa: E402
    apply_direct_costate_correction,
    array_sha256,
    torch_fixed_adjoint_jacobian_hvp,
)

SCHEMA = "jacobian_drift_full_costate_probe.v1"
SOURCE_RECEIPT = (
    REPO
    / "experiments/results/costate_trust_region_economics_20260713T032000Z/measurement_receipt.json"
)
SOURCE_RECEIPT_SHA256 = "60d76277ad02f0b0685fb369e8fbf9d11e4083fd5c34649528e963549d18c73e"
EXACT_FORWARD_THREAD_RECEIPT = (
    REPO / "experiments/results/segnet_exact_forward_20260713T020000Z/receipt.json"
)
EXACT_FORWARD_THREAD_RECEIPT_SHA256 = (
    "3b04a40c7c9e656cfc417dc60f2b73781e251a21fa02689a9e78523218ad3134"
)
SOURCE_PATHS = (
    "src/tac/scorer_surrogate/costate_trust_region.py",
    "src/tac/witness_dsl/costate_trust_region_policy.py",
    "tools/probe_jacobian_drift_certificate.py",
    "tools/probe_costate_trust_region_economics.py",
    "tools/probe_segnet_validation_certificate.py",
    "tools/probe_yopo_first_layer_costate.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _materialize_source_bundle(output_dir: Path, *, resume: bool) -> dict[str, Any]:
    """Copy exact uncommitted launch bytes and fail closed on resume drift."""

    manifest: dict[str, Any] = {}
    for relative in SOURCE_PATHS:
        source = REPO / relative
        destination = output_dir / "source_bundle" / relative
        source_bytes = source.read_bytes()
        source_sha = hashlib.sha256(source_bytes).hexdigest()
        if resume:
            if not destination.is_file() or hashlib.sha256(destination.read_bytes()).hexdigest() != source_sha:
                raise RuntimeError(f"source bundle custody changed: {relative}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + f".tmp.{os.getpid()}")
            with temporary.open("wb") as handle:
                handle.write(source_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        manifest[relative] = {
            "launch_path": str(source),
            "bundle_path": str(destination),
            "bytes": len(source_bytes),
            "sha256": source_sha,
        }
    return manifest


def _cosine(left: Any, right: Any) -> float:
    import torch

    # Accumulate in fp64 and clamp the final rounding residue.  The launch
    # bundle predating this guard emitted 37/64 fp32 values slightly above
    # one; those raw rows remain preserved but are not metric authority.
    left64 = left.detach().to(dtype=torch.float64)
    right64 = right.detach().to(dtype=torch.float64)
    denominator = torch.linalg.vector_norm(left64) * torch.linalg.vector_norm(right64)
    if float(denominator.item()) == 0.0:
        return 0.0
    value = float(torch.sum(left64 * right64).div(denominator).item())
    return min(1.0, max(-1.0, value))


def _norm(value: Any) -> float:
    import torch

    return float(torch.linalg.vector_norm(value).item())


def _exact_validate_theta(*, renderer: Any, theta: Any, segnet: Any, labels: Any) -> dict[str, float]:
    """Pure exact validation forward for one renderer state."""

    import torch
    import torch.nn.functional as functional

    started = time.perf_counter()
    with torch.inference_mode():
        scorer_input = sealed_probe._render_chart(renderer, theta)
        logits = segnet(scorer_input.permute(0, 3, 1, 2).contiguous())
        ce = float(functional.cross_entropy(logits, labels).item())
        dseg = float(
            np.mean(
                logits.detach().cpu().numpy().astype(np.float32, copy=False).argmax(1)
                != labels.detach().cpu().numpy()
            )
        )
    return {"ce": ce, "dseg_numpy_fp32": dseg, "seconds": time.perf_counter() - started}


def _fixed_q_terms(*, segnet: Any, scorer_input: Any, labels: Any, direction: Any) -> dict[str, Any]:
    """Measure fixed-q correction and full-loss HVP on the exact same graph."""

    import torch
    import torch.nn.functional as functional

    logits = segnet(scorer_input.permute(0, 3, 1, 2).contiguous())
    loss = functional.cross_entropy(logits, labels)
    q_anchor = torch.autograd.grad(loss, logits, create_graph=True, retain_graph=True)[0].detach()
    p_anchor = torch.autograd.grad(
        logits, scorer_input, grad_outputs=q_anchor, create_graph=True, retain_graph=True
    )[0]
    fixed_started = time.perf_counter()
    correction = torch_fixed_adjoint_jacobian_hvp(
        logits=logits,
        scorer_input=scorer_input,
        anchor_adjoint=q_anchor,
        direction=direction.detach(),
    )
    fixed_seconds = time.perf_counter() - fixed_started
    full_started = time.perf_counter()
    gradient = torch.autograd.grad(loss, scorer_input, create_graph=True, retain_graph=True)[0]
    full_hvp = torch.autograd.grad(
        torch.sum(gradient * direction.detach()), scorer_input, retain_graph=True
    )[0]
    full_seconds = time.perf_counter() - full_started
    return {
        "logits": logits,
        "loss": loss,
        "q_anchor": q_anchor,
        "p_anchor": p_anchor,
        "fixed_correction": correction,
        "full_loss_hvp": full_hvp,
        "adjoint_drift": full_hvp - correction,
        "timing": {
            "fixed_q_jacobian_hvp_seconds": fixed_seconds,
            "full_loss_hvp_diagnostic_seconds": full_seconds,
        },
    }


def _exact_decomposition(
    *, segnet: Any, anchor_input: Any, current_input: Any, labels: Any, anchor_terms: dict[str, Any]
) -> dict[str, float]:
    """Re-derive the exact adjoint split with independently evaluated VJPs."""

    import torch
    import torch.nn.functional as functional

    current_logits = segnet(current_input.permute(0, 3, 1, 2).contiguous())
    current_loss = functional.cross_entropy(current_logits, labels)
    q_current = torch.autograd.grad(
        current_loss, current_logits, create_graph=False, retain_graph=True
    )[0].detach()
    p_current = torch.autograd.grad(
        current_logits, current_input, grad_outputs=q_current, retain_graph=True
    )[0]
    j_current_q = p_current
    j_anchor_q_current = torch.autograd.grad(
        anchor_terms["logits"], anchor_input, grad_outputs=q_current, retain_graph=True
    )[0]
    j_anchor_delta_q = torch.autograd.grad(
        anchor_terms["logits"],
        anchor_input,
        grad_outputs=q_current - anchor_terms["q_anchor"],
        retain_graph=True,
    )[0]
    rhs = (j_current_q - j_anchor_q_current) + j_anchor_delta_q
    lhs = p_current - anchor_terms["p_anchor"]
    return {
        "fresh_costate_norm": _norm(p_current),
        "uncorrected_error": _norm(lhs),
        "exact_split_rhs_norm": _norm(rhs),
        "exact_split_residual": _norm(lhs - rhs),
        "fresh_ce": float(current_loss.item()),
        "fresh_dseg_numpy_fp32": float(
            np.mean(
                current_logits.detach().cpu().numpy().astype(np.float32, copy=False).argmax(1)
                != labels.detach().cpu().numpy()
            )
        ),
        "p_current": p_current,
    }


def _measure_regime(*, source_regime: dict[str, Any], segnet: Any, labels: Any) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    regime = str(source_regime["regime"])
    checkpoint = sealed_probe.CHECKPOINT_DIR / sealed_probe.REGIMES[regime]
    renderer, code, _model, _dash = sealed_probe._load_renderer(checkpoint)
    theta = torch.as_tensor(code[1], dtype=torch.float32).clone().requires_grad_(True)
    anchor_input = sealed_probe._render_chart(renderer, theta)
    anchor_logits = segnet(anchor_input.permute(0, 3, 1, 2).contiguous())
    anchor_loss = functional.cross_entropy(anchor_logits, labels)
    anchor_ce = float(anchor_loss.item())
    anchor_dseg = float(
        np.mean(
            anchor_logits.detach().cpu().numpy().astype(np.float32, copy=False).argmax(1)
            != labels.detach().cpu().numpy()
        )
    )
    theta_gradient = torch.autograd.grad(anchor_loss, theta)[0].detach()
    theta_norm = max(_norm(theta.detach()), 1.0)
    gradient_norm = _norm(theta_gradient)
    if not math.isfinite(gradient_norm) or gradient_norm == 0.0:
        raise RuntimeError("anchor renderer gradient is zero or nonfinite")
    measured_safe_steps = [
        row
        for row in source_regime["decisions"]
        if not bool(row["inherited_exact_control"]["ce_worsens"])
        and not bool(row["inherited_exact_control"]["dseg_worsens"])
    ]
    if not measured_safe_steps:
        raise RuntimeError("sealed source regime has no measured CE+dseg-safe step fraction")
    matched_step_fraction = max(float(row["fraction"]) for row in measured_safe_steps)
    rows: list[dict[str, Any]] = []
    anchor_source_sha = array_sha256(anchor_input.detach().cpu().numpy())
    theta_ladder_direction = -theta_norm / gradient_norm * theta_gradient
    jvp_started = time.perf_counter()
    _, collinear_input_direction = torch.autograd.functional.jvp(
        lambda value: sealed_probe._render_chart(renderer, value),
        theta,
        theta_ladder_direction,
        create_graph=False,
        strict=True,
    )
    jvp_seconds = time.perf_counter() - jvp_started
    collinear_anchor_graph = sealed_probe._render_chart(renderer, theta)
    collinear_terms = _fixed_q_terms(
        segnet=segnet,
        scorer_input=collinear_anchor_graph,
        labels=labels,
        direction=collinear_input_direction,
    )
    for source_row in source_regime["decisions"]:
        index = int(source_row["candidate_index"])
        fraction = float(source_row["fraction"])
        candidate_theta = (theta.detach() + fraction * theta_ladder_direction).requires_grad_(True)
        current_live = sealed_probe._render_chart(renderer, candidate_theta)
        current_input = current_live.detach().requires_grad_(True)
        displacement = current_input - anchor_input.detach()
        # Rebuild the anchor graph for each candidate so no retained graph is
        # accidentally shared across resumability boundaries.
        anchor_graph = sealed_probe._render_chart(renderer, theta)
        anchor_terms = _fixed_q_terms(
            segnet=segnet, scorer_input=anchor_graph, labels=labels, direction=displacement
        )
        corrected = apply_direct_costate_correction(
            anchor_costate=anchor_terms["p_anchor"].detach().cpu().numpy(),
            correction=anchor_terms["fixed_correction"].detach().cpu().numpy(),
        )
        exact_started = time.perf_counter()
        exact = _exact_decomposition(
            segnet=segnet,
            anchor_input=anchor_graph,
            current_input=current_input,
            labels=labels,
            anchor_terms=anchor_terms,
        )
        exact_seconds = time.perf_counter() - exact_started
        validation_started = time.perf_counter()
        with torch.inference_mode():
            validation_logits = segnet(current_input.detach().permute(0, 3, 1, 2).contiguous())
            _ = validation_logits.argmax(1)
        pure_validation_seconds = time.perf_counter() - validation_started
        corrected_tensor = torch.as_tensor(corrected, dtype=current_input.dtype)
        corrected_error = _norm(exact["p_current"] - corrected_tensor)
        corrected_cosine = _cosine(exact["p_current"], corrected_tensor)
        scaled_linear_displacement = fraction * collinear_input_direction
        scaled_correction = fraction * collinear_terms["fixed_correction"]
        scaled_estimate = collinear_terms["p_anchor"] + scaled_correction
        estimated_renderer_gradient = torch.autograd.grad(
            current_live,
            candidate_theta,
            grad_outputs=corrected_tensor,
            allow_unused=False,
        )[0]
        fresh_input_for_theta = sealed_probe._render_chart(renderer, candidate_theta)
        fresh_logits_for_theta = segnet(fresh_input_for_theta.permute(0, 3, 1, 2).contiguous())
        fresh_loss_for_theta = functional.cross_entropy(fresh_logits_for_theta, labels)
        fresh_renderer_gradient = torch.autograd.grad(fresh_loss_for_theta, candidate_theta)[0]
        estimated_norm = _norm(estimated_renderer_gradient)
        fresh_norm = _norm(fresh_renderer_gradient)
        candidate_norm = max(_norm(candidate_theta.detach()), 1.0)
        if estimated_norm == 0.0 or fresh_norm == 0.0:
            raise RuntimeError("matched-window renderer gradient is zero")
        corrected_step_theta = (
            candidate_theta.detach()
            - matched_step_fraction * candidate_norm / estimated_norm * estimated_renderer_gradient.detach()
        )
        fresh_step_theta = (
            candidate_theta.detach()
            - matched_step_fraction * candidate_norm / fresh_norm * fresh_renderer_gradient.detach()
        )
        corrected_step = _exact_validate_theta(
            renderer=renderer,
            theta=corrected_step_theta,
            segnet=segnet,
            labels=labels,
        )
        fresh_step = _exact_validate_theta(
            renderer=renderer,
            theta=fresh_step_theta,
            segnet=segnet,
            labels=labels,
        )
        rows.append(
            {
                "candidate_index": index,
                "fraction": fraction,
                "anchor_scorer_input_sha256": anchor_source_sha,
                "candidate_scorer_input_sha256": array_sha256(
                    current_input.detach().cpu().numpy()
                ),
                "scorer_input_displacement_l2": _norm(displacement),
                "uncorrected_banked_costate_error_l2": exact["uncorrected_error"],
                "corrected_costate_error_l2": corrected_error,
                "corrected_costate_cosine": corrected_cosine,
                "exact_decomposition_residual_l2": exact["exact_split_residual"],
                "fixed_q_correction_norm_l2": _norm(anchor_terms["fixed_correction"]),
                "full_loss_hvp_norm_l2": _norm(anchor_terms["full_loss_hvp"]),
                "adjoint_drift_norm_l2": _norm(anchor_terms["adjoint_drift"]),
                "collinear_anchor_hvp": {
                    "scaled_correction_error_l2": _norm(exact["p_current"] - scaled_estimate),
                    "scaled_correction_cosine": _cosine(exact["p_current"], scaled_estimate),
                    "jvp_vs_real_displacement_residual_l2": _norm(
                        displacement - scaled_linear_displacement
                    ),
                    "one_anchor_renderer_jvp_seconds": jvp_seconds,
                    "one_anchor_fixed_q_hvp_seconds": collinear_terms["timing"][
                        "fixed_q_jacobian_hvp_seconds"
                    ],
                    "authority": "EMPIRICAL_UNLESS_RENDERER_HESSIAN_AND_FIXED_R_CELL_BOUNDED",
                },
                "renderer_gradient_dot_fresh": float(
                    torch.sum(estimated_renderer_gradient * fresh_renderer_gradient).item()
                ),
                "renderer_gradient_cosine_fresh": _cosine(
                    estimated_renderer_gradient, fresh_renderer_gradient
                ),
                "fresh_exact_ce": exact["fresh_ce"],
                "fresh_exact_dseg_numpy_fp32": exact["fresh_dseg_numpy_fp32"],
                "fresh_exact_ce_delta_from_anchor": exact["fresh_ce"] - anchor_ce,
                "fresh_exact_dseg_delta_from_anchor": (
                    exact["fresh_dseg_numpy_fp32"] - anchor_dseg
                ),
                "matched_one_step_window": {
                    "step_fraction": matched_step_fraction,
                    "step_fraction_provenance": (
                        "MEASURED largest registered source fraction with inherited exact "
                        "CE and dseg both nonworsening"
                    ),
                    "current": {
                        "ce": exact["fresh_ce"],
                        "dseg_numpy_fp32": exact["fresh_dseg_numpy_fp32"],
                    },
                    "corrected_step": corrected_step,
                    "fresh_exact_step": fresh_step,
                    "corrected_ce_delta_vs_current": corrected_step["ce"] - exact["fresh_ce"],
                    "corrected_dseg_delta_vs_current": (
                        corrected_step["dseg_numpy_fp32"] - exact["fresh_dseg_numpy_fp32"]
                    ),
                    "fresh_ce_delta_vs_current": fresh_step["ce"] - exact["fresh_ce"],
                    "fresh_dseg_delta_vs_current": (
                        fresh_step["dseg_numpy_fp32"] - exact["fresh_dseg_numpy_fp32"]
                    ),
                    "corrected_minus_fresh_ce": corrected_step["ce"] - fresh_step["ce"],
                    "corrected_minus_fresh_dseg": (
                        corrected_step["dseg_numpy_fp32"] - fresh_step["dseg_numpy_fp32"]
                    ),
                    "measurement_only_shadow_forwards": 2,
                },
                "timing_seconds": anchor_terms["timing"]
                | {
                    "fresh_exact_input_costate_forward_backward_shadow": exact_seconds,
                    "pure_exact_validation_forward": pure_validation_seconds,
                },
                "authority": "EMPIRICAL_TRAINING_SIGNAL_ONLY",
            }
        )
    if not rows:
        raise RuntimeError("sealed regime has no registered candidates")
    ordered = sorted(rows, key=lambda row: float(row["scorer_input_displacement_l2"]))
    characterized: list[dict[str, Any]] = []
    for row in ordered:
        window = row["matched_one_step_window"]
        safe = (
            float(window["corrected_ce_delta_vs_current"]) <= 0.0
            and float(window["corrected_dseg_delta_vs_current"]) <= 0.0
            and float(window["corrected_minus_fresh_dseg"]) <= 0.0
        )
        if not safe:
            break
        characterized.append(row)
    return {
        "regime": regime,
        "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
        "anchor_scorer_input_sha256": anchor_source_sha,
        "anchor": {"exact_ce": anchor_ce, "exact_dseg_numpy_fp32": anchor_dseg},
        "candidates": rows,
        "rigorous_certified_reuses": 0,
        "rigorous_blockers": [
            "no custody-bearing Lip(DJ) whole-ball upper bound",
            "no full-SegNet C2,1 activation-cell radius artifact",
            "no coercive norm-conversion artifact for inherited margin-Fisher geometry",
            "no correction numerical-error upper-bound artifact",
            "no current corrected renderer-gradient lower bound over the ball",
        ],
        "empirical_characterization_only": True,
        "empirical_oracle_characterized_safe_ball": {
            "radius_l2": (
                max(float(row["scorer_input_displacement_l2"]) for row in characterized)
                if characterized
                else 0.0
            ),
            "corrected_reuse_count": len(characterized),
            "candidate_count": len(rows),
            "beats_inherited_1_of_64_count": len(characterized) > 1,
            "admission_semantics": (
                "matched one-step corrected update exact-validates CE/dseg against current and "
                "fresh-gradient control"
            ),
            "authority": "EMPIRICAL_ORACLE_CHARACTERIZATION_NOT_CERTIFICATE",
        },
    }


def _canaries() -> dict[str, Any]:
    import torch

    torch.set_default_dtype(torch.float64)
    x = torch.tensor([0.4], requires_grad=True)
    logits = torch.stack((x, 2.0 * x))
    q = torch.tensor([[3.0], [-2.0]])
    affine = torch_fixed_adjoint_jacobian_hvp(
        logits=logits, scorer_input=x, anchor_adjoint=q, direction=torch.ones_like(x)
    )
    y = torch.tensor([0.5], requires_grad=True)
    cubic_logits = torch.stack((y.square(), y.pow(3)))
    correction = torch_fixed_adjoint_jacobian_hvp(
        logits=cubic_logits,
        scorer_input=y,
        anchor_adjoint=torch.tensor([[2.0], [-1.0]]),
        direction=torch.tensor([0.25]),
    )
    expected = torch.tensor([0.25 * (4.0 - 6.0 * 0.5)])
    torch.set_default_dtype(torch.float32)
    return {
        "status": "PASS"
        if float(affine.abs().max().item()) == 0.0 and torch.allclose(correction, expected)
        else "FAIL",
        "affine_fixed_q_jacobian_drift_norm": float(affine.abs().max().item()),
        "quadratic_cubic_fixed_q_correction": float(correction.item()),
        "quadratic_cubic_expected": float(expected.item()),
        "affine_scope": "fixed-q Jacobian drift only; changing CE adjoint is not asserted zero",
    }


def _immutable_bundle(*, source_bundle: dict[str, Any] | None = None) -> dict[str, Any]:
    if _sha256(SOURCE_RECEIPT) != SOURCE_RECEIPT_SHA256:
        raise RuntimeError("sealed task-454 source receipt custody changed")
    if _sha256(EXACT_FORWARD_THREAD_RECEIPT) != EXACT_FORWARD_THREAD_RECEIPT_SHA256:
        raise RuntimeError("sealed task-456 thread-control receipt custody changed")
    thread_receipt = json.loads(EXACT_FORWARD_THREAD_RECEIPT.read_text())
    selected_threads = int(thread_receipt["control_law"]["selected_threads"])
    if selected_threads <= 0 or thread_receipt.get("verdict") != "GO":
        raise RuntimeError("task-456 thread-control receipt does not contain an admitted thread count")
    return {
        "schema": SCHEMA,
        "source_receipt": {"path": str(SOURCE_RECEIPT), "sha256": SOURCE_RECEIPT_SHA256},
        "source_bytes": (
            source_bundle
            if source_bundle is not None
            else {
                path: {
                    "launch_path": str(REPO / path),
                    "bytes": (REPO / path).stat().st_size,
                    "sha256": _sha256(REPO / path),
                }
                for path in SOURCE_PATHS
            }
        ),
        "inputs": {
            "exact_forward_thread_control": {
                "path": str(EXACT_FORWARD_THREAD_RECEIPT),
                "sha256": EXACT_FORWARD_THREAD_RECEIPT_SHA256,
                "selected_threads": selected_threads,
                "authority": "MEASURED task-456 local exact-argmax thread control",
            },
            "segnet": {"path": str(sealed_probe.SEGNET), "sha256": _sha256(sealed_probe.SEGNET)},
            "gt_cache": {
                "path": str(sealed_probe.GT_CACHE),
                "sha256": _sha256(sealed_probe.GT_CACHE),
            },
        },
        "authority": {
            "axis": "[macOS-CPU advisory; torch-fp32; training-signal]",
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from tac.boundary_math.seg_core import load_real_segnet

    output_root = (REPO / "experiments/results").resolve()
    output_dir = args.output_dir.resolve()
    output_dir.relative_to(output_root)
    receipt_path = output_dir / "measurement_receipt.json"
    source_bundle = _materialize_source_bundle(output_dir, resume=args.resume)
    immutable = _immutable_bundle(source_bundle=source_bundle)
    torch.set_num_threads(int(immutable["inputs"]["exact_forward_thread_control"]["selected_threads"]))
    if args.resume and not receipt_path.is_file():
        raise RuntimeError("--resume requires an existing receipt")
    if receipt_path.exists() and not args.resume:
        raise RuntimeError("fresh run refuses to overwrite existing evidence")
    receipt = json.loads(receipt_path.read_text()) if args.resume else immutable | {
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
        "controls": _canaries(),
        "regimes": [],
    }
    for key, value in immutable.items():
        if receipt.get(key) != value:
            raise RuntimeError(f"resume custody changed: {key}")
    if args.resume and receipt.get("status") == "MEASURED":
        return receipt
    _atomic_write_json(receipt_path, receipt)
    if receipt["controls"]["status"] != "PASS":
        raise RuntimeError("analytic fixed-q HVP canaries failed")
    source = json.loads(SOURCE_RECEIPT.read_text())
    torch.manual_seed(20260713)
    np.random.seed(20260713)
    torch.use_deterministic_algorithms(True)
    segnet = load_real_segnet("cpu").eval()
    for parameter in segnet.parameters():
        parameter.requires_grad_(False)
    with np.load(sealed_probe.GT_CACHE, allow_pickle=False) as cache:
        labels = torch.as_tensor(np.asarray(cache["lstars"])[0], dtype=torch.long).unsqueeze(0)
    completed = {str(row["regime"]) for row in receipt["regimes"]}
    for source_regime in source["regimes"]:
        regime = str(source_regime["regime"])
        if regime in completed:
            continue
        row = _measure_regime(source_regime=source_regime, segnet=segnet, labels=labels)
        checkpoint_path = output_dir / f"regime_{regime}.json"
        _atomic_write_json(checkpoint_path, row)
        receipt["regimes"].append(row)
        _atomic_write_json(receipt_path, receipt)
    fixed_times = [
        candidate["timing_seconds"]["fixed_q_jacobian_hvp_seconds"]
        for regime in receipt["regimes"]
        for candidate in regime["candidates"]
    ]
    validation_times = [
        candidate["timing_seconds"]["pure_exact_validation_forward"]
        for regime in receipt["regimes"]
        for candidate in regime["candidates"]
    ]
    shadow_times = [
        candidate["timing_seconds"]["fresh_exact_input_costate_forward_backward_shadow"]
        for regime in receipt["regimes"]
        for candidate in regime["candidates"]
    ]
    matched_window_times = [
        candidate["matched_one_step_window"][arm]["seconds"]
        for regime in receipt["regimes"]
        for candidate in regime["candidates"]
        for arm in ("corrected_step", "fresh_exact_step")
    ]
    baseline = source["baseline_counts"]
    baseline_forwards = int(baseline["operational_validation_forwards"])
    baseline_calls = int(baseline["total_teacher_forward_backward"])
    matched_validation_median = float(np.median(matched_window_times))
    faithful_equivalents = [
        len(regime["candidates"])
        * float(np.median([row["timing_seconds"]["fixed_q_jacobian_hvp_seconds"] for row in regime["candidates"]]))
        / float(np.median(validation_times))
        for regime in receipt["regimes"]
    ]
    collinear_equivalents = [
        (
            float(regime["candidates"][0]["collinear_anchor_hvp"]["one_anchor_renderer_jvp_seconds"])
            + float(regime["candidates"][0]["collinear_anchor_hvp"]["one_anchor_fixed_q_hvp_seconds"])
        )
        / float(np.median(validation_times))
        for regime in receipt["regimes"]
    ]
    faithful_totals = [1.0 + value for value in faithful_equivalents]
    collinear_totals = [1.0 + value for value in collinear_equivalents]
    faithful_matched_lower_bounds = []
    collinear_matched_characterizations = []
    for regime in receipt["regimes"]:
        regime_hvp_median = float(
            np.median(
                [
                    row["timing_seconds"]["fixed_q_jacobian_hvp_seconds"]
                    for row in regime["candidates"]
                ]
            )
        )
        sampled_prefix = int(
            regime["empirical_oracle_characterized_safe_ball"]["corrected_reuse_count"]
        )
        per_step_lower_bound = regime_hvp_median / matched_validation_median
        faithful_matched_lower_bounds.append(
            {
                "regime": regime["regime"],
                "sampled_ray_safe_prefix_count": sampled_prefix,
                "incremental_hvp_lower_bound_validation_equivalents_per_step": (
                    per_step_lower_bound
                ),
                "total_including_one_anchor_validation_lower_bound": (
                    1.0 + sampled_prefix * per_step_lower_bound
                ),
                "omitted_costs": (
                    "anchor forward/loss/q setup, renderer projection, gate, graph retention, "
                    "and any exact shadows"
                ),
            }
        )
        collinear = regime["candidates"][0]["collinear_anchor_hvp"]
        residual_ratios = [
            float(row["collinear_anchor_hvp"]["jvp_vs_real_displacement_residual_l2"])
            / float(row["scorer_input_displacement_l2"])
            for row in regime["candidates"]
            if float(row["scorer_input_displacement_l2"]) > 0.0
        ]
        collinear_matched_characterizations.append(
            {
                "regime": regime["regime"],
                "total_including_one_anchor_validation_lower_bound": (
                    1.0
                    + (
                        float(collinear["one_anchor_renderer_jvp_seconds"])
                        + float(collinear["one_anchor_fixed_q_hvp_seconds"])
                    )
                    / matched_validation_median
                ),
                "median_linearized_vs_real_displacement_residual_ratio": float(
                    np.median(residual_ratios)
                ),
                "admissible": False,
                "reason": (
                    "no renderer-Hessian/fixed-R-cell bound; timing is characterization only"
                ),
            }
        )
    receipt["economics"] = {
        "baseline_counts": {
            "operational_validation_forwards": baseline_forwards,
            "total_teacher_forward_backward": baseline_calls,
        },
        "baseline_validation_forwards_per_teacher_call": baseline_forwards / baseline_calls,
        "fixed_q_hvp_median_seconds": float(np.median(fixed_times)),
        "pure_exact_validation_forward_median_seconds": float(np.median(validation_times)),
        "fresh_exact_input_costate_forward_backward_shadow_median_seconds": float(
            np.median(shadow_times)
        ),
        "matched_one_step_measurement_only_shadow_forwards": len(matched_window_times),
        "matched_one_step_measurement_only_shadow_seconds_total": float(
            np.sum(np.asarray(matched_window_times, dtype=np.float64))
        ),
        "matched_full_through_r_validation_median_seconds": matched_validation_median,
        "faithful_matched_validation_lower_bounds": faithful_matched_lower_bounds,
        "faithful_matched_validation_verdict": (
            "NO_GO_WHERE_SAMPLED_HIGH_REUSE_EXISTS; incremental lower bound already exceeds "
            "8.375 before omitted costs"
        ),
        "collinear_matched_validation_characterization": collinear_matched_characterizations,
        "collinear_admissible": False,
        "correction_cost_in_validation_forward_equivalents": (
            float(np.median(fixed_times)) / float(np.median(validation_times))
        ),
        "faithful_per_candidate_hvp_validation_forward_equivalents_per_anchor": faithful_equivalents,
        "collinear_anchor_hvp_plus_renderer_jvp_validation_forward_equivalents_per_anchor": (
            collinear_equivalents
        ),
        "faithful_total_including_anchor_validation_equivalents_per_anchor": faithful_totals,
        "collinear_total_including_anchor_validation_equivalents_per_anchor": collinear_totals,
        "fresh_shadows_visible_and_excluded_only_from_hypothetical_operational_path": True,
        "legacy_model_only_forward_equivalents_warning": (
            "the pure_exact_validation_forward denominator excludes render and CE; compare it "
            "only as a model-kernel diagnostic, never apples-to-apples with task-454 validation"
        ),
    }
    receipt["verdict"] = {
        "rigorous_certificate": "BLOCKED",
        "rigorous_certified_reuses": 0,
        "empirical_rows": sum(len(row["candidates"]) for row in receipt["regimes"]),
        "inherited_margin_fisher_comparator": "1/64 proxy reuse",
        "go_eligible": False,
        "pointer_moved": False,
        "verdict_scope": (
            "pair0; sealed early/boundary/late regimes; fixed-adjoint first-order HVP; "
            "current Torch/CPU substrate; registered task-454 ladder/window"
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
        args.output_dir = REPO / "experiments/results" / f"jacobian_drift_certificate_{stamp}"
    if args.canaries_only:
        _atomic_write_json(
            args.output_dir / "measurement_receipt.json",
            {"schema": SCHEMA, "controls": _canaries(), "authority": "synthetic_only"},
        )
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
