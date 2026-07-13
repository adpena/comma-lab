#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Small deterministic checkpoint-bound probe for FiLM polar-chart SPEL.

This is deliberately not an evaluator or a score probe.  It reads one V9
checkpoint, verifies the function-preserving chart boundary, then compares the
NumPy MCSD/SPEL rule with an ambient-Muon reference on a constructed local
FiLM-map regression.  The JSON receipt labels that proxy scope explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from tac.optimization.film_polar_chart_spel_mlx import (
    FilmPolarChartSPELState,
    muon_aspect_ratio_scale,
    newton_schulz5_numpy,
    polar_chart_numpy,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_matmul(left: np.ndarray, right: np.ndarray, *, name: str) -> np.ndarray:
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        out = np.matmul(left, right)
    if not np.all(np.isfinite(out)):
        raise FloatingPointError(f"{name} produced non-finite values")
    return np.asarray(out)


def _loss_and_grad(weight: np.ndarray, inputs: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray]:
    residual = _finite_matmul(inputs, weight.T, name="local-regression forward") - target
    loss = float(np.mean(np.square(residual, dtype=np.float32), dtype=np.float32))
    scale = np.float32(2.0 / residual.size)
    grad = np.asarray(
        scale * _finite_matmul(residual.T, inputs, name="local-regression pullback"),
        dtype=np.float32,
    )
    return loss, grad


def _ambient_muon_step(
    weight: np.ndarray,
    momentum: np.ndarray,
    grad: np.ndarray,
    *,
    learning_rate: float,
    beta: float,
    weight_decay: float,
    ns_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    coupled_grad = np.asarray(grad + np.float32(weight_decay) * weight, dtype=np.float32)
    next_momentum = np.asarray(beta * momentum + (1.0 - beta) * coupled_grad, dtype=np.float32)
    drive = np.asarray((1.0 - beta) * coupled_grad + beta * next_momentum, dtype=np.float32)
    direction = newton_schulz5_numpy(drive, steps=ns_steps)
    effective_lr = np.float32(learning_rate * muon_aspect_ratio_scale(weight.shape))
    return np.asarray(weight - effective_lr * direction, dtype=np.float32), next_momentum


def _mlx_execution_status() -> dict[str, object]:
    try:
        import mlx.core as mx

        value = mx.array([1.0], dtype=mx.float32)
        mx.eval(value)
    except Exception as exc:  # exact infrastructure refusal belongs in the durable receipt
        return {
            "status": "BLOCKED",
            "exception_type": type(exc).__name__,
            "reason": str(exc),
        }
    return {"status": "AVAILABLE", "device": str(mx.default_device())}


def run_probe(checkpoint: Path, *, seed: int, steps: int) -> dict[str, object]:
    checkpoint = checkpoint.resolve()
    with np.load(checkpoint, allow_pickle=False) as payload:
        if "film.weight" not in payload:
            raise KeyError(f"{checkpoint} has no film.weight")
        weight0 = np.asarray(payload["film.weight"], dtype=np.float32)
        checkpoint_epoch = int(payload["__epoch"]) if "__epoch" in payload else None

    q0, h0 = polar_chart_numpy(weight0)
    reconstruction = np.asarray(_finite_matmul(q0, h0, name="boundary fold"), dtype=np.float32)
    reconstruction_relative_fro = float(
        np.linalg.norm(reconstruction - weight0, ord="fro") / np.linalg.norm(weight0, ord="fro")
    )
    direct_unit_projection_relative_fro = float(
        np.linalg.norm(q0 - weight0, ord="fro") / np.linalg.norm(weight0, ord="fro")
    )

    rng = np.random.default_rng(seed)
    inputs = rng.normal(size=(64, weight0.shape[1])).astype(np.float32)
    local_delta = rng.normal(size=(64, weight0.shape[0])).astype(np.float32)
    initial_output = _finite_matmul(inputs, weight0.T, name="initial local output")
    local_delta *= np.float32(0.01 * np.std(initial_output) / np.std(local_delta))
    target = np.asarray(initial_output + local_delta, dtype=np.float32)

    control = weight0.copy()
    control_momentum = np.zeros_like(control)
    treatment = FilmPolarChartSPELState()
    treatment.initialize_numpy(weight0)
    initial_loss, _ = _loss_and_grad(weight0, inputs, target)
    treatment_weight = weight0.copy()
    for _ in range(int(steps)):
        _, control_grad = _loss_and_grad(control, inputs, target)
        control, control_momentum = _ambient_muon_step(
            control,
            control_momentum,
            control_grad,
            learning_rate=0.002,
            beta=0.95,
            weight_decay=1e-4,
            ns_steps=5,
        )
        _, treatment_grad = _loss_and_grad(treatment_weight, inputs, target)
        treatment_weight = treatment.step_numpy(
            treatment_grad,
            learning_rate=0.002 * muon_aspect_ratio_scale(weight0.shape),
            ema_decay=0.997,
        )

    control_loss, _ = _loss_and_grad(control, inputs, target)
    treatment_loss, _ = _loss_and_grad(treatment_weight, inputs, target)
    _, control_h = polar_chart_numpy(control)
    treatment_h = np.asarray(treatment.h0, dtype=np.float32)
    control_h_relative_drift = float(np.linalg.norm(control_h - h0) / np.linalg.norm(h0))
    treatment_h_relative_drift = float(np.linalg.norm(treatment_h - h0) / np.linalg.norm(h0))
    mlx_status = _mlx_execution_status()

    return {
        "schema": "film_polar_chart_spel_local_micro_probe.v1",
        "authority": "MEASURED_LOCAL_NUMPY_FP32_PROXY_NON_PROMOTABLE",
        "verdict_scope": (
            "checkpoint-bound chart/reproducibility mechanics plus constructed local FiLM-map regression; "
            "NOT a trainer fine-tune, NOT through-R, NOT d_seg/d_pose, NOT n600"
        ),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_epoch": checkpoint_epoch,
        "film_shape": list(weight0.shape),
        "seed": int(seed),
        "steps": int(steps),
        "muon_learning_rate": 0.002,
        "muon_aspect_scale": muon_aspect_ratio_scale(weight0.shape),
        "effective_film_learning_rate": 0.002 * muon_aspect_ratio_scale(weight0.shape),
        "boundary": {
            "reconstruction_relative_fro": reconstruction_relative_fro,
            "direct_unit_projection_relative_fro": direct_unit_projection_relative_fro,
            "q_orthogonality_residual_fro": float(
                np.linalg.norm(
                    _finite_matmul(q0.T, q0, name="boundary Q.T@Q")
                    - np.eye(q0.shape[1]),
                    ord="fro",
                )
            ),
            "h0_sigma_min": float(np.linalg.svd(h0.astype(np.float64), compute_uv=False)[-1]),
            "h0_sigma_max": float(np.linalg.svd(h0.astype(np.float64), compute_uv=False)[0]),
        },
        "constructed_local_regression": {
            "initial_loss": initial_loss,
            "ambient_muon_final_loss": control_loss,
            "polar_spel_final_loss": treatment_loss,
            "ambient_muon_loss_ratio": control_loss / initial_loss,
            "polar_spel_loss_ratio": treatment_loss / initial_loss,
            "ambient_muon_h_relative_drift": control_h_relative_drift,
            "polar_spel_h_relative_drift": treatment_h_relative_drift,
        },
        "required_next_authority": (
            "matched governed local MLX fine-tune from this checkpoint, then n600 finishing-stage through-R verdict"
        ),
        "requested_local_mlx_micro_ab": {
            "status": "BLOCKED" if mlx_status["status"] == "BLOCKED" else "READY",
            "source_checkpoint_epoch": checkpoint_epoch,
            "derived_first_finisher_epoch": (None if checkpoint_epoch is None else checkpoint_epoch + 1),
            "matched_steps": int(steps),
            "seed": int(seed),
            "control_dsl_factory": "MuonAtCheckpointBoundary",
            "treatment_dsl_factory": "FilmPolarChartSPELManifoldMuon",
            "verdict_owed": "n600 at next real finishing stage",
        },
        "mlx_execution": mlx_status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=469)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be >= 1")
    receipt = run_probe(args.checkpoint, seed=args.seed, steps=args.steps)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, output)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
