#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Local OSS-reconciliation tournament for INSTANT on frozen SegNet.

The implementation under test adapts the official MIT INSTANT algorithm but
copies no upstream source bytes.  It keeps the scorer forward exact, projects
only frozen ungrouped 1x1-Conv2d input adjoints, calibrates rank by retained
singular energy plus the official oversampling rule, and chooses the smaller
channel/spatial projection axis per layer.

The probe is read-only with respect to trainers, live run directories, and the
frontier.  It writes only small, atomic receipts beneath ``experiments/results``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO / "tools"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from probe_yopo_first_layer_costate import (  # noqa: E402
    _evaluate_teacher,
    _load_renderer,
    _render_chart,
    _renderer_parity_canary,
    _select_candidate_recess,
)

from tac.boundary_math.instant_projected_adjoint import (  # noqa: E402
    AdaptiveProjectorCalibration,
    ProjectionProof,
    calibrate_adaptive_projector_numpy,
    instant_pointwise_conv2d,
    load_calibration,
    save_calibration,
)
from tac.boundary_math.segnet_gradient_replacement import measure_costate_agreement  # noqa: E402
from tac.witness_dsl.scorer_gradient_policy import (  # noqa: E402
    InstantAdmissionEconomics,
    instant_validation_economics,
)

SCHEMA = "instant_oss_reconciliation_probe.v2"
RUN_MANIFEST_SCHEMA = "instant_oss_run_manifest.v2"
CHECKPOINT_ENVELOPE_SCHEMA = "instant_run_checkpoint_envelope.v1"
CALIBRATION_STAGE_SCHEMA = "instant_calibration_stage.v2"
REGIME_STAGE_SCHEMA = "instant_regime_stage.v2"
SEED = 20260712
ENERGY_TARGETS = (0.90, 0.95, 0.99)
OVERSAMPLING = 5
VALIDATION_HORIZONS = (2, 4, 8)
validation_economics = instant_validation_economics
SEGNET_SHA256 = "68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6"
GT_SHA256 = "e3f5ce8e79374ed0b9a3f007167dd7488862b51420f0b25b7bcec7ee6865f63e"
REGIMES = (
    {
        "name": "early_ce_boundary",
        "stage": "CE",
        "epoch": 299,
        "path": "experiments/results/levelset_l7_preserved_snapshots/levelset_ckpt_stageCE_ep299.npz",
        "sha256": "e76e02c0c88c2a20b06e0faf800640f3b71798cb113cbd1f66e59021a5c5ac72",
    },
    {
        "name": "boundary_tau_boundary",
        "stage": "tau_softplus",
        "epoch": 899,
        "path": "experiments/results/levelset_l7_preserved_snapshots/levelset_ckpt_stageTau_ep899.npz",
        "sha256": "d8b579d8a7d7da109e1b0a5cc517fe2dee11584a5417c238553c6b473afdb82a",
    },
    {
        "name": "late_l7_boundary",
        "stage": "l7_softplus",
        "epoch": 1500,
        "path": "experiments/results/levelset_l7_preserved_snapshots/levelset_ckpt_stageL7_ep1500.npz",
        "sha256": "fc859c76977e6e9c853fe0d6cadbbe403ba3375fff3dfb2f13652b4fb01ad270",
    },
)
SOURCE_PATHS = (
    "tools/probe_instant_projected_adjoint.py",
    "tools/probe_yopo_first_layer_costate.py",
    "src/tac/boundary_math/instant_projected_adjoint.py",
    "src/tac/boundary_math/segnet_gradient_replacement.py",
    "src/tac/scorer.py",
    "src/tac/witness_dsl/scorer_gradient_policy.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_array(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_exact_keys(payload: Any, expected: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError(f"{label} schema mismatch")
    return payload


@dataclass(frozen=True)
class InstantRunCheckpointEnvelope:
    """Content-addressed partial stage bound to one exact run manifest."""

    kind: str
    run_binding: dict[str, Any]
    identity: dict[str, Any]
    payload: dict[str, Any]

    def body(self) -> dict[str, Any]:
        return {
            "schema": CHECKPOINT_ENVELOPE_SCHEMA,
            "kind": self.kind,
            "run_binding": self.run_binding,
            "identity": self.identity,
            "payload": self.payload,
        }

    def to_dict(self) -> dict[str, Any]:
        body = self.body()
        return {**body, "content_sha256": _json_sha256(body)}

    def write(self, path: Path) -> None:
        _atomic_json(path, self.to_dict())

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        expected_kind: str,
        expected_run_binding: dict[str, Any],
        expected_identity: dict[str, Any],
    ) -> InstantRunCheckpointEnvelope:
        raw = _require_exact_keys(
            json.loads(path.read_text(encoding="utf-8")),
            {"schema", "kind", "run_binding", "identity", "payload", "content_sha256"},
            label="checkpoint envelope",
        )
        body = {key: raw[key] for key in ("schema", "kind", "run_binding", "identity", "payload")}
        if raw["schema"] != CHECKPOINT_ENVELOPE_SCHEMA:
            raise ValueError("checkpoint envelope schema mismatch")
        if raw["kind"] != expected_kind:
            raise ValueError("checkpoint envelope kind mismatch")
        if raw["run_binding"] != expected_run_binding:
            raise ValueError("checkpoint envelope run binding mismatch")
        if raw["identity"] != expected_identity:
            raise ValueError("checkpoint envelope stage identity mismatch")
        if _json_sha256(body) != raw["content_sha256"]:
            raise ValueError("checkpoint envelope content hash mismatch")
        if not isinstance(raw["payload"], dict):
            raise ValueError("checkpoint envelope payload schema mismatch")
        return cls(
            kind=raw["kind"],
            run_binding=raw["run_binding"],
            identity=raw["identity"],
            payload=raw["payload"],
        )


def _run_binding(run_manifest: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    return {
        "schema": "instant_run_binding.v1",
        "run_manifest_sha256": _sha256(manifest_path),
        "source_custody_sha256": _json_sha256(run_manifest["source_custody"]),
        "input_custody_sha256": _json_sha256(run_manifest["input_custody"]),
    }


def _authenticate_terminal_receipt(
    path: Path,
    *,
    expected_sha256: str,
    expected_run_binding: dict[str, Any],
) -> dict[str, Any]:
    _require_sha256(expected_sha256, label="expected terminal receipt hash")
    if _sha256(path) != expected_sha256:
        raise ValueError("terminal receipt external SHA-256 mismatch")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("schema") != SCHEMA:
        raise ValueError("terminal receipt schema mismatch")
    if receipt.get("run_binding") != expected_run_binding:
        raise ValueError("terminal receipt run binding mismatch")
    if _json_sha256(receipt.get("source_custody")) != expected_run_binding.get(
        "source_custody_sha256"
    ):
        raise ValueError("terminal receipt source custody is not bound to the run manifest")
    if receipt.get("score_claim") is not False or receipt.get("pointer_moved") is not False:
        raise ValueError("terminal receipt exceeded research-only authority")
    return receipt


def _timing(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    return {
        "median_seconds": median,
        "mad_seconds": float(np.median(np.abs(array - median))),
        "samples_seconds": [float(value) for value in array],
        "sample_count": len(values),
    }


def _ratio_timing(numerators: list[float], denominators: list[float]) -> dict[str, Any]:
    if len(numerators) != len(denominators):
        raise ValueError("paired timing arms require the same sample count")
    ratios = [left / right for left, right in zip(numerators, denominators, strict=True)]
    record = _timing(ratios)
    record["unit"] = "dense_seconds_per_projected_second"
    record["conservative_lower_bound"] = record["median_seconds"] - record["mad_seconds"]
    return record


def _eligible_convolutions(model: Any) -> dict[str, Any]:
    import torch

    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Conv2d)
        and tuple(module.kernel_size) == (1, 1)
        and tuple(module.stride) == (1, 1)
        and tuple(module.padding) == (0, 0)
        and tuple(module.dilation) == (1, 1)
        and module.groups == 1
    }


def _render_regime(regime: dict[str, Any]) -> tuple[Any, Any, Any, dict[str, Any]]:
    import torch

    checkpoint = REPO / str(regime["path"])
    if _sha256(checkpoint) != regime["sha256"]:
        raise ValueError(f"checkpoint custody mismatch: {checkpoint}")
    renderer, code, _model, _dash = _load_renderer(checkpoint)
    theta = torch.as_tensor(code[1], dtype=torch.float32).clone().requires_grad_(True)
    parity = _renderer_parity_canary(renderer, theta)
    if parity["max_abs"] != 0.0:
        raise RuntimeError("settled receiver parity canary failed")
    frame_nhwc = _render_chart(renderer, theta)
    return renderer, theta, frame_nhwc.permute(0, 3, 1, 2).contiguous(), parity


def _capture_output_cotangents(model: Any, frame: Any, labels: Any) -> tuple[dict[str, np.ndarray], Any]:
    import torch.nn.functional as functional

    eligible = _eligible_convolutions(model)
    captured: dict[str, np.ndarray] = {}
    handles = []
    for name, module in eligible.items():

        def hook(_module: Any, _inputs: Any, output: Any, *, layer_name: str = name) -> None:
            output.register_hook(
                lambda gradient, bound_name=layer_name: captured.__setitem__(
                    bound_name, gradient.detach().cpu().contiguous().numpy().copy()
                )
            )

        handles.append(module.register_forward_hook(hook))
    value = frame.detach().clone().requires_grad_(True)
    try:
        functional.cross_entropy(model(value), labels).backward()
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != set(eligible):
        raise RuntimeError("calibration capture did not cover every eligible pointwise convolution")
    if value.grad is None:
        raise RuntimeError("calibration capture produced no scorer input costate")
    return captured, value.grad.detach().clone()


def _build_bank(
    calibration_samples: dict[str, list[np.ndarray]], *, energy_target: float
) -> tuple[dict[str, AdaptiveProjectorCalibration], dict[str, Any]]:
    started = time.perf_counter()
    bank = {
        name: calibrate_adaptive_projector_numpy(
            np.stack(samples), energy_target=energy_target, oversampling=OVERSAMPLING
        )
        for name, samples in calibration_samples.items()
    }
    elapsed = time.perf_counter() - started
    axes = {axis: sum(item.axis == axis for item in bank.values()) for axis in ("channels", "spatial")}
    ranks = [item.rank for item in bank.values()]
    base_ranks = [item.base_rank for item in bank.values()]
    return bank, {
        "seconds": elapsed,
        "layer_count": len(bank),
        "axes": axes,
        "base_rank": {"min": min(base_ranks), "median": float(np.median(base_ranks)), "max": max(base_ranks)},
        "rank_after_oversampling": {"min": min(ranks), "median": float(np.median(ranks)), "max": max(ranks)},
        "retained_energy_min": min(item.retained_energy for item in bank.values()),
        "calibration_samples_per_layer": len(next(iter(calibration_samples.values()))),
    }


def _persist_banks(
    output_dir: Path,
    banks: dict[float, tuple[dict[str, AdaptiveProjectorCalibration], dict[str, Any]]],
    *,
    run_binding: dict[str, Any],
    capture_custody: dict[str, Any],
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema": CALIBRATION_STAGE_SCHEMA,
        "capture_custody": capture_custody,
        "targets": {},
    }
    for target, (bank, metadata) in banks.items():
        target_key = f"{target:.2f}"
        layer_records: dict[str, Any] = {}
        target_dir = output_dir / "calibration" / target_key
        target_dir.mkdir(parents=True, exist_ok=True)
        for index, (name, calibration) in enumerate(sorted(bank.items())):
            path = target_dir / f"{index:03d}.npz"
            save_calibration(path, calibration)
            layer_records[name] = {
                "path": str(path.relative_to(output_dir)),
                "sha256": _sha256(path),
                "metadata": calibration.metadata(),
                "basis_sha256": _hash_array(calibration.basis),
                "singular_values_sha256": _hash_array(calibration.singular_values),
            }
        manifest["targets"][target_key] = {"summary": metadata, "layers": layer_records}
    identity = {
        "stage": "calibration",
        "energy_targets": [f"{target:.2f}" for target in ENERGY_TARGETS],
        "eligible_layers": sorted(next(iter(banks.values()))[0]),
    }
    InstantRunCheckpointEnvelope(
        kind="calibration",
        run_binding=run_binding,
        identity=identity,
        payload=manifest,
    ).write(output_dir / "calibration_checkpoint.json")
    return manifest


def _load_banks(
    output_dir: Path,
    *,
    run_binding: dict[str, Any],
    capture_custody: dict[str, Any],
    expected_layers: set[str],
    calibration_samples: dict[str, list[np.ndarray]],
) -> dict[float, tuple[dict[str, AdaptiveProjectorCalibration], dict[str, Any]]]:
    checkpoint = output_dir / "calibration_checkpoint.json"
    identity = {
        "stage": "calibration",
        "energy_targets": [f"{target:.2f}" for target in ENERGY_TARGETS],
        "eligible_layers": sorted(expected_layers),
    }
    manifest = InstantRunCheckpointEnvelope.load(
        checkpoint,
        expected_kind="calibration",
        expected_run_binding=run_binding,
        expected_identity=identity,
    ).payload
    _require_exact_keys(
        manifest, {"schema", "capture_custody", "targets"}, label="calibration stage"
    )
    if manifest.get("schema") != CALIBRATION_STAGE_SCHEMA:
        raise ValueError("unsupported calibration stage checkpoint")
    if manifest["capture_custody"] != capture_custody:
        raise ValueError("calibration capture custody mismatch")
    if set(manifest["targets"]) != {f"{target:.2f}" for target in ENERGY_TARGETS}:
        raise ValueError("calibration checkpoint target coverage mismatch")
    banks: dict[float, tuple[dict[str, AdaptiveProjectorCalibration], dict[str, Any]]] = {}
    for target_key, target_record in manifest["targets"].items():
        _require_exact_keys(target_record, {"summary", "layers"}, label="calibration target")
        if set(target_record["layers"]) != expected_layers:
            raise ValueError("calibration checkpoint layer coverage mismatch")
        bank: dict[str, AdaptiveProjectorCalibration] = {}
        for name, record in target_record["layers"].items():
            _require_exact_keys(
                record,
                {"path", "sha256", "metadata", "basis_sha256", "singular_values_sha256"},
                label="calibration layer",
            )
            path = output_dir / record["path"]
            if _sha256(path) != record["sha256"]:
                raise ValueError(f"calibration checkpoint custody mismatch: {path}")
            calibration = load_calibration(path)
            if calibration.metadata() != record["metadata"]:
                raise ValueError(f"calibration metadata mismatch: {name}")
            if _hash_array(calibration.basis) != record["basis_sha256"]:
                raise ValueError(f"calibration basis mismatch: {name}")
            if _hash_array(calibration.singular_values) != record["singular_values_sha256"]:
                raise ValueError(f"calibration singular values mismatch: {name}")
            if calibration.source_fingerprint != _hash_array(
                np.asarray(np.stack(calibration_samples[name]), dtype=np.float64)
            ):
                raise ValueError(f"calibration source cotangent mismatch: {name}")
            bank[name] = calibration
        banks[float(target_key)] = (bank, target_record["summary"])
    if set(banks) != set(ENERGY_TARGETS):
        raise ValueError("calibration checkpoint target coverage mismatch")
    return banks


@contextlib.contextmanager
def _projected_pointwise_context(
    model: Any, bank: dict[str, AdaptiveProjectorCalibration]
) -> Any:
    eligible = _eligible_convolutions(model)
    if set(eligible) != set(bank):
        raise ValueError("projection bank does not exactly cover eligible 1x1 convolutions")
    originals: dict[str, Any] = {}
    proof = ProjectionProof()
    try:
        for name, module in eligible.items():
            originals[name] = module.forward
            calibration = bank[name]

            def projected(value: Any, *, bound=module, bound_calibration=calibration) -> Any:
                return instant_pointwise_conv2d(
                    value,
                    bound.weight,
                    bound.bias,
                    bound_calibration,
                    proof=proof,
                )

            module.forward = projected
        yield proof
    finally:
        for name, module in eligible.items():
            module.forward = originals[name]


def _measure_dense(model: Any, frame: Any, labels: Any, *, samples: int, warmups: int) -> dict[str, Any]:
    import torch.nn.functional as functional

    for _ in range(warmups):
        value = frame.detach().clone().requires_grad_(True)
        functional.cross_entropy(model(value), labels).backward()
    forward: list[float] = []
    backward: list[float] = []
    costate = None
    logits = None
    for _ in range(samples):
        value = frame.detach().clone().requires_grad_(True)
        started = time.perf_counter()
        logits = model(value)
        forward_done = time.perf_counter()
        functional.cross_entropy(logits, labels).backward()
        backward_done = time.perf_counter()
        forward.append(forward_done - started)
        backward.append(backward_done - forward_done)
        costate = value.grad.detach().clone()
    if costate is None or logits is None:
        raise RuntimeError("dense arm produced no costate")
    return {
        "forward": _timing(forward),
        "backward": _timing(backward),
        "total": _timing([a + b for a, b in zip(forward, backward, strict=True)]),
        "total_samples": [a + b for a, b in zip(forward, backward, strict=True)],
        "costate": costate,
        "logits": logits.detach(),
    }


def _measure_projected(
    model: Any,
    frame: Any,
    labels: Any,
    bank: dict[str, AdaptiveProjectorCalibration],
    dense_logits: Any,
    *,
    samples: int,
    warmups: int,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    with _projected_pointwise_context(model, bank) as proof:
        # Compare in the same grad-enabled CPU execution mode as the dense arm.
        # PyTorch may select a numerically different Conv2d kernel when the
        # scorer input does not require gradients.
        forward_check = model(frame.detach().clone().requires_grad_(True))
        if not torch.equal(forward_check, dense_logits):
            raise RuntimeError(
                f"projected provider changed exact forward: maxabs={float((forward_check-dense_logits).abs().max())}"
            )
        for _ in range(warmups):
            value = frame.detach().clone().requires_grad_(True)
            functional.cross_entropy(model(value), labels).backward()
        forward: list[float] = []
        backward: list[float] = []
        costate = None
        for _ in range(samples):
            value = frame.detach().clone().requires_grad_(True)
            started = time.perf_counter()
            logits = model(value)
            forward_done = time.perf_counter()
            functional.cross_entropy(logits, labels).backward()
            backward_done = time.perf_counter()
            forward.append(forward_done - started)
            backward.append(backward_done - forward_done)
            costate = value.grad.detach().clone()
    if costate is None:
        raise RuntimeError("projected arm produced no input costate")
    expected_calls = len(bank) * (warmups + samples)
    if proof.backward_calls != expected_calls or proof.dense_conv2d_input_calls != 0:
        raise RuntimeError("projected pointwise kernel proof failed")
    return {
        "forward": _timing(forward),
        "backward": _timing(backward),
        "total": _timing([a + b for a, b in zip(forward, backward, strict=True)]),
        "total_samples": [a + b for a, b in zip(forward, backward, strict=True)],
        "costate": costate,
        "forward_exact": True,
        "kernel_proof": {
            "backward_calls": proof.backward_calls,
            "channel_axis_calls": proof.channel_axis_calls,
            "spatial_axis_calls": proof.spatial_axis_calls,
            "dense_conv2d_input_calls": proof.dense_conv2d_input_calls,
            "covered_pointwise_layers": len(bank),
            "all_other_convolutions_exact": True,
        },
    }


def _annulus_mask(margins: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    flat = np.asarray(margins, dtype=np.float64).reshape(-1)
    count = max(1, int(np.ceil(0.05 * flat.size)))
    threshold = float(np.partition(flat, count - 1)[count - 1])
    mask = np.asarray(margins) <= threshold
    return mask[None, None], {
        "control_law": "constant bottom-5-percent GT-margin annulus",
        "anchor": "canonical witness annulus_bottom_k=0.05",
        "threshold": threshold,
        "selected_pixels": int(mask.sum()),
    }


def renderer_descent_direction_gate(exact_gradient: Any, candidate_gradient: Any) -> dict[str, Any]:
    """Positive-inner-product gate with an explicit fp64 accumulation floor."""

    exact = np.asarray(exact_gradient, dtype=np.float64).reshape(-1)
    candidate = np.asarray(candidate_gradient, dtype=np.float64).reshape(-1)
    metrics = measure_costate_agreement(exact, candidate)
    dot_terms = int(exact.size)
    accumulated_roundoff = dot_terms * np.finfo(np.float64).eps
    if accumulated_roundoff >= 1.0:
        raise RuntimeError("renderer-gradient dot-product floor is not finite")
    cosine_floor = float(accumulated_roundoff / (1.0 - accumulated_roundoff))
    passed = bool(
        metrics.finite
        and metrics.cosine_similarity is not None
        and metrics.cosine_similarity > cosine_floor
    )
    return {
        "passed": passed,
        "cosine_similarity": metrics.cosine_similarity,
        "fp64_dot_gamma_n_floor": cosine_floor,
        "dot_terms": dot_terms,
        "predicate": "cosine(exact_renderer_gradient,candidate_renderer_gradient) > gamma_n",
    }


def _direction_receipt(
    *,
    renderer: Any,
    theta: Any,
    frame: Any,
    labels: Any,
    segnet: Any,
    exact_costate: Any,
    candidate_costate: Any,
    annulus: np.ndarray,
) -> dict[str, Any]:
    import torch

    global_metrics = measure_costate_agreement(
        exact_costate.detach().cpu().numpy(), candidate_costate.detach().cpu().numpy()
    )
    annulus_metrics = measure_costate_agreement(
        exact_costate.detach().cpu().numpy(),
        candidate_costate.detach().cpu().numpy(),
        mask=annulus,
    )
    exact_theta = torch.autograd.grad(
        frame, theta, grad_outputs=exact_costate, retain_graph=True
    )[0].detach()
    candidate_theta = torch.autograd.grad(
        frame, theta, grad_outputs=candidate_costate, retain_graph=True
    )[0].detach()
    renderer_metrics = measure_costate_agreement(
        exact_theta.cpu().numpy(), candidate_theta.cpu().numpy()
    )
    descent_gate = renderer_descent_direction_gate(
        exact_theta.cpu().numpy(), candidate_theta.cpu().numpy()
    )
    with torch.inference_mode():
        current_ce, current_dseg = _evaluate_teacher(segnet, frame.permute(0, 2, 3, 1), labels)
    exact_candidate, exact_step, exact_trials, exact_validations, exact_validation_seconds = _select_candidate_recess(
        renderer=renderer,
        theta=theta,
        candidate_grad=exact_theta,
        segnet=segnet,
        labels=labels,
        current_loss=current_ce,
        current_dseg=current_dseg,
    )
    candidate, candidate_step, trials, validations, validation_seconds = _select_candidate_recess(
        renderer=renderer,
        theta=theta,
        candidate_grad=candidate_theta,
        segnet=segnet,
        labels=labels,
        current_loss=current_ce,
        current_dseg=current_dseg,
    )
    return {
        "global_input_costate": global_metrics.to_dict(),
        "boundary_annulus_input_costate": annulus_metrics.to_dict(),
        "renderer_parameter_gradient": renderer_metrics.to_dict(),
        "renderer_descent_direction_gate": descent_gate,
        "exact_teacher_control": {
            "accepted": exact_candidate is not None,
            "step_norm": exact_step,
            "trials": exact_trials,
            "validation_forwards": exact_validations,
            "validation_seconds": exact_validation_seconds,
        },
        "projected_candidate": {
            "accepted": candidate is not None,
            "step_norm": candidate_step,
            "trials": trials,
            "validation_forwards": validations,
            "validation_seconds": validation_seconds,
        },
        "admission": bool(
            exact_candidate is not None
            and candidate is not None
            and global_metrics.finite
            and descent_gate["passed"]
        ),
        "control_law": {
            "type": "event-conditioned tested predicate with completion guarantee",
            "default_fraction": 1e-2,
            "recess": "halve until exact-teacher CE decreases and d_seg does not worsen; terminate at fp32 identity",
            "anchor": "same control used by the landed YOPO pair0 saved-regime probe",
        },
    }


def _authenticate_economics(record: dict[str, Any]) -> dict[str, Any]:
    terms = record.get("charged_terms")
    if not isinstance(terms, dict):
        raise ValueError("INSTANT economics charged terms schema mismatch")
    expected = InstantAdmissionEconomics(
        exact_seconds=terms.get("exact_refresh_seconds"),
        approximate_seconds=terms.get("projected_hot_step_seconds"),
        projected_candidate_validation_seconds=terms.get(
            "projected_candidate_validation_seconds"
        ),
        calibration_seconds=terms.get("calibration_seconds"),
        fallback_seconds=terms.get("fallback_seconds"),
    ).to_dict()
    if record != expected:
        raise ValueError("INSTANT economics derivation mismatch")
    return expected


def _arm_admitted(arm: dict[str, Any]) -> bool:
    economics = _authenticate_economics(arm["admission_economics"])
    return bool(
        arm["direction"]["admission"]
        and arm["paired_hot_step_speedup"]["conservative_lower_bound"] > 1.0
        and economics["admitted_cadences_K"]
    )


def _validate_regime_stage_payload(
    payload: dict[str, Any],
    *,
    expected_regime: dict[str, Any],
    expected_frame_sha256: str | None = None,
    expected_calibration: dict[float, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    _require_exact_keys(
        payload,
        {"regime", "renderer_parity_canary", "scorer_input_frame_sha256", "dense", "arms"},
        label="regime stage",
    )
    if payload["regime"] != expected_regime:
        raise ValueError("regime stage identity mismatch")
    _require_sha256(payload["scorer_input_frame_sha256"], label="scorer input frame")
    if (
        expected_frame_sha256 is not None
        and payload["scorer_input_frame_sha256"] != expected_frame_sha256
    ):
        raise ValueError("regime stage frame custody mismatch")
    arms = payload["arms"]
    if not isinstance(arms, list) or len(arms) != len(ENERGY_TARGETS):
        raise ValueError("regime stage arm coverage mismatch")
    if {arm.get("energy_target") for arm in arms if isinstance(arm, dict)} != set(
        ENERGY_TARGETS
    ):
        raise ValueError("regime stage energy target coverage mismatch")
    for arm in arms:
        _require_exact_keys(
            arm,
            {
                "energy_target",
                "oversampling",
                "calibration",
                "timing",
                "paired_hot_step_speedup",
                "admission_economics",
                "direction",
                "admitted",
            },
            label="regime arm",
        )
        target = float(arm["energy_target"])
        if arm["oversampling"] != OVERSAMPLING:
            raise ValueError("regime arm oversampling mismatch")
        if expected_calibration is not None and arm["calibration"] != expected_calibration[target]:
            raise ValueError("regime arm calibration summary mismatch")
        if arm["admitted"] != _arm_admitted(arm):
            raise ValueError("regime arm admission derivation mismatch")
    return payload


def _derive_receipt_science(receipt: dict[str, Any]) -> dict[str, Any]:
    regimes = receipt.get("regimes")
    if not isinstance(regimes, list) or len(regimes) != len(REGIMES):
        raise ValueError("terminal receipt regime coverage mismatch")
    by_name = {row.get("regime", {}).get("name"): row for row in regimes if isinstance(row, dict)}
    if set(by_name) != {row["name"] for row in REGIMES}:
        raise ValueError("terminal receipt regime identity mismatch")
    ordered = [
        _validate_regime_stage_payload(by_name[regime["name"]], expected_regime=regime)
        for regime in REGIMES
    ]
    admitted = [
        [row["regime"]["name"], arm["energy_target"]]
        for row in ordered
        for arm in row["arms"]
        if arm["admitted"]
    ]
    clears = [
        target
        for target in ENERGY_TARGETS
        if all(
            next(arm for arm in row["arms"] if arm["energy_target"] == target)["admitted"]
            for row in ordered
        )
    ]
    verdict = receipt.get("verdict")
    if not isinstance(verdict, dict):
        raise ValueError("terminal receipt verdict schema mismatch")
    expected_verdict = "GO" if clears else "NO_GO"
    if (
        verdict.get("verdict") != expected_verdict
        or verdict.get("admitted_regime_arms") != admitted
        or verdict.get("energy_targets_clearing_all_regimes") != clears
    ):
        raise ValueError("terminal receipt verdict derivation mismatch")
    return {"admitted_regime_arms": admitted, "energy_targets_clearing_all_regimes": clears}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--expected-terminal-sha256",
        help="external SHA-256 required to authenticate an already-complete receipt",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    try:
        output_dir.relative_to(REPO / "experiments/results")
    except ValueError as exc:
        raise ValueError("--output-dir must be beneath experiments/results") from exc
    if args.samples < 2 or args.warmups < 0 or args.threads < 1:
        raise ValueError("samples must be >=2, warmups >=0, and threads >=1")
    if output_dir.exists() and not args.resume:
        raise ValueError("output directory exists; pass --resume to authenticate and continue")
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    free_bytes = shutil.disk_usage(output_dir).free
    if free_bytes < 1024**3:
        raise RuntimeError("storage preflight refused: less than 1 GiB free")
    if _sha256(REPO / "upstream/models/segnet.safetensors") != SEGNET_SHA256:
        raise ValueError("SegNet custody mismatch")
    gt_path = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n6.npz"
    if _sha256(gt_path) != GT_SHA256:
        raise ValueError("GT cache custody mismatch")
    if os.environ.get("PYTHONHASHSEED") != str(args.seed):
        raise RuntimeError(f"launch requires PYTHONHASHSEED={args.seed}")

    import torch

    from tac.scorer import load_default_segnet

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(args.threads)
    with np.load(gt_path, allow_pickle=False) as payload:
        labels = torch.as_tensor(np.asarray(payload["lstars"][0], dtype=np.int64))[None]
        margins = np.asarray(payload["margins"][0], dtype=np.float32)
    annulus, annulus_metadata = _annulus_mask(margins)
    segnet = load_default_segnet(REPO / "upstream", device="cpu").eval()
    for parameter in segnet.parameters():
        parameter.requires_grad_(False)
    all_convolutions = sum(isinstance(module, torch.nn.Conv2d) for module in segnet.modules())
    eligible = _eligible_convolutions(segnet)
    source_custody = {path: _sha256(REPO / path) for path in SOURCE_PATHS}
    run_manifest = {
        "schema": RUN_MANIFEST_SCHEMA,
        "seed": args.seed,
        "samples": args.samples,
        "warmups": args.warmups,
        "threads": args.threads,
        "energy_targets": list(ENERGY_TARGETS),
        "oversampling": OVERSAMPLING,
        "source_custody": source_custody,
        "input_custody": {
            "segnet_sha256": SEGNET_SHA256,
            "gt_sha256": GT_SHA256,
            "checkpoint_sha256": {row["name"]: row["sha256"] for row in REGIMES},
        },
    }
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.is_file():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != run_manifest:
            raise ValueError("resume run manifest differs from current code/config/input custody")
    else:
        _atomic_json(manifest_path, run_manifest)
    run_binding = _run_binding(run_manifest, manifest_path)
    final_path = output_dir / "measurement_receipt.json"
    if args.resume and final_path.is_file():
        if args.expected_terminal_sha256 is None:
            raise ValueError(
                "completed resume requires --expected-terminal-sha256 from external custody"
            )
        terminal = _authenticate_terminal_receipt(
            final_path,
            expected_sha256=args.expected_terminal_sha256,
            expected_run_binding=run_binding,
        )
        _derive_receipt_science(terminal)
        print(
            json.dumps(
                {
                    "receipt": str(final_path.relative_to(REPO)),
                    "sha256": _sha256(final_path),
                    "verdict": terminal["verdict"]["verdict"],
                    "resume": "authenticated_terminal_no_rewrite",
                },
                sort_keys=True,
            )
        )
        return 0
    frames: dict[str, dict[str, Any]] = {}
    calibration_samples = {name: [] for name in eligible}
    capture_regimes: dict[str, Any] = {}
    calibration_checkpoint = output_dir / "calibration_checkpoint.json"
    reuse_calibration = args.resume and calibration_checkpoint.is_file()
    for regime in REGIMES:
        renderer, theta, frame, parity = _render_regime(regime)
        captured, exact_costate = _capture_output_cotangents(segnet, frame, labels)
        for name, value in captured.items():
            calibration_samples[name].append(value)
        frame_sha256 = _hash_array(frame.detach().cpu().numpy())
        exact_costate_sha256 = _hash_array(exact_costate.cpu().numpy())
        capture_regimes[regime["name"]] = {
            "regime": regime,
            "scorer_input_frame_sha256": frame_sha256,
            "exact_input_costate_sha256": exact_costate_sha256,
            "output_cotangent_sha256": {
                name: _hash_array(value) for name, value in sorted(captured.items())
            },
        }
        frames[regime["name"]] = {
            "renderer": renderer,
            "theta": theta,
            "frame": frame,
            "parity": parity,
            "calibration_exact_costate_sha256": exact_costate_sha256,
            "scorer_input_frame_sha256": frame_sha256,
        }

    capture_custody = {
        "schema": "instant_calibration_capture_custody.v1",
        "eligible_layers": sorted(eligible),
        "regimes": capture_regimes,
    }

    if reuse_calibration:
        banks = _load_banks(
            output_dir,
            run_binding=run_binding,
            capture_custody=capture_custody,
            expected_layers=set(eligible),
            calibration_samples=calibration_samples,
        )
    else:
        banks = {
            target: _build_bank(calibration_samples, energy_target=target)
            for target in ENERGY_TARGETS
        }
        _persist_banks(
            output_dir,
            banks,
            run_binding=run_binding,
            capture_custody=capture_custody,
        )

    regimes_receipt: list[dict[str, Any]] = []
    for regime in REGIMES:
        item = frames[regime["name"]]
        stage_path = output_dir / "stages" / f"{regime['name']}.json"
        stage_identity = {
            "regime": regime,
            "scorer_input_frame_sha256": item["scorer_input_frame_sha256"],
        }
        expected_calibration = {target: banks[target][1] for target in ENERGY_TARGETS}
        if args.resume and stage_path.is_file():
            stage = InstantRunCheckpointEnvelope.load(
                stage_path,
                expected_kind="regime",
                expected_run_binding=run_binding,
                expected_identity=stage_identity,
            ).payload
            regimes_receipt.append(
                _validate_regime_stage_payload(
                    stage,
                    expected_regime=regime,
                    expected_frame_sha256=item["scorer_input_frame_sha256"],
                    expected_calibration=expected_calibration,
                )
            )
            continue
        dense = _measure_dense(
            segnet, item["frame"], labels, samples=args.samples, warmups=args.warmups
        )
        if (
            item["calibration_exact_costate_sha256"] is not None
            and _hash_array(dense["costate"].cpu().numpy()) != item["calibration_exact_costate_sha256"]
        ):
            raise RuntimeError("dense costate changed between calibration and tournament")
        arms: list[dict[str, Any]] = []
        for target in ENERGY_TARGETS:
            bank, calibration_metadata = banks[target]
            projected = _measure_projected(
                segnet,
                item["frame"],
                labels,
                bank,
                dense["logits"],
                samples=args.samples,
                warmups=args.warmups,
            )
            direction = _direction_receipt(
                renderer=item["renderer"],
                theta=item["theta"],
                frame=item["frame"],
                labels=labels,
                segnet=segnet,
                exact_costate=dense["costate"],
                candidate_costate=projected["costate"],
                annulus=annulus,
            )
            speedup = _ratio_timing(dense["total_samples"], projected["total_samples"])
            economics = InstantAdmissionEconomics(
                exact_seconds=dense["total"]["median_seconds"],
                approximate_seconds=projected["total"]["median_seconds"],
                projected_candidate_validation_seconds=direction["projected_candidate"][
                    "validation_seconds"
                ],
            ).to_dict()
            arm = {
                "energy_target": target,
                "oversampling": OVERSAMPLING,
                "calibration": calibration_metadata,
                "timing": {
                    key: value
                    for key, value in projected.items()
                    if key not in {"costate", "total_samples"}
                },
                "paired_hot_step_speedup": speedup,
                "admission_economics": economics,
                "direction": direction,
                "admitted": False,
            }
            arm["admitted"] = _arm_admitted(arm)
            arms.append(arm)
        regime_receipt = {
            "regime": regime,
            "renderer_parity_canary": item["parity"],
            "scorer_input_frame_sha256": item["scorer_input_frame_sha256"],
            "dense": {
                key: value
                for key, value in dense.items()
                if key not in {"costate", "logits", "total_samples"}
            },
            "arms": arms,
        }
        _validate_regime_stage_payload(
            regime_receipt,
            expected_regime=regime,
            expected_frame_sha256=item["scorer_input_frame_sha256"],
            expected_calibration=expected_calibration,
        )
        InstantRunCheckpointEnvelope(
            kind="regime",
            run_binding=run_binding,
            identity=stage_identity,
            payload=regime_receipt,
        ).write(stage_path)
        regimes_receipt.append(regime_receipt)

    admitted = [
        [row["regime"]["name"], arm["energy_target"]]
        for row in regimes_receipt
        for arm in row["arms"]
        if arm["admitted"]
    ]
    target_clears_all = [
        target
        for target in ENERGY_TARGETS
        if all(
            next(arm for arm in row["arms"] if arm["energy_target"] == target)["admitted"]
            for row in regimes_receipt
        )
    ]
    receipt = {
        "schema": SCHEMA,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "authority": "macOS-CPU advisory saved-regime local probe",
        "review_status": "recovery-written-UNREVIEWED",
        "score_claim": False,
        "pointer_moved": False,
        "paid_dispatch": False,
        "live_trainer_touched": False,
        "live_run_directory_touched": False,
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, text=True, capture_output=True
        ).stdout.strip(),
        "command": {"argv": [sys.executable, *sys.argv], "cwd": str(Path.cwd())},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "seed": args.seed,
            "threads": args.threads,
            "across_seed_variance": "UNKNOWN",
        },
        "storage_preflight_free_bytes": free_bytes,
        "cleanup": (
            "large activations are in-memory scratch; small content-addressed calibration NPZ files, "
            "per-stage envelopes, and this terminal JSON persist for crash resume"
        ),
        "source_custody": source_custody,
        "run_binding": run_binding,
        "input_custody": {
            "segnet": {"path": "upstream/models/segnet.safetensors", "sha256": SEGNET_SHA256},
            "gt": {"path": str(gt_path.relative_to(REPO)), "sha256": GT_SHA256},
            "checkpoints": {row["name"]: {"path": row["path"], "sha256": row["sha256"]} for row in REGIMES},
        },
        "oss_reference": {
            "repository": "https://github.com/hieu-trannn/INSTANT",
            "license": "MIT (official GitHub repository license detector and operator correction)",
            "source_copy": False,
            "adaptation": "equations and control flow re-expressed locally; no upstream file bytes copied",
            "verified_algorithm_surfaces": [
                "exact full forward",
                "adaptive smaller-axis output-cotangent projection",
                "1x1 Conv2d-only computer-vision registration",
                "energy retained-rank calibration",
                "oversampling default 5",
            ],
            "citation": (
                "Tuan-Kiet Doan, Trung-Hieu Tran, Enzo Tartaglione, Nikola Simidjievski, "
                "Van-Tam Nguyen (2026), INSTANT: Compressing Gradients and Activations for "
                "Resource-Efficient Training, ICLR 2026, OpenReview:P2q6Y7UweV"
            ),
            "arxiv_or_doi": "none found on the official paper page",
        },
        "control_laws": {
            "rank": {
                "type": "self-deriving formula",
                "formula": "min dimension rank reaching retained energy target plus oversampling=5",
                "targets": list(ENERGY_TARGETS),
                "anchor": "official INSTANT computer-vision example uses var=0.95 and over_sam=5",
            },
            "projection_axis": {
                "type": "event-conditioned tested predicate",
                "predicate": "channels if C_out <= H*W else spatial",
                "anchor": "official LinearSVDOp backward chooses the smaller adjoint axis",
            },
            "zero_variance": {
                "type": "event-conditioned tested predicate",
                "predicate": "all-zero calibration energy raises and skips the projected update",
                "anchor": "Dr.GRPO/DAPO zero-variance skip; no standard-deviation division occurs here",
            },
        },
        "coverage": {
            "all_conv2d_layers": all_convolutions,
            "eligible_oss_pointwise_layers": len(eligible),
            "all_other_conv2d_layers_exact": all_convolutions - len(eligible),
            "calibration_regimes": len(REGIMES),
            "calibration_is_on_policy": True,
            "calibration_description": "exact cotangents on the three sealed renders the witness states produce",
        },
        "annulus": annulus_metadata,
        "regimes": regimes_receipt,
        "clean_room_delta": {
            "terminal_clean_room_measurement": "UNKNOWN: the prior INSTANT process died before a terminal receipt",
            "algorithmic_difference": (
                "prior WIP projected every Conv2d on the spatial axis; this pass projects only 1x1 Conv2d and "
                "chooses the smaller channel/spatial axis"
            ),
            "verdict_delta": "UNKNOWN because there is no clean-room terminal measurement to subtract",
        },
        "verdict": {
            "verdict": "GO" if target_clears_all else "NO_GO",
            "admitted_regime_arms": admitted,
            "energy_targets_clearing_all_regimes": target_clears_all,
            "verdict_scope": (
                "n=1 pair0 at three sealed CE/tau/L7 states on macOS-CPU advisory; exact frozen-SegNet forward; "
                "INSTANT adaptive projection on eligible 1x1 Conv2d input adjoints; samples="
                f"{args.samples}; score_claim=false"
            ),
            "economic_interpretation": (
                "admission requires direction, paired hot-step median-minus-MAD > 1, and at least one of "
                "K={2,4,8} to clear the registered equal-refresh formula after charging measured "
                "projected-candidate validation; zero calibration/fallback are labeled optimistic "
                "lower-cost assumptions, so any ratio <=1 is a decisive NO_GO"
            ),
        },
        "rl_transfer": {
            "on_policy_distillation": "not applicable: no learned surrogate or offline dataset is used",
            "rloo": "not applicable: no learned value/costate estimator or sampled reward group is used",
            "zero_variance_skip": "implemented for all-zero calibration energy",
        },
        "stores_consulted": (
            "research(5715), equations(622), memory(1893), dag(505), council(277), tasks(96), docs(92); "
            "official INSTANT GitHub repository and official ICLR/OpenReview paper surfaces loaded; "
            "no paid provider, live trainer, live run directory, or upstream evaluator consulted"
        ),
    }
    receipt_path = final_path
    _derive_receipt_science(receipt)
    _atomic_json(receipt_path, receipt)
    print(
        json.dumps(
            {
                "receipt": str(receipt_path.relative_to(REPO)),
                "sha256": _sha256(receipt_path),
                "verdict": receipt["verdict"]["verdict"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
