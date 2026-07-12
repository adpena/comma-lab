#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Behavioral receipt for the frozen-SegNet input-costate injection seam.

The default probe is a fast deterministic Torch proof on a non-linear renderer
and teacher.  ``--real-segnet-cache-slice`` optionally runs a bounded n=1,
direct-slice, non-authority experiment with the canonical frozen SegNet and
exact input-costate caches refreshed every 1/2/4 steps.  Neither mode calls
``upstream/evaluate.py`` or makes a score/throughput/generalization claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.boundary_math.segnet_gradient_replacement import (  # noqa: E402
    costate_injection_loss_torch,
    measure_costate_agreement,
)

DEFAULT_OUTPUT = (
    REPO_ROOT
    / ".omx"
    / "research"
    / "artifacts"
    / "segnet_costate_injection_probe_20260712.json"
)
DEFAULT_GT_CACHE = REPO_ROOT / "experiments/results/mlx_fleet_gt_cache/gt_n6.npz"
DEFAULT_UPSTREAM = REPO_ROOT / "upstream"
DEFAULT_GT_CACHE_SHA256 = "e3f5ce8e79374ed0b9a3f007167dd7488862b51420f0b25b7bcec7ee6865f63e"
DEFAULT_SEGNET_SHA256 = "68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "UNAVAILABLE"


def _source_sha256s() -> dict[str, str]:
    relative_paths = (
        "tools/probe_segnet_costate_injection.py",
        "src/tac/boundary_math/segnet_gradient_replacement.py",
        "src/tac/witness_dsl/scorer_gradient_policy.py",
        "src/tac/canonical_equations/segnet_costate_injection_20260712.py",
    )
    return {name: _sha256_file(REPO_ROOT / name) for name in relative_paths}


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def _parameter_gradient_metrics(reference: Any, candidate: Any) -> dict[str, Any]:
    ref_np = reference.detach().cpu().numpy()
    cand_np = candidate.detach().cpu().numpy()
    agreement = measure_costate_agreement(ref_np, cand_np)
    return {
        **agreement.to_dict(),
        "max_absolute_error": float(np.max(np.abs(ref_np - cand_np))),
    }


def run_synthetic_proof(*, seed: int) -> dict[str, Any]:
    """Prove direct == exact-costate-injected and refute a wrong costate."""

    import torch

    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    dtype = torch.float64
    n_parameters = 9
    n_frame_values = 24
    hidden = 13

    theta = torch.randn(n_parameters, generator=generator, dtype=dtype) * 0.25
    theta.requires_grad_(True)
    render_a = torch.randn(
        n_frame_values, n_parameters, generator=generator, dtype=dtype
    ) / np.sqrt(n_parameters)
    render_b = torch.randn(
        n_frame_values, n_parameters, generator=generator, dtype=dtype
    ) / np.sqrt(n_parameters)
    teacher_w = torch.randn(hidden, n_frame_values, generator=generator, dtype=dtype) / np.sqrt(
        n_frame_values
    )
    teacher_v = torch.randn(hidden, generator=generator, dtype=dtype) / np.sqrt(hidden)
    teacher_target = torch.randn(hidden, generator=generator, dtype=dtype) * 0.2

    def renderer(parameters: Any) -> Any:
        linear = render_a @ parameters
        coupled = render_b @ torch.tanh(parameters)
        frame_flat = torch.sin(linear) + 0.15 * linear.square() + 0.2 * coupled
        return frame_flat.reshape(2, 3, 4)

    def teacher_loss(frame: Any) -> Any:
        activations = torch.tanh(teacher_w @ frame.reshape(-1))
        regression = (activations - teacher_target).square().mean()
        nonlocal_term = 0.07 * torch.logsumexp(activations + teacher_v, dim=0)
        return regression + nonlocal_term

    started = time.perf_counter()
    frame = renderer(theta)
    loss = teacher_loss(frame)
    exact_costate = torch.autograd.grad(loss, frame, retain_graph=True)[0]
    direct_gradient = torch.autograd.grad(loss, theta, retain_graph=True)[0]
    injected_loss = costate_injection_loss_torch(frame, exact_costate)
    injected_gradient = torch.autograd.grad(injected_loss, theta, retain_graph=True)[0]

    # A sign-reversed costate is shape-compatible and finite, so it is a useful
    # behavioral negative control: metadata-only tests cannot make it pass.
    wrong_costate = -exact_costate
    wrong_loss = costate_injection_loss_torch(frame, wrong_costate)
    wrong_gradient = torch.autograd.grad(wrong_loss, theta)[0]
    wall_seconds = time.perf_counter() - started

    exact_metrics = _parameter_gradient_metrics(direct_gradient, injected_gradient)
    wrong_metrics = _parameter_gradient_metrics(direct_gradient, wrong_gradient)
    proof_pass = bool(
        exact_metrics["valid"]
        and exact_metrics["cosine_similarity"] >= 1.0 - 1.0e-12
        and exact_metrics["max_absolute_error"] <= 1.0e-12
        and wrong_metrics["cosine_similarity"] <= -0.99
        and wrong_metrics["relative_l2_error"] >= 1.9
    )
    return {
        "status": "PASS" if proof_pass else "FAIL",
        "proof_pass": proof_pass,
        "seed": seed,
        "framework": "torch",
        "framework_version": torch.__version__,
        "dtype": str(dtype),
        "shapes": {
            "theta": list(theta.shape),
            "frame": list(frame.shape),
            "teacher_costate": list(exact_costate.shape),
        },
        "teacher_loss": float(loss.detach()),
        "direct_vs_exact_costate_injected": exact_metrics,
        "negative_control": {
            "construction": "lambda_wrong = -lambda_exact",
            "direct_vs_wrong_costate_injected": wrong_metrics,
            "must_fail": True,
            "failed_as_required": bool(
                wrong_metrics["cosine_similarity"] <= -0.99
                and wrong_metrics["relative_l2_error"] >= 1.9
            ),
        },
        "wall_seconds": wall_seconds,
        "authority": "synthetic behavioral identity proof; research-only",
        "score_claim": False,
    }


def _make_low_dimensional_basis(torch: Any, *, height: int, width: int) -> Any:
    y = torch.linspace(-1.0, 1.0, height, dtype=torch.float32)
    x = torch.linspace(-1.0, 1.0, width, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    spatial = torch.stack(
        (
            torch.ones_like(xx),
            torch.sin(np.pi * xx),
            torch.sin(np.pi * yy),
            torch.cos(np.pi * xx) * torch.cos(np.pi * yy),
        )
    )
    spatial = spatial / torch.sqrt(torch.mean(spatial.square(), dim=(1, 2), keepdim=True))
    basis = torch.zeros(12, 3, height, width, dtype=torch.float32)
    index = 0
    for channel in range(3):
        for pattern in spatial:
            basis[index, channel] = pattern
            index += 1
    return basis


def _real_cache_arm(
    *,
    torch: Any,
    segnet: Any,
    base_frame: Any,
    basis: Any,
    target_labels: Any,
    theta_initial: Any,
    refresh_interval: int,
    steps: int,
    learning_rate: float,
    render_scale: float,
) -> dict[str, Any]:
    import torch.nn.functional as functional

    theta = theta_initial.clone().detach().requires_grad_(True)

    def render(parameters: Any) -> Any:
        delta = render_scale * torch.einsum("d,dchw->chw", parameters, basis)
        return torch.clamp(base_frame + delta.unsqueeze(0), 0.0, 255.0)

    teacher_training_forwards = 0
    teacher_training_backwards = 0
    cached_costate = None
    refresh_rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for step in range(steps):
        if cached_costate is None or step % refresh_interval == 0:
            teacher_frame = render(theta).detach().requires_grad_(True)
            teacher_loss = functional.cross_entropy(segnet(teacher_frame), target_labels)
            teacher_training_forwards += 1
            cached_costate = torch.autograd.grad(teacher_loss, teacher_frame)[0].detach()
            teacher_training_backwards += 1
            refresh_rows.append(
                {
                    "step": step,
                    "teacher_ce": float(teacher_loss.detach()),
                    "costate_l2": float(torch.linalg.vector_norm(cached_costate)),
                }
            )

        frame = render(theta)
        injection = costate_injection_loss_torch(frame, cached_costate)
        theta_gradient = torch.autograd.grad(injection, theta)[0]
        with torch.no_grad():
            theta -= learning_rate * theta_gradient

    optimization_wall = time.perf_counter() - started
    with torch.inference_mode():
        final_frame = render(theta)
        final_logits = segnet(final_frame)
        final_ce = functional.cross_entropy(final_logits, target_labels)
        final_dseg = torch.mean((final_logits.argmax(dim=1) != target_labels).float())
    total_wall = time.perf_counter() - started
    return {
        "refresh_interval_steps": refresh_interval,
        "steps": steps,
        "teacher_calls": {
            "training_forward": teacher_training_forwards,
            "training_backward": teacher_training_backwards,
            "final_check_forward": 1,
            "total_forward": teacher_training_forwards + 1,
            "total_backward": teacher_training_backwards,
        },
        "refresh_observations": refresh_rows,
        "final_teacher_ce": float(final_ce),
        "final_exact_teacher_argmax_disagreement_n1": float(final_dseg),
        "theta_final_l2": float(torch.linalg.vector_norm(theta.detach())),
        "optimization_wall_seconds": optimization_wall,
        "total_wall_seconds_including_final_teacher_check": total_wall,
    }


def run_real_segnet_cache_slice(
    *,
    seed: int,
    gt_cache: Path,
    expected_gt_cache_sha256: str,
    upstream_dir: Path,
    expected_segnet_sha256: str,
    steps: int,
    learning_rate: float,
    render_scale: float,
) -> dict[str, Any]:
    """Run a bounded exact-costate-cache slice or return an honest blocker."""

    base_receipt: dict[str, Any] = {
        "scope": "n=1 direct real-GT last-frame short-horizon endpoint/cadence slice",
        "authority": "[macOS-CPU advisory] NON-PROMOTABLE; no upstream/evaluate.py",
        "evidence_kind": "short_horizon_endpoint_and_teacher_refresh_cadence",
        "fresh_nonrefresh_costate_agreement_measured": False,
        "generalization_claim": False,
        "throughput_win_claim": False,
        "score_claim": False,
        "cache_kind": "exact SegNet input costate refreshed on-trajectory",
        "explicit_non_equivalence": "this is not the inert LEVER-4 texture proxy",
        "refresh_intervals": [1, 2, 4],
        "steps": steps,
        "learning_rate": learning_rate,
        "render_scale_pixels": render_scale,
        "gt_cache": str(gt_cache),
        "expected_gt_cache_sha256": expected_gt_cache_sha256,
        "upstream_dir": str(upstream_dir),
        "expected_segnet_sha256": expected_segnet_sha256,
    }
    try:
        import torch
        import torch.nn.functional as functional

        if not gt_cache.is_file():
            raise FileNotFoundError(f"GT cache is missing: {gt_cache}")
        segnet_path = upstream_dir / "models" / "segnet.safetensors"
        if not segnet_path.is_file():
            raise FileNotFoundError(f"canonical SegNet weights are missing: {segnet_path}")
        actual_gt_cache_sha256 = _sha256_file(gt_cache)
        actual_segnet_sha256 = _sha256_file(segnet_path)
        base_receipt["gt_cache_sha256"] = actual_gt_cache_sha256
        base_receipt["segnet_sha256"] = actual_segnet_sha256
        if actual_gt_cache_sha256 != expected_gt_cache_sha256:
            raise ValueError(
                "GT cache SHA-256 does not match the declared input: "
                f"expected={expected_gt_cache_sha256}, actual={actual_gt_cache_sha256}"
            )
        if actual_segnet_sha256 != expected_segnet_sha256:
            raise ValueError(
                "SegNet SHA-256 does not match the declared input: "
                f"expected={expected_segnet_sha256}, actual={actual_segnet_sha256}"
            )
        if steps < 1:
            raise ValueError("real cache slice requires --real-steps >= 1")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("--real-learning-rate must be finite and > 0")
        if not np.isfinite(render_scale) or render_scale <= 0:
            raise ValueError("--real-render-scale must be finite and > 0")

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        with np.load(gt_cache, allow_pickle=False) as cache:
            if "gt_f1" not in cache.files:
                raise KeyError(f"{gt_cache} lacks required key 'gt_f1'")
            camera_frame = np.asarray(cache["gt_f1"][0], dtype=np.float32).copy()
        if camera_frame.ndim != 3 or camera_frame.shape[-1] != 3:
            raise ValueError(f"gt_f1[0] has unexpected shape {camera_frame.shape}")

        from tac.scorer import load_default_segnet

        load_started = time.perf_counter()
        segnet = load_default_segnet(upstream_dir, device="cpu").eval()
        load_wall = time.perf_counter() - load_started
        base_camera = torch.from_numpy(camera_frame).permute(2, 0, 1).unsqueeze(0)
        base_frame = functional.interpolate(
            base_camera, size=(384, 512), mode="bilinear", align_corners=False
        )
        basis = _make_low_dimensional_basis(torch, height=384, width=512)
        # ``target_labels`` is reused by gradient-enabled CE at refresh steps.
        # ``inference_mode`` would stamp it as an inference tensor that autograd
        # refuses to save; ``no_grad`` preserves an ordinary immutable tensor.
        with torch.no_grad():
            target_logits = segnet(base_frame)
            target_labels = target_logits.argmax(dim=1)

        generator = torch.Generator(device="cpu").manual_seed(seed)
        theta_initial = torch.randn(12, generator=generator, dtype=torch.float32) * 0.35
        with torch.no_grad():
            initial_delta = render_scale * torch.einsum(
                "d,dchw->chw", theta_initial, basis
            )
            initial_frame = torch.clamp(base_frame + initial_delta.unsqueeze(0), 0.0, 255.0)
            initial_logits = segnet(initial_frame)
            initial_ce = functional.cross_entropy(initial_logits, target_labels)
            initial_dseg = torch.mean(
                (initial_logits.argmax(dim=1) != target_labels).float()
            )

        arms = [
            _real_cache_arm(
                torch=torch,
                segnet=segnet,
                base_frame=base_frame,
                basis=basis,
                target_labels=target_labels,
                theta_initial=theta_initial,
                refresh_interval=interval,
                steps=steps,
                learning_rate=learning_rate,
                render_scale=render_scale,
            )
            for interval in (1, 2, 4)
        ]
        return {
            **base_receipt,
            "status": "MEASURED_SHORT_HORIZON_ENDPOINT_CADENCE_N1",
            "seed": seed,
            "framework": "torch",
            "framework_version": torch.__version__,
            "device": "cpu",
            "scorer": "canonical upstream SegNet / EfficientNet-B2",
            "scorer_sha256": actual_segnet_sha256,
            "input_shapes": {
                "camera_gt_f1": list(camera_frame.shape),
                "segnet_frame": list(base_frame.shape),
                "renderer_parameters": list(theta_initial.shape),
            },
            "teacher_model_load_wall_seconds": load_wall,
            "shared_target_teacher_forward_calls": 1,
            "initial_check_teacher_forward_calls": 1,
            "initial_teacher_ce": float(initial_ce),
            "initial_exact_teacher_argmax_disagreement_n1": float(initial_dseg),
            "arms": arms,
            "interpretation": (
                "Descriptive n=1 short-horizon endpoint and refresh-cadence evidence only. "
                "Fresh provider-vs-teacher costate agreement was not measured on non-refresh "
                "steps, so this receipt does not establish cache-gradient faithfulness there. "
                "A training-loop choice still requires multi-regime on-trajectory gradients, "
                "short-horizon checks, measured end-to-end duty cycle, and governed full-P=600 A/B."
            ),
        }
    except Exception as exc:  # fail closed into a durable blocker receipt
        return {
            **base_receipt,
            "status": "BLOCKED",
            "blocker_type": type(exc).__name__,
            "blocker": str(exc),
            "interpretation": "No real-cache result was produced; do not infer a win or loss.",
        }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument(
        "--real-segnet-cache-slice",
        action="store_true",
        help="also run the bounded n=1 canonical-SegNet cache-refresh slice",
    )
    parser.add_argument("--gt-cache", type=Path, default=DEFAULT_GT_CACHE)
    parser.add_argument("--gt-cache-sha256", default=DEFAULT_GT_CACHE_SHA256)
    parser.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--segnet-sha256", default=DEFAULT_SEGNET_SHA256)
    parser.add_argument("--real-steps", type=int, default=4)
    parser.add_argument("--real-learning-rate", type=float, default=0.2)
    parser.add_argument("--real-render-scale", type=float, default=24.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    command_argv = [sys.executable, *sys.argv]
    config = {
        "seed": args.seed,
        "output": str(args.output),
        "real_segnet_cache_slice": bool(args.real_segnet_cache_slice),
        "gt_cache": str(args.gt_cache),
        "gt_cache_sha256": args.gt_cache_sha256,
        "upstream_dir": str(args.upstream_dir),
        "segnet_sha256": args.segnet_sha256,
        "real_steps": args.real_steps,
        "real_learning_rate": args.real_learning_rate,
        "real_render_scale": args.real_render_scale,
    }
    synthetic = run_synthetic_proof(seed=args.seed)
    receipt: dict[str, Any] = {
        "schema": "segnet_costate_injection_probe_v1_20260712",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo_root": str(REPO_ROOT),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "git_head": _git_head(),
        "source_sha256s": _source_sha256s(),
        "command_argv": command_argv,
        "command_shell": shlex.join(command_argv),
        "config": config,
        "identity": (
            "grad_theta <stopgrad(lambda_hat), x(theta)> = J_x(theta)^T lambda_hat"
        ),
        "synthetic_proof": synthetic,
        "real_segnet_cache_slice_requested": bool(args.real_segnet_cache_slice),
        "score_claim": False,
        "frontier_pointer_delta": None,
    }
    exit_code = 0 if synthetic["proof_pass"] else 1
    if args.real_segnet_cache_slice:
        real_receipt = run_real_segnet_cache_slice(
            seed=args.seed,
            gt_cache=args.gt_cache.resolve(),
            expected_gt_cache_sha256=args.gt_cache_sha256,
            upstream_dir=args.upstream_dir.resolve(),
            expected_segnet_sha256=args.segnet_sha256,
            steps=args.real_steps,
            learning_rate=args.real_learning_rate,
            render_scale=args.real_render_scale,
        )
        receipt["real_segnet_cache_slice"] = real_receipt
        if real_receipt["status"] == "BLOCKED":
            exit_code = max(exit_code, 2)

    output = args.output.resolve()
    _atomic_json_write(output, receipt)
    print(json.dumps({
        "output": str(output),
        "synthetic_status": synthetic["status"],
        "real_status": (
            receipt.get("real_segnet_cache_slice", {}).get("status", "NOT_REQUESTED")
        ),
        "score_claim": False,
    }, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
