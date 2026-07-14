#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure guarded K=2 exact-input-costate reuse on an n600 real-state replay.

This probe is deliberately outside the live trainer.  It replays one real
renderer-code state per video pair, stratified over the three immutable V9
checkpoints used by the task-455 frozen replay.  For every state it performs:

1. an exact teacher-costate anchor step;
2. a second, common-norm stale-costate versus exact-costate comparison; and
3. an exact forward-only CE/d_seg/d_pose guard on the stale candidate.

The output is training-signal evidence only.  It never trains a model, mutates
an input run, evaluates an archive, or claims a contest score.  The replay is
resumable because every pair is an atomic record and each checkpoint cohort is
sealed by a stage manifest before the aggregate receipt is written.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SCHEMA = "p0_costate_reuse_k2_n600.v2"
PAIR_SCHEMA = "p0_costate_reuse_k2_pair.v2"
STAGE_SCHEMA = "p0_costate_reuse_k2_stage.v2"
LANE_ID = "lane_p0_backward_closer_20260713"
AXIS = "[macOS-CPU advisory; Torch/NumPy-fp32 training-gradient MEANS only]"
DEFAULT_OUTPUT = REPO / "experiments/results/p0_costate_reuse_k2_n600_v3_20260713"
DEFAULT_STORAGE_PLAN = REPO / ".omx/research/p0_costate_reuse_k2_v3_storage_preflight_20260713.json"
N_PAIRS = 600
SEED = 455
HOLDOUT_PERIOD = 5
STEP_FRACTION = 0.01
MAX_HALVINGS = 23
K_MAX = 2
DIAGNOSTIC_FORWARD_SHARE = 0.1784755863
DIAGNOSTIC_FORWARD_SHARE_PROVENANCE = (
    ".omx/research/p0_costate_reuse_gradfree_20260713.md; "
    "task-455 diagnostic ratio, explicitly unresolved in-loop"
)

OBJECTIVE_SPEC = {
    "schema": "p0_costate_reuse_k2_objective.v2",
    "anchor_step": "exact teacher costate followed by bounded fractional line search",
    "reuse_step": "raw input-costate zero-order hold at Kmax=2 with matched renderer-step norm",
    "accept_guard": {
        "ce": "strict descent from the current exact state",
        "d_seg": "nonworsening through-R exact SegNet argmax distance",
        "d_pose": "nonworsening through-R official PoseNet distance",
    },
    "fallback": "byte-exact rollback plus full exact-teacher refresh",
    "calibration_admission": {
        "economics": (
            "guarded diagnostic teacher-slice speedup strictly exceeds the "
            "forward-elimination Amdahl ceiling"
        ),
        "gradient": "every behaviorally accepted calibration row has renderer-gradient relative L2 < 1",
        "d_seg_regret": "every behaviorally accepted calibration row has stale-minus-exact d_seg <= 0",
        "runtime": "exact gradients are calibration-only and unavailable to the live reuse controller",
    },
}

ADMISSION_SPEC = {
    "schema": "p0_costate_reuse_k2_admission.v1",
    "complete_state_count": N_PAIRS,
    "required_accept_fraction_formula": "2*forward_share/(1-forward_share), strict greater-than",
    "accept_fraction_denominator": (
        "all n600 calibration states; terminal/blocked states are charged as fallback"
    ),
    "amdahl_ceiling_formula": "1/(1-forward_share)",
    "gradient_relative_l2_threshold": 1.0,
    "gradient_relative_l2_comparator": "strict_lt",
    "accepted_stale_minus_exact_d_seg_threshold": 0.0,
    "accepted_stale_minus_exact_d_seg_comparator": "lte",
    "whole_epoch_speedup": "UNKNOWN_IN_LOOP_TIMER_OWED",
}

SOURCE_FILES = (
    "tools/probe_p0_costate_reuse_k2.py",
    "tools/probe_frozen_replay_convex_head.py",
    "tools/probe_onpolicy_costate_matched_window.py",
    "tools/probe_onpolicy_scorer_surrogate.py",
    "tools/probe_yopo_first_layer_costate.py",
    "experiments/train_witness_realized_through_R_mlx.py",
    "src/tac/scorer_surrogate/frozen_replay_convex_head.py",
    "src/tac/boundary_math/segnet_gradient_replacement.py",
    "src/tac/cuda_levelset_training.py",
    "src/tac/local_acceleration/torch_levelset_inflate.py",
    "upstream/modules.py",
)


class ProbeError(RuntimeError):
    """The replay, custody, or fidelity contract failed closed."""


def _utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    header = f"{array.dtype.str}|{array.shape}".encode()
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())


def _load_module(name: str, relative: str) -> Any:
    path = REPO / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ProbeError(f"cannot import {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _file_custody(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _source_custody() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for relative in SOURCE_FILES:
        path = REPO / relative
        if not path.is_file():
            raise ProbeError(f"missing source file {relative}")
        result[relative] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    return result


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def _validate_storage_plan(path: Path, output_dir: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ProbeError(f"storage preflight is missing: {path}")
    payload = json.loads(path.read_text())
    if payload.get("blockers"):
        raise ProbeError(f"storage waterfall remains blocked: {payload['blockers']}")
    if payload.get("score_claim") is not False or payload.get("promotion_eligible") is not False:
        raise ProbeError("storage plan carries false authority")
    selected = payload.get("selected_tier")
    selected_root = payload.get("selected_workload_root")
    if not isinstance(selected, str) or not isinstance(selected_root, str):
        raise ProbeError("storage waterfall did not select a usable tier")
    if Path(selected_root).resolve() != output_dir.resolve():
        raise ProbeError("storage plan selected_workload_root does not match output_dir")
    if selected == "local" and not payload.get("operator_storage_policy", {}).get("local_disk_enabled"):
        raise ProbeError("local storage tier was not explicitly opted in")
    requested = payload.get("requested_bytes")
    if isinstance(requested, bool) or not isinstance(requested, int) or requested < 0:
        raise ProbeError("storage plan requested_bytes must be a non-negative integer")
    free_bytes = payload.get("selected_free_bytes")
    if isinstance(free_bytes, bool) or not isinstance(free_bytes, int) or free_bytes < 0:
        # Canonical waterfall v1 records capacity on the selected tier row.
        # Accept that schema only when the row is unique and fully bound.
        tier_rows = payload.get("tiers")
        matches = [
            row
            for row in tier_rows
            if isinstance(row, dict)
            and row.get("name") == selected
            and row.get("workload_root") == selected_root
            and row.get("eligible") is True
        ] if isinstance(tier_rows, list) else []
        if len(matches) != 1:
            raise ProbeError("storage plan selected_free_bytes is missing or invalid")
        free_bytes = matches[0].get("free_bytes")
        tier_requested = matches[0].get("requested_bytes")
        if (
            isinstance(free_bytes, bool)
            or not isinstance(free_bytes, int)
            or free_bytes < 0
            or isinstance(tier_requested, bool)
            or tier_requested != requested
        ):
            raise ProbeError("storage plan selected-tier capacity is missing or invalid")
    if free_bytes < requested:
        raise ProbeError("storage plan does not reserve the requested bytes")
    return {
        "plan": _file_custody(path),
        "selected_tier": selected,
        "selected_free_bytes": free_bytes,
        "requested_bytes": requested,
        "output_dir": str(output_dir),
        "large_artifacts_written": False,
    }


def _objective_sha256() -> str:
    return _canonical_sha256(OBJECTIVE_SPEC)


def _scorer_sha256(input_custody: dict[str, Any]) -> str:
    scorer_paths = (
        "upstream/models/segnet.safetensors",
        "upstream/models/posenet.safetensors",
    )
    return _canonical_sha256({path: input_custody[path] for path in scorer_paths})


def _acquire_lock(output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output_dir / ".probe.lock", os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(descriptor)
        raise ProbeError(f"another K2 replay owns {output_dir}") from exc
    return descriptor


def _release_lock(descriptor: int) -> None:
    fcntl.flock(descriptor, fcntl.LOCK_UN)
    os.close(descriptor)


def _pair_path(output_dir: Path, pair_index: int) -> Path:
    return output_dir / "pairs" / f"pair_{pair_index:04d}.json"


def _quantiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p10": None, "median": None, "mean": None, "p90": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.quantile(array, 0.50)),
        "mean": float(array.mean()),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(array.max()),
    }


def vector_metrics(reference: Any, candidate: Any) -> dict[str, float]:
    """NumPy-fp32 direction fidelity plus float64 reduction audit scalars."""

    ref = np.asarray(reference, dtype=np.float32).reshape(-1)
    cand = np.asarray(candidate, dtype=np.float32).reshape(-1)
    if ref.shape != cand.shape or not (np.isfinite(ref).all() and np.isfinite(cand).all()):
        raise ProbeError("vector fidelity received mismatched or nonfinite inputs")
    dot32 = float(np.sum(ref * cand, dtype=np.float32))
    ref2_32 = float(np.sum(ref * ref, dtype=np.float32))
    cand2_32 = float(np.sum(cand * cand, dtype=np.float32))
    delta = np.asarray(ref - cand, dtype=np.float32)
    delta2_32 = float(np.sum(delta * delta, dtype=np.float32))
    cosine = dot32 / math.sqrt(ref2_32 * cand2_32) if ref2_32 and cand2_32 else 0.0
    return {
        "dot_fp32": dot32,
        "reference_square_norm_fp32": ref2_32,
        "candidate_square_norm_fp32": cand2_32,
        "delta_square_norm_fp32": delta2_32,
        "cosine_fp32": cosine,
        "relative_l2_error_fp32": math.sqrt(delta2_32 / ref2_32) if ref2_32 else 0.0,
        "candidate_to_reference_norm_fp32": math.sqrt(cand2_32 / ref2_32) if ref2_32 else 0.0,
    }


def joint_guard(current: dict[str, float], candidate: dict[str, float]) -> dict[str, bool]:
    return {
        "ce_strict_descent": bool(candidate["ce"] < current["ce"]),
        "d_seg_nonworsening": bool(candidate["d_seg"] <= current["d_seg"]),
        "d_pose_nonworsening": bool(candidate["d_pose"] <= current["d_pose"]),
    }


def guard_passes(predicates: dict[str, bool]) -> bool:
    return bool(predicates) and all(predicates.values())


def candidate_at_norm(theta: Any, gradient: Any, step_norm: float) -> Any | None:
    import torch

    gradient_norm = float(torch.linalg.vector_norm(gradient.detach()).item())
    if not math.isfinite(gradient_norm) or gradient_norm <= 0.0:
        return None
    candidate = theta.detach() - gradient.detach() * (float(step_norm) / gradient_norm)
    return None if torch.equal(candidate, theta.detach()) else candidate


def exact_call_amortization(
    *, calibration_states: int, accepted_reuses: int
) -> dict[str, float | int | None]:
    """Count exact costate calls for two-step K2 cycles with exact fallback."""

    if calibration_states < 0 or accepted_reuses < 0 or accepted_reuses > calibration_states:
        raise ValueError("invalid K2 exact-call accounting")
    if calibration_states == 0:
        return {
            "calibration_two_step_opportunities": 0,
            "accepted_reuses": 0,
            "fallback_refreshes": 0,
            "baseline_exact_costate_calls": 0,
            "guarded_k2_exact_costate_calls": 0,
            "exact_costate_calls_saved": 0,
            "exact_call_amortization_x": None,
            "backward_call_reduction_fraction": None,
        }
    fallbacks = calibration_states - accepted_reuses
    baseline = 2 * calibration_states
    guarded = calibration_states + fallbacks
    return {
        "calibration_two_step_opportunities": calibration_states,
        "accepted_reuses": accepted_reuses,
        "fallback_refreshes": fallbacks,
        "baseline_exact_costate_calls": baseline,
        "guarded_k2_exact_costate_calls": guarded,
        "exact_costate_calls_saved": baseline - guarded,
        "exact_call_amortization_x": baseline / guarded,
        "backward_call_reduction_fraction": (baseline - guarded) / baseline,
    }


def diagnostic_admission_threshold(forward_share: float) -> dict[str, float]:
    """Derive the strict reuse-rate gate from the same diagnostic cost split.

    With acceptance fraction ``p``, fallback fraction ``1-p``, and teacher
    forward share ``a``, guarded K=2 costs ``2 - p(1-a)`` per two baseline
    calls.  Requiring its speedup to beat the forward-elimination ceiling
    ``1/(1-a)`` gives ``p > 2a/(1-a)``.
    """

    if not 0.0 <= forward_share < 1.0:
        raise ValueError("forward_share must be in [0,1)")
    denominator = 1.0 - forward_share
    return {
        "forward_share_alpha": forward_share,
        "forward_elimination_amdahl_ceiling_x": 1.0 / denominator,
        "required_accept_fraction_strict_gt": 2.0 * forward_share / denominator,
    }


def _render_camera_pair(renderer: Any, theta: Any, pair_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Render the two official camera frames, changing only pair frame 1."""

    import torch

    from tac.local_acceleration import torch_levelset_inflate as tli
    from tac.local_acceleration.torch_levelset_inflate import _torch_act

    renderer.code[2 * pair_index + 1] = theta.detach()
    features_np = (
        renderer._self_orient_native(pair_index) if renderer.m["self_orient"] else renderer.curv_n
    )
    features = torch.as_tensor(features_np, dtype=torch.float32)
    model, parameters = renderer.m, renderer.P
    hidden0 = tli.torch_in_proj_h0(parameters, features, model)
    activation = (
        model["activation"],
        model["wire_w0"],
        model["wire_s0"],
        model["hosc_beta"],
        model["hosc_omega"],
    )
    frames: list[np.ndarray] = []
    for frame_index in (2 * pair_index, 2 * pair_index + 1):
        code_row = theta if frame_index == 2 * pair_index + 1 else renderer.code[frame_index]
        film = (code_row @ parameters["film.weight"].T + parameters["film.bias"]).reshape(
            renderer.nH, 2, renderer.hd
        )
        hidden = hidden0
        for layer in range(renderer.nH):
            hidden = _torch_act(
                (hidden @ parameters[f"hidden.{layer}.weight"].T + parameters[f"hidden.{layer}.bias"])
                * (1.0 + film[layer, 0])
                + film[layer, 1],
                *activation,
            )
        phi = hidden @ parameters["out_sdf.weight"].T + parameters["out_sdf.bias"]
        texture = hidden @ parameters["out_tex.weight"].T + parameters["out_tex.bias"]
        weights = torch.softmax(phi / float(model["softmax_temp"]), dim=-1)
        rgb = torch.sigmoid(weights @ parameters["palette"] + texture) * 255.0
        if not model["chroma"]:
            luma = 0.299 * rgb[:, :1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
            rgb = torch.cat((luma, luma, luma), dim=-1)
        frames.append(tli.torch_R(rgb, model["render_h"], model["render_w"], 874, 1164))
    return frames[0], frames[1]


def _metric_state(
    *, round2: Any, trainer: Any, renderer: Any, theta: Any, pair_index: int,
    segnet: Any, posenet: Any, labels_t: Any, labels_np: np.ndarray, pose: np.ndarray,
) -> dict[str, float]:
    import torch
    import torch.nn.functional as functional

    with torch.no_grad():
        frame = round2._render_chart_for_pair(renderer, theta, pair_index)
        logits = segnet(frame.permute(0, 3, 1, 2).contiguous())
        ce = float(functional.cross_entropy(logits, labels_t).item())
        d_seg = float((logits.argmax(dim=1).cpu().numpy()[0] != labels_np).mean())
        f0, f1 = _render_camera_pair(renderer, theta, pair_index)
        d_pose = float(trainer.cpu_verdict_d_pose(posenet, f0, f1, pose))
    return {"ce": ce, "d_seg": d_seg, "d_pose": d_pose}


def _exact_state(
    *, round2: Any, trainer: Any, renderer: Any, theta: Any, pair_index: int,
    segnet: Any, posenet: Any, labels_t: Any, labels_np: np.ndarray, pose: np.ndarray,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    theta_live = theta.detach().clone().requires_grad_(True)
    frame = round2._render_chart_for_pair(renderer, theta_live, pair_index)
    frame_nchw = frame.permute(0, 3, 1, 2).contiguous()
    teacher_started = time.perf_counter()
    logits = segnet(frame_nchw)
    forward_seconds = time.perf_counter() - teacher_started
    loss = functional.cross_entropy(logits, labels_t)
    backward_started = time.perf_counter()
    costate = torch.autograd.grad(loss, frame_nchw, retain_graph=True)[0].detach()
    backward_seconds = time.perf_counter() - backward_started
    renderer_started = time.perf_counter()
    gradient = torch.autograd.grad(
        (frame_nchw * costate).sum(), theta_live
    )[0].detach()
    renderer_vjp_seconds = time.perf_counter() - renderer_started
    if not bool(torch.isfinite(costate).all() and torch.isfinite(gradient).all()):
        raise ProbeError("exact teacher or renderer VJP produced a nonfinite tensor")
    d_seg = float((logits.detach().argmax(dim=1).cpu().numpy()[0] != labels_np).mean())
    f0, f1 = _render_camera_pair(renderer, theta_live.detach(), pair_index)
    d_pose = float(trainer.cpu_verdict_d_pose(posenet, f0, f1, pose))
    return {
        "theta": theta_live.detach(),
        "frame": frame_nchw.detach(),
        "costate": costate,
        "gradient": gradient,
        "metrics": {"ce": float(loss.detach().item()), "d_seg": d_seg, "d_pose": d_pose},
        "timing_seconds": {
            "teacher_forward": forward_seconds,
            "teacher_backward": backward_seconds,
            "renderer_vjp": renderer_vjp_seconds,
        },
    }


def _stale_renderer_gradient(
    *, round2: Any, renderer: Any, theta: Any, pair_index: int, anchor_costate: Any,
) -> tuple[Any, float, str]:
    import torch

    theta_live = theta.detach().clone().requires_grad_(True)
    frame = round2._render_chart_for_pair(renderer, theta_live, pair_index)
    frame_nchw = frame.permute(0, 3, 1, 2).contiguous()
    if tuple(frame_nchw.shape) != tuple(anchor_costate.shape):
        raise ProbeError("stale anchor costate does not match current frame geometry")
    started = time.perf_counter()
    gradient = torch.autograd.grad((frame_nchw * anchor_costate.detach()).sum(), theta_live)[0].detach()
    elapsed = time.perf_counter() - started
    if not bool(torch.isfinite(gradient).all()):
        raise ProbeError("stale renderer VJP produced a nonfinite gradient")
    return gradient, elapsed, _array_sha256(frame_nchw.detach().cpu().numpy())


def _select_exact_candidate(
    *, round2: Any, trainer: Any, renderer: Any, theta: Any, gradient: Any,
    pair_index: int, segnet: Any, posenet: Any, labels_t: Any,
    labels_np: np.ndarray, pose: np.ndarray, current: dict[str, float],
) -> tuple[Any | None, float | None, list[dict[str, Any]]]:
    import torch

    theta_norm = max(float(torch.linalg.vector_norm(theta.detach()).item()), float(torch.finfo(theta.dtype).eps))
    trials: list[dict[str, Any]] = []
    for halving in range(MAX_HALVINGS + 1):
        fraction = STEP_FRACTION * (0.5 ** halving)
        step_norm = fraction * theta_norm
        candidate = candidate_at_norm(theta, gradient, step_norm)
        if candidate is None:
            trials.append({
                "halving": halving,
                "fraction": fraction,
                "step_norm": step_norm,
                "accepted": False,
                "reason": "zero_gradient_or_bit_identical_completion",
            })
            return None, None, trials
        metrics = _metric_state(
            round2=round2, trainer=trainer, renderer=renderer, theta=candidate,
            pair_index=pair_index, segnet=segnet, posenet=posenet,
            labels_t=labels_t, labels_np=labels_np, pose=pose,
        )
        predicates = joint_guard(current, metrics)
        accepted = guard_passes(predicates)
        trials.append({
            "halving": halving,
            "fraction": fraction,
            "step_norm": step_norm,
            "metrics": metrics,
            "predicates": predicates,
            "accepted": accepted,
        })
        if accepted:
            return candidate, step_norm, trials
    return None, None, trials


def _measure_pair(
    *, assignment: Any, renderer: Any, round2: Any, trainer: Any, segnet: Any,
    posenet: Any, labels: Any, poses: Any, run_contract_sha256: str,
) -> dict[str, Any]:
    import torch

    pair_index = int(assignment.pair_index)
    labels_np = np.array(labels[pair_index], dtype=np.int64, copy=True)
    labels_t = torch.as_tensor(labels_np[None], dtype=torch.long)
    pose = np.array(poses[pair_index], dtype=np.float64, copy=True)
    theta0 = renderer.code[2 * pair_index + 1].detach().clone()
    state0 = _exact_state(
        round2=round2, trainer=trainer, renderer=renderer, theta=theta0,
        pair_index=pair_index, segnet=segnet, posenet=posenet,
        labels_t=labels_t, labels_np=labels_np, pose=pose,
    )
    theta1, step0_norm, step0_trials = _select_exact_candidate(
        round2=round2, trainer=trainer, renderer=renderer, theta=theta0,
        gradient=state0["gradient"], pair_index=pair_index, segnet=segnet,
        posenet=posenet, labels_t=labels_t, labels_np=labels_np, pose=pose,
        current=state0["metrics"],
    )
    base = {
        "schema": PAIR_SCHEMA,
        "run_contract_sha256": run_contract_sha256,
        "completed_at_utc": _utc(),
        "assignment": assignment.to_dict(),
        "axis": AXIS,
        "seed": SEED,
        "K_max": K_MAX,
        "label_sha256": _array_sha256(labels_np),
        "pose_sha256": _array_sha256(pose),
        "theta0_sha256": _array_sha256(theta0.cpu().numpy()),
        "anchor_frame_sha256": _array_sha256(state0["frame"].cpu().numpy()),
        "anchor_costate_sha256": _array_sha256(state0["costate"].cpu().numpy()),
        "anchor_metrics": state0["metrics"],
        "anchor_timing_seconds": state0["timing_seconds"],
        "anchor_step_norm": step0_norm,
        "anchor_line_search": step0_trials,
        "exact_costate_shadow_calls": 1,
    }
    if theta1 is None:
        renderer.code[2 * pair_index + 1] = theta0
        base.update({
            "status": "TERMINAL_OR_BLOCKED_AT_ANCHOR",
            "eligible_for_k2": False,
            "reuse_guard_accept": False,
            "verdict_scope": "real state had no full-facet exact anchor step under the preregistered halving law",
        })
        return base

    state1 = _exact_state(
        round2=round2, trainer=trainer, renderer=renderer, theta=theta1,
        pair_index=pair_index, segnet=segnet, posenet=posenet,
        labels_t=labels_t, labels_np=labels_np, pose=pose,
    )
    stale_gradient, stale_vjp_seconds, stale_frame_sha = _stale_renderer_gradient(
        round2=round2, renderer=renderer, theta=theta1, pair_index=pair_index,
        anchor_costate=state0["costate"],
    )
    if stale_frame_sha != _array_sha256(state1["frame"].cpu().numpy()):
        raise ProbeError(f"pair {pair_index} repeated current frame drifted")
    theta2_exact, step1_norm, step1_trials = _select_exact_candidate(
        round2=round2, trainer=trainer, renderer=renderer, theta=theta1,
        gradient=state1["gradient"], pair_index=pair_index, segnet=segnet,
        posenet=posenet, labels_t=labels_t, labels_np=labels_np, pose=pose,
        current=state1["metrics"],
    )
    base["exact_costate_shadow_calls"] = 2
    base.update({
        "theta1_sha256": _array_sha256(theta1.cpu().numpy()),
        "current_frame_sha256": _array_sha256(state1["frame"].cpu().numpy()),
        "current_costate_sha256": _array_sha256(state1["costate"].cpu().numpy()),
        "current_metrics": state1["metrics"],
        "current_timing_seconds": state1["timing_seconds"],
        "costate_fidelity": vector_metrics(
            state1["costate"].cpu().numpy(), state0["costate"].cpu().numpy()
        ),
        "renderer_gradient_fidelity": vector_metrics(
            state1["gradient"].cpu().numpy(), stale_gradient.cpu().numpy()
        ),
        "stale_renderer_vjp_seconds": stale_vjp_seconds,
        "exact_second_step_norm": step1_norm,
        "exact_second_line_search": step1_trials,
    })
    if theta2_exact is None or step1_norm is None:
        renderer.code[2 * pair_index + 1] = theta0
        base.update({
            "status": "TERMINAL_OR_BLOCKED_AT_REUSE_POINT",
            "eligible_for_k2": False,
            "reuse_guard_accept": False,
            "verdict_scope": "real state reached no full-facet exact second step under the matched schedule",
        })
        return base

    theta2_stale = candidate_at_norm(theta1, stale_gradient, step1_norm)
    if theta2_stale is None:
        renderer.code[2 * pair_index + 1] = theta0
        base.update({
            "status": "STALE_ZERO_OR_BIT_IDENTICAL",
            "eligible_for_k2": True,
            "reuse_guard_accept": False,
            "reuse_guard": {
                "ce_strict_descent": False,
                "d_seg_nonworsening": False,
                "d_pose_nonworsening": False,
            },
            "verdict_scope": "raw input-costate ZOH at Kmax=2 on the registered real-state replay",
        })
        return base

    exact_metrics = _metric_state(
        round2=round2, trainer=trainer, renderer=renderer, theta=theta2_exact,
        pair_index=pair_index, segnet=segnet, posenet=posenet,
        labels_t=labels_t, labels_np=labels_np, pose=pose,
    )
    stale_metrics = _metric_state(
        round2=round2, trainer=trainer, renderer=renderer, theta=theta2_stale,
        pair_index=pair_index, segnet=segnet, posenet=posenet,
        labels_t=labels_t, labels_np=labels_np, pose=pose,
    )
    predicates = joint_guard(state1["metrics"], stale_metrics)
    accepted = guard_passes(predicates)
    base.update({
        "status": "REUSE_GUARD_ACCEPT" if accepted else "REUSE_GUARD_FALLBACK",
        "eligible_for_k2": True,
        "reuse_guard_accept": accepted,
        "reuse_guard": predicates,
        "exact_second_metrics": exact_metrics,
        "stale_second_metrics": stale_metrics,
        "stale_minus_exact_regret": {
            key: float(stale_metrics[key] - exact_metrics[key])
            for key in ("ce", "d_seg", "d_pose")
        },
        "theta2_exact_sha256": _array_sha256(theta2_exact.cpu().numpy()),
        "theta2_stale_sha256": _array_sha256(theta2_stale.cpu().numpy()),
        "verdict_scope": "guarded raw input-costate ZOH Kmax=2 on one n600 stratified real-state replay",
    })
    renderer.code[2 * pair_index + 1] = theta0
    return base


def _validate_pair_record(
    row: dict[str, Any], *, assignment: Any, run_contract_sha256: str
) -> None:
    pair_index = int(assignment.pair_index)
    if row.get("schema") != PAIR_SCHEMA:
        raise ProbeError(f"pair record schema drift at {pair_index}")
    if row.get("assignment") != assignment.to_dict():
        raise ProbeError(f"pair record assignment drift at {pair_index}")
    if row.get("run_contract_sha256") != run_contract_sha256:
        raise ProbeError(f"pair record run contract drift at {pair_index}")
    claimed_content_sha256 = row.get("record_content_sha256")
    unsigned = {key: value for key, value in row.items() if key != "record_content_sha256"}
    if claimed_content_sha256 != _canonical_sha256(unsigned):
        raise ProbeError(f"pair record content custody drift at {pair_index}")


def _record_custody(output_dir: Path, assignment: Any, run_contract_sha256: str) -> dict[str, Any]:
    pair_index = int(assignment.pair_index)
    path = _pair_path(output_dir, pair_index)
    if not path.is_file():
        raise ProbeError(f"stage record is missing pair {pair_index}")
    raw = path.read_bytes()
    row = json.loads(raw)
    _validate_pair_record(row, assignment=assignment, run_contract_sha256=run_contract_sha256)
    return {
        "pair_index": pair_index,
        "path": str(path.relative_to(output_dir)),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _verify_stage_manifest(
    output_dir: Path,
    checkpoint_name: str,
    rows: Sequence[Any],
    run_contract_sha256: str,
) -> dict[str, Any]:
    path = output_dir / f"stage_{checkpoint_name}_complete.json"
    if not path.is_file():
        raise ProbeError(f"stage {checkpoint_name} manifest is missing")
    payload = json.loads(path.read_text())
    records = [_record_custody(output_dir, row, run_contract_sha256) for row in rows]
    expected = {
        "schema": STAGE_SCHEMA,
        "run_contract_sha256": run_contract_sha256,
        "checkpoint_name": checkpoint_name,
        "state_count": len(records),
        "records": records,
        "tree_sha256": _canonical_sha256(records),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ProbeError(f"stage {checkpoint_name} manifest custody drift at {key}")
    return payload


def _stage_manifest(
    output_dir: Path,
    checkpoint_name: str,
    rows: Sequence[Any],
    run_contract_sha256: str,
) -> dict[str, Any]:
    manifest_path = output_dir / f"stage_{checkpoint_name}_complete.json"
    if manifest_path.is_file():
        return _verify_stage_manifest(output_dir, checkpoint_name, rows, run_contract_sha256)
    records: list[dict[str, Any]] = []
    for assignment in rows:
        records.append(_record_custody(output_dir, assignment, run_contract_sha256))
    payload = {
        "schema": STAGE_SCHEMA,
        "completed_at_utc": _utc(),
        "run_contract_sha256": run_contract_sha256,
        "checkpoint_name": checkpoint_name,
        "state_count": len(records),
        "records": records,
        "tree_sha256": _canonical_sha256(records),
    }
    _atomic_json(manifest_path, payload)
    return payload


def _load_records(
    output_dir: Path, assignments: Sequence[Any], run_contract_sha256: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for assignment in assignments:
        path = _pair_path(output_dir, int(assignment.pair_index))
        if not path.is_file():
            raise ProbeError(f"final aggregate is missing pair {assignment.pair_index}")
        row = json.loads(path.read_text())
        _validate_pair_record(
            row, assignment=assignment, run_contract_sha256=run_contract_sha256
        )
        rows.append(row)
    return rows


def aggregate_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in records if row.get("eligible_for_k2") is True]
    accepted = [
        row
        for row in eligible
        if row.get("reuse_guard_accept") is True
        and guard_passes(row.get("reuse_guard") or {})
    ]
    inconsistent_accept_count = sum(
        row.get("reuse_guard_accept") is True
        and not guard_passes(row.get("reuse_guard") or {})
        for row in eligible
    )
    calls = exact_call_amortization(
        calibration_states=len(records), accepted_reuses=len(accepted)
    )
    status_counts = Counter(str(row.get("status")) for row in records)
    failure_counts: Counter[str] = Counter()
    for row in eligible:
        if row.get("reuse_guard_accept") is True:
            continue
        for key, passed in (row.get("reuse_guard") or {}).items():
            if not passed:
                failure_counts[key] += 1
    grad_rows = [row["renderer_gradient_fidelity"] for row in records if "renderer_gradient_fidelity" in row]
    costate_rows = [row["costate_fidelity"] for row in records if "costate_fidelity" in row]
    regret_rows = [row["stale_minus_exact_regret"] for row in eligible if "stale_minus_exact_regret" in row]
    accepted_regret_rows = [
        row["stale_minus_exact_regret"]
        for row in accepted
        if "stale_minus_exact_regret" in row
    ]
    accepted_grad_rows = [
        row["renderer_gradient_fidelity"]
        for row in accepted
        if "renderer_gradient_fidelity" in row
    ]
    accepted_d_seg_regrets = [
        float(row["stale_minus_exact_regret"]["d_seg"])
        for row in accepted
        if "stale_minus_exact_regret" in row
        and "d_seg" in row["stale_minus_exact_regret"]
    ]
    fallback_rate = (len(records) - len(accepted)) / len(records) if records else None
    diagnostic_speedup = (
        2.0
        / (
            1.0
            + DIAGNOSTIC_FORWARD_SHARE
            + fallback_rate * (1.0 - DIAGNOSTIC_FORWARD_SHARE)
        )
        if fallback_rate is not None else None
    )
    return {
        "state_count": len(records),
        "unique_pair_count": len({int(row["assignment"]["pair_index"]) for row in records}),
        "checkpoint_counts": dict(Counter(row["assignment"]["checkpoint_name"] for row in records)),
        "status_counts": dict(status_counts),
        "eligible_state_count": len(eligible),
        "terminal_or_blocked_state_count": len(records) - len(eligible),
        "reuse_guard_accept_count": len(accepted),
        "behavioral_full_facet_accept_count": len(accepted),
        "inconsistent_accept_flag_count": inconsistent_accept_count,
        "reuse_guard_fallback_count": len(eligible) - len(accepted),
        "reuse_guard_accept_fraction": len(accepted) / len(eligible) if eligible else None,
        "calibration_fallback_count": len(records) - len(accepted),
        "calibration_accept_fraction": len(accepted) / len(records) if records else None,
        "guard_failure_counts": dict(failure_counts),
        "costate_fidelity": {
            "cosine_fp32": _quantiles([row["cosine_fp32"] for row in costate_rows]),
            "relative_l2_error_fp32": _quantiles([row["relative_l2_error_fp32"] for row in costate_rows]),
        },
        "renderer_gradient_fidelity": {
            "cosine_fp32": _quantiles([row["cosine_fp32"] for row in grad_rows]),
            "relative_l2_error_fp32": _quantiles([row["relative_l2_error_fp32"] for row in grad_rows]),
            "descent_sufficient_relative_l2_lt_one_count": sum(
                row["relative_l2_error_fp32"] < 1.0 for row in grad_rows
            ),
            "accepted_calibration_row_count": len(accepted),
            "accepted_calibration_fidelity_present_count": len(accepted_grad_rows),
            "accepted_calibration_relative_l2_lt_one_count": sum(
                row["relative_l2_error_fp32"] < 1.0 for row in accepted_grad_rows
            ),
            "accepted_calibration_relative_l2_threshold": 1.0,
            "accepted_calibration_relative_l2_comparator": "strict_lt",
        },
        "accepted_d_seg_regret_gate": {
            "accepted_calibration_row_count": len(accepted),
            "accepted_calibration_regret_present_count": len(accepted_d_seg_regrets),
            "accepted_calibration_d_seg_regret_lte_zero_count": sum(
                value <= 0.0 for value in accepted_d_seg_regrets
            ),
            "threshold": 0.0,
            "comparator": "lte",
        },
        "accepted_stale_minus_exact_regret": {
            key: _quantiles([row[key] for row in accepted_regret_rows])
            for key in ("ce", "d_seg", "d_pose")
        },
        "all_eligible_stale_minus_exact_regret": {
            key: _quantiles([row[key] for row in regret_rows])
            for key in ("ce", "d_seg", "d_pose")
        },
        "exact_costate_call_economics": calls,
        "diagnostic_teacher_slice_economics": {
            "forward_share_alpha": DIAGNOSTIC_FORWARD_SHARE,
            "forward_share_provenance": DIAGNOSTIC_FORWARD_SHARE_PROVENANCE,
            "fallback_rate": fallback_rate,
            "fallback_scope": (
                "all non-accepted calibration states, including terminal/blocked states, "
                "are conservatively charged as exact fallback"
            ),
            "conditional_speedup_x": diagnostic_speedup,
            "formula": "2/(1+forward_share+fallback_rate*(1-forward_share))",
            **diagnostic_admission_threshold(DIAGNOSTIC_FORWARD_SHARE),
            "evidence_grade": "DERIVED_DIAGNOSTIC_NOT_IN_LOOP",
            "whole_epoch_speedup": "UNKNOWN_IN_LOOP_TIMER_OWED",
        },
    }


def evaluate_admission_gate(
    aggregate: dict[str, Any], *, complete_n600: bool
) -> dict[str, Any]:
    """Evaluate the preregistered, calibration-only K=2 admission contract."""

    accepted = int(aggregate["behavioral_full_facet_accept_count"])
    eligible = int(aggregate["eligible_state_count"])
    accept_fraction = aggregate["calibration_accept_fraction"]
    economics = aggregate["diagnostic_teacher_slice_economics"]
    required_fraction = float(economics["required_accept_fraction_strict_gt"])
    speedup = economics["conditional_speedup_x"]
    ceiling = float(economics["forward_elimination_amdahl_ceiling_x"])
    gradient = aggregate["renderer_gradient_fidelity"]
    regret = aggregate["accepted_d_seg_regret_gate"]
    observed_state_count = int(aggregate["state_count"])
    observed_unique_pair_count = int(aggregate["unique_pair_count"])
    predicates = {
        "complete_n600": (
            bool(complete_n600)
            and observed_state_count == N_PAIRS
            and observed_unique_pair_count == N_PAIRS
        ),
        "has_behavioral_full_facet_accepts": accepted > 0,
        "accept_fraction_strictly_exceeds_required": (
            accept_fraction is not None and accept_fraction > required_fraction
        ),
        "diagnostic_speedup_strictly_exceeds_amdahl_ceiling": (
            speedup is not None and float(speedup) > ceiling
        ),
        "all_accepted_gradient_fidelity_present": (
            gradient["accepted_calibration_fidelity_present_count"] == accepted
        ),
        "all_accepted_gradient_relative_l2_strict_lt_one": (
            gradient["accepted_calibration_relative_l2_lt_one_count"] == accepted
        ),
        "all_accepted_d_seg_regret_present": (
            regret["accepted_calibration_regret_present_count"] == accepted
        ),
        "all_accepted_stale_d_seg_regret_lte_exact": (
            regret["accepted_calibration_d_seg_regret_lte_zero_count"] == accepted
        ),
        "no_inconsistent_accept_flags": aggregate["inconsistent_accept_flag_count"] == 0,
    }
    return {
        "schema": ADMISSION_SPEC["schema"],
        "spec": ADMISSION_SPEC,
        "spec_sha256": _canonical_sha256(ADMISSION_SPEC),
        "predicates": predicates,
        "passed": all(predicates.values()),
        "behavioral_full_facet_accept_count": accepted,
        "eligible_state_count": eligible,
        "observed_state_count": observed_state_count,
        "observed_unique_pair_count": observed_unique_pair_count,
        "measured_accept_fraction": accept_fraction,
        "required_accept_fraction_strict_gt": required_fraction,
        "diagnostic_teacher_slice_speedup_x": speedup,
        "forward_elimination_amdahl_ceiling_x": ceiling,
        "evidence_grade": "DERIVED_DIAGNOSTIC_NOT_IN_LOOP",
        "runtime_exact_gradient_access": False,
        "whole_epoch_speedup": "UNKNOWN_IN_LOOP_TIMER_OWED",
    }


def build_admission_content(
    *,
    run_contract: dict[str, Any],
    stage_manifest_custody: Sequence[dict[str, Any]],
    aggregate: dict[str, Any],
    admission_gate: dict[str, Any],
    admission_verdict: str,
) -> dict[str, Any]:
    """Bind admission to the exact objective, scorers, records, and aggregate."""

    return {
        "run_contract_sha256": run_contract["sha256"],
        "objective_sha256": run_contract["payload"]["objective_sha256"],
        "scorer_sha256": run_contract["payload"]["scorer_sha256"],
        "admission_spec_sha256": admission_gate["spec_sha256"],
        "stage_manifest_custody": list(stage_manifest_custody),
        "aggregate_sha256": _canonical_sha256(aggregate),
        "admission_verdict": admission_verdict,
    }


def _run_contract(
    *, output_dir: Path, storage_plan: Path, round2: Any, max_pairs: int | None,
) -> dict[str, Any]:
    input_custody: dict[str, Any] = {}
    for relative, expected in round2.EXPECTED_INPUTS.items():
        path = REPO / relative
        if not path.is_file():
            raise ProbeError(f"missing input {relative}")
        actual = {"path": relative, "bytes": path.stat().st_size, "sha256": _sha256(path)}
        if actual["bytes"] != expected["bytes"] or actual["sha256"] != expected["sha256"]:
            raise ProbeError(f"input custody drift for {relative}")
        input_custody[relative] = actual
    posenet = REPO / "upstream/models/posenet.safetensors"
    input_custody[str(posenet.relative_to(REPO))] = {
        "path": str(posenet.relative_to(REPO)),
        "bytes": posenet.stat().st_size,
        "sha256": _sha256(posenet),
    }
    objective_sha256 = _objective_sha256()
    scorer_sha256 = _scorer_sha256(input_custody)
    payload = {
        "schema": SCHEMA,
        "git_head_at_launch": _git_head(),
        "output_dir": str(output_dir),
        "storage_plan": _file_custody(storage_plan),
        "source_custody": _source_custody(),
        "input_custody": input_custody,
        "objective_spec": OBJECTIVE_SPEC,
        "objective_sha256": objective_sha256,
        "admission_spec": ADMISSION_SPEC,
        "admission_spec_sha256": _canonical_sha256(ADMISSION_SPEC),
        "scorer_sha256": scorer_sha256,
        "constants": {
            "n_pairs": N_PAIRS,
            "seed": SEED,
            "checkpoint_count": len(round2.CHECKPOINTS),
            "holdout_period": HOLDOUT_PERIOD,
            "K_max": K_MAX,
            "step_fraction": STEP_FRACTION,
            "max_halvings": MAX_HALVINGS,
            "diagnostic_forward_share": DIAGNOSTIC_FORWARD_SHARE,
        },
        "constant_provenance": {
            "n_pairs": "operator n600 authority rule",
            "seed": "task-455 lineage",
            "checkpoint_count": "three immutable V9 stages available to the frozen replay",
            "holdout_period": "inherited deterministic frozen-replay split; split is descriptive here",
            "K_max": "operator-directed smallest changed formulation after K>=4 failures",
            "step_fraction": "task-455 exact-branch registered maximum; bounded fractional halving",
            "max_halvings": "binary32 fraction-bit count",
            "diagnostic_forward_share": DIAGNOSTIC_FORWARD_SHARE_PROVENANCE,
        },
        "max_pairs": max_pairs,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
    }
    semantic_payload = {
        key: value for key, value in payload.items() if key != "git_head_at_launch"
    }
    return {
        "sha256": _canonical_sha256(semantic_payload),
        "payload": payload,
        "launch_provenance_sha256": _canonical_sha256(payload),
    }


def _semantic_contract_payload(contract: dict[str, Any]) -> dict[str, Any]:
    """Return crash-resume semantics while retaining launch HEAD as provenance."""

    payload = contract.get("payload")
    if not isinstance(payload, dict):
        raise ProbeError("run contract payload is missing")
    return {key: value for key, value in payload.items() if key != "git_head_at_launch"}


def _validate_resume_contract(
    prior: dict[str, Any], current: dict[str, Any]
) -> dict[str, Any]:
    """Adopt the launch contract when every semantic byte identity still matches."""

    prior_semantic = _semantic_contract_payload(prior)
    current_semantic = _semantic_contract_payload(current)
    if prior.get("sha256") != _canonical_sha256(prior_semantic):
        raise ProbeError("stored resume run contract self-hash changed")
    if current.get("sha256") != _canonical_sha256(current_semantic):
        raise ProbeError("current resume run contract self-hash changed")
    if prior_semantic != current_semantic or prior.get("sha256") != current.get("sha256"):
        raise ProbeError("resume run contract changed")
    launch_payload = prior.get("payload")
    if prior.get("launch_provenance_sha256") != _canonical_sha256(launch_payload):
        raise ProbeError("stored launch provenance hash changed")
    return prior


def _load_completed_receipt(
    output_dir: Path,
    *,
    run_contract: dict[str, Any],
    assignments: Sequence[Any],
    checkpoint_names: Sequence[str],
    complete_n600: bool,
) -> dict[str, Any] | None:
    """Re-derive a completed run from sealed rows before returning its receipt."""

    complete_path = output_dir / "complete.json"
    if not complete_path.is_file():
        return None
    try:
        complete = json.loads(complete_path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbeError("completed-run seal is unreadable") from exc
    if not isinstance(complete, dict) or complete.get("schema") != (
        "p0_costate_reuse_k2_complete.v2"
    ):
        raise ProbeError("completed-run seal schema changed")
    if complete.get("receipt") != "measurement_receipt.json":
        raise ProbeError("completed-run receipt path changed")
    receipt_path = output_dir / "measurement_receipt.json"
    if not receipt_path.is_file():
        raise ProbeError("completed-run receipt bytes are unavailable")
    raw = receipt_path.read_bytes()
    if complete.get("receipt_bytes") != len(raw) or complete.get("receipt_sha256") != (
        hashlib.sha256(raw).hexdigest()
    ):
        raise ProbeError("completed-run receipt custody changed")
    try:
        receipt = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbeError("completed-run receipt is not valid JSON") from exc
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema") != SCHEMA
        or receipt.get("status") != "completed"
        or receipt.get("n_pairs") != len(assignments)
    ):
        raise ProbeError("completed-run receipt status changed")
    if receipt.get("run_contract") != run_contract:
        raise ProbeError("completed-run receipt contract changed")
    if receipt.get("objective_sha256") != run_contract["payload"]["objective_sha256"]:
        raise ProbeError("completed-run objective custody changed")
    if receipt.get("scorer_sha256") != run_contract["payload"]["scorer_sha256"]:
        raise ProbeError("completed-run scorer custody changed")

    stage_manifests: list[dict[str, Any]] = []
    for checkpoint_name in checkpoint_names:
        cohort = [
            assignment
            for assignment in assignments
            if assignment.checkpoint_name == checkpoint_name
        ]
        if cohort:
            stage_manifests.append(
                _verify_stage_manifest(
                    output_dir, checkpoint_name, cohort, run_contract["sha256"]
                )
            )
    expected_stage_custody = [
        {
            "checkpoint_name": manifest["checkpoint_name"],
            "run_contract_sha256": manifest["run_contract_sha256"],
            "state_count": manifest["state_count"],
            "tree_sha256": manifest["tree_sha256"],
            **_file_custody(
                output_dir / f"stage_{manifest['checkpoint_name']}_complete.json"
            ),
        }
        for manifest in stage_manifests
    ]
    if receipt.get("stage_manifest_custody") != expected_stage_custody:
        raise ProbeError("completed-run stage-manifest custody changed")
    records = _load_records(output_dir, assignments, run_contract["sha256"])
    if len(records) != len(assignments):
        raise ProbeError("completed-run pair-record count changed")
    aggregate = aggregate_records(records)
    if receipt.get("measurement") != aggregate:
        raise ProbeError("completed-run aggregate changed")
    gate = evaluate_admission_gate(aggregate, complete_n600=complete_n600)
    verdict = "ADMIT_K2_GUARDED_REUSE" if gate["passed"] else "NOT_ADMITTED"
    fidelity_gate = receipt.get("fidelity_gate")
    if (
        not isinstance(fidelity_gate, dict)
        or fidelity_gate.get("calibration_admission_gate") != gate
        or fidelity_gate.get("admission") != verdict
        or fidelity_gate.get("complete_n600") is not complete_n600
        or fidelity_gate.get("live_trainer_activation") is not False
        or fidelity_gate.get("runtime_exact_gradient_access") is not False
    ):
        raise ProbeError("completed-run fidelity gate changed")
    if receipt.get("admission_verdict") != verdict:
        raise ProbeError("completed-run admission verdict changed")
    expected_admission_content = build_admission_content(
        run_contract=run_contract,
        stage_manifest_custody=expected_stage_custody,
        aggregate=aggregate,
        admission_gate=gate,
        admission_verdict=verdict,
    )
    if (
        receipt.get("admission_content") != expected_admission_content
        or receipt.get("admission_content_sha256")
        != _canonical_sha256(expected_admission_content)
    ):
        raise ProbeError("completed-run admission content changed")
    authority = receipt.get("authority")
    if not isinstance(authority, dict) or any(
        authority.get(field) is not False
        for field in ("score_claim", "promotion_eligible", "pointer_moved")
    ):
        raise ProbeError("completed-run receipt carries false authority")
    return receipt


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    output_dir.relative_to((REPO / "experiments/results").resolve())
    storage = _validate_storage_plan(args.storage_plan.resolve(), output_dir)
    descriptor = _acquire_lock(output_dir)
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.manual_seed(SEED)
        torch.use_deterministic_algorithms(True)

        round2 = _load_module("_p0_k2_round2", "tools/probe_frozen_replay_convex_head.py")
        old_probe = _load_module("_p0_k2_old_probe", "tools/probe_onpolicy_scorer_surrogate.py")
        yopo = _load_module("_p0_k2_yopo", "tools/probe_yopo_first_layer_costate.py")
        trainer = _load_module("_p0_k2_trainer", "experiments/train_witness_realized_through_R_mlx.py")
        contract = _run_contract(
            output_dir=output_dir,
            storage_plan=args.storage_plan.resolve(),
            round2=round2,
            max_pairs=args.max_pairs,
        )
        contract_path = output_dir / "run_contract.json"
        if contract_path.is_file():
            prior = json.loads(contract_path.read_text())
            if not args.resume:
                raise ProbeError("output already exists; pass --resume")
            contract = _validate_resume_contract(prior, contract)
        else:
            _atomic_json(contract_path, contract)

        checkpoint_names = tuple(row[0] for row in round2.CHECKPOINTS)
        assignments = round2.deterministic_replay_assignments(
            n_pairs=N_PAIRS,
            checkpoint_names=checkpoint_names,
            holdout_period=HOLDOUT_PERIOD,
            seed=SEED,
        )
        if args.max_pairs is not None:
            assignments = assignments[: args.max_pairs]
        completed_receipt = _load_completed_receipt(
            output_dir,
            run_contract=contract,
            assignments=assignments,
            checkpoint_names=checkpoint_names,
            complete_n600=(len(assignments) == N_PAIRS and args.max_pairs is None),
        )
        if completed_receipt is not None:
            return completed_receipt
        run_contract_sha256 = contract["sha256"]
        segnet, posenet = old_probe._load_scorers()
        with np.load(round2.GT_CACHE, allow_pickle=False) as cache:
            labels = np.array(cache["lstars"][:N_PAIRS], dtype=np.int64, copy=True)
            poses = np.array(cache["gt_poses"][:N_PAIRS], dtype=np.float64, copy=True)
        for checkpoint_index, (checkpoint_name, checkpoint_path, _epoch) in enumerate(round2.CHECKPOINTS):
            cohort = [row for row in assignments if row.checkpoint_index == checkpoint_index]
            if not cohort:
                continue
            manifest_path = output_dir / f"stage_{checkpoint_name}_complete.json"
            if manifest_path.is_file():
                _verify_stage_manifest(
                    output_dir, checkpoint_name, cohort, run_contract_sha256
                )
                continue
            renderer, code, model, _dash = yopo._load_renderer(checkpoint_path)
            if model["n_pairs"] != N_PAIRS or code.shape[0] != 2 * N_PAIRS:
                raise ProbeError(f"checkpoint {checkpoint_name} is not n600")
            for assignment in cohort:
                path = _pair_path(output_dir, int(assignment.pair_index))
                if path.is_file():
                    prior = json.loads(path.read_text())
                    _validate_pair_record(
                        prior,
                        assignment=assignment,
                        run_contract_sha256=run_contract_sha256,
                    )
                    continue
                row = _measure_pair(
                    assignment=assignment,
                    renderer=renderer,
                    round2=round2,
                    trainer=trainer,
                    segnet=segnet,
                    posenet=posenet,
                    labels=labels,
                    poses=poses,
                    run_contract_sha256=run_contract_sha256,
                )
                row["record_content_sha256"] = _canonical_sha256(row)
                _atomic_json(path, row)
                print(json.dumps({
                    "stage": "pair_complete",
                    "pair_index": int(assignment.pair_index),
                    "checkpoint": checkpoint_name,
                    "status": row["status"],
                }, sort_keys=True), flush=True)
            _stage_manifest(
                output_dir, checkpoint_name, cohort, run_contract_sha256
            )

        stage_manifests: list[dict[str, Any]] = []
        for checkpoint_index, (checkpoint_name, _checkpoint_path, _epoch) in enumerate(
            round2.CHECKPOINTS
        ):
            cohort = [row for row in assignments if row.checkpoint_index == checkpoint_index]
            if cohort:
                stage_manifests.append(
                    _verify_stage_manifest(
                        output_dir, checkpoint_name, cohort, run_contract_sha256
                    )
                )
        records = _load_records(output_dir, assignments, run_contract_sha256)
        aggregate = aggregate_records(records)
        complete_n600 = len(records) == N_PAIRS and args.max_pairs is None
        admission_gate = evaluate_admission_gate(aggregate, complete_n600=complete_n600)
        admission_verdict = (
            "ADMIT_K2_GUARDED_REUSE" if admission_gate["passed"] else "NOT_ADMITTED"
        )
        stage_manifest_custody = [
            {
                "checkpoint_name": manifest["checkpoint_name"],
                "run_contract_sha256": manifest["run_contract_sha256"],
                "state_count": manifest["state_count"],
                "tree_sha256": manifest["tree_sha256"],
                **_file_custody(
                    output_dir / f"stage_{manifest['checkpoint_name']}_complete.json"
                ),
            }
            for manifest in stage_manifests
        ]
        admission_content = build_admission_content(
            run_contract=contract,
            stage_manifest_custody=stage_manifest_custody,
            aggregate=aggregate,
            admission_gate=admission_gate,
            admission_verdict=admission_verdict,
        )
        receipt = {
            "schema": SCHEMA,
            "completed_at_utc": _utc(),
            "axis": AXIS,
            "lane_id": LANE_ID,
            "run_contract": contract,
            "status": "completed",
            "admission_verdict": admission_verdict,
            "n_pairs": len(records),
            "objective_sha256": contract["payload"]["objective_sha256"],
            "scorer_sha256": contract["payload"]["scorer_sha256"],
            "stage_manifest_custody": stage_manifest_custody,
            "admission_content": admission_content,
            "admission_content_sha256": _canonical_sha256(admission_content),
            "storage_preflight": storage,
            "measurement": aggregate,
            "fidelity_gate": {
                "policy": (
                    "per-reuse exact forward CE strict descent and through-R d_seg/d_pose "
                    "nonworsening; failures force exact refresh and byte-exact rollback"
                ),
                "gradient_diagnostic": (
                    "renderer-gradient relative L2 < 1 is a sufficient first-order descent "
                    "certificate, but the exact full-facet guard owns behavioral admission"
                ),
                "complete_n600": complete_n600,
                "calibration_admission_gate": admission_gate,
                "admission": admission_verdict,
                "live_trainer_activation": False,
                "runtime_exact_gradient_access": False,
                "in_loop_timing": "OWED_OPERATOR_GO",
            },
            "cleanup": {
                "status": "NO_BULK_SCRATCH_CREATED",
                "pair_records_are_durable_small_metadata": True,
                "auto_cleanup": "atomic temporary JSON files are replaced in-place; no raw frames/costates persisted",
            },
            "authority": {
                "score_claim": False,
                "promotion_eligible": False,
                "pointer_moved": False,
                "whole_epoch_speedup": "UNKNOWN_IN_LOOP_TIMER_OWED",
                "contest_cpu_cuda": "NOT_MEASURED",
            },
            "host": {
                "platform": platform.platform(),
                "python": sys.version,
                "torch": torch.__version__,
                "numpy": np.__version__,
            },
        }
        receipt_path = output_dir / "measurement_receipt.json"
        _atomic_json(receipt_path, receipt)
        _atomic_json(output_dir / "complete.json", {
            "schema": "p0_costate_reuse_k2_complete.v2",
            "receipt": str(receipt_path.relative_to(output_dir)),
            "receipt_bytes": receipt_path.stat().st_size,
            "receipt_sha256": _sha256(receipt_path),
            "completed_at_utc": _utc(),
        })
        return receipt
    finally:
        _release_lock(descriptor)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--storage-plan", type=Path, default=DEFAULT_STORAGE_PLAN)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--max-pairs", type=int,
        help="bounded non-authority smoke prefix; omit for the sealed n600 replay",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_pairs is not None and not (1 <= args.max_pairs <= N_PAIRS):
        raise ProbeError("--max-pairs must be in [1,600]")
    receipt = run(args)
    print(json.dumps({
        "receipt": str(args.output_dir / "measurement_receipt.json"),
        "admission": receipt["fidelity_gate"]["admission"],
        "eligible": receipt["measurement"]["eligible_state_count"],
        "accepted": receipt["measurement"]["reuse_guard_accept_count"],
        "exact_call_amortization_x": receipt["measurement"]["exact_costate_call_economics"][
            "exact_call_amortization_x"
        ],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
