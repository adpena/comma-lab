#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Local, resumable saved-regime probe for task #455.

The operational non-anchor path runs the nonlinear costate surrogate and the
renderer VJP without calling SegNet.  Exact SegNet/PoseNet calls made after the
candidate is formed are measurement-only controls.  The receipt reports both
counts so validation work cannot be mistaken for deployed economics.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
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
sys.path.insert(0, str(REPO / "src"))

from tac.boundary_math.segnet_gradient_replacement import measure_costate_agreement  # noqa: E402
from tac.ib_lagrangian_aux_scorer import _EMA  # noqa: E402
from tac.scorer_surrogate.onpolicy_costate import (  # noqa: E402
    NonlinearCostateSurrogate,
    OnPolicyTransition,
    ProviderCustody,
    fit_onpolicy_transitions,
    predict_detached_costate,
    whole_step_economics,
)
from tac.witness_dsl.onpolicy_scorer_surrogate_policy import (  # noqa: E402
    OnPolicyScorerSurrogatePolicy,
)

SCHEMA = "onpolicy_scorer_surrogate_probe.v2"
REGIMES = {
    "early": "frozen_ep299_CEend.npz",
    "boundary": "frozen_ep726_MuonStart.npz",
    "late": "frozen_ep925_liveEMA.npz",
}
CHECKPOINT_DIR = REPO / "experiments/results/tau_crossover_trainflow_20260707"
GT_CACHE = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n6.npz"
SEGNET = REPO / "upstream/models/segnet.safetensors"
POSENET = REPO / "upstream/models/posenet.safetensors"
VIDEO = REPO / "upstream/videos/0.mkv"
SOURCE_FILES = (
    "tools/probe_onpolicy_scorer_surrogate.py",
    "src/tac/scorer_surrogate/onpolicy_costate.py",
    "src/tac/witness_dsl/onpolicy_scorer_surrogate_policy.py",
    "tools/probe_yopo_first_layer_costate.py",
    "tools/dash_comb_probe_n600.py",
    "experiments/train_witness_realized_through_R_mlx.py",
    "src/tac/boundary_math/segnet_gradient_replacement.py",
    "src/tac/cuda_levelset_training.py",
    "src/tac/local_acceleration/torch_levelset_inflate.py",
    "upstream/modules.py",
)
VERDICT_SCOPE = (
    "formulation: n=1 pair0, saved early/boundary/late witness regimes, macOS-CPU advisory, "
    "nonlinear 9-channel on-policy input-costate student with EMA shadow, K={1,4,20}, "
    "40 post-bootstrap steps"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())


def _atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    import torch

    buffer = io.BytesIO()
    torch.save(payload, buffer)
    _atomic_bytes(path, buffer.getvalue())


def _import_file(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _source_custody() -> dict[str, dict[str, Any]]:
    return {name: {"sha256": _sha256(REPO / name), "bytes": (REPO / name).stat().st_size} for name in SOURCE_FILES}


def _materialize_source_bundle(output_dir: Path) -> dict[str, dict[str, Any]]:
    """Preserve the exact launch sources so an uncommitted probe is reproducible."""

    bundle: dict[str, dict[str, Any]] = {}
    for name in SOURCE_FILES:
        source = REPO / name
        destination = output_dir / "source_bundle" / name
        _atomic_bytes(destination, source.read_bytes())
        bundle[name] = {
            "path": str(destination.relative_to(output_dir)),
            "sha256": _sha256(destination),
            "bytes": destination.stat().st_size,
        }
    return bundle


def _verify_source_bundle(
    output_dir: Path,
    bundle: dict[str, dict[str, Any]],
    expected_source_custody: dict[str, dict[str, Any]],
) -> None:
    expected_names = set(SOURCE_FILES)
    if set(bundle) != expected_names:
        raise RuntimeError("source bundle does not cover the run-contract source set")
    for name, meta in bundle.items():
        if {"sha256": meta.get("sha256"), "bytes": meta.get("bytes")} != expected_source_custody.get(name):
            raise RuntimeError(f"source bundle does not match run-contract custody for {name}")
        path = output_dir / meta["path"]
        path.resolve().relative_to(output_dir.resolve())
        if not path.is_file() or path.stat().st_size != meta["bytes"] or _sha256(path) != meta["sha256"]:
            raise RuntimeError(f"source bundle custody failed for {name}")


def _git_head() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, text=True, capture_output=True, check=True)
    return result.stdout.strip()


def _load_scorers() -> tuple[Any, Any]:
    from safetensors.torch import load_file

    upstream = str(REPO / "upstream")
    if upstream not in sys.path:
        sys.path.insert(0, upstream)
    from modules import PoseNet, SegNet

    segnet, posenet = SegNet().eval(), PoseNet().eval()
    segnet.load_state_dict(load_file(str(SEGNET), device="cpu"))
    posenet.load_state_dict(load_file(str(POSENET), device="cpu"))
    for model in (segnet, posenet):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return segnet, posenet


def _render_camera_pair(renderer: Any, theta: Any) -> tuple[np.ndarray, np.ndarray]:
    """Render both receiver frames at camera uint8 resolution."""
    import torch

    from tac.local_acceleration import torch_levelset_inflate as tli
    from tac.local_acceleration.torch_levelset_inflate import _torch_act

    renderer.code[1] = theta.detach()
    feats_np = renderer._self_orient_native(0) if renderer.m["self_orient"] else renderer.curv_n
    feats = torch.as_tensor(feats_np, dtype=torch.float32)
    m, p = renderer.m, renderer.P
    h = tli.torch_in_proj_h0(p, feats, m)
    kw = (m["activation"], m["wire_w0"], m["wire_s0"], m["hosc_beta"], m["hosc_omega"])
    frames: list[np.ndarray] = []
    for frame_index in (0, 1):
        code_row = renderer.code[frame_index]
        film = (code_row @ p["film.weight"].T + p["film.bias"]).reshape(renderer.nH, 2, renderer.hd)
        state = h
        for layer in range(renderer.nH):
            state = _torch_act(
                (state @ p[f"hidden.{layer}.weight"].T + p[f"hidden.{layer}.bias"])
                * (1.0 + film[layer, 0])
                + film[layer, 1],
                *kw,
            )
        phi = state @ p["out_sdf.weight"].T + p["out_sdf.bias"]
        texture = state @ p["out_tex.weight"].T + p["out_tex.bias"]
        logits = phi / float(m["softmax_temp"])
        weights = torch.softmax(logits, dim=-1)
        rgb = torch.sigmoid(weights @ p["palette"] + texture) * 255.0
        if not m["chroma"]:
            luma = 0.299 * rgb[:, :1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
            rgb = torch.cat((luma, luma, luma), dim=-1)
        frames.append(tli.torch_R(rgb, m["render_h"], m["render_w"], 874, 1164))
    return frames[0], frames[1]


def _verdict(base: Any, segnet: Any, posenet: Any, renderer: Any, theta: Any, labels: np.ndarray, pose: np.ndarray) -> dict[str, float]:
    f0, f1 = _render_camera_pair(renderer, theta)
    return {
        "d_seg": float(base.cpu_verdict_d_seg(segnet, f1, labels)),
        "d_pose": float(base.cpu_verdict_d_pose(posenet, f0, f1, pose)),
    }


def _reference_step(
    *,
    yopo: Any,
    renderer: Any,
    theta: Any,
    gradient: Any,
    segnet: Any,
    labels_t: Any,
    current_ce: float,
) -> tuple[Any | None, float | None, float | None, list[dict[str, Any]]]:
    """Fractional CE-descent gate with bit-identical completion.

    Hard d_seg and d_pose are sequence-level endpoint gates.  They are still
    measured at every trial, but a one-pixel argmax crossing cannot veto an
    otherwise valid teacher-relaxation trajectory before the sequence exists.
    """
    import torch

    grad_norm = float(torch.linalg.vector_norm(gradient).item())
    theta_norm = max(float(torch.linalg.vector_norm(theta).item()), 1.0)
    if not math.isfinite(grad_norm) or grad_norm == 0.0:
        return None, None, None, [{"accepted": False, "reason": "zero_or_nonfinite_gradient"}]
    fraction = 1.0e-2
    trials: list[dict[str, Any]] = []
    while True:
        step_norm = fraction * theta_norm
        candidate = theta.detach() - step_norm / grad_norm * gradient.detach()
        if torch.equal(candidate, theta.detach()):
            trials.append({"accepted": False, "reason": "bit_identical_completion", "fraction": fraction})
            return None, None, None, trials
        frame = yopo._render_chart(renderer, candidate)
        ce, _ = yopo._evaluate_teacher(segnet, frame, labels_t)
        accepted = ce < current_ce
        trials.append(
            {
                "fraction": fraction,
                "step_norm": step_norm,
                "ce": ce,
                "predicates": {"ce_descent": accepted},
                "accepted": accepted,
            }
        )
        if accepted:
            return candidate, step_norm, fraction, trials
        fraction *= 0.5


def _candidate_at_norm(theta: Any, gradient: Any, step_norm: float) -> Any | None:
    import torch

    norm = float(torch.linalg.vector_norm(gradient).item())
    if not math.isfinite(norm) or norm == 0.0:
        return None
    candidate = theta.detach() - step_norm / norm * gradient.detach()
    return None if torch.equal(candidate, theta.detach()) else candidate


def _input_custody(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "gt_cache": {"path": str(GT_CACHE), "sha256": _sha256(GT_CACHE), "bytes": GT_CACHE.stat().st_size},
        "segnet": {"path": str(SEGNET), "sha256": _sha256(SEGNET), "bytes": SEGNET.stat().st_size},
        "posenet": {"path": str(POSENET), "sha256": _sha256(POSENET), "bytes": POSENET.stat().st_size},
        "video": {"path": str(VIDEO), "sha256": _sha256(VIDEO), "bytes": VIDEO.stat().st_size},
        "checkpoints": {
            regime: {"path": str(CHECKPOINT_DIR / REGIMES[regime]), "sha256": _sha256(CHECKPOINT_DIR / REGIMES[regime])}
            for regime in args.regimes
        },
    }


def _run_contract(args: argparse.Namespace, policy: OnPolicyScorerSurrogatePolicy) -> dict[str, Any]:
    payload = {
        "schema": SCHEMA,
        "seed": args.seed,
        "regimes": list(args.regimes),
        "steps": args.steps,
        "output_dir": str(args.output_dir.resolve()),
        "storage_plan_sha256": _sha256(args.storage_plan),
        "policy": policy.compile_measurement_contract(),
        "inputs": _input_custody(args),
        "source_custody": _source_custody(),
        "objective": "exact_segnet_ce_input_costate_through_R",
        "pair_index": 0,
    }
    return {"sha256": _payload_sha256(payload), "payload": payload}


def _write_checkpoint(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    _atomic_torch(path, payload)
    return {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _load_checkpoint(meta: dict[str, Any], *, output_dir: Path, expected: dict[str, Any]) -> dict[str, Any]:
    import torch

    path = Path(meta["path"])
    path.resolve().relative_to(output_dir.resolve())
    if not path.is_file() or path.stat().st_size != meta["bytes"] or _sha256(path) != meta["sha256"]:
        raise RuntimeError("checkpoint bytes do not match receipt custody")
    state = torch.load(path, map_location="cpu", weights_only=False)
    for key, value in expected.items():
        if state.get(key) != value:
            raise RuntimeError(f"checkpoint {key} does not match run contract")
    return state


def _arm_seed(seed: int, regime: str, cadence: int) -> int:
    return int(hashlib.sha256(f"{seed}:{regime}:K{cadence}".encode()).hexdigest()[:8], 16)


def _steady_economics(cadence: int, steps: list[dict[str, Any]]) -> dict[str, float]:
    exact_times = [
        row["timing_seconds"]["operational_whole_step"]
        for row in steps
        if row["refresh"] and (cadence == 1 or row["step"] > 1)
    ]
    surrogate_times = [
        row["timing_seconds"]["operational_whole_step"] for row in steps if not row["refresh"]
    ]
    if not exact_times:
        raise RuntimeError("steady-state exact anchor timing is missing")
    t_exact = float(np.mean(exact_times))
    t_surrogate = float(np.mean(surrogate_times)) if surrogate_times else t_exact
    return whole_step_economics(
        cadence=cadence,
        t_exact_seconds=t_exact,
        t_surrogate_seconds=t_surrogate,
    )


def _classify_verdict(regime_rows: list[dict[str, Any]], expected_regimes: int) -> tuple[str, str]:
    if len(regime_rows) != expected_regimes:
        return "NEEDS-MORE", "one or more exact-teacher baselines could not complete"
    control_rows = [row["arms"]["K1"] for row in regime_rows if "K1" in row.get("arms", {})]
    target_rows = [row["arms"]["K20"] for row in regime_rows if "K20" in row.get("arms", {})]
    if not control_rows or not all(row["sequence_holds_exact_dseg_dpose_descent"] for row in control_rows):
        return (
            "NEEDS-MORE",
            "the K=1 exact-teacher control did not hold exact sequence-endpoint d_seg/d_pose descent in every regime",
        )
    if target_rows and len(target_rows) == expected_regimes and all(
        row["all_nonrefresh_cycle_validations_hold_teacher_relaxation_descent"]
        and row["sequence_holds_exact_dseg_dpose_descent"]
        for row in target_rows
    ):
        return "GO", "K=20 held exact through-R sequence-endpoint CE/d_seg/d_pose descent in every saved regime"
    return "NO-GO", "K=20 nonlinear surrogate failed exact sequence-endpoint descent after a valid K=1 control"


def _base_receipt(args: argparse.Namespace, policy: OnPolicyScorerSurrogatePolicy) -> dict[str, Any]:
    run_contract = _run_contract(args, policy)
    return {
        "schema": SCHEMA,
        "status": "RUNNING",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "axis": "[macOS-CPU advisory training-gradient]",
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_expected_unmoved": True,
        "verdict_scope": VERDICT_SCOPE,
        "review_status": "recovery-written-UNREVIEWED",
        "argv": sys.argv,
        "cwd": str(Path.cwd()),
        "storage_plan": {
            "path": str(args.storage_plan),
            "sha256": _sha256(args.storage_plan),
            "payload": json.loads(args.storage_plan.read_text()),
        },
        "git_head_at_launch": _git_head(),
        "runtime": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
        "policy": policy.compile_measurement_contract(),
        "inputs": run_contract["payload"]["inputs"],
        "source_custody": _source_custody(),
        "run_contract": run_contract,
        "measurement_controls": {
            "positive": "K=1 exact-provider arm",
            "negative": "sign-reversed exact costate at first bootstrap state",
            "admission": (
                "non-anchor steps use no exact-teacher decision; exact validation is measurement-only; the full "
                "sequence endpoint must decrease exact CE and not worsen exact camera-roundtripped d_seg or d_pose"
            ),
        },
        "regimes": {},
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    permitted = (REPO / "experiments/results").resolve()
    args.output_dir.resolve().relative_to(permitted)
    storage_plan = json.loads(args.storage_plan.read_text())
    if storage_plan.get("blockers") or not storage_plan.get("selected_workload_root"):
        raise RuntimeError("storage waterfall did not select a writable tier")
    if Path(storage_plan["selected_workload_root"]).resolve() != args.output_dir.resolve():
        raise RuntimeError("storage plan workload root does not match output-dir")
    policy = OnPolicyScorerSurrogatePolicy()
    if args.steps != policy.measurement_horizon:
        raise RuntimeError(f"final measurement requires derived horizon {policy.measurement_horizon}")
    receipt_path = args.output_dir / "measurement_receipt.json"
    if receipt_path.exists() and not args.resume:
        raise RuntimeError("output exists; pass --resume or choose a fresh directory")
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        _verify_source_bundle(
            args.output_dir,
            receipt.get("source_bundle", {}),
            receipt.get("source_custody", {}),
        )
    else:
        receipt = _base_receipt(args, policy)
        receipt["source_bundle"] = _materialize_source_bundle(args.output_dir)
    current_contract = _run_contract(args, policy)
    if receipt.get("run_contract") != current_contract:
        raise RuntimeError("source, input, seed, policy, or storage custody changed; refusing resume")
    _atomic_json(receipt_path, receipt)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    yopo = _import_file("_task455_yopo_harness", REPO / "tools/probe_yopo_first_layer_costate.py")
    base = _import_file("_task455_base_trainer", REPO / "experiments/train_witness_realized_through_R_mlx.py")
    segnet, posenet = _load_scorers()
    teacher_calls = {"segnet": 0, "posenet": 0}

    def _count_segnet(_module: Any, _inputs: Any, _output: Any) -> None:
        teacher_calls["segnet"] += 1

    def _count_posenet(_module: Any, _inputs: Any, _output: Any) -> None:
        teacher_calls["posenet"] += 1

    segnet.register_forward_hook(_count_segnet)
    posenet.register_forward_hook(_count_posenet)
    receipt["runtime"]["torch"] = torch.__version__
    with np.load(GT_CACHE, allow_pickle=False) as cache:
        labels_np = np.asarray(cache["lstars"][0], np.int64)
        labels_t = torch.as_tensor(labels_np)[None]
        pose = np.asarray(cache["gt_poses"][0], np.float64)

    for regime in args.regimes:
        if receipt["regimes"].get(regime, {}).get("status") == "MEASURED":
            for cadence in policy.measured_cadences:
                arm = receipt["regimes"][regime]["arms"][f"K{cadence}"]
                _load_checkpoint(
                    arm["stage_checkpoint"],
                    output_dir=args.output_dir,
                    expected={
                        "schema": SCHEMA,
                        "run_contract_sha256": current_contract["sha256"],
                        "regime": regime,
                        "cadence": cadence,
                        "completed_step": args.steps,
                    },
                )
            continue
        renderer, code, _model, _dash = yopo._load_renderer(CHECKPOINT_DIR / REGIMES[regime])
        theta0 = torch.as_tensor(code[1], dtype=torch.float32).requires_grad_(True)
        parity = yopo._renderer_parity_canary(renderer, theta0)
        if parity["max_abs"] != 0.0:
            raise RuntimeError("saved renderer chart parity failed")

        torch.manual_seed(_arm_seed(args.seed, regime, 0))
        np.random.seed(_arm_seed(args.seed, regime, 0))
        random.seed(_arm_seed(args.seed, regime, 0))
        provider_custody = ProviderCustody(
            fingerprint_sha256=_payload_sha256(
                {"run_contract": current_contract["sha256"], "regime": regime, "pair_index": 0}
            ),
            regime=regime,
        )
        # Bootstrap uses only witness-produced states and exact through-R teacher labels.
        frame0 = yopo._render_chart(renderer, theta0)
        cost0, holder0, exact0_s = yopo._capture_exact_teacher_costate(
            segnet=segnet, frame_nchw=frame0.permute(0, 3, 1, 2), labels=labels_t
        )
        grad0 = torch.autograd.grad((frame0.permute(0, 3, 1, 2) * cost0).sum(), theta0)[0]
        verdict0 = _verdict(base, segnet, posenet, renderer, theta0, labels_np, pose)
        theta1, _, bootstrap_fraction, bootstrap_trials = _reference_step(
            yopo=yopo,
            renderer=renderer,
            theta=theta0,
            gradient=grad0,
            segnet=segnet,
            labels_t=labels_t,
            current_ce=holder0["ce"],
        )
        if theta1 is None:
            receipt["regimes"][regime] = {
                "status": "BLOCKED",
                "reason": "exact_teacher_bootstrap_cannot_decrease_teacher_CE_before_bit-identical_completion",
                "current": {"ce": holder0["ce"], **verdict0},
                "bootstrap_trials": bootstrap_trials,
            }
            _atomic_json(receipt_path, receipt)
            continue
        frame1 = yopo._render_chart(renderer, theta1.requires_grad_(True))
        cost1, holder1, exact1_s = yopo._capture_exact_teacher_costate(
            segnet=segnet, frame_nchw=frame1.permute(0, 3, 1, 2), labels=labels_t
        )
        transition = OnPolicyTransition(
            frame0.permute(0, 3, 1, 2).detach(),
            cost0.detach(),
            frame1.permute(0, 3, 1, 2).detach(),
            cost1.detach(),
            1,
            provider_custody,
        )
        model = NonlinearCostateSurrogate(policy.hidden_channels)
        optimizer = torch.optim.Adam(model.parameters(), lr=2.0e-3)
        ema = _EMA(model, decay=policy.ema_decay)
        fit_started = time.perf_counter()
        fit = fit_onpolicy_transitions(
            model,
            [transition],
            optimizer=optimizer,
            steps=policy.fit_steps_per_anchor,
            ema=ema,
        )
        bootstrap_fit_s = time.perf_counter() - fit_started
        if not fit.finite or not fit.improved:
            receipt["regimes"][regime] = {
                "status": "BLOCKED",
                "reason": "bootstrap_onpolicy_fit_did_not_complete_with_finite_improvement",
                "fit": fit.to_dict(),
            }
            _atomic_json(receipt_path, receipt)
            continue
        # Required negative canary: exact sign reversal must not masquerade as the descent direction.
        reverse = _candidate_at_norm(theta0, -grad0, float(bootstrap_trials[-1]["step_norm"]))
        reverse_verdict = _verdict(base, segnet, posenet, renderer, reverse, labels_np, pose) if reverse is not None else None
        reverse_ce = yopo._evaluate_teacher(segnet, yopo._render_chart(renderer, reverse), labels_t)[0] if reverse is not None else None
        negative_pass = bool(
            reverse is not None
            and (reverse_ce >= holder0["ce"] or reverse_verdict["d_seg"] > verdict0["d_seg"] or reverse_verdict["d_pose"] > verdict0["d_pose"])
        )
        verdict0_repeat = _verdict(base, segnet, posenet, renderer, theta0, labels_np, pose)
        deterministic_noise_floor = {
            key: abs(verdict0_repeat[key] - verdict0[key]) for key in ("d_seg", "d_pose")
        }
        bootstrap_state = {
            "model": model.state_dict(),
            "ema_shadow": ema.state_dict(),
            "ema_updates": ema._num_updates,
            "optimizer": optimizer.state_dict(),
            "theta": theta1.detach(),
            "anchor_frame": frame1.permute(0, 3, 1, 2).detach(),
            "anchor_costate": cost1.detach(),
            "anchor_theta": theta1.detach(),
            "anchor_fraction": bootstrap_fraction,
        }
        row = receipt["regimes"].setdefault(
            regime,
            {
                "status": "RUNNING",
                "renderer_parity": parity,
                "bootstrap": {
                    "sequence_start": {"ce": holder1["ce"], **_verdict(base, segnet, posenet, renderer, theta1, labels_np, pose)},
                    "exact_teacher_seconds": exact0_s + exact1_s,
                    "fit_seconds": bootstrap_fit_s,
                    "fit": fit.to_dict(),
                    "joint_trials": bootstrap_trials,
                    "sign_reversed_negative": {"status": "PASS" if negative_pass else "FAIL", "ce": reverse_ce, **(reverse_verdict or {})},
                    "deterministic_repeat_noise_floor": deterministic_noise_floor,
                    "across_seed_variance": "UNKNOWN_single_seed_spine",
                },
                "arms": {},
            },
        )
        bootstrap_path = args.output_dir / f"stage_bootstrap_{regime}.pt"
        if "stage_checkpoint" not in row["bootstrap"]:
            row["bootstrap"]["stage_checkpoint"] = _write_checkpoint(
                bootstrap_path,
                {
                    "schema": SCHEMA,
                    "run_contract_sha256": current_contract["sha256"],
                    "regime": regime,
                    "cadence": 0,
                    "completed_step": 0,
                    "model": bootstrap_state["model"],
                    "ema_shadow": bootstrap_state["ema_shadow"],
                    "ema_updates": bootstrap_state["ema_updates"],
                    "optimizer": bootstrap_state["optimizer"],
                    "theta": bootstrap_state["theta"],
                    "anchor_theta": bootstrap_state["anchor_theta"],
                    "anchor_fraction": bootstrap_state["anchor_fraction"],
                    "torch_rng_state": torch.get_rng_state(),
                    "numpy_rng_state": np.random.get_state(),
                    "python_rng_state": random.getstate(),
                    "inference_weights": "ema_shadow",
                },
            )
        else:
            _load_checkpoint(
                row["bootstrap"]["stage_checkpoint"],
                output_dir=args.output_dir,
                expected={
                    "schema": SCHEMA,
                    "run_contract_sha256": current_contract["sha256"],
                    "regime": regime,
                    "cadence": 0,
                    "completed_step": 0,
                },
            )
        _atomic_json(receipt_path, receipt)

        for cadence in policy.measured_cadences:
            arm_key = f"K{cadence}"
            arm = row["arms"].setdefault(arm_key, {"status": "RUNNING", "cadence": cadence, "steps": []})
            if arm.get("status") == "MEASURED":
                _load_checkpoint(
                    arm["stage_checkpoint"],
                    output_dir=args.output_dir,
                    expected={
                        "schema": SCHEMA,
                        "run_contract_sha256": current_contract["sha256"],
                        "regime": regime,
                        "cadence": cadence,
                        "completed_step": args.steps,
                    },
                )
                continue
            if arm.get("status") == "BLOCKED":
                continue
            arm_seed = _arm_seed(args.seed, regime, cadence)
            torch.manual_seed(arm_seed)
            np.random.seed(arm_seed)
            random.seed(arm_seed)
            model = NonlinearCostateSurrogate(policy.hidden_channels)
            provider_model = NonlinearCostateSurrogate(policy.hidden_channels)
            optimizer = torch.optim.Adam(model.parameters(), lr=2.0e-3)
            ema = _EMA(model, decay=policy.ema_decay)
            model.load_state_dict(bootstrap_state["model"])
            ema.shadow = {key: value.clone() for key, value in bootstrap_state["ema_shadow"].items()}
            ema._num_updates = int(bootstrap_state["ema_updates"])
            provider_model.load_state_dict(ema.state_dict())
            optimizer.load_state_dict(bootstrap_state["optimizer"])
            theta = bootstrap_state["theta"].clone().requires_grad_(True)
            anchor_frame = bootstrap_state["anchor_frame"].clone()
            anchor_costate = bootstrap_state["anchor_costate"].clone()
            anchor_theta = bootstrap_state["anchor_theta"].clone()
            anchor_fraction = float(bootstrap_state["anchor_fraction"])
            start_step = 1
            if args.resume and arm["steps"]:
                last_step = arm["steps"][-1]
                state = _load_checkpoint(
                    arm["resume_checkpoint"],
                    output_dir=args.output_dir,
                    expected={
                        "schema": SCHEMA,
                        "run_contract_sha256": current_contract["sha256"],
                        "regime": regime,
                        "cadence": cadence,
                        "completed_step": int(last_step["step"]),
                    },
                )
                model.load_state_dict(state["model"])
                ema.shadow = {key: value.clone() for key, value in state["ema_shadow"].items()}
                ema._num_updates = int(state["ema_updates"])
                provider_model.load_state_dict(ema.state_dict())
                optimizer.load_state_dict(state["optimizer"])
                theta = state["theta"].requires_grad_(True)
                anchor_theta = state["anchor_theta"]
                reconstruction_counts_before = dict(teacher_calls)
                reconstruction_started = time.perf_counter()
                anchor_frame_nhwc = yopo._render_chart(renderer, anchor_theta)
                anchor_frame = anchor_frame_nhwc.permute(0, 3, 1, 2).detach()
                anchor_costate = yopo._capture_exact_teacher_costate(
                    segnet=segnet, frame_nchw=anchor_frame, labels=labels_t
                )[0]
                arm["resume_reconstruction"] = {
                    "status": "MEASURED_OPERATIONAL_OVERHEAD",
                    "seconds": time.perf_counter() - reconstruction_started,
                    "teacher_work_counts": {
                        key: teacher_calls[key] - reconstruction_counts_before[key] for key in teacher_calls
                    },
                    "reason": "reconstruct exact anchor frame and costate from contract-bound anchor_theta",
                }
                anchor_fraction = float(state["anchor_fraction"])
                torch.set_rng_state(state["torch_rng_state"])
                np.random.set_state(state["numpy_rng_state"])
                random.setstate(state["python_rng_state"])
                start_step = int(state["completed_step"]) + 1

            for step in range(start_step, args.steps + 1):
                refresh = step == 1 or (step - 1) % cadence == 0
                validation_due = refresh or step % cadence == 0 or step == args.steps
                counts_before = dict(teacher_calls)
                current_verdict = (
                    _verdict(base, segnet, posenet, renderer, theta, labels_np, pose)
                    if validation_due
                    else None
                )
                counts_before_operational = dict(teacher_calls)
                operational_started = time.perf_counter()
                frame = yopo._render_chart(renderer, theta)
                frame_nchw = frame.permute(0, 3, 1, 2)
                fit_row = None
                training_s = 0.0
                operational_exact_s = 0.0
                operational_trials: list[dict[str, Any]] = []
                if refresh:
                    exact_costate, holder, operational_exact_s = yopo._capture_exact_teacher_costate(
                        segnet=segnet, frame_nchw=frame_nchw, labels=labels_t
                    )
                    if step > 1 and cadence > 1:
                        sample = OnPolicyTransition(
                            anchor_frame,
                            anchor_costate,
                            frame_nchw.detach(),
                            exact_costate.detach(),
                            step,
                            provider_custody,
                        )
                        started = time.perf_counter()
                        fit_row = fit_onpolicy_transitions(
                            model,
                            [sample],
                            optimizer=optimizer,
                            steps=policy.fit_steps_per_anchor,
                            ema=ema,
                        ).to_dict()
                        provider_model.load_state_dict(ema.state_dict())
                        training_s = time.perf_counter() - started
                        if not fit_row["finite"] or not fit_row["improved"]:
                            arm["steps"].append(
                                {
                                    "step": step,
                                    "status": "BLOCKED",
                                    "reason": "anchor_onpolicy_fit_did_not_complete_with_finite_improvement",
                                    "fit": fit_row,
                                }
                            )
                            arm["status"] = "BLOCKED"
                            _atomic_json(receipt_path, receipt)
                            break
                    anchor_frame, anchor_costate = frame_nchw.detach(), exact_costate.detach()
                    anchor_theta = theta.detach().clone()
                    candidate_costate = exact_costate
                    provider_s = operational_exact_s + training_s
                    provider_mode = "exact_teacher_anchor"
                else:
                    started = time.perf_counter()
                    candidate_costate = predict_detached_costate(
                        provider_model,
                        current_frame=frame_nchw.detach(),
                        anchor_frame=anchor_frame,
                        anchor_costate=anchor_costate,
                    )
                    provider_s = time.perf_counter() - started
                    provider_mode = "nonlinear_surrogate_no_teacher"

                if arm.get("status") == "BLOCKED":
                    break

                grad_started = time.perf_counter()
                candidate_grad = torch.autograd.grad((frame_nchw * candidate_costate).sum(), theta, retain_graph=False)[0]
                renderer_vjp_s = time.perf_counter() - grad_started
                if refresh:
                    candidate, step_norm, next_fraction, operational_trials = _reference_step(
                        yopo=yopo,
                        renderer=renderer,
                        theta=theta,
                        gradient=candidate_grad,
                        segnet=segnet,
                        labels_t=labels_t,
                        current_ce=holder["ce"],
                    )
                    if candidate is not None and next_fraction is not None:
                        anchor_fraction = next_fraction
                else:
                    step_norm = anchor_fraction * max(float(torch.linalg.vector_norm(theta).item()), 1.0)
                    candidate = _candidate_at_norm(theta, candidate_grad, step_norm)
                if candidate is None or step_norm is None:
                    arm["steps"].append(
                        {
                            "step": step,
                            "status": "BLOCKED",
                            "reason": "operational_control_law_reached_bit-identical_or_nonfinite_completion",
                            "operational_trials": operational_trials,
                        }
                    )
                    arm["status"] = "BLOCKED"
                    _atomic_json(receipt_path, receipt)
                    break
                next_theta = candidate.detach().requires_grad_(True)
                operational_step_s = time.perf_counter() - operational_started
                counts_after_operational = dict(teacher_calls)

                # Exact validation occurs only at anchors and cycle endpoints.
                # It never changes the operational trajectory or its timing.
                control_exact_s = 0.0
                control_holder = None
                candidate_ce = None
                candidate_verdict = None
                reference = None
                reference_ce = None
                reference_verdict = None
                reference_trials: list[dict[str, Any]] = []
                agreement = None
                holds: bool | None = None
                if validation_due:
                    if refresh:
                        control_costate, control_holder = exact_costate, holder
                        reference, reference_trials = candidate, operational_trials
                    else:
                        control_costate, control_holder, control_exact_s = yopo._capture_exact_teacher_costate(
                            segnet=segnet, frame_nchw=frame_nchw, labels=labels_t
                        )
                        exact_grad = torch.autograd.grad(
                            (
                                yopo._render_chart(renderer, theta).permute(0, 3, 1, 2)
                                * control_costate
                            ).sum(),
                            theta,
                        )[0]
                        reference, _, _, reference_trials = _reference_step(
                            yopo=yopo,
                            renderer=renderer,
                            theta=theta,
                            gradient=exact_grad,
                            segnet=segnet,
                            labels_t=labels_t,
                            current_ce=control_holder["ce"],
                        )
                    candidate_ce = yopo._evaluate_teacher(
                        segnet, yopo._render_chart(renderer, candidate), labels_t
                    )[0]
                    candidate_verdict = _verdict(
                        base, segnet, posenet, renderer, candidate, labels_np, pose
                    )
                    holds = bool(candidate_ce < control_holder["ce"])
                    if reference is not None and refresh:
                        reference_ce, reference_verdict = candidate_ce, candidate_verdict
                    elif reference is not None:
                        reference_ce = yopo._evaluate_teacher(
                            segnet, yopo._render_chart(renderer, reference), labels_t
                        )[0]
                        reference_verdict = _verdict(
                            base, segnet, posenet, renderer, reference, labels_np, pose
                        )
                    agreement = measure_costate_agreement(
                        control_costate.detach().numpy(), candidate_costate.detach().numpy()
                    ).to_dict()
                row_step = {
                    "step": step,
                    "status": "MEASURED",
                    "provider_mode": provider_mode,
                    "refresh": refresh,
                    "validation_due": validation_due,
                    "fit": fit_row,
                    "current": (
                        {"ce": control_holder["ce"], **current_verdict}
                        if control_holder is not None and current_verdict is not None
                        else None
                    ),
                    "candidate": ({"ce": candidate_ce, **candidate_verdict} if candidate_verdict else None),
                    "reference": ({"ce": reference_ce, **reference_verdict} if reference_verdict else None),
                    "holds_teacher_relaxation_descent": holds,
                    "component_step_predicates": (
                        {
                            "dseg_nonworsening": candidate_verdict["d_seg"] <= current_verdict["d_seg"],
                            "dpose_nonworsening": candidate_verdict["d_pose"] <= current_verdict["d_pose"],
                        }
                        if candidate_verdict is not None and current_verdict is not None
                        else None
                    ),
                    "operational_trajectory_used_exact_fallback": False,
                    "regret": (
                        {
                            "ce": candidate_ce - reference_ce,
                            "d_seg": candidate_verdict["d_seg"] - reference_verdict["d_seg"],
                            "d_pose": candidate_verdict["d_pose"] - reference_verdict["d_pose"],
                        }
                        if candidate_verdict and reference_verdict
                        else None
                    ),
                    "costate_agreement": agreement,
                    "timing_seconds": {
                        "operational_provider": provider_s,
                        "operational_renderer_vjp": renderer_vjp_s,
                        "operational_whole_step": operational_step_s,
                        "measurement_only_exact_costate": control_exact_s,
                    },
                    "teacher_work_counts": {
                        "operational": {
                            key: counts_after_operational[key] - counts_before_operational[key]
                            for key in teacher_calls
                        },
                        "measurement_only_excluded_from_economics": {
                            key: (counts_before_operational[key] - counts_before[key])
                            + (teacher_calls[key] - counts_after_operational[key])
                            for key in teacher_calls
                        },
                    },
                    "operational_control_fraction": anchor_fraction,
                    "operational_trials": operational_trials,
                    "reference_trials": reference_trials,
                }
                theta = next_theta
                checkpoint_path = args.output_dir / f"checkpoint_{regime}_{arm_key}_slot{step % 2}.pt"
                checkpoint = _write_checkpoint(
                    checkpoint_path,
                    {
                        "schema": SCHEMA,
                        "run_contract_sha256": current_contract["sha256"],
                        "regime": regime,
                        "cadence": cadence,
                        "completed_step": step,
                        "model": model.state_dict(),
                        "ema_shadow": ema.state_dict(),
                        "ema_updates": ema._num_updates,
                        "optimizer": optimizer.state_dict(),
                        "theta": theta.detach(),
                        "anchor_theta": anchor_theta,
                        "anchor_fraction": anchor_fraction,
                        "torch_rng_state": torch.get_rng_state(),
                        "numpy_rng_state": np.random.get_state(),
                        "python_rng_state": random.getstate(),
                        "inference_weights": "ema_shadow",
                    },
                )
                arm["steps"].append(row_step)
                arm["resume_checkpoint"] = checkpoint
                _atomic_json(receipt_path, receipt)

            if arm.get("status") == "BLOCKED":
                continue
            validation_rows = [
                s for s in arm["steps"] if not s["refresh"] and s["validation_due"]
            ]
            holds = bool(validation_rows) and all(
                s["holds_teacher_relaxation_descent"] for s in validation_rows
            )
            endpoint_frame = yopo._render_chart(renderer, theta)
            endpoint_ce = yopo._evaluate_teacher(segnet, endpoint_frame, labels_t)[0]
            endpoint_verdict = _verdict(base, segnet, posenet, renderer, theta, labels_np, pose)
            sequence_start = row["bootstrap"]["sequence_start"]
            endpoint_descent = {
                "ce_descent": endpoint_ce < sequence_start["ce"],
                "dseg_nonworsening": endpoint_verdict["d_seg"] <= sequence_start["d_seg"],
                "dpose_nonworsening": endpoint_verdict["d_pose"] <= sequence_start["d_pose"],
            }
            arm.update(
                {
                    "status": "MEASURED",
                    "all_nonrefresh_cycle_validations_hold_teacher_relaxation_descent": holds,
                    "sequence_start": sequence_start,
                    "sequence_endpoint": {"ce": endpoint_ce, **endpoint_verdict},
                    "sequence_endpoint_descent_predicates": endpoint_descent,
                    "sequence_holds_exact_dseg_dpose_descent": all(endpoint_descent.values()),
                    "full_teacher_fallbacks": 0,
                    "economics": _steady_economics(cadence, arm["steps"]),
                    "mean_regret_nonrefresh": {
                        key: float(
                            np.mean(
                                [
                                    s["regret"][key]
                                    for s in arm["steps"]
                                    if not s["refresh"] and s["regret"] is not None
                                ]
                            )
                        )
                        if any(not s["refresh"] and s["regret"] is not None for s in arm["steps"])
                        else 0.0
                        for key in ("ce", "d_seg", "d_pose")
                    },
                    "review_status": "recovery-written-UNREVIEWED",
                }
            )
            final_path = args.output_dir / f"stage_final_{regime}_{arm_key}.pt"
            last_checkpoint_path = Path(arm["resume_checkpoint"]["path"])
            _atomic_bytes(final_path, last_checkpoint_path.read_bytes())
            arm["stage_checkpoint"] = {
                "path": str(final_path),
                "sha256": _sha256(final_path),
                "bytes": final_path.stat().st_size,
                "inference_weights": "ema_shadow",
            }
            _atomic_json(receipt_path, receipt)
        row["status"] = "MEASURED" if all(arm.get("status") == "MEASURED" for arm in row["arms"].values()) else "BLOCKED"
        _atomic_json(receipt_path, receipt)

    measured = [row for row in receipt["regimes"].values() if row.get("status") == "MEASURED"]
    verdict, reason = _classify_verdict(measured, len(args.regimes))
    receipt.update(
        {
            "status": "MEASURED",
            "verdict": verdict,
            "verdict_reason": reason,
            "verdict_scope": VERDICT_SCOPE,
            "review_status": "recovery-written-UNREVIEWED",
            "completed_at_utc": datetime.now(UTC).isoformat(),
        }
    )
    _atomic_json(receipt_path, receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--storage-plan", type=Path, required=True)
    parser.add_argument("--regimes", nargs="+", choices=tuple(REGIMES), default=tuple(REGIMES))
    parser.add_argument("--steps", type=int, default=40, choices=(40,))
    parser.add_argument("--seed", type=int, default=455)
    parser.add_argument("--resume", action="store_true")
    return parser


if __name__ == "__main__":
    args = _parser().parse_args()
    result = run(args)
    print(json.dumps({"status": result["status"], "verdict": result.get("verdict")}, sort_keys=True))
