#!/usr/bin/env python3
"""Executable JO2 objective, checkpoint, and package-validation primitives.

The current no-launch build intentionally stops before scorer loading.  The
functions below are the real hybrid actuator, joint loss, dual update,
checkpoint bundle, and receiver-package validators that a remote trainer must
use.  The fresh same-object Schur/package consumer now lives in
``ddm_jo2_receiver_close``; the CLI still refuses until the remote trainer calls
that consumer at every stage boundary.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from experiments import ddm_jo1_joint_objective_design as design

AXIS = "[JO1 build-only hybrid objective; no scorer authority]"
TRAINING_IMPLEMENTATION_BLOCKER = design.REMOTE_TRAINER_BLOCKER


class JO1WorkerError(RuntimeError):
    """Fail-closed JO1 worker error."""


@dataclass
class DualState:
    collateral: float = 0.0
    pose: float = 0.0
    collateral_penalty: float = 1.0
    pose_penalty: float = 1.0


@dataclass
class ResumeCursor:
    stage_id: str
    step: int
    field_pass_cursor: int
    package_cursor: int


@dataclass(frozen=True)
class FieldDecomposition:
    fixed: int
    introduced: int
    wrong_to_wrong: int
    candidate_flips: int

    @property
    def net_fixed_flips(self) -> int:
        return self.fixed - self.introduced


class HybridOutputResidual(nn.Module):
    """Shared oriented token context applied after all semantic TokenBlocks.

    The learned path is deliberately small.  Five token indicator planes and
    their fixed horizontal/vertical derivatives form the 15-channel input.
    All learned operations occur after that generic context construction and
    produce a bounded RGB residual on the 384x512 renderer lattice.
    """

    def __init__(self, hidden_channels: int, max_rgb_delta: float) -> None:
        super().__init__()
        if hidden_channels < 1 or hidden_channels > 64 or max_rgb_delta <= 0.0:
            raise JO1WorkerError("hybrid actuator geometry is invalid")
        self.hidden_channels = int(hidden_channels)
        self.max_rgb_delta = float(max_rgb_delta)
        self.context = nn.Conv2d(15, hidden_channels, 1)
        self.oriented = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            3,
            padding=1,
            groups=hidden_channels,
        )
        self.head = nn.Conv2d(hidden_channels, 3, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def _oriented_context(tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise JO1WorkerError("token batch must have shape (B,H,W)")
        one_hot = functional.one_hot(tokens.long(), num_classes=5).permute(0, 3, 1, 2).float()
        horizontal = functional.pad(one_hot, (1, 1, 0, 0), mode="replicate")
        horizontal = 0.5 * (horizontal[:, :, :, 2:] - horizontal[:, :, :, :-2])
        vertical = functional.pad(one_hot, (0, 0, 1, 1), mode="replicate")
        vertical = 0.5 * (vertical[:, :, 2:, :] - vertical[:, :, :-2, :])
        return torch.cat((one_hot, horizontal, vertical), dim=1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        context = self._oriented_context(tokens)
        hidden = functional.gelu(self.context(context))
        hidden = hidden + functional.gelu(self.oriented(hidden))
        return torch.tanh(self.head(hidden)) * self.max_rgb_delta


def apply_output_residual(base_renderer_rgb: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """Actuate immediately before exact R; no semantic TokenBlock follows."""
    if base_renderer_rgb.shape != residual.shape or base_renderer_rgb.ndim != 4:
        raise JO1WorkerError("base/residual renderer tensors differ")
    return (base_renderer_rgb + residual).clamp(0.0, 255.0)


def joint_inner_objective(
    *,
    seg_logits: torch.Tensor,
    target: torch.Tensor,
    retained_base_argmax: torch.Tensor,
    pose6_candidate: torch.Tensor,
    pose6_target: torch.Tensor,
    rate_proxy: torch.Tensor,
    duals: DualState,
    stage: design.StageConfig,
    collateral_rho: float = design.COLLATERAL_CAP,
    pose_cap: float = design.BASE_DPOSE,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Differentiable benefit/harm/pose/rate augmented-Lagrangian objective."""
    if seg_logits.ndim != 4 or seg_logits.shape[1] != 5:
        raise JO1WorkerError("SegNet logits must have shape (B,5,H,W)")
    if target.shape != retained_base_argmax.shape or target.shape != seg_logits.shape[:1] + seg_logits.shape[2:]:
        raise JO1WorkerError("Seg objective field geometry differs")
    if pose6_candidate.shape != pose6_target.shape or pose6_candidate.shape[-1] != 6:
        raise JO1WorkerError("Pose objective must use matching first-six vectors")
    if rate_proxy.ndim != 0:
        raise JO1WorkerError("rate_proxy must be one named scalar")
    probability = seg_logits.softmax(dim=1).gather(1, target.long().unsqueeze(1)).squeeze(1)
    base_error = retained_base_argmax != target
    base_correct = ~base_error
    zero = probability.sum() * 0.0
    # Both terms share the full-field denominator.  Separate conditional means
    # would make one base-error pixel and one of ~118M base-correct pixels carry
    # equal aggregate weight, erasing the collateral physics this objective is
    # required to price.
    field_denominator = float(probability.numel())
    soft_benefit = (
        probability[base_error].sum() / field_denominator if torch.any(base_error) else zero
    )
    soft_harm = (
        (1.0 - probability[base_correct]).sum() / field_denominator
        if torch.any(base_correct)
        else zero
    )
    collateral_violation = soft_harm - float(collateral_rho) * soft_benefit
    pose_mse = functional.mse_loss(pose6_candidate, pose6_target)
    pose_violation = pose_mse - float(pose_cap)
    collateral_positive = functional.relu(collateral_violation)
    pose_positive = functional.relu(pose_violation)
    loss = (
        -float(stage.benefit_weight.value) * soft_benefit
        + float(stage.harm_weight.value) * soft_harm
        + float(stage.pose_weight.value) * pose_mse
        + duals.collateral * collateral_positive
        + 0.5 * duals.collateral_penalty * collateral_positive.square()
        + duals.pose * pose_positive
        + 0.5 * duals.pose_penalty * pose_positive.square()
        + float(stage.rate_proxy_weight.value) * rate_proxy
    )
    return loss, {
        "loss": loss.detach(),
        "soft_benefit": soft_benefit.detach(),
        "soft_harm": soft_harm.detach(),
        "collateral_violation": collateral_violation.detach(),
        "pose_mse": pose_mse.detach(),
        "pose_violation": pose_violation.detach(),
        "rate_proxy": rate_proxy.detach(),
    }


def exact_field_decomposition(
    base: np.ndarray,
    candidate: np.ndarray,
    target: np.ndarray,
) -> FieldDecomposition:
    arrays = tuple(np.asarray(value, dtype=np.uint8) for value in (base, candidate, target))
    if any(value.shape != arrays[0].shape for value in arrays) or arrays[0].shape != (
        design.N_PAIRS,
        design.SEG_H,
        design.SEG_W,
    ):
        raise JO1WorkerError("B/H/W fields must all have exact n600 geometry")
    base_error = arrays[0] != arrays[2]
    candidate_error = arrays[1] != arrays[2]
    return FieldDecomposition(
        fixed=int(np.count_nonzero(base_error & ~candidate_error)),
        introduced=int(np.count_nonzero(~base_error & candidate_error)),
        wrong_to_wrong=int(np.count_nonzero(base_error & candidate_error & (arrays[1] != arrays[0]))),
        candidate_flips=int(np.count_nonzero(candidate_error)),
    )


def update_duals_at_stage_boundary(
    duals: DualState,
    *,
    field: FieldDecomposition,
    d_pose_candidate: float,
) -> DualState:
    """Update exact constraints only at a measured stage boundary."""
    ratio_violation = (
        float(field.introduced > 0)
        if field.fixed == 0
        else field.introduced / field.fixed - design.COLLATERAL_CAP
    )
    pose_violation = d_pose_candidate - design.BASE_DPOSE
    return DualState(
        collateral=max(0.0, duals.collateral + duals.collateral_penalty * ratio_violation),
        pose=max(0.0, duals.pose + duals.pose_penalty * pose_violation),
        collateral_penalty=duals.collateral_penalty,
        pose_penalty=duals.pose_penalty,
    )


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def _atomic_torch(path: Path, value: Any) -> dict[str, Any]:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return design._atomic_bytes(path, buffer.getvalue())


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def save_checkpoint_bundle(
    root: Path,
    *,
    model: nn.Module,
    ema_state: Mapping[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    duals: DualState,
    cursor: ResumeCursor,
    config_sha256: str,
) -> dict[str, Any]:
    """Persist distinct live/EMA/optimizer/RNG/dual/cursor payloads atomically."""
    if cursor.stage_id not in design.REQUIRED_STAGE_IDS or min(
        cursor.step, cursor.field_pass_cursor, cursor.package_cursor
    ) < 0:
        raise JO1WorkerError("resume cursor is invalid")
    root.mkdir(parents=True, exist_ok=True)
    records = {
        "live": _atomic_torch(root / "live.pt", _cpu_tree(model.state_dict())),
        "ema": _atomic_torch(root / "ema.pt", _cpu_tree(dict(ema_state))),
        "optimizer": _atomic_torch(root / "optimizer.pt", _cpu_tree(optimizer.state_dict())),
        "rng": _atomic_torch(root / "rng.pt", _rng_state()),
        "duals": design.atomic_json(root / "duals.json", asdict(duals)),
        "resume_cursor": design.atomic_json(root / "resume_cursor.json", asdict(cursor)),
    }
    manifest = {
        "schema": design.CHECKPOINT_SCHEMA,
        "config_sha256": config_sha256,
        "stage_id": cursor.stage_id,
        "step": cursor.step,
        "field_pass_cursor": cursor.field_pass_cursor,
        "package_cursor": cursor.package_cursor,
        "payloads": records,
        "atomic": True,
    }
    design.atomic_json(root / "CHECKPOINT.json", manifest)
    return manifest


def _verify_record(value: Mapping[str, Any]) -> None:
    path = Path(str(value["path"]))
    if not path.is_file() or path.stat().st_size != int(value["bytes"]):
        raise JO1WorkerError(f"retained payload is absent or size-drifted: {path}")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != value["sha256"]:
        raise JO1WorkerError(f"retained payload hash drifted: {path}")


def validate_checkpoint_bundle(root: Path, expected_config_sha256: str) -> dict[str, Any]:
    manifest = json.loads((root / "CHECKPOINT.json").read_text(encoding="utf-8"))
    required = {
        "schema",
        "config_sha256",
        "stage_id",
        "step",
        "field_pass_cursor",
        "package_cursor",
        "payloads",
        "atomic",
    }
    if set(manifest) != required or manifest["schema"] != design.CHECKPOINT_SCHEMA:
        raise JO1WorkerError("checkpoint manifest schema differs")
    if manifest["config_sha256"] != expected_config_sha256 or manifest["atomic"] is not True:
        raise JO1WorkerError("checkpoint config/atomic binding differs")
    if set(manifest["payloads"]) != {"live", "ema", "optimizer", "rng", "duals", "resume_cursor"}:
        raise JO1WorkerError("checkpoint payload set is incomplete")
    for record in manifest["payloads"].values():
        _verify_record(record)
    return manifest


def validate_stage_package(
    *,
    archive: Path,
    repeat_archive: Path,
    retained_payloads: Mapping[str, Mapping[str, Any]],
    receiver_parseback_identity: bool,
    compensation_object_sha256: str,
    expected_object_sha256: str,
) -> dict[str, Any]:
    """Validate a real, retained, single-p, same-object stage package."""
    required = {
        "candidate_argmax_field",
        "bhw_decomposition",
        "pose6_outputs",
        "exact_package",
        "decoded_render_identity",
        "metrics_json",
    }
    if set(retained_payloads) != required:
        raise JO1WorkerError("stage package retained-payload set differs")
    for record in retained_payloads.values():
        _verify_record(record)
    member = design.read_single_p_archive(archive)
    repeat_member = design.read_single_p_archive(repeat_archive)
    if archive.read_bytes() != repeat_archive.read_bytes() or member != repeat_member:
        raise JO1WorkerError("stage archive deterministic repeat differs")
    if not receiver_parseback_identity:
        raise JO1WorkerError("receiver parse-back identity was not proved")
    if compensation_object_sha256 != expected_object_sha256:
        raise JO1WorkerError("fresh Schur compensation is bound to a different object")
    return {
        "schema": "ddm_jo1_validated_stage_package.v1",
        "archive": {
            "path": str(archive.resolve()),
            "bytes": archive.stat().st_size,
            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        },
        "member_p_bytes": len(member),
        "single_p": True,
        "repeat_identity": True,
        "receiver_parseback_identity": True,
        "fresh_same_object_compensation": True,
        "retained_payloads": dict(retained_payloads),
    }


def _write_blocker(output: Path, config: design.CompiledConfig, reasons: Sequence[str]) -> dict[str, Any]:
    result = {
        "schema": "ddm_jo1_worker_blocked.v1",
        "status": "BLOCKED",
        "blockers": list(reasons),
        "action": config.action,
        "workload_config_sha256": config.workload_config_sha256,
        "axis": AXIS,
        "score_claim": False,
        "frontier_moved": False,
    }
    design.atomic_json(output / "WORKER_BLOCKED.json", result)
    return result


def run(compiled_config: Path, expected_config_sha256: str) -> dict[str, Any]:
    config = design.load_compiled_config(compiled_config, expected_config_sha256)
    output = Path(config.output_root).resolve() / config.run_id
    ready = design.readiness(config)
    blockers = list(ready["blockers"])
    # This build does not pretend that a package validator is the rc2 trainer.
    # The named closure is left explicit for the parent landing to route.
    if (
        config.action in {"memory_preflight", "train"}
        and TRAINING_IMPLEMENTATION_BLOCKER not in blockers
    ):
        blockers.append(TRAINING_IMPLEMENTATION_BLOCKER)
    return _write_blocker(output, config, blockers or ["NO_REMOTE_ACTION_IN_BUILD_ONLY_WORKER"])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled-config", required=True, type=Path)
    parser.add_argument("--expected-config-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run(args.compiled_config, args.expected_config_sha256)
    except (OSError, ValueError, design.JO1Error, JO1WorkerError) as error:
        print(json.dumps({"status": "BLOCKED", "reason": str(error)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
