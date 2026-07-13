#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Resumable matched-window probe for the task-455 input-costate surrogate.

This is a local research harness, not an evaluator.  It trains only on dense
exact labels collected from the witness's own realized-through-R trajectory,
then compares an exact-costate branch and an EMA-surrogate branch from the same
start under the exact branch's immutable per-step norm schedule.  Frozen scorer
calls used to observe CE/d_seg/d_pose are accounted separately from operational
teacher calls, so validation cannot manufacture a speedup.
"""

from __future__ import annotations

import argparse
import copy
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
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

from tac.scorer_surrogate.amortized_onpolicy_costate import (  # noqa: E402
    AUTHORITY_SCOPE,
    RESEARCH_ONLY,
    AmortizedCostateConfig,
    AmortizedOnPolicyCostate,
    EMACostateProvider,
    OnPolicyTransition,
    ProviderCustody,
    checkpoint_payload,
    fit_dense_onpolicy_batch,
    predict_ema_detached_costate,
    restore_checkpoint_payload,
)
from tac.scorer_surrogate.onpolicy_matched_verdict import (  # noqa: E402
    ArmEvidence,
    CommonStepSchedule,
    EvidenceStatus,
    ExactMetricAuthority,
    MetricObservation,
    MetricTrace,
    RegimeEvidence,
    adjudicate_matched_windows,
    aggregate_isolated_timings,
    derive_deterministic_repeat_noise_floor,
)
from tac.witness_dsl.onpolicy_scorer_surrogate_policy import (  # noqa: E402
    OnPolicyScorerSurrogatePolicy,
)

SCHEMA: Final[str] = "onpolicy_costate_matched_window_probe.v1"
CHECKPOINT_SCHEMA: Final[str] = "onpolicy_costate_matched_window_checkpoint.v1"
AXIS: Final[str] = "[macOS-CPU advisory training-gradient]"
OPERATOR_REFERENCE_MS: Final[float] = 1656.0
OPERATOR_REFERENCE_PROVENANCE: Final[str] = "operator-supplied exact-forward baseline; not measured by this run"
PAIR_INDEX: Final[int] = 0
FRAME_CHANNELS: Final[int] = 3
FRAME_VALUE_SCALE: Final[float] = 255.0
DEFAULT_NORMALIZATION_FLOOR: Final[float] = torch.finfo(torch.float32).eps
DEFAULT_MSE_WEIGHT: Final[float] = 1.0
DEFAULT_COSINE_WEIGHT: Final[float] = 1.0
DEFAULT_ADMISSION_IMPROVEMENT: Final[float] = 0.0
TARGET_TEACHER_SKIP_FRACTION: Final[float] = 0.95
DERIVED_TARGET_CADENCE: Final[int] = math.ceil(1.0 / (1.0 - TARGET_TEACHER_SKIP_FRACTION))
DEFAULT_WINDOW_STEPS: Final[int] = math.ceil(math.sqrt(DERIVED_TARGET_CADENCE))
DEFAULT_COLLECTION_STEPS: Final[int] = DEFAULT_WINDOW_STEPS
DEFAULT_BRANCH_KERNELS: Final[tuple[int, ...]] = (3, 5)
DEFAULT_OPTIMIZER_STEPS: Final[int] = len(DEFAULT_BRANCH_KERNELS)
DEFAULT_HIDDEN_CHANNELS: Final[int] = 2 * (FRAME_CHANNELS * 3 - 1)
DEFAULT_EMA_DECAY: Final[float] = 1.0 - 1.0 / DEFAULT_WINDOW_STEPS
DEFAULT_REGIME: Final[str] = "boundary"
DEFAULT_SEED: Final[int] = 455
LARGE_ARTIFACT_THRESHOLD_BYTES: Final[int] = 10 * 1024 * 1024
RECEIVER_HEIGHT: Final[int] = 874
RECEIVER_WIDTH: Final[int] = 1164
FP32_BYTES: Final[int] = torch.finfo(torch.float32).bits // 8
ANCHOR_TENSORS_PER_CHECKPOINT: Final[int] = 2
ROLLING_CHECKPOINT_SLOTS: Final[int] = 2
PRESERVED_STAGE_CHECKPOINTS: Final[int] = 4
MODEL_OPTIMIZER_ALLOWANCE_BYTES: Final[int] = 16 * 1024 * 1024
DERIVED_MIN_STORAGE_BYTES: Final[int] = (
    FRAME_CHANNELS
    * RECEIVER_HEIGHT
    * RECEIVER_WIDTH
    * FP32_BYTES
    * ANCHOR_TENSORS_PER_CHECKPOINT
    * (ROLLING_CHECKPOINT_SLOTS + PRESERVED_STAGE_CHECKPOINTS)
    + MODEL_OPTIMIZER_ALLOWANCE_BYTES
)

REGIMES: Final[dict[str, str]] = {
    "early": "frozen_ep299_CEend.npz",
    "boundary": "frozen_ep726_MuonStart.npz",
    "late": "frozen_ep925_liveEMA.npz",
}
CHECKPOINT_DIR = REPO / "experiments/results/tau_crossover_trainflow_20260707"
GT_CACHE = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n6.npz"
SEGNET = REPO / "upstream/models/segnet.safetensors"
POSENET = REPO / "upstream/models/posenet.safetensors"
VIDEO = REPO / "upstream/videos/0.mkv"
SOURCE_FILES: Final[tuple[str, ...]] = (
    "tools/probe_onpolicy_costate_matched_window.py",
    "tools/probe_onpolicy_scorer_surrogate.py",
    "tools/probe_yopo_first_layer_costate.py",
    "src/tac/scorer_surrogate/amortized_onpolicy_costate.py",
    "src/tac/scorer_surrogate/onpolicy_costate.py",
    "src/tac/scorer_surrogate/onpolicy_matched_verdict.py",
    "src/tac/witness_dsl/onpolicy_scorer_surrogate_policy.py",
    "src/tac/boundary_math/segnet_gradient_replacement.py",
    "experiments/train_witness_realized_through_R_mlx.py",
)


class ProbeError(RuntimeError):
    """Fail-closed probe contract or custody error."""


@dataclass(frozen=True)
class MatchedProbeConfig:
    regime: str
    seed: int
    collection_steps: int
    window_steps: int
    optimizer_steps_per_label: int
    step_fraction: float
    learning_rate: float
    ema_decay: float
    hidden_channels: int
    branch_kernel_sizes: tuple[int, ...]
    target_teacher_skip_fraction: float

    @property
    def anchor_cadence(self) -> int:
        # K is derived from the requested skip fraction: 1 - 1/K >= target.
        return math.ceil(1.0 / (1.0 - self.target_teacher_skip_fraction))

    def validate(self) -> None:
        if self.regime not in REGIMES:
            raise ProbeError(f"unknown regime {self.regime!r}")
        integer_fields = {
            "seed": self.seed,
            "collection_steps": self.collection_steps,
            "window_steps": self.window_steps,
            "optimizer_steps_per_label": self.optimizer_steps_per_label,
            "hidden_channels": self.hidden_channels,
        }
        if any(not isinstance(value, int) for value in integer_fields.values()):
            raise ProbeError("seed, horizons, optimizer steps, and hidden channels must be integers")
        if self.collection_steps < 1 or self.window_steps < 1 or self.optimizer_steps_per_label < 1:
            raise ProbeError("collection/window/optimizer step counts must be >= 1")
        if self.hidden_channels < FRAME_CHANNELS:
            raise ProbeError("hidden channels must be >= frame channels")
        if not math.isfinite(self.step_fraction) or self.step_fraction <= 0.0:
            raise ProbeError("step fraction must be finite > 0")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ProbeError("learning rate must be finite > 0")
        if not math.isfinite(self.ema_decay) or not 0.0 <= self.ema_decay < 1.0:
            raise ProbeError("EMA decay must be in [0, 1)")
        if not 0.0 <= self.target_teacher_skip_fraction < 1.0:
            raise ProbeError("target teacher skip fraction must be in [0, 1)")
        if not self.branch_kernel_sizes or any(kernel < 1 or kernel % 2 == 0 for kernel in self.branch_kernel_sizes):
            raise ProbeError("branch kernels must be positive odd integers")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["branch_kernel_sizes"] = list(self.branch_kernel_sizes)
        payload["anchor_cadence"] = self.anchor_cadence
        payload["anchor_cadence_derivation"] = "ceil(1/(1-target_teacher_skip_fraction))"
        payload["value_provenance"] = {
            "window_steps": (
                "typed selection: ceil(sqrt(anchor_cadence)) smoke or anchor_cadence decisive proof"
            ),
            "collection_steps": "equal to matched window for dense one-label-per-state collection",
            "branch_kernel_sizes": "first two nontrivial odd spatial supports",
            "hidden_channels": "2*(nine input channels - reference channel)=16",
            "ema_decay": "1 - 1/window_steps",
            "optimizer_steps_per_label": "number of multiscale branches",
            "normalization_floor": "torch.finfo(float32).eps",
            "mse_weight": "equal-weight loss component",
            "cosine_weight": "equal-weight loss component",
            "step_fraction": (
                "required explicit maximum for joint CE/d_seg/d_pose fractional halving; "
                "halving is bounded by torch.finfo(parameter_dtype).bits; exhaustion is BLOCKED"
            ),
            "learning_rate": "required explicit typed optimizer input; no hidden default",
        }
        return payload


@dataclass
class TimingLedger:
    """Disjoint timing buckets; the operator reference is never a sample."""

    measured_seconds: dict[str, list[float]] = field(
        default_factory=lambda: {
            "collection_exact_costate": [],
            "anchor_fit": [],
            "exact_forward_only": [],
            "exact_costate_forward_backward": [],
            "surrogate_inference": [],
            "surrogate_matched_warmup_exact_forward": [],
            "renderer_vjp_collection": [],
            "renderer_vjp_exact_control": [],
            "renderer_vjp_surrogate_target": [],
            "candidate_update_exact_control": [],
            "candidate_update_surrogate_target": [],
            "exact_window_operational_step": [],
            "surrogate_window_operational_step": [],
            "surrogate_anchor_operational_step": [],
            "surrogate_nonanchor_operational_step": [],
            "exact_validation": [],
            "surrogate_validation": [],
            "surrogate_anchor_exact_costate": [],
            "repeat_exact_costate_forward_backward": [],
            "repeat_exact_forward_only": [],
            "repeat_window_operational_step": [],
            "repeat_validation": [],
            "schedule_derivation_line_search": [],
            "line_search_completion_exact_costate": [],
        }
    )

    def add(self, bucket: str, elapsed_seconds: float) -> None:
        if bucket not in self.measured_seconds:
            raise ProbeError(f"undeclared timing bucket {bucket!r}")
        if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0.0:
            raise ProbeError("timing samples must be finite and nonnegative")
        self.measured_seconds[bucket].append(float(elapsed_seconds))

    def summary(self) -> dict[str, Any]:
        measured: dict[str, dict[str, float | int | None]] = {}
        for name, values in self.measured_seconds.items():
            measured[name] = {
                "count": len(values),
                "total_seconds": float(sum(values)),
                "mean_seconds": float(np.mean(values)) if values else None,
            }
        return {
            "measured": measured,
            "operator_supplied_reference": {
                "exact_forward_ms": OPERATOR_REFERENCE_MS,
                "provenance": OPERATOR_REFERENCE_PROVENANCE,
                "included_in_measured_samples": False,
            },
        }

    @classmethod
    def from_samples(cls, samples: dict[str, list[float]]) -> TimingLedger:
        ledger = cls()
        if set(samples) != set(ledger.measured_seconds):
            raise ProbeError("checkpoint timing bucket schema mismatch")
        ledger.measured_seconds = {key: [float(value) for value in values] for key, values in samples.items()}
        for key, values in ledger.measured_seconds.items():
            for value in values:
                if not math.isfinite(value) or value < 0.0:
                    raise ProbeError(f"checkpoint timing bucket {key!r} contains an invalid sample")
        return ledger


def derive_common_step_norm(theta: torch.Tensor, *, step_fraction: float) -> float:
    """Select one norm from the exact branch state for use by both branches."""

    theta_norm = float(torch.linalg.vector_norm(theta.detach()).item())
    floor = float(torch.finfo(theta.dtype).eps)
    value = step_fraction * max(theta_norm, floor)
    if not math.isfinite(value) or value <= 0.0:
        raise ProbeError("derived exact-branch step norm is not finite and positive")
    return value


def candidate_at_common_norm(theta: torch.Tensor, gradient: torch.Tensor, step_norm: float) -> torch.Tensor:
    """Apply a branch-specific direction with the exact branch's scalar norm."""

    gradient_norm = float(torch.linalg.vector_norm(gradient.detach()).item())
    if not math.isfinite(gradient_norm) or gradient_norm <= 0.0:
        raise ProbeError("cannot apply common schedule to zero/nonfinite branch gradient")
    return theta.detach() - gradient.detach() * (step_norm / gradient_norm)


def complete_operational_step_seconds(provider_path: float, candidate_update: float) -> float:
    """Compose the same complete operational boundary for both matched arms."""

    values = (provider_path, candidate_update)
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ProbeError("operational step components must be finite and nonnegative")
    return math.fsum(values)


def joint_descent_predicates(
    current: dict[str, float], candidate: dict[str, float]
) -> dict[str, bool]:
    """Require strict CE descent while holding both exact through-R metrics."""

    return {
        "ce_strict_descent": candidate["ce"] < current["ce"],
        "d_seg_nonworsening": candidate["d_seg"] <= current["d_seg"],
        "d_pose_nonworsening": candidate["d_pose"] <= current["d_pose"],
    }


def classify_exact_window_completion(
    *, certified_zero_renderer_gradient: bool, observed_updates: int, requested_updates: int
) -> str:
    """Distinguish a proved zero-gradient floor from blocked line-search exhaustion."""

    if certified_zero_renderer_gradient:
        return "CERTIFIED_ZERO_RENDERER_GRADIENT"
    if observed_updates >= requested_updates:
        return "REQUESTED_HORIZON_COMPLETED"
    return "LINE_SEARCH_BLOCKED_AFTER_MEASURED_PREFIX"


def matched_schedule_record(step: int, exact_selected_norm: float) -> dict[str, Any]:
    """One immutable schedule row explicitly bound to both branches."""

    return {
        "step": step,
        "selected_by": "exact_branch",
        "exact_step_norm": exact_selected_norm,
        "surrogate_step_norm": exact_selected_norm,
        "identical_norm_predicate": True,
    }


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
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())


def _torch_bytes(payload: dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


@dataclass(frozen=True)
class CheckpointRecord:
    path: str
    sha256: str
    bytes: int
    stage: str
    sequence: int
    preserved: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TwoSlotCheckpointStore:
    """Atomic rolling two-slot checkpoints plus immutable stage checkpoints."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _write(
        self, path: Path, payload: dict[str, Any], *, stage: str, sequence: int, preserved: bool
    ) -> CheckpointRecord:
        encoded = _torch_bytes(payload)
        _atomic_bytes(path, encoded)
        record = CheckpointRecord(
            path=str(path),
            sha256=hashlib.sha256(encoded).hexdigest(),
            bytes=len(encoded),
            stage=stage,
            sequence=sequence,
            preserved=preserved,
        )
        _atomic_json(path.with_suffix(path.suffix + ".json"), record.to_dict())
        return record

    def save_slot(self, payload: dict[str, Any], *, stage: str, sequence: int) -> CheckpointRecord:
        if sequence < 0 or not stage:
            raise ProbeError("checkpoint stage and nonnegative sequence are required")
        slot = sequence % 2
        # Slots are global across stages so retained rolling state is bounded
        # to exactly two files; the stage lives in the custody sidecar/payload.
        path = self.output_dir / f"checkpoint_slot{slot}.pt"
        record = self._write(path, payload, stage=stage, sequence=sequence, preserved=False)
        _atomic_json(self.output_dir / "checkpoint_latest.json", record.to_dict())
        return record

    def preserve_stage(self, payload: dict[str, Any], *, stage: str, sequence: int) -> CheckpointRecord:
        path = self.output_dir / f"stage_{stage}_complete_seq{sequence}.pt"
        if path.exists() or path.with_suffix(path.suffix + ".json").exists():
            raise ProbeError(f"refusing to overwrite preserved stage checkpoint {path}")
        return self._write(path, payload, stage=stage, sequence=sequence, preserved=True)

    def load(self, path: Path, *, expected_run_contract_sha256: str) -> dict[str, Any]:
        resolved = path.resolve()
        try:
            resolved.relative_to(self.output_dir)
        except ValueError as error:
            raise ProbeError("resume checkpoint is outside the selected output directory") from error
        metadata_path = resolved.with_suffix(resolved.suffix + ".json")
        if not resolved.is_file() or not metadata_path.is_file():
            raise ProbeError("resume checkpoint bytes or custody sidecar are missing")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if resolved.stat().st_size != metadata.get("bytes") or _sha256(resolved) != metadata.get("sha256"):
            raise ProbeError("resume checkpoint byte custody mismatch")
        payload = torch.load(resolved, map_location="cpu", weights_only=False)
        if payload.get("schema") != CHECKPOINT_SCHEMA:
            raise ProbeError("resume checkpoint schema mismatch")
        if payload.get("run_contract_sha256") != expected_run_contract_sha256:
            raise ProbeError("source/input/config/storage custody changed; refusing resume")
        if not isinstance(payload.get("anchor_frame"), torch.Tensor) or not isinstance(
            payload.get("anchor_costate"), torch.Tensor
        ):
            raise ProbeError("checkpoint lacks anchor frame/costate; unscheduled teacher restore is forbidden")
        return payload


def build_probe_checkpoint(
    *,
    run_contract_sha256: str,
    stage: str,
    next_step: int,
    anchor_frame: torch.Tensor,
    anchor_costate: torch.Tensor,
    state: dict[str, Any],
) -> dict[str, Any]:
    """Build complete state; anchors make restoration a zero-teacher operation."""

    if anchor_frame.shape != anchor_costate.shape or not bool(torch.isfinite(anchor_costate).all()):
        raise ProbeError("checkpoint anchor frame/costate are incomplete or incompatible")
    return {
        "schema": CHECKPOINT_SCHEMA,
        "run_contract_sha256": run_contract_sha256,
        "stage": stage,
        "next_step": next_step,
        "anchor_frame": anchor_frame.detach().cpu().clone(),
        "anchor_costate": anchor_costate.detach().cpu().clone(),
        "state": state,
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "research_only": True,
        "authority_scope": AUTHORITY_SCOPE,
    }


def restore_probe_checkpoint(
    store: TwoSlotCheckpointStore,
    path: Path,
    *,
    expected_run_contract_sha256: str,
) -> dict[str, Any]:
    """Restore without accepting or invoking an exact-teacher callback."""

    payload = store.load(path, expected_run_contract_sha256=expected_run_contract_sha256)
    torch.set_rng_state(payload["torch_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])
    random.setstate(payload["python_rng_state"])
    return payload


def _import_file(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ProbeError(f"cannot import helper {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _file_custody(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ProbeError(f"required custody input is missing: {path}")
    return {"path": str(path.resolve()), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _source_custody() -> dict[str, dict[str, Any]]:
    return {name: _file_custody(REPO / name) for name in SOURCE_FILES}


def _materialize_source_bundle(output_dir: Path) -> dict[str, dict[str, Any]]:
    bundle: dict[str, dict[str, Any]] = {}
    for name in SOURCE_FILES:
        destination = output_dir / "source_bundle" / name
        _atomic_bytes(destination, (REPO / name).read_bytes())
        bundle[name] = {
            "path": str(destination.relative_to(output_dir)),
            "sha256": _sha256(destination),
            "bytes": destination.stat().st_size,
        }
    return bundle


def _verify_source_bundle(
    output_dir: Path,
    bundle: dict[str, dict[str, Any]],
    expected: dict[str, dict[str, Any]],
) -> None:
    if set(bundle) != set(expected):
        raise ProbeError("source bundle does not cover the run-contract source set")
    for name, meta in bundle.items():
        if meta.get("sha256") != expected[name]["sha256"] or meta.get("bytes") != expected[name]["bytes"]:
            raise ProbeError(f"source bundle does not match run-contract custody for {name}")
        path = (output_dir / meta["path"]).resolve()
        try:
            path.relative_to(output_dir.resolve())
        except ValueError as error:
            raise ProbeError("source bundle path escapes the output directory") from error
        if not path.is_file() or path.stat().st_size != meta["bytes"] or _sha256(path) != meta["sha256"]:
            raise ProbeError(f"source bundle byte custody failed for {name}")


def _input_custody(config: MatchedProbeConfig) -> dict[str, Any]:
    return {
        "gt_cache": _file_custody(GT_CACHE),
        "segnet": _file_custody(SEGNET),
        "posenet": _file_custody(POSENET),
        "video": _file_custody(VIDEO),
        "renderer_checkpoint": _file_custody(CHECKPOINT_DIR / REGIMES[config.regime]),
    }


def _validate_storage_plan(path: Path, output_dir: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ProbeError("--storage-plan is required and must exist")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("blockers"):
        raise ProbeError(f"storage preflight is blocked: {payload['blockers']}")
    selected = payload.get("selected_workload_root")
    if not selected or Path(selected).resolve() != output_dir.resolve():
        raise ProbeError("storage plan selected_workload_root does not match --output-dir")
    if payload.get("requested_bytes", 0) < DERIVED_MIN_STORAGE_BYTES:
        raise ProbeError(
            "storage plan requested_bytes is below the derived rolling-plus-preserved checkpoint requirement"
        )
    if payload.get("selected_tier") == "local" and not payload.get("operator_storage_policy", {}).get(
        "local_disk_enabled"
    ):
        raise ProbeError("local storage requires explicit operator-policy opt-in")
    return payload


def _run_contract(args: argparse.Namespace, config: MatchedProbeConfig) -> dict[str, Any]:
    storage = _validate_storage_plan(args.storage_plan, args.output_dir)
    policy = OnPolicyScorerSurrogatePolicy()
    corrected_policy = policy.compile_corrected_measurement_contract()
    expected = {
        "target_teacher_skip_fraction": policy.target_exact_teacher_skip_fraction,
        "collection_steps": policy.matched_window_steps,
        "optimizer_steps_per_label": policy.dense_optimizer_steps_per_observation,
        "ema_decay": policy.dense_ema_decay,
        "hidden_channels": policy.amortized_hidden_channels,
        "branch_kernel_sizes": policy.branch_kernel_sizes,
    }
    observed = {
        "target_teacher_skip_fraction": config.target_teacher_skip_fraction,
        "collection_steps": config.collection_steps,
        "optimizer_steps_per_label": config.optimizer_steps_per_label,
        "ema_decay": config.ema_decay,
        "hidden_channels": config.hidden_channels,
        "branch_kernel_sizes": config.branch_kernel_sizes,
    }
    if observed != expected:
        raise ProbeError(
            "probe tunables do not match the corrected typed policy contract; all values remain explicit CLI surfaces"
        )
    if config.window_steps not in {
        int(corrected_policy["matched_window_steps"]),
        int(corrected_policy["decisive_window_steps"]),
    }:
        raise ProbeError(
            "window_steps must be the typed smoke horizon or the complete target anchor cadence"
        )
    payload = {
        "schema": SCHEMA,
        "config": config.to_dict(),
        "output_dir": str(args.output_dir.resolve()),
        "source_custody": _source_custody(),
        "input_custody": _input_custody(config),
        "storage_custody": {
            "plan": _file_custody(args.storage_plan),
            "selected_workload_root": storage["selected_workload_root"],
            "requested_bytes": storage["requested_bytes"],
        },
        "typed_policy": corrected_policy,
        "objective": "exact_segnet_ce_input_costate_through_R_training_signal_only",
        "pair_index": PAIR_INDEX,
        "axis": AXIS,
    }
    return {"sha256": _payload_sha256(payload), "payload": payload}


def _git_head() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, text=True, capture_output=True, check=True)
    return result.stdout.strip()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _amortized_config(config: MatchedProbeConfig) -> AmortizedCostateConfig:
    return AmortizedCostateConfig(
        frame_channels=FRAME_CHANNELS,
        hidden_channels=config.hidden_channels,
        branch_kernel_sizes=config.branch_kernel_sizes,
        frame_value_scale=FRAME_VALUE_SCALE,
        normalization_floor=DEFAULT_NORMALIZATION_FLOOR,
        mse_weight=DEFAULT_MSE_WEIGHT,
        cosine_weight=DEFAULT_COSINE_WEIGHT,
        ema_decay=config.ema_decay,
        admission_min_relative_improvement=DEFAULT_ADMISSION_IMPROVEMENT,
    )


def _renderer_gradient(frame_nhwc: torch.Tensor, theta: torch.Tensor, costate_nchw: torch.Tensor) -> torch.Tensor:
    if frame_nhwc.shape != costate_nchw.permute(0, 2, 3, 1).shape:
        raise ProbeError("renderer frame and input costate shape mismatch")
    started = time.perf_counter()
    gradient = torch.autograd.grad((frame_nhwc.permute(0, 3, 1, 2) * costate_nchw.detach()).sum(), theta)[0]
    elapsed = time.perf_counter() - started
    if not bool(torch.isfinite(gradient).all()):
        raise ProbeError("renderer VJP produced a nonfinite gradient")
    return gradient, elapsed


def _metric_row(
    *,
    helper: Any,
    base: Any,
    renderer: Any,
    theta: torch.Tensor,
    segnet: Any,
    posenet: Any,
    labels_t: torch.Tensor,
    labels_np: np.ndarray,
    pose: np.ndarray,
) -> dict[str, float]:
    frame = helper._render_chart(renderer, theta)
    ce, teacher_argmax_debt = helper._evaluate_teacher(segnet, frame, labels_t)
    verdict = base._verdict(base._base_module, segnet, posenet, renderer, theta, labels_np, pose)
    return {
        "ce": ce,
        "teacher_argmax_debt": teacher_argmax_debt,
        "d_seg": verdict["d_seg"],
        "d_pose": verdict["d_pose"],
    }


def _noise_floor(first: list[dict[str, Any]], second: list[dict[str, Any]]) -> dict[str, Any]:
    if len(first) != len(second):
        raise ProbeError("deterministic repeat traces have different lengths")
    keys = ("ce", "d_seg", "d_pose")
    maximum = {key: max(abs(a[key] - b[key]) for a, b in zip(first, second, strict=True)) for key in keys}
    return {
        "status": "MEASURED",
        "repeat_count": 2,
        "max_abs_by_metric": maximum,
        "bit_identical_metrics": all(value == 0.0 for value in maximum.values()),
        "across_seed_variance": "UNKNOWN_single_seed_spine",
    }


def _trajectory_verdict(
    exact: list[dict[str, Any]], surrogate: list[dict[str, Any]], noise: dict[str, Any]
) -> dict[str, Any]:
    if len(exact) != len(surrogate) or not exact:
        raise ProbeError("matched trajectory traces are incomplete")
    floors = noise["max_abs_by_metric"]
    regret: dict[str, list[float]] = {
        key: [s[key] - e[key] for e, s in zip(exact, surrogate, strict=True)] for key in ("ce", "d_seg", "d_pose")
    }
    within_noise = {key: all(abs(value) <= floors[key] for value in values) for key, values in regret.items()}
    matched = all(within_noise.values())
    return {
        "status": "MATCH" if matched else "FAIL",
        "fidelity_gate": "full_window_ce_dseg_dpose_trajectory_parity_at_deterministic_repeat_noise_floor",
        "trajectory_regret": regret,
        "max_abs_regret": {key: max(abs(value) for value in values) for key, values in regret.items()},
        "within_repeat_noise_floor": within_noise,
        "matched_exact_teacher_descent": matched,
        "verdict_scope": (
            "single pair0 saved regime and seed; exact/surrogate common start and exact-selected norm schedule; "
            "through-R macOS-CPU advisory training-gradient; negative does not close the surrogate family"
        ),
        "score_claim": False,
    }


def _canonical_matched_verdict(
    *,
    regime: str,
    schedule_rows: list[dict[str, Any]],
    exact_trace: list[dict[str, Any]],
    surrogate_trace: list[dict[str, Any]],
    repeat_trace: list[dict[str, Any]],
    exact_control_valid: bool,
    exact_terminal_floor: bool,
    ema_provider_admitted: bool,
) -> tuple[dict[str, Any], CommonStepSchedule, Any]:
    """Bind measured traces to the canonical pure verdict implementation."""

    schedule = CommonStepSchedule(
        step_indices=tuple(row["step"] for row in schedule_rows),
        control_values=tuple(row["exact_step_norm"] for row in schedule_rows),
        control_name="exact_branch_selected_normalized_step_norm",
        derivation="step_fraction_times_current_exact_branch_parameter_norm",
    )

    def observation(rows: list[dict[str, Any]], label: str) -> MetricObservation:
        trace = MetricTrace(
            step_indices=tuple(row["step"] for row in rows),
            d_seg=tuple(row["d_seg"] for row in rows),
            ce=tuple(row["ce"] for row in rows),
            d_pose=tuple(row["d_pose"] for row in rows),
            common_step_schedule_sha256=schedule.sha256,
        )
        authority = ExactMetricAuthority(
            ce_exact_teacher_through_r=True,
            d_seg_exact_argmax_through_r=True,
            d_pose_exact_frozen_posenet_through_r=True,
            axis=AXIS,
            evidence_sha256=_payload_sha256(
                {
                    "label": label,
                    "schedule_sha256": schedule.sha256,
                    "trace": rows,
                    "axis": AXIS,
                }
            ),
        )
        return MetricObservation(trace=trace, authority=authority)

    exact_observation = observation(exact_trace, "exact_control")
    repeat_observation = observation(repeat_trace, "exact_control_deterministic_repeat")
    surrogate_observation = observation(surrogate_trace, "ema_surrogate_target")
    noise_floor = derive_deterministic_repeat_noise_floor(
        (exact_observation, repeat_observation), common_step_schedule=schedule
    )
    exact_status = (
        EvidenceStatus.VALID_TERMINAL_FLOOR
        if exact_terminal_floor
        else EvidenceStatus.MEASURED
        if exact_control_valid
        else EvidenceStatus.BLOCKED
    )
    target_status = EvidenceStatus.MEASURED if ema_provider_admitted else EvidenceStatus.BLOCKED
    evidence = RegimeEvidence(
        regime=regime,
        exact_control=ArmEvidence(
            arm_id="exact_costate_control",
            status=exact_status,
            observation=exact_observation,
            status_reason=(
                "exact teacher reached a bit-identical renderer terminal floor"
                if exact_terminal_floor
                else None
                if exact_control_valid
                else (
                    "exact joint control failed strict CE descent or non-worsening exact "
                    "through-R d_seg/d_pose"
                )
            ),
            status_evidence_sha256=(
                exact_observation.authority.evidence_sha256 if exact_terminal_floor or not exact_control_valid else None
            ),
        ),
        surrogate_target=ArmEvidence(
            arm_id="ema_costate_surrogate",
            status=target_status,
            observation=surrogate_observation if ema_provider_admitted else None,
            status_reason=None if ema_provider_admitted else "final EMA-shadow fit did not pass admission",
            status_evidence_sha256=None
            if ema_provider_admitted
            else _payload_sha256({"ema_provider_admitted": False, "regime": regime}),
        ),
    )
    verdict = adjudicate_matched_windows(
        requested_regimes=(regime,),
        regime_evidence=(evidence,),
        common_step_schedule=schedule,
        deterministic_repeat_noise_floor=noise_floor,
    )
    return verdict.to_dict(), schedule, noise_floor


def _base_receipt(args: argparse.Namespace, config: MatchedProbeConfig, run_contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "RUNNING",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "research_only": RESEARCH_ONLY,
        "authority_scope": AUTHORITY_SCOPE,
        "surrogate_predicts": "dL_teacher/d(frame)",
        "surrogate_does_not_predict": "d_seg",
        "git_head_at_launch": _git_head(),
        "argv": sys.argv,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "config": config.to_dict(),
        "run_contract": run_contract,
        "teacher_accounting": {
            "collection_exact_costate_labels": 0,
            "deployment_exact_branch_costates": 0,
            "deployment_surrogate_branch_anchors": 0,
            "deterministic_repeat_exact_costates": 0,
            "measurement_only_segnet_forwards": 0,
            "measurement_only_posenet_forwards": 0,
            "timing_only_segnet_forwards": 0,
            "schedule_derivation_segnet_forwards": 0,
            "schedule_derivation_posenet_forwards": 0,
            "line_search_completion_exact_costates": 0,
            "observed_total_segnet_forwards": 0,
            "observed_total_posenet_forwards": 0,
            "resume_restore_teacher_calls": 0,
        },
        "timing": TimingLedger().summary(),
        "disk_hygiene": {
            "large_artifacts_created": "PENDING_MEASUREMENT",
            "large_artifact_threshold_bytes": LARGE_ARTIFACT_THRESHOLD_BYTES,
            "derived_min_storage_bytes": DERIVED_MIN_STORAGE_BYTES,
            "derivation": (
                "RGB fp32 receiver elements * anchor-frame-and-costate * "
                "(two rolling slots + four preserved stages) + model/optimizer allowance"
            ),
            "cleanup": "atomic temporary files are success-cleaned; stage checkpoints are durable evidence",
        },
    }


def _make_model(config: MatchedProbeConfig) -> tuple[AmortizedOnPolicyCostate, EMACostateProvider, Any]:
    model = AmortizedOnPolicyCostate(_amortized_config(config))
    provider = EMACostateProvider(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    return model, provider, optimizer


def _checkpoint_model_state(
    model: AmortizedOnPolicyCostate,
    provider: EMACostateProvider,
    optimizer: Any,
    next_step: int,
) -> dict[str, Any]:
    return checkpoint_payload(model, provider, optimizer, next_trajectory_step=next_step)


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = MatchedProbeConfig(
        regime=args.regime,
        seed=args.seed,
        collection_steps=args.collection_steps,
        window_steps=args.window_steps,
        optimizer_steps_per_label=args.optimizer_steps_per_label,
        step_fraction=args.step_fraction,
        learning_rate=args.learning_rate,
        ema_decay=args.ema_decay,
        hidden_channels=args.hidden_channels,
        branch_kernel_sizes=tuple(args.branch_kernel_sizes),
        target_teacher_skip_fraction=args.target_teacher_skip_fraction,
    )
    config.validate()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.relative_to((REPO / "experiments/results").resolve())
    run_contract = _run_contract(args, config)
    _seed_everything(config.seed)
    store = TwoSlotCheckpointStore(args.output_dir)
    receipt_path = args.output_dir / "measurement_receipt.json"
    if receipt_path.exists() and args.resume_from is None:
        raise ProbeError("output receipt exists; pass --resume-from or choose a fresh output directory")
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        _verify_source_bundle(
            args.output_dir,
            receipt.get("source_bundle", {}),
            run_contract["payload"]["source_custody"],
        )
    else:
        receipt = _base_receipt(args, config, run_contract)
        receipt["source_bundle"] = _materialize_source_bundle(args.output_dir)
    if receipt.get("run_contract") != run_contract:
        raise ProbeError("source/input/config/storage custody changed; refusing resume")
    _atomic_json(receipt_path, receipt)

    yopo = _import_file("_task455_matched_yopo", REPO / "tools/probe_yopo_first_layer_costate.py")
    old_probe = _import_file("_task455_matched_old_probe", REPO / "tools/probe_onpolicy_scorer_surrogate.py")
    trainer = _import_file("_task455_matched_trainer", REPO / "experiments/train_witness_realized_through_R_mlx.py")
    # Bind the helper expected by _metric_row without broadening its public API.
    old_probe._base_module = trainer
    segnet, posenet = old_probe._load_scorers()
    teacher_counts = receipt["teacher_accounting"]
    segnet.register_forward_hook(
        lambda _module, _inputs, _output: teacher_counts.__setitem__(
            "observed_total_segnet_forwards", teacher_counts["observed_total_segnet_forwards"] + 1
        )
    )
    posenet.register_forward_hook(
        lambda _module, _inputs, _output: teacher_counts.__setitem__(
            "observed_total_posenet_forwards", teacher_counts["observed_total_posenet_forwards"] + 1
        )
    )
    with np.load(GT_CACHE, allow_pickle=False) as cache:
        labels_np = np.asarray(cache["lstars"][PAIR_INDEX], np.int64)
        labels_t = torch.as_tensor(labels_np)[None]
        pose = np.asarray(cache["gt_poses"][PAIR_INDEX], np.float64)
    renderer, code, _model_cfg, _dash = yopo._load_renderer(CHECKPOINT_DIR / REGIMES[config.regime])
    theta_start = torch.as_tensor(code[1], dtype=torch.float32).detach()
    parity = yopo._renderer_parity_canary(renderer, theta_start.requires_grad_(True))
    if parity["max_abs"] != 0.0:
        raise ProbeError("saved renderer chart failed bit-identical parity canary")

    model, provider, optimizer = _make_model(config)
    timing = TimingLedger()
    theta_collection = theta_start.clone()
    anchor_frame: torch.Tensor | None = None
    anchor_costate: torch.Tensor | None = None
    collection_next = 0
    resume_stage: str | None = None
    resume_state: dict[str, Any] = {}
    if args.resume_from is not None:
        resumed = restore_probe_checkpoint(store, args.resume_from, expected_run_contract_sha256=run_contract["sha256"])
        if resumed.get("authority_scope") != AUTHORITY_SCOPE or resumed.get("research_only") is not True:
            raise ProbeError("resume checkpoint changed training-signal-only authority")
        saved = resumed["state"]
        resume_stage = resumed["stage"]
        resume_state = saved
        restore_checkpoint_payload(saved["model"], model, provider, optimizer)
        anchor_frame = resumed["anchor_frame"]
        anchor_costate = resumed["anchor_costate"]
        theta_collection = saved["theta_collection"]
        collection_next = saved["collection_next"]
        timing = TimingLedger.from_samples(saved["timing_samples"])
        teacher_counts.update(saved["teacher_accounting"])
        teacher_counts["resume_restore_teacher_calls"] = 0

    custody = ProviderCustody(
        fingerprint_sha256=_payload_sha256(
            {"run_contract": run_contract["sha256"], "regime": config.regime, "pair_index": PAIR_INDEX}
        ),
        regime=config.regime,
    )
    fit_rows: list[dict[str, Any]] = receipt.get("collection_fit_rows", [])
    if args.resume_from is not None:
        fit_rows = list(saved["fit_rows"])
    for step in range(collection_next, config.collection_steps):
        theta = theta_collection.detach().requires_grad_(True)
        frame = yopo._render_chart(renderer, theta)
        exact_costate, _holder, elapsed_exact = yopo._capture_exact_teacher_costate(
            segnet=segnet, frame_nchw=frame.permute(0, 3, 1, 2), labels=labels_t
        )
        teacher_counts["collection_exact_costate_labels"] += 1
        timing.add("collection_exact_costate", elapsed_exact)
        current_frame = frame.permute(0, 3, 1, 2).detach()
        if anchor_frame is None:
            anchor_frame, anchor_costate = current_frame.clone(), exact_costate.clone()
        assert anchor_costate is not None
        transition = OnPolicyTransition(
            anchor_frame=anchor_frame,
            anchor_costate=anchor_costate,
            current_frame=current_frame,
            current_costate=exact_costate,
            trajectory_step=step,
            custody=custody,
        )
        fit_started = time.perf_counter()
        admission = fit_dense_onpolicy_batch(
            model,
            [transition],
            optimizer=optimizer,
            optimizer_steps=config.optimizer_steps_per_label,
            ema_provider=provider,
        )
        fit_elapsed = time.perf_counter() - fit_started
        timing.add("anchor_fit", fit_elapsed)
        fit_rows.append(
            {
                "step": step,
                "timing_seconds": fit_elapsed,
                "transition_sha256": _payload_sha256(
                    {
                        "trajectory_step": step,
                        "current_frame_sha256": hashlib.sha256(
                            current_frame.contiguous().numpy().tobytes()
                        ).hexdigest(),
                        "current_costate_sha256": hashlib.sha256(
                            exact_costate.contiguous().numpy().tobytes()
                        ).hexdigest(),
                    }
                ),
                **admission.to_dict(),
            }
        )
        candidate_costate = (
            exact_costate
            if step == 0
            else predict_ema_detached_costate(
                provider,
                current_frame=current_frame,
                anchor_frame=anchor_frame,
                anchor_costate=anchor_costate,
            )
        )
        gradient, renderer_vjp_elapsed = _renderer_gradient(frame, theta, candidate_costate)
        timing.add("renderer_vjp_collection", renderer_vjp_elapsed)
        norm = derive_common_step_norm(theta, step_fraction=config.step_fraction)
        theta_collection = candidate_at_common_norm(theta, gradient, norm)
        state = {
            "model": _checkpoint_model_state(model, provider, optimizer, step + 1),
            "theta_collection": theta_collection,
            "collection_next": step + 1,
            "fit_rows": fit_rows,
            "timing_samples": copy.deepcopy(timing.measured_seconds),
            "teacher_accounting": dict(teacher_counts),
        }
        checkpoint = build_probe_checkpoint(
            run_contract_sha256=run_contract["sha256"],
            stage="collection",
            next_step=step + 1,
            anchor_frame=anchor_frame,
            anchor_costate=anchor_costate,
            state=state,
        )
        receipt["latest_checkpoint"] = store.save_slot(checkpoint, stage="collection", sequence=step).to_dict()
        receipt["collection_fit_rows"] = fit_rows
        receipt["timing"] = timing.summary()
        _atomic_json(receipt_path, receipt)

    assert anchor_frame is not None and anchor_costate is not None
    collection_state = {
        "model": _checkpoint_model_state(model, provider, optimizer, config.collection_steps),
        "theta_collection": theta_collection,
        "collection_next": config.collection_steps,
        "fit_rows": fit_rows,
        "timing_samples": copy.deepcopy(timing.measured_seconds),
        "teacher_accounting": dict(teacher_counts),
    }
    collection_checkpoint = build_probe_checkpoint(
        run_contract_sha256=run_contract["sha256"],
        stage="collection",
        next_step=config.collection_steps,
        anchor_frame=anchor_frame,
        anchor_costate=anchor_costate,
        state=collection_state,
    )
    if "collection_stage_checkpoint" not in receipt:
        receipt["collection_stage_checkpoint"] = store.preserve_stage(
            collection_checkpoint, stage="collection", sequence=config.collection_steps
        ).to_dict()
    # Freeze the admitted EMA provider before both deployment branches.
    provider.eval()
    provider_state = copy.deepcopy(provider.state_dict())
    ema_provider_admitted = bool(fit_rows and fit_rows[-1]["admitted"])

    def deployment_state(**extra: Any) -> dict[str, Any]:
        return {
            "model": _checkpoint_model_state(model, provider, optimizer, config.collection_steps),
            "theta_collection": theta_collection,
            "collection_next": config.collection_steps,
            "fit_rows": fit_rows,
            "timing_samples": copy.deepcopy(timing.measured_seconds),
            "teacher_accounting": dict(teacher_counts),
            **extra,
        }

    def exact_window(
        *,
        deterministic_repeat: bool,
        resume: dict[str, Any] | None,
        fixed_schedule: list[dict[str, Any]] | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float, bool]:
        stage = "repeat_window" if deterministic_repeat else "exact_window"
        if resume is not None:
            theta = resume["branch_theta"]
            trace = list(resume["branch_trace"])
            schedule = list(resume["branch_schedule"])
            first_step = resume["branch_next"]
            total_operational = float(resume["branch_operational_seconds"])
            terminal_floor = bool(resume.get("exact_terminal_floor", False))
        else:
            theta = theta_collection.detach().clone()
            start_validation = time.perf_counter()
            start_metrics = _metric_row(
                helper=yopo,
                base=old_probe,
                renderer=renderer,
                theta=theta,
                segnet=segnet,
                posenet=posenet,
                labels_t=labels_t,
                labels_np=labels_np,
                pose=pose,
            )
            timing.add(
                "repeat_validation" if deterministic_repeat else "exact_validation",
                time.perf_counter() - start_validation,
            )
            teacher_counts["measurement_only_segnet_forwards"] += 2
            teacher_counts["measurement_only_posenet_forwards"] += 1
            trace = [{"step": 0, **start_metrics}]
            schedule = [matched_schedule_record(0, 0.0)]
            first_step = 0
            total_operational = 0.0
            terminal_floor = False
        if deterministic_repeat and fixed_schedule is None:
            raise ProbeError("deterministic repeat requires the exact branch's fixed schedule")
        target_updates = len(fixed_schedule) - 1 if fixed_schedule is not None else config.window_steps
        for step in range(first_step, target_updates):
            theta = theta.detach().requires_grad_(True)
            timing_frame = yopo._render_chart(renderer, theta)
            forward_started = time.perf_counter()
            with torch.no_grad():
                segnet(timing_frame.permute(0, 3, 1, 2).contiguous())
            timing.add(
                "repeat_exact_forward_only" if deterministic_repeat else "exact_forward_only",
                time.perf_counter() - forward_started,
            )
            teacher_counts["timing_only_segnet_forwards"] += 1
            operational_started = time.perf_counter()
            frame = yopo._render_chart(renderer, theta)
            costate, holder, costate_elapsed = yopo._capture_exact_teacher_costate(
                segnet=segnet, frame_nchw=frame.permute(0, 3, 1, 2), labels=labels_t
            )
            gradient, vjp_elapsed = _renderer_gradient(frame, theta, costate)
            provider_path_elapsed = time.perf_counter() - operational_started
            if deterministic_repeat:
                assert fixed_schedule is not None
                schedule_row = fixed_schedule[step + 1]
                step_norm = schedule_row["exact_step_norm"]
                candidate_started = time.perf_counter()
                candidate = candidate_at_common_norm(theta, gradient, step_norm)
                candidate_elapsed = time.perf_counter() - candidate_started
                candidate_metrics: dict[str, float] | None = None
            else:
                line_search_started = time.perf_counter()
                fraction = config.step_fraction
                line_search_trials: list[dict[str, Any]] = []
                candidate_metrics = None
                current_metrics = trace[-1]
                candidate = None
                step_norm = None
                gradient_is_exactly_zero = bool(torch.count_nonzero(gradient.detach()).item() == 0)
                if gradient_is_exactly_zero:
                    terminal_floor = True
                    line_search_trials.append(
                        {
                            "accepted": False,
                            "reason": "certified_zero_renderer_gradient",
                            "fraction": fraction,
                        }
                    )
                for _trial_index in range(0 if terminal_floor else torch.finfo(theta.dtype).bits):
                    step_norm = derive_common_step_norm(theta, step_fraction=fraction)
                    candidate = candidate_at_common_norm(theta, gradient, step_norm)
                    if torch.equal(candidate, theta.detach()):
                        line_search_trials.append(
                            {
                                "accepted": False,
                                "reason": "parameter_quantization_exhaustion_blocked",
                                "fraction": fraction,
                            }
                        )
                        candidate = None
                        step_norm = None
                        break
                    candidate_metrics = _metric_row(
                        helper=yopo,
                        base=old_probe,
                        renderer=renderer,
                        theta=candidate,
                        segnet=segnet,
                        posenet=posenet,
                        labels_t=labels_t,
                        labels_np=labels_np,
                        pose=pose,
                    )
                    teacher_counts["schedule_derivation_segnet_forwards"] += 2
                    teacher_counts["schedule_derivation_posenet_forwards"] += 1
                    predicates = joint_descent_predicates(current_metrics, candidate_metrics)
                    accepted = all(predicates.values())
                    line_search_trials.append(
                        {
                            "accepted": accepted,
                            "fraction": fraction,
                            "step_norm": step_norm,
                            "metrics": candidate_metrics,
                            "predicates": predicates,
                        }
                    )
                    if accepted:
                        break
                    fraction *= 0.5
                else:
                    if not terminal_floor:
                        line_search_trials.append(
                            {
                                "accepted": False,
                                "reason": "derived_trial_limit_exhausted_blocked",
                                "trial_limit": torch.finfo(theta.dtype).bits,
                            }
                        )
                        candidate = None
                        step_norm = None
                line_search_elapsed = time.perf_counter() - line_search_started
                timing.add("schedule_derivation_line_search", line_search_elapsed)
                if candidate is None or step_norm is None:
                    timing.add("line_search_completion_exact_costate", costate_elapsed)
                    teacher_counts["line_search_completion_exact_costates"] += 1
                    break
                candidate_started = time.perf_counter()
                candidate = candidate_at_common_norm(theta, gradient, step_norm)
                candidate_elapsed = time.perf_counter() - candidate_started
            operational_elapsed = complete_operational_step_seconds(
                provider_path_elapsed, candidate_elapsed
            )
            total_operational += operational_elapsed
            timing.add("candidate_update_exact_control", candidate_elapsed)
            if deterministic_repeat:
                timing.add("repeat_exact_costate_forward_backward", costate_elapsed)
                timing.add("repeat_window_operational_step", operational_elapsed)
                teacher_counts["deterministic_repeat_exact_costates"] += 1
            else:
                timing.add("exact_costate_forward_backward", costate_elapsed)
                timing.add("renderer_vjp_exact_control", vjp_elapsed)
                timing.add("exact_window_operational_step", operational_elapsed)
                teacher_counts["deployment_exact_branch_costates"] += 1
            if deterministic_repeat:
                validation_started = time.perf_counter()
                metrics = _metric_row(
                    helper=yopo,
                    base=old_probe,
                    renderer=renderer,
                    theta=candidate,
                    segnet=segnet,
                    posenet=posenet,
                    labels_t=labels_t,
                    labels_np=labels_np,
                    pose=pose,
                )
                timing.add("repeat_validation", time.perf_counter() - validation_started)
                teacher_counts["measurement_only_segnet_forwards"] += 2
                teacher_counts["measurement_only_posenet_forwards"] += 1
            else:
                if candidate_metrics is None:
                    raise ProbeError("joint exact-control line search accepted without exact metrics")
                metrics = candidate_metrics
            schedule.append(matched_schedule_record(step + 1, step_norm))
            trace.append({"step": step + 1, **metrics})
            theta = candidate
            checkpoint = build_probe_checkpoint(
                run_contract_sha256=run_contract["sha256"],
                stage=stage,
                next_step=step + 1,
                anchor_frame=anchor_frame,
                anchor_costate=anchor_costate,
                state=deployment_state(
                    branch_theta=theta,
                    branch_trace=trace,
                    branch_schedule=schedule,
                    branch_next=step + 1,
                    branch_operational_seconds=total_operational,
                    exact_trace=exact_trace if deterministic_repeat else trace,
                    exact_schedule=schedule,
                    exact_window_seconds=exact_window_seconds if deterministic_repeat else total_operational,
                    exact_terminal_floor=terminal_floor,
                ),
            )
            receipt["latest_checkpoint"] = store.save_slot(checkpoint, stage=stage, sequence=step).to_dict()
            receipt["teacher_accounting"] = teacher_counts
            receipt["timing"] = timing.summary()
            _atomic_json(receipt_path, receipt)
        return trace, schedule, total_operational, terminal_floor

    if resume_stage in {"repeat_window", "surrogate_window"}:
        exact_trace = list(resume_state["exact_trace"])
        schedule = list(resume_state["exact_schedule"])
        exact_window_seconds = float(resume_state["exact_window_seconds"])
    else:
        exact_trace, schedule, exact_window_seconds, exact_terminal_floor = exact_window(
            deterministic_repeat=False,
            resume=resume_state if resume_stage == "exact_window" else None,
            fixed_schedule=None,
        )
    if resume_stage in {"repeat_window", "surrogate_window"}:
        exact_terminal_floor = bool(resume_state.get("exact_terminal_floor", False))
    exact_stage_state = deployment_state(
        exact_trace=exact_trace,
        exact_schedule=schedule,
        exact_window_seconds=exact_window_seconds,
        exact_terminal_floor=exact_terminal_floor,
    )
    exact_stage_checkpoint = build_probe_checkpoint(
        run_contract_sha256=run_contract["sha256"],
        stage="exact_window",
        next_step=len(schedule) - 1,
        anchor_frame=anchor_frame,
        anchor_costate=anchor_costate,
        state=exact_stage_state,
    )
    if "exact_stage_checkpoint" not in receipt:
        receipt["exact_stage_checkpoint"] = store.preserve_stage(
            exact_stage_checkpoint, stage="exact_window", sequence=len(schedule) - 1
        ).to_dict()
    # Deterministic repeat is measurement-only and excluded from deployment economics.
    if resume_stage == "surrogate_window":
        repeat_trace = list(resume_state["repeat_trace"])
        repeat_schedule = list(resume_state["repeat_schedule"])
        repeat_seconds = float(resume_state["repeat_window_seconds"])
    else:
        if resume_stage == "repeat_window":
            resume_state["exact_trace"] = exact_trace
            resume_state["exact_schedule"] = schedule
            resume_state["exact_window_seconds"] = exact_window_seconds
        repeat_trace, repeat_schedule, repeat_seconds, _repeat_floor = exact_window(
            deterministic_repeat=True,
            resume=resume_state if resume_stage == "repeat_window" else None,
            fixed_schedule=schedule,
        )
    if schedule != repeat_schedule:
        raise ProbeError("deterministic exact repeat selected a different norm schedule")
    repeat_stage_state = deployment_state(
        exact_trace=exact_trace,
        exact_schedule=schedule,
        exact_window_seconds=exact_window_seconds,
        repeat_trace=repeat_trace,
        repeat_schedule=repeat_schedule,
        repeat_window_seconds=repeat_seconds,
        exact_terminal_floor=exact_terminal_floor,
    )
    repeat_stage_checkpoint = build_probe_checkpoint(
        run_contract_sha256=run_contract["sha256"],
        stage="repeat_window",
        next_step=len(repeat_schedule) - 1,
        anchor_frame=anchor_frame,
        anchor_costate=anchor_costate,
        state=repeat_stage_state,
    )
    if "repeat_stage_checkpoint" not in receipt:
        receipt["repeat_stage_checkpoint"] = store.preserve_stage(
            repeat_stage_checkpoint, stage="repeat_window", sequence=len(repeat_schedule) - 1
        ).to_dict()

    def surrogate_window(
        *, resume: dict[str, Any] | None
    ) -> tuple[list[dict[str, Any]], float, torch.Tensor, torch.Tensor]:
        provider.load_state_dict(provider_state, strict=True)
        if resume is not None:
            theta = resume["branch_theta"]
            local_anchor_frame = anchor_frame.clone()
            local_anchor_costate = anchor_costate.clone()
            trace = list(resume["branch_trace"])
            first_step = resume["branch_next"]
            total_operational = float(resume["branch_operational_seconds"])
        else:
            theta = theta_collection.detach().clone()
            local_anchor_frame = anchor_frame.clone()
            local_anchor_costate = anchor_costate.clone()
            start_validation = time.perf_counter()
            start_metrics = _metric_row(
                helper=yopo,
                base=old_probe,
                renderer=renderer,
                theta=theta,
                segnet=segnet,
                posenet=posenet,
                labels_t=labels_t,
                labels_np=labels_np,
                pose=pose,
            )
            timing.add("surrogate_validation", time.perf_counter() - start_validation)
            teacher_counts["measurement_only_segnet_forwards"] += 2
            teacher_counts["measurement_only_posenet_forwards"] += 1
            trace = [{"step": 0, "exact_anchor": False, **start_metrics}]
            first_step = 0
            total_operational = 0.0
        for step in range(first_step, len(schedule) - 1):
            schedule_row = schedule[step + 1]
            theta = theta.detach().requires_grad_(True)
            # Identical excluded warm-up treatment keeps arm cache state symmetric.
            timing_frame = yopo._render_chart(renderer, theta)
            warmup_started = time.perf_counter()
            with torch.no_grad():
                segnet(timing_frame.permute(0, 3, 1, 2).contiguous())
            timing.add("surrogate_matched_warmup_exact_forward", time.perf_counter() - warmup_started)
            teacher_counts["timing_only_segnet_forwards"] += 1
            operational_started = time.perf_counter()
            frame = yopo._render_chart(renderer, theta)
            frame_nchw = frame.permute(0, 3, 1, 2)
            refresh = step % config.anchor_cadence == 0
            if refresh:
                costate, _holder, exact_elapsed = yopo._capture_exact_teacher_costate(
                    segnet=segnet, frame_nchw=frame_nchw, labels=labels_t
                )
                local_anchor_frame = frame_nchw.detach().clone()
                local_anchor_costate = costate.detach().clone()
                teacher_counts["deployment_surrogate_branch_anchors"] += 1
                timing.add("surrogate_anchor_exact_costate", exact_elapsed)
            else:
                inference_started = time.perf_counter()
                costate = predict_ema_detached_costate(
                    provider,
                    current_frame=frame_nchw.detach(),
                    anchor_frame=local_anchor_frame,
                    anchor_costate=local_anchor_costate,
                )
                timing.add("surrogate_inference", time.perf_counter() - inference_started)
            gradient, vjp_elapsed = _renderer_gradient(frame, theta, costate)
            provider_path_elapsed = time.perf_counter() - operational_started
            candidate_started = time.perf_counter()
            candidate = candidate_at_common_norm(theta, gradient, schedule_row["exact_step_norm"])
            candidate_elapsed = time.perf_counter() - candidate_started
            operational_elapsed = complete_operational_step_seconds(
                provider_path_elapsed, candidate_elapsed
            )
            total_operational += operational_elapsed
            timing.add("renderer_vjp_surrogate_target", vjp_elapsed)
            timing.add("candidate_update_surrogate_target", candidate_elapsed)
            timing.add("surrogate_window_operational_step", operational_elapsed)
            timing.add(
                "surrogate_anchor_operational_step" if refresh else "surrogate_nonanchor_operational_step",
                operational_elapsed,
            )
            validation_started = time.perf_counter()
            metrics = _metric_row(
                helper=yopo,
                base=old_probe,
                renderer=renderer,
                theta=candidate,
                segnet=segnet,
                posenet=posenet,
                labels_t=labels_t,
                labels_np=labels_np,
                pose=pose,
            )
            timing.add("surrogate_validation", time.perf_counter() - validation_started)
            teacher_counts["measurement_only_segnet_forwards"] += 2
            teacher_counts["measurement_only_posenet_forwards"] += 1
            trace.append({"step": step + 1, "exact_anchor": refresh, **metrics})
            theta = candidate
            checkpoint = build_probe_checkpoint(
                run_contract_sha256=run_contract["sha256"],
                stage="surrogate_window",
                next_step=step + 1,
                anchor_frame=local_anchor_frame,
                anchor_costate=local_anchor_costate,
                state=deployment_state(
                    exact_trace=exact_trace,
                    exact_schedule=schedule,
                    exact_window_seconds=exact_window_seconds,
                    repeat_trace=repeat_trace,
                    repeat_schedule=repeat_schedule,
                    repeat_window_seconds=repeat_seconds,
                    branch_theta=theta,
                    branch_trace=trace,
                    branch_next=step + 1,
                    branch_operational_seconds=total_operational,
                ),
            )
            receipt["latest_checkpoint"] = store.save_slot(
                checkpoint, stage="surrogate_window", sequence=step
            ).to_dict()
            receipt["teacher_accounting"] = teacher_counts
            receipt["timing"] = timing.summary()
            _atomic_json(receipt_path, receipt)
        return trace, total_operational, local_anchor_frame, local_anchor_costate

    surrogate_trace, surrogate_window_seconds, final_anchor_frame, final_anchor_costate = surrogate_window(
        resume=resume_state if resume_stage == "surrogate_window" else None
    )

    exact_ce_strict_descent = all(
        exact_trace[index + 1]["ce"] < exact_trace[index]["ce"] for index in range(len(exact_trace) - 1)
    )
    exact_dseg_nonworsening = exact_trace[-1]["d_seg"] <= exact_trace[0]["d_seg"]
    exact_dpose_nonworsening = exact_trace[-1]["d_pose"] <= exact_trace[0]["d_pose"]
    exact_control_valid = exact_ce_strict_descent and exact_dseg_nonworsening and exact_dpose_nonworsening
    verdict, canonical_schedule, canonical_noise = _canonical_matched_verdict(
        regime=config.regime,
        schedule_rows=schedule,
        exact_trace=exact_trace,
        surrogate_trace=surrogate_trace,
        repeat_trace=repeat_trace,
        exact_control_valid=exact_control_valid,
        exact_terminal_floor=exact_terminal_floor and exact_control_valid,
        ema_provider_admitted=ema_provider_admitted,
    )
    canonical_verdict = verdict["verdict"]
    mission_verdict = (
        "NO-GO"
        if canonical_verdict == "NO-GO"
        else "NEEDS-MORE"
        if len(schedule) - 1 < config.anchor_cadence
        else canonical_verdict
    )
    mission_verdict_reason = (
        verdict["reason"]
        if mission_verdict == canonical_verdict
        else "the bounded matched window does not validate fidelity through the full K20 anchor interval"
    )
    exact_step_seconds = timing.measured_seconds["exact_window_operational_step"]
    surrogate_step_seconds = timing.measured_seconds["surrogate_nonanchor_operational_step"]
    t_exact = float(np.mean(exact_step_seconds)) if exact_step_seconds else None
    t_surrogate = float(np.mean(surrogate_step_seconds)) if surrogate_step_seconds else None
    projected_k20_speedup = (
        None
        if t_exact is None or t_surrogate is None
        else config.anchor_cadence * t_exact / (t_exact + (config.anchor_cadence - 1) * t_surrogate)
    )
    observed_updates = len(schedule) - 1
    speedup = (
        None
        if exact_window_seconds <= 0.0 or surrogate_window_seconds <= 0.0
        else exact_window_seconds / surrogate_window_seconds
    )
    isolated_timings = (
        aggregate_isolated_timings(
            common_step_schedule=canonical_schedule,
            exact_schedule_sha256=canonical_schedule.sha256,
            surrogate_schedule_sha256=canonical_schedule.sha256,
            exact_forward_only=timing.measured_seconds["exact_forward_only"],
            exact_costate_forward_backward=timing.measured_seconds["exact_costate_forward_backward"],
            anchor_fit=timing.measured_seconds["anchor_fit"],
            surrogate_inference=timing.measured_seconds["surrogate_inference"],
            renderer_vjp_exact_control=timing.measured_seconds["renderer_vjp_exact_control"],
            renderer_vjp_surrogate_target=timing.measured_seconds["renderer_vjp_surrogate_target"],
            whole_matched_window_exact_control=(exact_window_seconds,),
            whole_matched_window_surrogate_target=(surrogate_window_seconds,),
        )
        if all(
            (
                timing.measured_seconds["exact_forward_only"],
                timing.measured_seconds["exact_costate_forward_backward"],
                timing.measured_seconds["anchor_fit"],
                timing.measured_seconds["surrogate_inference"],
                timing.measured_seconds["renderer_vjp_exact_control"],
                timing.measured_seconds["renderer_vjp_surrogate_target"],
            )
        )
        and exact_window_seconds > 0.0
        and surrogate_window_seconds > 0.0
        else {
            "status": "BLOCKED",
            "reason": "measured prefix left one or more isolated matched timing surfaces empty",
            "score_claim": False,
        }
    )
    measured_forward_mean_seconds = (
        float(np.mean(timing.measured_seconds["exact_forward_only"]))
        if timing.measured_seconds["exact_forward_only"]
        else None
    )
    window_economics = {
        "exact_window_operational_seconds": exact_window_seconds,
        "surrogate_window_operational_seconds": surrogate_window_seconds,
        "speedup": speedup,
        "saved_fraction": None if speedup is None else 1.0 - 1.0 / speedup,
        "provider_replacement_includes_exact_forward_and_backward": True,
        "teacher_forward_replacement_only": False,
        "comparison_basis": (
            "sums of symmetric complete per-step operational timers under one exact-derived common norm "
            "schedule; each includes render, provider, renderer VJP, and candidate update; line-search and "
            "exact validation calls excluded"
        ),
        "target_cadence": config.anchor_cadence,
        "observed_window_steps": observed_updates,
        "observed_exact_teacher_skip_fraction": (None if observed_updates == 0 else 1.0 - 1.0 / observed_updates),
        "target_exact_teacher_skip_fraction": config.target_teacher_skip_fraction,
        "target_cadence_fidelity_validated": observed_updates >= config.anchor_cadence,
        "projected_k20_speedup_from_measured_step_means": projected_k20_speedup,
        "projection_formula": "K*t_exact/(t_exact+(K-1)*t_surrogate)",
        "projection_admission_authority": False,
        "validation_excluded": True,
        "isolated_timings": isolated_timings,
        "operator_reference": {
            "exact_forward_ms": OPERATOR_REFERENCE_MS,
            "provenance": OPERATOR_REFERENCE_PROVENANCE,
            "measured_current_probe_exact_forward_mean_ms": None
            if measured_forward_mean_seconds is None
            else measured_forward_mean_seconds * 1000.0,
            "included_in_measured_samples": False,
        },
    }
    terminal_state = deployment_state(
        exact_schedule=schedule,
        exact_window_seconds=exact_window_seconds,
        exact_trace=exact_trace,
        surrogate_trace=surrogate_trace,
        repeat_trace=repeat_trace,
        repeat_schedule=repeat_schedule,
        repeat_window_seconds=repeat_seconds,
    )
    terminal_checkpoint = build_probe_checkpoint(
        run_contract_sha256=run_contract["sha256"],
        stage="surrogate_window",
        next_step=observed_updates,
        anchor_frame=final_anchor_frame,
        anchor_costate=final_anchor_costate,
        state=terminal_state,
    )
    if "surrogate_stage_checkpoint" not in receipt:
        receipt["surrogate_stage_checkpoint"] = store.preserve_stage(
            terminal_checkpoint, stage="surrogate_window", sequence=observed_updates
        ).to_dict()
    expected_segnet_forwards = sum(
        teacher_counts[key]
        for key in (
            "collection_exact_costate_labels",
            "deployment_exact_branch_costates",
            "deployment_surrogate_branch_anchors",
            "deterministic_repeat_exact_costates",
            "measurement_only_segnet_forwards",
            "timing_only_segnet_forwards",
            "schedule_derivation_segnet_forwards",
            "line_search_completion_exact_costates",
        )
    )
    teacher_counts["expected_total_segnet_forwards_from_categories"] = expected_segnet_forwards
    teacher_counts["expected_total_posenet_forwards_from_categories"] = teacher_counts[
        "measurement_only_posenet_forwards"
    ] + teacher_counts["schedule_derivation_posenet_forwards"]
    teacher_counts["segnet_forward_reconciliation"] = (
        "PASS" if teacher_counts["observed_total_segnet_forwards"] == expected_segnet_forwards else "FAIL"
    )
    teacher_counts["posenet_forward_reconciliation"] = (
        "PASS"
        if teacher_counts["observed_total_posenet_forwards"]
        == teacher_counts["expected_total_posenet_forwards_from_categories"]
        else "FAIL"
    )
    if (
        teacher_counts["segnet_forward_reconciliation"] != "PASS"
        or teacher_counts["posenet_forward_reconciliation"] != "PASS"
    ):
        raise ProbeError("hook-observed scorer calls do not reconcile with operational/measurement categories")
    receipt.update(
        {
            "status": "MEASURED",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "renderer_parity": parity,
            "common_step_schedule": {
                "rows": schedule,
                "sha256": canonical_schedule.sha256,
                "control_name": canonical_schedule.control_name,
                "derivation": canonical_schedule.derivation,
            },
            "exact_trace": exact_trace,
            "surrogate_trace": surrogate_trace,
            "deterministic_repeat_trace": repeat_trace,
            "deterministic_repeat_noise_floor": verdict["deterministic_repeat_noise_floor"],
            "fidelity_verdict": verdict,
            "ema_provider_admitted": ema_provider_admitted,
            "exact_control": {
                "valid": exact_control_valid,
                "strict_ce_descent": exact_ce_strict_descent,
                "d_seg_nonworsening": exact_dseg_nonworsening,
                "d_pose_nonworsening": exact_dpose_nonworsening,
                "valid_terminal_floor": exact_terminal_floor and exact_control_valid,
                "completion_reason": classify_exact_window_completion(
                    certified_zero_renderer_gradient=exact_terminal_floor,
                    observed_updates=observed_updates,
                    requested_updates=config.window_steps,
                ),
            },
            "mission_verdict": mission_verdict,
            "mission_verdict_reason": mission_verdict_reason,
            "window_economics": window_economics,
            "timing": timing.summary(),
            "pointer_delta": "none",
            "false_authority_flags": {
                "score_claim": False,
                "mps_authority": False,
                "surrogate_eval_authority": False,
                "contest_cpu_or_cuda_eval": False,
            },
        }
    )
    tree_bytes = sum(path.stat().st_size for path in args.output_dir.rglob("*") if path.is_file())
    receipt["disk_hygiene"].update(
        {
            "result_tree_bytes_before_final_receipt": tree_bytes,
            "large_artifacts_created": tree_bytes >= LARGE_ARTIFACT_THRESHOLD_BYTES,
            "bytes_deleted_or_moved": 0,
            "retention_reason": "source-custodied resumable research evidence",
        }
    )
    _atomic_json(receipt_path, receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--storage-plan", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--regime", choices=tuple(REGIMES), default=DEFAULT_REGIME)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--collection-steps", type=int, default=DEFAULT_COLLECTION_STEPS)
    parser.add_argument("--window-steps", type=int, default=DEFAULT_WINDOW_STEPS)
    parser.add_argument("--optimizer-steps-per-label", type=int, default=DEFAULT_OPTIMIZER_STEPS)
    parser.add_argument("--step-fraction", type=float, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--ema-decay", type=float, default=DEFAULT_EMA_DECAY)
    parser.add_argument("--hidden-channels", type=int, default=DEFAULT_HIDDEN_CHANNELS)
    parser.add_argument("--branch-kernel-sizes", type=int, nargs="+", default=DEFAULT_BRANCH_KERNELS)
    parser.add_argument("--target-teacher-skip-fraction", type=float, default=TARGET_TEACHER_SKIP_FRACTION)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = run(args)
    print(
        json.dumps(
            {
                "receipt": str(args.output_dir / "measurement_receipt.json"),
                "fidelity": receipt["fidelity_verdict"]["verdict"],
                "speedup": receipt["window_economics"]["speedup"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
