#!/usr/bin/env python3
"""Launch-admissible PR130 QAT trainer with EMA and full-state resumption.

borrowed_substrate_accounting:
  source_repo: /Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo
  source_head: e34f31bc4969042c0051ac81aa3c56884419a231
  source_path: code/train_semantic_quantized.py
  source_sha256: 4bcaf8a5c581c1e5eb057ea0ef760f269e1eabfdd2fca926bcbf61f4163a248d
  theirs: renderer/QAT forward, curriculum, optimizer ordering, selection key,
    exact-R evaluation, and packed-size accounting.
  ours: canonical tac.training.EMA integration, LawRef-resolved decay,
    snapshot/restore evaluation, deployed pack/parse argmax gate, atomic
    complete checkpoints, stage-boundary preservation, and --resume-from.

This wrapper deliberately leaves the reconstructive ``lifted/`` custody copy
untouched.  It imports and executes that copy's scientific primitives while
owning the operational mechanics that PR130 did not ship.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from tac.admission_guard import assert_governed_admission
from tac.pr130_lift.band_objective import (
    band_weight_field,
    band_weight_stats,
    band_weight_table_sha256,
    curriculum_loss_weighted,
    label_boundary,
    load_band_weight_table,
)
from tac.pr130_lift.checkpoint_schema import (
    architecture_config_from_checkpoint,
    build_semantic_checkpoint_metadata,
)
from tac.pr130_lift.editability_levers import (
    DEFAULT_FILM_CRITICAL_MULTIPLIER,
    EditabilityLeverConfig,
    EditabilityLevers,
)
from tac.pr130_lift.pose.source_loader import load_lifted_module
from tac.training import EMA
from tac.witness_dsl.lawref import (
    LADDER_DERIVED_AT_CONFIG,
    InputRef,
    LawRef,
    lawref_to_declaration,
    resolve,
)

N_TOTAL_PAIRS = 600
EMA_LAW_REF = "ema_decay_run_geometry_v1"
EMA_TARGET_SEED_FRACTION = 0.01
LEGACY_EMA_DECAY_FALLBACK = 0.997
RESUME_SCHEMA = "tac.pr130.semantic_qat_resume.v1"
REPO_ROOT = Path(__file__).resolve().parents[3]
LIFTED_DIR = Path(__file__).resolve().parent / "lifted"


def _jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _load_lifted_qat() -> ModuleType:
    path = LIFTED_DIR / "train_semantic_quantized.py"
    spec = importlib.util.spec_from_file_location(
        "tac.pr130_lift.dynamic.train_semantic_quantized", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load lifted QAT trainer at {path}")
    module = importlib.util.module_from_spec(spec)
    previous = list(sys.path)
    if str(LIFTED_DIR) not in sys.path:
        sys.path.insert(0, str(LIFTED_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = previous
    return module


def resolve_ema_policy(
    updates_per_run: int | None,
    *,
    target_seed_fraction: float = EMA_TARGET_SEED_FRACTION,
) -> dict[str, Any]:
    """Resolve target EMA decay from run geometry, with a genuine legacy fallback."""

    if updates_per_run is None:
        return {
            "equation_id": EMA_LAW_REF,
            "decay": LEGACY_EMA_DECAY_FALLBACK,
            "fallback_used": True,
            "fallback_reason": "updates_per_run absent in legacy caller",
            "warmup": True,
        }
    updates = int(updates_per_run)
    if updates <= 0:
        raise ValueError("updates_per_run must be positive")
    if not 0.0 < target_seed_fraction < 1.0:
        raise ValueError("target_seed_fraction must be in (0,1)")
    ref = LawRef(
        equation_id=EMA_LAW_REF,
        inputs={
            "mode": InputRef.literal(
                1, "mode 1 = decay_from_seed_fraction"
            ),
            "updates_per_run": InputRef.literal(
                updates, "QAT run geometry: one EMA update after each optimizer step"
            ),
            "target_seed_fraction": InputRef.literal(
                float(target_seed_fraction),
                "canonical EmaDecayCalibrated terminal seed-fraction default",
            ),
        },
        ladder_class=LADDER_DERIVED_AT_CONFIG,
    )
    resolved = resolve(ref, repo_root=REPO_ROOT)
    decay = float(resolved.value)
    if not 0.0 < decay < 1.0:
        raise ValueError(f"LawRef resolved invalid EMA decay {decay}")
    manifest = resolved.to_dict()
    # Wall-clock resolution time is observability, not a causal config input.
    manifest.pop("resolved_at", None)
    manifest.pop("resolved_at_utc", None)
    return {
        "equation_id": EMA_LAW_REF,
        "decay": decay,
        "fallback_used": False,
        "lawref_declaration": lawref_to_declaration(ref),
        "resolved_manifest": manifest,
        "warmup": True,
        **_ema_warmup_dominance(decay, updates),
    }


#: Above this ``target_seed_fraction`` the warmup ramp dominates the resolved
#: decay at EVERY run length.  Derived, not measured: the LawRef resolves
#: ``decay = f**(1/N)``, so ``1 - decay ~ ln(1/f)/N`` and the warmup crossover
#: ``t* = (10*decay - 1)/(1 - decay) ~ 9N/ln(1/f)``.  Then ``N < t*`` iff
#: ``ln(1/f) < 9`` iff ``f > exp(-9)`` -- independent of N.
EMA_WARMUP_DOMINANCE_SEED_FRACTION_THRESHOLD = 1.2340980408667956e-04  # exp(-9)


def _ema_warmup_dominance(decay: float, updates: int) -> dict[str, Any]:
    """Say whether the resolved decay ever actually governs the run.

    ``EMA.effective_decay()`` returns ``min(decay, (1 + t) / (10 + t))``.  The
    warmup branch only reaches the target at ``t = (10 * decay - 1) / (1 - decay)``.
    For a 600-step run at decay 0.99235 that crossover is **t = 1,167**, so the
    target decay is applied on **no step at all** and the realised seed retention
    is ~3.3e-19, not the declared 0.01 -- seventeen orders of magnitude apart.

    ``ddm_av3`` F7 recorded this as an instance ("inert for any run shorter than
    ~1,167 steps").  It is STRUCTURAL: because the LawRef derives the decay FROM
    the run geometry as ``f**(1/N)``, the crossover scales as ``9N/ln(1/f)``,
    i.e. **~1.954 x N at the canonical f = 0.01, for every N**.  At this seed
    fraction the target decay is therefore never applied at ANY run length --
    the warmup ramp is the whole policy.  It would govern only for
    ``f < exp(-9)`` (see :data:`EMA_WARMUP_DOMINANCE_SEED_FRACTION_THRESHOLD`).

    This is NOT the anti-freeze bug; the warmup ramp is the deliberate cure and
    ``ddm_av3`` F1 measured it working (end lag 0.006-0.010 in all four arms).
    It is a PROVENANCE honesty defect: the manifest recorded a
    ``derived_at_config`` decay as governing a run it never governed.  These
    derived fields are observability, not causal inputs, so they are excluded
    from the resume equality check by
    :data:`EMA_POLICY_DERIVED_OBSERVABILITY_KEYS`.
    """

    crossover = (10.0 * decay - 1.0) / (1.0 - decay) if decay < 1.0 else float("inf")
    dominates = updates < crossover
    # Realised retention of the seed weight under the warmup ramp:
    # prod_{t=0}^{updates-1} min(decay, (1 + t) / (10 + t)).
    retention = 1.0
    for step in range(int(updates)):
        retention *= min(decay, (1.0 + step) / (10.0 + step))
    return {
        "warmup_dominates_target": bool(dominates),
        "warmup_target_crossover_updates": crossover,
        "warmup_crossover_over_updates": crossover / max(updates, 1),
        "declared_target_seed_fraction": decay**updates,
        "realized_seed_retention_under_warmup": retention,
        "governing_policy": "warmup_ramp" if dominates else "target_decay",
    }


#: Fields of ``ema_policy`` that are DERIVED observability, never causal inputs.
#: Excluded from the resume equality check so that (a) a checkpoint written
#: before they existed still resumes, and (b) a checkpoint written after them
#: does not refuse against an older reader.  This cannot weaken the guard: every
#: one is a pure function of ``decay`` (still compared, inside ``ema_policy``)
#: and ``updates_per_run`` (still compared, as ``steps``, in the producing
#: config), so an actual policy change is still caught by the surviving keys.
EMA_POLICY_DERIVED_OBSERVABILITY_KEYS: frozenset[str] = frozenset(
    {
        "warmup_dominates_target",
        "warmup_target_crossover_updates",
        "warmup_crossover_over_updates",
        "declared_target_seed_fraction",
        "realized_seed_retention_under_warmup",
        "governing_policy",
    }
)


def _causal_ema_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    """``policy`` minus the derived-observability fields."""

    return {
        key: value
        for key, value in policy.items()
        if key not in EMA_POLICY_DERIVED_OBSERVABILITY_KEYS
    }


@contextmanager
def ema_eval_scope(model: torch.nn.Module, ema: EMA) -> Iterator[None]:
    """Apply the EMA shadow only for evaluation and restore live weights exactly."""

    original = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    was_training = model.training
    ema.apply(model)
    model.eval()
    try:
        yield
    finally:
        model.load_state_dict(original)
        model.train(was_training)


def _ema_training_state(ema: EMA) -> dict[str, Any]:
    return {
        "decay": float(ema.decay),
        "warmup": bool(ema.warmup),
        "num_updates": int(ema._num_updates),  # canonical class state, no public codec
        "shadow": {
            key: value.detach().cpu().clone()
            for key, value in ema.state_dict().items()
        },
    }


def _restore_ema_training_state(
    ema: EMA,
    payload: Mapping[str, Any],
    *,
    device: torch.device,
) -> None:
    if float(payload["decay"]) != float(ema.decay):
        raise ValueError("resume EMA decay differs from the resolved run policy")
    if bool(payload["warmup"]) != bool(ema.warmup):
        raise ValueError("resume EMA warmup policy differs from the current run")
    shadow = payload.get("shadow")
    if not isinstance(shadow, Mapping) or set(shadow) != set(ema.shadow):
        raise ValueError("resume EMA shadow key set differs from the model")
    ema.shadow = {
        key: value.detach().to(device).clone() for key, value in shadow.items()
    }
    ema._num_updates = int(payload["num_updates"])


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(dict(payload), temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode()
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _phase_for_step(args: argparse.Namespace, step: int) -> str:
    if not 1 <= step <= args.steps:
        raise ValueError("step outside run horizon")
    if step <= args.float_warmup_steps:
        return "float_ce"
    qat_steps = args.steps - args.float_warmup_steps
    qat_index = step - args.float_warmup_steps - 1
    progress = qat_index / max(qat_steps - 1, 1)
    if progress < args.ce_fraction:
        return "ce"
    if progress < args.softplus_fraction:
        return "softplus_margin"
    return "expected_flip"


def phase_end_steps(args: argparse.Namespace) -> dict[int, str]:
    """Return every actual curriculum boundary, including the terminal stage."""

    ends: dict[int, str] = {}
    previous = _phase_for_step(args, 1)
    for step in range(2, args.steps + 1):
        current = _phase_for_step(args, step)
        if current != previous:
            ends[step - 1] = previous
        previous = current
    ends[args.steps] = previous
    return ends


def _stable_training_config(args: argparse.Namespace) -> dict[str, Any]:
    non_causal = {
        "out",
        "save",
        "resume_from",
        "checkpoint_every",
        "parity_pairs",
    }
    return {
        key: value
        for key, value in _jsonable_args(args).items()
        if key not in non_causal
    }


#: Config keys introduced by the B2E editability levers, with the INERT value
#: that reproduces the pre-lever trainer exactly.  A resume checkpoint written
#: before these flags existed has no such keys; per the canonical
#: additive-legacy-compatible resume law, that checkpoint may still resume IFF
#: every absent key currently sits at its inert value.  If a lever is ON, the run
#: genuinely differs from the parent and the guard MUST refuse.
ADDITIVE_LEVER_CONFIG_DEFAULTS: dict[str, Any] = {
    "weight_perturb_robustness": 0.0,
    "weight_perturb_shape": "quantization",
    "film_critical_multiplier": DEFAULT_FILM_CRITICAL_MULTIPLIER,
    "weight_qat_q3q4": False,
    "film_row_dropout": 0.0,
    "film_row_dropout_protect_top": 0,
    "carrier_rank_penalty": 0.0,
    "carrier_tensors": [],
    "lever_seed": 20260816,
    # RG1 band objective: inert at alpha 0, and the table sha is recorded only
    # when the term is active, so a pre-RG1 checkpoint resumes into an off run.
    "band_objective_weight": 0.0,
    "band_weight_table_sha256": None,
}


def _reconcile_additive_resume_config(
    prior_config: dict[str, Any], current_config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Make a pre-lever resume checkpoint comparable to a lever-aware invocation.

    Returns the two configs with legitimately-absent additive keys removed from
    BOTH sides.  A key is legitimately absent only when the checkpoint predates
    it AND the current run leaves it inert; otherwise it is left in place so the
    caller's equality check refuses, loudly and for the right reason.

    This EXTENDS the guard rather than weakening it: an unknown missing key, a
    key that disappeared, or an active lever against a pre-lever parent all still
    refuse.
    """
    prior = dict(prior_config)
    current = dict(current_config)
    for key, inert in ADDITIVE_LEVER_CONFIG_DEFAULTS.items():
        if key in prior:
            continue  # checkpoint knows the key; compare it normally.
        if key not in current:
            continue  # neither side has it; nothing to reconcile.
        if current[key] == inert:
            # Pre-lever parent + inert lever == byte-identical training.
            del current[key]
    return prior, current


def _artifact_manifest(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    paths = {
        "init": args.init,
        "input_cache": args.input_cache or args.cache,
        "target_cache": args.target_cache or args.cache,
        "master_cache": args.master_cache,
    }
    digest_cache: dict[Path, tuple[int, str]] = {}
    manifest: dict[str, dict[str, Any]] = {}
    for role, path in paths.items():
        if path is None:
            continue
        resolved = Path(path).resolve()
        if resolved not in digest_cache:
            digest_cache[resolved] = (resolved.stat().st_size, _sha256_file(resolved))
        byte_count, digest = digest_cache[resolved]
        manifest[role] = {
            "path": str(resolved),
            "bytes": byte_count,
            "sha256": digest,
        }
    return manifest


def _capture_rng_state(device: torch.device) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "torch_cpu": torch.get_rng_state(),
        "numpy": np.random.get_state(),
    }
    if device.type == "cuda":
        payload["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    elif device.type == "mps" and hasattr(torch.mps, "get_rng_state"):
        payload["torch_mps"] = torch.mps.get_rng_state()
    return payload


def _restore_rng_state(payload: Mapping[str, Any], device: torch.device) -> None:
    torch.set_rng_state(payload["torch_cpu"])
    np.random.set_state(payload["numpy"])
    if device.type == "cuda" and "torch_cuda_all" in payload:
        torch.cuda.set_rng_state_all(payload["torch_cuda_all"])
    elif (
        device.type == "mps"
        and "torch_mps" in payload
        and hasattr(torch.mps, "set_rng_state")
    ):
        torch.mps.set_rng_state(payload["torch_mps"])


def _checkpoint_payload(
    *,
    args: argparse.Namespace,
    architecture_config: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    ema_policy: Mapping[str, Any],
    model: torch.nn.Module,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    generator: torch.Generator,
    order: torch.Tensor,
    cursor: int,
    step: int,
    phase: str,
    checkpoint_kind: str,
    history: list[dict[str, Any]],
    best_key: tuple[Any, ...],
    best_seg: float,
    best_step: int,
    init_seg: float,
    best_rgb: float | None,
    best_state: Mapping[str, torch.Tensor],
    deployment_state: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    producing = {
        **_stable_training_config(args),
        "ema_policy": dict(ema_policy),
        "checkpoint_step": int(step),
        "checkpoint_phase": phase,
        "checkpoint_kind": checkpoint_kind,
    }
    parent = dict(artifacts["init"])
    payload = build_semantic_checkpoint_metadata(
        architecture_config=architecture_config,
        producing_stage_config=producing,
        parent_checkpoint=parent,
    )
    payload.update(
        {
            "resume_schema": RESUME_SCHEMA,
            "state_dict": {
                key: value.detach().cpu().clone()
                for key, value in deployment_state.items()
            },
            "quant_bits": int(args.bits),
            "score_claim": False,
            "deployment_weights": "ema_shadow",
            "training_state": {
                "step": int(step),
                "phase": phase,
                "model_state_dict": {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
                "ema": _ema_training_state(ema),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "generator_state": generator.get_state(),
                "order": order.detach().cpu().clone(),
                "cursor": int(cursor),
                "rng_state": _capture_rng_state(next(model.parameters()).device),
                "history": copy.deepcopy(history),
                "best_key": tuple(best_key),
                "best_seg": float(best_seg),
                "best_step": int(best_step),
                "init_seg": float(init_seg),
                "best_rgb": None if best_rgb is None else float(best_rgb),
                "best_ema_state_dict": {
                    key: value.detach().cpu().clone()
                    for key, value in best_state.items()
                },
            },
            "input_artifacts": copy.deepcopy(dict(artifacts)),
        }
    )
    return payload


def _load_resume_state(
    path: Path,
    *,
    args: argparse.Namespace,
    architecture_config: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    ema_policy: Mapping[str, Any],
    model: torch.nn.Module,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    generator: torch.Generator,
    device: torch.device,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("resume_schema") != RESUME_SCHEMA:
        raise ValueError("--resume-from is not a complete PR130 QAT resume checkpoint")
    restored_architecture = architecture_config_from_checkpoint(
        payload, consumer="semantic_qat_resume"
    )
    if restored_architecture != dict(architecture_config):
        raise ValueError("resume architecture differs from the initialization checkpoint")
    prior_config = dict(payload["producing_stage_config"])
    prior_ema_policy = prior_config.pop("ema_policy")
    for key in ("checkpoint_step", "checkpoint_phase", "checkpoint_kind"):
        prior_config.pop(key, None)
    prior_config, current_config = _reconcile_additive_resume_config(
        prior_config, _stable_training_config(args)
    )
    if prior_config != current_config:
        raise ValueError("resume producing-stage config differs from the current invocation")
    if _causal_ema_policy(prior_ema_policy) != _causal_ema_policy(ema_policy):
        raise ValueError("resume LawRef-resolved EMA policy differs from current geometry")
    if payload.get("input_artifacts") != dict(artifacts):
        raise ValueError("resume input artifact bytes differ from the original run")
    state = payload["training_state"]
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    _restore_ema_training_state(ema, state["ema"], device=device)
    generator.set_state(state["generator_state"])
    _restore_rng_state(state["rng_state"], device)
    return {
        "start_step": int(state["step"]) + 1,
        "order": state["order"].long(),
        "cursor": int(state["cursor"]),
        "history": list(state["history"]),
        "best_key": tuple(state["best_key"]),
        "best_seg": float(state["best_seg"]),
        # Additive + legacy-compatible: a pre-F3-fix checkpoint has neither key.
        # Computed LAZILY -- dict.get(k, default) evaluates its default eagerly,
        # so an inline fallback would read history[0] even when the key exists.
        "best_step": int(state.get("best_step", 0)),
        "init_seg": _restored_init_seg(state),
        "best_rgb": (
            None if state["best_rgb"] is None else float(state["best_rgb"])
        ),
        "best_state": {
            key: value.detach().cpu().clone()
            for key, value in state["best_ema_state_dict"].items()
        },
    }


def _restored_init_seg(state: Mapping[str, Any]) -> float:
    """The step-0 metric of a resumed run, tolerant of pre-F3-fix checkpoints.

    Preference order: the recorded ``init_seg``; else the first history row (the
    step-0 evaluation, which every run writes); else ``best_seg``, which for a
    pre-fix checkpoint IS the seeded step-0 value whenever nothing displaced it.
    """

    if "init_seg" in state:
        return float(state["init_seg"])
    history = state.get("history") or []
    if history and "quantized_exact_seg" in history[0]:
        return float(history[0]["quantized_exact_seg"])
    return float(state["best_seg"])


def _checkpoint_paths(save: Path, *, step: int, phase: str, stage_boundary: bool) -> tuple[Path, Path]:
    role = f"stage-{phase}" if stage_boundary else "periodic"
    preserved = save.with_name(f"{save.stem}.{role}.step{step:06d}.full_state.pt")
    latest = save.with_name(f"{save.stem}.resume.latest.pt")
    return preserved, latest


@torch.no_grad()
def deployed_argmax_parity(
    *,
    qat: ModuleType,
    model: torch.nn.Module,
    segnet: torch.nn.Module,
    conditioning_tokens: torch.Tensor,
    bits: int,
    pair_ids: list[int],
    batch_size: int,
    architecture_config: Mapping[str, Any],
    deployment_state: Mapping[str, torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    """Compare QAT fake-quantized EMA inference to actual pack/parse inference."""

    if not pair_ids:
        raise ValueError("deployed argmax parity requires at least one pair")
    packer = load_lifted_module("pack_semantic_pose")
    checkpoint = {
        "state_dict": {
            key: value.detach().cpu().clone()
            for key, value in deployment_state.items()
        },
        "quant_bits": int(bits),
    }
    semantic_blob, expected = packer.pack_semantic(checkpoint)
    original = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    was_training = model.training
    try:
        model.load_state_dict(deployment_state)
        model.eval()
        parsed = packer.unpack_semantic(
            semantic_blob, dict(architecture_config), model.state_dict()
        )
        if set(parsed) != set(expected):
            raise RuntimeError("deployed semantic pack/parse key set drifted")
        for name in parsed:
            if not torch.equal(parsed[name], expected[name]):
                raise RuntimeError(f"deployed semantic pack/parse tensor drifted: {name}")
        deployed_model = qat.SemanticTokenRenderer(
            width=int(architecture_config["width"]),
            blocks=int(architecture_config["blocks"]),
            frame_dim=int(architecture_config["frame_dim"]),
            num_pairs=int(architecture_config.get("num_pairs", N_TOTAL_PAIRS)),
            num_tokens=int(architecture_config.get("num_tokens", 5)),
            phase_y=int(architecture_config.get("phase_y", 1)),
            phase_x=int(architecture_config.get("phase_x", 1)),
            temporal_radius=int(architecture_config.get("temporal_radius", 0)),
        ).to(device)
        deployed_model.load_state_dict(parsed)
        deployed_model.eval()
        argmax_diff = 0
        total_pixels = 0
        frame_max_abs_delta = 0.0
        frame_equal = True
        qat_nonfinite_values = 0
        deployed_nonfinite_values = 0
        for start in range(0, len(pair_ids), batch_size):
            selected = pair_ids[start : start + batch_size]
            batch_ids_cpu = torch.tensor(selected, dtype=torch.long)
            idx = batch_ids_cpu.to(device)
            conditioning = qat.gather_conditioning(
                conditioning_tokens,
                batch_ids_cpu,
                int(architecture_config.get("temporal_radius", 0)),
            ).to(device)
            qat_frame = qat.render_quantized(
                model, conditioning, idx, bits, exact_path=True
            )
            deployed_frame = qat.render_float(
                deployed_model, conditioning, idx, exact_path=True
            )
            frame_equal = frame_equal and torch.equal(qat_frame, deployed_frame)
            qat_nonfinite_values += int((~torch.isfinite(qat_frame)).sum())
            deployed_nonfinite_values += int((~torch.isfinite(deployed_frame)).sum())
            frame_max_abs_delta = max(
                frame_max_abs_delta,
                float((qat_frame - deployed_frame).abs().max()),
            )
            qat_pred = segnet(qat_frame).argmax(1)
            deployed_pred = segnet(deployed_frame).argmax(1)
            argmax_diff += int((qat_pred != deployed_pred).sum())
            total_pixels += qat_pred.numel()
        return {
            "schema": "tac.pr130.ema_deployed_argmax_parity.v1",
            "pair_ids": list(pair_ids),
            "pairs": len(pair_ids),
            "semantic_blob_bytes": len(semantic_blob),
            "argmax_diff_pixels": argmax_diff,
            "total_pixels": total_pixels,
            "argmax_diff_rate": argmax_diff / total_pixels,
            "frame_max_abs_delta": frame_max_abs_delta,
            "frame_equal": frame_equal,
            "qat_nonfinite_values": qat_nonfinite_values,
            "deployed_nonfinite_values": deployed_nonfinite_values,
            "passed": (
                argmax_diff == 0
                and frame_equal
                and qat_nonfinite_values == 0
                and deployed_nonfinite_values == 0
            ),
            "score_claim": False,
        }
    finally:
        model.load_state_dict(original)
        model.train(was_training)


def _parity_pair_ids(count: int, seed: int) -> list[int]:
    if not 1 <= count <= N_TOTAL_PAIRS:
        raise ValueError("--parity-pairs must be in [1,600]")
    if count == N_TOTAL_PAIRS:
        return list(range(N_TOTAL_PAIRS))
    rng = np.random.default_rng(seed + 20260809)
    return sorted(
        int(value)
        for value in rng.choice(N_TOTAL_PAIRS, size=count, replace=False).tolist()
    )


def _smoke_pair_ids(count: int | None, seed: int) -> torch.Tensor:
    """Return a deterministic real-pair population for a declared smoke scope."""

    if count is None:
        return torch.arange(N_TOTAL_PAIRS, dtype=torch.long)
    if not 1 <= count <= N_TOTAL_PAIRS:
        raise ValueError("--smoke-pairs must be in [1,600]")
    rng = np.random.default_rng(seed + 20260809)
    selected = sorted(
        int(value)
        for value in rng.choice(N_TOTAL_PAIRS, size=count, replace=False).tolist()
    )
    return torch.tensor(selected, dtype=torch.long)


@torch.no_grad()
def _evaluate_semantic_pairs(
    *,
    qat: ModuleType,
    model: torch.nn.Module,
    segnet: torch.nn.Module,
    conditioning_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    bits: int,
    pair_ids: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> float:
    """Run the borrowed exact QAT/scorer path on an explicit real-pair set."""

    model.eval()
    mismatches = 0
    pixels = 0
    for start in range(0, pair_ids.numel(), batch_size):
        batch_ids_cpu = pair_ids[start : start + batch_size]
        idx = batch_ids_cpu.to(device)
        conditioning = qat.gather_conditioning(
            conditioning_tokens, batch_ids_cpu, model.temporal_radius
        ).to(device)
        target = target_tokens[batch_ids_cpu].to(device)
        frame = qat.render_quantized(model, conditioning, idx, bits, exact_path=True)
        prediction = segnet(frame).argmax(1)
        mismatches += int((prediction != target).sum())
        pixels += target.numel()
    return mismatches / pixels


@torch.no_grad()
def _evaluate_rgb_pairs(
    *,
    qat: ModuleType,
    model: torch.nn.Module,
    conditioning_tokens: torch.Tensor,
    master_targets: torch.Tensor,
    bits: int,
    pair_ids: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> float:
    """Run the borrowed exact QAT/RGB diagnostic on an explicit real-pair set."""

    model.eval()
    squared_error = 0.0
    values = 0
    for start in range(0, pair_ids.numel(), batch_size):
        batch_ids_cpu = pair_ids[start : start + batch_size]
        idx = batch_ids_cpu.to(device)
        conditioning = qat.gather_conditioning(
            conditioning_tokens, batch_ids_cpu, model.temporal_radius
        ).to(device)
        frame = qat.render_quantized(model, conditioning, idx, bits, exact_path=True)
        target = F.interpolate(
            master_targets[batch_ids_cpu].to(device=device, dtype=torch.float32),
            size=frame.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        squared_error += float(((frame - target) / 255.0).square().sum())
        values += frame.numel()
    return squared_error / values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--input-cache", type=Path)
    parser.add_argument("--target-cache", type=Path)
    parser.add_argument("--master-cache", type=Path)
    parser.add_argument("--distill-weight", type=float, default=0.0)
    parser.add_argument("--distill-max-seg", type=float, default=4e-4)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--steps", type=int, default=12_000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--float-warmup-steps", type=int, default=0)
    parser.add_argument("--ce-fraction", type=float, default=0.50)
    parser.add_argument("--softplus-fraction", type=float, default=0.85)
    parser.add_argument("--ema-target-seed-fraction", type=float, default=EMA_TARGET_SEED_FRACTION)
    parser.add_argument("--parity-pairs", type=int, default=N_TOTAL_PAIRS)
    parser.add_argument(
        "--smoke-pairs",
        type=int,
        help="seeded real-pair scope reduction for mechanism-only CPU smokes",
    )
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--disable-tf32", action="store_true")
    parser.add_argument("--fixed-zero-mask", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--save", type=Path, required=True)
    # --- B2E train-for-editability levers (ns1 P1) -------------------------
    # Every one is DEFAULT-OFF and byte-identical to the pre-lever trainer when
    # off, including RNG consumption.  See src/tac/pr130_lift/editability_levers.py.
    parser.add_argument(
        "--weight-perturb-robustness",
        type=float,
        default=0.0,
        help="F1: weight-noise sigma in quantiser steps (0 = off)",
    )
    parser.add_argument(
        "--weight-perturb-shape",
        choices=("quantization", "gaussian"),
        default="quantization",
        help="F1: noise shape (inert while --weight-perturb-robustness is 0)",
    )
    parser.add_argument(
        "--film-critical-multiplier",
        type=float,
        default=DEFAULT_FILM_CRITICAL_MULTIPLIER,
        help="F1: extra perturbation on the pose-critical FiLM rows (inert when F1 is off)",
    )
    parser.add_argument(
        "--weight-qat-q3q4",
        action="store_true",
        help="F2: train through the exact mp2 mixed q3/q4 weight grid instead of uniform --bits",
    )
    parser.add_argument(
        "--film-row-dropout",
        type=float,
        default=0.0,
        help="F3: FiLM output-row dropout probability (0 = off)",
    )
    parser.add_argument(
        "--film-row-dropout-protect-top",
        type=int,
        default=0,
        help="F3: number of highest-norm FiLM rows never dropped",
    )
    parser.add_argument(
        "--carrier-rank-penalty",
        type=float,
        default=0.0,
        help="F4: nuclear-norm penalty weight on --carrier-tensors (0 = off)",
    )
    parser.add_argument(
        "--carrier-tensors",
        nargs="*",
        default=[],
        help="F4: parameter names the rank penalty applies to (empty = off)",
    )
    parser.add_argument(
        "--lever-seed",
        type=int,
        default=20260816,
        help="seed for the DEDICATED lever RNG stream (never the global stream)",
    )
    parser.add_argument(
        "--band-objective-weight",
        type=float,
        default=0.0,
        help=(
            "RG1: mixing fraction alpha for the debt-proportional 1-px label-band "
            "loss weight. 0 = off (exactly the stock objective); 1 = pure measured "
            "debt density. mean(weight) == 1 at every alpha, so this never rescales "
            "the effective learning rate."
        ),
    )
    args = parser.parse_args(argv)
    # Derived, never supplied: present only when the term is actually active, so a
    # pre-RG1 resume checkpoint reconciles additively against an inert run.
    args.band_weight_table_sha256 = None
    if args.input_cache is None:
        args.input_cache = args.cache
    if args.target_cache is None:
        args.target_cache = args.cache
    if args.input_cache is None or args.target_cache is None:
        parser.error("provide --cache or both --input-cache and --target-cache")
    if args.bits != 4:
        parser.error("--bits must be 4 because the deployed semantic packer is int4-only")
    if args.steps < 1:
        parser.error("--steps must be positive")
    if not 0 <= args.float_warmup_steps < args.steps:
        parser.error("--float-warmup-steps must be in [0, steps)")
    if args.batch_size < 1 or args.eval_batch_size < 1:
        parser.error("batch sizes must be positive")
    if args.eval_every < 1 or args.checkpoint_every < 1:
        parser.error("eval/checkpoint cadence must be positive")
    if not 0.0 < args.ema_target_seed_fraction < 1.0:
        parser.error("--ema-target-seed-fraction must be in (0,1)")
    if not 1 <= args.parity_pairs <= N_TOTAL_PAIRS:
        parser.error("--parity-pairs must be in [1,600]")
    if args.smoke_pairs is not None and not 1 <= args.smoke_pairs <= N_TOTAL_PAIRS:
        parser.error("--smoke-pairs must be in [1,600]")
    if args.smoke_pairs is not None and args.batch_size > args.smoke_pairs:
        parser.error("--batch-size cannot exceed --smoke-pairs")
    # ddm_av3 F2: --save and --out are FILE paths that the atomic writers finish
    # with os.replace().  An existing DIRECTORY at either path raises
    # IsADirectoryError -- but only at the END of the run, after the full final
    # evaluation.  ddm_lr1/A2 hit exactly this: 379 s of Metal, a computed
    # final_seg / parity / packed-bytes / verdict, and no result.json at all.
    # Refuse in milliseconds instead, so the payload is never computed-then-lost.
    for flag, path in (("--save", args.save), ("--out", args.out)):
        if path is not None and Path(path).is_dir():
            parser.error(
                f"{flag} must be a file path, but {path} is an existing directory "
                "(the run would compute its full result and then fail to write it)"
            )
    if args.band_objective_weight > 0.0:
        # Record the table by CONTENT, never by location: the path differs across
        # machines, the bytes do not, and the bytes are what makes the run causal.
        args.band_weight_table_sha256 = band_weight_table_sha256()
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    assert_governed_admission("train_semantic_quantized_resumable")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if args.disable_tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    root = args.challenge_root.resolve()
    sys.path.insert(0, str(root))
    import modules  # pylint: disable=import-error,import-outside-toplevel

    qat = _load_lifted_qat()
    conditioning_tokens = torch.load(
        args.input_cache, map_location="cpu", weights_only=False
    )["seg"].long()
    target_tokens = torch.load(
        args.target_cache, map_location="cpu", weights_only=False
    )["seg"].long()
    if conditioning_tokens.shape[0] != N_TOTAL_PAIRS or target_tokens.shape[0] != N_TOTAL_PAIRS:
        raise ValueError("semantic caches must contain all 600 pairs")
    master_targets = None
    if args.master_cache is not None:
        master_targets = torch.load(
            args.master_cache, map_location="cpu", weights_only=False
        )["masters"].to(torch.uint8)

    init_checkpoint = torch.load(args.init, map_location="cpu", weights_only=False)
    architecture = architecture_config_from_checkpoint(
        init_checkpoint, consumer="train_semantic_quantized_resumable.init"
    )
    model = qat.SemanticTokenRenderer(
        width=architecture["width"],
        blocks=architecture["blocks"],
        frame_dim=architecture["frame_dim"],
        num_pairs=architecture["num_pairs"],
        num_tokens=architecture["num_tokens"],
        phase_y=architecture["phase_y"],
        phase_x=architecture["phase_x"],
        temporal_radius=architecture["temporal_radius"],
    ).to(device)
    model.load_state_dict(init_checkpoint["state_dict"])
    zero_masks = {
        name: (value == 0).to(device)
        for name, value in init_checkpoint["state_dict"].items()
        if args.fixed_zero_mask and value.ndim >= 2
    }
    segnet = modules.SegNet().eval().to(device)
    segnet.load_state_dict(load_file(modules.segnet_sd_path, device=str(device)))
    for parameter in segnet.parameters():
        parameter.requires_grad_(False)

    lever_config = EditabilityLeverConfig(
        weight_perturb_robustness=args.weight_perturb_robustness,
        weight_perturb_shape=args.weight_perturb_shape,
        film_critical_multiplier=args.film_critical_multiplier,
        weight_qat_q3q4=args.weight_qat_q3q4,
        weight_qat_low_bits=3,
        weight_qat_high_bits=args.bits,
        film_row_dropout=args.film_row_dropout,
        film_row_dropout_protect_top=args.film_row_dropout_protect_top,
        carrier_rank_penalty=args.carrier_rank_penalty,
        carrier_tensors=tuple(args.carrier_tensors),
        seed=args.lever_seed,
    )
    levers = EditabilityLevers(lever_config)
    if lever_config.any_active:
        # F2 deliberately EXTENDS the uniform-int4 refusal surface: --bits stays
        # hard-pinned at 4 (the deployed packer is int4-only), and the mixed q3/q4
        # grid is requested through its own flag so the uniform path is untouched
        # and byte-identical whenever F2 is off.
        print(
            "[b2e] editability levers ACTIVE: "
            + json.dumps(lever_config.activation_ledger()["levers"], sort_keys=True)
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.lr * 0.01
    )
    ema_policy = resolve_ema_policy(
        args.steps, target_seed_fraction=args.ema_target_seed_fraction
    )
    ema = EMA(model, decay=float(ema_policy["decay"]), warmup=True)
    generator = torch.Generator().manual_seed(args.seed)
    active_pair_ids = _smoke_pair_ids(args.smoke_pairs, args.seed)
    order = active_pair_ids[
        torch.randperm(active_pair_ids.numel(), generator=generator)
    ]
    cursor = 0
    artifacts = _artifact_manifest(args)
    checkpoints_written: list[str] = []

    if args.resume_from is None:
        with ema_eval_scope(model, ema):
            best_seg = _evaluate_semantic_pairs(
                qat=qat,
                model=model,
                segnet=segnet,
                conditioning_tokens=conditioning_tokens,
                target_tokens=target_tokens,
                bits=args.bits,
                pair_ids=active_pair_ids,
                batch_size=args.eval_batch_size,
                device=device,
            )
            best_rgb = (
                _evaluate_rgb_pairs(
                    qat=qat,
                    model=model,
                    conditioning_tokens=conditioning_tokens,
                    master_targets=master_targets,
                    bits=args.bits,
                    pair_ids=active_pair_ids,
                    batch_size=args.eval_batch_size,
                    device=device,
                )
                if master_targets is not None
                else None
            )
        best_key = (
            (
                best_seg > args.distill_max_seg,
                best_seg if best_seg > args.distill_max_seg else best_rgb,
                best_rgb if best_seg > args.distill_max_seg else best_seg,
            )
            if best_rgb is not None
            else (best_seg,)
        )
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in ema.state_dict().items()
        }
        # ddm_av3 F3: the selection is seeded from step 0, so a run that only
        # degrades never displaces its own INIT.  That is correct SELECTION and
        # misleading REPORTING -- all four ddm_lr1 arms headline `verdict: PASS`
        # on a number that is their input.  Tracking which step won makes the
        # difference legible in the result instead of only in `history`.
        best_step = 0
        init_seg = best_seg
        history = [
            {
                "step": 0,
                "quantized_exact_seg": best_seg,
                "normalized_rgb_mse": best_rgb,
                "evaluated_weights": "ema_shadow",
            }
        ]
        start_step = 1
        print(json.dumps(history[-1]), flush=True)
    else:
        restored = _load_resume_state(
            args.resume_from,
            args=args,
            architecture_config=architecture,
            artifacts=artifacts,
            ema_policy=ema_policy,
            model=model,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            generator=generator,
            device=device,
        )
        start_step = restored["start_step"]
        order = restored["order"]
        cursor = restored["cursor"]
        history = restored["history"]
        best_key = restored["best_key"]
        best_seg = restored["best_seg"]
        best_rgb = restored["best_rgb"]
        best_state = restored["best_state"]
        best_step = restored["best_step"]
        init_seg = restored["init_seg"]
        print(
            json.dumps(
                {
                    "event": "resume_loaded",
                    "resume_from": str(args.resume_from),
                    "start_step": start_step,
                }
            ),
            flush=True,
        )

    # RG1 band objective.  Default OFF, but "off" is a TRACKED state: the table
    # is resolved (and its content sha recorded) only when the term is active,
    # and the activation counter below makes a held-but-never-fired term
    # detectable in the receipt rather than silent.
    band_table = (
        load_band_weight_table() if args.band_objective_weight > 0.0 else None
    )
    band_activations = 0
    band_stats_first: dict[str, Any] | None = None
    if band_table is not None:
        print(
            json.dumps(
                {
                    "event": "band_objective_active",
                    "alpha": args.band_objective_weight,
                    "basis": band_table.basis,
                    "table_sha256": args.band_weight_table_sha256,
                    "off_band_weight": band_table.off_band_weight,
                    "band_px_ring0": band_table.provenance.get("band_px_ring0"),
                }
            ),
            flush=True,
        )

    boundaries = phase_end_steps(args)
    last_phase = _phase_for_step(args, min(max(start_step, 1), args.steps))
    for step in range(start_step, args.steps + 1):
        if cursor + args.batch_size > active_pair_ids.numel():
            order = active_pair_ids[
                torch.randperm(active_pair_ids.numel(), generator=generator)
            ]
            cursor = 0
        batch_ids_cpu = order[cursor : cursor + args.batch_size]
        cursor += args.batch_size
        idx = batch_ids_cpu.to(device)
        conditioning = qat.gather_conditioning(
            conditioning_tokens, batch_ids_cpu, model.temporal_radius
        ).to(device)
        target = target_tokens[batch_ids_cpu].to(device)
        model.train()
        in_float_warmup = step <= args.float_warmup_steps
        if in_float_warmup:
            frame = qat.render_float(model, conditioning, idx, exact_path=True)
        elif not lever_config.any_active:
            # Untouched stock path: byte-identical to the pre-lever trainer.
            frame = qat.render_quantized(
                model, conditioning, idx, args.bits, exact_path=True
            )
        else:
            # Lever path. The levers OWN the weight quantization here (always at
            # least uniform --bits), so we must call render_FLOAT to avoid
            # double-quantizing what parameter_overrides already quantized. The
            # render tail (ste_uint8 + the two interpolates) is identical in both
            # helpers, so only the parameter values differ.
            with levers.applied(model, base_bits=args.bits):
                frame = qat.render_float(model, conditioning, idx, exact_path=True)
        logits = segnet(frame)
        band_weight = (
            band_weight_field(target, band_table, args.band_objective_weight)
            if band_table is not None
            else None
        )
        if in_float_warmup:
            # The band weight applies here too when active, so the term is not
            # silently phase-dependent for a future run with a warmup.
            if band_weight is None:
                segmentation_loss = F.cross_entropy(logits, target)
            else:
                per_pixel = F.cross_entropy(logits, target, reduction="none")
                segmentation_loss = (per_pixel * band_weight).sum() / band_weight.sum()
                band_activations += 1
            phase = "float_ce"
        elif band_weight is None:
            # Untouched stock path: byte-identical to the pre-RG1 trainer.
            qat_steps = args.steps - args.float_warmup_steps
            segmentation_loss, phase = qat.curriculum_loss(
                logits,
                target,
                step - args.float_warmup_steps - 1,
                qat_steps,
                args.ce_fraction,
                args.softplus_fraction,
            )
        else:
            qat_steps = args.steps - args.float_warmup_steps
            segmentation_loss, phase = curriculum_loss_weighted(
                logits,
                target,
                step - args.float_warmup_steps - 1,
                qat_steps,
                args.ce_fraction,
                args.softplus_fraction,
                band_weight,
            )
            band_activations += 1
            if band_stats_first is None:
                band_stats_first = band_weight_stats(
                    band_weight, label_boundary(target)
                )
        if phase != _phase_for_step(args, step):
            raise RuntimeError("curriculum phase derivation drifted from lifted PR130 code")
        last_phase = phase
        distill_loss = torch.zeros((), device=device)
        if master_targets is not None:
            master = F.interpolate(
                master_targets[batch_ids_cpu].to(
                    device=device, dtype=torch.float32
                ),
                size=frame.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            distill_loss = F.mse_loss(frame / 255.0, master / 255.0)
        loss = segmentation_loss + args.distill_weight * distill_loss
        if lever_config.f4_active:
            # F4 contributes exactly nothing (and no graph node) when inactive,
            # so the stock loss is untouched with levers off.
            loss = loss + levers.rank_penalty(model).to(loss.device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and name in zero_masks:
                parameter.grad.masked_fill_(zero_masks[name], 0.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        if zero_masks:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name in zero_masks:
                        parameter.masked_fill_(zero_masks[name], 0.0)
        ema.update(model)
        scheduler.step()

        if step % args.eval_every == 0 or step == args.steps:
            with ema_eval_scope(model, ema):
                exact = _evaluate_semantic_pairs(
                    qat=qat,
                    model=model,
                    segnet=segnet,
                    conditioning_tokens=conditioning_tokens,
                    target_tokens=target_tokens,
                    bits=args.bits,
                    pair_ids=active_pair_ids,
                    batch_size=args.eval_batch_size,
                    device=device,
                )
                rgb_mse = (
                    _evaluate_rgb_pairs(
                        qat=qat,
                        model=model,
                        conditioning_tokens=conditioning_tokens,
                        master_targets=master_targets,
                        bits=args.bits,
                        pair_ids=active_pair_ids,
                        batch_size=args.eval_batch_size,
                        device=device,
                    )
                    if master_targets is not None
                    else None
                )
            candidate_key = (
                (
                    exact > args.distill_max_seg,
                    exact if exact > args.distill_max_seg else rgb_mse,
                    rgb_mse if exact > args.distill_max_seg else exact,
                )
                if rgb_mse is not None
                else (exact,)
            )
            if candidate_key < best_key:
                best_key = candidate_key
                best_seg = exact
                best_rgb = rgb_mse
                best_step = step
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in ema.state_dict().items()
                }
            record = {
                "step": step,
                "phase": phase,
                "loss": float(loss.detach()),
                "segmentation_loss": float(segmentation_loss.detach()),
                "distill_loss": float(distill_loss.detach()),
                "quantized_exact_seg": exact,
                "best_quantized_exact_seg": best_seg,
                "normalized_rgb_mse": rgb_mse,
                "best_normalized_rgb_mse": best_rgb,
                "lr": optimizer.param_groups[0]["lr"],
                "evaluated_weights": "ema_shadow",
            }
            history.append(record)
            print(json.dumps(record), flush=True)

        stage_boundary = step in boundaries
        should_checkpoint = stage_boundary or step % args.checkpoint_every == 0
        if should_checkpoint:
            preserved, latest = _checkpoint_paths(
                args.save,
                step=step,
                phase=phase,
                stage_boundary=stage_boundary,
            )
            checkpoint_kind = "stage_boundary" if stage_boundary else "periodic"
            payload = _checkpoint_payload(
                args=args,
                architecture_config=architecture,
                artifacts=artifacts,
                ema_policy=ema_policy,
                model=model,
                ema=ema,
                optimizer=optimizer,
                scheduler=scheduler,
                generator=generator,
                order=order,
                cursor=cursor,
                step=step,
                phase=phase,
                checkpoint_kind=checkpoint_kind,
                history=history,
                best_key=best_key,
                best_seg=best_seg,
                best_step=best_step,
                init_seg=init_seg,
                best_rgb=best_rgb,
                best_state=best_state,
                deployment_state=ema.state_dict(),
            )
            _atomic_torch_save(payload, preserved)
            _atomic_torch_save(payload, latest)
            checkpoints_written.append(str(preserved))
            print(
                json.dumps(
                    {
                        "event": "checkpoint_saved",
                        "step": step,
                        "phase": phase,
                        "kind": checkpoint_kind,
                        "preserved": str(preserved),
                        "latest": str(latest),
                    }
                ),
                flush=True,
            )

    # Fail fast, BEFORE the expensive finalize: a term that was requested but never
    # fired must refuse rather than report as an active lever.  Placed here, not
    # after the final evaluation, so this guard can never discard a computed result
    # -- which is the exact defect the write-order fix below repairs.
    if band_table is not None and band_activations == 0:
        raise RuntimeError(
            "band objective requested but never fired -- a held-but-inert term "
            "would silently report as an active lever"
        )

    deployment_ema = EMA(model, decay=float(ema_policy["decay"]), warmup=True)
    deployment_ema.shadow = {
        key: value.detach().to(device).clone() for key, value in best_state.items()
    }
    deployment_ema._num_updates = int(ema._num_updates)
    with ema_eval_scope(model, deployment_ema):
        final_seg = _evaluate_semantic_pairs(
            qat=qat,
            model=model,
            segnet=segnet,
            conditioning_tokens=conditioning_tokens,
            target_tokens=target_tokens,
            bits=args.bits,
            pair_ids=active_pair_ids,
            batch_size=args.eval_batch_size,
            device=device,
        )
        final_packed_bytes = qat.packed_size(model, args.bits)
    parity = deployed_argmax_parity(
        qat=qat,
        model=model,
        segnet=segnet,
        conditioning_tokens=conditioning_tokens,
        bits=args.bits,
        pair_ids=_parity_pair_ids(args.parity_pairs, args.seed),
        batch_size=args.eval_batch_size,
        architecture_config=architecture,
        deployment_state=best_state,
        device=device,
    )
    axis = (
        "[macOS-MPS advisory]"
        if device.type == "mps"
        else "[contest-CUDA mechanism-only; no score]"
        if device.type == "cuda"
        else "[CPU mechanism smoke; no score]"
    )
    parity["axis"] = axis
    if not parity["passed"]:
        blocked = {
            "verdict": "BLOCKED_EMA_DEPLOYED_ARGMAX_PARITY",
            "score_claim": False,
            "axis": axis,
            "ema_deployed_argmax_parity": parity,
        }
        _atomic_write_json(blocked, args.out)
        raise RuntimeError(
            "EMA shadow refused: deployed QAT path changed SegNet argmax pixels"
        )

    result = {
        "verdict": "PASS" if final_seg < 4e-4 else "FAIL",
        "score_claim": False,
        "axis": axis,
        "config": _jsonable_args(args),
        "ema_policy": ema_policy,
        "mechanism_pair_ids": active_pair_ids.tolist(),
        "mechanism_pair_denominator": int(active_pair_ids.numel()),
        "ema_deployed_argmax_parity": parity,
        "quantized_exact_seg": final_seg,
        # ddm_av3 F3: `quantized_exact_seg` is the argmin INCLUDING step 0, so a
        # degrade-only run reports its own input.  These three fields say so.
        "init_quantized_exact_seg": init_seg,
        "best_step": int(best_step),
        "improved_over_init": bool(best_step != 0 and final_seg < init_seg),
        "normalized_rgb_mse": best_rgb,
        "packed_parameter_bytes": final_packed_bytes,
        "band_objective": {
            "alpha": args.band_objective_weight,
            "active": band_table is not None,
            "activations": band_activations,
            "table_sha256": args.band_weight_table_sha256,
            "basis": band_table.basis if band_table is not None else None,
            "first_step_weight_stats": band_stats_first,
            "reason_if_off": (
                None if band_table is not None else "alpha == 0 (default)"
            ),
        },
        "history": history,
        "checkpoints_written": checkpoints_written,
    }
    # ddm_av3 F2, ALWAYS KEEP THE PAYLOAD: the result JSON is the CHEAP,
    # IRREPLACEABLE artifact (final_seg, parity, packed bytes, the full history);
    # the 1.7 MB checkpoint is the expensive but REBUILDABLE one.  ddm_lr1/A2 had
    # this order inverted, the save raised IsADirectoryError, and 379 s of Metal
    # produced no readable result at all.  Cheap-and-irreplaceable goes first.
    _atomic_write_json(result, args.out)
    final_payload = _checkpoint_payload(
        args=args,
        architecture_config=architecture,
        artifacts=artifacts,
        ema_policy=ema_policy,
        model=model,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        generator=generator,
        order=order,
        cursor=cursor,
        step=args.steps,
        phase=last_phase,
        checkpoint_kind="final_selected_ema",
        history=history,
        best_key=best_key,
        best_seg=best_seg,
        best_step=best_step,
        init_seg=init_seg,
        best_rgb=best_rgb,
        best_state=best_state,
        deployment_state=best_state,
    )
    final_payload["best_exact_seg"] = final_seg
    final_payload["result"] = result
    _atomic_torch_save(final_payload, args.save)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
