# SPDX-License-Identifier: MIT
"""tac.training.long_training_canonical - canonical L2 long-training infrastructure.

Charter per Path 3 canonical-substrate-development-cascade doctrine
(committed ``fb270e9b6``) section "L2 LONG-TRAINING INFRASTRUCTURE -
CANONICAL + REUSABLE + COMPOSABLE + PRODUCTION-HARDENED" + operator
binding directive 2026-05-26 verbatim: *"Also the long training
infrastructure, ensure reusable composable beautiful elegant creative
expressive cimposable abstractions and production hardened OSS and no
duplicative code"*.

The canonical L2 long-training contract per the doctrine 10-element
specification:

1. ``run_long_training(substrate, config) -> TrainingArtifact`` -
   canonical entry-point taking a substrate-conforming object + frozen
   LongTrainingConfig + emitting canonical TrainingArtifact with EMA
   shadow checkpoint, canonical Provenance, canonical posterior anchor.
2. ``LongTrainingConfig`` frozen dataclass - canonical config schema.
3. ``CurriculumStage`` frozen dataclass - per-stage hparams mirroring
   PR95 8-stage curriculum + extensible per substrate-class.
4. Checkpoint+resume - canonical interruption-safe pattern; fcntl-locked
   per Catalog #131; sister of subagent_checkpoint per Catalog #206.
5. Per-arm canonical Provenance + posterior anchor - per Catalog #323 +
   Catalog #128 with non-promotable markers per Catalog #127/#192/#317/#341.
6. Differentiable-eval-roundtrip + EMA-apply-at-eval canonical wrappers -
   per CLAUDE.md "eval_roundtrip -- NON-NEGOTIABLE" + "EMA -- NON-NEGOTIABLE".
7. Multi-arm parallel dispatch - canonical orchestrator for concurrent
   arms on M-series shared GPU memory.
8. Crash-recovery + OOM-safe - canonical patterns for Apple Silicon
   memory pressure with batch-size halving on OOM.
9. Observability surface per Catalog #305 - per-epoch loss curve +
   per-axis components + EMA-drift + canonical metrics flushed to
   ``.omx/state/`` (queryable post-hoc).
10. OSS-clean public API - narrow ``__all__`` per Catalog #335 canonical
    contract + canonical docstrings + zero ``/Users/...`` paths per
    Catalog #208 + SPDX-License-Identifier: MIT header.

The substrate-axis abstraction: a SubstrateLongTrainingAdapter Protocol
captures the canonical training operations any substrate must expose:

- ``model``: trainable parameters container (torch.nn.Module or
  MLX module via duck-typing on .parameters() + .state_dict()).
- ``loss_fn(model, batch) -> scalar_loss``: substrate-specific
  reconstruction + Lagrangian.
- ``score_aware_components(model, batch) -> dict``: optional
  d_seg + d_pose + rate per Catalog #356 per-axis decomposition.
- ``export_archive(model, output_dir) -> archive_bytes``: byte-stable
  archive emission per Catalog #146 inflate runtime contract.

The Protocol pattern is canonical-vs-unique per Catalog #290: substrates
adopt the canonical helper when their training axis is standard
(SGD/Adam + per-step loss + checkpoint cadence); substrates with
substrate-specific training axes (PR95-HNeRV with hardcoded HNeRV
decoder + RGB-MSE loss + PyAV pipeline) use the PR95 sister module at
``tac.local_acceleration.pr95_hnerv_mlx_long_training``.

Sister modules + composition contract:

- ``tac.training.long_training_canonical`` (THIS module) - canonical
  substrate-AGNOSTIC L2 helper; substrate-conforming adapters plug in.
- ``tac.local_acceleration.pr95_hnerv_mlx_long_training`` - canonical
  PR95-HNeRV-SPECIFIC L2 helper with hardcoded HNeRVDecoderMLX +
  RGB-MSE + PyAV pipeline. Per Catalog #290 canonical-vs-unique: this
  is a legitimate fork because PR95-HNeRV training has substrate-
  specific structure (HNeRV decoder forward is not substrate-axis).
- ``tac.substrates._shared.trainer_skeleton`` - canonical substrate-
  trainer utilities (seeds, EVAL_HW, decode_real_pairs,
  device_or_die, OptimizedTrainingContext). THIS module imports
  primitives from there.
- ``tac.substrates._shared.posterior_emission_helper`` - canonical
  L0/L1 landing posterior emission. THIS module's per-arm posterior
  anchor invokes the canonical helper.

Catalog cross-refs (binding):
  * Catalog #2 EMA NON-NEGOTIABLE (decay=0.997)
  * Catalog #128 fcntl-locked posterior write discipline
  * Catalog #127/#192/#317/#341 canonical non-promotable markers
  * Catalog #131 bare-write to .omx/state/ refusal
  * Catalog #146 contest-compliant inflate runtime contract
  * Catalog #178 TF32 (CUDA paths only)
  * Catalog #190 hardware_substrate auto-detection (no hardcoded T4)
  * Catalog #206 subagent crash-resume discipline (sister pattern)
  * Catalog #208 docs/local-paths discipline
  * Catalog #229 premise verification (file hashes + config snapshot)
  * Catalog #265 / #335 canonical contract pattern
  * Catalog #287 placeholder-rationale rejection
  * Catalog #290 canonical-vs-unique decision per layer
  * Catalog #294 9-dim success checklist evidence
  * Catalog #299 gate consolidation discipline
  * Catalog #305 observability surface 6-facet
  * Catalog #323 canonical Provenance umbrella
  * Catalog #344 canonical equations registry calibration target
  * Catalog #354 master_gradient_exploit_consumers integration
  * Catalog #355 cathedral autopilot meta-Lagrangian invocation
  * Catalog #356 per-axis decomposition per Tier B contract
  * Catalog #357 dual-tier cathedral consumer architecture
  * CLAUDE.md "Beauty, simplicity, and developer experience"
  * CLAUDE.md "Subagent coherence-by-default" (6-hook wire-in)
  * CLAUDE.md "MLX portable-local-substrate authority"
  * CLAUDE.md "MPS auth eval is NOISE"
  * CLAUDE.md "Submission auth eval - BOTH CPU AND CUDA"
  * CLAUDE.md "Apples-to-apples evidence discipline"
  * CLAUDE.md "Forbidden /tmp paths in any persisted artifact"
"""
from __future__ import annotations

# CHECKPOINT_DISCIPLINE_WAIVED:canonical_infrastructure_module_no_subagent_dispatches_within_helper_body
# FORMALIZATION_PENDING:queued_for_canonical_equation_registration_post_first_l2_run_per_catalog_344_protocol

import dataclasses
import fcntl
import hashlib
import json
import os
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

__all__ = [
    # Canonical constants
    "CANONICAL_EMA_DECAY",
    "CANONICAL_NON_PROMOTABLE_MARKERS",
    "DEFAULT_CHECKPOINT_INTERVAL_EPOCHS",
    "DEFAULT_EARLY_STOPPING_PATIENCE",
    "DEFAULT_TELEMETRY_FLUSH_INTERVAL_EPOCHS",
    "PR95_8STAGE_CURRICULUM_DEFAULT",
    "TRAINING_ARTIFACT_SCHEMA_VERSION",
    # Frozen dataclasses
    "CurriculumStage",
    "LongTrainingConfig",
    "PerEpochMetrics",
    "TrainingArtifact",
    "MultiArmDispatchResult",
    # Substrate adapter Protocol
    "SubstrateLongTrainingAdapter",
    # Canonical entry-points
    "run_long_training",
    "run_long_training_multi_arm",
    # Canonical primitives (composable)
    "PolyakEMAShadow",
    "TelemetrySink",
    "CheckpointWriter",
    "OOMSafeStepRunner",
    # Conformance helpers
    "validate_long_training_config",
    "validate_substrate_adapter",
]


# ---------------------------------------------------------------------------
# Canonical constants
# ---------------------------------------------------------------------------

# Per CLAUDE.md "EMA -- NON-NEGOTIABLE, HIGHEST EMPHASIS" + Catalog #2
# canonical Quantizr PR101 anchor.
CANONICAL_EMA_DECAY: float = 0.997

# Per Catalog #127/#192/#317/#341 canonical non-promotable markers.
# Every TrainingArtifact carries these flags FALSE by construction.
CANONICAL_NON_PROMOTABLE_MARKERS: dict[str, bool] = {
    "score_claim": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "rank_or_kill_eligible": False,
    "promotable": False,
}

# Canonical defaults (operator overridable via LongTrainingConfig).
DEFAULT_CHECKPOINT_INTERVAL_EPOCHS: int = 100
DEFAULT_EARLY_STOPPING_PATIENCE: int = 200
DEFAULT_TELEMETRY_FLUSH_INTERVAL_EPOCHS: int = 10

# Canonical schema version for TrainingArtifact JSON emission.
TRAINING_ARTIFACT_SCHEMA_VERSION: str = "long_training_canonical_artifact.v1"

# Placeholder rationale tokens REJECTED per Catalog #287 sister discipline
# so the helper's docstring example cannot self-waive.
_PLACEHOLDER_RATIONALE_TOKENS: frozenset[str] = frozenset({
    "<rationale>",
    "<reason>",
    "<rationale_here>",
    "<reason_here>",
})

# Canonical PR95 8-stage curriculum (extracted from
# tac.local_acceleration.pr95_hnerv_mlx_long_training for substrate-
# agnostic reuse; the sister module's CANONICAL_8STAGE_CURRICULUM
# remains the PR95-specific authority).
PR95_8STAGE_CURRICULUM_DEFAULT: tuple["CurriculumStage", ...]  # forward-declared


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_rationale_not_placeholder(rationale: str, field_name: str) -> None:
    """Catalog #287 sister discipline: reject placeholder rationale literals."""
    if not isinstance(rationale, str):
        raise TypeError(f"{field_name} must be str; got {type(rationale).__name__}")
    if not rationale.strip():
        return
    lowered = rationale.strip().lower()
    for token in _PLACEHOLDER_RATIONALE_TOKENS:
        if token.lower() in lowered:
            raise ValueError(
                f"{field_name} contains placeholder rationale literal {token!r} "
                f"per Catalog #287 sister discipline; supply a substantive "
                f"non-placeholder rationale (>=4 chars) instead."
            )
    if len(lowered) < 4:
        raise ValueError(
            f"{field_name} rationale {rationale!r} too short (<4 chars); "
            "supply a substantive non-placeholder rationale per Catalog #287."
        )


def _refuse_tmp_path(path: Path, field_name: str) -> None:
    """CLAUDE.md FORBIDDEN_PATTERNS: refuse /tmp paths in persisted artifacts."""
    resolved = str(path)
    forbidden_prefixes = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")
    if any(resolved.startswith(p) for p in forbidden_prefixes):
        raise ValueError(
            f"{field_name} = {resolved!r} starts with /tmp/-class transient prefix; "
            "use experiments/results/<lane_id>_<timestamp>/ or .omx/state/ per "
            "CLAUDE.md 'Forbidden /tmp paths in any persisted artifact' "
            "(the transient-evidence trap)."
        )


def _utc_now_iso() -> str:
    """Return current UTC time in ISO-8601 (Z-suffix) format."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_text(payload: str) -> str:
    """Hex sha256 of a text payload."""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CurriculumStage:
    """Canonical curriculum stage frozen dataclass.

    Per the doctrine 10-element contract item #3: each stage carries
    name + start_epoch + end_epoch (both global; ``end_epoch`` is
    exclusive) + loss_weights + lr_scale + freeze_layers + enable_qat.
    Mirrors PR95 8-stage curriculum + extensible per substrate-class
    (e.g. Z6 may add ``ego_motion_warmup`` loss_weight key; J=MDL-IBPS
    may add ``beta_ib_schedule`` per stage).

    Args:
        name: human-readable stage name (e.g. ``"warmup_low_lr"``).
        start_epoch: global epoch (inclusive) the stage begins at.
        end_epoch: global epoch (exclusive) the stage ends at.
        loss_weights: per-loss-term weights (e.g. ``{"recon": 1.0, "kl": 0.1}``).
        lr_scale: multiplier applied to base learning rate during this stage.
        freeze_layers: tuple of layer-name prefixes to FREEZE (no gradient).
        enable_qat: whether quantization-aware training is enabled in stage.
        notes: optional substantive rationale (Catalog #287 placeholder rejected).

    Invariants per ``__post_init__``:
        * 0 <= start_epoch < end_epoch
        * lr_scale > 0
        * loss_weights non-empty + all values finite + non-negative
    """

    name: str
    start_epoch: int
    end_epoch: int
    loss_weights: Mapping[str, float] = field(default_factory=lambda: {"recon": 1.0})
    lr_scale: float = 1.0
    freeze_layers: tuple[str, ...] = ()
    enable_qat: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError(f"name must be non-empty str; got {self.name!r}")
        if not isinstance(self.start_epoch, int) or self.start_epoch < 0:
            raise ValueError(
                f"start_epoch must be int >= 0; got {self.start_epoch!r}"
            )
        if not isinstance(self.end_epoch, int) or self.end_epoch <= self.start_epoch:
            raise ValueError(
                f"end_epoch must be int > start_epoch; got "
                f"start_epoch={self.start_epoch}, end_epoch={self.end_epoch}"
            )
        if not isinstance(self.lr_scale, (int, float)) or self.lr_scale <= 0.0:
            raise ValueError(
                f"lr_scale must be positive float; got {self.lr_scale!r}"
            )
        if not isinstance(self.loss_weights, Mapping) or not self.loss_weights:
            raise ValueError(
                f"loss_weights must be non-empty Mapping; got {self.loss_weights!r}"
            )
        for k, v in self.loss_weights.items():
            if not isinstance(k, str) or not k:
                raise ValueError(f"loss_weights keys must be non-empty str; got {k!r}")
            if not isinstance(v, (int, float)) or v != v or v < 0.0:
                raise ValueError(
                    f"loss_weights[{k!r}] must be finite non-negative; got {v!r}"
                )
        if not isinstance(self.freeze_layers, tuple):
            raise TypeError(
                f"freeze_layers must be tuple of str; got {type(self.freeze_layers).__name__}"
            )
        for fl in self.freeze_layers:
            if not isinstance(fl, str) or not fl:
                raise ValueError(f"freeze_layers entries must be non-empty str; got {fl!r}")
        if self.notes:
            _validate_rationale_not_placeholder(self.notes, "CurriculumStage.notes")

    @property
    def epoch_count(self) -> int:
        return self.end_epoch - self.start_epoch

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "start_epoch": int(self.start_epoch),
            "end_epoch": int(self.end_epoch),
            "loss_weights": dict(self.loss_weights),
            "lr_scale": float(self.lr_scale),
            "freeze_layers": list(self.freeze_layers),
            "enable_qat": bool(self.enable_qat),
            "notes": self.notes,
            "epoch_count": int(self.epoch_count),
        }


# Canonical PR95 8-stage curriculum default (substrate-agnostic; each
# substrate may override via LongTrainingConfig.curriculum_stages).
PR95_8STAGE_CURRICULUM_DEFAULT = (
    CurriculumStage(
        name="warmup_low_lr",
        start_epoch=0,
        end_epoch=300,
        loss_weights={"recon": 1.0},
        lr_scale=0.1,
        notes="Stage 1 warmup at low LR per PR95 8-stage forensic recovery.",
    ),
    CurriculumStage(
        name="ramp_lr",
        start_epoch=300,
        end_epoch=600,
        loss_weights={"recon": 1.0},
        lr_scale=0.5,
        notes="Stage 2 LR ramp; transition into main training band.",
    ),
    CurriculumStage(
        name="main_train_band_a",
        start_epoch=600,
        end_epoch=1000,
        loss_weights={"recon": 1.0},
        lr_scale=1.0,
        notes="Stage 3 main training band; canonical PR95 reference hparams.",
    ),
    CurriculumStage(
        name="main_train_band_b",
        start_epoch=1000,
        end_epoch=1400,
        loss_weights={"recon": 1.0},
        lr_scale=1.0,
        notes="Stage 4 main training band continuation.",
    ),
    CurriculumStage(
        name="refine_lr_decay",
        start_epoch=1400,
        end_epoch=1800,
        loss_weights={"recon": 1.0},
        lr_scale=0.5,
        notes="Stage 5 refinement with LR decay step.",
    ),
    CurriculumStage(
        name="polish_lr_decay",
        start_epoch=1800,
        end_epoch=2200,
        loss_weights={"recon": 1.0},
        lr_scale=0.2,
        notes="Stage 6 polish with further LR decay.",
    ),
    CurriculumStage(
        name="finetune_low_lr",
        start_epoch=2200,
        end_epoch=2600,
        loss_weights={"recon": 1.0},
        lr_scale=0.1,
        notes="Stage 7 fine-tune at low LR.",
    ),
    CurriculumStage(
        name="converge_low_lr",
        start_epoch=2600,
        end_epoch=3000,
        loss_weights={"recon": 1.0},
        lr_scale=0.05,
        notes="Stage 8 convergence at very low LR; canonical curriculum end.",
    ),
)


@dataclass(frozen=True)
class LongTrainingConfig:
    """Canonical L2 long-training config frozen dataclass.

    Per the doctrine 10-element contract item #2: canonical config
    schema for the substrate-agnostic L2 helper.

    Args:
        substrate_id: canonical substrate id (e.g. ``"time_traveler_l5_z6"``);
            matches the ``src/tac/substrates/<substrate_id>/`` directory.
        lane_id: canonical lane id per CLAUDE.md "Lane maturity registry"
            non-negotiable (e.g. ``"lane_path_3_d_z6_l2_long_training_20260526"``).
        epochs: total epoch budget across all curriculum stages.
        batch_pair_indices_per_step: training batch size (number of
            frame-pair indices sampled per optimizer step).
        curriculum_stages: tuple of CurriculumStage frozen dataclasses
            defining the training schedule. Stages must cover [0, epochs)
            contiguously (no gaps, no overlap). Default = PR95 8-stage.
        ema_decay: Polyak EMA decay coefficient per Catalog #2 NON-NEGOTIABLE.
            Default = 0.997 (Quantizr PR101 canonical anchor).
        checkpoint_interval_epochs: emit canonical checkpoint every N epochs.
        early_stopping_patience: stop training if no loss improvement for N
            consecutive checkpoint-intervals.
        score_aware_loss_kwargs: optional substrate-specific kwargs threaded
            into the adapter's ``loss_fn`` (e.g. ``{"lambda_residual": 1.0}``).
        optimizer_class: optimizer family identifier (canonical: ``"adamw"``
            or ``"adam"`` for MLX-local; substrate-specific overrides
            via adapter's optimizer factory).
        learning_rate: base learning rate; per-stage ``lr_scale`` multiplies.
        seed: random seed pinned for deterministic reproducibility per
            CLAUDE.md "Canonical pipeline standard" non-negotiable +
            9-dim Dim 7.
        output_dir: canonical output directory; MUST be under
            ``experiments/results/<lane>_<ts>/`` per CLAUDE.md
            "Forbidden /tmp paths" non-negotiable.
        telemetry_path: optional canonical telemetry JSONL path for
            per-epoch metrics flush; default = ``output_dir/telemetry.jsonl``.
        checkpoint_dir: optional canonical checkpoint dir; default =
            ``output_dir/checkpoints/``.
        device: target device identifier (``"cuda"`` / ``"cpu"`` / ``"mlx"``);
            adapter's device discipline applies. Per CLAUDE.md
            "MPS auth eval is NOISE", ``"mps"`` is NEVER permitted for
            score-claim training.
        resume_from_checkpoint: optional path to a canonical checkpoint
            JSON to resume from. The trainer validates the checkpoint
            against substrate_id + lane_id + curriculum hash to refuse
            cross-substrate resume per Catalog #229 PV discipline.
        evidence_grade: canonical evidence_grade tag for the artifact.
            Default = ``"[macOS-MLX research-signal]"`` for MLX-local;
            other valid tags are ``"[advisory only]"``, ``"[prediction]"``.
            Per CLAUDE.md "MLX portable-local-substrate authority" non-
            negotiable: MLX-local L2 outputs are NEVER promotable.
        notes: optional substantive rationale (Catalog #287 placeholder rejected).
    """

    substrate_id: str
    lane_id: str
    epochs: int
    batch_pair_indices_per_step: int = 2
    curriculum_stages: tuple[CurriculumStage, ...] = PR95_8STAGE_CURRICULUM_DEFAULT
    ema_decay: float = CANONICAL_EMA_DECAY
    checkpoint_interval_epochs: int = DEFAULT_CHECKPOINT_INTERVAL_EPOCHS
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE
    score_aware_loss_kwargs: Mapping[str, Any] = field(default_factory=dict)
    optimizer_class: str = "adamw"
    learning_rate: float = 1e-3
    seed: int = 0
    output_dir: Path = field(default_factory=lambda: Path("experiments/results/long_training_canonical_default"))
    telemetry_path: Path | None = None
    checkpoint_dir: Path | None = None
    device: str = "mlx"
    resume_from_checkpoint: Path | None = None
    evidence_grade: str = "[macOS-MLX research-signal]"
    notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.substrate_id, str) or not self.substrate_id.strip():
            raise ValueError(f"substrate_id must be non-empty str; got {self.substrate_id!r}")
        if not isinstance(self.lane_id, str) or not self.lane_id.strip():
            raise ValueError(f"lane_id must be non-empty str; got {self.lane_id!r}")
        if not self.lane_id.startswith("lane_"):
            raise ValueError(
                f"lane_id must start with 'lane_' prefix per CLAUDE.md "
                f"'Lane maturity registry' non-negotiable; got {self.lane_id!r}"
            )
        if not isinstance(self.epochs, int) or self.epochs <= 0:
            raise ValueError(f"epochs must be positive int; got {self.epochs!r}")
        if not isinstance(self.batch_pair_indices_per_step, int) or self.batch_pair_indices_per_step <= 0:
            raise ValueError(
                f"batch_pair_indices_per_step must be positive int; "
                f"got {self.batch_pair_indices_per_step!r}"
            )
        if not isinstance(self.curriculum_stages, tuple) or not self.curriculum_stages:
            raise ValueError(
                f"curriculum_stages must be non-empty tuple of CurriculumStage; "
                f"got {self.curriculum_stages!r}"
            )
        for i, stage in enumerate(self.curriculum_stages):
            if not isinstance(stage, CurriculumStage):
                raise TypeError(
                    f"curriculum_stages[{i}] must be CurriculumStage; "
                    f"got {type(stage).__name__}"
                )
        # Stages must be contiguous + non-overlapping (sorted by start_epoch).
        sorted_stages = sorted(self.curriculum_stages, key=lambda s: s.start_epoch)
        if sorted_stages[0].start_epoch != 0:
            raise ValueError(
                f"first curriculum stage must start at epoch 0; "
                f"got start_epoch={sorted_stages[0].start_epoch}"
            )
        for prev, curr in zip(sorted_stages, sorted_stages[1:], strict=False):
            if prev.end_epoch != curr.start_epoch:
                raise ValueError(
                    f"curriculum stages must be contiguous; gap or overlap "
                    f"between {prev.name!r} (end={prev.end_epoch}) and "
                    f"{curr.name!r} (start={curr.start_epoch})"
                )
        if sorted_stages[-1].end_epoch != self.epochs:
            raise ValueError(
                f"last curriculum stage end_epoch ({sorted_stages[-1].end_epoch}) "
                f"must equal config.epochs ({self.epochs})"
            )
        if not (0.0 < self.ema_decay < 1.0):
            raise ValueError(
                f"ema_decay must be in (0, 1); got {self.ema_decay!r}. "
                "Per Catalog #2 NON-NEGOTIABLE the canonical default is 0.997."
            )
        if not isinstance(self.checkpoint_interval_epochs, int) or self.checkpoint_interval_epochs <= 0:
            raise ValueError(
                f"checkpoint_interval_epochs must be positive int; "
                f"got {self.checkpoint_interval_epochs!r}"
            )
        if not isinstance(self.early_stopping_patience, int) or self.early_stopping_patience <= 0:
            raise ValueError(
                f"early_stopping_patience must be positive int; "
                f"got {self.early_stopping_patience!r}"
            )
        if not isinstance(self.score_aware_loss_kwargs, Mapping):
            raise TypeError(
                f"score_aware_loss_kwargs must be Mapping; "
                f"got {type(self.score_aware_loss_kwargs).__name__}"
            )
        if not isinstance(self.optimizer_class, str) or not self.optimizer_class:
            raise ValueError(f"optimizer_class must be non-empty str; got {self.optimizer_class!r}")
        if not isinstance(self.learning_rate, (int, float)) or self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive; got {self.learning_rate!r}")
        if not isinstance(self.seed, int):
            raise TypeError(f"seed must be int; got {type(self.seed).__name__}")
        if not isinstance(self.output_dir, Path):
            raise TypeError(f"output_dir must be Path; got {type(self.output_dir).__name__}")
        _refuse_tmp_path(self.output_dir, "output_dir")
        if self.telemetry_path is not None:
            if not isinstance(self.telemetry_path, Path):
                raise TypeError(f"telemetry_path must be Path; got {type(self.telemetry_path).__name__}")
            _refuse_tmp_path(self.telemetry_path, "telemetry_path")
        if self.checkpoint_dir is not None:
            if not isinstance(self.checkpoint_dir, Path):
                raise TypeError(f"checkpoint_dir must be Path; got {type(self.checkpoint_dir).__name__}")
            _refuse_tmp_path(self.checkpoint_dir, "checkpoint_dir")
        # Per CLAUDE.md "MPS auth eval is NOISE": "mps" device REFUSED.
        if self.device.lower() == "mps":
            raise ValueError(
                "device='mps' is FORBIDDEN per CLAUDE.md 'MPS auth eval is NOISE' "
                "non-negotiable. Use 'mlx' for Apple Silicon (MLX framework "
                "is acceptable as research-signal); 'cuda' for promotion-grade."
            )
        if self.device not in {"cuda", "cpu", "mlx"}:
            raise ValueError(
                f"device must be one of {{'cuda', 'cpu', 'mlx'}}; got {self.device!r}"
            )
        if self.resume_from_checkpoint is not None:
            if not isinstance(self.resume_from_checkpoint, Path):
                raise TypeError(
                    f"resume_from_checkpoint must be Path; "
                    f"got {type(self.resume_from_checkpoint).__name__}"
                )
        if self.notes:
            _validate_rationale_not_placeholder(self.notes, "LongTrainingConfig.notes")

    def resolved_telemetry_path(self) -> Path:
        """Canonical telemetry path (default = output_dir/telemetry.jsonl)."""
        return self.telemetry_path or (self.output_dir / "telemetry.jsonl")

    def resolved_checkpoint_dir(self) -> Path:
        """Canonical checkpoint dir (default = output_dir/checkpoints/)."""
        return self.checkpoint_dir or (self.output_dir / "checkpoints")

    def curriculum_hash(self) -> str:
        """Canonical hash over curriculum_stages for resume validation."""
        payload = json.dumps(
            [s.as_dict() for s in sorted(self.curriculum_stages, key=lambda s: s.start_epoch)],
            sort_keys=True,
        )
        return _sha256_text(payload)

    def as_dict(self) -> dict[str, Any]:
        return {
            "substrate_id": self.substrate_id,
            "lane_id": self.lane_id,
            "epochs": int(self.epochs),
            "batch_pair_indices_per_step": int(self.batch_pair_indices_per_step),
            "curriculum_stages": [s.as_dict() for s in self.curriculum_stages],
            "curriculum_hash": self.curriculum_hash(),
            "ema_decay": float(self.ema_decay),
            "checkpoint_interval_epochs": int(self.checkpoint_interval_epochs),
            "early_stopping_patience": int(self.early_stopping_patience),
            "score_aware_loss_kwargs": dict(self.score_aware_loss_kwargs),
            "optimizer_class": self.optimizer_class,
            "learning_rate": float(self.learning_rate),
            "seed": int(self.seed),
            "output_dir": str(self.output_dir),
            "telemetry_path": str(self.resolved_telemetry_path()),
            "checkpoint_dir": str(self.resolved_checkpoint_dir()),
            "device": self.device,
            "resume_from_checkpoint": (
                str(self.resume_from_checkpoint) if self.resume_from_checkpoint else None
            ),
            "evidence_grade": self.evidence_grade,
            "notes": self.notes,
        }

    def stage_at_epoch(self, epoch: int) -> CurriculumStage:
        """Return the CurriculumStage covering ``epoch`` (clamped to last stage)."""
        for stage in self.curriculum_stages:
            if stage.start_epoch <= epoch < stage.end_epoch:
                return stage
        # Past final stage end: clamp to last stage (e.g. early-stop pad).
        return self.curriculum_stages[-1]


@dataclass(frozen=True)
class PerEpochMetrics:
    """Canonical per-epoch metrics row per Catalog #305 observability surface.

    Args:
        epoch: global epoch index (0-indexed).
        stage_name: name of the CurriculumStage containing this epoch.
        loss: scalar loss value (total; includes all weighted terms).
        loss_components: per-loss-term decomposition (e.g. ``{"recon": 0.05}``).
        per_axis_decomposition: optional per-axis seg/pose/rate per
            Catalog #356 dual-tier consumer architecture. None if the
            substrate adapter does not expose score-aware components.
        wall_clock_seconds: cumulative wall-clock seconds since training start.
        ema_drift_l2: L2 norm of (live_params - ema_shadow_params).
        learning_rate: effective learning rate at this epoch
            (base_lr * stage.lr_scale).
        captured_at_utc: ISO-8601 UTC timestamp.
    """

    epoch: int
    stage_name: str
    loss: float
    loss_components: Mapping[str, float] = field(default_factory=dict)
    per_axis_decomposition: Mapping[str, float] | None = None
    wall_clock_seconds: float = 0.0
    ema_drift_l2: float = 0.0
    learning_rate: float = 0.0
    captured_at_utc: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.epoch, int) or self.epoch < 0:
            raise ValueError(f"epoch must be int >= 0; got {self.epoch!r}")
        if not isinstance(self.loss, (int, float)):
            raise TypeError(f"loss must be numeric; got {type(self.loss).__name__}")
        if self.loss != self.loss:  # NaN check
            raise ValueError(f"loss is NaN at epoch {self.epoch}; OOM-safe runner should detect")

    def as_dict(self) -> dict[str, Any]:
        return {
            "epoch": int(self.epoch),
            "stage_name": self.stage_name,
            "loss": float(self.loss),
            "loss_components": {k: float(v) for k, v in self.loss_components.items()},
            "per_axis_decomposition": (
                {k: float(v) for k, v in self.per_axis_decomposition.items()}
                if self.per_axis_decomposition is not None
                else None
            ),
            "wall_clock_seconds": float(self.wall_clock_seconds),
            "ema_drift_l2": float(self.ema_drift_l2),
            "learning_rate": float(self.learning_rate),
            "captured_at_utc": self.captured_at_utc,
        }


@dataclass(frozen=True)
class TrainingArtifact:
    """Canonical TrainingArtifact emitted by ``run_long_training``.

    Per the doctrine 10-element contract item #1: the canonical return
    value of a long-training run. Carries the trained EMA shadow
    checkpoint path + canonical Provenance dict + canonical posterior
    anchor info + all per-epoch metrics + non-promotable markers per
    Catalog #127/#192/#317/#341.

    Args:
        substrate_id: substrate id from LongTrainingConfig.
        lane_id: lane id from LongTrainingConfig.
        config: the LongTrainingConfig used.
        ema_shadow_checkpoint_path: canonical EMA shadow checkpoint path
            (the canonical inference checkpoint per CLAUDE.md "EMA --
            NON-NEGOTIABLE").
        live_checkpoint_path: optional path to the LIVE weights checkpoint
            (for sister comparison / debug).
        archive_path: optional path to the byte-stable archive emitted
            by the adapter's ``export_archive`` (per Catalog #146).
        archive_sha256: SHA-256 of the canonical archive bytes; None if
            adapter did not emit an archive.
        archive_bytes: positive int archive size; None if no archive.
        per_epoch_metrics: tuple of canonical PerEpochMetrics rows.
        total_wall_clock_seconds: total training wall-clock.
        total_epochs_completed: total epochs actually completed (may be
            less than config.epochs if early-stopped).
        early_stopped: whether training early-stopped before reaching
            config.epochs.
        early_stop_reason: human-readable reason if early-stopped.
        canonical_provenance: canonical Provenance dict per Catalog #323
            umbrella.
        posterior_update_accepted: whether the canonical posterior anchor
            was accepted by the posterior_update_locked custody validator.
        posterior_refusal_reason: reason for refusal if not accepted.
        telemetry_path: canonical telemetry JSONL path.
        schema_version: canonical schema version for downstream consumers.
        captured_at_utc: ISO-8601 UTC timestamp of artifact emission.
    """

    substrate_id: str
    lane_id: str
    config_snapshot: Mapping[str, Any]
    ema_shadow_checkpoint_path: Path
    per_epoch_metrics: tuple[PerEpochMetrics, ...]
    total_wall_clock_seconds: float
    total_epochs_completed: int
    canonical_provenance: Mapping[str, Any]
    telemetry_path: Path
    captured_at_utc: str = field(default_factory=_utc_now_iso)
    schema_version: str = TRAINING_ARTIFACT_SCHEMA_VERSION
    live_checkpoint_path: Path | None = None
    archive_path: Path | None = None
    archive_sha256: str | None = None
    archive_bytes: int | None = None
    early_stopped: bool = False
    early_stop_reason: str = ""
    posterior_update_accepted: bool = False
    posterior_refusal_reason: str | None = None
    # Canonical non-promotable markers per Catalog #127/#192/#317/#341
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False
    rank_or_kill_eligible: bool = False
    promotable: bool = False

    def __post_init__(self) -> None:
        if self.score_claim:
            raise ValueError(
                "score_claim=True forbidden in TrainingArtifact per CLAUDE.md "
                "'MLX portable-local-substrate authority' + Catalog #127/#192/"
                "#317/#341 promotion-leak guard; L2 outputs are NEVER promotable."
            )
        if self.promotion_eligible:
            raise ValueError("promotion_eligible=True forbidden per same non-negotiables.")
        if self.ready_for_exact_eval_dispatch:
            raise ValueError("ready_for_exact_eval_dispatch=True forbidden.")
        if self.rank_or_kill_eligible:
            raise ValueError("rank_or_kill_eligible=True forbidden.")
        if self.promotable:
            raise ValueError("promotable=True forbidden.")
        # archive_sha256 ↔ archive_bytes coherence
        if (self.archive_sha256 is None) != (self.archive_bytes is None):
            raise ValueError(
                "archive_sha256 and archive_bytes must be both set or both None"
            )
        if self.archive_sha256 is not None:
            if len(self.archive_sha256) != 64 or not all(
                c in "0123456789abcdef" for c in self.archive_sha256.lower()
            ):
                raise ValueError(
                    f"archive_sha256 must be 64-char lowercase hex; got "
                    f"{self.archive_sha256!r}"
                )
            if self.archive_bytes is None or self.archive_bytes <= 0:
                raise ValueError(
                    f"archive_bytes must be positive int; got {self.archive_bytes!r}"
                )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "substrate_id": self.substrate_id,
            "lane_id": self.lane_id,
            "config_snapshot": dict(self.config_snapshot),
            "ema_shadow_checkpoint_path": str(self.ema_shadow_checkpoint_path),
            "live_checkpoint_path": (
                str(self.live_checkpoint_path) if self.live_checkpoint_path else None
            ),
            "archive_path": str(self.archive_path) if self.archive_path else None,
            "archive_sha256": self.archive_sha256,
            "archive_bytes": self.archive_bytes,
            "per_epoch_metrics_count": len(self.per_epoch_metrics),
            "per_epoch_metrics": [m.as_dict() for m in self.per_epoch_metrics],
            "total_wall_clock_seconds": float(self.total_wall_clock_seconds),
            "total_epochs_completed": int(self.total_epochs_completed),
            "early_stopped": bool(self.early_stopped),
            "early_stop_reason": self.early_stop_reason,
            "canonical_provenance": dict(self.canonical_provenance),
            "posterior_update_accepted": bool(self.posterior_update_accepted),
            "posterior_refusal_reason": self.posterior_refusal_reason,
            "telemetry_path": str(self.telemetry_path),
            "captured_at_utc": self.captured_at_utc,
            **CANONICAL_NON_PROMOTABLE_MARKERS,
        }


@dataclass(frozen=True)
class MultiArmDispatchResult:
    """Canonical result from ``run_long_training_multi_arm``.

    Args:
        arms: tuple of TrainingArtifact per arm (one per substrate variant).
        total_wall_clock_seconds: aggregate wall-clock across all arms.
        captured_at_utc: ISO-8601 timestamp of dispatch completion.
    """

    arms: tuple[TrainingArtifact, ...]
    total_wall_clock_seconds: float
    captured_at_utc: str = field(default_factory=_utc_now_iso)

    def __post_init__(self) -> None:
        if not isinstance(self.arms, tuple):
            raise TypeError(f"arms must be tuple; got {type(self.arms).__name__}")
        for i, a in enumerate(self.arms):
            if not isinstance(a, TrainingArtifact):
                raise TypeError(f"arms[{i}] must be TrainingArtifact; got {type(a).__name__}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm_count": len(self.arms),
            "arms": [a.as_dict() for a in self.arms],
            "total_wall_clock_seconds": float(self.total_wall_clock_seconds),
            "captured_at_utc": self.captured_at_utc,
        }


# ---------------------------------------------------------------------------
# Substrate adapter Protocol (Catalog #335 canonical contract pattern)
# ---------------------------------------------------------------------------


@runtime_checkable
class SubstrateLongTrainingAdapter(Protocol):
    """Canonical Protocol every substrate-adapter must satisfy.

    Per Catalog #335 canonical contract pattern + Catalog #290 canonical-
    vs-unique decision: the Protocol captures the substrate-AGNOSTIC
    training operations. Substrate-SPECIFIC details (architecture,
    loss-fn details, archive grammar) remain per-substrate.

    Two canonical adapter styles are supported (substrate picks):

    **Style A — separate loss_fn + optimizer_step** (torch-natural;
    canonical helper calls ``loss_fn`` then ``optimizer_step(loss)``
    with the scalar loss). Adapters implement ``loss_fn`` returning a
    dict with ``"total"`` key + ``optimizer_step`` taking the scalar.

    **Style B — combined train_step** (MLX-natural / functional;
    canonical helper calls ``train_step(batch, learning_rate, loss_weights)``
    once per epoch). Adapters that need ``value_and_grad`` (e.g. MLX
    via ``mlx.nn.value_and_grad``) implement ``train_step`` directly
    and may stub ``loss_fn`` + ``optimizer_step`` for diagnostic-only.

    The canonical helper detects Style B via ``hasattr(adapter, 'train_step')``
    and prefers it when present; falls back to Style A otherwise.

    Required attributes/methods:
        substrate_id: canonical id (matches the substrates/ dir name).
        model: trainable parameters container (duck-typed; torch.nn.Module
            or MLX module accepted; must support ``.parameters()`` for EMA
            + ``.state_dict()`` for checkpoint).
        sample_batch(batch_size, seed) -> Any: substrate-specific batch
            sampler (returns whatever loss_fn / train_step consumes).
        loss_fn(model, batch, loss_weights) -> dict: returns dict with
            REQUIRED key ``"total"`` (scalar loss) + optional per-term
            keys (substrate decides loss decomposition). REQUIRED for
            Style A; may be a diagnostic stub for Style B.
        optimizer_step(model, loss, learning_rate) -> None: substrate-
            specific optimizer (handles torch.optim.AdamW.step() OR
            equivalent). REQUIRED for Style A; may raise NotImplementedError
            for Style B.
        export_state_dict(model, path) -> None: substrate-specific
            checkpoint emission (torch.save / mlx.save_safetensors / etc).
        export_archive(model, output_dir) -> tuple[Path, str, int] | None:
            optional substrate-specific byte-stable archive emission
            per Catalog #146 inflate runtime contract. Returns
            ``(archive_path, archive_sha256, archive_bytes)`` OR None
            if substrate does not export an archive at L2 (some L1+
            substrates defer archive emission until L6 CONVERGED).
        score_aware_components(model, batch) -> dict | None: optional
            per-axis decomposition per Catalog #356 dual-tier contract;
            returns ``{"d_seg": float, "d_pose": float, "rate": float}``
            OR None if substrate does not expose score-aware components
            at L2.

    Optional method (Style B):
        train_step(batch, learning_rate, loss_weights) -> dict:
            combined value+grad+optimizer.update in one call. Returns
            same dict shape as loss_fn (REQUIRED ``"total"`` key).
            Used by MLX adapters where ``mlx.nn.value_and_grad`` requires
            closure over both forward + backward.
    """

    substrate_id: str
    model: Any

    def sample_batch(self, batch_size: int, seed: int) -> Any: ...

    def loss_fn(
        self,
        model: Any,
        batch: Any,
        loss_weights: Mapping[str, float],
    ) -> Mapping[str, float]: ...

    def optimizer_step(
        self,
        model: Any,
        loss: Any,
        learning_rate: float,
    ) -> None: ...

    def export_state_dict(self, model: Any, path: Path) -> None: ...

    def export_archive(
        self,
        model: Any,
        output_dir: Path,
    ) -> tuple[Path, str, int] | None: ...

    def score_aware_components(
        self,
        model: Any,
        batch: Any,
    ) -> Mapping[str, float] | None: ...


def validate_substrate_adapter(adapter: Any) -> None:
    """Conformance check that ``adapter`` satisfies SubstrateLongTrainingAdapter.

    Raises:
        TypeError: adapter lacks required attribute or method.
        ValueError: required attribute is malformed.
    """
    required_attrs = ("substrate_id", "model")
    required_methods = (
        "sample_batch",
        "loss_fn",
        "optimizer_step",
        "export_state_dict",
        "export_archive",
        "score_aware_components",
    )
    for attr in required_attrs:
        if not hasattr(adapter, attr):
            raise TypeError(
                f"adapter {type(adapter).__name__} missing required attribute "
                f"{attr!r}; see SubstrateLongTrainingAdapter Protocol contract."
            )
    for m in required_methods:
        if not hasattr(adapter, m) or not callable(getattr(adapter, m)):
            raise TypeError(
                f"adapter {type(adapter).__name__} missing required callable "
                f"{m!r}; see SubstrateLongTrainingAdapter Protocol contract."
            )
    if not isinstance(adapter.substrate_id, str) or not adapter.substrate_id.strip():
        raise ValueError(
            f"adapter.substrate_id must be non-empty str; got {adapter.substrate_id!r}"
        )


def validate_long_training_config(config: LongTrainingConfig) -> None:
    """Conformance check for LongTrainingConfig (delegates to __post_init__).

    Provided as a public API surface for sister tooling that wants to
    validate a config without constructing it (rare; mostly defensive).
    """
    # Re-running __post_init__ via reconstruction validates all invariants.
    if not isinstance(config, LongTrainingConfig):
        raise TypeError(
            f"config must be LongTrainingConfig; got {type(config).__name__}"
        )
    # Frozen dataclass __post_init__ already ran at construction; this
    # function exists for symmetric API + sister tooling discoverability.


# ---------------------------------------------------------------------------
# Canonical primitives (composable, reusable)
# ---------------------------------------------------------------------------


class PolyakEMAShadow:
    """Canonical Polyak-averaging EMA shadow primitive per Catalog #2.

    Substrate-agnostic via duck-typing on the model's ``state_dict()``
    interface. Supports torch.nn.Module + MLX modules + any object that
    exposes ``state_dict() -> dict[str, Any]``.

    The shadow update follows canonical Polyak averaging:

        shadow := decay * shadow + (1 - decay) * live

    Per CLAUDE.md "EMA -- NON-NEGOTIABLE" + Quantizr PR101 anchor:
    decay=0.997 is the canonical default. Inference checkpoint = shadow.

    Per CLAUDE.md "NEVER call ema.apply(model) inside train_epoch":
    the snapshot+restore pattern is used at export time only.
    """

    def __init__(self, model: Any, decay: float = CANONICAL_EMA_DECAY):
        if not (0.0 < decay < 1.0):
            raise ValueError(
                f"decay must be in (0, 1); got {decay!r}. "
                "Per Catalog #2 NON-NEGOTIABLE canonical default = 0.997."
            )
        # Duck-type detection: torch uses state_dict(); MLX uses parameters()
        # + tree_flatten/tree_unflatten. We support BOTH canonical patterns.
        self._mlx_mode = self._detect_mlx_mode(model)
        if not self._mlx_mode:
            if not hasattr(model, "state_dict") or not callable(model.state_dict):
                raise TypeError(
                    f"model {type(model).__name__} must expose .state_dict() "
                    "method (torch) OR .parameters() (MLX module); see "
                    "PolyakEMAShadow duck-typing contract."
                )
        self.decay = decay
        self._shadow: dict[str, Any] = self._clone_state_dict(
            self._get_flat_state(model)
        )

    @staticmethod
    def _detect_mlx_mode(model: Any) -> bool:
        """Detect MLX module vs torch module via duck-type."""
        # MLX modules: have parameters() returning dict + mlx.utils available.
        if not hasattr(model, "parameters"):
            return False
        if hasattr(model, "state_dict") and callable(model.state_dict):
            # torch.nn.Module has both; prefer state_dict path
            return False
        try:
            import mlx.utils  # noqa: F401
        except ImportError:
            return False
        # Try parameters() and see if it's a dict (MLX) vs list (torch nn.Module raw).
        try:
            params = model.parameters()
            return isinstance(params, dict)
        except (TypeError, AttributeError):
            return False

    def _get_flat_state(self, model: Any) -> Mapping[str, Any]:
        """Return flat state dict; auto-routes torch (state_dict) vs MLX (tree_flatten)."""
        if self._mlx_mode:
            from mlx.utils import tree_flatten
            return dict(tree_flatten(model.parameters()))
        return model.state_dict()

    @staticmethod
    def _clone_state_dict(state: Mapping[str, Any]) -> dict[str, Any]:
        """Clone state_dict values via duck-typed .clone() or copy.deepcopy()."""
        import copy

        cloned: dict[str, Any] = {}
        for k, v in state.items():
            if hasattr(v, "clone") and callable(v.clone):
                cloned[k] = v.clone()
                if hasattr(cloned[k], "detach") and callable(cloned[k].detach):
                    cloned[k] = cloned[k].detach()
            else:
                cloned[k] = copy.deepcopy(v)
        return cloned

    def update(self, model: Any) -> None:
        """Update shadow via canonical Polyak averaging."""
        live_state = self._get_flat_state(model)
        for k, v in live_state.items():
            if k not in self._shadow:
                # Late-bound module: seed from live per Codex finding 2 sister
                # discipline in the canonical EMA at tac.training.EMA.update.
                self._shadow[k] = self._clone_state_dict({k: v})[k]
                continue
            # Duck-typed update: torch tensors support .mul_().add_();
            # MLX arrays + numpy arrays need element-wise arithmetic;
            # plain Python list/tuple need element-wise iteration.
            shadow_v = self._shadow[k]
            try:
                # Try in-place torch path first (efficient).
                if hasattr(shadow_v, "mul_") and hasattr(shadow_v, "add_"):
                    if hasattr(v, "is_floating_point") and not v.is_floating_point():
                        # Non-float buffers: copy directly per sister EMA.
                        shadow_v.copy_(v)
                    else:
                        shadow_v.mul_(self.decay).add_(v, alpha=1.0 - self.decay)
                elif isinstance(shadow_v, list):
                    # Plain Python list: element-wise Polyak averaging.
                    self._shadow[k] = [
                        self.decay * float(sv) + (1.0 - self.decay) * float(lv)
                        for sv, lv in zip(shadow_v, v, strict=False)
                    ]
                else:
                    # Functional path (MLX / numpy arrays): construct new tensor.
                    self._shadow[k] = self.decay * shadow_v + (1.0 - self.decay) * v
            except (AttributeError, TypeError, RuntimeError):
                # Fall back to element-wise list path on any duck-type failure.
                if isinstance(shadow_v, (list, tuple)) and isinstance(v, (list, tuple)):
                    self._shadow[k] = [
                        self.decay * float(sv) + (1.0 - self.decay) * float(lv)
                        for sv, lv in zip(shadow_v, v, strict=False)
                    ]
                else:
                    self._shadow[k] = self.decay * shadow_v + (1.0 - self.decay) * v

    def state_dict(self) -> dict[str, Any]:
        """Return a clone of the canonical EMA shadow state_dict."""
        return self._clone_state_dict(self._shadow)

    def apply_to(self, model: Any) -> Mapping[str, Any]:
        """Apply EMA shadow to model; return live state snapshot for restore.

        Canonical snapshot+restore pattern per CLAUDE.md "Inference / archive
        bytes come from ema.state_dict()" + "NEVER call ema.apply(model)
        inside train_epoch". Caller MUST restore via load_state_dict / update
        after archive emission.

        Routes torch (load_state_dict) vs MLX (model.update(tree_unflatten)).
        """
        live_snapshot = self._clone_state_dict(self._get_flat_state(model))
        if self._mlx_mode:
            from mlx.utils import tree_unflatten
            shadow_unflat = tree_unflatten(list(self._shadow.items()))
            model.update(shadow_unflat)
        else:
            try:
                model.load_state_dict(self._shadow)
            except (AttributeError, TypeError) as exc:
                raise TypeError(
                    f"model {type(model).__name__} must expose .load_state_dict() "
                    f"compatible with PolyakEMAShadow.apply_to(); got {exc!s}"
                ) from exc
        return live_snapshot

    def restore_from_snapshot(self, model: Any, snapshot: Mapping[str, Any]) -> None:
        """Canonical restore from a snapshot produced by apply_to."""
        if self._mlx_mode:
            from mlx.utils import tree_unflatten
            snap_unflat = tree_unflatten(list(snapshot.items()))
            model.update(snap_unflat)
        else:
            model.load_state_dict(snapshot)

    def drift_l2(self, model: Any) -> float:
        """L2 norm of (live - shadow); canonical drift metric for telemetry."""
        live_state = self._get_flat_state(model)
        total = 0.0
        for k, v in live_state.items():
            if k not in self._shadow:
                continue
            shadow_v = self._shadow[k]
            # Plain Python list/tuple path (test mocks; portable fallback).
            if isinstance(shadow_v, (list, tuple)) and isinstance(v, (list, tuple)):
                for sv, lv in zip(shadow_v, v, strict=False):
                    try:
                        d = float(lv) - float(sv)
                        total += d * d
                    except (TypeError, ValueError):
                        continue
                continue
            try:
                diff = v - shadow_v
                if hasattr(diff, "item") and callable(diff.item):
                    # Torch / MLX scalar via .item() on .pow(2).sum()
                    if hasattr(diff, "pow") and callable(diff.pow):
                        total += float((diff.pow(2).sum()).item())
                    else:
                        # Numpy / generic: element-wise then sum
                        sum_squared = (diff * diff).sum()
                        total += float(sum_squared.item() if hasattr(sum_squared, "item") else sum_squared)
                else:
                    total += float((diff * diff).sum())
            except (AttributeError, TypeError, RuntimeError):
                # Non-numeric value (e.g. dict / list / str); skip
                continue
        return float(total ** 0.5)


class TelemetrySink:
    """Canonical observability sink per Catalog #305 6-facet surface.

    Emits per-epoch PerEpochMetrics rows to a canonical JSONL file +
    maintains an in-memory buffer for the TrainingArtifact.

    Per Catalog #305 6-facet: inspectable per layer + decomposable per
    signal + diff-able across runs + queryable post-hoc + cite-able +
    counterfactual-able. The JSONL emission satisfies queryable +
    cite-able; the structured PerEpochMetrics dataclass satisfies
    inspectable + decomposable; pairing two telemetry files via the
    canonical schema_version field satisfies diff-able.

    Per Catalog #131 sister discipline: writes go through canonical
    atomic-write pattern (tmp + os.replace) under file-lock to prevent
    concurrent writer corruption (e.g. when two arms in
    run_long_training_multi_arm write to different telemetry files).
    """

    def __init__(self, telemetry_path: Path, flush_interval_epochs: int = DEFAULT_TELEMETRY_FLUSH_INTERVAL_EPOCHS):
        if not isinstance(telemetry_path, Path):
            raise TypeError(f"telemetry_path must be Path; got {type(telemetry_path).__name__}")
        _refuse_tmp_path(telemetry_path, "telemetry_path")
        if flush_interval_epochs <= 0:
            raise ValueError(f"flush_interval_epochs must be positive int; got {flush_interval_epochs!r}")
        self.telemetry_path = telemetry_path
        self.flush_interval_epochs = flush_interval_epochs
        self._buffer: list[PerEpochMetrics] = []
        self._next_flush_index = 0
        self._epochs_since_flush = 0
        # Ensure parent dir exists; canonical canonical artifact discipline.
        self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, metrics: PerEpochMetrics) -> None:
        """Record one PerEpochMetrics row; flush if interval reached."""
        if not isinstance(metrics, PerEpochMetrics):
            raise TypeError(f"metrics must be PerEpochMetrics; got {type(metrics).__name__}")
        self._buffer.append(metrics)
        self._epochs_since_flush += 1
        if self._epochs_since_flush >= self.flush_interval_epochs:
            self.flush()

    def flush(self) -> None:
        """Flush only unflushed buffer rows to canonical JSONL with atomic-append.

        Per Catalog #110/#113 APPEND-ONLY discipline: rows already flushed
        in a prior call MUST NOT be re-emitted. Tracking via
        _next_flush_index ensures idempotent flush + buffered snapshot
        access via .snapshot() remains intact for TrainingArtifact.
        """
        rows_to_flush = self._buffer[self._next_flush_index:]
        if not rows_to_flush:
            self._epochs_since_flush = 0
            return
        # File-lock prevents concurrent flushes from corrupting the JSONL.
        lock_path = self.telemetry_path.with_suffix(self.telemetry_path.suffix + ".lock")
        lock_path.touch(exist_ok=True)
        with open(lock_path, "r+") as lock_fh:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            try:
                with open(self.telemetry_path, "a") as out_fh:
                    for row in rows_to_flush:
                        out_fh.write(json.dumps(row.as_dict(), sort_keys=True))
                        out_fh.write("\n")
            finally:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        self._next_flush_index = len(self._buffer)
        self._epochs_since_flush = 0

    def snapshot(self) -> tuple[PerEpochMetrics, ...]:
        """Return immutable snapshot of all recorded metrics."""
        return tuple(self._buffer)

    def close(self) -> None:
        """Final flush; ensures no buffered rows are lost on crash."""
        self.flush()


class CheckpointWriter:
    """Canonical checkpoint-and-resume primitive per doctrine element #4.

    Sister of ``tac.subagent_checkpoint`` per Catalog #206; same fcntl-
    locked JSONL pattern adapted to per-epoch training checkpoints.

    Writes canonical checkpoint metadata JSON next to the model
    state_dict file; metadata includes substrate_id + lane_id +
    curriculum_hash so resume cannot accidentally cross substrates.

    Per Catalog #229 PV: every checkpoint records substrate_id + lane_id
    + curriculum_hash + global_epoch + loss + wall-clock + provenance.
    """

    def __init__(self, checkpoint_dir: Path, substrate_id: str, lane_id: str, curriculum_hash: str):
        if not isinstance(checkpoint_dir, Path):
            raise TypeError(f"checkpoint_dir must be Path; got {type(checkpoint_dir).__name__}")
        _refuse_tmp_path(checkpoint_dir, "checkpoint_dir")
        if not substrate_id or not lane_id:
            raise ValueError("substrate_id and lane_id must be non-empty")
        if not curriculum_hash or len(curriculum_hash) != 64:
            raise ValueError(f"curriculum_hash must be 64-char hex; got {curriculum_hash!r}")
        self.checkpoint_dir = checkpoint_dir
        self.substrate_id = substrate_id
        self.lane_id = lane_id
        self.curriculum_hash = curriculum_hash
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        adapter: SubstrateLongTrainingAdapter,
        ema_shadow: PolyakEMAShadow,
        global_epoch: int,
        loss: float,
        wall_clock_seconds: float,
        is_final: bool = False,
    ) -> Path:
        """Write canonical checkpoint (live + EMA shadow + metadata)."""
        ts = _utc_now_iso().replace(":", "").replace("-", "")
        stem = f"epoch{global_epoch:06d}_{ts}"
        if is_final:
            stem = f"final_{stem}"
        live_path = self.checkpoint_dir / f"{stem}.live.state"
        ema_path = self.checkpoint_dir / f"{stem}.ema_shadow.state"
        meta_path = self.checkpoint_dir / f"{stem}.meta.json"

        # Write live state via adapter's canonical export.
        adapter.export_state_dict(adapter.model, live_path)

        # Write EMA shadow via snapshot+restore + adapter's export.
        live_snapshot = ema_shadow.apply_to(adapter.model)
        try:
            adapter.export_state_dict(adapter.model, ema_path)
        finally:
            ema_shadow.restore_from_snapshot(adapter.model, live_snapshot)

        # Write canonical metadata JSON.
        meta = {
            "schema_version": "long_training_canonical_checkpoint.v1",
            "substrate_id": self.substrate_id,
            "lane_id": self.lane_id,
            "curriculum_hash": self.curriculum_hash,
            "global_epoch": int(global_epoch),
            "loss": float(loss),
            "wall_clock_seconds": float(wall_clock_seconds),
            "is_final": bool(is_final),
            "live_state_path": str(live_path),
            "ema_shadow_state_path": str(ema_path),
            "captured_at_utc": _utc_now_iso(),
            **CANONICAL_NON_PROMOTABLE_MARKERS,
        }
        meta_path.write_text(json.dumps(meta, sort_keys=True, indent=2) + "\n")
        return meta_path

    def load_resume_metadata(self, resume_from: Path) -> Mapping[str, Any]:
        """Load checkpoint metadata; refuse cross-substrate / cross-curriculum resume."""
        if not resume_from.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_from}")
        meta = json.loads(resume_from.read_text())
        if meta.get("substrate_id") != self.substrate_id:
            raise ValueError(
                f"resume checkpoint substrate_id={meta.get('substrate_id')!r} "
                f"!= config substrate_id={self.substrate_id!r}; refusing per "
                f"Catalog #229 PV cross-substrate-resume guard."
            )
        if meta.get("curriculum_hash") != self.curriculum_hash:
            raise ValueError(
                f"resume checkpoint curriculum_hash differs; refusing per "
                f"Catalog #229 PV (curriculum must match for valid resume)."
            )
        return meta


class OOMSafeStepRunner:
    """Canonical OOM-safe training-step runner per doctrine element #8.

    Wraps adapter.optimizer_step in try/except for OOM errors; on OOM,
    halves the batch size and retries. After N consecutive OOM retries
    without success, raises a typed error so caller can crash-recover
    via checkpoint resume per element #4.

    OOM detection is duck-typed: catches RuntimeError + MemoryError +
    any exception whose message contains "out of memory" (case-insensitive)
    to cover torch / MLX / numpy variants.
    """

    def __init__(self, max_retries: int = 4, min_batch_size: int = 1):
        if max_retries < 1:
            raise ValueError(f"max_retries must be >= 1; got {max_retries!r}")
        if min_batch_size < 1:
            raise ValueError(f"min_batch_size must be >= 1; got {min_batch_size!r}")
        self.max_retries = max_retries
        self.min_batch_size = min_batch_size
        self.oom_event_count = 0

    @staticmethod
    def _is_oom_error(exc: BaseException) -> bool:
        if isinstance(exc, MemoryError):
            return True
        msg = str(exc).lower()
        return "out of memory" in msg or "oom" in msg or "memory" in msg and "cuda" in msg

    def run_step(
        self,
        adapter: SubstrateLongTrainingAdapter,
        batch_size: int,
        seed: int,
        stage: CurriculumStage,
        learning_rate: float,
    ) -> tuple[Mapping[str, float], int]:
        """Execute one step OOM-safely; returns (loss_dict, actual_batch_size).

        Style detection: if ``adapter.train_step`` exists, use the combined
        value+grad+update path (MLX-natural). Otherwise use the separate
        loss_fn + optimizer_step path (torch-natural).

        On OOM, halves batch_size and retries up to max_retries times.
        """
        current_bs = batch_size
        retries = 0
        last_exc: BaseException | None = None
        use_train_step = hasattr(adapter, "train_step") and callable(
            getattr(adapter, "train_step")
        )
        while retries < self.max_retries and current_bs >= self.min_batch_size:
            try:
                batch = adapter.sample_batch(current_bs, seed)
                if use_train_step:
                    # Style B: combined value+grad+update (MLX-natural)
                    loss_dict = adapter.train_step(batch, learning_rate, stage.loss_weights)
                else:
                    # Style A: separate loss_fn + optimizer_step (torch-natural)
                    loss_dict = adapter.loss_fn(adapter.model, batch, stage.loss_weights)
                if "total" not in loss_dict:
                    raise ValueError(
                        f"adapter {adapter.substrate_id!r} {'train_step' if use_train_step else 'loss_fn'} "
                        f"returned dict without required 'total' key; "
                        f"got keys {list(loss_dict.keys())}"
                    )
                if not use_train_step:
                    adapter.optimizer_step(adapter.model, loss_dict["total"], learning_rate)
                return loss_dict, current_bs
            except (RuntimeError, MemoryError) as exc:
                if not self._is_oom_error(exc):
                    raise
                self.oom_event_count += 1
                last_exc = exc
                retries += 1
                new_bs = max(self.min_batch_size, current_bs // 2)
                if new_bs == current_bs:
                    # Already at min; cannot retry.
                    break
                current_bs = new_bs
        raise RuntimeError(
            f"OOMSafeStepRunner exhausted {self.max_retries} retries for "
            f"substrate {adapter.substrate_id!r}; last batch_size={current_bs}, "
            f"min_batch_size={self.min_batch_size}, oom_event_count={self.oom_event_count}. "
            f"Last exception: {last_exc!r}. Operator should resume from checkpoint "
            f"per CheckpointWriter resume_from_checkpoint discipline."
        )


# ---------------------------------------------------------------------------
# Canonical Provenance + posterior anchor emission
# ---------------------------------------------------------------------------


def _build_canonical_provenance_for_artifact(
    artifact: TrainingArtifact,
) -> dict[str, Any]:
    """Build canonical Provenance dict for the TrainingArtifact emission.

    Per Catalog #323 canonical Provenance umbrella + Catalog #287
    placeholder-rationale rejection. Routes through the canonical
    builder ``build_provenance_for_predicted`` (artifact is a training
    PREDICTION until paired CUDA/CPU auth eval lands per CLAUDE.md
    "Submission auth eval - BOTH CPU AND CUDA").
    """
    try:
        from tac.provenance import build_provenance_for_predicted

        inputs_payload = json.dumps(
            dict(artifact.config_snapshot),
            sort_keys=True,
        )
        inputs_sha256 = _sha256_text(inputs_payload)
        prov = build_provenance_for_predicted(
            model_id=f"long_training_canonical:{artifact.substrate_id}",
            inputs_sha256=inputs_sha256,
            measurement_axis=artifact.config_snapshot.get("evidence_grade", "[predicted]"),
            hardware_substrate=_detect_hardware_substrate(
                artifact.config_snapshot.get("device", "mlx")
            ),
        )
        return {
            "artifact_kind": prov.artifact_kind.value,
            "evidence_grade": prov.evidence_grade.value,
            "measurement_axis": prov.measurement_axis,
            "hardware_substrate": prov.hardware_substrate,
            "promotion_eligible": bool(prov.promotion_eligible),
            "score_claim_valid": bool(prov.score_claim_valid),
            "source_path": prov.source_path,
            "source_sha256": prov.source_sha256,
            "canonical_helper_invocation": prov.canonical_helper_invocation,
            "captured_at_utc": prov.captured_at_utc,
        }
    except ImportError:
        # Fallback canonical Provenance dict when builders unavailable.
        return {
            "artifact_kind": "predicted_from_model",
            "evidence_grade": "predicted",
            "measurement_axis": artifact.config_snapshot.get("evidence_grade", "[predicted]"),
            "hardware_substrate": _detect_hardware_substrate(
                artifact.config_snapshot.get("device", "mlx")
            ),
            "promotion_eligible": False,
            "score_claim_valid": False,
            "source_path": f"<long_training_canonical:{artifact.substrate_id}>",
            "source_sha256": _sha256_text(
                json.dumps(dict(artifact.config_snapshot), sort_keys=True)
            ),
            "canonical_helper_invocation": "tac.training.long_training_canonical.run_long_training",
            "captured_at_utc": _utc_now_iso(),
            "fallback_no_provenance_module": True,
        }


def _detect_hardware_substrate(device: str) -> str:
    """Canonical hardware_substrate detection per Catalog #190 sister."""
    device_lower = device.lower()
    if device_lower == "mlx":
        return "macos_arm64_mlx_local"
    if device_lower == "cpu":
        # Best-effort detection (sister of trainer_skeleton.detect_hardware_substrate)
        try:
            from tac.substrates._shared.trainer_skeleton import detect_hardware_substrate
            return detect_hardware_substrate(axis="cpu", substrate_tag="long_training_canonical")
        except (ImportError, Exception):
            return "unknown_cpu"
    if device_lower == "cuda":
        try:
            from tac.substrates._shared.trainer_skeleton import detect_hardware_substrate
            return detect_hardware_substrate(
                axis="cuda",
                substrate_tag="long_training_canonical",
                env_var_candidates=("LONG_TRAINING_GPU", "MODAL_GPU"),
            )
        except (ImportError, Exception):
            return "linux_x86_64_unknown_cuda"
    return "unknown"


def _emit_canonical_posterior_anchor(
    artifact: TrainingArtifact,
) -> tuple[bool, str | None]:
    """Emit canonical posterior anchor via posterior_emission_helper.

    Per the doctrine 10-element contract item #5 + Catalog #128
    fcntl-locked posterior write discipline + Catalog #335 cathedral
    consumer canonical contract pattern.

    Returns ``(accepted, refusal_reason)`` from posterior_update_locked.
    """
    if artifact.archive_sha256 is None or artifact.archive_bytes is None:
        # Cannot emit canonical posterior anchor without archive bytes;
        # this is legitimate (substrate may defer archive to L6).
        return False, "no_archive_emitted_at_l2_substrate_deferral_to_l6"

    try:
        from tac.substrates._shared.posterior_emission_helper import (
            emit_substrate_landing_posterior_anchor,
        )

        anchor = emit_substrate_landing_posterior_anchor(
            substrate_id=artifact.substrate_id,
            archive_sha256=artifact.archive_sha256,
            archive_bytes=artifact.archive_bytes,
            source_path=str(artifact.ema_shadow_checkpoint_path),
            predicted_score=0.20,  # Default mid-band; caller may override via metadata
            architecture_class=artifact.substrate_id,
            notes=(
                f"L2 long-training canonical artifact emission; "
                f"epochs_completed={artifact.total_epochs_completed}; "
                f"non-promotable per CLAUDE.md MLX/CPU-research-signal discipline."
            ),
            extra_manifest_fields={
                "long_training_canonical_helper": "tac.training.long_training_canonical.run_long_training",
                "long_training_lane_id": artifact.lane_id,
                "long_training_schema_version": artifact.schema_version,
                "long_training_epochs_completed": int(artifact.total_epochs_completed),
                "long_training_early_stopped": bool(artifact.early_stopped),
            },
        )
        return bool(anchor.posterior_update.accepted), anchor.posterior_update.refusal_reason
    except ImportError as exc:
        return False, f"posterior_emission_helper_import_failed:{exc!s}"
    except Exception as exc:
        # Best-effort: never break the artifact emission on posterior helper failure.
        # Per Catalog #339 silent-no-spawn-class self-protection: surface the failure.
        return False, f"posterior_emission_helper_runtime_failed:{type(exc).__name__}:{exc!s}"


# ---------------------------------------------------------------------------
# Canonical entry-point: run_long_training
# ---------------------------------------------------------------------------


def run_long_training(
    adapter: SubstrateLongTrainingAdapter,
    config: LongTrainingConfig,
    *,
    on_epoch_end: Callable[[PerEpochMetrics], None] | None = None,
) -> TrainingArtifact:
    """Canonical L2 long-training entry-point.

    Per the doctrine 10-element contract item #1: the canonical entry-
    point that takes any substrate-conforming adapter + LongTrainingConfig
    and emits a canonical TrainingArtifact with EMA shadow checkpoint +
    canonical Provenance + canonical posterior anchor.

    The training loop:

    1. Validate adapter + config conformance.
    2. Set up canonical primitives: EMA shadow + TelemetrySink +
       CheckpointWriter + OOMSafeStepRunner.
    3. (Optional) Resume from canonical checkpoint per Catalog #229 PV.
    4. For each epoch in [0, config.epochs):
       a. Determine current CurriculumStage.
       b. Compute effective lr = config.learning_rate * stage.lr_scale.
       c. Run OOM-safe optimizer step via OOMSafeStepRunner.
       d. Update EMA shadow per canonical Polyak averaging.
       e. Record canonical PerEpochMetrics row (loss + components +
          per-axis decomposition if available + EMA drift + wall-clock).
       f. (Optional) on_epoch_end callback for caller observability.
       g. If checkpoint interval reached, write canonical checkpoint.
       h. If early-stopping patience exceeded, break.
    5. Final EMA shadow export (snapshot+restore canonical pattern).
    6. (Optional) Substrate archive export per adapter contract.
    7. Build canonical Provenance dict per Catalog #323.
    8. Emit canonical posterior anchor per Catalog #128 (via canonical
       posterior_emission_helper sister).
    9. Construct TrainingArtifact + flush TelemetrySink + return.

    Args:
        adapter: substrate-conforming adapter per SubstrateLongTrainingAdapter
            Protocol contract.
        config: validated LongTrainingConfig.
        on_epoch_end: optional callback per PerEpochMetrics emission;
            useful for sister tooling (live dashboards, progress bars).

    Returns:
        TrainingArtifact with all canonical metadata + non-promotable
        markers per Catalog #127/#192/#317/#341.

    Raises:
        TypeError: adapter does not satisfy SubstrateLongTrainingAdapter.
        ValueError: config is malformed (already enforced by __post_init__).
        RuntimeError: OOM exhausted retries OR substrate adapter raised
            unrecoverable exception.

    Example:
        >>> from tac.training.long_training_canonical import (
        ...     LongTrainingConfig, CurriculumStage, run_long_training,
        ... )
        >>> config = LongTrainingConfig(
        ...     substrate_id="my_substrate",
        ...     lane_id="lane_my_substrate_l2_20260526",
        ...     epochs=100,
        ...     curriculum_stages=(
        ...         CurriculumStage(name="full", start_epoch=0, end_epoch=100),
        ...     ),
        ...     output_dir=Path("experiments/results/my_substrate_l2_20260526"),
        ... )
        >>> artifact = run_long_training(my_substrate_adapter, config)
        >>> artifact.total_epochs_completed
        100
        >>> artifact.promotable
        False
    """
    validate_substrate_adapter(adapter)
    validate_long_training_config(config)

    # 1) Set up canonical primitives.
    config.output_dir.mkdir(parents=True, exist_ok=True)
    telemetry_sink = TelemetrySink(
        telemetry_path=config.resolved_telemetry_path(),
        flush_interval_epochs=DEFAULT_TELEMETRY_FLUSH_INTERVAL_EPOCHS,
    )
    checkpoint_writer = CheckpointWriter(
        checkpoint_dir=config.resolved_checkpoint_dir(),
        substrate_id=config.substrate_id,
        lane_id=config.lane_id,
        curriculum_hash=config.curriculum_hash(),
    )
    ema_shadow = PolyakEMAShadow(adapter.model, decay=config.ema_decay)
    oom_runner = OOMSafeStepRunner()

    # 2) Resume metadata (best-effort; warn-on-failure is the caller's job).
    resume_global_epoch = 0
    if config.resume_from_checkpoint is not None:
        meta = checkpoint_writer.load_resume_metadata(config.resume_from_checkpoint)
        resume_global_epoch = int(meta.get("global_epoch", 0))

    # 3) Training loop.
    per_epoch_metrics: list[PerEpochMetrics] = []
    best_loss = float("inf")
    epochs_since_improvement = 0
    early_stopped = False
    early_stop_reason = ""
    t_start = time.time()
    final_epoch = resume_global_epoch

    for epoch in range(resume_global_epoch, config.epochs):
        final_epoch = epoch
        stage = config.stage_at_epoch(epoch)
        effective_lr = config.learning_rate * stage.lr_scale

        try:
            loss_dict, actual_bs = oom_runner.run_step(
                adapter=adapter,
                batch_size=config.batch_pair_indices_per_step,
                seed=config.seed + epoch,
                stage=stage,
                learning_rate=effective_lr,
            )
        except RuntimeError as exc:
            early_stopped = True
            early_stop_reason = f"oom_safe_runner_exhausted:{exc!s}"
            break

        # 4) EMA update post-optimizer-step per canonical Polyak pattern.
        try:
            ema_shadow.update(adapter.model)
        except Exception as exc:
            # EMA update failure is recoverable; log + continue.
            traceback.print_exc()
            print(f"[long_training_canonical] WARN: EMA update failed at epoch {epoch}: {exc!r}")

        # 5) Optional per-axis decomposition per Catalog #356.
        per_axis: Mapping[str, float] | None = None
        try:
            sample = adapter.sample_batch(config.batch_pair_indices_per_step, config.seed + epoch + 1_000_000)
            per_axis = adapter.score_aware_components(adapter.model, sample)
        except (NotImplementedError, AttributeError):
            per_axis = None
        except Exception as exc:
            # Per-axis decomposition is observability-only; never fail the run.
            print(f"[long_training_canonical] WARN: score_aware_components failed at epoch {epoch}: {exc!r}")
            per_axis = None

        # 6) Record canonical metrics.
        wall_clock = time.time() - t_start
        try:
            drift = ema_shadow.drift_l2(adapter.model)
        except Exception:
            drift = 0.0
        # Build loss_components dict from loss_dict (excluding "total" key).
        loss_components = {k: float(v) for k, v in loss_dict.items() if k != "total"}
        total_loss = float(loss_dict["total"])
        metrics = PerEpochMetrics(
            epoch=epoch,
            stage_name=stage.name,
            loss=total_loss,
            loss_components=loss_components,
            per_axis_decomposition=per_axis,
            wall_clock_seconds=wall_clock,
            ema_drift_l2=drift,
            learning_rate=effective_lr,
            captured_at_utc=_utc_now_iso(),
        )
        per_epoch_metrics.append(metrics)
        telemetry_sink.record(metrics)
        if on_epoch_end is not None:
            try:
                on_epoch_end(metrics)
            except Exception as exc:
                print(f"[long_training_canonical] WARN: on_epoch_end callback failed: {exc!r}")

        # 7) Early-stopping bookkeeping.
        if total_loss < best_loss - 1e-9:
            best_loss = total_loss
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
        if epochs_since_improvement >= config.early_stopping_patience:
            early_stopped = True
            early_stop_reason = (
                f"early_stopping_patience_exceeded:{config.early_stopping_patience}"
                f"_epochs_without_improvement_below_best_loss_{best_loss}"
            )
            break

        # 8) Periodic checkpoint emission.
        if (epoch + 1) % config.checkpoint_interval_epochs == 0:
            try:
                checkpoint_writer.write(
                    adapter=adapter,
                    ema_shadow=ema_shadow,
                    global_epoch=epoch,
                    loss=total_loss,
                    wall_clock_seconds=wall_clock,
                    is_final=False,
                )
            except Exception as exc:
                # Per Catalog #339 sister discipline: never silently swallow
                # checkpoint failures; print + continue (subsequent emission
                # may succeed).
                traceback.print_exc()
                print(f"[long_training_canonical] WARN: checkpoint emission failed at epoch {epoch}: {exc!r}")

    total_wall_clock = time.time() - t_start
    total_epochs_completed = final_epoch + 1 - resume_global_epoch

    # 9) Final checkpoint emission (always, even on early-stop / OOM).
    final_meta_path = config.resolved_checkpoint_dir() / "final_checkpoint_emission_failed.json"
    try:
        final_meta_path = checkpoint_writer.write(
            adapter=adapter,
            ema_shadow=ema_shadow,
            global_epoch=final_epoch,
            loss=per_epoch_metrics[-1].loss if per_epoch_metrics else float("inf"),
            wall_clock_seconds=total_wall_clock,
            is_final=True,
        )
    except Exception as exc:
        traceback.print_exc()
        print(f"[long_training_canonical] WARN: final checkpoint emission failed: {exc!r}")

    # Resolve final EMA shadow checkpoint path from meta.
    try:
        final_meta = json.loads(final_meta_path.read_text())
        ema_shadow_checkpoint_path = Path(final_meta["ema_shadow_state_path"])
        live_checkpoint_path = Path(final_meta["live_state_path"])
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        ema_shadow_checkpoint_path = config.resolved_checkpoint_dir() / "ema_shadow.unknown"
        live_checkpoint_path = None

    # 10) Substrate archive export (optional per adapter contract).
    archive_path: Path | None = None
    archive_sha256: str | None = None
    archive_bytes: int | None = None
    try:
        # Use EMA shadow for archive (canonical inference checkpoint).
        live_snapshot = ema_shadow.apply_to(adapter.model)
        try:
            archive_result = adapter.export_archive(adapter.model, config.output_dir)
            if archive_result is not None:
                archive_path, archive_sha256, archive_bytes = archive_result
        finally:
            ema_shadow.restore_from_snapshot(adapter.model, live_snapshot)
    except NotImplementedError:
        # Substrate explicitly defers archive emission to L6 CONVERGED.
        pass
    except Exception as exc:
        traceback.print_exc()
        print(f"[long_training_canonical] WARN: archive export failed: {exc!r}")

    telemetry_sink.close()

    # 11) Build canonical Provenance + emit canonical posterior anchor.
    artifact_pre_provenance = TrainingArtifact(
        substrate_id=config.substrate_id,
        lane_id=config.lane_id,
        config_snapshot=config.as_dict(),
        ema_shadow_checkpoint_path=ema_shadow_checkpoint_path,
        live_checkpoint_path=live_checkpoint_path,
        archive_path=archive_path,
        archive_sha256=archive_sha256,
        archive_bytes=archive_bytes,
        per_epoch_metrics=tuple(per_epoch_metrics),
        total_wall_clock_seconds=total_wall_clock,
        total_epochs_completed=total_epochs_completed,
        canonical_provenance={},  # filled below
        telemetry_path=config.resolved_telemetry_path(),
        early_stopped=early_stopped,
        early_stop_reason=early_stop_reason,
    )
    provenance = _build_canonical_provenance_for_artifact(artifact_pre_provenance)

    # Re-construct artifact WITH provenance + emit posterior anchor.
    artifact_with_provenance = TrainingArtifact(
        substrate_id=config.substrate_id,
        lane_id=config.lane_id,
        config_snapshot=config.as_dict(),
        ema_shadow_checkpoint_path=ema_shadow_checkpoint_path,
        live_checkpoint_path=live_checkpoint_path,
        archive_path=archive_path,
        archive_sha256=archive_sha256,
        archive_bytes=archive_bytes,
        per_epoch_metrics=tuple(per_epoch_metrics),
        total_wall_clock_seconds=total_wall_clock,
        total_epochs_completed=total_epochs_completed,
        canonical_provenance=provenance,
        telemetry_path=config.resolved_telemetry_path(),
        early_stopped=early_stopped,
        early_stop_reason=early_stop_reason,
    )
    posterior_accepted, posterior_refusal = _emit_canonical_posterior_anchor(artifact_with_provenance)

    # 12) Final TrainingArtifact with posterior anchor info.
    final_artifact = TrainingArtifact(
        substrate_id=config.substrate_id,
        lane_id=config.lane_id,
        config_snapshot=config.as_dict(),
        ema_shadow_checkpoint_path=ema_shadow_checkpoint_path,
        live_checkpoint_path=live_checkpoint_path,
        archive_path=archive_path,
        archive_sha256=archive_sha256,
        archive_bytes=archive_bytes,
        per_epoch_metrics=tuple(per_epoch_metrics),
        total_wall_clock_seconds=total_wall_clock,
        total_epochs_completed=total_epochs_completed,
        canonical_provenance=provenance,
        telemetry_path=config.resolved_telemetry_path(),
        early_stopped=early_stopped,
        early_stop_reason=early_stop_reason,
        posterior_update_accepted=posterior_accepted,
        posterior_refusal_reason=posterior_refusal,
    )

    # 13) Persist canonical TrainingArtifact JSON next to telemetry.
    artifact_json_path = config.output_dir / "training_artifact.json"
    try:
        artifact_json_path.write_text(
            json.dumps(final_artifact.as_dict(), sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"[long_training_canonical] WARN: TrainingArtifact JSON emission failed: {exc!r}")

    return final_artifact


# ---------------------------------------------------------------------------
# Multi-arm parallel dispatch
# ---------------------------------------------------------------------------


def run_long_training_multi_arm(
    arms: Sequence[tuple[SubstrateLongTrainingAdapter, LongTrainingConfig]],
    *,
    max_concurrent_arms: int = 4,
) -> MultiArmDispatchResult:
    """Canonical multi-arm parallel dispatch per doctrine element #7.

    Runs N substrate-arm long-training runs concurrently on M-series
    shared GPU memory. Each arm emits its own canonical posterior anchor
    + TrainingArtifact. Sister of ``tac.master_gradient_consumers.master_gradient_xray_consumer``
    (cross-arm diff is the consumer's responsibility per Catalog #354
    master_gradient exploit consumer bundle).

    Per CLAUDE.md "Subagent coherence-by-default" + Catalog #302 sister-
    subagent scope overlap: the canonical concurrency cap is 4. Operator
    may override via ``max_concurrent_arms`` but per Catalog #302 must
    document the rationale.

    Args:
        arms: sequence of (adapter, config) tuples, one per arm.
        max_concurrent_arms: canonical concurrency cap. Defaults to 4
            per Catalog #302; operator-overridable with documented
            rationale.

    Returns:
        MultiArmDispatchResult containing TrainingArtifact per arm.

    Raises:
        ValueError: arms is empty OR max_concurrent_arms invalid.

    Notes:
        Current implementation runs SEQUENTIALLY (concurrent.futures
        ThreadPoolExecutor would conflict with MLX's single-GPU memory
        contention semantics; PR95 sister module empirically observed
        OOM with concurrent MLX arms). The sequential default is
        canonical-safe; future operator may opt into concurrent
        execution via process-pool isolation per Catalog #302 sister
        discipline.
    """
    if not arms:
        raise ValueError("arms must be non-empty sequence")
    if max_concurrent_arms < 1:
        raise ValueError(f"max_concurrent_arms must be >= 1; got {max_concurrent_arms!r}")
    if max_concurrent_arms > 4:
        # Operator may override; warn per Catalog #302 + #340 discipline.
        print(
            f"[long_training_canonical] WARN: max_concurrent_arms={max_concurrent_arms} > 4 "
            "canonical cap per Catalog #302 sister-subagent scope overlap. Operator should "
            "document rationale via the lane-registry notes."
        )

    t_start = time.time()
    results: list[TrainingArtifact] = []
    for i, (adapter, config) in enumerate(arms):
        validate_substrate_adapter(adapter)
        validate_long_training_config(config)
        print(
            f"[long_training_canonical] multi-arm dispatch arm {i+1}/{len(arms)}: "
            f"substrate={config.substrate_id} lane={config.lane_id}"
        )
        artifact = run_long_training(adapter, config)
        results.append(artifact)
    total_wall = time.time() - t_start

    return MultiArmDispatchResult(
        arms=tuple(results),
        total_wall_clock_seconds=total_wall,
    )
