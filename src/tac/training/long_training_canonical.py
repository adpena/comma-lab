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
import fcntl
import hashlib
import json
import math
import os
import shutil
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "CANONICAL_EMA_DECAY",
    "CANONICAL_NON_PROMOTABLE_MARKERS",
    "CANONICAL_SEGNET_ARGMAX_MIN_OCCUPIED_CLASS_FRACTION_FOR_FIT_GATE",
    "CANONICAL_SEGNET_TARGET_CLASS_COVERAGE_FRACTION_FOR_FIT_GATE",
    "CANONICAL_SEGNET_TARGET_CLASS_MIN_RATIO_FOR_FIT_GATE",
    "DEFAULT_CHECKPOINT_INTERVAL_EPOCHS",
    "DEFAULT_CHECKPOINT_RETENTION_KEEP_BEST_N",
    "DEFAULT_CHECKPOINT_RETENTION_KEEP_LAST_N",
    "DEFAULT_EARLY_STOPPING_PATIENCE",
    "DEFAULT_TELEMETRY_FLUSH_INTERVAL_EPOCHS",
    "EMA_ACCUMULATION_MODES",
    "PR95_8STAGE_CURRICULUM_DEFAULT",
    "TRAINING_ARTIFACT_SCHEMA_VERSION",
    "CheckpointWriter",
    "CurriculumStage",
    "KahanCompensatedPolyakEMAShadow",
    "LongTrainingConfig",
    "LongTrainingStopRequested",
    "MultiArmDispatchResult",
    "OOMSafeStepRunner",
    "PerEpochMetrics",
    "PolyakEMAShadow",
    "SubstrateLongTrainingAdapter",
    "TelemetrySink",
    "TrainingArtifact",
    "apply_checkpoint_retention",
    "run_long_training",
    "run_long_training_multi_arm",
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

_SUBSTRATE_ARTIFACT_METADATA_FORBIDDEN_AUTHORITY_KEYS: frozenset[str] = frozenset(
    {
        *CANONICAL_NON_PROMOTABLE_MARKERS,
        "score_claim_valid",
    }
)

# Canonical defaults (operator overridable via LongTrainingConfig).
DEFAULT_CHECKPOINT_INTERVAL_EPOCHS: int = 100
DEFAULT_CHECKPOINT_RETENTION_KEEP_LAST_N: int | None = None
DEFAULT_CHECKPOINT_RETENTION_KEEP_BEST_N: int = 1
DEFAULT_EARLY_STOPPING_PATIENCE: int = 200
DEFAULT_TELEMETRY_FLUSH_INTERVAL_EPOCHS: int = 10
EMA_ACCUMULATION_MODES: frozenset[str] = frozenset({"kahan", "naive"})
CONTEST_RATE_SCORE_PER_BYTE: float = 25.0 / 37_545_489.0
CANONICAL_SEGNET_ARGMAX_MIN_OCCUPIED_CLASS_FRACTION_FOR_FIT_GATE = 0.400001
CANONICAL_SEGNET_TARGET_CLASS_COVERAGE_FRACTION_FOR_FIT_GATE = 1.0
CANONICAL_SEGNET_TARGET_CLASS_MIN_RATIO_FOR_FIT_GATE = 0.2

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
PR95_8STAGE_CURRICULUM_DEFAULT: tuple[CurriculumStage, ...]  # forward-declared


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


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(v) for v in value]
    try:
        return value.item()
    except AttributeError:
        return str(value)


def _jsonable_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _jsonable(value) for key, value in mapping.items()}


def _sha256_text(payload: str) -> str:
    """Hex sha256 of a text payload."""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _curriculum_hash_for_stages(stages: Sequence[CurriculumStage]) -> str:
    payload = json.dumps(
        [s.as_dict() for s in sorted(stages, key=lambda s: s.start_epoch)],
        sort_keys=True,
    )
    return _sha256_text(payload)


def _prefix_curriculum_hash_at_epoch_boundary(
    stages: Sequence[CurriculumStage],
    *,
    boundary_epoch: int,
) -> str | None:
    """Hash ``stages`` truncated to the already-executed epoch boundary."""

    if boundary_epoch <= 0:
        return None
    prefix: list[CurriculumStage] = []
    for stage in sorted(stages, key=lambda s: s.start_epoch):
        if stage.start_epoch >= boundary_epoch:
            break
        end_epoch = min(stage.end_epoch, boundary_epoch)
        if end_epoch <= stage.start_epoch:
            continue
        prefix.append(replace(stage, end_epoch=end_epoch))
        if stage.end_epoch >= boundary_epoch:
            break
    if not prefix or prefix[0].start_epoch != 0 or prefix[-1].end_epoch != boundary_epoch:
        return None
    for prev, curr in pairwise(prefix):
        if prev.end_epoch != curr.start_epoch:
            return None
    return _curriculum_hash_for_stages(tuple(prefix))


def _matching_prefix_curriculum_boundary(
    stages: Sequence[CurriculumStage],
    *,
    min_boundary_epoch: int,
    expected_hash: str,
) -> int | None:
    """Return the earliest safe prefix boundary matching ``expected_hash``."""

    max_boundary_epoch = max((stage.end_epoch for stage in stages), default=0)
    for boundary_epoch in range(max(1, min_boundary_epoch), max_boundary_epoch + 1):
        if (
            _prefix_curriculum_hash_at_epoch_boundary(
                stages,
                boundary_epoch=boundary_epoch,
            )
            == expected_hash
        ):
            return int(boundary_epoch)
    return None


def _validate_substrate_artifact_metadata(
    metadata: Mapping[str, Any],
    field_name: str = "substrate_artifact_metadata",
) -> dict[str, Any]:
    """Validate optional adapter-supplied metadata for artifact emission.

    This metadata channel is deliberately non-authoritative. It can carry
    substrate-local facts (backend lineage, math-fidelity blockers, portable
    export notes), but cannot carry duplicate score/readiness keys. The
    canonical ``TrainingArtifact`` false-authority fields remain the only
    custody surface downstream consumers should read.
    """
    if not isinstance(metadata, Mapping):
        raise TypeError(
            f"{field_name} must be Mapping; got {type(metadata).__name__}"
        )
    def _reject_authority_keys(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if not isinstance(key, str) or not key:
                    raise ValueError(
                        f"{field_name} keys must be non-empty str; got "
                        f"{key!r} at {path}"
                    )
                child_path = f"{path}.{key}"
                if key in _SUBSTRATE_ARTIFACT_METADATA_FORBIDDEN_AUTHORITY_KEYS:
                    raise ValueError(
                        f"{field_name} cannot carry canonical "
                        f"authority/readiness key {key!r} at {child_path}; "
                        f"use TrainingArtifact.{key} as the single custody "
                        "surface."
                    )
                _reject_authority_keys(child, child_path)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                _reject_authority_keys(child, f"{path}[{index}]")

    _reject_authority_keys(metadata, field_name)
    normalized = dict(metadata)
    try:
        json.dumps(normalized, sort_keys=True)
    except TypeError as exc:
        raise TypeError(
            f"{field_name} must be JSON-serializable; {exc}"
        ) from exc
    return normalized


def _collect_adapter_artifact_metadata(
    adapter: Any,
) -> dict[str, Any]:
    """Collect optional adapter metadata, failing closed on malformed output."""
    metadata_fn = getattr(adapter, "artifact_metadata", None)
    if metadata_fn is None:
        return {}
    if not callable(metadata_fn):
        raise TypeError("adapter.artifact_metadata exists but is not callable")
    metadata = metadata_fn()
    if metadata is None:
        return {}
    return _validate_substrate_artifact_metadata(metadata, "adapter.artifact_metadata")


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
        ema_accumulation: floating-point accumulation mode for the EMA shadow.
            ``"naive"`` preserves historical Polyak updates; ``"kahan"``
            enables compensated accumulation and strict fallback refusal.
        checkpoint_interval_epochs: emit canonical checkpoint every N epochs.
        checkpoint_retention_keep_last_n: optional hot-checkpoint retention
            guard. ``None`` disables pruning and preserves legacy behavior.
            When set, periodic checkpoints outside the last-N / best-N /
            milestone keep set are moved to cold store with a manifest.
        checkpoint_retention_keep_best_n: keep the N lowest-loss periodic
            checkpoints in the hot checkpoint directory when retention is
            enabled. Default 1 preserves non-monotone training signal.
        checkpoint_retention_keep_every_n_epochs: optional milestone spacing
            using the same ``epoch + 1`` convention as checkpoint cadence.
        checkpoint_retention_action: ``"move"`` (lossless hot-retention) or
            explicit ``"delete"``. Long carrier campaigns should use move.
        checkpoint_retention_cold_store_roots: optional cold-store roots for
            moved checkpoints. Empty means use the operator storage waterfall.
        checkpoint_retention_manifest_path: optional JSONL manifest path; by
            default lives next to the checkpoint directory.
        best_checkpoint_for_archive_export: when True, the trainer keeps the
            best live/EMA state in memory, emits one durable best checkpoint at
            the end, and uses that checkpoint for archive export. By default
            "best" means lowest total loss. Score-aware carriers may set
            checkpoint_selection_metric_key to a concrete loss component so
            archive export follows the scorer-facing objective rather than a
            guard/coder aggregate.
        checkpoint_selection_metric_key: loss_dict key used for best-checkpoint
            selection and early-stopping improvement tests. ``"total"`` selects
            the scalar total loss. Missing or non-finite component values record
            a blocker and fall back to total loss for that epoch.
        checkpoint_selection_metric_mode: ``"min"`` for losses/errors,
            ``"max"`` for rewards/occupancy metrics.
        checkpoint_selection_metric_required: when True, a missing, nonnumeric,
            or nonfinite named checkpoint metric is a hard training failure
            instead of a fallback to total loss. Score-aware compact renderers
            use this for direct scorer metrics so a collapsed archive cannot be
            exported just because guard/coder total loss decreased.
        checkpoint_selection_tie_break_metric_key: optional secondary loss_dict
            key used only when the primary checkpoint metric is exactly tied
            within the canonical tolerance. This is for discontinuous scorer
            proxies such as argmax disagreement: when the discrete metric is
            flat, archive export should prefer the checkpoint with the best
            continuous distance-to-boundary surrogate.
        checkpoint_selection_tie_break_metric_mode: ``"min"`` or ``"max"`` for
            the secondary metric.
        checkpoint_selection_tie_break_metric_required: hard-fail if the
            configured secondary metric is missing/non-finite.
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
        telemetry_flush_interval_epochs: flush JSONL rows every N epochs.
            Default preserves the canonical historical cadence; long expensive
            carrier rows should set this to 1 so progress is queryable while
            the run is alive.
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
    ema_accumulation: str = "naive"
    checkpoint_interval_epochs: int = DEFAULT_CHECKPOINT_INTERVAL_EPOCHS
    checkpoint_retention_keep_last_n: int | None = DEFAULT_CHECKPOINT_RETENTION_KEEP_LAST_N
    checkpoint_retention_keep_best_n: int = DEFAULT_CHECKPOINT_RETENTION_KEEP_BEST_N
    checkpoint_retention_keep_every_n_epochs: int | None = None
    checkpoint_retention_action: str = "move"
    checkpoint_retention_cold_store_roots: tuple[Path, ...] = field(default_factory=tuple)
    checkpoint_retention_manifest_path: Path | None = None
    best_checkpoint_for_archive_export: bool = True
    checkpoint_selection_metric_key: str = "total"
    checkpoint_selection_metric_mode: str = "min"
    checkpoint_selection_metric_required: bool = False
    checkpoint_selection_tie_break_metric_key: str = ""
    checkpoint_selection_tie_break_metric_mode: str = "min"
    checkpoint_selection_tie_break_metric_required: bool = False
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE
    score_aware_loss_kwargs: Mapping[str, Any] = field(default_factory=dict)
    optimizer_class: str = "adamw"
    learning_rate: float = 1e-3
    seed: int = 0
    output_dir: Path = field(default_factory=lambda: Path("experiments/results/long_training_canonical_default"))
    telemetry_path: Path | None = None
    telemetry_flush_interval_epochs: int = DEFAULT_TELEMETRY_FLUSH_INTERVAL_EPOCHS
    checkpoint_dir: Path | None = None
    device: str = "mlx"
    resume_from_checkpoint: Path | None = None
    evidence_grade: str = "[macOS-MLX research-signal]"
    ema_archive_selection_enabled: bool = False
    archive_selection_replay_required: bool = False
    archive_selection_replay_batch_size: int | None = None
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
        for prev, curr in pairwise(sorted_stages):
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
        if self.ema_accumulation not in EMA_ACCUMULATION_MODES:
            raise ValueError(
                "ema_accumulation must be one of "
                f"{sorted(EMA_ACCUMULATION_MODES)}; got {self.ema_accumulation!r}"
            )
        if not isinstance(self.checkpoint_interval_epochs, int) or self.checkpoint_interval_epochs <= 0:
            raise ValueError(
                f"checkpoint_interval_epochs must be positive int; "
                f"got {self.checkpoint_interval_epochs!r}"
            )
        if (
            self.checkpoint_retention_keep_last_n is not None
            and (
                isinstance(self.checkpoint_retention_keep_last_n, bool)
                or not isinstance(self.checkpoint_retention_keep_last_n, int)
                or self.checkpoint_retention_keep_last_n < 0
            )
        ):
            raise ValueError(
                "checkpoint_retention_keep_last_n must be non-negative int "
                f"or None; got {self.checkpoint_retention_keep_last_n!r}"
            )
        if (
            isinstance(self.checkpoint_retention_keep_best_n, bool)
            or not isinstance(self.checkpoint_retention_keep_best_n, int)
            or self.checkpoint_retention_keep_best_n < 0
        ):
            raise ValueError(
                "checkpoint_retention_keep_best_n must be non-negative int; "
                f"got {self.checkpoint_retention_keep_best_n!r}"
            )
        if (
            self.checkpoint_retention_keep_every_n_epochs is not None
            and (
                isinstance(self.checkpoint_retention_keep_every_n_epochs, bool)
                or not isinstance(self.checkpoint_retention_keep_every_n_epochs, int)
                or self.checkpoint_retention_keep_every_n_epochs <= 0
            )
        ):
            raise ValueError(
                "checkpoint_retention_keep_every_n_epochs must be positive int "
                "or None; got "
                f"{self.checkpoint_retention_keep_every_n_epochs!r}"
            )
        if self.checkpoint_retention_action not in {"move", "delete"}:
            raise ValueError(
                "checkpoint_retention_action must be 'move' or 'delete'; got "
                f"{self.checkpoint_retention_action!r}"
            )
        if not isinstance(self.checkpoint_retention_cold_store_roots, tuple):
            raise TypeError(
                "checkpoint_retention_cold_store_roots must be tuple[Path, ...]; "
                f"got {type(self.checkpoint_retention_cold_store_roots).__name__}"
            )
        for index, root in enumerate(self.checkpoint_retention_cold_store_roots):
            if not isinstance(root, Path):
                raise TypeError(
                    "checkpoint_retention_cold_store_roots"
                    f"[{index}] must be Path; got {type(root).__name__}"
                )
            _refuse_tmp_path(root, f"checkpoint_retention_cold_store_roots[{index}]")
        if self.checkpoint_retention_manifest_path is not None:
            if not isinstance(self.checkpoint_retention_manifest_path, Path):
                raise TypeError(
                    "checkpoint_retention_manifest_path must be Path; got "
                    f"{type(self.checkpoint_retention_manifest_path).__name__}"
                )
            _refuse_tmp_path(
                self.checkpoint_retention_manifest_path,
                "checkpoint_retention_manifest_path",
            )
        if self.checkpoint_retention_enabled and (
            (self.checkpoint_retention_keep_last_n or 0)
            + self.checkpoint_retention_keep_best_n
            <= 0
            and self.checkpoint_retention_keep_every_n_epochs is None
        ):
            raise ValueError(
                "checkpoint retention would prune every periodic checkpoint; "
                "keep at least one last/best checkpoint or configure milestone spacing"
            )
        if not isinstance(self.best_checkpoint_for_archive_export, bool):
            raise TypeError(
                "best_checkpoint_for_archive_export must be bool; got "
                f"{type(self.best_checkpoint_for_archive_export).__name__}"
            )
        if (
            not isinstance(self.checkpoint_selection_metric_key, str)
            or not self.checkpoint_selection_metric_key.strip()
        ):
            raise ValueError(
                "checkpoint_selection_metric_key must be non-empty str; got "
                f"{self.checkpoint_selection_metric_key!r}"
            )
        if self.checkpoint_selection_metric_mode not in {"min", "max"}:
            raise ValueError(
                "checkpoint_selection_metric_mode must be one of {'min', 'max'}; "
                f"got {self.checkpoint_selection_metric_mode!r}"
            )
        if not isinstance(self.checkpoint_selection_metric_required, bool):
            raise TypeError(
                "checkpoint_selection_metric_required must be bool; got "
                f"{type(self.checkpoint_selection_metric_required).__name__}"
            )
        if not isinstance(self.checkpoint_selection_tie_break_metric_key, str):
            raise TypeError(
                "checkpoint_selection_tie_break_metric_key must be str; got "
                f"{type(self.checkpoint_selection_tie_break_metric_key).__name__}"
            )
        if self.checkpoint_selection_tie_break_metric_mode not in {"min", "max"}:
            raise ValueError(
                "checkpoint_selection_tie_break_metric_mode must be one of "
                "{'min', 'max'}; got "
                f"{self.checkpoint_selection_tie_break_metric_mode!r}"
            )
        if not isinstance(self.checkpoint_selection_tie_break_metric_required, bool):
            raise TypeError(
                "checkpoint_selection_tie_break_metric_required must be bool; got "
                f"{type(self.checkpoint_selection_tie_break_metric_required).__name__}"
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
        if (
            not isinstance(self.telemetry_flush_interval_epochs, int)
            or self.telemetry_flush_interval_epochs <= 0
        ):
            raise ValueError(
                "telemetry_flush_interval_epochs must be positive int; got "
                f"{self.telemetry_flush_interval_epochs!r}"
            )
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
        if self.resume_from_checkpoint is not None and not isinstance(
            self.resume_from_checkpoint, Path
        ):
            raise TypeError(
                f"resume_from_checkpoint must be Path; "
                f"got {type(self.resume_from_checkpoint).__name__}"
            )
        if not isinstance(self.ema_archive_selection_enabled, bool):
            raise TypeError(
                "ema_archive_selection_enabled must be bool; got "
                f"{type(self.ema_archive_selection_enabled).__name__}"
            )
        if not isinstance(self.archive_selection_replay_required, bool):
            raise TypeError(
                "archive_selection_replay_required must be bool; got "
                f"{type(self.archive_selection_replay_required).__name__}"
            )
        if self.archive_selection_replay_batch_size is not None and (
            isinstance(self.archive_selection_replay_batch_size, bool)
            or not isinstance(self.archive_selection_replay_batch_size, int)
            or self.archive_selection_replay_batch_size <= 0
        ):
            raise ValueError(
                "archive_selection_replay_batch_size must be positive int "
                f"or None; got {self.archive_selection_replay_batch_size!r}"
            )
        if self.notes:
            _validate_rationale_not_placeholder(self.notes, "LongTrainingConfig.notes")

    def resolved_telemetry_path(self) -> Path:
        """Canonical telemetry path (default = output_dir/telemetry.jsonl)."""
        return self.telemetry_path or (self.output_dir / "telemetry.jsonl")

    def resolved_checkpoint_dir(self) -> Path:
        """Canonical checkpoint dir (default = output_dir/checkpoints/)."""
        return self.checkpoint_dir or (self.output_dir / "checkpoints")

    @property
    def checkpoint_retention_enabled(self) -> bool:
        """Whether periodic checkpoint hot-retention is active."""

        return (
            self.checkpoint_retention_keep_last_n is not None
            or self.checkpoint_retention_keep_every_n_epochs is not None
        )

    def resolved_checkpoint_retention_manifest_path(self) -> Path:
        """Canonical retention manifest path."""

        return self.checkpoint_retention_manifest_path or (
            self.resolved_checkpoint_dir() / "checkpoint_retention_manifest.jsonl"
        )

    def curriculum_hash(self) -> str:
        """Canonical hash over curriculum_stages for resume validation."""
        return _curriculum_hash_for_stages(self.curriculum_stages)

    def as_dict(self) -> dict[str, Any]:
        return {
            "substrate_id": self.substrate_id,
            "lane_id": self.lane_id,
            "epochs": int(self.epochs),
            "batch_pair_indices_per_step": int(self.batch_pair_indices_per_step),
            "curriculum_stages": [s.as_dict() for s in self.curriculum_stages],
            "curriculum_hash": self.curriculum_hash(),
            "ema_decay": float(self.ema_decay),
            "ema_accumulation": self.ema_accumulation,
            "checkpoint_interval_epochs": int(self.checkpoint_interval_epochs),
            "checkpoint_retention": {
                "schema": "long_training_checkpoint_retention_config.v1",
                "enabled": bool(self.checkpoint_retention_enabled),
                "keep_last_n": self.checkpoint_retention_keep_last_n,
                "keep_best_n": int(self.checkpoint_retention_keep_best_n),
                "keep_every_n_epochs": self.checkpoint_retention_keep_every_n_epochs,
                "action": self.checkpoint_retention_action,
                "cold_store_roots": [
                    str(path) for path in self.checkpoint_retention_cold_store_roots
                ],
                "manifest_path": str(
                    self.resolved_checkpoint_retention_manifest_path()
                ),
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            "best_checkpoint_for_archive_export": bool(
                self.best_checkpoint_for_archive_export
            ),
            "checkpoint_selection_metric_key": str(
                self.checkpoint_selection_metric_key
            ),
            "checkpoint_selection_metric_mode": str(
                self.checkpoint_selection_metric_mode
            ),
            "checkpoint_selection_metric_required": bool(
                self.checkpoint_selection_metric_required
            ),
            "checkpoint_selection_tie_break_metric_key": str(
                self.checkpoint_selection_tie_break_metric_key
            ),
            "checkpoint_selection_tie_break_metric_mode": str(
                self.checkpoint_selection_tie_break_metric_mode
            ),
            "checkpoint_selection_tie_break_metric_required": bool(
                self.checkpoint_selection_tie_break_metric_required
            ),
            "early_stopping_patience": int(self.early_stopping_patience),
            "score_aware_loss_kwargs": dict(self.score_aware_loss_kwargs),
            "optimizer_class": self.optimizer_class,
            "learning_rate": float(self.learning_rate),
            "seed": int(self.seed),
            "output_dir": str(self.output_dir),
            "telemetry_path": str(self.resolved_telemetry_path()),
            "telemetry_flush_interval_epochs": int(
                self.telemetry_flush_interval_epochs
            ),
            "checkpoint_dir": str(self.resolved_checkpoint_dir()),
            "device": self.device,
            "resume_from_checkpoint": (
                str(self.resume_from_checkpoint) if self.resume_from_checkpoint else None
            ),
            "evidence_grade": self.evidence_grade,
            "ema_archive_selection_enabled": bool(
                self.ema_archive_selection_enabled
            ),
            "archive_selection_replay_required": bool(
                self.archive_selection_replay_required
            ),
            "archive_selection_replay_batch_size": (
                None
                if self.archive_selection_replay_batch_size is None
                else int(self.archive_selection_replay_batch_size)
            ),
            "notes": self.notes,
        }

    def stage_at_epoch(self, epoch: int) -> CurriculumStage:
        """Return the CurriculumStage covering ``epoch`` (clamped to last stage)."""
        for stage in self.curriculum_stages:
            if stage.start_epoch <= epoch < stage.end_epoch:
                return stage
        # Past final stage end: clamp to last stage (e.g. early-stop pad).
        return self.curriculum_stages[-1]


def _contest_score_from_axes(
    *, seg: float, pose: float, archive_bytes: float
) -> float:
    """The canonical contest score for the per-epoch proxy decomposition.

    ``100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37_545_489``. Inlined here
    (rather than imported from ``tac.substrates.hi_nerv.launch_manifest``) to
    keep ``long_training_canonical`` substrate-agnostic with no import cycle.
    Proxy / [macOS-MLX research-signal] only — exact CPU/CUDA eval is authority.
    """
    pose_term = math.sqrt(10.0 * max(0.0, float(pose)))
    return (
        100.0 * float(seg)
        + pose_term
        + 25.0 * float(archive_bytes) / 37_545_489.0
    )


def _assemble_gating_row(
    *,
    adapter: Any,
    epoch: int,
    total_loss: float,
    per_axis: Mapping[str, float] | None,
    last_checkpoint_path: Path | None = None,
    checkpoint_save_scheduled: bool = False,
) -> dict[str, Any] | None:
    """Assemble the canonical RICH GATING telemetry row for one epoch.

    Observability-ONLY per CLAUDE.md "Max observability" + the B1 clean-relaunch
    BLOCKER 1 directive. The adapter's ``rich_gating_telemetry(epoch)`` supplies
    the stage geometry + optimizer-partition kind + per-step grad-norm / clip /
    nan-inf; this helper augments it with the per-axis proxy loss decomposition
    (seg / pose / rate / recon) + the canonical proxy contest score so a
    diverging run is diagnosable from a single telemetry tail.

    Returns ``None`` if the adapter does not expose ``rich_gating_telemetry``
    (sister substrates stay byte-identical: their rows omit the gating fields).

    The per-axis values are the UNWEIGHTED proxy axis losses. ``loss_total`` is
    the WEIGHTED training total. They are NOT expected to be equal; the row
    reports both plus an arithmetically-consistent ``per_axis_sum`` (the literal
    sum of the per-axis components emitted) so a NO-FAKE sum-check verifies the
    arithmetic without forcing a fabricated equality.
    """
    rich_fn = getattr(adapter, "rich_gating_telemetry", None)
    if not callable(rich_fn):
        return None
    try:
        gating: dict[str, Any] = dict(rich_fn(epoch))
    except Exception as exc:  # observability-only; never fail the run.
        print(
            f"[long_training_canonical] WARN: rich_gating_telemetry failed "
            f"at epoch {epoch}: {exc!r}"
        )
        return None

    gating["loss_total"] = float(total_loss)

    axes = dict(per_axis or {})
    seg = float(axes.get("seg", 0.0))
    pose = float(axes.get("pose", 0.0))
    rate = float(axes.get("archive_bytes", axes.get("rate", 0.0)))
    recon_aux = float(axes.get("recon_aux", 0.0))
    margin = float(axes.get("margin", axes.get("seg_margin", 0.0)))
    c1a = float(axes.get("c1a", axes.get("c1a_entropy", 0.0)))

    gating["loss_seg"] = seg
    gating["loss_pose"] = pose
    gating["loss_rate"] = rate
    gating["loss_recon_aux"] = recon_aux
    gating["loss_margin"] = margin
    gating["loss_c1a"] = c1a
    # Literal arithmetic sum of the per-axis components emitted above (the
    # NO-FAKE sum-check target — verifies the emitted numbers add up, NOT a
    # forced equality to the weighted total).
    gating["per_axis_sum"] = seg + pose + rate + recon_aux + margin + c1a

    # Proxy axis values + proxy contest score (research-signal only).
    gating["proxy_d_seg"] = seg
    gating["proxy_d_pose"] = pose
    gating["proxy_rate"] = rate
    gating["proxy_score"] = _contest_score_from_axes(
        seg=seg, pose=pose, archive_bytes=rate
    )
    # ``checkpoint_path`` records the LAST durably-written checkpoint meta path
    # (the periodic write happens after this row is recorded, so this is the
    # most-recent saved checkpoint — an honest record, never a predicted path).
    # ``checkpoint_save_scheduled`` is the honest pre-write fact that THIS epoch
    # is a checkpoint-cadence boundary.
    gating["checkpoint_path"] = (
        str(last_checkpoint_path) if last_checkpoint_path is not None else None
    )
    gating["checkpoint_save_scheduled"] = bool(checkpoint_save_scheduled)
    return gating


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
        batch_observability: optional non-authority sampling telemetry from
            the adapter, such as pair indices touched by the train/eval
            batches. This exists to prevent small-subset distortion wins from
            being mistaken for full-video coverage.
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
    batch_observability: Mapping[str, Any] | None = None
    wall_clock_seconds: float = 0.0
    ema_drift_l2: float = 0.0
    learning_rate: float = 0.0
    captured_at_utc: str = ""
    # Optional RICH GATING telemetry per CLAUDE.md "Max observability" +
    # the B1 clean-relaunch directive: the adapter may expose
    # ``rich_gating_telemetry(epoch)`` returning stage geometry +
    # optimizer-partition kind + per-step grad-norm/clip + nan/inf so a
    # diverging run is diagnosable from a single telemetry tail. ``None``
    # for adapters that do NOT expose the accessor (byte-stable — these
    # fields are absent from such adapters' rows by default).
    gating: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.epoch, int) or self.epoch < 0:
            raise ValueError(f"epoch must be int >= 0; got {self.epoch!r}")
        if not isinstance(self.loss, (int, float)):
            raise TypeError(f"loss must be numeric; got {type(self.loss).__name__}")
        if self.loss != self.loss:  # NaN check
            raise ValueError(f"loss is NaN at epoch {self.epoch}; OOM-safe runner should detect")

    def as_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "epoch": int(self.epoch),
            "stage_name": self.stage_name,
            "loss": float(self.loss),
            "loss_components": {k: float(v) for k, v in self.loss_components.items()},
            "per_axis_decomposition": (
                {k: float(v) for k, v in self.per_axis_decomposition.items()}
                if self.per_axis_decomposition is not None
                else None
            ),
            "batch_observability": (
                _jsonable_mapping(self.batch_observability)
                if self.batch_observability is not None
                else None
            ),
            "wall_clock_seconds": float(self.wall_clock_seconds),
            "ema_drift_l2": float(self.ema_drift_l2),
            "learning_rate": float(self.learning_rate),
            "captured_at_utc": self.captured_at_utc,
        }
        if self.gating is not None:
            gating = _jsonable_mapping(self.gating)
            # Preserve the full nested gating block AND surface the canonical
            # gating fields at the row top-level so operator diagnosis can tail
            # the JSONL and read stage_index / loss_form / muon_active /
            # grad_norm / grad_clip_applied / nan_inf_count / sidecar_exported
            # directly (the B1 clean-relaunch BLOCKER 1 contract).
            row["gating"] = gating
            for _top_key in (
                "stage_index",
                "loss_form",
                "muon_active",
                "optimizer_kind_by_group",
                "grad_norm",
                "grad_clip_applied",
                "nan_inf_count",
                "sidecar_exported",
                "pay_rent_gate_active",
                "loss_total",
                "loss_seg",
                "loss_pose",
                "loss_rate",
                "loss_margin",
                "loss_c1a",
                "proxy_d_seg",
                "proxy_d_pose",
                "proxy_rate",
                "proxy_score",
                "checkpoint_path",
            ):
                if _top_key in gating:
                    row[_top_key] = gating[_top_key]
        return row


class LongTrainingStopRequested(RuntimeError):
    """Typed callback request to stop canonical long training.

    Ordinary ``on_epoch_end`` callback errors remain observability warnings.
    This exception is reserved for intentional fail-closed monitors such as
    scorer-axis instability guards that should stop a doomed expensive run while
    preserving canonical telemetry, checkpoints, and artifact provenance.
    """

    def __init__(self, reason: str) -> None:
        reason_text = str(reason).strip()
        if not reason_text:
            reason_text = "long_training_stop_requested"
        self.reason = reason_text
        super().__init__(reason_text)


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
        checkpoint_selection: machine-readable selection contract for final
            versus best-loss checkpoint. The top-level live/EMA checkpoint
            paths refer to the selected checkpoint.
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
        substrate_artifact_metadata: optional non-authority adapter metadata
            such as backend lineage or substrate-local blockers. Canonical
            score/readiness keys are rejected here to prevent stale duplicate
            custody readers.
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
    substrate_artifact_metadata: Mapping[str, Any] = field(default_factory=dict)
    live_checkpoint_path: Path | None = None
    archive_path: Path | None = None
    archive_sha256: str | None = None
    archive_bytes: int | None = None
    archive_selection_manifest_path: Path | None = None
    checkpoint_selection: Mapping[str, Any] = field(default_factory=dict)
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
        _validate_substrate_artifact_metadata(self.substrate_artifact_metadata)
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
        if not isinstance(self.checkpoint_selection, Mapping):
            raise TypeError(
                "checkpoint_selection must be Mapping; got "
                f"{type(self.checkpoint_selection).__name__}"
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
            "archive_selection_manifest_path": (
                str(self.archive_selection_manifest_path)
                if self.archive_selection_manifest_path
                else None
            ),
            "checkpoint_selection": _jsonable_mapping(self.checkpoint_selection),
            "per_epoch_metrics_count": len(self.per_epoch_metrics),
            "per_epoch_metrics": [m.as_dict() for m in self.per_epoch_metrics],
            "total_wall_clock_seconds": float(self.total_wall_clock_seconds),
            "total_epochs_completed": int(self.total_epochs_completed),
            "early_stopped": bool(self.early_stopped),
            "early_stop_reason": self.early_stop_reason,
            "substrate_artifact_metadata": dict(self.substrate_artifact_metadata),
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
        archive_replay_components(archive_path, batch, candidate_kind) -> dict | None:
            optional source-bound archive parse-back hook for live-vs-EMA archive
            selection. When ``LongTrainingConfig.archive_selection_replay_required``
            is True, the selector fails closed unless this hook returns finite
            scorer-axis components from the exported archive bytes.

    Optional method (Style B):
        train_step(batch, learning_rate, loss_weights) -> dict:
            combined value+grad+optimizer.update in one call. Returns
            same dict shape as loss_fn (REQUIRED ``"total"`` key).
            Used by MLX adapters where ``mlx.nn.value_and_grad`` requires
            closure over both forward + backward.
        post_optimizer_projection(epoch) -> Mapping | None:
            optional proximal projection hook called immediately after the
            optimizer step and before EMA, scorer-component probes, metrics,
            checkpoints, and archive export. Use this for real weight-space
            projections such as train-time quantization/pruning controls; do
            not put parameter mutation in on_epoch_end observability callbacks.
        notify_curriculum_stage(epoch, stage) -> None:
            optional stage-control hook called once per epoch after
            ``notify_global_epoch`` and before ``train_step`` / ``loss_fn``.
            This is the canonical surface for non-loss stage controls such as
            ``CurriculumStage.enable_qat``; if an adapter implements it and it
            raises, the training run fails closed instead of silently ignoring
            a requested train-time control.
        artifact_metadata() -> Mapping[str, Any] | None:
            optional non-authority metadata threaded into TrainingArtifact
            JSON and MLX posterior rows. This may carry substrate lineage or
            blocker facts, but canonical score/readiness keys are rejected.
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


def _resolve_checkpoint_selection_metric(
    *,
    loss_dict: Mapping[str, Any],
    total_loss: float,
    metric_key: str,
    strict: bool = False,
) -> tuple[float, str | None]:
    """Return the metric that controls best-checkpoint archive export."""

    key = str(metric_key).strip()
    if not key or key == "total":
        return float(total_loss), None
    if key not in loss_dict:
        if strict:
            raise RuntimeError(f"checkpoint_selection_metric_missing:{key}")
        return float(total_loss), f"checkpoint_selection_metric_missing:{key}"
    try:
        value = float(loss_dict[key])
    except (TypeError, ValueError) as err:
        if strict:
            raise RuntimeError(
                f"checkpoint_selection_metric_non_numeric:{key}"
            ) from err
        return float(total_loss), f"checkpoint_selection_metric_non_numeric:{key}"
    if not math.isfinite(value):
        if strict:
            raise RuntimeError(f"checkpoint_selection_metric_nonfinite:{key}")
        return float(total_loss), f"checkpoint_selection_metric_nonfinite:{key}"
    return value, None


def _checkpoint_metric_improved(
    *,
    current: float,
    best: float,
    mode: str,
) -> bool:
    if mode == "max":
        return current > best + 1e-9
    return current < best - 1e-9


def _checkpoint_selection_improved(
    *,
    current: float,
    best: float,
    mode: str,
    current_tie_break: float | None = None,
    best_tie_break: float | None = None,
    tie_break_mode: str = "min",
) -> bool:
    if _checkpoint_metric_improved(current=current, best=best, mode=mode):
        return True
    if abs(float(current) - float(best)) > 1e-9:
        return False
    if current_tie_break is None:
        return False
    if best_tie_break is None:
        return True
    return _checkpoint_metric_improved(
        current=float(current_tie_break),
        best=float(best_tie_break),
        mode=tie_break_mode,
    )


def _ordered_unique(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


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

    def __init__(
        self,
        model: Any,
        decay: float = CANONICAL_EMA_DECAY,
        *,
        enable_kahan: bool = False,
        strict_kahan: bool = False,
    ):
        """Construct canonical Polyak EMA shadow primitive.

        Args:
            model: substrate model (torch / MLX / duck-typed state_dict).
            decay: Polyak EMA decay coefficient in (0, 1); per Catalog #2
                NON-NEGOTIABLE canonical default = 0.997.
            enable_kahan: when True, the per-tensor EMA shadow update uses
                Kahan (1965) compensated summation to bound floating-point
                truncation error at ULP-level instead of accumulating
                O(N * ULP) over N updates. Per T3 grand council OP #2
                (commit ``7d04474cb``) Class 1-SCOPED Kahan-EMA surgical
                mitigation of M2 (EMA shadow drift accumulation through
                Polyak 0.997 decay's ~333-step exponential moving average).
                Defaults to False to preserve canonical backward
                compatibility per Catalog #110/#113 APPEND-ONLY discipline;
                the canonical wrapper class ``KahanCompensatedPolyakEMAShadow``
                sets this flag True by default for callers who want the
                hardened semantics by construction.
            strict_kahan: when True, any Kahan branch fallback raises instead
                of silently degrading to naive Polyak for that tensor key.
        """
        if not (0.0 < decay < 1.0):
            raise ValueError(
                f"decay must be in (0, 1); got {decay!r}. "
                "Per Catalog #2 NON-NEGOTIABLE canonical default = 0.997."
            )
        # Duck-type detection: torch uses state_dict(); MLX uses parameters()
        # + tree_flatten/tree_unflatten. We support BOTH canonical patterns.
        self._mlx_mode = self._detect_mlx_mode(model)
        if not self._mlx_mode and (
            not hasattr(model, "state_dict") or not callable(model.state_dict)
        ):
            raise TypeError(
                f"model {type(model).__name__} must expose .state_dict() "
                "method (torch) OR .parameters() (MLX module); see "
                "PolyakEMAShadow duck-typing contract."
            )
        self.decay = decay
        self.enable_kahan = bool(enable_kahan)
        self.strict_kahan = bool(strict_kahan)
        if self.strict_kahan and not self.enable_kahan:
            raise ValueError("strict_kahan=True requires enable_kahan=True")
        self._shadow: dict[str, Any] = self._clone_state_dict(
            self._get_flat_state(model)
        )
        # Per Kahan 1965 compensated summation: a per-tensor compensation
        # buffer captures the low-order bits truncated during each
        # `shadow = decay*shadow + (1-decay)*live` update so the next update
        # restores them before the next truncation. Initialized to zeros and
        # only populated when ``enable_kahan=True``. The per-key zero-init
        # is deferred to first update() so we can match each shadow tensor's
        # type / shape via duck-typed subtraction `shadow - shadow`.
        self._kahan_compensation: dict[str, Any] = {}
        self._kahan_fallback_keys: set[str] = set()

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
        """Update shadow via canonical Polyak averaging.

        When ``enable_kahan=True``, the per-tensor update uses Kahan (1965)
        compensated summation to bound truncation error at ULP-level
        regardless of N updates. The canonical Kahan recurrence on
        ``S_new = decay*S + (1-decay)*L`` (treating the update as a sum of
        two terms with the compensation tracking the lost low-order bits):

            y = (1-decay)*L - c_prev
            t = decay*S + y
            c_new = (t - decay*S) - y
            S_new = t

        Per T3 grand council OP #2 (commit ``7d04474cb``) the M2 mechanism
        (EMA shadow drift accumulation through Polyak 0.997 decay's
        ~333-step exponential moving average) is bounded by Kahan;
        per Carmack 30-min smoke (OP #3) the empirical reduction ratio is
        verifiable at $0 MLX-local on the Z6 L2 substrate.
        """
        live_state = self._get_flat_state(model)
        for k, v in live_state.items():
            if k not in self._shadow:
                # Late-bound module: seed from live per Codex finding 2 sister
                # discipline in the canonical EMA at tac.training.EMA.update.
                self._shadow[k] = self._clone_state_dict({k: v})[k]
                continue
            shadow_v = self._shadow[k]
            if self.enable_kahan:
                # Kahan-compensated path: route the SAME duck-typed dispatch
                # tree as the naive path, but each branch threads c_prev /
                # c_new compensation via element-wise arithmetic. Non-float
                # buffers cannot accumulate truncation error meaningfully so
                # we bypass Kahan for them (copy live directly).
                try:
                    if hasattr(shadow_v, "mul_") and hasattr(shadow_v, "add_"):
                        # Torch path: explicit functional Kahan to retain
                        # the compensation buffer; we sacrifice in-place
                        # efficiency for ULP-bounded accumulation.
                        if hasattr(v, "is_floating_point") and not v.is_floating_point():
                            shadow_v.copy_(v)
                            self._kahan_compensation.pop(k, None)
                        else:
                            c_prev = self._kahan_compensation.get(k)
                            if c_prev is None:
                                c_prev = shadow_v - shadow_v  # type-matched zero tensor
                            y = (1.0 - self.decay) * v - c_prev
                            decay_shadow = self.decay * shadow_v
                            t = decay_shadow + y
                            c_new = (t - decay_shadow) - y
                            # In-place assign so downstream readers (drift_l2)
                            # see the updated shadow without re-binding.
                            shadow_v.copy_(t) if hasattr(shadow_v, "copy_") else None
                            self._shadow[k] = t if not hasattr(shadow_v, "copy_") else shadow_v
                            self._kahan_compensation[k] = c_new
                    elif isinstance(shadow_v, list):
                        # Plain Python list: element-wise Kahan.
                        c_prev_list = self._kahan_compensation.get(k)
                        if c_prev_list is None:
                            c_prev_list = [0.0] * len(shadow_v)
                        new_shadow: list[float] = []
                        new_c: list[float] = []
                        for sv, lv, cp in zip(shadow_v, v, c_prev_list, strict=False):
                            y = (1.0 - self.decay) * float(lv) - cp
                            decay_sv = self.decay * float(sv)
                            t = decay_sv + y
                            c_new_i = (t - decay_sv) - y
                            new_shadow.append(t)
                            new_c.append(c_new_i)
                        self._shadow[k] = new_shadow
                        self._kahan_compensation[k] = new_c
                    else:
                        # Functional path (MLX / numpy): construct new array.
                        c_prev = self._kahan_compensation.get(k)
                        if c_prev is None:
                            c_prev = shadow_v - shadow_v
                        y = (1.0 - self.decay) * v - c_prev
                        decay_shadow = self.decay * shadow_v
                        t = decay_shadow + y
                        c_new = (t - decay_shadow) - y
                        self._shadow[k] = t
                        self._kahan_compensation[k] = c_new
                except (AttributeError, TypeError, RuntimeError):
                    # Duck-type failure: degrade to naive Polyak per the
                    # canonical contract; discard this key's compensation so
                    # a later successful Kahan step restarts from a typed zero
                    # buffer instead of mixing stale arithmetic state.
                    self._kahan_compensation.pop(k, None)
                    self._kahan_fallback_keys.add(k)
                    if self.strict_kahan:
                        raise RuntimeError(
                            "Kahan EMA update failed in strict mode for "
                            f"state key {k!r}; refusing silent naive fallback"
                        ) from None
                    if isinstance(shadow_v, (list, tuple)) and isinstance(v, (list, tuple)):
                        self._shadow[k] = [
                            self.decay * float(sv) + (1.0 - self.decay) * float(lv)
                            for sv, lv in zip(shadow_v, v, strict=False)
                        ]
                    else:
                        self._shadow[k] = self.decay * shadow_v + (1.0 - self.decay) * v
                continue
            # Naive (canonical backward-compat) Polyak path:
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

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Replace EMA shadow state from a checkpoint mapping."""
        if not isinstance(state, Mapping):
            raise TypeError(f"state must be Mapping; got {type(state).__name__}")
        self._shadow = self._clone_state_dict(state)

    def compensation_state_dict(self) -> dict[str, Any]:
        """Return a clone of the Kahan compensation state."""
        return self._clone_state_dict(self._kahan_compensation)

    def load_compensation_state_dict(self, state: Mapping[str, Any]) -> None:
        """Replace Kahan compensation state from a checkpoint mapping."""
        if not self.enable_kahan and state:
            raise ValueError("cannot load Kahan compensation when enable_kahan=False")
        if not isinstance(state, Mapping):
            raise TypeError(f"state must be Mapping; got {type(state).__name__}")
        self._kahan_compensation = self._clone_state_dict(state)

    @property
    def kahan_fallback_keys(self) -> tuple[str, ...]:
        """Tensor keys that degraded from Kahan to naive in non-strict mode."""
        return tuple(sorted(self._kahan_fallback_keys))

    def write_compensation_state(self, path: Path) -> None:
        """Persist Kahan compensation state for checkpoint/resume durability."""
        import pickle

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self._kahan_compensation, fh, protocol=pickle.HIGHEST_PROTOCOL)

    def load_compensation_state(self, path: Path) -> None:
        """Load persisted Kahan compensation state."""
        import pickle

        with path.open("rb") as fh:
            state = pickle.load(fh)
        self.load_compensation_state_dict(state)

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


class KahanCompensatedPolyakEMAShadow(PolyakEMAShadow):
    """Canonical Kahan-compensated Polyak EMA shadow primitive.

    Sister of ``PolyakEMAShadow`` per Catalog #265 narrow public API:
    callers who want the hardened M2-mitigated semantics by construction
    instantiate this wrapper class instead of the canonical primitive;
    callers who want the canonical backward-compatible behavior keep
    using ``PolyakEMAShadow`` (which defaults ``enable_kahan=False``).

    Per T3 grand council OP #2 (commit ``7d04474cb``) Class 1-SCOPED
    Kahan-EMA surgical mitigation of M2 (EMA shadow drift accumulation
    through Polyak 0.997 decay's ~333-step exponential moving average).
    Per Kahan's classical 1965 result, error of N additions reduces
    from O(N * ULP) to O(ULP); for Polyak EMA the effective window is
    ``1 / (1 - decay) ≈ 333`` steps at the canonical 0.997 decay so
    the unmitigated accumulation upper-bound is ~333 ULPs per shadow
    element, while Kahan reduces this to ~1 ULP regardless of training
    depth.

    Per CLAUDE.md "consolidate everything into META layer or canonical
    helpers" standing directive: this is a thin canonical wrapper that
    inherits PolyakEMAShadow's full duck-typed surface (torch / MLX /
    plain Python list / numpy) and only overrides the default
    ``enable_kahan`` flag. Canonical long-training callers opt into this
    path with ``LongTrainingConfig(ema_accumulation="kahan")``; default
    ``"naive"`` preserves historical behavior.

    Per CLAUDE.md "EMA -- NON-NEGOTIABLE" + Quantizr PR101 anchor:
    decay=0.997 is the canonical default. Per CLAUDE.md "Forbidden
    empirical-claim-without-evidence-tag": empirical drift-reduction
    ratio vs naive is verifiable at $0 MLX-local via
    ``tools/smoke_kahan_ema_vs_naive_z6.py`` (T3 OP #3 Carmack 30-min
    smoke).
    """

    def __init__(
        self,
        model: Any,
        decay: float = CANONICAL_EMA_DECAY,
        *,
        strict_kahan: bool = False,
    ):
        super().__init__(
            model,
            decay=decay,
            enable_kahan=True,
            strict_kahan=strict_kahan,
        )


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

    @staticmethod
    def _resolved_export_path(requested_path: Path) -> Path:
        """Return the file the adapter actually emitted for a requested state path."""
        if requested_path.is_file():
            return requested_path
        for suffix in (".npsd", ".npz"):
            candidate = requested_path.with_suffix(requested_path.suffix + suffix)
            if candidate.is_file():
                return candidate
        return requested_path

    def write(
        self,
        adapter: SubstrateLongTrainingAdapter,
        ema_shadow: PolyakEMAShadow,
        global_epoch: int,
        loss: float,
        wall_clock_seconds: float,
        is_final: bool = False,
        checkpoint_role: str | None = None,
        selection_metric_key: str | None = None,
        selection_metric_value: float | None = None,
        selection_metric_mode: str | None = None,
    ) -> Path:
        """Write canonical checkpoint (live + EMA shadow + metadata)."""
        role = str(checkpoint_role or ("final" if is_final else "periodic"))
        if role not in {"periodic", "best", "final"}:
            raise ValueError(
                "checkpoint_role must be one of {'periodic', 'best', 'final'}; "
                f"got {role!r}"
            )
        ts = _utc_now_iso().replace(":", "").replace("-", "")
        stem = f"epoch{global_epoch:06d}_{ts}"
        if role == "final":
            stem = f"final_{stem}"
        elif role == "best":
            stem = f"best_{stem}"
        live_path = self.checkpoint_dir / f"{stem}.live.state"
        ema_path = self.checkpoint_dir / f"{stem}.ema_shadow.state"
        kahan_compensation_path = (
            self.checkpoint_dir / f"{stem}.ema_kahan_compensation.pkl"
            if ema_shadow.enable_kahan
            else None
        )
        meta_path = self.checkpoint_dir / f"{stem}.meta.json"

        # Write live state via adapter's canonical export.
        adapter.export_state_dict(adapter.model, live_path)
        live_state_path = self._resolved_export_path(live_path)

        # Write EMA shadow via snapshot+restore + adapter's export.
        live_snapshot = ema_shadow.apply_to(adapter.model)
        try:
            adapter.export_state_dict(adapter.model, ema_path)
        finally:
            ema_shadow.restore_from_snapshot(adapter.model, live_snapshot)
        ema_state_path = self._resolved_export_path(ema_path)

        if kahan_compensation_path is not None:
            ema_shadow.write_compensation_state(kahan_compensation_path)

        # Write canonical metadata JSON.
        meta = {
            "schema_version": "long_training_canonical_checkpoint.v1",
            "substrate_id": self.substrate_id,
            "lane_id": self.lane_id,
            "curriculum_hash": self.curriculum_hash,
            "global_epoch": int(global_epoch),
            "loss": float(loss),
            "checkpoint_selection_metric_key": (
                str(selection_metric_key) if selection_metric_key else "total"
            ),
            "checkpoint_selection_metric_value": (
                float(selection_metric_value)
                if selection_metric_value is not None
                else float(loss)
            ),
            "checkpoint_selection_metric_mode": (
                str(selection_metric_mode) if selection_metric_mode else "min"
            ),
            "wall_clock_seconds": float(wall_clock_seconds),
            "is_final": role == "final",
            "is_best": role == "best",
            "checkpoint_role": role,
            "ema_accumulation": "kahan" if ema_shadow.enable_kahan else "naive",
            "ema_kahan_enabled": bool(ema_shadow.enable_kahan),
            "ema_kahan_strict": bool(ema_shadow.strict_kahan),
            "ema_kahan_fallback_keys": list(ema_shadow.kahan_fallback_keys),
            "ema_kahan_compensation_key_count": len(
                ema_shadow.compensation_state_dict()
            ),
            "ema_kahan_compensation_state_path": (
                str(kahan_compensation_path) if kahan_compensation_path else None
            ),
            "live_state_path": str(live_state_path),
            "ema_shadow_state_path": str(ema_state_path),
            "captured_at_utc": _utc_now_iso(),
            **CANONICAL_NON_PROMOTABLE_MARKERS,
        }
        meta_path.write_text(json.dumps(meta, sort_keys=True, indent=2) + "\n")
        return meta_path

    @staticmethod
    def resolve_resume_metadata_path(resume_from: Path) -> Path:
        """Resolve a resume path to canonical checkpoint metadata JSON.

        Operators naturally copy the hot state path from training reports
        (``*.live.state.npsd`` / ``*.ema_shadow.state.npsd``).  The metadata
        remains the authority for substrate/curriculum guards, so accept the
        state path only by resolving its sibling ``*.meta.json``.
        """

        if not resume_from.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_from}")
        if resume_from.name.endswith(".meta.json"):
            return resume_from
        for suffix in (".live.state.npsd", ".ema_shadow.state.npsd"):
            if resume_from.name.endswith(suffix):
                meta_path = resume_from.with_name(
                    resume_from.name[: -len(suffix)] + ".meta.json"
                )
                if meta_path.is_file():
                    return meta_path
                raise FileNotFoundError(
                    "resume checkpoint state exists but sibling metadata is "
                    f"missing: state={resume_from}, expected_meta={meta_path}"
                )
        return resume_from

    def load_resume_metadata(
        self,
        resume_from: Path,
        *,
        current_curriculum_stages: Sequence[CurriculumStage] | None = None,
    ) -> Mapping[str, Any]:
        """Load checkpoint metadata; refuse cross-substrate / cross-curriculum resume."""
        resume_from = self.resolve_resume_metadata_path(resume_from)
        if not resume_from.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_from}")
        meta = json.loads(resume_from.read_text(encoding="utf-8"))
        if meta.get("substrate_id") != self.substrate_id:
            raise ValueError(
                f"resume checkpoint substrate_id={meta.get('substrate_id')!r} "
                f"!= config substrate_id={self.substrate_id!r}; refusing per "
                f"Catalog #229 PV cross-substrate-resume guard."
            )
        if meta.get("curriculum_hash") != self.curriculum_hash:
            boundary_epoch = int(meta.get("global_epoch", -1)) + 1
            matching_boundary_epoch = (
                _matching_prefix_curriculum_boundary(
                    current_curriculum_stages,
                    min_boundary_epoch=boundary_epoch,
                    expected_hash=str(meta.get("curriculum_hash")),
                )
                if current_curriculum_stages is not None
                else None
            )
            if matching_boundary_epoch is not None:
                meta = {
                    **dict(meta),
                    "resume_curriculum_validation": {
                        "schema": "long_training_prefix_compatible_resume.v1",
                        "mode": "prefix_compatible_future_extension",
                        "checkpoint_curriculum_hash": str(meta.get("curriculum_hash")),
                        "current_curriculum_hash": self.curriculum_hash,
                        "prefix_curriculum_hash": str(meta.get("curriculum_hash")),
                        "prefix_boundary_epoch": int(matching_boundary_epoch),
                        "resume_state_next_epoch": int(boundary_epoch),
                    },
                }
                return meta
            raise ValueError(
                "resume checkpoint curriculum_hash differs; refusing per "
                "Catalog #229 PV (curriculum must match for valid resume)."
            )
        return meta


CHECKPOINT_RETENTION_PASS_SCHEMA = "long_training_checkpoint_retention_pass.v1"


def _load_checkpoint_record(meta_path: Path) -> dict[str, Any] | None:
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(meta, Mapping):
        return None
    files: list[Path] = [meta_path]
    for key in (
        "live_state_path",
        "ema_shadow_state_path",
        "ema_kahan_compensation_state_path",
    ):
        value = meta.get(key)
        if isinstance(value, str) and value:
            candidate = Path(value).expanduser().resolve(strict=False)
            if candidate.exists() and candidate not in files:
                files.append(candidate)
    return {
        "meta_path": meta_path,
        "meta": dict(meta),
        "global_epoch": int(meta.get("global_epoch", -1)),
        "loss": float(meta.get("loss", float("inf"))),
        "is_final": bool(meta.get("is_final")),
        "is_best": bool(meta.get("is_best"))
        or str(meta.get("checkpoint_role") or "") == "best",
        "checkpoint_role": str(meta.get("checkpoint_role") or ""),
        "files": files,
    }


def _checkpoint_retention_default_cold_store_roots() -> tuple[Path, ...]:
    try:
        from comma_lab.operator_storage_waterfall import operator_cold_store_roots

        return tuple(Path(root) for root in operator_cold_store_roots())
    except Exception:
        return ()


def _checkpoint_retention_cold_store_roots(config: LongTrainingConfig) -> tuple[Path, ...]:
    roots = tuple(config.checkpoint_retention_cold_store_roots)
    return roots or _checkpoint_retention_default_cold_store_roots()


def _checkpoint_record_file_status(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": path.as_posix(),
        "bytes": int(stat.st_size),
        "sha256": _sha256_file(path),
    }


def _checkpoint_retention_group_bytes(record: Mapping[str, Any]) -> int:
    total = 0
    for path in record.get("files", ()):
        if isinstance(path, Path) and path.exists():
            total += int(path.stat().st_size)
    return total


def _checkpoint_retention_root_slug(checkpoint_dir: Path) -> str:
    resolved = checkpoint_dir.expanduser().resolve(strict=False)
    return "_".join(part for part in resolved.parts if part and part != "/")


def _select_checkpoint_retention_cold_store_root(
    roots: Sequence[Path],
    *,
    required_bytes: int,
    source: Path,
) -> tuple[Path, dict[str, Any]] | None:
    eligible: list[tuple[bool, Path, dict[str, Any]]] = []
    try:
        source_device = source.stat().st_dev
    except OSError:
        source_device = None
    for root in roots:
        resolved = root.expanduser().resolve(strict=False)
        try:
            resolved.mkdir(parents=True, exist_ok=True)
            usage = shutil.disk_usage(resolved)
            if int(usage.free) < int(required_bytes):
                continue
            same_device = (
                source_device is not None and resolved.stat().st_dev == source_device
            )
            contract = {
                "schema": "long_training_checkpoint_retention_cold_store_root.v1",
                "root": resolved.as_posix(),
                "free_bytes": int(usage.free),
                "required_bytes": int(required_bytes),
                "same_device_as_source": bool(same_device),
            }
            eligible.append((bool(same_device), resolved, contract))
        except OSError:
            continue
    if not eligible:
        return None
    eligible.sort(key=lambda item: (item[0], item[1].as_posix()))
    _, root, contract = eligible[0]
    return root, contract


def _move_checkpoint_file_to_cold_store(
    source: Path,
    destination: Path,
    *,
    source_status: Mapping[str, Any],
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing_sha = _sha256_file(destination)
        if existing_sha != source_status["sha256"]:
            raise RuntimeError(
                "checkpoint retention destination exists with different sha256: "
                f"{destination}"
            )
        source.unlink()
        return {
            "source": source.as_posix(),
            "cold_store_path": destination.as_posix(),
            "bytes": int(source_status["bytes"]),
            "sha256": source_status["sha256"],
            "status": "moved_existing_verified",
        }
    tmp = destination.with_name(
        f"{destination.name}.tmp.{os.getpid()}.{time.time_ns()}"
    )
    try:
        shutil.copy2(source, tmp)
        copied_sha = _sha256_file(tmp)
        if copied_sha != source_status["sha256"]:
            raise RuntimeError(
                "checkpoint retention copy sha256 mismatch for "
                f"{source}: {copied_sha} != {source_status['sha256']}"
            )
        tmp.replace(destination)
        final_sha = _sha256_file(destination)
        if final_sha != source_status["sha256"]:
            raise RuntimeError(
                "checkpoint retention final sha256 mismatch for "
                f"{destination}: {final_sha} != {source_status['sha256']}"
            )
        source.unlink()
    finally:
        if tmp.exists():
            tmp.unlink()
    return {
        "source": source.as_posix(),
        "cold_store_path": destination.as_posix(),
        "bytes": int(source_status["bytes"]),
        "sha256": source_status["sha256"],
        "status": "moved_verified",
    }


def _apply_checkpoint_retention_to_record(
    record: Mapping[str, Any],
    *,
    config: LongTrainingConfig,
    roots: Sequence[Path],
) -> dict[str, Any]:
    meta_path = record["meta_path"]
    group_files = [path for path in record.get("files", ()) if isinstance(path, Path)]
    existing_files = [path for path in group_files if path.exists()]
    file_statuses = [_checkpoint_record_file_status(path) for path in existing_files]
    group_bytes = sum(int(status["bytes"]) for status in file_statuses)
    row: dict[str, Any] = {
        "schema": "long_training_checkpoint_retention_group.v1",
        "meta_path": meta_path.as_posix(),
        "global_epoch": int(record.get("global_epoch", -1)),
        "loss": float(record.get("loss", float("inf"))),
        "is_final": bool(record.get("is_final")),
        "is_best": bool(record.get("is_best")),
        "checkpoint_role": str(record.get("checkpoint_role") or ""),
        "action": config.checkpoint_retention_action,
        "bytes": int(group_bytes),
        "files": file_statuses,
        **CANONICAL_NON_PROMOTABLE_MARKERS,
    }
    if not existing_files:
        row["status"] = "skipped_missing"
        return row
    if config.checkpoint_retention_action == "delete":
        for path in existing_files:
            path.unlink()
        row["status"] = "deleted_explicit"
        return row
    selected = _select_checkpoint_retention_cold_store_root(
        roots,
        required_bytes=group_bytes,
        source=meta_path,
    )
    if selected is None:
        row["status"] = "blocked_no_cold_store_capacity"
        row["blockers"] = ["checkpoint_retention_no_cold_store_capacity"]
        return row
    cold_root, cold_contract = selected
    destination_dir = (
        cold_root
        / "long_training_checkpoint_retention"
        / _checkpoint_retention_root_slug(config.resolved_checkpoint_dir())
        / str(meta_path.stem)
    )
    moved_files: list[dict[str, Any]] = []
    for path, status in zip(existing_files, file_statuses, strict=True):
        moved_files.append(
            _move_checkpoint_file_to_cold_store(
                path,
                destination_dir / path.name,
                source_status=status,
            )
        )
    row["status"] = "moved"
    row["cold_store_contract"] = cold_contract
    row["cold_store_dir"] = destination_dir.as_posix()
    row["moved_files"] = moved_files
    return row


def _append_checkpoint_retention_manifest(
    manifest_path: Path,
    payload: Mapping[str, Any],
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = manifest_path.with_suffix(manifest_path.suffix + ".lock")
    lock_path.touch(exist_ok=True)
    with open(lock_path, "r+") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        try:
            with open(manifest_path, "a", encoding="utf-8") as out_fh:
                out_fh.write(json.dumps(_jsonable(payload), sort_keys=True))
                out_fh.write("\n")
        finally:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)


def apply_checkpoint_retention(
    config: LongTrainingConfig,
    *,
    trigger_meta_path: Path | None = None,
) -> dict[str, Any] | None:
    """Apply loss-ranked hot checkpoint retention with no silent signal loss."""

    if not config.checkpoint_retention_enabled:
        return None
    checkpoint_dir = config.resolved_checkpoint_dir()
    records = [
        record
        for meta_path in sorted(checkpoint_dir.glob("*.meta.json"))
        if (record := _load_checkpoint_record(meta_path)) is not None
    ]
    periodic = [
        record
        for record in records
        if not bool(record["is_final"]) and not bool(record["is_best"])
    ]
    protected_meta_paths = {
        record["meta_path"]
        for record in records
        if bool(record["is_final"]) or bool(record["is_best"])
    }
    if config.resume_from_checkpoint is not None:
        protected_meta_paths.add(
            CheckpointWriter.resolve_resume_metadata_path(
                config.resume_from_checkpoint.expanduser().resolve(strict=False)
            )
        )
    keep_last_n = config.checkpoint_retention_keep_last_n
    if keep_last_n is not None and keep_last_n > 0:
        by_epoch = sorted(
            periodic,
            key=lambda record: (
                int(record["global_epoch"]),
                record["meta_path"].as_posix(),
            ),
        )
        protected_meta_paths.update(record["meta_path"] for record in by_epoch[-keep_last_n:])
    keep_best_n = int(config.checkpoint_retention_keep_best_n)
    if keep_best_n > 0:
        finite_records = [
            record
            for record in periodic
            if record["loss"] == record["loss"] and record["loss"] != float("inf")
        ]
        by_loss = sorted(
            finite_records,
            key=lambda record: (
                float(record["loss"]),
                int(record["global_epoch"]),
                record["meta_path"].as_posix(),
            ),
        )
        protected_meta_paths.update(record["meta_path"] for record in by_loss[:keep_best_n])
    keep_every = config.checkpoint_retention_keep_every_n_epochs
    if keep_every is not None:
        protected_meta_paths.update(
            record["meta_path"]
            for record in periodic
            if (int(record["global_epoch"]) + 1) % int(keep_every) == 0
        )
    candidates = [
        record
        for record in periodic
        if record["meta_path"] not in protected_meta_paths
    ]
    roots = _checkpoint_retention_cold_store_roots(config)
    rows = [
        _apply_checkpoint_retention_to_record(record, config=config, roots=roots)
        for record in candidates
    ]
    if not rows:
        return None
    payload = {
        "schema": CHECKPOINT_RETENTION_PASS_SCHEMA,
        "captured_at_utc": _utc_now_iso(),
        "checkpoint_dir": checkpoint_dir.as_posix(),
        "trigger_meta_path": (
            None if trigger_meta_path is None else trigger_meta_path.as_posix()
        ),
        "policy": config.as_dict()["checkpoint_retention"],
        "record_count_before": len(records),
        "periodic_record_count_before": len(periodic),
        "protected_meta_paths": sorted(path.as_posix() for path in protected_meta_paths),
        "candidate_count": len(candidates),
        "rows": rows,
        "moved_count": sum(1 for row in rows if row.get("status") == "moved"),
        "deleted_count": sum(
            1 for row in rows if row.get("status") == "deleted_explicit"
        ),
        "blocked_count": sum(
            1 for row in rows if str(row.get("status", "")).startswith("blocked")
        ),
        "bytes_processed": sum(int(row.get("bytes", 0)) for row in rows),
        "cold_store_roots": [path.as_posix() for path in roots],
        **CANONICAL_NON_PROMOTABLE_MARKERS,
    }
    _append_checkpoint_retention_manifest(
        config.resolved_checkpoint_retention_manifest_path(),
        payload,
    )
    return payload


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
        return "out of memory" in msg or "oom" in msg or ("memory" in msg and "cuda" in msg)

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
            adapter.train_step
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
            {
                "config_snapshot": dict(artifact.config_snapshot),
                "substrate_artifact_metadata": dict(
                    artifact.substrate_artifact_metadata
                ),
            },
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
                json.dumps(
                    {
                        "config_snapshot": dict(artifact.config_snapshot),
                        "substrate_artifact_metadata": dict(
                            artifact.substrate_artifact_metadata
                        ),
                    },
                    sort_keys=True,
                )
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

        extra_manifest_fields: dict[str, Any] = {
            "long_training_canonical_helper": "tac.training.long_training_canonical.run_long_training",
            "long_training_lane_id": artifact.lane_id,
            "long_training_schema_version": artifact.schema_version,
            "long_training_epochs_completed": int(artifact.total_epochs_completed),
            "long_training_early_stopped": bool(artifact.early_stopped),
        }
        if artifact.substrate_artifact_metadata:
            extra_manifest_fields["substrate_artifact_metadata"] = dict(
                artifact.substrate_artifact_metadata
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
            extra_manifest_fields=extra_manifest_fields,
        )
        return bool(anchor.posterior_update.accepted), anchor.posterior_update.refusal_reason
    except ImportError as exc:
        return False, f"posterior_emission_helper_import_failed:{exc!s}"
    except Exception as exc:
        # Best-effort: never break the artifact emission on posterior helper failure.
        # Per Catalog #339 silent-no-spawn-class self-protection: surface the failure.
        return False, f"posterior_emission_helper_runtime_failed:{type(exc).__name__}:{exc!s}"


def _load_checkpoint_model_state(
    adapter: SubstrateLongTrainingAdapter,
    state_path: Path,
) -> None:
    """Load a checkpoint state into ``adapter.model`` or fail closed.

    The canonical writer only requires ``export_state_dict`` because
    substrates serialize differently. Resume is stricter: an adapter that uses
    a non-JSON/binary format must provide ``import_state_dict(path)`` so the
    helper never pretends that metadata-only resume restored real weights.
    """

    if hasattr(adapter, "import_state_dict") and callable(adapter.import_state_dict):
        adapter.import_state_dict(adapter.model, state_path)
        return
    if not hasattr(adapter.model, "load_state_dict") or not callable(
        adapter.model.load_state_dict
    ):
        raise RuntimeError(
            "resume_from_checkpoint requires adapter.import_state_dict(model, path) "
            "or model.load_state_dict(JSON_state); refusing metadata-only resume"
        )
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "resume_from_checkpoint points at a non-JSON state file but adapter "
            "does not expose import_state_dict(model, path); refusing fake resume"
        ) from exc
    if not isinstance(state, Mapping):
        raise RuntimeError(
            f"resume state file must decode to a Mapping; got {type(state).__name__}"
        )
    adapter.model.load_state_dict(state)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _archive_selection_written_archive_evidence(candidate_dir: Path) -> dict[str, Any]:
    """Return archive evidence emitted before a selector failure, if present."""

    archive_path = candidate_dir / "archive.zip"
    if not archive_path.is_file():
        return {}
    archive_bytes = archive_path.stat().st_size
    return {
        "archive_path": archive_path.as_posix(),
        "archive_sha256": _sha256_file(archive_path),
        "archive_bytes": int(archive_bytes),
        "emitted_archive_available": True,
        "emitted_archive_evidence_source": (
            "candidate_archive_zip_written_before_selection_failure"
        ),
    }


def _archive_selection_components(
    adapter: SubstrateLongTrainingAdapter,
    config: LongTrainingConfig,
    *,
    seed_offset: int,
) -> Mapping[str, float] | None:
    """Return deterministic local score-aware components for archive selection."""

    try:
        batch = adapter.sample_batch(
            config.batch_pair_indices_per_step,
            config.seed + int(seed_offset),
        )
        components = adapter.score_aware_components(adapter.model, batch)
    except (NotImplementedError, AttributeError):
        return None
    except Exception as exc:
        print(
            "[long_training_canonical] WARN: archive-selection "
            f"score_aware_components failed: {exc!r}"
        )
        return None
    if components is None:
        out: dict[str, float] = {}
    else:
        out = {}
        for key, value in components.items():
            try:
                out[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
    health_hook = getattr(adapter, "archive_selection_health", None)
    if callable(health_hook):
        try:
            health = health_hook(adapter.model, batch)
        except Exception as exc:
            print(
                "[long_training_canonical] WARN: archive-selection "
                f"health probe failed: {exc!r}"
            )
            health = None
        if isinstance(health, Mapping):
            for key, value in health.items():
                try:
                    out[f"selection_health_{key}"] = float(value)
                except (TypeError, ValueError):
                    continue
    if not out:
        return None
    return out


def _adapter_batch_observability(adapter: Any, batch: Any | None = None) -> Mapping[str, Any] | None:
    hook = getattr(adapter, "batch_observability", None)
    if callable(hook):
        try:
            observed = hook(batch)
        except TypeError:
            observed = hook()
        if isinstance(observed, Mapping):
            return observed
    observed = getattr(adapter, "last_batch_observability", None)
    if isinstance(observed, Mapping):
        return observed
    return None


def _archive_selection_proxy_score(
    components: Mapping[str, float] | None,
    *,
    archive_bytes: int,
) -> float:
    """Compose the local false-authority proxy score for live-vs-EMA archives."""

    proxy = float(int(archive_bytes) * CONTEST_RATE_SCORE_PER_BYTE)
    if not components:
        return proxy
    for key, value in components.items():
        if key in {"archive_bytes", "bytes", "rate"}:
            continue
        if str(key).startswith("selection_health_"):
            continue
        proxy += float(value)
    return proxy


def _archive_selection_replay_components(
    adapter: SubstrateLongTrainingAdapter,
    config: LongTrainingConfig,
    *,
    archive_path: Path,
    seed_offset: int,
    candidate_kind: str,
) -> Mapping[str, float] | None:
    """Return parse-back scorer components for an exported candidate archive."""

    hook = getattr(adapter, "archive_replay_components", None)
    if not callable(hook):
        if config.archive_selection_replay_required:
            raise RuntimeError(
                "archive_selection_replay_required_but_adapter_missing_archive_replay_components"
            )
        return None
    batch_size = (
        int(config.archive_selection_replay_batch_size)
        if config.archive_selection_replay_batch_size is not None
        else int(config.batch_pair_indices_per_step)
    )
    batch = adapter.sample_batch(
        batch_size,
        config.seed + int(seed_offset) + 7_000_000,
    )
    try:
        replay = hook(
            Path(archive_path),
            batch,
            candidate_kind=str(candidate_kind),
        )
    except TypeError:
        replay = hook(Path(archive_path), batch)
    if replay is None:
        if config.archive_selection_replay_required:
            raise RuntimeError(
                "archive_selection_replay_required_but_archive_replay_components_returned_none"
            )
        return None
    if not isinstance(replay, Mapping):
        raise TypeError(
            "archive_replay_components must return Mapping[str, float] or None; "
            f"got {type(replay).__name__}"
        )
    out: dict[str, float] = {}
    for key, value in replay.items():
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed):
            out[str(key)] = parsed
    if not out and config.archive_selection_replay_required:
        raise RuntimeError(
            "archive_selection_replay_required_but_archive_replay_components_empty"
        )
    return out or None


def _finite_component(
    components: Mapping[str, Any],
    key: str,
) -> float | None:
    raw = components.get(key)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _archive_selection_health_sort_key(
    row: Mapping[str, Any]
) -> tuple[int, float, float, float]:
    """Return hard health tier before proxy-score archive selection.

    Training-time direct-live SegNet escape can be erased by EMA smoothing.
    When an adapter exposes target-class coverage or target-class mass, prefer
    an archive view that preserves the material target classes the upstream
    SegNet actually sees. Older adapters that only expose occupied-class
    fraction retain the previous non-collapse sort behavior.
    """

    components = row.get("parseback_score_components")
    if not isinstance(components, Mapping) or not components:
        components = row.get("score_components")
    if not isinstance(components, Mapping):
        return (0, 0.0, 0.0, 0.0)
    occupied = _finite_component(
        components,
        "selection_health_segnet_direct_live_candidate_occupied_class_fraction"
    )
    target_coverage = _finite_component(
        components,
        "selection_health_segnet_direct_live_candidate_target_class_coverage_fraction",
    )
    target_any_coverage = _finite_component(
        components,
        "selection_health_segnet_direct_live_candidate_target_any_class_coverage_fraction",
    )
    target_min_ratio = _finite_component(
        components,
        "selection_health_segnet_direct_live_candidate_target_class_min_ratio",
    )
    if occupied is None and target_coverage is None and target_any_coverage is None:
        return (0, 0.0, 0.0, 0.0)
    effective_target_coverage = (
        target_coverage if target_coverage is not None else target_any_coverage
    )
    target_collapsed = (
        effective_target_coverage is not None
        and effective_target_coverage
        < CANONICAL_SEGNET_TARGET_CLASS_COVERAGE_FRACTION_FOR_FIT_GATE
    )
    target_min_ratio_collapsed = (
        target_min_ratio is not None
        and target_min_ratio < CANONICAL_SEGNET_TARGET_CLASS_MIN_RATIO_FOR_FIT_GATE
    )
    # Match the receiver/export SegNet argmax survival gate: two occupied
    # classes out of five is still a collapse for scorer-faithful HiNeRV.
    occupied_collapsed = (
        occupied is not None
        and occupied
        < CANONICAL_SEGNET_ARGMAX_MIN_OCCUPIED_CLASS_FRACTION_FOR_FIT_GATE
    )
    tier = (
        2
        if target_collapsed or target_min_ratio_collapsed
        else 1
        if occupied_collapsed
        else 0
    )
    return (
        tier,
        -(effective_target_coverage or 0.0),
        -(target_min_ratio or 0.0),
        -(occupied or 0.0),
    )


def _export_live_ema_archive_selection(
    *,
    adapter: SubstrateLongTrainingAdapter,
    config: LongTrainingConfig,
    ema_shadow: PolyakEMAShadow,
) -> tuple[Path | None, str | None, int | None, Path]:
    """Export live and EMA archives, then select by local proxy score.

    The manifest is the durable truth. It records both candidate archive hashes,
    bytes, local score-aware components when available, and false-authority
    markers. This closes the engineering gap where a run blindly exported the
    EMA view without proving that the archive-selected view was preferable.
    """

    selection_dir = config.output_dir / "ema_archive_selection"
    selection_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = selection_dir / "ema_archive_selection.json"
    rows: list[dict[str, Any]] = []

    def _record_failure(
        kind: str,
        exc: Exception,
        *,
        candidate_dir: Path | None = None,
    ) -> None:
        written_archive_evidence = (
            _archive_selection_written_archive_evidence(candidate_dir)
            if candidate_dir is not None
            else {}
        )
        rows.append(
            {
                "schema": "long_training_archive_selection_candidate.v1",
                "candidate_kind": kind,
                "status": "failed",
                "failure": f"{type(exc).__name__}:{exc!s}",
                **written_archive_evidence,
                **CANONICAL_NON_PROMOTABLE_MARKERS,
            }
        )

    def _export_candidate(kind: str, *, seed_offset: int) -> None:
        candidate_dir = selection_dir / kind
        try:
            components = _archive_selection_components(
                adapter,
                config,
                seed_offset=seed_offset,
            )
            result = adapter.export_archive(adapter.model, candidate_dir)
            if result is None:
                rows.append(
                    {
                        "schema": "long_training_archive_selection_candidate.v1",
                        "candidate_kind": kind,
                        "status": "deferred",
                        "archive_path": None,
                        "archive_sha256": None,
                        "archive_bytes": None,
                        "score_components": dict(components or {}),
                        **CANONICAL_NON_PROMOTABLE_MARKERS,
                    }
                )
                return
            archive_path, archive_sha256, archive_bytes = result
            archive_path = Path(archive_path)
            archive_bytes = int(archive_bytes)
            if not archive_path.is_file():
                raise FileNotFoundError(str(archive_path))
            actual_sha = _sha256_file(archive_path)
            if actual_sha != str(archive_sha256):
                raise ValueError(
                    "archive sha mismatch: "
                    f"reported={archive_sha256} actual={actual_sha}"
                )
            proxy = _archive_selection_proxy_score(
                components,
                archive_bytes=archive_bytes,
            )
            parseback_components = _archive_selection_replay_components(
                adapter,
                config,
                archive_path=archive_path,
                seed_offset=seed_offset,
                candidate_kind=kind,
            )
            parseback_proxy = (
                None
                if parseback_components is None
                else _archive_selection_proxy_score(
                    parseback_components,
                    archive_bytes=archive_bytes,
                )
            )
            selection_proxy = proxy if parseback_proxy is None else parseback_proxy
            selection_authority = (
                "local_training_proxy_false_authority"
                if parseback_proxy is None
                else "archive_parseback_replay_proxy_false_authority"
            )
            rows.append(
                {
                    "schema": "long_training_archive_selection_candidate.v1",
                    "candidate_kind": kind,
                    "status": "exported",
                    "archive_path": archive_path.as_posix(),
                    "archive_sha256": str(archive_sha256),
                    "archive_bytes": archive_bytes,
                    "score_components": dict(components or {}),
                    "proxy_score": float(proxy),
                    "parseback_score_components": dict(parseback_components or {}),
                    "parseback_proxy_score": (
                        None if parseback_proxy is None else float(parseback_proxy)
                    ),
                    "selection_proxy_score": float(selection_proxy),
                    "selection_authority": selection_authority,
                    "proxy_score_terms": {
                        "rate_score_per_byte": CONTEST_RATE_SCORE_PER_BYTE,
                        "score_components_are_local_training_proxy": True,
                        "parseback_score_components_available": (
                            parseback_components is not None
                        ),
                        "selection_uses_parseback_score_components": (
                            parseback_proxy is not None
                        ),
                    },
                    **CANONICAL_NON_PROMOTABLE_MARKERS,
                }
            )
        except Exception as exc:
            _record_failure(kind, exc, candidate_dir=candidate_dir)

    _export_candidate("live", seed_offset=2_000_000)
    live_snapshot = ema_shadow.apply_to(adapter.model)
    try:
        _export_candidate("ema", seed_offset=2_000_001)
    finally:
        ema_shadow.restore_from_snapshot(adapter.model, live_snapshot)

    exported = [row for row in rows if row.get("status") == "exported"]
    selected = None
    if exported:
        selected = min(
            exported,
            key=lambda row: (
                *_archive_selection_health_sort_key(row),
                float(row.get("selection_proxy_score", row["proxy_score"])),
                0 if row.get("candidate_kind") == "ema" else 1,
            ),
        )
    manifest = {
        "schema": "long_training_ema_archive_selection.v1",
        "enabled": True,
        "substrate_id": config.substrate_id,
        "lane_id": config.lane_id,
        "selection_metric": (
            "archive_parseback_proxy_when_available_else_local_proxy_plus_charged_archive_rate"
        ),
        "authority": (
            "archive_parseback_replay_proxy_false_authority"
            if any(
                row.get("selection_authority")
                == "archive_parseback_replay_proxy_false_authority"
                for row in exported
            )
            else "local_training_proxy_false_authority"
        ),
        "archive_selection_replay_required": bool(
            config.archive_selection_replay_required
        ),
        "archive_selection_replay_batch_size": (
            None
            if config.archive_selection_replay_batch_size is None
            else int(config.archive_selection_replay_batch_size)
        ),
        "rate_score_per_byte": CONTEST_RATE_SCORE_PER_BYTE,
        "candidate_count": len(rows),
        "exported_candidate_count": len(exported),
        "selected_candidate_kind": (
            None if selected is None else selected.get("candidate_kind")
        ),
        "selected_archive_path": (
            None if selected is None else selected.get("archive_path")
        ),
        "selected_archive_sha256": (
            None if selected is None else selected.get("archive_sha256")
        ),
        "selected_archive_bytes": (
            None if selected is None else selected.get("archive_bytes")
        ),
        "rows": rows,
        "captured_at_utc": _utc_now_iso(),
        **CANONICAL_NON_PROMOTABLE_MARKERS,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if selected is None:
        return None, None, None, manifest_path
    return (
        Path(str(selected["archive_path"])),
        str(selected["archive_sha256"]),
        int(selected["archive_bytes"]),
        manifest_path,
    )


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
        ...     lane_id="lane_my_substrate_l2_20260526",  # FAKE_LANE_OK:test_fixture_or_docstring_or_dict_key_reference_to_lane_token_lane_my_substrate_l2_20260526_NOT_a_real_lane_registry_pre_registration_per_catalog_126_false_positive_per_comprehensive_bug_audit_cascade_20260526
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
        flush_interval_epochs=config.telemetry_flush_interval_epochs,
    )
    checkpoint_writer = CheckpointWriter(
        checkpoint_dir=config.resolved_checkpoint_dir(),
        substrate_id=config.substrate_id,
        lane_id=config.lane_id,
        curriculum_hash=config.curriculum_hash(),
    )
    oom_runner = OOMSafeStepRunner()

    # 2) Resume metadata (best-effort; warn-on-failure is the caller's job).
    resume_global_epoch = 0
    resume_meta: Mapping[str, Any] | None = None
    if config.resume_from_checkpoint is not None:
        resume_meta = checkpoint_writer.load_resume_metadata(
            config.resume_from_checkpoint,
            current_curriculum_stages=config.curriculum_stages,
        )
        resume_global_epoch = int(resume_meta.get("global_epoch", -1)) + 1
        checkpoint_accumulation = str(resume_meta.get("ema_accumulation") or "naive")
        if checkpoint_accumulation != config.ema_accumulation:
            raise RuntimeError(
                "resume_from_checkpoint ema_accumulation mismatch: "
                f"checkpoint={checkpoint_accumulation!r}, config={config.ema_accumulation!r}"
            )
        ema_state_path = Path(str(resume_meta.get("ema_shadow_state_path") or ""))
        live_state_path = Path(str(resume_meta.get("live_state_path") or ""))
        _load_checkpoint_model_state(adapter, ema_state_path)

    ema_shadow = PolyakEMAShadow(
        adapter.model,
        decay=config.ema_decay,
        enable_kahan=config.ema_accumulation == "kahan",
        strict_kahan=config.ema_accumulation == "kahan",
    )
    if resume_meta is not None:
        kahan_compensation_path = resume_meta.get("ema_kahan_compensation_state_path")
        if config.ema_accumulation == "kahan":
            if not isinstance(kahan_compensation_path, str) or not kahan_compensation_path:
                raise RuntimeError(
                    "resume_from_checkpoint in Kahan mode requires "
                    "ema_kahan_compensation_state_path"
                )
            ema_shadow.load_compensation_state(Path(kahan_compensation_path))
        _load_checkpoint_model_state(adapter, live_state_path)

    # 3) Training loop.
    per_epoch_metrics: list[PerEpochMetrics] = []
    best_loss = float("inf")
    best_metric = (
        float("-inf")
        if config.checkpoint_selection_metric_mode == "max"
        else float("inf")
    )
    best_tie_break_metric: float | None = None
    best_epoch = -1
    best_wall_clock = 0.0
    best_live_state: Mapping[str, Any] | None = None
    best_ema_state: Mapping[str, Any] | None = None
    best_ema_compensation_state: Mapping[str, Any] | None = None
    best_state_capture_error: str | None = None
    checkpoint_selection_metric_blockers: list[str] = []
    last_selection_metric: float | None = None
    epochs_since_improvement = 0
    early_stopped = False
    early_stop_reason = ""
    t_start = time.time()
    final_epoch = resume_global_epoch
    # Most-recent durably-written checkpoint meta path; surfaced into the next
    # epoch's gating telemetry ``checkpoint_path`` (the canonical checkpoint
    # write happens AFTER the per-epoch telemetry record, so the row carries
    # the LAST saved checkpoint — an honest record, never a predicted path).
    last_checkpoint_meta_path: Path | None = None

    for epoch in range(resume_global_epoch, config.epochs):
        final_epoch = epoch
        stage = config.stage_at_epoch(epoch)
        effective_lr = config.learning_rate * stage.lr_scale

        # Notify the adapter of the current global epoch so any curriculum-aware
        # adapter (e.g. MlxScoreAwareAdapter with pr95_faithful_curriculum_enabled
        # per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline"
        # L14 + L15) can advance per-stage optimizer/loss-family/sigma/lambda/qat
        # state. Adapters that don't implement notify_global_epoch are a silent
        # no-op (backward compat per the canonical Protocol contract). This is
        # the canonical wiring point for the m9-v3 PR95-faithful 8-stage
        # Muon+AdamW curriculum sister-wave (commit c91481212 canonical
        # helper + adapter wiring + this notify_global_epoch invocation).
        notify_fn = getattr(adapter, "notify_global_epoch", None)
        if callable(notify_fn):
            try:
                notify_fn(epoch)
            except Exception as exc:
                # notify_global_epoch is observability-only; never fail the run.
                print(
                    f"[long_training_canonical] WARN: notify_global_epoch failed at epoch {epoch}: {exc!r}"
                )
        stage_notify_fn = getattr(adapter, "notify_curriculum_stage", None)
        if callable(stage_notify_fn):
            stage_notify_fn(epoch, stage)

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
        projection_report: Mapping[str, Any] | None = None
        projection_fn = getattr(adapter, "post_optimizer_projection", None)
        if callable(projection_fn):
            try:
                raw_projection_report = projection_fn(epoch=epoch)
                if isinstance(raw_projection_report, Mapping):
                    projection_report = raw_projection_report
            except Exception as exc:
                early_stopped = True
                early_stop_reason = f"post_optimizer_projection_failed:{exc!s}"
                traceback.print_exc()
                break
        train_batch_observability = _adapter_batch_observability(adapter)

        # 4) EMA update post-optimizer-step per canonical Polyak pattern.
        try:
            ema_shadow.update(adapter.model)
        except Exception as exc:
            if config.ema_accumulation == "kahan":
                raise
            # EMA update failure is recoverable; log + continue.
            traceback.print_exc()
            print(f"[long_training_canonical] WARN: EMA update failed at epoch {epoch}: {exc!r}")

        # 5) Optional per-axis decomposition per Catalog #356.
        per_axis: Mapping[str, float] | None = None
        try:
            sample = adapter.sample_batch(config.batch_pair_indices_per_step, config.seed + epoch + 1_000_000)
            per_axis_batch_observability = _adapter_batch_observability(adapter, sample)
            per_axis = adapter.score_aware_components(adapter.model, sample)
        except (NotImplementedError, AttributeError):
            per_axis = None
            per_axis_batch_observability = None
        except Exception as exc:
            # Per-axis decomposition is observability-only; never fail the run.
            print(f"[long_training_canonical] WARN: score_aware_components failed at epoch {epoch}: {exc!r}")
            per_axis = None
            per_axis_batch_observability = None

        # 6) Record canonical metrics.
        wall_clock = time.time() - t_start
        try:
            drift = ema_shadow.drift_l2(adapter.model)
        except Exception:
            drift = 0.0
        # Build loss_components dict from loss_dict (excluding "total" key).
        loss_components = {k: float(v) for k, v in loss_dict.items() if k != "total"}
        total_loss = float(loss_dict["total"])
        selection_metric, selection_metric_blocker = _resolve_checkpoint_selection_metric(
            loss_dict=loss_dict,
            total_loss=total_loss,
            metric_key=config.checkpoint_selection_metric_key,
            strict=bool(config.checkpoint_selection_metric_required),
        )
        tie_break_metric: float | None = None
        tie_break_metric_blocker: str | None = None
        if str(config.checkpoint_selection_tie_break_metric_key).strip():
            tie_break_metric, tie_break_metric_blocker = (
                _resolve_checkpoint_selection_metric(
                    loss_dict=loss_dict,
                    total_loss=total_loss,
                    metric_key=config.checkpoint_selection_tie_break_metric_key,
                    strict=bool(
                        config.checkpoint_selection_tie_break_metric_required
                    ),
                )
            )
        last_selection_metric = float(selection_metric)
        if selection_metric_blocker is not None:
            checkpoint_selection_metric_blockers.append(selection_metric_blocker)
        if tie_break_metric_blocker is not None:
            checkpoint_selection_metric_blockers.append(
                f"checkpoint_selection_tie_break_{tie_break_metric_blocker}"
            )
        # RICH GATING telemetry per CLAUDE.md "Max observability" + the B1
        # clean-relaunch BLOCKER 1 directive. Observability-ONLY: querying the
        # adapter's ``rich_gating_telemetry`` does not mutate loss/grad/optimizer
        # state, so sister substrates without the accessor stay byte-identical
        # (gating stays None -> the new fields are absent from their rows).
        gating_row = _assemble_gating_row(
            adapter=adapter,
            epoch=epoch,
            total_loss=total_loss,
            per_axis=per_axis,
            last_checkpoint_path=last_checkpoint_meta_path,
            checkpoint_save_scheduled=bool(
                (epoch + 1) % config.checkpoint_interval_epochs == 0
            ),
        )
        metrics = PerEpochMetrics(
            epoch=epoch,
            stage_name=stage.name,
            loss=total_loss,
            loss_components=loss_components,
            per_axis_decomposition=per_axis,
            gating=gating_row,
            batch_observability={
                "schema": "long_training_batch_observability.v1",
                "train_batch": train_batch_observability,
                "per_axis_batch": per_axis_batch_observability,
                "post_optimizer_projection": (
                    _jsonable_mapping(projection_report)
                    if projection_report is not None
                    else None
                ),
                "actual_train_batch_size": int(actual_bs),
                "requested_batch_size": int(config.batch_pair_indices_per_step),
                "coverage_scope": "sampled_pair_indices_not_full_video_replay",
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
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
            except LongTrainingStopRequested as exc:
                early_stopped = True
                early_stop_reason = exc.reason
                break
            except Exception as exc:
                print(f"[long_training_canonical] WARN: on_epoch_end callback failed: {exc!r}")

        # 7) Early-stopping bookkeeping.
        if _checkpoint_selection_improved(
            current=float(selection_metric),
            best=float(best_metric),
            mode=config.checkpoint_selection_metric_mode,
            current_tie_break=tie_break_metric,
            best_tie_break=best_tie_break_metric,
            tie_break_mode=config.checkpoint_selection_tie_break_metric_mode,
        ):
            best_metric = float(selection_metric)
            best_tie_break_metric = (
                float(tie_break_metric) if tie_break_metric is not None else None
            )
            best_loss = total_loss
            best_epoch = int(epoch)
            best_wall_clock = float(wall_clock)
            epochs_since_improvement = 0
            if config.best_checkpoint_for_archive_export:
                try:
                    best_live_state = PolyakEMAShadow._clone_state_dict(
                        ema_shadow._get_flat_state(adapter.model)
                    )
                    best_ema_state = ema_shadow.state_dict()
                    best_ema_compensation_state = (
                        ema_shadow.compensation_state_dict()
                    )
                    best_state_capture_error = None
                except Exception as exc:
                    best_state_capture_error = f"{type(exc).__name__}:{exc!s}"
                    print(
                        "[long_training_canonical] WARN: best checkpoint state "
                        f"capture failed at epoch {epoch}: {exc!r}"
                    )
        else:
            epochs_since_improvement += 1
        if epochs_since_improvement >= config.early_stopping_patience:
            early_stopped = True
            early_stop_reason = (
                f"early_stopping_patience_exceeded:{config.early_stopping_patience}"
                "_epochs_without_improvement_for_checkpoint_metric_"
                f"{config.checkpoint_selection_metric_key}_{best_metric}"
            )
            break

        # 8) Periodic checkpoint emission.
        if (epoch + 1) % config.checkpoint_interval_epochs == 0:
            try:
                periodic_meta_path = checkpoint_writer.write(
                    adapter=adapter,
                    ema_shadow=ema_shadow,
                    global_epoch=epoch,
                    loss=total_loss,
                    wall_clock_seconds=wall_clock,
                    is_final=False,
                    selection_metric_key=config.checkpoint_selection_metric_key,
                    selection_metric_value=float(selection_metric),
                    selection_metric_mode=config.checkpoint_selection_metric_mode,
                )
                apply_checkpoint_retention(
                    config,
                    trigger_meta_path=periodic_meta_path,
                )
                # Surface this durable checkpoint path into the NEXT epoch's
                # gating telemetry ``checkpoint_path`` (honest most-recent
                # record; the write succeeded above).
                last_checkpoint_meta_path = periodic_meta_path
            except Exception as exc:
                if config.ema_accumulation == "kahan":
                    raise
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
            selection_metric_key=config.checkpoint_selection_metric_key,
            selection_metric_value=last_selection_metric,
            selection_metric_mode=config.checkpoint_selection_metric_mode,
        )
    except Exception as exc:
        if config.ema_accumulation == "kahan":
            raise
        traceback.print_exc()
        print(f"[long_training_canonical] WARN: final checkpoint emission failed: {exc!r}")
    try:
        apply_checkpoint_retention(config, trigger_meta_path=final_meta_path)
    except Exception as exc:
        if config.ema_accumulation == "kahan":
            raise
        traceback.print_exc()
        print(f"[long_training_canonical] WARN: checkpoint retention failed: {exc!r}")

    # 9b) Emit one durable best checkpoint when final drifted away from the
    # observed best loss. This captures non-monotone valleys without every-epoch
    # checkpoint bloat.
    selected_meta_path = final_meta_path
    selected_checkpoint_role = "final"
    best_meta_path: Path | None = None
    best_checkpoint_emission_error: str | None = None
    if (
        config.best_checkpoint_for_archive_export
        and best_live_state is not None
        and best_ema_state is not None
    ):
        if int(best_epoch) == int(final_epoch):
            best_meta_path = final_meta_path
        else:
            final_live_state = PolyakEMAShadow._clone_state_dict(
                ema_shadow._get_flat_state(adapter.model)
            )
            final_ema_state = ema_shadow.state_dict()
            final_ema_compensation_state = ema_shadow.compensation_state_dict()
            try:
                ema_shadow.restore_from_snapshot(adapter.model, best_live_state)
                ema_shadow.load_state_dict(best_ema_state)
                if ema_shadow.enable_kahan:
                    ema_shadow.load_compensation_state_dict(
                        best_ema_compensation_state or {}
                    )
                best_meta_path = checkpoint_writer.write(
                    adapter=adapter,
                    ema_shadow=ema_shadow,
                    global_epoch=int(best_epoch),
                    loss=float(best_loss),
                    wall_clock_seconds=float(best_wall_clock),
                    is_final=False,
                    checkpoint_role="best",
                    selection_metric_key=config.checkpoint_selection_metric_key,
                    selection_metric_value=float(best_metric),
                    selection_metric_mode=config.checkpoint_selection_metric_mode,
                )
                selected_meta_path = best_meta_path
                selected_checkpoint_role = "best"
            except Exception as exc:
                if config.ema_accumulation == "kahan":
                    raise
                best_checkpoint_emission_error = f"{type(exc).__name__}:{exc!s}"
                traceback.print_exc()
                print(
                    "[long_training_canonical] WARN: best checkpoint emission "
                    f"failed: {exc!r}"
                )
            finally:
                ema_shadow.restore_from_snapshot(adapter.model, final_live_state)
                ema_shadow.load_state_dict(final_ema_state)
                if ema_shadow.enable_kahan:
                    ema_shadow.load_compensation_state_dict(
                        final_ema_compensation_state
                    )
            if best_meta_path is not None:
                try:
                    apply_checkpoint_retention(config, trigger_meta_path=best_meta_path)
                except Exception as exc:
                    if config.ema_accumulation == "kahan":
                        raise
                    traceback.print_exc()
                    print(
                        "[long_training_canonical] WARN: checkpoint retention "
                        f"after best emission failed: {exc!r}"
                    )

    # Resolve selected and final checkpoint paths from metadata. The top-level
    # TrainingArtifact checkpoint paths intentionally point at the selected
    # archive-export state; final paths remain in checkpoint_selection.
    try:
        final_meta = json.loads(final_meta_path.read_text())
        final_ema_shadow_checkpoint_path = Path(final_meta["ema_shadow_state_path"])
        final_live_checkpoint_path = Path(final_meta["live_state_path"])
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        final_meta = {}
        final_ema_shadow_checkpoint_path = (
            config.resolved_checkpoint_dir() / "ema_shadow.unknown"
        )
        final_live_checkpoint_path = None
    try:
        selected_meta = json.loads(selected_meta_path.read_text())
        ema_shadow_checkpoint_path = Path(selected_meta["ema_shadow_state_path"])
        live_checkpoint_path = Path(selected_meta["live_state_path"])
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        selected_meta = {}
        ema_shadow_checkpoint_path = final_ema_shadow_checkpoint_path
        live_checkpoint_path = final_live_checkpoint_path
        selected_meta_path = final_meta_path
        selected_checkpoint_role = "final"

    selected_checkpoint_restore_error: str | None = None
    if selected_checkpoint_role == "best" and best_live_state is not None:
        try:
            ema_shadow.restore_from_snapshot(adapter.model, best_live_state)
            if best_ema_state is not None:
                ema_shadow.load_state_dict(best_ema_state)
            if ema_shadow.enable_kahan:
                ema_shadow.load_compensation_state_dict(
                    best_ema_compensation_state or {}
                )
        except Exception as exc:
            selected_checkpoint_restore_error = f"{type(exc).__name__}:{exc!s}"
            traceback.print_exc()
            print(
                "[long_training_canonical] WARN: selected best checkpoint restore "
                f"failed before archive export: {exc!r}"
            )
            if "final_live_state" in locals() and "final_ema_state" in locals():
                try:
                    ema_shadow.restore_from_snapshot(adapter.model, final_live_state)
                    ema_shadow.load_state_dict(final_ema_state)
                    if ema_shadow.enable_kahan:
                        ema_shadow.load_compensation_state_dict(
                            final_ema_compensation_state
                        )
                except Exception as restore_exc:
                    print(
                        "[long_training_canonical] WARN: fallback final checkpoint "
                        f"restore also failed: {restore_exc!r}"
                    )
            selected_meta_path = final_meta_path
            selected_checkpoint_role = "final"
            selected_meta = final_meta
            ema_shadow_checkpoint_path = final_ema_shadow_checkpoint_path
            live_checkpoint_path = final_live_checkpoint_path

    checkpoint_selection: dict[str, Any] = {
        "schema": "long_training_checkpoint_selection.v1",
        "policy": (
            f"best_{config.checkpoint_selection_metric_key}_checkpoint_for_archive_export"
            if config.best_checkpoint_for_archive_export
            else "final_checkpoint_for_archive_export"
        ),
        "selection_metric_key": config.checkpoint_selection_metric_key,
        "selection_metric_mode": config.checkpoint_selection_metric_mode,
        "selection_metric_required": bool(
            config.checkpoint_selection_metric_required
        ),
        "tie_break_metric_key": str(
            config.checkpoint_selection_tie_break_metric_key
        ),
        "tie_break_metric_mode": str(
            config.checkpoint_selection_tie_break_metric_mode
        ),
        "tie_break_metric_required": bool(
            config.checkpoint_selection_tie_break_metric_required
        ),
        "selected_role": selected_checkpoint_role,
        "selected_meta_path": selected_meta_path.as_posix(),
        "selected_global_epoch": selected_meta.get("global_epoch"),
        "selected_loss": selected_meta.get("loss"),
        "selected_metric": selected_meta.get("checkpoint_selection_metric_value"),
        "selected_live_state_path": (
            live_checkpoint_path.as_posix() if live_checkpoint_path else None
        ),
        "selected_ema_shadow_state_path": ema_shadow_checkpoint_path.as_posix(),
        "final_meta_path": final_meta_path.as_posix(),
        "final_global_epoch": final_meta.get("global_epoch"),
        "final_loss": final_meta.get("loss"),
        "final_metric": final_meta.get("checkpoint_selection_metric_value"),
        "final_live_state_path": (
            final_live_checkpoint_path.as_posix()
            if final_live_checkpoint_path
            else None
        ),
        "final_ema_shadow_state_path": final_ema_shadow_checkpoint_path.as_posix(),
        "best_observed_epoch": int(best_epoch),
        "best_observed_loss": float(best_loss),
        "best_observed_metric": float(best_metric),
        "best_observed_tie_break_metric": (
            float(best_tie_break_metric)
            if best_tie_break_metric is not None
            else None
        ),
        "best_meta_path": best_meta_path.as_posix() if best_meta_path else None,
        "best_state_capture_error": best_state_capture_error,
        "best_checkpoint_emission_error": best_checkpoint_emission_error,
        "selected_checkpoint_restore_error": selected_checkpoint_restore_error,
        "checkpoint_selection_metric_blockers": _ordered_unique(
            checkpoint_selection_metric_blockers
        ),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }

    # 10) Substrate archive export (optional per adapter contract).
    archive_path: Path | None = None
    archive_sha256: str | None = None
    archive_bytes: int | None = None
    archive_selection_manifest_path: Path | None = None
    try:
        if config.ema_archive_selection_enabled:
            (
                archive_path,
                archive_sha256,
                archive_bytes,
                archive_selection_manifest_path,
            ) = _export_live_ema_archive_selection(
                adapter=adapter,
                config=config,
                ema_shadow=ema_shadow,
            )
        else:
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
    substrate_artifact_metadata = _collect_adapter_artifact_metadata(adapter)

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
        archive_selection_manifest_path=archive_selection_manifest_path,
        checkpoint_selection=checkpoint_selection,
        per_epoch_metrics=tuple(per_epoch_metrics),
        total_wall_clock_seconds=total_wall_clock,
        total_epochs_completed=total_epochs_completed,
        canonical_provenance={},  # filled below
        telemetry_path=config.resolved_telemetry_path(),
        substrate_artifact_metadata=substrate_artifact_metadata,
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
        archive_selection_manifest_path=archive_selection_manifest_path,
        checkpoint_selection=checkpoint_selection,
        per_epoch_metrics=tuple(per_epoch_metrics),
        total_wall_clock_seconds=total_wall_clock,
        total_epochs_completed=total_epochs_completed,
        canonical_provenance=provenance,
        telemetry_path=config.resolved_telemetry_path(),
        substrate_artifact_metadata=substrate_artifact_metadata,
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
        archive_selection_manifest_path=archive_selection_manifest_path,
        checkpoint_selection=checkpoint_selection,
        per_epoch_metrics=tuple(per_epoch_metrics),
        total_wall_clock_seconds=total_wall_clock,
        total_epochs_completed=total_epochs_completed,
        canonical_provenance=provenance,
        telemetry_path=config.resolved_telemetry_path(),
        substrate_artifact_metadata=substrate_artifact_metadata,
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
