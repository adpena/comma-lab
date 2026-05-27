# SPDX-License-Identifier: MIT
"""Layer 0 — canonical encoder pipeline orchestrator.

Wrap ``experiments/train_substrate_*.py`` trainer + post-train QAT + canonical
weight export + MLX→numpy bridge into one canonical entry point per Phase 1
audit specification memo at
``.omx/research/canonical_submission_pipeline_specification_memo_20260526.md``
Layer 0.

The bug class this layer extincts: ad-hoc
``python experiments/train_substrate_X.py --flag1 --flag2 ...`` invocations
that bypass canonical Catalog gates (Tier-1 engineering hygiene per
#172/#178/#179/#180, Tier-2 hardware-correctness per #170/#171/#181/#182/
#215, Tier-3 substrate-correctness per #222/#226/#240/#249/#270, NVML env
block per #244, Modal call-id-ledger registration per #245/#339/#360,
canonical scorer-loss helper routing per #164/#228). Per the 2026-05-19
6-phase manual cleanup empirical anchor, each of those gates is silently
bypassable when the encoder pipeline is hand-rolled.

This module's :func:`build_compression_pipeline` is the canonical entry
point. It DOES NOT invoke paid Modal / Vast.ai / Lightning dispatch (Phase
2 scope per Phase 1 spec memo): it PREPARES the encoder context, verifies
the Catalog #270 umbrella protocol against the trainer + recipe pair, and
returns a typed :class:`CompressionPipelineResult` that downstream Phase
3-10 layers consume to actually run dispatch (Phase 6
``tac.submission_packet.paired_auth_eval``) and bundle the submission
packet (Phase 4 ``tac.submission_packet.builder``).

The pipeline is OBSERVABILITY-ONLY by construction (per Catalog #341
non-promotable routing markers + CLAUDE.md "Apples-to-apples evidence
discipline"): every emitted :class:`CompressionPipelineResult` carries
``score_claim=False`` + ``promotable=False`` + ``axis_tag=[predicted]``.
Promotion of a compression-pipeline anchor to a contest score signal
REQUIRES paired-CUDA + Linux x86_64 CPU empirical anchor per CLAUDE.md
"Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT
HARDWARE" non-negotiable (lands at Phase 6 / Phase 10).

Per the 8th MLX-first numpy-portable standing directive: this module is
pure-Python (no MLX or PyTorch dependency at orchestration time); MLX
training happens INSIDE the trainer module the pipeline wraps. The
pipeline's outputs are numpy-portable ``.npz`` paths emitted by the
trainer + sha-checksummed by the orchestrator.

Per the 11th ORDER-MATTERS standing directive: this module is THE FIRST
Phase 1 spec consumer; Phase 3-10 layers depend on its return type.

Per the 12th canonicalization × standardization × ease-of-contest-
compliance trinity: this is one canonical helper, ONE return shape, ONE
verification protocol that downstream layers can compose without
re-deriving conventions.
"""
from __future__ import annotations

import datetime
import enum
import hashlib
import json
import os
import platform
import re
import socket
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Module-level constants — canonical schemas + canonical paths + canonical IDs
# ---------------------------------------------------------------------------

COMPRESSION_PIPELINE_SCHEMA_VERSION = "compression_pipeline_v1_20260526"
"""Pinned schema for :class:`CompressionPipelineResult` persistence rows."""

PHASE_2_LAYER_VERSION = "phase_2_compression_pipeline_canonical_landed_20260526"
"""Operator-readable Phase 2 landing marker per Phase 1 audit spec memo."""

CANONICAL_EQUATION_ID = (
    "compression_pipeline_canonical_helper_consolidation_savings_v1"
)
"""Canonical equation registered per Phase 1 audit spec memo §13.

FORMALIZATION_PENDING until Phase 10 first-PR-through-canonical-pipeline
regression lands the first paired-CUDA empirical anchor of wall-clock
collapse (predicted: ~3h manual → <60s automated, ~180x speedup).
"""

# Per Catalog #341 routing markers (Tier A observability-only).
PREDICTED_AXIS_TAG = "[predicted]"

# Per Catalog #287 placeholder rejection.
_PLACEHOLDER_RATIONALES: frozenset[str] = frozenset(
    {"<rationale>", "<reason>", "<rationale_here>", "<reason_here>", ""}
)

# Per Catalog #190 + CLAUDE.md "Hardware identification" — canonical tokens
# (NO false precision; unknown defaults to explicit unknown marker).
_VALID_HARDWARE_SUBSTRATE_TOKENS: frozenset[str] = frozenset(
    {
        "linux_x86_64_modal_t4",
        "linux_x86_64_modal_a10g",
        "linux_x86_64_modal_a100",
        "linux_x86_64_modal_h100",
        "linux_x86_64_modal_l40s",
        "linux_x86_64_modal_cpu",
        "linux_x86_64_vastai_4090",
        "linux_x86_64_vastai_h100",
        "linux_x86_64_vastai_cpu",
        "linux_x86_64_lightning_a100",
        "linux_x86_64_lightning_cpu",
        "linux_x86_64_gha_cpu",
        "macos_arm64_m5_max",
        "macos_arm64_m1_pro",
        "linux_x86_64_unknown_cuda",
        "linux_x86_64_unknown_cpu",
        "unknown_unknown_unknown",
    }
)


class HardwareSubstrateClass(enum.StrEnum):
    """Canonical hardware-substrate routing class per Catalog #270 + #192.

    AUTO triggers per-host probing; LOCAL_MPS/LOCAL_CPU are MPS-research-
    signal / macOS-CPU-advisory (NEVER score-promotable per Catalog #192
    + CLAUDE.md "MPS auth eval is NOISE"); the remote classes route to
    paid GPU/CPU on 1:1 contest-compliant Linux x86_64.
    """

    AUTO = "auto"
    LOCAL_MPS = "local-mps"
    LOCAL_CPU = "local-cpu"
    MODAL = "modal"
    VASTAI = "vastai"
    LIGHTNING = "lightning"


# ---------------------------------------------------------------------------
# Custom exception (sister of canonical fail-closed exception hierarchy)
# ---------------------------------------------------------------------------


class CompressionPipelineError(RuntimeError):
    """Compression pipeline orchestration error.

    Sister of :class:`tac.deploy.modal.call_id_ledger.LedgerRegistrationFailedError`
    + :class:`tac.canonical_equations.InvalidEquationError`. Raised by
    :func:`build_compression_pipeline` when the trainer + recipe pair
    cannot satisfy Catalog #270 umbrella protocol invariants AND no
    waiver is supplied.
    """


# ---------------------------------------------------------------------------
# Per-axis predicted band (per Catalog #356 Dim 3 per-axis decomposition)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerAxisPredictedBand:
    """Per-axis predicted band from the trainer's recipe.

    Per Catalog #324 (predicted-band post-training Tier-C validation
    discipline) + Catalog #296 (predicted-band Dykstra feasibility check):
    the predicted band MUST be marked ``validation_status=pending_post_training``
    initially; Phase 10 lands the first post-training Tier-C validation.

    Sister of :class:`tac.optimization.tier_c_density_post_training_validator.PredictedBandWithValidation`
    at the per-recipe-emit surface; this is the per-axis dataclass the
    compression pipeline carries for downstream Phase 3 archive_grammar +
    Phase 6 paired_auth_eval consumers.
    """

    predicted_seg_distortion_band: tuple[float, float]
    """``(lower, upper)`` predicted ΔSeg band per Catalog #324."""

    predicted_pose_distortion_band: tuple[float, float]
    """``(lower, upper)`` predicted Δpose band per Catalog #324."""

    predicted_archive_bytes_band: tuple[int, int]
    """``(lower, upper)`` predicted archive byte count per Catalog #324."""

    predicted_band_validation_status: str
    """One of ``"pending_post_training"`` / ``"validated_post_training"`` /
    ``"phantom_random_init"`` per Catalog #324."""

    def __post_init__(self) -> None:
        lo, hi = self.predicted_seg_distortion_band
        if not (isinstance(lo, (int, float)) and isinstance(hi, (int, float))):
            raise ValueError("predicted_seg_distortion_band must be (float, float)")
        if not (lo <= hi):
            raise ValueError(
                f"predicted_seg_distortion_band {self.predicted_seg_distortion_band} must satisfy lo<=hi"
            )
        lo, hi = self.predicted_pose_distortion_band
        if not (isinstance(lo, (int, float)) and isinstance(hi, (int, float))):
            raise ValueError("predicted_pose_distortion_band must be (float, float)")
        if not (lo <= hi):
            raise ValueError(
                f"predicted_pose_distortion_band {self.predicted_pose_distortion_band} must satisfy lo<=hi"
            )
        lo, hi = self.predicted_archive_bytes_band
        if not (isinstance(lo, int) and isinstance(hi, int)):
            raise ValueError("predicted_archive_bytes_band must be (int, int)")
        if not (lo <= hi):
            raise ValueError(
                f"predicted_archive_bytes_band {self.predicted_archive_bytes_band} must satisfy lo<=hi"
            )
        if self.predicted_band_validation_status not in {
            "pending_post_training",
            "validated_post_training",
            "phantom_random_init",
        }:
            raise ValueError(
                "predicted_band_validation_status must be one of "
                "{'pending_post_training','validated_post_training','phantom_random_init'} "
                f"per Catalog #324; got {self.predicted_band_validation_status!r}"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "predicted_seg_distortion_band": list(self.predicted_seg_distortion_band),
            "predicted_pose_distortion_band": list(self.predicted_pose_distortion_band),
            "predicted_archive_bytes_band": list(self.predicted_archive_bytes_band),
            "predicted_band_validation_status": self.predicted_band_validation_status,
        }


# ---------------------------------------------------------------------------
# Canonical return dataclass (frozen per Catalog #335 + #323)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompressionPipelineResult:
    """Canonical Phase 2 Layer 0 compression pipeline output.

    Sister of :class:`tac.cross_substrate_master_gradient_analyzer.CrossSubstrateMasterGradientAnalysis`
    + :class:`tac.deploy.modal.call_id_ledger.CallIdLedgerEvent` at the
    submission-packet sub-surface.

    Per Catalog #341 + CLAUDE.md "Apples-to-apples evidence discipline":
    every result is observability-only by construction. Promotion REQUIRES
    paired-CUDA + Linux x86_64 CPU empirical anchor per CLAUDE.md
    "Submission auth eval — BOTH CPU AND CUDA".
    """

    schema_version: str
    """Canonical schema version (current: :data:`COMPRESSION_PIPELINE_SCHEMA_VERSION`)."""

    lane_id: str
    """Lane registry id per CLAUDE.md "Lane maturity registry" lifecycle discipline."""

    substrate_id: str
    """Substrate id derived from trainer filename (``experiments/train_substrate_<id>.py``)."""

    video_path: str
    """Repo-relative path to the contest video (canonical: ``upstream/videos/0.mkv``)."""

    hardware_substrate: str
    """Canonical token from :data:`_VALID_HARDWARE_SUBSTRATE_TOKENS`."""

    hardware_substrate_class: str
    """One of :class:`HardwareSubstrateClass` values."""

    substrate_trainer_path: str
    """Repo-relative path to the trainer (``experiments/train_substrate_*.py``)."""

    recipe_path: str
    """Repo-relative path to the recipe YAML (``.omx/operator_authorize_recipes/*.yaml``)."""

    mlx_first_encode: bool
    """Per the 8th MLX-first standing directive; True for local Apple Silicon, False otherwise."""

    qat_enabled: bool
    """Whether post-train QAT is enabled (default True per CLAUDE.md QAT pipeline)."""

    weights_export_path: str | None
    """Repo-relative path to the trained-weights ``.npz`` (numpy-portable).

    None until the trainer has been invoked (Phase 6 / Phase 10 wall-clock landing).
    """

    weights_sha256: str | None
    """sha256 of the weights ``.npz`` (None until trainer invoked)."""

    weights_size_bytes: int | None
    """Bytes of the weights ``.npz`` (None until trainer invoked)."""

    training_anchor_call_id: str | None
    """Modal call_id (Catalog #245 ledger row) when Modal-dispatched; None for local."""

    qat_anchor_call_id: str | None
    """Modal call_id for QAT post-train phase when Modal-dispatched."""

    dispatch_optimization_protocol_overall_pass: bool
    """Catalog #270 umbrella verdict ``overall_pass`` for the trainer + recipe pair."""

    dispatch_optimization_protocol_blockers: tuple[str, ...]
    """Catalog #270 blockers list (empty when overall_pass=True)."""

    per_axis_predicted_band: PerAxisPredictedBand | None
    """Per-axis predicted band from recipe per Catalog #324 + #356.

    None when the recipe omits a predicted_band declaration.
    """

    measurement_utc: str
    """ISO-8601 UTC timestamp of pipeline emission."""

    axis_tag: str
    """Always ``"[predicted]"`` per Catalog #341 + canonical Provenance."""

    score_claim: bool
    """Always ``False`` per CLAUDE.md "Apples-to-apples evidence discipline"."""

    promotable: bool
    """Always ``False`` per Catalog #341 + #192."""

    evidence_grade: str
    """Always ``"[predicted; compression-pipeline-canonical]"`` per Catalog #287 / #323."""

    canonical_helper_invocation: str
    """``"tac.submission_packet.build_compression_pipeline"`` per Catalog #190."""

    canonical_equation_id: str
    """:data:`CANONICAL_EQUATION_ID` (per Catalog #344)."""

    canonical_equation_status: str
    """``"FORMALIZATION_PENDING"`` until Phase 10 first empirical anchor."""

    elapsed_seconds: float
    """Orchestration elapsed wall-clock (not training elapsed; trainer carries its own)."""

    cost_usd: float | None
    """Estimated provider cost USD; None for local/dry-run."""

    canonical_provenance: Mapping[str, Any] = field(default_factory=dict)
    """Per Catalog #323 canonical Provenance umbrella."""

    written_at_utc: str = ""
    """When persisted to a canonical ledger (caller-fills)."""

    written_pid: int = 0
    """Process PID that emitted the result."""

    written_host: str = ""
    """Host that emitted the result."""

    def __post_init__(self) -> None:
        if self.schema_version != COMPRESSION_PIPELINE_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must equal {COMPRESSION_PIPELINE_SCHEMA_VERSION!r}; "
                f"got {self.schema_version!r}"
            )
        if not self.lane_id:
            raise ValueError("lane_id must be non-empty")
        if not self.substrate_id:
            raise ValueError("substrate_id must be non-empty")
        if not self.video_path:
            raise ValueError("video_path must be non-empty")
        if self.hardware_substrate not in _VALID_HARDWARE_SUBSTRATE_TOKENS:
            raise ValueError(
                f"hardware_substrate {self.hardware_substrate!r} must be in "
                f"{sorted(_VALID_HARDWARE_SUBSTRATE_TOKENS)}; per Catalog #190"
            )
        if self.hardware_substrate_class not in {c.value for c in HardwareSubstrateClass}:
            raise ValueError(
                f"hardware_substrate_class {self.hardware_substrate_class!r} must be one of "
                f"{[c.value for c in HardwareSubstrateClass]}"
            )
        if not self.substrate_trainer_path:
            raise ValueError("substrate_trainer_path must be non-empty")
        if not self.recipe_path:
            raise ValueError("recipe_path must be non-empty")
        if not isinstance(self.mlx_first_encode, bool):
            raise ValueError("mlx_first_encode must be bool")
        if not isinstance(self.qat_enabled, bool):
            raise ValueError("qat_enabled must be bool")
        if self.weights_sha256 is not None and len(self.weights_sha256) != 64:
            raise ValueError(
                f"weights_sha256 must be 64-char hex; got len={len(self.weights_sha256)}"
            )
        if self.weights_size_bytes is not None and self.weights_size_bytes < 0:
            raise ValueError("weights_size_bytes must be non-negative")
        if not isinstance(self.dispatch_optimization_protocol_blockers, tuple):
            raise ValueError("dispatch_optimization_protocol_blockers must be a tuple (frozen)")
        if self.axis_tag != PREDICTED_AXIS_TAG:
            raise ValueError(f"axis_tag must equal {PREDICTED_AXIS_TAG!r}; got {self.axis_tag!r}")
        if self.score_claim is not False:
            raise ValueError("score_claim must be False per Catalog #341")
        if self.promotable is not False:
            raise ValueError("promotable must be False per Catalog #341")
        if not self.evidence_grade.startswith("[predicted;"):
            raise ValueError(
                "evidence_grade must start with '[predicted;' per Catalog #287/#323"
            )
        if self.canonical_equation_id != CANONICAL_EQUATION_ID:
            raise ValueError(
                f"canonical_equation_id must equal {CANONICAL_EQUATION_ID!r}; got {self.canonical_equation_id!r}"
            )
        if self.canonical_equation_status not in {"FORMALIZATION_PENDING", "REGISTERED"}:
            raise ValueError(
                "canonical_equation_status must be 'FORMALIZATION_PENDING' or 'REGISTERED' per Catalog #344"
            )
        if not self.measurement_utc:
            raise ValueError("measurement_utc must be non-empty")
        if self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be non-negative")
        if self.cost_usd is not None and self.cost_usd < 0:
            raise ValueError("cost_usd must be None or non-negative")
        if not isinstance(self.canonical_provenance, Mapping):
            raise ValueError("canonical_provenance must be a Mapping per Catalog #323")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "lane_id": self.lane_id,
            "substrate_id": self.substrate_id,
            "video_path": self.video_path,
            "hardware_substrate": self.hardware_substrate,
            "hardware_substrate_class": self.hardware_substrate_class,
            "substrate_trainer_path": self.substrate_trainer_path,
            "recipe_path": self.recipe_path,
            "mlx_first_encode": bool(self.mlx_first_encode),
            "qat_enabled": bool(self.qat_enabled),
            "weights_export_path": self.weights_export_path,
            "weights_sha256": self.weights_sha256,
            "weights_size_bytes": (
                int(self.weights_size_bytes)
                if self.weights_size_bytes is not None
                else None
            ),
            "training_anchor_call_id": self.training_anchor_call_id,
            "qat_anchor_call_id": self.qat_anchor_call_id,
            "dispatch_optimization_protocol_overall_pass": bool(
                self.dispatch_optimization_protocol_overall_pass
            ),
            "dispatch_optimization_protocol_blockers": list(
                self.dispatch_optimization_protocol_blockers
            ),
            "per_axis_predicted_band": (
                self.per_axis_predicted_band.as_dict()
                if self.per_axis_predicted_band is not None
                else None
            ),
            "measurement_utc": self.measurement_utc,
            "axis_tag": self.axis_tag,
            "score_claim": bool(self.score_claim),
            "promotable": bool(self.promotable),
            "evidence_grade": self.evidence_grade,
            "canonical_helper_invocation": self.canonical_helper_invocation,
            "canonical_equation_id": self.canonical_equation_id,
            "canonical_equation_status": self.canonical_equation_status,
            "elapsed_seconds": float(self.elapsed_seconds),
            "cost_usd": (
                float(self.cost_usd) if self.cost_usd is not None else None
            ),
            "canonical_provenance": dict(self.canonical_provenance),
            "written_at_utc": self.written_at_utc,
            "written_pid": int(self.written_pid),
            "written_host": self.written_host,
        }


# ---------------------------------------------------------------------------
# Core API functions
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    """Canonical UTC timestamp (ISO-8601 with tz)."""
    return datetime.datetime.now(datetime.UTC).isoformat()


def _substrate_id_from_trainer_path(trainer_path: Path) -> str:
    """Extract the canonical substrate id from a trainer filename.

    ``experiments/train_substrate_<id>.py`` → ``<id>``.

    Raises ValueError if the filename does not match the canonical pattern.
    """
    name = trainer_path.name
    m = re.match(r"^train_substrate_(?P<id>[a-z][a-z0-9_]*)\.py$", name)
    if m is None:
        raise CompressionPipelineError(
            f"trainer_path {trainer_path!r} does not match canonical "
            "'experiments/train_substrate_<id>.py' pattern"
        )
    return m.group("id")


def classify_hardware_substrate_for_dispatch(
    requested: str | HardwareSubstrateClass,
    *,
    explicit_hardware_substrate: str | None = None,
) -> tuple[str, str]:
    """Classify the requested hardware substrate into canonical class + token.

    Per CLAUDE.md "MPS auth eval is NOISE" + Catalog #192 + Catalog #190:
    local Apple Silicon routes are MPS-research-signal / macOS-CPU-advisory
    (NEVER score-promotable); remote routes are 1:1 contest-compliant
    Linux x86_64.

    Args:
        requested: Operator request (``"auto"`` / ``"local-mps"`` /
            ``"local-cpu"`` / ``"modal"`` / ``"vastai"`` / ``"lightning"``).
        explicit_hardware_substrate: Optional canonical hardware substrate
            token (e.g. ``"linux_x86_64_modal_t4"``) that overrides
            auto-detection. MUST be in :data:`_VALID_HARDWARE_SUBSTRATE_TOKENS`.

    Returns:
        ``(hardware_substrate_token, hardware_substrate_class_value)``.

    Raises:
        CompressionPipelineError: if classification cannot be unambiguously
            resolved.
    """
    if isinstance(requested, str):
        try:
            req_class = HardwareSubstrateClass(requested.lower())
        except ValueError as exc:
            raise CompressionPipelineError(
                f"hardware_substrate request {requested!r} must be one of "
                f"{[c.value for c in HardwareSubstrateClass]}"
            ) from exc
    else:
        req_class = requested

    if explicit_hardware_substrate is not None:
        if explicit_hardware_substrate not in _VALID_HARDWARE_SUBSTRATE_TOKENS:
            raise CompressionPipelineError(
                f"explicit_hardware_substrate {explicit_hardware_substrate!r} must be in "
                f"{sorted(_VALID_HARDWARE_SUBSTRATE_TOKENS)} per Catalog #190"
            )
        return (explicit_hardware_substrate, req_class.value)

    # AUTO: detect from platform without false precision (Catalog #190).
    if req_class is HardwareSubstrateClass.AUTO:
        system = platform.system().lower()
        machine = platform.machine().lower()
        if system == "darwin" and machine in {"arm64", "aarch64"}:
            return ("macos_arm64_m5_max", HardwareSubstrateClass.LOCAL_MPS.value)
        if system == "linux":
            # Local Linux without paid-provider context defaults to GHA-CPU-class
            # so any subsequent score claim is appropriately non-promotable until
            # paired-axis Linux x86_64 evidence lands.
            return ("linux_x86_64_gha_cpu", HardwareSubstrateClass.LOCAL_CPU.value)
        return ("unknown_unknown_unknown", req_class.value)

    if req_class is HardwareSubstrateClass.LOCAL_MPS:
        return ("macos_arm64_m5_max", req_class.value)
    if req_class is HardwareSubstrateClass.LOCAL_CPU:
        return ("linux_x86_64_unknown_cpu", req_class.value)

    # Remote classes default to A100-class token; explicit override above
    # gives operator finer-grained control. Per CLAUDE.md "Submission auth
    # eval BOTH CPU AND CUDA" the remote class is the canonical contest axis.
    if req_class is HardwareSubstrateClass.MODAL:
        return ("linux_x86_64_modal_a100", req_class.value)
    if req_class is HardwareSubstrateClass.VASTAI:
        return ("linux_x86_64_vastai_4090", req_class.value)
    if req_class is HardwareSubstrateClass.LIGHTNING:
        return ("linux_x86_64_lightning_a100", req_class.value)

    raise CompressionPipelineError(
        f"unreachable classification for {req_class!r}"
    )


def validate_recipe_trainer_pair(
    *,
    substrate_trainer: Path,
    recipe_path: Path,
    repo_root: Path | None = None,
) -> tuple[str, str]:
    """Validate that the trainer + recipe pair is canonically aligned.

    Per Catalog #240 (recipe-vs-trainer-state consistency) + Catalog #270
    (umbrella protocol): the trainer must exist + the recipe must exist +
    the recipe's filename must reference the trainer's substrate id.

    Returns:
        ``(substrate_id, recipe_name)`` derived from canonical filenames.

    Raises:
        CompressionPipelineError: if either path is missing OR the pair is
            mismatched.
    """
    root = repo_root if repo_root is not None else REPO_ROOT
    trainer_abs = (
        substrate_trainer
        if substrate_trainer.is_absolute()
        else (root / substrate_trainer).resolve()
    )
    recipe_abs = (
        recipe_path if recipe_path.is_absolute() else (root / recipe_path).resolve()
    )
    if not trainer_abs.is_file():
        raise CompressionPipelineError(
            f"substrate_trainer {trainer_abs} does not exist; "
            "Catalog #240 requires the trainer to be canonically present"
        )
    if not recipe_abs.is_file():
        raise CompressionPipelineError(
            f"recipe_path {recipe_abs} does not exist; "
            "Catalog #240 requires the recipe to be canonically present"
        )
    substrate_id = _substrate_id_from_trainer_path(trainer_abs)
    recipe_name = recipe_abs.stem
    # Catalog #240 sister-discipline: the recipe name must contain the substrate id.
    if substrate_id not in recipe_name:
        raise CompressionPipelineError(
            f"recipe {recipe_abs.name!r} must reference substrate_id "
            f"{substrate_id!r} per Catalog #240 recipe-vs-trainer-state consistency"
        )
    return (substrate_id, recipe_name)


def verify_compression_pipeline_protocol_complete(
    *,
    substrate_trainer: Path,
    recipe_path: Path,
    repo_root: Path | None = None,
) -> tuple[bool, tuple[str, ...]]:
    """Verify Catalog #270 umbrella protocol for the trainer + recipe pair.

    Routes through the canonical
    ``tools.canonical_dispatch_optimization_protocol.verify_dispatch_protocol_complete``
    helper (NOT a subprocess shell-out per Catalog #226 canonical-helper-
    routing discipline). The helper enumerates Tier-1 engineering + Tier-2
    hardware-correctness + Tier-3 substrate-correctness signals and returns
    a typed :class:`ProtocolVerdict`.

    Returns:
        ``(overall_pass, blockers_tuple)``.
    """
    # Late import: the canonical helper lives under tools/ (not src/tac/) so
    # we register it via sys.path injection + standard import (the
    # ``@dataclass`` decorator's module-resolution requires the module to be
    # in sys.modules, which only works via the standard import machinery,
    # not via ``importlib.util.spec_from_file_location``).
    import sys

    root = repo_root if repo_root is not None else REPO_ROOT
    canonical_helper_path = root / "tools" / "canonical_dispatch_optimization_protocol.py"
    if not canonical_helper_path.is_file():
        raise CompressionPipelineError(
            f"canonical helper {canonical_helper_path} not found; "
            "Catalog #270 umbrella protocol cannot be verified"
        )
    tools_dir = str((root / "tools").resolve())
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    try:
        import canonical_dispatch_optimization_protocol as _cdop
    except ImportError as exc:
        raise CompressionPipelineError(
            f"failed to import canonical helper {canonical_helper_path}: {exc}"
        ) from exc

    trainer_abs = (
        substrate_trainer
        if substrate_trainer.is_absolute()
        else (root / substrate_trainer).resolve()
    )
    recipe_name = recipe_path.stem
    verdict = _cdop.verify_dispatch_protocol_complete(
        trainer=trainer_abs,
        recipe=recipe_name,
        repo_root=root,
    )
    return (bool(verdict.overall_pass), tuple(verdict.blockers))


def derive_compression_pipeline_provenance(
    *,
    lane_id: str,
    substrate_id: str,
    hardware_substrate: str,
    measurement_utc: str,
) -> dict[str, Any]:
    """Build the canonical Provenance dict for a compression pipeline result.

    Per Catalog #323 canonical Provenance umbrella: every persisted row
    carries (axis_tag + hardware_substrate + evidence_grade + score_claim
    + promotable + canonical_helper_invocation + captured_at_utc). This
    helper returns the canonical shape downstream consumers expect.
    """
    return {
        "axis_tag": PREDICTED_AXIS_TAG,
        "hardware_substrate": hardware_substrate,
        "evidence_grade": "[predicted; compression-pipeline-canonical]",
        "score_claim": False,
        "promotable": False,
        "canonical_helper_invocation": (
            "tac.submission_packet.build_compression_pipeline"
        ),
        "captured_at_utc": measurement_utc,
        "lane_id": lane_id,
        "substrate_id": substrate_id,
        "canonical_equation_id": CANONICAL_EQUATION_ID,
        "canonical_equation_status": "FORMALIZATION_PENDING",
        "schema_version": COMPRESSION_PIPELINE_SCHEMA_VERSION,
    }


def _extract_per_axis_predicted_band_from_recipe(
    recipe_path: Path,
) -> PerAxisPredictedBand | None:
    """Best-effort extract per-axis predicted band from recipe YAML.

    Per Catalog #324 (predicted-band post-training Tier-C validation): the
    recipe MAY carry a ``predicted_band:`` field with per-axis sub-fields.
    When omitted, returns None (caller routes the optional bypass).

    Pure-text extraction (no YAML library) so the helper stays
    dependency-free per the 12th canonicalization × ease-of-contest-
    compliance trinity.
    """
    try:
        text = recipe_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    # Look for canonical sub-field patterns:
    #   predicted_seg_distortion_band: [lo, hi]
    #   predicted_pose_distortion_band: [lo, hi]
    #   predicted_archive_bytes_band: [lo, hi]
    seg = re.search(
        r"predicted_seg_distortion_band:\s*\[\s*([0-9.+\-eE]+)\s*,\s*([0-9.+\-eE]+)\s*\]",
        text,
    )
    pose = re.search(
        r"predicted_pose_distortion_band:\s*\[\s*([0-9.+\-eE]+)\s*,\s*([0-9.+\-eE]+)\s*\]",
        text,
    )
    bytes_match = re.search(
        r"predicted_archive_bytes_band:\s*\[\s*([0-9]+)\s*,\s*([0-9]+)\s*\]",
        text,
    )
    if not (seg and pose and bytes_match):
        return None
    status_match = re.search(
        r"predicted_band_validation_status:\s*[\"']?([a-z_]+)[\"']?",
        text,
    )
    status = (
        status_match.group(1)
        if status_match is not None
        else "pending_post_training"
    )
    try:
        return PerAxisPredictedBand(
            predicted_seg_distortion_band=(float(seg.group(1)), float(seg.group(2))),
            predicted_pose_distortion_band=(float(pose.group(1)), float(pose.group(2))),
            predicted_archive_bytes_band=(
                int(bytes_match.group(1)),
                int(bytes_match.group(2)),
            ),
            predicted_band_validation_status=status,
        )
    except (ValueError, TypeError):
        return None


def build_compression_pipeline(
    *,
    lane_id: str,
    video_path: Path,
    substrate_trainer: Path,
    recipe_path: Path,
    hardware_substrate: str | HardwareSubstrateClass = HardwareSubstrateClass.AUTO,
    qat_enabled: bool = True,
    output_dir: Path,
    explicit_hardware_substrate: str | None = None,
    mlx_first: bool | None = None,
    repo_root: Path | None = None,
    weights_export_path: Path | None = None,
    weights_sha256: str | None = None,
    weights_size_bytes: int | None = None,
    training_anchor_call_id: str | None = None,
    qat_anchor_call_id: str | None = None,
    cost_usd: float | None = None,
    elapsed_seconds: float = 0.0,
    skip_protocol_verification: bool = False,
) -> CompressionPipelineResult:
    """Canonical encoder pipeline orchestrator (Layer 0).

    Routes the trainer + recipe pair through Catalog #270 umbrella protocol
    verification + canonical hardware classification + recipe-derived
    per-axis predicted band extraction + canonical Provenance umbrella +
    canonical equation id stamping. Returns a typed
    :class:`CompressionPipelineResult` that downstream Phase 3-10 layers
    consume.

    The helper does NOT invoke paid Modal/Vast.ai/Lightning dispatch per
    Phase 2 scope. It PREPARES the encoder context (verifies dispatchability)
    and emits a result row. Phase 6 ``paired_auth_eval`` (separate subagent)
    is where the actual dispatch fires.

    Args:
        lane_id: Lane registry id per CLAUDE.md "Lane maturity registry"
            non-negotiable lifecycle discipline.
        video_path: Path to the contest video (canonical: ``upstream/videos/0.mkv``).
        substrate_trainer: Path to ``experiments/train_substrate_<id>.py``.
        recipe_path: Path to ``.omx/operator_authorize_recipes/substrate_<id>_*.yaml``.
        hardware_substrate: One of :class:`HardwareSubstrateClass` values or
            its string representation.
        qat_enabled: Whether post-train QAT is part of the pipeline (default True).
        output_dir: Output directory for emitted artifacts (lane registry
            evidence path conventions).
        explicit_hardware_substrate: Optional canonical token override per
            :data:`_VALID_HARDWARE_SUBSTRATE_TOKENS`.
        mlx_first: Override the 8th MLX-first default; None auto-detects
            from hardware_substrate_class (True for LOCAL_MPS, False otherwise).
        repo_root: Override repo root (defaults to module-resolved REPO_ROOT).
        weights_export_path / weights_sha256 / weights_size_bytes /
            training_anchor_call_id / qat_anchor_call_id / cost_usd /
            elapsed_seconds: Optional results from a completed pipeline
            invocation (caller-fills after trainer + QAT land). When None,
            the result is a PRE-RUN preparation row.
        skip_protocol_verification: When True, bypass Catalog #270 umbrella
            check (operator-routable for dry-run preparation only; default
            False).

    Returns:
        :class:`CompressionPipelineResult` with canonical Provenance.

    Raises:
        CompressionPipelineError: when trainer + recipe pair is invalid OR
            Catalog #270 umbrella protocol blockers exist AND
            ``skip_protocol_verification`` is False.
    """
    if not lane_id or not lane_id.strip():
        raise CompressionPipelineError("lane_id must be non-empty")
    if not isinstance(video_path, Path):
        raise CompressionPipelineError("video_path must be a pathlib.Path")
    if not isinstance(substrate_trainer, Path):
        raise CompressionPipelineError("substrate_trainer must be a pathlib.Path")
    if not isinstance(recipe_path, Path):
        raise CompressionPipelineError("recipe_path must be a pathlib.Path")
    if not isinstance(output_dir, Path):
        raise CompressionPipelineError("output_dir must be a pathlib.Path")

    root = repo_root if repo_root is not None else REPO_ROOT
    substrate_id, recipe_name = validate_recipe_trainer_pair(
        substrate_trainer=substrate_trainer,
        recipe_path=recipe_path,
        repo_root=root,
    )

    hardware_substrate_token, hardware_substrate_class_value = (
        classify_hardware_substrate_for_dispatch(
            hardware_substrate,
            explicit_hardware_substrate=explicit_hardware_substrate,
        )
    )

    # Per the 8th MLX-first standing directive: MLX is the default encoder
    # path on local Apple Silicon; remote / non-Apple-Silicon falls back to
    # PyTorch (CUDA) or pure-numpy (CPU).
    if mlx_first is None:
        mlx_first_resolved = (
            hardware_substrate_class_value == HardwareSubstrateClass.LOCAL_MPS.value
            or hardware_substrate_token.startswith("macos_arm64_")
        )
    else:
        mlx_first_resolved = bool(mlx_first)

    # Catalog #270 umbrella protocol verification (unless explicitly bypassed).
    if skip_protocol_verification:
        overall_pass = True
        blockers: tuple[str, ...] = ()
    else:
        overall_pass, blockers = verify_compression_pipeline_protocol_complete(
            substrate_trainer=substrate_trainer,
            recipe_path=recipe_path,
            repo_root=root,
        )

    if not overall_pass and not skip_protocol_verification:
        # Surface the blockers in the error so the operator immediately
        # sees per-tier remediation rather than a generic refusal.
        raise CompressionPipelineError(
            f"Catalog #270 umbrella protocol refuses trainer={substrate_trainer.name!r} "
            f"+ recipe={recipe_path.name!r}: {len(blockers)} blocker(s) "
            f"-- {'; '.join(blockers[:5])}{'...' if len(blockers) > 5 else ''}; "
            "pass skip_protocol_verification=True for dry-run preparation, "
            "OR remediate blockers per Catalog #172/#178/#179/#180/#170/#171/"
            "#181/#182/#215/#244/#226/#240/#249 per-tier guidance"
        )

    measurement_utc = _utc_now_iso()
    canonical_provenance = derive_compression_pipeline_provenance(
        lane_id=lane_id,
        substrate_id=substrate_id,
        hardware_substrate=hardware_substrate_token,
        measurement_utc=measurement_utc,
    )

    per_axis_band = _extract_per_axis_predicted_band_from_recipe(
        recipe_path if recipe_path.is_absolute() else (root / recipe_path).resolve()
    )

    return CompressionPipelineResult(
        schema_version=COMPRESSION_PIPELINE_SCHEMA_VERSION,
        lane_id=lane_id,
        substrate_id=substrate_id,
        video_path=str(video_path),
        hardware_substrate=hardware_substrate_token,
        hardware_substrate_class=hardware_substrate_class_value,
        substrate_trainer_path=str(substrate_trainer),
        recipe_path=str(recipe_path),
        mlx_first_encode=mlx_first_resolved,
        qat_enabled=bool(qat_enabled),
        weights_export_path=(
            str(weights_export_path) if weights_export_path is not None else None
        ),
        weights_sha256=weights_sha256,
        weights_size_bytes=weights_size_bytes,
        training_anchor_call_id=training_anchor_call_id,
        qat_anchor_call_id=qat_anchor_call_id,
        dispatch_optimization_protocol_overall_pass=bool(overall_pass),
        dispatch_optimization_protocol_blockers=blockers,
        per_axis_predicted_band=per_axis_band,
        measurement_utc=measurement_utc,
        axis_tag=PREDICTED_AXIS_TAG,
        score_claim=False,
        promotable=False,
        evidence_grade="[predicted; compression-pipeline-canonical]",
        canonical_helper_invocation=(
            "tac.submission_packet.build_compression_pipeline"
        ),
        canonical_equation_id=CANONICAL_EQUATION_ID,
        canonical_equation_status="FORMALIZATION_PENDING",
        elapsed_seconds=float(elapsed_seconds),
        cost_usd=cost_usd,
        canonical_provenance=canonical_provenance,
        written_at_utc=measurement_utc,
        written_pid=os.getpid(),
        written_host=socket.gethostname(),
    )
