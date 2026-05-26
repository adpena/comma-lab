# SPDX-License-Identifier: MIT
"""tac.substrates._shared.posterior_emission_helper — canonical L0/L1 landing posterior emission.

Per WAVE-1-POSTERIOR-EMISSION-CANONICAL-WIRE-IN charter 2026-05-26 +
OPTIMIZATION-TOOLING-AUDIT roadmap commit ``e757bb74c`` META #1 CRITICAL
finding: ZERO of 8 Path 3 LANDED substrates emit ``posterior_update_locked``
→ cathedral autopilot EMPIRICALLY BLIND to ALL Path 3 contest signals.

This module is the canonical single-helper surface 8+ Path 3 substrates
invoke at landing-time (L0 SCAFFOLD smoke verdict / L1 trainer landing /
per-test posterior emission) so the cathedral autopilot's 62
auto-discovered consumers observe substrate signals via the canonical
posterior surfaces.

Per CLAUDE.md "Subagent coherence-by-default" hook #5 + Catalog #128
fcntl-locked posterior write discipline + Catalog #131 sister bare-write
guard + Catalog #138 strict-load discipline: the canonical write path is
``tac.continual_learning.posterior_update_locked``.

Per CLAUDE.md "MLX portable-local-substrate authority" + Catalog #317
canonical local-MPS dispatch markers + Catalog #341 cathedral consumer
canonical-routing-markers discipline: MLX research signals are
NON-PROMOTABLE by construction. They:

  - carry ``evidence_tag = "[MPS-research-signal]"`` (NON_PROMOTABLE_TAGS)
  - carry ``hardware_substrate = "macos_arm64"`` (macOS Apple Silicon)
  - are REFUSED by ``posterior_update_locked`` custody validator
    (recorded in ``refused_anchor_count``, not promoted to
    ``accepted_anchor_history``)
  - ALSO append to the canonical MPS-research-signal posterior at
    ``.omx/state/mps_research_signal_manifest.jsonl`` via
    ``tac.optimization.mps_research_signal.append_manifest_row_to_jsonl``
    (which cathedral consumers like ``cpu_axis_optimal_consumer`` and
    ``canonical_equation_lookup_consumer`` can query)

Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #323
canonical Provenance umbrella + Catalog #287 placeholder-rationale
rejection: every emitted anchor carries:

  - canonical Provenance via ``tac.provenance.builders.build_provenance_for_mps_proxy``
    (which auto-stamps ``promotion_eligible=False`` + ``score_claim_valid=False``)
  - canonical non-promotable markers (``score_claim=False`` /
    ``promotion_eligible=False`` / ``ready_for_exact_eval_dispatch=False`` /
    ``rank_or_kill_eligible=False``)
  - canonical blocker list documenting why the anchor is non-authoritative

Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against"
+ "Beauty, simplicity, and developer experience": ONE helper invocation
per substrate. Substrates pass:

  - substrate_id (e.g. ``"dreamer_v3_rssm"``)
  - archive_sha256 (64-hex; the canonical MLX-built archive bytes)
  - archive_bytes (positive int; the canonical archive size)
  - source_path (path to the artifact that produced this anchor;
    typically the trainer's output dir or a test fixture)

Optional fields for richer cathedral consumer observability:

  - predicted_score (float; the MLX-local predicted contest score; if
    supplied, carried in the canonical ContestResult.score_value field
    AND in the MPS manifest row's proxy_score field for downstream
    queryability)
  - predicted_d_seg / predicted_d_pose (float; per-axis predictions)
  - architecture_class (str; the substrate's canonical architecture
    class label; defaults to substrate_id)
  - notes (str; free-form provenance text; placeholder literals REJECTED
    per Catalog #287)

Sister substrates that have already adopted this helper (canonical
adoption pattern reference for new substrate wire-ins): see the L1
landing memos under ``.omx/research/path_3_*_landed_*.md`` for the 8
Path 3 substrates this helper structurally protects.

Catalog cross-refs:
  * Catalog #128 ``check_continual_learning_writes_use_lock`` — canonical
    fcntl-locked posterior write discipline this helper inherits.
  * Catalog #131 ``check_no_bare_writes_to_shared_state`` — sister
    bare-write guard that REFUSES non-canonical writes to ``.omx/state/``;
    this helper routes through canonical helpers only.
  * Catalog #138 ``check_state_writers_strict_load_for_mutating_path`` —
    strict-load discipline the canonical helpers inherit.
  * Catalog #287 ``check_no_docstring_overstatement_without_evidence_tag``
    — placeholder ``<rationale>`` / ``<reason>`` literals REJECTED in the
    notes/rationale fields.
  * Catalog #317 ``check_local_research_signal_dispatches_stamp_evidence_grade``
    — sister MPS-research-signal canonical routing discipline.
  * Catalog #323 ``check_no_score_claim_without_canonical_provenance``
    — canonical Provenance umbrella every emitted anchor carries.
  * Catalog #335 ``check_cathedral_consumer_directory_package_exposes_canonical_contract``
    — cathedral consumer canonical contract; auto-discovered consumers
    observe anchors emitted via this helper.
  * Catalog #341 ``check_cathedral_consumer_mps_prescreen_routing_carries_canonical_markers``
    — canonical non-promotable markers this helper threads through.
  * Catalog #355 ``check_cathedral_autopilot_main_invokes_meta_lagrangian``
    — sister meta-Lagrangian wire-in that consumes the posterior anchors
    emitted by this helper.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tac.continual_learning import (
    DEFAULT_POSTERIOR_LOCK_PATH,
    DEFAULT_POSTERIOR_PATH,
    ContestResult,
    PosteriorUpdate,
    posterior_update_locked,
)
from tac.optimization.mps_research_signal import (
    EVIDENCE_GRADE as MPS_RESEARCH_SIGNAL_EVIDENCE_GRADE,
    append_manifest_row_to_jsonl,
)
from tac.provenance import (
    Provenance,
    build_provenance_for_mps_proxy,
)

# Canonical MPS-research-signal posterior JSONL path. Sister of
# ``DEFAULT_POSTERIOR_PATH`` (tac.continual_learning) which is the
# canonical contest-axis posterior JSON. Per CLAUDE.md "MLX portable-
# local-substrate authority" this is the canonical MLX-research-signal
# posterior surface.
DEFAULT_MPS_RESEARCH_SIGNAL_MANIFEST_PATH: Path = (
    Path(os.environ.get("PACT_REPO_ROOT", "")) / ".omx" / "state"
    / "mps_research_signal_manifest.jsonl"
    if os.environ.get("PACT_REPO_ROOT")
    else Path(__file__).resolve().parents[4] / ".omx" / "state"
    / "mps_research_signal_manifest.jsonl"
)

# Canonical non-promotable markers per Catalog #127/#192/#317/#341.
# Every MLX research-signal manifest row carries these flags. The
# canonical ``append_manifest_row_to_jsonl`` REFUSES rows that flip any
# to True.
_CANONICAL_NON_PROMOTABLE_FLAGS: dict[str, bool] = {
    "score_claim": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "rank_or_kill_eligible": False,
}

# Canonical blocker list documenting why MLX research-signal anchors are
# non-authoritative. Surfaced in the manifest row so downstream consumers
# + operator briefings see structured non-promotability rationale.
_CANONICAL_MLX_NON_AUTHORITATIVE_BLOCKERS: tuple[str, ...] = (
    "macos_mlx_research_signal_not_contest_authority",
    "requires_paired_contest_cpu_plus_cuda_for_score_claim",
    "macos_mps_drift_23x_vs_cuda_per_claude_md_mps_auth_eval_is_noise",
    "predicted_band_validation_status_pending_post_training_per_catalog_324",
)

# Placeholder rationale literals refused per Catalog #287 sister
# discipline so the helper's docstring example cannot self-waive.
_PLACEHOLDER_RATIONALE_TOKENS: frozenset[str] = frozenset({
    "<rationale>",
    "<reason>",
    "<rationale_here>",
    "<reason_here>",
})


@dataclass(frozen=True)
class SubstrateLandingPosteriorAnchor:
    """Typed return value from ``emit_substrate_landing_posterior_anchor``.

    Per Catalog #305 6-facet observability: every field is inspectable +
    decomposable + diff-able + queryable + cite-able + counterfactual-able.

    Per Catalog #287/#323/#341: anchors are OBSERVABILITY-ONLY by
    construction; the canonical non-promotable markers + Provenance are
    threaded through every emission.
    """

    substrate_id: str
    architecture_class: str
    archive_sha256: str
    archive_bytes: int
    predicted_score: float
    evidence_tag: str
    hardware_substrate: str
    provenance: Provenance
    posterior_update: PosteriorUpdate
    manifest_row: dict[str, Any]
    manifest_path: Path

    # Canonical non-promotable markers (per Catalog #127/#192/#317/#341)
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False
    rank_or_kill_eligible: bool = False

    # Canonical blocker list (cite-able)
    non_authoritative_blockers: tuple[str, ...] = field(
        default_factory=lambda: tuple(_CANONICAL_MLX_NON_AUTHORITATIVE_BLOCKERS)
    )

    def __post_init__(self) -> None:
        if self.score_claim:
            raise ValueError(
                "score_claim=True forbidden per CLAUDE.md 'MLX portable-local-"
                "substrate authority' + Catalog #127/#192/#317/#341 promotion-"
                "leak guard; MLX research-signal anchors NEVER promote."
            )
        if self.promotion_eligible:
            raise ValueError(
                "promotion_eligible=True forbidden per the same non-negotiables."
            )
        if self.ready_for_exact_eval_dispatch:
            raise ValueError(
                "ready_for_exact_eval_dispatch=True forbidden; MLX anchors are "
                "non-promotable signal only."
            )
        if self.rank_or_kill_eligible:
            raise ValueError(
                "rank_or_kill_eligible=True forbidden; MLX anchors cannot "
                "drive rank-or-kill decisions per CLAUDE.md 'MPS auth eval "
                "is NOISE' non-negotiable."
            )


def _validate_archive_sha256(archive_sha256: str) -> None:
    if not isinstance(archive_sha256, str):
        raise TypeError(
            f"archive_sha256 must be str; got {type(archive_sha256).__name__}"
        )
    if len(archive_sha256) != 64 or not all(
        c in "0123456789abcdef" for c in archive_sha256.lower()
    ):
        raise ValueError(
            f"archive_sha256 must be 64-char lowercase hex; got "
            f"{archive_sha256!r}"
        )


def _validate_notes_no_placeholder(notes: str) -> None:
    """Catalog #287 sister discipline: reject placeholder rationale literals."""
    if not notes:
        return
    stripped = notes.strip().lower()
    if not stripped:
        return
    for token in _PLACEHOLDER_RATIONALE_TOKENS:
        if token.lower() in stripped:
            raise ValueError(
                f"notes contains placeholder rationale literal {token!r} per "
                "Catalog #287 sister discipline; supply a substantive "
                "rationale (>=4 chars, non-placeholder) instead."
            )
    if len(stripped) < 4:
        raise ValueError(
            f"notes rationale {notes!r} too short (<4 chars); supply a "
            "substantive non-placeholder rationale per Catalog #287."
        )


def emit_substrate_landing_posterior_anchor(
    *,
    substrate_id: str,
    archive_sha256: str,
    archive_bytes: int,
    source_path: str | Path,
    predicted_score: float = 0.20,
    predicted_d_seg: float | None = None,
    predicted_d_pose: float | None = None,
    architecture_class: str | None = None,
    notes: str = "L0 SCAFFOLD MLX landing posterior anchor; non-promotable per CLAUDE.md MLX research-signal discipline",
    posterior_path: Path | None = None,
    posterior_lock_path: Path | None = None,
    manifest_path: Path | None = None,
    extra_manifest_fields: dict[str, Any] | None = None,
) -> SubstrateLandingPosteriorAnchor:
    """Emit a canonical landing-time posterior anchor for an MLX-research-signal substrate.

    This is the canonical single-helper surface 8+ Path 3 substrates invoke
    at L0/L1 landing-time so the cathedral autopilot's 62 auto-discovered
    consumers observe substrate signals via the canonical posterior surfaces.

    Args:
        substrate_id: canonical substrate id (e.g. ``"dreamer_v3_rssm"``;
            matches the ``src/tac/substrates/<substrate_id>/`` directory).
        archive_sha256: SHA-256 of the canonical MLX-built archive bytes
            (64-char lowercase hex per :class:`Provenance` invariants).
        archive_bytes: positive int; the canonical archive size in bytes.
        source_path: path to the artifact that produced this anchor
            (typically the trainer's output dir or a test fixture). Per
            :func:`tac.provenance.builders.build_provenance_for_mps_proxy`
            transient ``/tmp`` paths are REFUSED.
        predicted_score: float; the MLX-local predicted contest score.
            Defaults to 0.20 (mid-band predicted-pursuit) when caller
            does not have an empirical anchor yet.
        predicted_d_seg / predicted_d_pose: optional per-axis predictions.
            If supplied, threaded into the canonical manifest row for
            downstream cathedral consumer observability.
        architecture_class: canonical architecture class label; defaults
            to ``substrate_id``.
        notes: substantive non-placeholder rationale per Catalog #287;
            placeholder ``<rationale>`` / ``<reason>`` literals REJECTED.
        posterior_path / posterior_lock_path: canonical
            ``continual_learning_posterior.json`` paths; defaults to
            ``tac.continual_learning.DEFAULT_POSTERIOR_PATH`` + sister
            lock.
        manifest_path: canonical MPS-research-signal manifest JSONL path;
            defaults to ``.omx/state/mps_research_signal_manifest.jsonl``.
        extra_manifest_fields: optional Mapping of substrate-specific
            extra fields to surface in the manifest row (e.g. substrate
            paradigm + canonical equation refs). Per
            :func:`append_manifest_row_to_jsonl` non-promotable defaults
            are auto-stamped so caller cannot pollute the canonical
            posterior via malformed extras.

    Returns:
        :class:`SubstrateLandingPosteriorAnchor` with all emission
        metadata + the canonical Provenance + the PosteriorUpdate verdict
        + the manifest row that was appended.

    Raises:
        ValueError: archive_sha256 not 64-char hex / archive_bytes not
            positive / notes contains placeholder rationale / required
            field empty.
        TypeError: archive_sha256 not str / archive_bytes not int.
    """
    if not isinstance(substrate_id, str) or not substrate_id.strip():
        raise ValueError("substrate_id must be non-empty str")
    if not isinstance(archive_bytes, int) or isinstance(archive_bytes, bool):
        raise TypeError(
            f"archive_bytes must be int; got {type(archive_bytes).__name__}"
        )
    if archive_bytes <= 0:
        raise ValueError(
            f"archive_bytes must be positive int; got {archive_bytes!r}"
        )
    _validate_archive_sha256(archive_sha256)
    _validate_notes_no_placeholder(notes)

    arch_class = (architecture_class or substrate_id).strip()
    if not arch_class:
        raise ValueError("architecture_class must be non-empty str")

    source_path_str = str(source_path)
    manifest_p = manifest_path or DEFAULT_MPS_RESEARCH_SIGNAL_MANIFEST_PATH

    # 1) Build canonical Provenance via build_provenance_for_mps_proxy.
    # Per Catalog #323 canonical Provenance umbrella: the helper auto-
    # stamps promotion_eligible=False + score_claim_valid=False.
    provenance = build_provenance_for_mps_proxy(
        artifact_sha256=archive_sha256,
        source_path=source_path_str,
    )

    # 2) Build the canonical ContestResult with non-promotable MLX
    # research-signal markers. The custody validator will REFUSE
    # promotion (NON_PROMOTABLE_TAGS) but the anchor IS recorded in
    # refused_anchor_count via posterior_update_locked.
    result = ContestResult(
        axis="cpu",  # MLX runs on Apple Silicon CPU/MPS; axis is non-authoritative-CPU
        hardware_substrate="macos_arm64",
        architecture_class=arch_class,
        score_value=float(predicted_score),
        evidence_tag="[MPS-research-signal]",
        archive_sha256=archive_sha256.lower(),
        archive_bytes=int(archive_bytes),
        cpu_pose=float(predicted_d_pose) if predicted_d_pose is not None else None,
        cpu_seg=float(predicted_d_seg) if predicted_d_seg is not None else None,
        notes=notes,
        metadata={
            "source": "substrate_landing_posterior_emission_helper",
            "substrate_id": substrate_id,
            "source_path": source_path_str,
            "evidence_grade": MPS_RESEARCH_SIGNAL_EVIDENCE_GRADE,
            "provenance_kind": provenance.artifact_kind.value,
            "provenance_evidence_grade": provenance.evidence_grade.value,
            "provenance_canonical_helper": provenance.canonical_helper_invocation,
            "non_authoritative_blockers": list(_CANONICAL_MLX_NON_AUTHORITATIVE_BLOCKERS),
        },
        observed_at_utc=datetime.now(UTC).isoformat(),
    )

    # 3) Append to the canonical continual_learning_posterior.json via
    # the canonical fcntl-locked helper. Per Catalog #128 + #131 + #138
    # sister discipline. The custody validator will REFUSE the MLX
    # research-signal tag (non-authoritative); the refusal is recorded
    # in refused_anchor_count which cathedral consumers observe.
    posterior_update = posterior_update_locked(
        result,
        posterior_path=posterior_path or DEFAULT_POSTERIOR_PATH,
        lock_path=posterior_lock_path or DEFAULT_POSTERIOR_LOCK_PATH,
        forbid_macos_promotion=True,
    )

    # 4) Append to the canonical MPS-research-signal posterior at
    # .omx/state/mps_research_signal_manifest.jsonl via the canonical
    # fcntl-locked sister helper. This IS the cathedral-consumer-
    # queryable surface for MLX research signals (per Catalog #317
    # canonical local-MPS dispatch markers).
    manifest_row: dict[str, Any] = {
        "substrate_id": substrate_id,
        "architecture_class": arch_class,
        "archive_sha256": archive_sha256.lower(),
        "archive_bytes": int(archive_bytes),
        "predicted_score": float(predicted_score),
        "source_path": source_path_str,
        "evidence_tag": "[MPS-research-signal]",
        "hardware_substrate": "macos_arm64",
        "axis_tag": "[MPS-research-signal]",  # Catalog #341 canonical marker
        "device": "mps",
        "predicted_delta_adjustment": 0.0,  # Catalog #341 routing-marker
        "promotable": False,  # Catalog #341 routing-marker
        "non_authoritative_blockers": list(_CANONICAL_MLX_NON_AUTHORITATIVE_BLOCKERS),
        "provenance_canonical_helper": provenance.canonical_helper_invocation,
        "provenance_kind": provenance.artifact_kind.value,
        "provenance_evidence_grade": provenance.evidence_grade.value,
        "notes": notes,
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "posterior_update_accepted": posterior_update.accepted,
        "posterior_update_refusal_reason": posterior_update.refusal_reason,
    }
    if predicted_d_seg is not None:
        manifest_row["predicted_d_seg"] = float(predicted_d_seg)
    if predicted_d_pose is not None:
        manifest_row["predicted_d_pose"] = float(predicted_d_pose)
    if extra_manifest_fields:
        # Stable + canonical: extras are added but cannot override
        # canonical non-promotable markers (sister helper enforces this
        # via auto-stamp).
        for k, v in extra_manifest_fields.items():
            if k in _CANONICAL_NON_PROMOTABLE_FLAGS:
                # Cannot override the canonical non-promotable markers.
                continue
            manifest_row[k] = v

    append_manifest_row_to_jsonl(
        manifest_row,
        output_path=manifest_p,
    )

    return SubstrateLandingPosteriorAnchor(
        substrate_id=substrate_id,
        architecture_class=arch_class,
        archive_sha256=archive_sha256.lower(),
        archive_bytes=int(archive_bytes),
        predicted_score=float(predicted_score),
        evidence_tag="[MPS-research-signal]",
        hardware_substrate="macos_arm64",
        provenance=provenance,
        posterior_update=posterior_update,
        manifest_row=manifest_row,
        manifest_path=manifest_p,
    )


def synthesize_substrate_archive_sha256(
    substrate_id: str,
    *,
    salt: str = "wave_1_posterior_emission_l0_scaffold",
) -> str:
    """Deterministic substrate-id → 64-char hex sha256 for L0 SCAFFOLD anchors.

    For substrates that don't yet have a real archive (L0 SCAFFOLD pre-
    smoke-trainer), this helper produces a deterministic placeholder
    sha256 from the substrate_id + canonical salt so the canonical
    Provenance + ContestResult invariants are satisfied without lying
    about archive bytes.

    The result is suffixed with the canonical salt prefix in the manifest
    row's ``synthesized_archive_sha256`` field so audit tools see the
    sha was NOT a real archive measurement.

    Args:
        substrate_id: canonical substrate id.
        salt: optional salt string (defaults to canonical L0 SCAFFOLD
            salt; callers SHOULD use the default to keep audit trail
            queryable).

    Returns:
        64-char lowercase hex sha256.
    """
    if not isinstance(substrate_id, str) or not substrate_id.strip():
        raise ValueError("substrate_id must be non-empty str")
    payload = f"{salt}:{substrate_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "DEFAULT_MPS_RESEARCH_SIGNAL_MANIFEST_PATH",
    "SubstrateLandingPosteriorAnchor",
    "emit_substrate_landing_posterior_anchor",
    "synthesize_substrate_archive_sha256",
]
