"""B1 HiNeRV PR95 full-curriculum training telemetry surface.

This module is the operator-facing "learn and analyze" surface for the B1
HiNeRV 229K full-curriculum MLX campaign (operator frontier-override
2026-06-09: "get full telemetry all stages all behavior best checkpoint to
learn and analyze").

It does NOT re-implement the training loop. The canonical
``tac.training.long_training_canonical.run_long_training`` already emits the
rich per-epoch telemetry JSONL (``per_axis_decomposition`` seg/pose/rate +
``ema_drift_l2`` + Muon/AdamW tensor counts + loss decomposition), the EMA
shadow best checkpoint, and the ``TrainingArtifact`` with canonical
Provenance per Catalog #323. This module READS those canonical artifacts and
EMITS three consolidated, queryable surfaces that the canonical telemetry does
NOT provide as a single artifact:

1. **Per-STAGE L14/L15 rollup** (:func:`build_pr95_curriculum_stage_records`)
   — the canonical 8-stage tuple (loss_family, lr/optimizer descriptor,
   qat_active, c1a_lambda, sigma, muon_active, epoch range, epochs-in-stage)
   read directly from the canonical
   :class:`tac.substrates._shared.mlx_score_aware.pr95_faithful_curriculum.PR95FaithfulCurriculumFactory`
   per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L14 +
   L15. ADOPT_CANONICAL: the factory IS the canonical L14 source; this module
   only projects it into a per-stage record list.

2. **Best-checkpoint manifest** (:func:`build_best_checkpoint_manifest`) — a
   small durable manifest carrying the EMA-shadow checkpoint path + its
   SHA-256 + epoch + stage + selection-metric proxy score + param count + the
   cite-able tuple (run_id, git_sha, config_hash, seed, arch_param_count) per
   CLAUDE.md "Max observability" facet 5 (cite-able). The canonical checkpoint
   ``.meta.json`` carries the path + selection metric but NOT the content
   SHA-256 nor the consolidated cite-tuple.

3. **Per-epoch decomposition rows** (:func:`iter_decomposed_epoch_rows`) — a
   thin projection of the canonical per-epoch telemetry into the operator's
   exact ask: ``{epoch, stage, seg, pose, rate, total, grad_norm, lr,
   seconds, ema_drift, ema_applied_proxy}`` per CLAUDE.md "Max observability"
   facet 2 (decomposable per signal). The canonical telemetry has all of
   these but buried inside ~200 ``dynamics_*`` keys; this projection surfaces
   exactly the operator's analysis signals.

All outputs carry the canonical non-promotable markers per CLAUDE.md "MLX
portable-local-substrate authority" + Catalog #127/#192/#317/#341:
``score_claim=False``, ``promotable=False``,
``ready_for_exact_eval_dispatch=False``, ``measurement_axis="[macOS-MLX
research-signal]"``. Every training number this module surfaces is research
signal only; the archive's SCORE is exact-eval'd later on contest hardware
(CPU Linux x86_64 + CUDA T4), NEVER from MLX.

Cross-references:
    - CLAUDE.md "Max observability" (the 6-facet definition).
    - CLAUDE.md "EMA -- NON-NEGOTIABLE" (best checkpoint = EMA shadow).
    - CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L14/L15.
    - ``tac.training.long_training_canonical`` (the canonical training loop +
      PerEpochMetrics + TrainingArtifact this module reads).
    - ``tac.substrates._shared.mlx_score_aware.pr95_faithful_curriculum``
      (the canonical L14 8-stage factory this module projects).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "MLX_RESEARCH_SIGNAL_AXIS",
    "TELEMETRY_SCHEMA_VERSION",
    "BestCheckpointManifest",
    "DecomposedEpochRow",
    "Pr95StageRecord",
    "build_best_checkpoint_manifest",
    "build_pr95_curriculum_stage_records",
    "iter_decomposed_epoch_rows",
    "read_decomposed_epoch_rows",
    "sha256_file",
    "summarize_run",
]

TELEMETRY_SCHEMA_VERSION: str = "b1_hinerv_training_telemetry.v1"

# Per CLAUDE.md "MPS auth eval is NOISE" + "MLX portable-local-substrate
# authority" — every number surfaced here is research signal, never a score
# claim. The exact score is measured later on contest hardware.
MLX_RESEARCH_SIGNAL_AXIS: str = "[macOS-MLX research-signal]"

# Canonical non-promotable marker block stamped on every emitted artifact per
# Catalog #127/#192/#317/#341.
_NON_PROMOTABLE_MARKERS: dict[str, Any] = {
    "measurement_axis": MLX_RESEARCH_SIGNAL_AXIS,
    "score_claim": False,
    "promotable": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}


def sha256_file(path: str | Path) -> str:
    """Return the lowercase-hex SHA-256 of a file's bytes.

    Used to bind a best-checkpoint manifest to the exact EMA-shadow bytes per
    CLAUDE.md "Max observability" facet 5 (cite-able) + the "certify or block"
    SSD-spill provenance rule (a checkpoint manifest must carry the content
    hash so a moved/rebuilt checkpoint can be verified).
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"sha256_file: not a file: {p}")
    digest = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# 1) Per-STAGE L14/L15 rollup (read from the canonical PR95 curriculum factory)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Pr95StageRecord:
    """One row of the canonical PR95 8-stage L14/L15 curriculum tuple.

    Mirrors
    :class:`tac.substrates._shared.mlx_score_aware.pr95_faithful_curriculum.PR95FaithfulCurriculumStageVerdict`
    projected into a JSON-able record per CLAUDE.md "HNeRV /
    leaderboard-implementation parity discipline" L14 (8-stage curriculum) +
    L15 (Muon final-stage-only). This is observability-only — the factory is
    the canonical authority.

    Attributes:
        stage_index: 1-indexed canonical PR95 stage (1..8).
        descriptor_id: canonical per-stage optimizer descriptor id
            (e.g. ``pr95_stage8_muon_adamw_mlx``).
        loss_family: canonical per-stage loss family (CE / tau_softplus /
            smooth / l7_softplus etc.) per L14.
        start_epoch: half-open stage start global epoch (inclusive).
        end_epoch: half-open stage end global epoch (exclusive).
        epochs_in_stage: ``end_epoch - start_epoch``.
        qat_active: whether quantization-aware training is active this stage.
        c1a_lambda: C1a coder-aware regularization weight (L16; 0.0 / 0.01 /
            0.02 across the canonical curriculum).
        sigma: Gaussian noise-injection sigma (L17; 0.2 / 0.1).
        muon_active: whether Muon is the optimizer this stage (L15 — stage 8
            only under ``faithful_stage8_only``).
        muon_policy: the active Muon policy.
    """

    stage_index: int
    descriptor_id: str
    loss_family: str
    start_epoch: int
    end_epoch: int
    epochs_in_stage: int
    qat_active: bool
    c1a_lambda: float
    sigma: float
    muon_active: bool
    muon_policy: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage_index": int(self.stage_index),
            "descriptor_id": self.descriptor_id,
            "loss_family": self.loss_family,
            "start_epoch": int(self.start_epoch),
            "end_epoch": int(self.end_epoch),
            "epochs_in_stage": int(self.epochs_in_stage),
            "qat_active": bool(self.qat_active),
            "c1a_lambda": float(self.c1a_lambda),
            "sigma": float(self.sigma),
            "muon_active": bool(self.muon_active),
            "muon_policy": self.muon_policy,
        }


def build_pr95_curriculum_stage_records(
    *,
    total_epoch_budget: int,
    muon_policy: str = "faithful_stage8_only",
) -> tuple[Pr95StageRecord, ...]:
    """Project the canonical PR95 8-stage curriculum into per-stage records.

    Reads the canonical
    :class:`PR95FaithfulCurriculumFactory` (the L14/L15 authority) and emits
    one :class:`Pr95StageRecord` per canonical stage. This is the per-STAGE
    consolidated table the operator asked for ("all stages") — distinct from
    the canonical per-epoch telemetry which only records the generic
    ``stage_name`` string.

    Args:
        total_epoch_budget: total curriculum epoch budget; passed verbatim to
            the factory (29,650 = canonical PR95; smaller budgets scale the
            stage shares keeping the canonical ratio per L14).
        muon_policy: ``faithful_stage8_only`` (canonical PR95) or
            ``every_stage``.

    Returns:
        the 8 canonical per-stage records in stage order.

    Raises:
        ImportError: MLX unavailable (the factory requires MLX).
        the factory's errors propagate for an invalid budget (< 8).
    """
    from tac.substrates._shared.mlx_score_aware.pr95_faithful_curriculum import (
        PR95FaithfulCurriculumFactory,
    )

    factory = PR95FaithfulCurriculumFactory(
        total_epoch_budget=int(total_epoch_budget),
        muon_policy=str(muon_policy),
    )
    records: list[Pr95StageRecord] = []
    for stage_index, start_epoch, end_epoch in factory.stage_epoch_boundaries:
        # current_stage_verdict expects an epoch INSIDE the stage; use the
        # start_epoch (inclusive lower bound of the half-open range).
        verdict = factory.current_stage_verdict(int(start_epoch))
        records.append(
            Pr95StageRecord(
                stage_index=int(stage_index),
                descriptor_id=str(verdict.descriptor_id),
                loss_family=str(verdict.loss_family),
                start_epoch=int(start_epoch),
                end_epoch=int(end_epoch),
                epochs_in_stage=int(end_epoch) - int(start_epoch),
                qat_active=bool(verdict.qat_active),
                c1a_lambda=float(verdict.cat_lambda),
                sigma=float(verdict.cat_sigma),
                muon_active=bool(verdict.uses_muon),
                muon_policy=str(verdict.muon_policy),
            )
        )
    return tuple(records)


# ---------------------------------------------------------------------------
# 2) Best-checkpoint manifest (EMA shadow + sha256 + cite-tuple)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BestCheckpointManifest:
    """Durable manifest binding the best EMA-shadow checkpoint to its bytes.

    Per CLAUDE.md "EMA -- NON-NEGOTIABLE" the inference checkpoint is the EMA
    shadow, NOT the live final-epoch weights. The canonical checkpoint
    ``.meta.json`` records the path + selection metric but NOT the content
    SHA-256 nor the consolidated cite-tuple. This manifest adds both so the
    operator (and a downstream B2 exact-eval) can verify the exact bytes and
    cite the run unambiguously per "Max observability" facet 5.

    Attributes:
        ema_shadow_checkpoint_path: path to the best EMA-shadow checkpoint.
        ema_shadow_sha256: SHA-256 of the EMA-shadow checkpoint bytes.
        is_ema_shadow: always True (the inference checkpoint is the EMA shadow
            per the EMA non-negotiable). Carried explicitly so consumers can
            assert it.
        best_epoch: global epoch at which the best checkpoint was captured.
        best_stage_name: the curriculum stage_name active at the best epoch.
        selection_metric_key / selection_metric_value /
            selection_metric_mode: the canonical checkpoint selector that
            chose this checkpoint (e.g. ``total`` / ``min``). This is a
            training-proxy value, NOT a contest score.
        arch_param_count: total trainable parameter count of the substrate.
        run_id / git_sha / config_hash / seed: the cite-able tuple per
            facet 5.
    """

    ema_shadow_checkpoint_path: str
    ema_shadow_sha256: str
    is_ema_shadow: bool
    best_epoch: int
    best_stage_name: str
    selection_metric_key: str
    selection_metric_value: float
    selection_metric_mode: str
    arch_param_count: int
    run_id: str
    git_sha: str
    config_hash: str
    seed: int

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": "b1_hinerv_best_checkpoint_manifest.v1",
            "ema_shadow_checkpoint_path": self.ema_shadow_checkpoint_path,
            "ema_shadow_sha256": self.ema_shadow_sha256,
            "is_ema_shadow": bool(self.is_ema_shadow),
            "best_epoch": int(self.best_epoch),
            "best_stage_name": self.best_stage_name,
            "selection_metric_key": self.selection_metric_key,
            # Proxy training metric, NOT a contest score. Named explicitly so
            # no downstream consumer mistakes it for an authoritative score.
            "selection_metric_value_proxy": float(self.selection_metric_value),
            "selection_metric_mode": self.selection_metric_mode,
            "arch_param_count": int(self.arch_param_count),
            "cite_tuple": {
                "run_id": self.run_id,
                "git_sha": self.git_sha,
                "config_hash": self.config_hash,
                "seed": int(self.seed),
                "arch_param_count": int(self.arch_param_count),
            },
        }
        payload.update(_NON_PROMOTABLE_MARKERS)
        return payload


def _coerce_float(value: Any, *, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out != out:  # NaN
        return default
    return out


def build_best_checkpoint_manifest(
    *,
    best_checkpoint_meta_path: str | Path,
    arch_param_count: int,
    run_id: str,
    git_sha: str,
    config_hash: str,
    seed: int,
    best_stage_name: str = "",
) -> BestCheckpointManifest:
    """Build the best-checkpoint manifest from a canonical checkpoint meta.

    Reads the canonical ``best_epoch*.meta.json`` emitted by
    ``tac.training.long_training_canonical`` (schema
    ``long_training_canonical_checkpoint.v1``), locates the EMA-shadow state
    file it references, hashes the EMA-shadow bytes, and packages the cite-able
    tuple.

    Args:
        best_checkpoint_meta_path: path to the canonical best checkpoint
            ``.meta.json``.
        arch_param_count: total trainable parameter count of the substrate.
        run_id / git_sha / config_hash / seed: the cite-able tuple.
        best_stage_name: optional curriculum stage_name active at the best
            epoch (the canonical meta does not always carry it).

    Returns:
        a :class:`BestCheckpointManifest`.

    Raises:
        FileNotFoundError: the meta or the referenced EMA-shadow file is
            missing.
        ValueError: the meta is not the canonical best-checkpoint schema, or
            does not reference an EMA-shadow state path.
    """
    meta_path = Path(best_checkpoint_meta_path)
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"build_best_checkpoint_manifest: meta not found: {meta_path}"
        )
    meta = json.loads(meta_path.read_text())
    if not isinstance(meta, Mapping):
        raise ValueError(
            f"checkpoint meta must be a JSON object; got {type(meta).__name__}"
        )
    ema_path_str = meta.get("ema_shadow_state_path")
    if not ema_path_str:
        raise ValueError(
            "checkpoint meta does not reference an EMA-shadow state path; per "
            "CLAUDE.md 'EMA -- NON-NEGOTIABLE' the best inference checkpoint "
            "MUST be the EMA shadow, not live weights. "
            f"meta={meta_path}"
        )
    ema_path = Path(ema_path_str)
    if not ema_path.is_file():
        # The canonical meta stores an absolute path; if the run was moved to
        # cold store the relative sibling is the fallback.
        sibling = meta_path.parent / ema_path.name
        if sibling.is_file():
            ema_path = sibling
        else:
            raise FileNotFoundError(
                f"EMA-shadow state file referenced by meta not found: "
                f"{ema_path} (and no sibling {sibling})"
            )
    return BestCheckpointManifest(
        ema_shadow_checkpoint_path=str(ema_path),
        ema_shadow_sha256=sha256_file(ema_path),
        is_ema_shadow=True,
        best_epoch=int(meta.get("global_epoch", -1)),
        best_stage_name=str(best_stage_name or meta.get("stage_name") or ""),
        selection_metric_key=str(meta.get("checkpoint_selection_metric_key") or "total"),
        selection_metric_value=_coerce_float(
            meta.get("checkpoint_selection_metric_value", meta.get("loss"))
        ),
        selection_metric_mode=str(meta.get("checkpoint_selection_metric_mode") or "min"),
        arch_param_count=int(arch_param_count),
        run_id=str(run_id),
        git_sha=str(git_sha),
        config_hash=str(config_hash),
        seed=int(seed),
    )


# ---------------------------------------------------------------------------
# 3) Per-epoch decomposition projection (seg / pose / rate / grad-norm / lr)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecomposedEpochRow:
    """The operator's per-epoch analysis signals projected from canonical telemetry.

    The canonical per-epoch telemetry JSONL row carries ~200 ``dynamics_*``
    keys; this projection surfaces exactly the operator's "all behavior"
    analysis signals per CLAUDE.md "Max observability" facet 2 (decomposable
    per signal):

    Attributes:
        epoch: global epoch index.
        stage_name: curriculum stage_name active this epoch.
        seg: per-axis SegNet score contribution (``per_axis_decomposition.seg``).
        pose: per-axis PoseNet score contribution
            (``per_axis_decomposition.pose``). This is already the sqrt-term
            score contribution per the canonical adapter.
        rate: per-axis archive-bytes contribution
            (``per_axis_decomposition.archive_bytes``).
        recon_aux: auxiliary reconstruction contribution if present.
        total: total proxy loss for the epoch.
        grad_norm: pre-clip global gradient L2 norm.
        learning_rate: effective learning rate this epoch.
        seconds: cumulative wall-clock seconds since training start.
        ema_drift_l2: L2 norm of (live - EMA shadow).
        uses_muon: whether the optimizer used Muon this epoch (1.0/0.0 → bool).
    """

    epoch: int
    stage_name: str
    seg: float
    pose: float
    rate: float
    recon_aux: float
    total: float
    grad_norm: float
    learning_rate: float
    seconds: float
    ema_drift_l2: float
    uses_muon: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "epoch": int(self.epoch),
            "stage_name": self.stage_name,
            "seg": float(self.seg),
            "pose": float(self.pose),
            "rate": float(self.rate),
            "recon_aux": float(self.recon_aux),
            "total": float(self.total),
            "grad_norm": float(self.grad_norm),
            "learning_rate": float(self.learning_rate),
            "seconds": float(self.seconds),
            "ema_drift_l2": float(self.ema_drift_l2),
            "uses_muon": bool(self.uses_muon),
        }


def _project_epoch_row(row: Mapping[str, Any]) -> DecomposedEpochRow:
    """Project a canonical per-epoch telemetry row into a DecomposedEpochRow."""
    per_axis = row.get("per_axis_decomposition") or {}
    if not isinstance(per_axis, Mapping):
        per_axis = {}
    components = row.get("loss_components") or {}
    if not isinstance(components, Mapping):
        components = {}
    # grad_norm: prefer the canonical pre-clip global norm key.
    grad_norm = _coerce_float(
        components.get(
            "gradient_global_norm_pre_clip",
            components.get("dynamics_gradient_all_l2", 0.0),
        )
    )
    uses_muon = _coerce_float(components.get("pact_optimizer_uses_muon", 0.0)) > 0.0
    return DecomposedEpochRow(
        epoch=int(row.get("epoch", -1)),
        stage_name=str(row.get("stage_name") or ""),
        seg=_coerce_float(per_axis.get("seg")),
        pose=_coerce_float(per_axis.get("pose")),
        rate=_coerce_float(per_axis.get("archive_bytes")),
        recon_aux=_coerce_float(per_axis.get("recon_aux")),
        total=_coerce_float(row.get("loss")),
        grad_norm=grad_norm,
        learning_rate=_coerce_float(row.get("learning_rate")),
        seconds=_coerce_float(row.get("wall_clock_seconds")),
        ema_drift_l2=_coerce_float(row.get("ema_drift_l2")),
        uses_muon=uses_muon,
    )


def iter_decomposed_epoch_rows(
    rows: Iterable[Mapping[str, Any]],
) -> Iterator[DecomposedEpochRow]:
    """Project an iterable of canonical telemetry rows into decomposed rows."""
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        yield _project_epoch_row(row)


def read_decomposed_epoch_rows(
    telemetry_jsonl_path: str | Path,
) -> tuple[DecomposedEpochRow, ...]:
    """Read a canonical telemetry JSONL file and project every epoch row.

    Args:
        telemetry_jsonl_path: path to the canonical ``telemetry.jsonl``.

    Returns:
        decomposed epoch rows in file order.

    Raises:
        FileNotFoundError: the telemetry file is missing.
    """
    path = Path(telemetry_jsonl_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"read_decomposed_epoch_rows: telemetry not found: {path}"
        )
    rows: list[DecomposedEpochRow] = []
    with path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # Skip a partially-flushed trailing line in a live run; the
                # next read picks it up once complete.
                continue
            if isinstance(obj, Mapping):
                rows.append(_project_epoch_row(obj))
    return tuple(rows)


# ---------------------------------------------------------------------------
# Consolidated run summary (the "learn and analyze" headline)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunSummary:
    """Consolidated headline summary of a B1 training run for the operator.

    Attributes:
        epochs_observed: number of telemetry epoch rows seen.
        first_total / last_total / best_total: proxy loss trajectory headline.
        first_pose / last_pose / best_pose: pose-axis trajectory headline.
        first_seg / last_seg / best_seg: seg-axis trajectory headline.
        last_rate: most-recent archive-bytes axis value.
        total_wall_clock_seconds: cumulative wall-clock seconds.
        mean_seconds_per_epoch: average seconds/epoch.
        muon_epochs_observed: count of epochs where Muon was the optimizer.
        stages_observed: distinct stage_name values seen.
        loss_is_finite: whether all observed totals were finite.
    """

    epochs_observed: int
    first_total: float
    last_total: float
    best_total: float
    first_pose: float
    last_pose: float
    best_pose: float
    first_seg: float
    last_seg: float
    best_seg: float
    last_rate: float
    total_wall_clock_seconds: float
    mean_seconds_per_epoch: float
    muon_epochs_observed: int
    stages_observed: tuple[str, ...]
    loss_is_finite: bool = field(default=True)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": "b1_hinerv_run_summary.v1",
            "epochs_observed": int(self.epochs_observed),
            "first_total": float(self.first_total),
            "last_total": float(self.last_total),
            "best_total": float(self.best_total),
            "first_pose": float(self.first_pose),
            "last_pose": float(self.last_pose),
            "best_pose": float(self.best_pose),
            "first_seg": float(self.first_seg),
            "last_seg": float(self.last_seg),
            "best_seg": float(self.best_seg),
            "last_rate": float(self.last_rate),
            "total_wall_clock_seconds": float(self.total_wall_clock_seconds),
            "mean_seconds_per_epoch": float(self.mean_seconds_per_epoch),
            "muon_epochs_observed": int(self.muon_epochs_observed),
            "stages_observed": list(self.stages_observed),
            "loss_is_finite": bool(self.loss_is_finite),
        }
        payload.update(_NON_PROMOTABLE_MARKERS)
        return payload


def summarize_run(rows: Iterable[DecomposedEpochRow]) -> RunSummary:
    """Compute the consolidated run summary from decomposed epoch rows.

    Args:
        rows: decomposed epoch rows (e.g. from :func:`read_decomposed_epoch_rows`).

    Returns:
        a :class:`RunSummary`.

    Raises:
        ValueError: no rows were provided (cannot summarize an empty run).
    """
    materialized = list(rows)
    if not materialized:
        raise ValueError("summarize_run: no epoch rows to summarize")
    totals = [r.total for r in materialized]
    poses = [r.pose for r in materialized]
    segs = [r.seg for r in materialized]
    loss_is_finite = all(t == t and t not in (float("inf"), float("-inf")) for t in totals)
    total_wall = max((r.seconds for r in materialized), default=0.0)
    n = len(materialized)
    stages: list[str] = []
    for r in materialized:
        if r.stage_name and r.stage_name not in stages:
            stages.append(r.stage_name)
    return RunSummary(
        epochs_observed=n,
        first_total=totals[0],
        last_total=totals[-1],
        best_total=min(totals),
        first_pose=poses[0],
        last_pose=poses[-1],
        best_pose=min(poses),
        first_seg=segs[0],
        last_seg=segs[-1],
        best_seg=min(segs),
        last_rate=materialized[-1].rate,
        total_wall_clock_seconds=total_wall,
        mean_seconds_per_epoch=(total_wall / n) if n else 0.0,
        muon_epochs_observed=sum(1 for r in materialized if r.uses_muon),
        stages_observed=tuple(stages),
        loss_is_finite=loss_is_finite,
    )
