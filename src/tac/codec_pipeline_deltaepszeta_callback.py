# SPDX-License-Identifier: MIT
"""δεζ pipeline-aware training callback — Phase 2.

Per the four-way stack composition contract
(``.omx/research/four_way_stack_composition_contract_20260507_claude.md``),
the δεζ paradigm runs **upstream** of the canonical
:class:`tac.codec_pipeline.CodecPipeline`: it produces a state_dict at
training time that is natively friendlier to downstream codec ops (split-Brotli,
arithmetic coding, intN substrate). For δεζ training to actually optimize
toward smaller archives, the trainer needs an empirical signal — at each
epoch — of how the current state_dict encodes through the pipeline.

This module exposes :class:`CodecPipelineAwareTrainingCallback`. The callback:

  - runs the full :class:`tac.codec_pipeline.CodecPipeline` over the current
    ``state_dict`` at end-of-epoch (or whenever called),
  - reports per-op byte counts (from the
    :class:`tac.codec_pipeline.PipelineManifest`) as a co-training signal,
  - writes a JSONL log under ``experiments/results/<lane_id>/training_log/``
    so the run is durable + auditable post-hoc,
  - optionally adds a soft archive-size penalty to the loss
    (``add_to_loss``). Stub-mode by default (``lambda_penalty == 0``);
    real-mode (``lambda_penalty > 0``) returns a ramped quadratic penalty
    on archive overshoot.

Usage::

    pipeline = CodecPipeline([
        Op3_ApogeeIntN_Substrate(bits=6),
        Op1_PR101SplitBrotli(),
    ])
    callback = CodecPipelineAwareTrainingCallback(
        pipeline=pipeline,
        log_dir="experiments/results/lane_<id>/training_log",
    )
    # inside training loop:
    epoch_bytes = callback.report(model.state_dict(), epoch=N)
    loss = pixel_loss + callback.add_to_loss(0.0, archive_size_target=200_000)

CLAUDE.md compliance:
  - Strict-scorer-rule: pure CPU codec measurement; **no scorer load** at
    callback time.
  - No /tmp paths: log_dir is operator-supplied; default convention is
    ``experiments/results/<lane_id>/training_log/``.
  - Don't touch other ``codec_pipeline_*.py`` modules — only import.
  - No score claims: the callback reports BYTES, not scores. Any score
    derived from the pipeline-produced archive must come from contest-CUDA
    replay separately.

Cross-references:
  - Canonical orchestrator: :mod:`tac.codec_pipeline`
  - δεζ Phase 1 stubs: :mod:`tac.joint_scorer_aware_training`,
    :mod:`tac.learnable_entropy_model`,
    :mod:`tac.self_compress_full_renderer`
  - Composition contract:
    ``.omx/research/four_way_stack_composition_contract_20260507_claude.md``
  - Lane registry: ``lane_codec_pipeline_deltaepszeta_callback`` (L0).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tac.codec_pipeline import CodecPipeline

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-epoch report record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EpochReport:
    """One row in the JSONL training log.

    Each field is JSON-serializable. The JSONL log is the durable empirical
    record of how the pipeline byte counts evolve across training epochs;
    consumed downstream by reseed/calibration tooling and by post-hoc
    correlation against contest-CUDA scores.
    """
    epoch: int
    timestamp_utc: str
    total_bytes: int
    final_blob_sha256: str
    elapsed_seconds: float
    per_op_bytes: dict[str, int]
    per_op_delta_bytes: dict[str, int]
    archive_size_target: int | None = None
    overshoot_bytes: int = 0
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "timestamp_utc": self.timestamp_utc,
            "total_bytes": self.total_bytes,
            "final_blob_sha256": self.final_blob_sha256,
            "elapsed_seconds": round(self.elapsed_seconds, 4),
            "per_op_bytes": dict(self.per_op_bytes),
            "per_op_delta_bytes": dict(self.per_op_delta_bytes),
            "archive_size_target": self.archive_size_target,
            "overshoot_bytes": self.overshoot_bytes,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------

@dataclass
class CodecPipelineAwareTrainingCallback:
    """δεζ training callback that evaluates the current state_dict through
    the canonical :class:`CodecPipeline` at end of each epoch and reports
    per-op byte impact as a co-training signal.

    Attributes:
        pipeline: A constructed :class:`CodecPipeline`; the callback does
            NOT mutate it. The pipeline's ops are run in order at each
            ``report`` call.
        log_dir: Directory under which ``training_pipeline_bytes.jsonl`` is
            written. Created if absent. Per CLAUDE.md, must NOT be a /tmp
            path; canonical convention is
            ``experiments/results/<lane_id>/training_log/``.
        log_filename: Override for the JSONL filename; default is
            ``training_pipeline_bytes.jsonl``.
        skip_validate: Forwarded to :meth:`CodecPipeline.encode`. Default
            False — the Contrarian gate runs at every report. If the gate
            reliably trips on a synthetic-state-dict path that you know is
            safe, set True at your own risk.
        lambda_penalty: When > 0, ``add_to_loss`` returns a real soft
            penalty for archive overshoot. When == 0 (default), it returns
            0 (stub-mode) — the callback is observability-only.
        history: Per-epoch reports collected at ``report`` time. Useful
            for in-process correlation; the JSONL log is the durable record.

    Usage::

        callback = CodecPipelineAwareTrainingCallback(
            pipeline=CodecPipeline([Op3(bits=6), Op1()]),
            log_dir="experiments/results/lane_xyz_<UTC>/training_log",
        )
        # per epoch:
        epoch_bytes = callback.report(model.state_dict(), epoch=N)
        loss = pixel_loss + callback.add_to_loss(0.0, archive_size_target=200_000)
    """
    pipeline: CodecPipeline
    log_dir: Path | str
    log_filename: str = "training_pipeline_bytes.jsonl"
    skip_validate: bool = False
    lambda_penalty: float = 0.0
    history: list[EpochReport] = field(default_factory=list)

    # Cached last-encoded artifacts for ``add_to_loss``. Populated by
    # ``report``. Until ``report`` has been called at least once,
    # ``add_to_loss`` returns 0 because there is no archive size to
    # measure against.
    _last_total_bytes: int | None = field(default=None, init=False, repr=False)
    _last_per_op_bytes: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.pipeline, CodecPipeline):
            raise TypeError(
                f"pipeline must be a CodecPipeline, got {type(self.pipeline)!r}"
            )
        # Forbid /tmp paths per CLAUDE.md transient-evidence rule.
        log_dir_str = str(self.log_dir)
        if log_dir_str.startswith("/tmp/") or log_dir_str == "/tmp":
            raise ValueError(
                f"log_dir must not be under /tmp (transient-evidence trap); "
                f"got {log_dir_str!r}. Use experiments/results/<lane_id>/... "
                "or .omx/state/ instead."
            )
        if self.lambda_penalty < 0.0:
            raise ValueError(
                f"lambda_penalty must be >= 0, got {self.lambda_penalty!r}"
            )
        self.log_dir = Path(self.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    @property
    def log_path(self) -> Path:
        return Path(self.log_dir) / self.log_filename

    # ---------------------------------------------------------------
    # Core API
    # ---------------------------------------------------------------

    def report(
        self,
        state_dict: dict[str, torch.Tensor],
        epoch: int,
        *,
        archive_size_target: int | None = None,
        notes: str = "",
    ) -> dict[str, int]:
        """Encode ``state_dict`` through the pipeline; record per-op bytes.

        Returns a dict mapping ``op_name -> bytes_out``. The same dict
        (plus deltas, total, sha256, timestamp, etc.) is appended to the
        JSONL log and to ``self.history``.

        Strict-scorer-rule: this calls only ``pipeline.encode`` which is
        pure CPU codec; no scorer load.
        """
        if epoch < 0:
            raise ValueError(f"epoch must be >= 0, got {epoch!r}")

        _, manifest = self.pipeline.encode(
            state_dict, skip_validate=self.skip_validate
        )
        per_op_bytes: dict[str, int] = {}
        per_op_delta_bytes: dict[str, int] = {}
        for res in manifest.op_results:
            per_op_bytes[res.op_name] = int(res.bytes_out)
            per_op_delta_bytes[res.op_name] = int(res.bytes_delta)

        overshoot = 0
        if archive_size_target is not None:
            overshoot = max(0, manifest.final_bytes - archive_size_target)

        report = EpochReport(
            epoch=int(epoch),
            timestamp_utc=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            total_bytes=int(manifest.final_bytes),
            final_blob_sha256=manifest.final_blob_sha256,
            elapsed_seconds=float(manifest.elapsed_seconds),
            per_op_bytes=per_op_bytes,
            per_op_delta_bytes=per_op_delta_bytes,
            archive_size_target=archive_size_target,
            overshoot_bytes=int(overshoot),
            notes=notes,
        )
        self.history.append(report)
        self._append_jsonl(report)
        self._last_total_bytes = report.total_bytes
        self._last_per_op_bytes = dict(per_op_bytes)
        return dict(per_op_bytes)

    def add_to_loss(
        self,
        loss: float | torch.Tensor,
        archive_size_target: int = 200_000,
    ) -> float | torch.Tensor:
        """Return a soft penalty for archive-size overshoot.

        Stub-mode (default, ``self.lambda_penalty == 0``): returns 0. The
        callback is observability-only; the trainer's loss is unchanged.

        Real-mode (``self.lambda_penalty > 0``): returns
        ``lambda_penalty * max(0, archive_bytes - archive_size_target)``.
        ``archive_bytes`` comes from the most recent :meth:`report` call.
        Until ``report`` has been called at least once, returns 0 (the
        callback has no archive measurement yet).

        Note: this is intentionally a Python ``float`` arithmetic — the
        penalty is a non-differentiable signal (codec encoding is not
        end-to-end differentiable; bytes are an integer count). The
        downstream trainer should treat this as a Lagrangian-style
        regularizer added at loss-aggregate time, not as a backprop
        gradient source. Wiring it into a learnable Lagrange multiplier
        loop is Phase 3 (deferred to δεζ blueprint section 5).
        """
        if archive_size_target <= 0:
            raise ValueError(
                f"archive_size_target must be > 0, got {archive_size_target!r}"
            )
        # Stub-mode: lambda is 0 → penalty is 0 regardless of overshoot.
        if self.lambda_penalty == 0.0:
            return loss * 0 + 0  # preserves dtype/device for torch.Tensor loss
        # Real-mode: need a measurement. Without one, return 0 — the
        # callback hasn't been called yet, so the trainer should not be
        # penalized.
        if self._last_total_bytes is None:
            return loss * 0 + 0
        overshoot = max(0, self._last_total_bytes - archive_size_target)
        penalty = float(self.lambda_penalty) * float(overshoot)
        # Return loss + penalty, preserving torch.Tensor dtype/device when
        # the caller passed a tensor.
        return loss + penalty

    # ---------------------------------------------------------------
    # JSONL logging
    # ---------------------------------------------------------------

    def _append_jsonl(self, report: EpochReport) -> None:
        line = json.dumps(report.to_dict(), sort_keys=True, separators=(",", ":"))
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def read_log(self) -> list[dict[str, Any]]:
        """Read the JSONL log back. Useful for tests + post-hoc audit."""
        if not self.log_path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with self.log_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows


__all__ = [
    "CodecPipelineAwareTrainingCallback",
    "EpochReport",
]
