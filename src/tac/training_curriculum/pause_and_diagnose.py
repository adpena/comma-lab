# SPDX-License-Identifier: MIT
"""Pause-and-diagnose: canonical instrumented checkpoint for post-hoc inspection.

Generalizes A1's empirically-validated pause-and-diagnose pattern: the A1
0.19285 [contest-CPU] frontier was achieved by pausing an already-trained
PR95-paradigm substrate at inflate time and applying bias corrections to head-
output bytes (see `tools/build_a1_inflate_time_bias_correction_sweep.py` +
`feedback_a1_inflate_bias_sweep_exact_cpu_review_20260509_codex.md`).

The Karpathy nanoGPT pattern: pause at smallest viable scale, instrument
EVERY layer's activation magnitude / gradient norm / weight delta / scorer
component, then use the diagnostic to decide whether to (a) continue training,
(b) tune LR, (c) swap loss, (d) extract teacher checkpoint, or (e) abort and
re-design the substrate. The diagnostic IS the design feedback loop.

`[derived]` claims:
- A pause-checkpoint adds O(state_dict_bytes) memory + O(forward_pass_time)
  diagnostic compute. For a 100k-param substrate at fp16, this is ~200KB +
  ~1s; negligible vs. 1000-epoch training cost.

`[literature-extrapolation]` claims:
- Karpathy nanoGPT (2022 + 2024 lectures) explicitly demonstrates pause-and-
  instrument as the dominant debugging discipline; "let compute speak" tag.

`[would-need-empirical]` claims:
- Whether the post-hoc bias correction generalizes from A1 (PR95-paradigm
  substrate) to T4 Priority 1 BOLT-ON-on-A1 lanes (Ballé hyperprior /
  PR101 entropy stack / VQ-codebook) is empirically unknown until at least
  one bolt-on lane is dispatched.

Canonical-vs-unique decision per layer (Catalog #290)
─────────────────────────────────────────────────────
* Checkpoint serialization → ADOPT canonical (`torch.save` + atomic write
  via `tac.archive.utils` is the canonical pattern). FORK rationale: NONE.
* Diagnostic metric set → UNIQUE (substrate-specific; user supplies callable
  registry per design memo §"Observability surface").
* State-dict deep-clone → ADOPT canonical (every EMA helper in `tac.training`
  uses `{k: v.detach().clone() for k, v in ...}`).

Cargo-cult audit per assumption
───────────────────────────────
* "Pause cost is negligible compared to training cost" — HARD-EARNED for
  100k-param contest substrates at fp16; CARGO-CULTED for 100M+ param LLMs
  (Karpathy's nanoGPT regime). Contest scope: HARD-EARNED.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch


class PauseAndDiagnoseError(RuntimeError):
    """Raised when pause-and-diagnose checkpoint capture fails."""


@dataclass(frozen=True)
class DiagnosticMetric:
    """One diagnostic metric captured at a pause point.

    Args:
        name: Operator-readable diagnostic key (e.g.
            ``"head0_grad_norm_mean"``).
        value: Scalar metric value (float; serializable).
        axis: One of ``"contest-CUDA"`` / ``"contest-CPU"`` / ``"macOS-CPU
            advisory"`` / ``"MPS-PROXY"`` / ``"diagnostic"`` /
            ``"derived"``. ``"diagnostic"`` means the metric is the
            substrate's own internal state (no scorer axis); ``"derived"``
            means computed from first principles (rate-distortion / MDL).
        rationale: 1-line operator-readable explanation. Mandatory; empty
            strings rejected to prevent comment-only contracts per CLAUDE.md
            "Comment-only contracts are FORBIDDEN" non-negotiable.
    """

    name: str
    value: float
    axis: str
    rationale: str

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise PauseAndDiagnoseError("DiagnosticMetric.name must be non-empty")
        if self.axis not in {
            "contest-CUDA",
            "contest-CPU",
            "macOS-CPU advisory",
            "MPS-PROXY",
            "diagnostic",
            "derived",
        }:
            raise PauseAndDiagnoseError(
                f"DiagnosticMetric.axis={self.axis!r} not in canonical set"
            )
        if not self.rationale or not self.rationale.strip():
            raise PauseAndDiagnoseError(
                f"DiagnosticMetric(name={self.name!r}).rationale must be non-empty "
                "per CLAUDE.md 'Comment-only contracts are FORBIDDEN'"
            )


@dataclass(frozen=True)
class DiagnosticCheckpoint:
    """A pause-checkpoint capturing model state + diagnostic metrics.

    Args:
        epoch: Epoch number at which the pause was captured.
        state_dict_path: Path to the ``torch.save``'d state_dict (deep-cloned;
            CPU; ``torch.float32`` casts NOT applied so the live precision is
            preserved per Catalog #205 inflate-runtime discipline).
        metrics: Tuple of :class:`DiagnosticMetric` entries (variadic length).
        substrate_id: Canonical substrate id from
            ``.omx/state/lane_registry.json``; e.g. ``"nscs01_nullspace_split
            _renderer"``.
        utc_iso: ISO-formatted UTC timestamp of capture.
        notes: Operator-readable free-form notes (1-line; rejected if empty).
    """

    epoch: int
    state_dict_path: str
    metrics: tuple[DiagnosticMetric, ...]
    substrate_id: str
    utc_iso: str
    notes: str

    def __post_init__(self) -> None:
        if self.epoch < 0:
            raise PauseAndDiagnoseError(
                f"DiagnosticCheckpoint.epoch={self.epoch} must be >= 0"
            )
        if not self.state_dict_path:
            raise PauseAndDiagnoseError(
                "DiagnosticCheckpoint.state_dict_path must be non-empty"
            )
        if not self.substrate_id:
            raise PauseAndDiagnoseError(
                "DiagnosticCheckpoint.substrate_id must be non-empty"
            )
        if not self.notes or not self.notes.strip():
            raise PauseAndDiagnoseError(
                "DiagnosticCheckpoint.notes must be non-empty per CLAUDE.md "
                "'Comment-only contracts are FORBIDDEN'"
            )

    def to_manifest_dict(self) -> dict[str, Any]:
        """Serialize for JSON manifest persistence.

        Per CLAUDE.md "Beauty, simplicity, and developer experience":
        "make artifacts human-readable where possible and machine-checkable
        always".
        """
        return {
            "epoch": self.epoch,
            "state_dict_path": self.state_dict_path,
            "substrate_id": self.substrate_id,
            "utc_iso": self.utc_iso,
            "notes": self.notes,
            "metrics": [asdict(m) for m in self.metrics],
        }


def pause_and_capture(
    model: torch.nn.Module,
    *,
    epoch: int,
    output_dir: Path,
    substrate_id: str,
    metric_fns: dict[str, tuple[Callable[[torch.nn.Module], float], str, str]] | None = None,
    notes: str,
    utc_iso: str,
    ema_shadow: dict[str, torch.Tensor] | None = None,
) -> DiagnosticCheckpoint:
    """Pause training, capture model state + diagnostic metrics, return checkpoint.

    Generalizes A1's pause-and-diagnose pattern. Persists the state-dict to
    ``output_dir / f"pause_epoch_{epoch:04d}.pt"`` and a JSON manifest
    alongside.

    Args:
        model: Live :class:`torch.nn.Module` whose state to snapshot. If
            ``ema_shadow`` is provided, the EMA shadow IS snapshotted (per
            CLAUDE.md "EMA — non-negotiable" rule: "inference / archive bytes
            come from ``ema.state_dict()``").
        epoch: Epoch number (>= 0).
        output_dir: Output directory; created if missing.
        substrate_id: Canonical substrate id.
        metric_fns: Dict of ``{metric_name: (callable, axis, rationale)}``
            where ``callable(model) -> float`` computes the metric. Each
            callable is invoked with the model in its current state (live or
            EMA shadow per ``ema_shadow`` arg).
        notes: 1-line operator-readable notes; required.
        utc_iso: UTC ISO timestamp; required (caller supplies for determinism
            in tests).
        ema_shadow: Optional EMA shadow dict; if provided, snapshot the EMA
            state (NOT the live model state) per CLAUDE.md "EMA — non-
            negotiable".

    Returns:
        :class:`DiagnosticCheckpoint` with state-dict path + serialized
        metrics.

    Raises:
        :class:`PauseAndDiagnoseError` on any I/O or validation failure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dict_path = output_dir / f"pause_epoch_{epoch:04d}.pt"
    manifest_path = output_dir / f"pause_epoch_{epoch:04d}.manifest.json"

    # Snapshot: EMA shadow if provided, else live model state.
    if ema_shadow is not None:
        snapshot = {k: v.detach().cpu().clone() for k, v in ema_shadow.items()}
    else:
        snapshot = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    try:
        # Atomic write via .tmp + os.replace (Catalog #128/#131/#245 sister
        # discipline applied to checkpoint persistence).
        tmp_path = state_dict_path.with_suffix(state_dict_path.suffix + ".tmp")
        torch.save(snapshot, tmp_path)
        tmp_path.replace(state_dict_path)
    except OSError as e:
        raise PauseAndDiagnoseError(
            f"Failed to write pause checkpoint to {state_dict_path}: {e}"
        ) from e

    # Compute diagnostic metrics in eval mode (no gradient bleed).
    metrics: list[DiagnosticMetric] = []
    if metric_fns:
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                for name, (fn, axis, rationale) in metric_fns.items():
                    try:
                        value = float(fn(model))
                    except Exception as e:
                        raise PauseAndDiagnoseError(
                            f"Diagnostic metric {name!r} raised: {e}"
                        ) from e
                    metrics.append(
                        DiagnosticMetric(
                            name=name, value=value, axis=axis, rationale=rationale
                        )
                    )
        finally:
            if was_training:
                model.train()

    checkpoint = DiagnosticCheckpoint(
        epoch=epoch,
        state_dict_path=str(state_dict_path),
        metrics=tuple(metrics),
        substrate_id=substrate_id,
        utc_iso=utc_iso,
        notes=notes,
    )

    # Persist manifest as JSON for operator-readable inspection.
    try:
        manifest_path.write_text(
            json.dumps(checkpoint.to_manifest_dict(), indent=2, sort_keys=True)
        )
    except OSError as e:
        raise PauseAndDiagnoseError(
            f"Failed to write pause manifest to {manifest_path}: {e}"
        ) from e

    return checkpoint
