# SPDX-License-Identifier: MIT
"""Per-tensor ``||W||`` telemetry producer (#515 B0 instrument — the optdyn unlock).

The measured motivation (memo ``.omx/research/optimizer_dynamics_followup_20260715.md``
T1, equation ``inr_weight_norm_radial_ode_v1``): the Muon/NS update norm is
weight-independent, so ``eta_rel = ||u|| / ||W(t)||`` is a STATE variable — with a
flat Muon LR the mod32cap finisher ran a hidden per-layer relative-LR INCREASE
(x1.40 film / x1.09-1.22 hidden) as norms shrank. ``||W||`` is also SPECTRAL
content for sin/hosc rows, so the drift is an NTK band shift. The eta_rel-pin /
row-norm-projection / restoring-decay cures all READ this stream — sensors read
trainer streams, never recompute (poison taxonomy #3).

This module is the PRODUCER half only: pure functions over the trainer's already
materialized ``live_np`` / ``ema_np`` name->np.ndarray dicts (the exact shapes at
the trainer's verdict/checkpoint sites). It owns no optimizer, EMA, model, or
scorer state, allocates nothing persistent, and is score-neutral by construction
(read-only over copies the trainer already made). Trainer emission-site wiring is
QUEUED behind the live dry-start (pid 31576) — see
``tac.witness_dsl.constants_telemetry_build_wave_20260715.TRAINER_WIREIN_QUEUE``.

Read-only observability defaults ON per the "off is a tracked queue" law.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

WEIGHT_NORM_SCHEMA = "witness_weight_norm.v1"

# Trainer-internal cfg/bookkeeping arrays (``__cfg_*`` / ``__resume_*`` prefixed)
# are NOT weights; excluding them keeps the row a pure parameter-norm stream.
_EXCLUDED_PREFIXES: tuple[str, ...] = ("__",)


def _is_weight_key(name: str) -> bool:
    return not any(name.startswith(p) for p in _EXCLUDED_PREFIXES)


def per_tensor_norms(params: Mapping[str, np.ndarray]) -> dict[str, float]:
    """L2 norm per tensor (sorted keys; finite-checked, fail-loud on NaN/Inf).

    Accepts the trainer's ``live_np`` / ``ema_np`` dicts directly. Scalars and
    empty arrays are legal (norm 0.0 for empty).
    """
    out: dict[str, float] = {}
    for name in sorted(params):
        if not _is_weight_key(name):
            continue
        arr = np.asarray(params[name], dtype=np.float64)
        norm = float(np.linalg.norm(arr.ravel()))
        if not math.isfinite(norm):
            raise ValueError(
                f"weight_norm_telemetry: non-finite ||W|| for tensor {name!r} "
                f"(norm={norm!r}) — upstream weight corruption, fail loud"
            )
        out[name] = norm
    return out


@dataclass(frozen=True)
class WeightNormBaseline:
    """The t0 norms the drift ratios are measured against (resume-safe).

    Capture at the FIRST emission of a run (or restore from the first emitted
    row via :func:`baseline_from_row` after a resume) so ``rel_from_t0`` is
    stable across crash-resume — never re-anchored mid-run.
    """

    epoch: int
    norms: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.epoch) < 0:
            raise ValueError(f"WeightNormBaseline.epoch must be >= 0, got {self.epoch!r}")
        for k, v in self.norms.items():
            if not math.isfinite(float(v)) or float(v) < 0.0:
                raise ValueError(f"WeightNormBaseline: bad norm for {k!r}: {v!r}")


def baseline_from_row(row: Mapping[str, object]) -> WeightNormBaseline:
    """Rebuild the t0 baseline from a previously emitted row (crash-resume path)."""
    if row.get("schema") != WEIGHT_NORM_SCHEMA:
        raise ValueError(
            f"baseline_from_row: row schema {row.get('schema')!r} != {WEIGHT_NORM_SCHEMA!r}"
        )
    per_tensor = row.get("per_tensor")
    if not isinstance(per_tensor, Mapping) or not per_tensor:
        raise ValueError("baseline_from_row: row has no per_tensor mapping")
    norms = {str(k): float(v["norm"]) for k, v in per_tensor.items()}
    return WeightNormBaseline(epoch=int(row["ep"]), norms=norms)


def weight_norm_row(
    epoch: int,
    live: Mapping[str, np.ndarray],
    ema: Mapping[str, np.ndarray] | None = None,
    *,
    baseline: WeightNormBaseline | None = None,
    update_norms: Mapping[str, float] | None = None,
) -> dict[str, object]:
    """One emission-ready ``weight_norm`` telemetry row (score-neutral, read-only).

    Fields per tensor: ``norm`` (live L2), ``ema_norm`` (when the EMA shadow is
    supplied), ``rel_from_t0`` (norm/baseline - 1; the radial-ODE drift series),
    and ``eta_rel`` (= update_norm/norm when the caller supplies measured
    per-tensor update norms — the eta_rel-pin read stream). Global summary:
    total norm, min/max per-tensor drift.

    The trainer wire-in emits this at verdict cadence adjacent to the EMA-verdict
    row (where ``live_np``/``ema_np`` are already materialized), then routes it
    through the same run.log JSON stream every other telemetry row uses.
    """
    live_norms = per_tensor_norms(live)
    ema_norms = per_tensor_norms(ema) if ema is not None else {}
    per_tensor: dict[str, dict[str, float]] = {}
    drifts: list[float] = []
    for name, norm in live_norms.items():
        entry: dict[str, float] = {"norm": norm}
        if name in ema_norms:
            entry["ema_norm"] = ema_norms[name]
        if baseline is not None and name in baseline.norms and baseline.norms[name] > 0.0:
            rel = norm / baseline.norms[name] - 1.0
            entry["rel_from_t0"] = rel
            drifts.append(rel)
        if update_norms is not None and name in update_norms and norm > 0.0:
            un = float(update_norms[name])
            if not math.isfinite(un) or un < 0.0:
                raise ValueError(
                    f"weight_norm_row: bad update norm for {name!r}: {un!r}"
                )
            entry["eta_rel"] = un / norm
        per_tensor[name] = entry
    row: dict[str, object] = {
        "stage": "weight_norm",
        "schema": WEIGHT_NORM_SCHEMA,
        "ep": int(epoch),
        "per_tensor": per_tensor,
        "global": {
            "n_tensors": len(per_tensor),
            "total_norm": float(math.sqrt(sum(v * v for v in live_norms.values()))),
            "rel_from_t0_min": (min(drifts) if drifts else None),
            "rel_from_t0_max": (max(drifts) if drifts else None),
        },
    }
    return row


__all__ = [
    "WEIGHT_NORM_SCHEMA",
    "WeightNormBaseline",
    "baseline_from_row",
    "per_tensor_norms",
    "weight_norm_row",
]
