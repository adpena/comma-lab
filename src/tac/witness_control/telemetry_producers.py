"""Additive, score-neutral trainer telemetry producers for the v7.5/v8 vehicle.

This module owns the small, testable state machines behind the D-A/D-B launch
observers and the queued task-408 boundary telemetry.  It deliberately owns no
optimizer, EMA, model, or scorer state.  The trainer supplies real measured
durations and gradient vectors; these helpers aggregate and persist rows only.

All JSONL writes route through :func:`tac.jsonl_store.append_locked_jsonl`.
State that must not duplicate after a crash implements the canonical resume
registry protocol with additive ``__dtp_*`` keys.  A legacy sidecar containing
none of those keys restores to a deterministic empty observer state.
"""
from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tac.causal_manifest import (
    EVENT_FAMILY_PRIORITY,
    ArtifactRef,
    CausalManifestWriter,
    ClassEdgeMark,
    EventMarkRow,
    IncidenceMark,
    ReceiverStateMark,
    SpacetimeMark,
    StratumMark,
)
from tac.jsonl_store import append_locked_jsonl

COMPONENT_WALLCLOCK_SCHEMA = "witness_component_wallclock.v1"
SPS_ENGAGEMENT_SCHEMA = "sps_gradient_role_conflict_engagement.v1"
# The 8 v1 fields (probe-derived + cadence-gated + total) PLUS the 2026-07-15 #480
# REAL-PATH fields (``real_*``): disjoint main-thread wall intervals measured around
# the ACTUAL training-loop regions (no probes, no graph changes — pure perf_counter_ns
# around existing statements => score-neutral by construction). Unlike the inclusive
# probe fields, the real_* fields ARE summable: real_path_sum_s <= epoch_total_s and
# (epoch_total_s - real_path_sum_s - checkpoint_io_s) is the honest UNATTRIBUTED
# remainder (python loop + staging + everything not yet timed). Additive + legacy-
# compatible: v1 rows simply lack the real_* keys; readers keying the 8 v1 names are
# unchanged (schema string deliberately stays .v1 — the launch-ticket AST gate pins it).
COMPONENT_FIELDS: tuple[str, ...] = (
    "teacher_forward_s",
    "teacher_backward_s",
    "witness_forward_s",
    "witness_backward_s",
    "realized_R_s",
    "verdict_s",
    "checkpoint_io_s",
    "real_grad_accum_s",             # per-chunk fused value_and_grad + mx.eval (the real update path)
    "real_optimizer_s",              # clip + per-group clip + opt.update + EMA + mx.eval
    "real_loss_terms_telemetry_s",   # the per-term loss recompute/alarm block (_lt_stride)
    "real_epoch_probes_s",           # the D-A decomposition probe + SPS emit themselves
    "real_verdict_submit_s",         # MAIN-thread verdict join/decide/schedule (the submit block)
    # (#480 v2, 2026-07-15) three DISJOINT SPANS covering the whole epoch body on a SECOND
    # axis (complementary to the real_* update-path intervals; deliberately NOT real_-prefixed
    # so REAL_PATH_FIELDS/real_path_sum_s are unchanged): span_pre_loop + span_accum_loop +
    # span_epoch_tail ~= epoch_total. Classifies the measured 77% unattributed remainder
    # (ep2 drystart3: 848.5s of 1095.6s): in-loop-gap = span_accum_loop_s - (real_grad_accum
    # + real_optimizer + real_loss_terms + real_epoch_probes) vs epoch-tail (event sensors /
    # annulus / jacobian-basin / causal-manifest / verdict+checkpoint region) vs pre-loop.
    "span_pre_loop_s",
    "span_accum_loop_s",
    "span_epoch_tail_s",
    "epoch_total_s",
)
REAL_PATH_FIELDS: tuple[str, ...] = tuple(
    name for name in COMPONENT_FIELDS if name.startswith("real_"))
SPS_MATERIAL_FRACTION_THRESHOLD = 0.10
SPS_MATERIAL_COSINE_THRESHOLD = -0.05
_TIMER_COMPONENTS = frozenset(COMPONENT_FIELDS[:-1])


def _finite_nonnegative(value: float, *, name: str) -> float:
    out = float(value)
    if not math.isfinite(out) or out < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
    return out


@dataclass
class ComponentWallclock:
    """Per-epoch monotonic component timer with explicit missing/error semantics.

    The four forward/backward fields may be same-function, read-only decomposition
    probes because the production MLX loss is a fused lazy graph.  Such observations
    are labelled by ``measurement_scope`` and never represented as a disjoint sum.
    ``None`` means unmeasured; zero is used only for an actually measured zero-length
    interval returned by the injected clock.
    """

    path: Path
    clock_ns: Callable[[], int] = time.perf_counter_ns
    measurement_scope: str = "real_path_plus_same_function_read_only_decomposition_probe"
    _seconds: dict[str, float] = field(default_factory=dict)
    _calls: dict[str, int] = field(default_factory=dict)
    _not_invoked: set[str] = field(default_factory=set)
    _errors: list[dict[str, str]] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def reset(self) -> None:
        with self._lock:
            self._seconds.clear()
            self._calls.clear()
            self._not_invoked.clear()
            self._errors.clear()

    def add(self, component: str, seconds: float, *, calls: int = 1) -> None:
        if component not in _TIMER_COMPONENTS:
            raise KeyError(f"unknown wall-clock component {component!r}")
        if int(calls) < 1:
            raise ValueError(f"calls must be >=1, got {calls!r}")
        measured = _finite_nonnegative(seconds, name=component)
        with self._lock:
            self._seconds[component] = self._seconds.get(component, 0.0) + measured
            self._calls[component] = self._calls.get(component, 0) + int(calls)
            self._not_invoked.discard(component)

    def mark_not_invoked(self, component: str) -> None:
        """Record a measured structural zero without pretending work ran.

        Verdict and checkpoint I/O are cadence-gated.  An epoch where their
        predicates are false has a real duration of zero, distinct from an
        attempted measurement that failed or was omitted.
        """

        if component not in _TIMER_COMPONENTS:
            raise KeyError(f"unknown wall-clock component {component!r}")
        with self._lock:
            if component not in self._seconds:
                self._seconds[component] = 0.0
                self._calls.setdefault(component, 0)
                self._not_invoked.add(component)

    def record_error(self, component: str, exc: BaseException) -> None:
        if component not in _TIMER_COMPONENTS and component != "decomposition_probe":
            raise KeyError(f"unknown wall-clock component {component!r}")
        with self._lock:
            self._errors.append({
                "component": component,
                "error": f"{type(exc).__name__}: {exc}",
            })

    def add_ns(self, component: str, elapsed_ns: int, *, calls: int = 1) -> None:
        if int(elapsed_ns) < 0:
            raise ValueError(f"elapsed_ns must be non-negative, got {elapsed_ns!r}")
        self.add(component, int(elapsed_ns) / 1_000_000_000.0, calls=calls)

    @contextmanager
    def measure(self, component: str) -> Iterator[None]:
        start = int(self.clock_ns())
        try:
            yield
        finally:
            stop = int(self.clock_ns())
            self.add_ns(component, stop - start)

    def measure_probe(self, component: str, fn: Callable[[], Any]) -> Any | None:
        """Measure a read-only probe, recording a loud error without raising.

        Training-critical work must use :meth:`measure` so its exception semantics
        remain unchanged.  This method is only for observer probes whose failure must
        not kill or perturb a training run.
        """

        start = int(self.clock_ns())
        try:
            value = fn()
        except Exception as exc:  # observability is fail-open for training
            self.record_error(component, exc)
            return None
        self.add_ns(component, int(self.clock_ns()) - start)
        return value

    def row(self, *, epoch: int, epoch_total_s: float) -> dict[str, Any]:
        total = _finite_nonnegative(epoch_total_s, name="epoch_total_s")
        with self._lock:
            values: dict[str, float | None] = {
                name: self._seconds.get(name) for name in COMPONENT_FIELDS[:-1]
            }
            values["epoch_total_s"] = total
            observed_sum = sum(
                float(values[name])
                for name in COMPONENT_FIELDS[:-1]
                if values[name] is not None
            )
            real_path_sum = sum(
                float(values[name])
                for name in REAL_PATH_FIELDS
                if values[name] is not None
            )
            return {
                "schema": COMPONENT_WALLCLOCK_SCHEMA,
                "stage": "witness_component_wallclock",
                "epoch": int(epoch),
                **values,
                "component_calls": {
                    name: int(self._calls.get(name, 0)) for name in COMPONENT_FIELDS[:-1]
                },
                "not_invoked": sorted(self._not_invoked),
                "measurement_scope": self.measurement_scope,
                "clock": "time.perf_counter_ns",
                "overlap_policy": (
                    "inclusive backward probes and real-path intervals are not asserted disjoint; "
                    "do not sum components as an epoch decomposition — EXCEPT the real_* fields, "
                    "which are disjoint main-thread intervals: real_path_sum_s is summable and "
                    "epoch_total_s - real_path_sum_s - checkpoint_io_s = unattributed remainder"
                ),
                "observed_component_sum_s": float(observed_sum),
                "real_path_sum_s": float(real_path_sum),
                "errors": list(self._errors),
                "complete": not self._errors and all(
                    values[name] is not None for name in COMPONENT_FIELDS
                ),
                "score_neutral": True,
                "promotable": False,
                "axis": "[macOS-MLX timing observer] NON-PROMOTABLE",
            }

    def emit(self, *, epoch: int, epoch_total_s: float) -> dict[str, Any]:
        row = self.row(epoch=epoch, epoch_total_s=epoch_total_s)
        append_locked_jsonl(self.path, row)
        return row


def deterministic_strata(n_pairs: int, k: int = 4) -> tuple[int, ...]:
    """Return midpoint-stratified deterministic pair indices.

    For the canonical n600/K=4 configuration this is exactly
    ``(75, 225, 375, 525)``.
    """

    n = int(n_pairs)
    count = int(k)
    if n < 1:
        raise ValueError(f"n_pairs must be >=1, got {n_pairs!r}")
    if count < 1 or count > n:
        raise ValueError(f"k must be in [1,n_pairs], got k={k!r}, n_pairs={n}")
    return tuple(min(n - 1, int((i + 0.5) * n / count)) for i in range(count))


def _flat_mapping(vectors: Mapping[str, Any]) -> tuple[list[str], dict[str, np.ndarray]]:
    names = sorted(str(name) for name in vectors)
    flat = {name: np.asarray(vectors[name], dtype=np.float64).reshape(-1) for name in names}
    return names, flat


def _cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return None
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


def gradient_role_conflict_stats(
    prediction: Mapping[str, Any], temporal: Mapping[str, Any]
) -> dict[str, Any]:
    """NumPy twin of ``probe_sps_gradient_role_conflict._gradient_stats``.

    Keys, flattening order, coactivity epsilon, scalar fractions, tensor-weighted
    negative fraction, and conflict thresholds match the standalone probe.
    """

    pred_names, pred = _flat_mapping(prediction)
    temp_names, temp = _flat_mapping(temporal)
    if pred_names != temp_names:
        raise ValueError(f"gradient role keys differ: {pred_names!r} != {temp_names!r}")
    for name in pred_names:
        if pred[name].shape != temp[name].shape:
            raise ValueError(
                f"gradient shape differs for {name!r}: {pred[name].shape} != {temp[name].shape}"
            )
    gp = np.concatenate([pred[name] for name in pred_names]) if pred_names else np.zeros(0)
    gt = np.concatenate([temp[name] for name in pred_names]) if pred_names else np.zeros(0)
    prediction_norm = float(np.linalg.norm(gp))
    temporal_norm = float(np.linalg.norm(gt))
    scale = max(float(np.max(np.abs(gp))) if gp.size else 0.0,
                float(np.max(np.abs(gt))) if gt.size else 0.0, 1.0)
    active_eps = 1e-12 * scale
    coactive = (np.abs(gp) > active_eps) & (np.abs(gt) > active_eps)
    negative = coactive & ((gp * gt) < 0.0)
    tensor_rows: list[dict[str, Any]] = []
    negative_tensor_params = 0
    active_tensor_params = 0
    for name in pred_names:
        pnorm = float(np.linalg.norm(pred[name]))
        tnorm = float(np.linalg.norm(temp[name]))
        cosine = _cosine(pred[name], temp[name])
        n = int(pred[name].size)
        active = pnorm > 0.0 and tnorm > 0.0
        if active:
            active_tensor_params += n
            if cosine is not None and cosine < 0.0:
                negative_tensor_params += n
        tensor_rows.append({
            "name": name,
            "numel": n,
            "cosine": cosine,
            "prediction_norm": pnorm,
            "temporal_norm": tnorm,
        })
    total = int(gp.size)
    coactive_count = int(np.count_nonzero(coactive))
    negative_count = int(np.count_nonzero(negative))
    global_cosine = _cosine(gp, gt)
    negative_all = negative_count / total if total else 0.0
    negative_coactive = negative_count / coactive_count if coactive_count else 0.0
    negative_tensor_fraction = (
        negative_tensor_params / active_tensor_params if active_tensor_params else 0.0
    )
    conflict = bool(
        global_cosine is not None
        and global_cosine <= SPS_MATERIAL_COSINE_THRESHOLD
        and negative_all >= SPS_MATERIAL_FRACTION_THRESHOLD
    )
    return {
        "global_cosine": global_cosine,
        "prediction_norm": prediction_norm,
        "temporal_norm": temporal_norm,
        "trunk_scalar_count": total,
        "coactive_scalar_count": coactive_count,
        "coactive_weight_fraction": coactive_count / total if total else 0.0,
        "negative_product_scalar_count": negative_count,
        "negative_product_weight_fraction_all": negative_all,
        "negative_product_weight_fraction_coactive": negative_coactive,
        "negative_cosine_tensor_weight_fraction": negative_tensor_fraction,
        "negative_tensor_weight_fraction": negative_tensor_fraction,
        "material_fraction_threshold_assumed": SPS_MATERIAL_FRACTION_THRESHOLD,
        "material_cosine_threshold_assumed": SPS_MATERIAL_COSINE_THRESHOLD,
        "conflict_exists_under_preregistered_rule": conflict,
        "conflict_exists": conflict,
        "per_tensor": tensor_rows,
    }


def sps_engagement_row(
    prediction: Mapping[str, Any],
    temporal: Mapping[str, Any],
    *,
    seg: Mapping[str, Any] | None = None,
    pose: Mapping[str, Any] | None = None,
    epoch: int,
    engagement: str,
    reason: str,
    actual_event: bool,
    pair_indices: Sequence[int],
    active_temporal_terms: Mapping[str, float],
) -> dict[str, Any]:
    if engagement not in {"temporal_screw_engaged", "phase_advection_engaged"}:
        raise ValueError(f"unknown SPS engagement {engagement!r}")
    combined_stats = gradient_role_conflict_stats(prediction, temporal)
    mechanism_stats = {
        "fully_armed_seg_plus_pose_vs_temporal": combined_stats,
    }
    if seg is not None:
        mechanism_stats["seg_vs_temporal"] = gradient_role_conflict_stats(seg, temporal)
    if pose is not None:
        mechanism_stats["pose_vs_temporal"] = gradient_role_conflict_stats(pose, temporal)
    # Match the standalone probe's primary verdict: seg-vs-temporal is the SPS
    # reformulation discriminator; the fully-armed and pose rows remain visible.
    primary = mechanism_stats.get("seg_vs_temporal", combined_stats)
    return {
        "schema": SPS_ENGAGEMENT_SCHEMA,
        "stage": "sps_gradient_role_conflict_engagement",
        "epoch": int(epoch),
        "engagement": engagement,
        "cadence_reason": str(reason),
        "actual_event": bool(actual_event),
        "pair_indices": [int(v) for v in pair_indices],
        "k_pairs": len(pair_indices),
        "active_temporal_terms": {
            str(name): float(weight) for name, weight in active_temporal_terms.items()
        },
        "primary_conflict": "seg_vs_temporal" if seg is not None else (
            "fully_armed_seg_plus_pose_vs_temporal"
        ),
        **primary,
        "gradient_conflict": mechanism_stats,
        "score_neutral": True,
        "promotable": False,
        "axis": "[macOS-MLX gradient observer] NON-PROMOTABLE",
        "verdict_scope": "engaged-regime mechanism diagnostic; not a score or family verdict",
    }


def engagement_reasons(
    epoch: int,
    *,
    nominal_epoch: int,
    window: int,
    actual_event: bool = False,
) -> tuple[str, ...]:
    ep = int(epoch)
    nominal = int(nominal_epoch)
    radius = max(0, int(window))
    if actual_event:
        # The actual transition is the higher-authority reason.  When it
        # coincides with the nominal window, emit one K-pair gradient probe,
        # not an identical second expensive probe for the same epoch.
        return ("actual_engagement_transition",)
    reasons: list[str] = []
    if abs(ep - nominal) <= radius:
        offset = ep - nominal
        reasons.append("nominal_boundary" if offset == 0 else f"nominal_boundary_offset_{offset:+d}")
    return tuple(dict.fromkeys(reasons))


@dataclass
class ProducerResumeState:
    """Canonical resume payload for nonduplicating D-B/Q2/Q6 emissions."""

    sps_emitted: set[str] = field(default_factory=set)
    ladder_emitted: set[str] = field(default_factory=set)
    event_mark_resume_keys: set[str] = field(default_factory=set)
    inert_streaks: dict[str, int] = field(default_factory=dict)

    def mark_sps(self, *, engagement: str, epoch: int, reason: str) -> bool:
        key = f"{engagement}:{int(epoch)}:{reason}"
        if key in self.sps_emitted:
            return False
        self.sps_emitted.add(key)
        return True

    def mark_ladder(self, name: str) -> bool:
        key = str(name)
        if key in self.ladder_emitted:
            return False
        self.ladder_emitted.add(key)
        return True

    def mark_event_resume_key(self, resume_key: str) -> bool:
        key = str(resume_key)
        if key in self.event_mark_resume_keys:
            return False
        self.event_mark_resume_keys.add(key)
        return True

    def state_arrays(self, prefix: str) -> dict[str, Any]:
        if not (
            self.sps_emitted
            or self.ladder_emitted
            or self.event_mark_resume_keys
            or self.inert_streaks
        ):
            return {}
        return {
            prefix + "sps_emitted_json": np.asarray(json.dumps(sorted(self.sps_emitted))),
            prefix + "ladder_emitted_json": np.asarray(json.dumps(sorted(self.ladder_emitted))),
            prefix + "event_mark_resume_keys_json": np.asarray(
                json.dumps(sorted(self.event_mark_resume_keys))
            ),
            prefix + "inert_streaks_json": np.asarray(json.dumps(self.inert_streaks, sort_keys=True)),
        }

    def restore_from_cfg(self, prefix: str, cfg: dict) -> bool:
        key = prefix + "sps_emitted_json"
        event_key = prefix + "event_mark_resume_keys_json"
        if key not in cfg and event_key not in cfg:
            return False

        def _load(name: str, default: str) -> Any:
            value = cfg.get(prefix + name, default)
            if hasattr(value, "item"):
                value = value.item()
            return json.loads(str(value))

        self.sps_emitted = {str(v) for v in _load("sps_emitted_json", "[]")}
        self.ladder_emitted = {str(v) for v in _load("ladder_emitted_json", "[]")}
        self.event_mark_resume_keys = {
            str(v) for v in _load("event_mark_resume_keys_json", "[]")
        }
        self.inert_streaks = {
            str(k): int(v) for k, v in _load("inert_streaks_json", "{}").items()
        }
        return True


def select_marked_event_family(
    detector_matches: Mapping[str, Sequence[str]],
) -> tuple[str, tuple[str, ...]]:
    """Apply D39's disjoint priority partition and preserve all matched detectors."""

    normalized: dict[str, tuple[str, ...]] = {}
    for family, detectors in detector_matches.items():
        if family not in EVENT_FAMILY_PRIORITY:
            raise ValueError(f"unknown marked-event family {family!r}")
        items = tuple(sorted({str(item) for item in detectors if str(item).strip()}))
        if items:
            normalized[str(family)] = items
    if not normalized:
        raise ValueError("marked event requires at least one matched detector")
    family = min(normalized, key=EVENT_FAMILY_PRIORITY.__getitem__)
    flattened = tuple(
        sorted(
            f"{matched_family}:{detector}"
            for matched_family, detectors in normalized.items()
            for detector in detectors
        )
    )
    return family, flattened


@dataclass
class MarkedEventTelemetryProducer:
    """Resume-safe D39 producer over the canonical causal-manifest writer."""

    writer: CausalManifestWriter
    state: ProducerResumeState

    def record(
        self,
        *,
        stage_id: str,
        checkpoint_id: str | None,
        pair_index: int,
        frame_from: int,
        frame_to: int,
        observed_at_utc: str,
        detector_matches: Mapping[str, Sequence[str]],
        kind_by_family: Mapping[str, str],
        class_edge: ClassEdgeMark,
        location: SpacetimeMark,
        attachment: IncidenceMark,
        stratum_before: StratumMark,
        stratum_after: StratumMark,
        receiver_state: ReceiverStateMark,
        evidence: Sequence[ArtifactRef],
        receiver_derivable: bool,
        public_derivation_ref: ArtifactRef | None,
        notes: Mapping[str, Any] | None = None,
    ) -> tuple[EventMarkRow, bool]:
        family, detectors = select_marked_event_family(detector_matches)
        if family not in kind_by_family:
            raise ValueError(f"missing kind for selected marked-event family {family!r}")
        row = EventMarkRow.build(
            run_id=self.writer.run_id,
            stage_id=stage_id,
            checkpoint_id=checkpoint_id,
            pair_index=pair_index,
            frame_from=frame_from,
            frame_to=frame_to,
            observed_at_utc=observed_at_utc,
            family=family,
            kind=str(kind_by_family[family]),
            detectors_matched=detectors,
            class_edge=class_edge,
            location=location,
            attachment=attachment,
            stratum_before=stratum_before,
            stratum_after=stratum_after,
            receiver_state=receiver_state,
            evidence=evidence,
            receiver_derivable=receiver_derivable,
            public_derivation_ref=public_derivation_ref,
            notes=notes,
        )
        appended = self.writer.record_event_mark(row)
        # Only advance the checkpoint cursor after the locked append/no-op has
        # proved the immutable row exists with identical bytes.
        self.state.mark_event_resume_key(row.resume_key)
        return row, appended


@dataclass
class ClipActivationAggregator:
    threshold: float
    _global: list[float] = field(default_factory=list)
    _groups: dict[str, list[float]] = field(default_factory=dict)

    def observe(self, global_norm: float, group_norms: Mapping[str, float]) -> None:
        self._global.append(_finite_nonnegative(global_norm, name="global_norm"))
        for name, value in group_norms.items():
            self._groups.setdefault(str(name), []).append(
                _finite_nonnegative(value, name=f"group_norm[{name}]")
            )

    def _summary(self, values: Sequence[float]) -> dict[str, Any]:
        if not values:
            return {"n": 0, "frac_clipped": 0.0, "norm_mean": None, "norm_max": None}
        arr = np.asarray(values, dtype=np.float64)
        return {
            "n": int(arr.size),
            "frac_clipped": float(np.mean(arr > float(self.threshold))),
            "norm_mean": float(np.mean(arr)),
            "norm_max": float(np.max(arr)),
        }

    def row(self, *, epoch: int) -> dict[str, Any]:
        return {
            "stage": "grad_clip_activation",
            "epoch": int(epoch),
            "ep": int(epoch),
            "threshold": float(self.threshold),
            "global": self._summary(self._global),
            "per_group": {name: self._summary(values) for name, values in sorted(self._groups.items())},
            "score_neutral": True,
        }


def term_inert_rows(
    terms: Mapping[str, float],
    *,
    engaged: Mapping[str, bool],
    epoch: int,
    state: ProducerResumeState,
    share_threshold: float = 1e-6,
    sustained_rows: int = 3,
) -> list[dict[str, Any]]:
    total = sum(abs(float(value)) for value in terms.values()) + 1e-30
    rows: list[dict[str, Any]] = []
    for name, is_engaged in engaged.items():
        share = abs(float(terms.get(name, 0.0))) / total
        streak = state.inert_streaks.get(name, 0)
        streak = streak + 1 if is_engaged and share < float(share_threshold) else 0
        state.inert_streaks[name] = streak
        if streak == int(sustained_rows):
            rows.append({
                "stage": "term_inert",
                "alarm": "term_inert",
                "epoch": int(epoch),
                "term": str(name),
                "post_weight_share": float(share),
                "share_threshold": float(share_threshold),
                "sustained_rows": int(streak),
                "score_neutral": True,
                "verdict_scope": "engaged loss-term telemetry only",
            })
    return rows


def live_gap_fields(ema_verdict: Mapping[str, float], live_verdict: Mapping[str, float]) -> dict[str, float]:
    return {
        "d_seg_live": float(live_verdict["d_seg"]),
        "d_pose_live": float(live_verdict["d_pose"]),
        "d_seg_ema_minus_live": float(ema_verdict["d_seg"]) - float(live_verdict["d_seg"]),
        "d_pose_ema_minus_live": float(ema_verdict["d_pose"]) - float(live_verdict["d_pose"]),
    }


def tail_cycle_endpoint_row(
    verdict: Mapping[str, Any] | None,
    *,
    epoch: int,
    cycle: int,
    start_epoch: int,
    boundary_reason: str,
) -> dict[str, Any] | None:
    if verdict is None or "d_seg" not in verdict or "d_pose" not in verdict:
        return None
    return {
        "stage": "tail_cycle_endpoint",
        "epoch": int(epoch),
        "cycle": int(cycle),
        "cycle_start_epoch": int(start_epoch),
        "cycle_end_epoch": int(epoch),
        "boundary_reason": str(boundary_reason),
        "verdict_epoch": int(verdict.get("epoch", -1)),
        "d_seg": float(verdict["d_seg"]),
        "d_pose": float(verdict["d_pose"]),
        "implied_S": (float(verdict["implied_S"]) if "implied_S" in verdict else None),
        "score_neutral": True,
    }


def would_fire_row(
    *,
    epoch: int,
    lever: str,
    metric: str,
    value: float | None,
    threshold: Any,
    dwell: int,
    sensor_data_epoch: int | None,
    event_mode: bool,
    fired: bool,
) -> dict[str, Any]:
    return {
        "stage": "lever_would_fire",
        "epoch": int(epoch),
        "lever": str(lever),
        "metric": str(metric),
        "value": None if value is None else float(value),
        "threshold": threshold,
        "dwell": int(dwell),
        "sensor_data_epoch": None if sensor_data_epoch is None else int(sensor_data_epoch),
        "event_mode": bool(event_mode),
        "status": "fired" if fired else "held",
        "score_neutral": True,
        "verdict_scope": "would-fire sensor telemetry; not a lever-family verdict",
    }


def ladder_birth_complete_row(
    *, epoch: int, class_id: int, class_name: str, final_radius: float
) -> dict[str, Any]:
    return {
        "stage": "ladder_birth_complete",
        "epoch": int(epoch),
        "class_id": int(class_id),
        "class_name": str(class_name),
        "r_final": float(final_radius),
        "score_neutral": True,
    }


def lever_engage_row(
    lever: str, *, status: str, epoch: int, via: str
) -> dict[str, Any]:
    if status not in {"armed", "fired", "complete"}:
        raise ValueError(f"invalid lever engagement status {status!r}")
    return {
        "stage": "lever_engage",
        "lever": str(lever),
        "status": status,
        "epoch": int(epoch),
        "via": str(via),
    }


__all__ = [
    "COMPONENT_FIELDS",
    "COMPONENT_WALLCLOCK_SCHEMA",
    "REAL_PATH_FIELDS",
    "SPS_ENGAGEMENT_SCHEMA",
    "SPS_MATERIAL_COSINE_THRESHOLD",
    "SPS_MATERIAL_FRACTION_THRESHOLD",
    "ClipActivationAggregator",
    "ComponentWallclock",
    "MarkedEventTelemetryProducer",
    "ProducerResumeState",
    "deterministic_strata",
    "engagement_reasons",
    "gradient_role_conflict_stats",
    "ladder_birth_complete_row",
    "lever_engage_row",
    "live_gap_fields",
    "select_marked_event_family",
    "sps_engagement_row",
    "tail_cycle_endpoint_row",
    "term_inert_rows",
    "would_fire_row",
]
