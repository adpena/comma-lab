# SPDX-License-Identifier: MIT
"""L4 survival + hysteresis producers for the HiNeRV target-region birth actuator.

The launch gate (``tac.analysis.nerv_long_run_launch_gate``) DEFINES the
survival/hysteresis schemas before any producer existed; this module is the
first conforming producer.  It re-measures one ALREADY-ACCEPTED live birth on a
second surface and emits the typed rows the gate consumes, carrying the live
receipt's ``action_id`` verbatim so the gate can trace one action identity
across surfaces (a re-solved birth under fake-quant would be a different
experiment and is explicitly NOT what this does).

Two surfaces are produced honestly here:

* ``fakequant_mlx`` - flips the renderer's REAL decoder-weight fake-quant
  forward on (via its existing ``configure_decoder_fake_quant_forward`` API,
  restoring the prior config afterwards), forwards the SAME pairs, and
  recomputes the region's target support on the quantized receiver image.
* hysteresis - runs M further steps of the EXISTING plain scorer-domain
  bootstrap (``fit_scorer_domain_bootstrap_from_targets`` with the hard-birth
  weight pinned to zero) to simulate continued ordinary training, then
  re-measures the same region.

The ``parseback_mlx`` and ``inflated_torch_cpu`` surfaces require an archive
build / inflate runtime that this scoped producer cannot stand up faithfully;
rather than fabricate a ``survived`` boolean, it returns a typed BLOCKED row
(``hi_nerv_target_region_birth_survival_blocked.v1``) naming the blocker, so the
gate keeps blocking those surfaces.  Honesty over coverage.

NO-FAKE discipline: every ``survived`` boolean is the result of a real forward
recomputation of region target support on the named surface.  If this module
were replaced by a canonical-marker stub returning ``survived=True`` constants,
the behavioral tests (which compare against a real fake-quant forward, prove the
fake-quant config is restored, and feed the rows to the real gate) would fail.

Rows travel inside ``substrate_artifact_metadata`` whose custody validator
refuses nested authority/readiness keys; every row carries the non-authority
marker and NONE of ``score_claim`` / ``promotion_eligible`` /
``ready_for_exact_eval_dispatch`` / ``rank_or_kill_eligible`` / ``promotable``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy import ndimage

from tac.substrates.hi_nerv.target_region_birth import region_margin_stats

# Mirror the gate's canonical schema strings exactly (the gate is the contract).
BIRTH_SURVIVAL_SCHEMA = "hi_nerv_target_region_birth_survival.v1"
BIRTH_HYSTERESIS_SCHEMA = "hi_nerv_target_region_birth_hysteresis.v1"
BIRTH_SURVIVAL_BLOCKED_SCHEMA = "hi_nerv_target_region_birth_survival_blocked.v1"

# Surfaces this producer can re-measure faithfully without an archive/inflate
# runtime.  ``fakequant_mlx`` is the only L4 surface a scoped MLX producer can
# stand up honestly; the other two need real packet bytes / a torch inflate.
FAITHFUL_SURVIVAL_SURFACES = ("fakequant_mlx",)
BLOCKED_SURVIVAL_SURFACES = ("parseback_mlx", "inflated_torch_cpu")
_SURVIVAL_SURFACE_BLOCKER = {
    "parseback_mlx": "birth_survival_parseback_requires_archive_parse_back_runtime",
    "inflated_torch_cpu": "birth_survival_inflate_requires_torch_inflate_runtime",
}

# The non-authority marker the receipt builder uses; survival rows are planning
# control evidence, never a score claim.
_NON_AUTHORITY_MARKER = "planning_control_false_authority"


class BirthSurvivalError(ValueError):
    """Raised when survival-producer inputs are structurally invalid."""


def _require_mlx() -> None:
    import importlib.util

    if importlib.util.find_spec("mlx") is None:  # pragma: no cover - env guard
        raise BirthSurvivalError("MLX is required for HiNeRV birth-survival re-measurement")


def _coerce_labels_bhw(target_labels: Any) -> np.ndarray:
    labels = np.asarray(target_labels)
    if labels.ndim == 4 and labels.shape[-1] == 1:
        labels = labels[..., 0]
    if labels.ndim != 3:
        raise BirthSurvivalError(f"target_labels must be BHW (or BHW1); got shape {labels.shape}")
    if labels.size == 0:
        raise BirthSurvivalError("target_labels must be non-empty")
    return labels.astype(np.int64, copy=False)


def reconstruct_birth_region_mask(
    target_labels: Any,
    worst_region: Mapping[str, Any],
) -> tuple[np.ndarray, int]:
    """Deterministically rebuild the named birth region's BHW float32 mask.

    Reconstruction uses ONLY the labels and the live receipt's named region
    coordinates (``batch_index`` / ``class_index`` / ``region_label``) plus the
    same scipy 4-connected ``ndimage.label`` the selector used, so the mask is
    byte-identical to the region the live birth named.  The reconstructed pixel
    count is verified against the receipt's ``region_pixel_count`` to refuse a
    drifted region (mismatched labels for a different live run).
    """

    labels = _coerce_labels_bhw(target_labels)
    try:
        batch = int(worst_region["batch_index"])
        cls = int(worst_region["class_index"])
        region_label = int(worst_region["region_label"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BirthSurvivalError(f"worst_region missing batch/class/region fields: {worst_region!r}") from exc
    if not 0 <= batch < int(labels.shape[0]):
        raise BirthSurvivalError(f"worst_region batch_index {batch} outside labels batch {int(labels.shape[0])}")
    frame = labels[batch]
    labeled, _count = ndimage.label(frame == cls)
    mask = np.zeros(labels.shape, dtype=np.float32)
    mask[batch] = (labeled == region_label).astype(np.float32)
    region_pixels = int(np.count_nonzero(mask))
    expected = worst_region.get("region_pixel_count")
    if expected is not None and region_pixels != int(expected):
        raise BirthSurvivalError(
            "reconstructed birth region drifted from the live receipt; "
            f"labels={region_pixels} receipt={int(expected)} "
            "(target_labels do not match the live birth run)"
        )
    if region_pixels == 0:
        raise BirthSurvivalError("reconstructed birth region selects zero pixels")
    return mask, region_pixels


def _initial_in_region_target_count(worst_region: Mapping[str, Any]) -> int:
    """Region pixels that already won the target class at the live before-state.

    The live receipt does not persist per-pixel initial argmax, but it records
    the exact integer counts: a region with ``region_unsolved_pixel_count`` of
    its ``region_pixel_count`` pixels wrong had ``count - unsolved`` pixels
    already winning the target class.  This is the exact before-count the net
    target-support delta is measured against (the gate needs the count, not the
    spatial layout).
    """

    total = int(worst_region["region_pixel_count"])
    unsolved = int(worst_region["region_unsolved_pixel_count"])
    return max(0, total - unsolved)


def _live_action_id(live_birth_payload: Mapping[str, Any]) -> str:
    action_id = live_birth_payload.get("action_id")
    if not action_id:
        # Some payloads carry the id only inside the nested receipt.
        receipt = live_birth_payload.get("receipt")
        if isinstance(receipt, Mapping):
            action_id = receipt.get("action_id")
    if not action_id:
        raise BirthSurvivalError("live_birth_payload missing action_id; survival cannot trace one action identity")
    return str(action_id)


def _live_worst_region(live_birth_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    worst = live_birth_payload.get("worst_region")
    if not isinstance(worst, Mapping):
        raise BirthSurvivalError("live_birth_payload missing worst_region; cannot reconstruct the birth region")
    return worst


def _predict_frame1_nhwc01(model: Any, pair_indices: Any) -> Any:
    """Return frame-1 NHWC in [0, 1] from the renderer for the given pairs."""

    import mlx.core as mx

    pair01 = model(pair_indices) / 255.0
    return mx.transpose(pair01[:, 1], (0, 2, 3, 1))  # type: ignore[union-attr]


def _candidate_logits_np(scorer_teacher: Any, frame1_nhwc01: Any) -> np.ndarray:
    """Receiver-faithful SegNet logits: clamp/round uint8 roundtrip then teacher."""

    import mlx.core as mx

    from tac.substrates.hi_nerv.mlx_renderer import _receiver_uint8_roundtrip_ste_nhwc01

    live_fn = getattr(scorer_teacher, "teacher_logits_for_frames_nhwc01", None)
    if not callable(live_fn):
        raise BirthSurvivalError(
            "scorer_teacher must expose teacher_logits_for_frames_nhwc01 for survival re-measurement"
        )
    logits = live_fn(_receiver_uint8_roundtrip_ste_nhwc01(frame1_nhwc01))
    mx.eval(logits)  # type: ignore[union-attr]
    out = np.asarray(logits, dtype=np.float32)
    if out.ndim != 4:
        raise BirthSurvivalError(f"live SegNet candidate logits must be BHWC; got {out.shape}")
    return out


def _region_support_on_surface(
    model: Any,
    *,
    scorer_teacher: Any,
    pair_indices: Any,
    region_mask_np: np.ndarray,
    birth_class: int,
) -> dict[str, Any]:
    """Forward the model and measure the region's target support on this surface."""

    frame1 = _predict_frame1_nhwc01(model, pair_indices)
    logits_np = _candidate_logits_np(scorer_teacher, frame1)
    class_count = int(logits_np.shape[-1])
    if not 0 <= int(birth_class) < class_count:
        raise BirthSurvivalError(f"birth class {birth_class} outside live logits classes {class_count}")
    stats = region_margin_stats(logits_np, region_mask_np, int(birth_class))
    region_pixels = int(stats["region_pixel_count"])
    region_hard_won = int(stats["region_hard_won_pixels"])
    return {
        "stats": stats,
        "region_pixel_count": region_pixels,
        "region_hard_won": region_hard_won,
        "region_unsolved": int(region_pixels - region_hard_won),
        # exact contest-units debt for THIS region under the surface:
        # 100 * unsolved / total_scored_pixels (batch-local normalizer).
        "total_scored_pixels": int(logits_np.size // class_count),
    }


def _region_debt_units(region_unsolved: int, total_scored_pixels: int) -> float:
    if total_scored_pixels <= 0:
        return 0.0
    return float(100.0 * int(region_unsolved) / int(total_scored_pixels))


def _fake_quant_config_snapshot(model: Any) -> dict[str, Any]:
    return {
        "configured_enabled": bool(getattr(model, "decoder_fake_quant_forward_configured_enabled", False)),
        "stage_controlled": bool(getattr(model, "decoder_fake_quant_forward_stage_controlled", False)),
        "stage_qat_active": bool(getattr(model, "decoder_fake_quant_forward_stage_qat_active", True)),
        "bits": getattr(model, "decoder_fake_quant_bits", None),
        "bits_by_name": dict(getattr(model, "decoder_fake_quant_bits_by_name", {}) or {}),
        "forward_enabled": bool(getattr(model, "decoder_fake_quant_forward_enabled", False)),
    }


def _restore_fake_quant_config(model: Any, snapshot: Mapping[str, Any]) -> None:
    # Re-apply the configured geometry exactly, then restore the live flag/stage
    # bits so the model is bit-identical in behavior to before the survival call.
    model.configure_decoder_fake_quant_forward(
        enabled=bool(snapshot["configured_enabled"]),
        quant_bits=snapshot["bits"],
        per_tensor_bits=dict(snapshot["bits_by_name"]),
        stage_controlled=bool(snapshot["stage_controlled"]),
    )
    model.decoder_fake_quant_forward_stage_qat_active = bool(snapshot["stage_qat_active"])
    model.decoder_fake_quant_forward_enabled = bool(snapshot["forward_enabled"])


def measure_birth_survival(
    model: Any,
    *,
    scorer_teacher: Any,
    target_labels: Any,
    live_birth_payload: Mapping[str, Any],
    surface: str,
    pair_indices: Any,
    fakequant_bits: int = 8,
) -> dict[str, Any]:
    """Re-measure one accepted live birth on ``surface`` and emit a typed row.

    For ``surface="fakequant_mlx"`` the renderer's REAL decoder-weight fake-quant
    forward is enabled, the SAME pairs are forwarded, the named birth region is
    reconstructed deterministically, and its target support is recomputed on the
    quantized receiver image.  ``survived`` is ``True`` only when the region's
    ``region_hard_won`` under fake-quant is at least one pixel AND the net target
    support (won-under-fakequant minus the live before-count) is still positive.
    The fake-quant config is restored before returning.  ``action_id`` is COPIED
    from the live payload (never recomputed).

    For ``parseback_mlx`` / ``inflated_torch_cpu`` a faithful cheap path is not
    achievable in this scoped producer, so a typed BLOCKED row is returned with
    no ``survived`` boolean (the gate keeps blocking that surface).
    """

    surface_name = str(surface)
    action_id = _live_action_id(live_birth_payload)
    if surface_name in BLOCKED_SURVIVAL_SURFACES:
        return _blocked_survival_row(action_id=action_id, surface=surface_name)
    if surface_name not in FAITHFUL_SURVIVAL_SURFACES:
        raise BirthSurvivalError(
            f"surface must be one of {FAITHFUL_SURVIVAL_SURFACES + BLOCKED_SURVIVAL_SURFACES}; got {surface_name!r}"
        )

    _require_mlx()
    import mlx.core as mx

    bits = int(fakequant_bits)
    if bits < 1 or bits > 16:
        raise BirthSurvivalError(f"fakequant_bits must be in [1, 16]; got {fakequant_bits}")

    worst_region = _live_worst_region(live_birth_payload)
    birth_class = int(worst_region["class_index"])
    region_mask_np, region_pixels = reconstruct_birth_region_mask(target_labels, worst_region)
    initial_in_region_target = _initial_in_region_target_count(worst_region)

    idx = mx.array(pair_indices, dtype=mx.int32)  # type: ignore[union-attr]
    if idx.ndim != 1 or int(idx.shape[0]) <= 0:
        raise BirthSurvivalError("pair_indices must be a non-empty rank-1 tensor")

    fq_snapshot = _fake_quant_config_snapshot(model)
    try:
        model.configure_decoder_fake_quant_forward(
            enabled=True,
            quant_bits=bits,
            per_tensor_bits=None,
            stage_controlled=False,
        )
        if not bool(getattr(model, "decoder_fake_quant_forward_enabled", False)):
            raise BirthSurvivalError("fake-quant forward did not enable; survival surface would be unfaithful")
        support = _region_support_on_surface(
            model,
            scorer_teacher=scorer_teacher,
            pair_indices=idx,
            region_mask_np=region_mask_np,
            birth_class=birth_class,
        )
    finally:
        _restore_fake_quant_config(model, fq_snapshot)

    region_hard_won = int(support["region_hard_won"])
    net_target_support_delta = int(region_hard_won - initial_in_region_target)
    survived = bool(region_hard_won >= 1 and net_target_support_delta > 0)
    region_debt_units = _region_debt_units(int(support["region_unsolved"]), int(support["total_scored_pixels"]))

    return _survival_row(
        action_id=action_id,
        surface=surface_name,
        survived=survived,
        birth_class=birth_class,
        region_pixels=region_pixels,
        region_hard_won=region_hard_won,
        target_hard_lost=max(0, initial_in_region_target - region_hard_won),
        net_target_support_delta=net_target_support_delta,
        initial_in_region_target=initial_in_region_target,
        region_unsolved=int(support["region_unsolved"]),
        region_debt_units=region_debt_units,
        margin_stats=support["stats"],
        fakequant_bits=bits,
        total_scored_pixels=int(support["total_scored_pixels"]),
        worst_region=worst_region,
    )


def measure_birth_hysteresis(
    model: Any,
    *,
    scorer_teacher: Any,
    target_rgb_0: Any,
    target_rgb_1: Any,
    target_labels: Any,
    live_birth_payload: Mapping[str, Any],
    pair_indices: Any,
    extra_steps: int = 3,
    learning_rate: float = 2.0e-3,
) -> dict[str, Any]:
    """Run M more plain training steps then re-measure the birth region.

    Continued training is simulated with the EXISTING plain scorer-domain
    bootstrap (``fit_scorer_domain_bootstrap_from_targets``) with the hard-birth
    weight pinned to zero — i.e. ordinary RGB/YUV6/temporal/contrast objectives,
    NOT another birth solve.  ``passed`` is ``True`` only when the region's
    ``region_hard_won`` has NOT collapsed below 50% of the t0 (pre-extra-steps)
    value AND the region's debt at t+M is not above the pre-birth debt.
    """

    steps = int(extra_steps)
    if steps <= 0:
        raise BirthSurvivalError(f"extra_steps must be positive; got {extra_steps}")
    _require_mlx()
    import mlx.core as mx

    action_id = _live_action_id(live_birth_payload)
    worst_region = _live_worst_region(live_birth_payload)
    birth_class = int(worst_region["class_index"])
    region_mask_np, region_pixels = reconstruct_birth_region_mask(target_labels, worst_region)

    idx = mx.array(pair_indices, dtype=mx.int32)  # type: ignore[union-attr]
    if idx.ndim != 1 or int(idx.shape[0]) <= 0:
        raise BirthSurvivalError("pair_indices must be a non-empty rank-1 tensor")
    labels_mx = mx.array(_coerce_labels_bhw(target_labels), dtype=mx.int32)  # type: ignore[union-attr]

    # Pre-birth debt anchor: 100 * region_unsolved_at_birth / total_scored.
    # worst_region records the live-surface total_scored_pixels normalizer.
    pre_birth_total_scored = int(worst_region.get("total_scored_pixels") or 0)
    pre_birth_debt = _region_debt_units(
        int(worst_region["region_unsolved_pixel_count"]),
        pre_birth_total_scored,
    )

    # t0: region support on the live float surface right now (post-birth).
    support_t0 = _region_support_on_surface(
        model,
        scorer_teacher=scorer_teacher,
        pair_indices=idx,
        region_mask_np=region_mask_np,
        birth_class=birth_class,
    )
    hard_won_t0 = int(support_t0["region_hard_won"])
    debt_t0 = _region_debt_units(int(support_t0["region_unsolved"]), int(support_t0["total_scored_pixels"]))

    # Continued ordinary training: hard-birth weight pinned to 0 (no re-solve).
    bootstrap = model.fit_scorer_domain_bootstrap_from_targets(
        target_rgb_0,
        target_rgb_1,
        pair_indices=idx,
        target_segnet_argmax_1=labels_mx,
        scorer_teacher=scorer_teacher,
        segnet_hard_birth_bootstrap_weight=0.0,
        steps=steps,
        learning_rate=float(learning_rate),
    )

    # t+M: region support after the extra steps.
    support_tm = _region_support_on_surface(
        model,
        scorer_teacher=scorer_teacher,
        pair_indices=idx,
        region_mask_np=region_mask_np,
        birth_class=birth_class,
    )
    hard_won_tm = int(support_tm["region_hard_won"])
    debt_tm = _region_debt_units(int(support_tm["region_unsolved"]), int(support_tm["total_scored_pixels"]))

    no_collapse = bool(float(hard_won_tm) >= 0.5 * float(hard_won_t0))
    debt_not_above_pre_birth = bool(debt_tm <= pre_birth_debt + 1.0e-12)
    passed = bool(no_collapse and debt_not_above_pre_birth)

    row: dict[str, Any] = {
        "schema": BIRTH_HYSTERESIS_SCHEMA,
        "action_id": action_id,
        "surface": "live_mlx_continued",
        "passed": passed,
        "extra_steps": steps,
        "birth_class_index": birth_class,
        "region_pixel_count": region_pixels,
        # t0 / t+M hard-won and debt, exactly as the gate's hysteresis schema asks.
        "hard_won_t0": hard_won_t0,
        "hard_won_tm": hard_won_tm,
        "debt_t0": float(debt_t0),
        "debt_tm": float(debt_tm),
        "pre_birth_debt": float(pre_birth_debt),
        "margin_mean_t0": float(support_t0["stats"]["margin_mean"]),
        "margin_mean_tm": float(support_tm["stats"]["margin_mean"]),
        "no_hard_won_collapse": no_collapse,
        "debt_not_above_pre_birth": debt_not_above_pre_birth,
        "bootstrap_accepted_step_count": int(bootstrap.get("accepted_step_count") or 0),
        "bootstrap_schema": str(bootstrap.get("schema") or ""),
        "human_visual_fidelity_objective": False,
        "authority": _NON_AUTHORITY_MARKER,
    }
    return row


def _survival_row(
    *,
    action_id: str,
    surface: str,
    survived: bool,
    birth_class: int,
    region_pixels: int,
    region_hard_won: int,
    target_hard_lost: int,
    net_target_support_delta: int,
    initial_in_region_target: int,
    region_unsolved: int,
    region_debt_units: float,
    margin_stats: Mapping[str, float],
    fakequant_bits: int,
    total_scored_pixels: int,
    worst_region: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the survival row in the gate's canonical schema.

    The gate prices target support via ``receiver_surface_target_support_breakdown``,
    which reads ``target_hard_won_count`` / ``net_target_support_delta`` (and
    their ``receiver_surface_*`` canonical aliases) both at the row top level and
    inside a nested ``argmax_transitions`` mapping.  Both are populated so the
    gate sees a positive, re-measured support without producer-specific adapters.
    """

    transitions = {
        "target_hard_won_count": int(region_hard_won),
        "target_hard_lost_count": int(target_hard_lost),
        "net_target_support_delta": int(net_target_support_delta),
        "wrong_to_target_count": int(region_hard_won),
        "target_to_wrong_count": int(target_hard_lost),
    }
    row: dict[str, Any] = {
        "schema": BIRTH_SURVIVAL_SCHEMA,
        "action_id": str(action_id),
        "surface": str(surface),
        "survived": bool(survived),
        "family": "hinerv",
        "birth_class_index": int(birth_class),
        "region_pixel_count": int(region_pixels),
        "region_hard_won_count": int(region_hard_won),
        "region_unsolved_pixel_count": int(region_unsolved),
        "initial_in_region_target_count": int(initial_in_region_target),
        "region_score_debt_units": float(region_debt_units),
        "region_margin_mean": float(margin_stats["margin_mean"]),
        "region_margin_min": float(margin_stats["margin_min"]),
        "region_hard_ratio": float(margin_stats["region_hard_ratio"]),
        "fakequant_bits": int(fakequant_bits),
        "total_scored_pixels": int(total_scored_pixels),
        # Canonical receiver-surface aliases the gate's support breakdown reads.
        "receiver_surface_target_hard_won_count": int(region_hard_won),
        "receiver_surface_target_hard_lost_count": int(target_hard_lost),
        "receiver_surface_net_target_support_delta": int(net_target_support_delta),
        # Top-level transition aliases (the gate merges these too).
        **transitions,
        "argmax_transitions": transitions,
        "worst_region": dict(worst_region),
        "human_visual_fidelity_objective": False,
        "authority": _NON_AUTHORITY_MARKER,
    }
    return row


def _blocked_survival_row(*, action_id: str, surface: str) -> dict[str, Any]:
    """Typed FAIL-CLOSED row for surfaces this producer cannot stand up.

    Deliberately carries NO ``survived`` boolean: the gate finds no satisfying
    survival row for the surface and keeps emitting
    ``birth_survival_receipt_missing:<surface>``.  Honesty over coverage — a
    fabricated ``survived=True`` here would be a fake implementation.
    """

    blocker = _SURVIVAL_SURFACE_BLOCKER.get(str(surface), "birth_survival_surface_unsupported")
    return {
        "schema": BIRTH_SURVIVAL_BLOCKED_SCHEMA,
        "action_id": str(action_id),
        "surface": str(surface),
        "blocked": True,
        "blocker": blocker,
        "reason": (
            "scoped MLX survival producer cannot build a faithful "
            f"{surface} surface (no archive parse-back / torch inflate runtime here)"
        ),
        "human_visual_fidelity_objective": False,
        "authority": _NON_AUTHORITY_MARKER,
    }


__all__ = [
    "BIRTH_HYSTERESIS_SCHEMA",
    "BIRTH_SURVIVAL_BLOCKED_SCHEMA",
    "BIRTH_SURVIVAL_SCHEMA",
    "BLOCKED_SURVIVAL_SURFACES",
    "FAITHFUL_SURVIVAL_SURFACES",
    "BirthSurvivalError",
    "measure_birth_hysteresis",
    "measure_birth_survival",
    "reconstruct_birth_region_mask",
]
