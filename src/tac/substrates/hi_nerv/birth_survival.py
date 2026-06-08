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

Calling ``measure_birth_survival(..., surface="parseback_mlx")`` still returns
a typed BLOCKED row unless a real parse-back artifact is supplied.  The bounded
parse-back path is ``measure_birth_parseback_survival_from_report``: it consumes
an existing receiver-cache/parse-back report, maps the live receipt's pair ids,
and re-measures the named region by rendering the charged archive bytes through
the receiver.  If the report names a different action id, or says it cached a
pair set that excludes the birth pair, it returns a typed BLOCKED row with no
``survived`` boolean.  Honesty over coverage.

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

import base64
import hashlib
import json
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from tac.contest_eval_contract import CAMERA_SIZE_WH, RGB_CHANNELS, SEQ_LEN
from tac.submission_archive import MINIMAL_SINGLE_MEMBER_NAME
from tac.substrates.hi_nerv.archive import hiv1_quantize_dequantize_for_training
from tac.substrates.hi_nerv.target_region_birth import (
    lset_subset_conditioned_margin_certificate,
    region_margin_stats,
)

# Mirror the gate's canonical schema strings exactly (the gate is the contract).
BIRTH_SURVIVAL_SCHEMA = "hi_nerv_target_region_birth_survival.v1"
BIRTH_HYSTERESIS_SCHEMA = "hi_nerv_target_region_birth_hysteresis.v1"
BIRTH_SURVIVAL_BLOCKED_SCHEMA = "hi_nerv_target_region_birth_survival_blocked.v1"
DECODER_SECTION_SHADOW_ABLATION_SCHEMA = "hi_nerv_decoder_section_shadow_ablation.v1"
TARGET_MARGIN_CERTIFICATE_SCHEMA = "tac.target_margin_certificate.v1"
MIN_SCORER_EFFECT_RETENTION_FOR_SURVIVAL = 0.5
TARGET_MARGIN_SAFETY_FLOOR = 0.0
# Decoder-weight sections the shadow ablation can swap to their archive-decoded
# value (via the round-trip-identity-proven import_torch_state_dict).  Latent
# sections are covered by archive_roundtrip_shadow; these are the conv/proj/head
# modules nearest the SegNet decision surface.
DECODER_SHADOW_SECTIONS = (
    "head_rgb_1",
    "head_rgb_0",
    "fine_injector",
    "mid_injector",
    "feature_grids",
    "convnext_blocks",
    "blocks",
    "latent_embed",
)

# Surfaces this producer can re-measure faithfully without an archive/inflate
# runtime.  ``fakequant_mlx`` is the only L4 surface a scoped MLX producer can
# stand up honestly; the other two need real packet bytes / a torch inflate.
FAITHFUL_SURVIVAL_SURFACES = ("fakequant_mlx", "archive_roundtrip_shadow")
BLOCKED_SURVIVAL_SURFACES = ("parseback_mlx", "inflated_torch_cpu")
# The HIV1 latent quantizer mirror only supports latent sections (decoder-weight
# sections are covered by the fakequant surface).  ``archive_roundtrip_shadow``
# routes these named latent attributes through the exact int16 archive
# encode/decode forward, keeping decoder weights live, so a collapse here
# attributes the parse-back scorer-effect loss to the LATENT quantizer.
DEFAULT_ARCHIVE_ROUNDTRIP_LATENT_SECTIONS = ("latents_fine",)
DEFAULT_ARCHIVE_ROUNDTRIP_LATENT_CODEC = "int16_raw"
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
    packed_mask = worst_region.get("region_mask_bhw_packbits")
    if isinstance(packed_mask, Mapping):
        try:
            shape = tuple(int(v) for v in packed_mask["shape"])
            true_count = int(packed_mask["true_count"])
            data_b64 = str(packed_mask["data_b64"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BirthSurvivalError(
                f"invalid region_mask_bhw_packbits payload: {packed_mask!r}"
            ) from exc
        if shape != tuple(int(v) for v in labels.shape):
            raise BirthSurvivalError(
                "packed birth region shape drifted from target_labels; "
                f"mask={shape} labels={tuple(int(v) for v in labels.shape)}"
            )
        try:
            packed = np.frombuffer(base64.b64decode(data_b64), dtype=np.uint8)
        except Exception as exc:  # pragma: no cover - defensive bad-base64 path.
            raise BirthSurvivalError("packed birth region mask is not valid base64") from exc
        flat_count = int(np.prod(shape))
        unpacked = np.unpackbits(packed, bitorder="big", count=flat_count).astype(bool)
        mask = unpacked.reshape(shape).astype(np.float32)
        region_pixels = int(np.count_nonzero(mask))
        expected = worst_region.get("region_pixel_count")
        if expected is not None and region_pixels != int(expected):
            raise BirthSurvivalError(
                "packed birth region drifted from the live receipt; "
                f"mask={region_pixels} receipt={int(expected)}"
            )
        if region_pixels != true_count:
            raise BirthSurvivalError(
                "packed birth region true_count drifted from decoded mask; "
                f"decoded={region_pixels} payload={true_count}"
            )
        if region_pixels == 0:
            raise BirthSurvivalError("packed birth region selects zero pixels")
        return mask, region_pixels

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


def _first_int_from_mappings(*nodes: Mapping[str, Any], keys: tuple[str, ...]) -> int | None:
    for node in nodes:
        if not isinstance(node, Mapping):
            continue
        for key in keys:
            value = node.get(key)
            try:
                if value is not None:
                    return int(value)
            except (TypeError, ValueError):
                continue
    return None


def _live_wrong_to_target_count(live_birth_payload: Mapping[str, Any]) -> int | None:
    """Return the live accepted scorer-effect magnitude for retention gates."""

    receipt = live_birth_payload.get("receipt")
    receipt = receipt if isinstance(receipt, Mapping) else {}
    nodes = (
        live_birth_payload,
        receipt,
        live_birth_payload.get("argmax_transitions"),
        receipt.get("argmax_transitions"),
        live_birth_payload.get("region_argmax_transitions"),
        receipt.get("region_argmax_transitions"),
        live_birth_payload.get("argmax_transition_counts"),
        receipt.get("argmax_transition_counts"),
    )
    return _first_int_from_mappings(
        *(node for node in nodes if isinstance(node, Mapping)),
        keys=(
            "wrong_to_target",
            "wrong_to_target_count",
            "seg_wrong_to_target_count",
            "target_hard_won",
            "target_hard_won_count",
            "hard_won_count",
            "receiver_surface_wrong_to_target",
            "receiver_surface_wrong_to_target_count",
            "receiver_surface_target_hard_won_count",
        ),
    )


def _predict_pair01_nhwc01(model: Any, pair_indices: Any) -> tuple[Any, Any]:
    """Return both pair frames as NHWC in [0, 1] from the renderer."""

    import mlx.core as mx

    pair01 = model(pair_indices) / 255.0
    return (
        mx.transpose(pair01[:, 0], (0, 2, 3, 1)),  # type: ignore[union-attr]
        mx.transpose(pair01[:, 1], (0, 2, 3, 1)),  # type: ignore[union-attr]
    )


def _predict_frame1_nhwc01(model: Any, pair_indices: Any) -> Any:
    """Return frame-1 NHWC in [0, 1] from the renderer for the given pairs."""

    _pred0, pred1 = _predict_pair01_nhwc01(model, pair_indices)
    return pred1


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


def _region_support_from_frame1_nhwc01(
    *,
    scorer_teacher: Any,
    frame1_nhwc01: Any,
    region_mask_np: np.ndarray,
    birth_class: int,
) -> dict[str, Any]:
    """Measure the named birth region on a supplied receiver frame-1 surface."""

    logits_np = _candidate_logits_np(scorer_teacher, frame1_nhwc01)
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
    return _region_support_from_frame1_nhwc01(
        scorer_teacher=scorer_teacher,
        frame1_nhwc01=frame1,
        region_mask_np=region_mask_np,
        birth_class=birth_class,
    )


def _archive_roundtrip_shadow_support(
    model: Any,
    *,
    scorer_teacher: Any,
    pair_indices: Any,
    region_mask_np: np.ndarray,
    birth_class: int,
    latent_sections: Sequence[str],
    latent_codec: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Measure the named region with latent sections routed through HIV1 decode.

    Each named latent attribute is replaced (forward value only) by its exact
    ``hiv1_quantize_dequantize_for_training`` archive-decoded value, the SAME
    pairs are forwarded through the live decoder weights, and the region's
    target support is recomputed.  This isolates the latent-quantizer
    contribution to parse-back scorer-effect collapse from the decoder-weight
    quantizer (which the ``fakequant_mlx`` surface covers).  MLX arrays are
    immutable, so holding the original reference is a faithful snapshot; the
    originals are restored before returning.
    """

    sections = tuple(str(name) for name in latent_sections)
    if not sections:
        raise BirthSurvivalError("archive_roundtrip_shadow requires at least one latent section")
    snapshot: dict[str, Any] = {}
    applied: list[str] = []
    section_deltas: dict[str, float] = {}
    try:
        for name in sections:
            original = getattr(model, name, None)
            if original is None:
                raise BirthSurvivalError(
                    f"model has no latent section {name!r} for archive_roundtrip_shadow"
                )
            snapshot[name] = original
            decoded = hiv1_quantize_dequantize_for_training(
                original,
                section_config={"section_id": name, "latent_codec": str(latent_codec)},
            )
            section_deltas[name] = _max_abs_delta(original, decoded)
            setattr(model, name, decoded)
            applied.append(name)
        support = _region_support_on_surface(
            model,
            scorer_teacher=scorer_teacher,
            pair_indices=pair_indices,
            region_mask_np=region_mask_np,
            birth_class=birth_class,
        )
    finally:
        for name, original in snapshot.items():
            setattr(model, name, original)
    meta = {
        "surface_kind": "hiv1_latent_archive_roundtrip_shadow",
        "latent_sections": list(applied),
        "latent_codec": str(latent_codec),
        "section_hiv1_roundtrip_max_abs_delta": section_deltas,
        "decoder_weights_live": True,
        "contract": "latent_forward_routed_through_hiv1_archive_decode",
    }
    return support, meta


def _max_abs_delta(original: Any, decoded: Any) -> float:
    try:
        import mlx.core as mx

        delta = mx.max(mx.abs(decoded - original))  # type: ignore[union-attr]
        mx.eval(delta)  # type: ignore[union-attr]
        return float(np.asarray(delta, dtype=np.float32).reshape(-1)[0])
    except Exception:
        return float(
            np.abs(
                np.asarray(decoded, dtype=np.float32) - np.asarray(original, dtype=np.float32)
            ).max()
        )


def _region_debt_units(region_unsolved: int, total_scored_pixels: int) -> float:
    if total_scored_pixels <= 0:
        return 0.0
    return float(100.0 * int(region_unsolved) / int(total_scored_pixels))


def _margin_float_or_none(payload: Mapping[str, Any], key: str) -> float | None:
    try:
        value = float(payload.get(key))
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _margin_int_or_default(payload: Mapping[str, Any], key: str, default: int) -> int:
    try:
        return int(payload.get(key, default))
    except (TypeError, ValueError):
        return int(default)


def _target_margin_value(
    margin_stats: Mapping[str, Any],
    target_key: str,
    legacy_runner_up_key: str,
) -> float | None:
    direct = _margin_float_or_none(margin_stats, target_key)
    if direct is not None:
        return direct
    # ``runner_up - target`` with the same percentile can be sign-flipped for
    # mean/p50 only.  For min/max percentiles the exact paired statistic is not
    # recoverable unless the producer emitted explicit target-positive fields.
    if target_key.endswith("_mean") or target_key.endswith("_p50"):
        legacy = _margin_float_or_none(margin_stats, legacy_runner_up_key)
        return None if legacy is None else -legacy
    return None


def _target_margin_certificate(
    *,
    action_id: str,
    surface: str,
    birth_class: int,
    region_pixels: int,
    margin_stats: Mapping[str, Any],
    worst_region: Mapping[str, Any],
    live_wrong_to_target: int | None,
    surface_wrong_to_target: int,
    retention_ratio: float | None,
    retention_floor: float,
    retention_required: bool,
    scorer_effect_survived: bool,
    first_failed_surface: str | None,
) -> dict[str, Any]:
    """Return the target-positive wall-crossing certificate for this surface."""

    target_margin_min = _target_margin_value(
        margin_stats,
        "target_margin_min",
        "margin_max",
    )
    target_margin_p10 = _target_margin_value(
        margin_stats,
        "target_margin_p10",
        "margin_p90",
    )
    target_margin_p50 = _target_margin_value(
        margin_stats,
        "target_margin_p50",
        "margin_p50",
    )
    target_margin_mean = _target_margin_value(
        margin_stats,
        "target_margin_mean",
        "margin_mean",
    )
    floor_satisfied = (
        None
        if target_margin_p10 is None
        else bool(target_margin_p10 >= TARGET_MARGIN_SAFETY_FLOOR)
    )
    return {
        "schema": TARGET_MARGIN_CERTIFICATE_SCHEMA,
        "action_id": str(action_id),
        "surface": str(surface),
        "family": "hinerv",
        "pair_id": worst_region.get("pair_index"),
        "frame_index": 1,
        "birth_class_index": int(birth_class),
        "target_class": int(birth_class),
        "support_pixels": int(region_pixels),
        "region_pixel_count": int(region_pixels),
        "worst_region_key": [
            _margin_int_or_default(worst_region, "batch_index", -1),
            _margin_int_or_default(worst_region, "class_index", int(birth_class)),
            _margin_int_or_default(worst_region, "region_label", -1),
        ],
        "margin_convention": "target_minus_runner_up",
        "runner_up_margin_convention": "runner_up_minus_target",
        "target_margin_floor": float(TARGET_MARGIN_SAFETY_FLOOR),
        "target_margin_floor_satisfied": floor_satisfied,
        "target_margin_min": target_margin_min,
        "target_margin_p10": target_margin_p10,
        "target_margin_p50": target_margin_p50,
        "target_margin_mean": target_margin_mean,
        "runner_up_minus_target_margin_min": _margin_float_or_none(
            margin_stats,
            "margin_min",
        ),
        "runner_up_minus_target_margin_p10": _margin_float_or_none(
            margin_stats,
            "margin_p10",
        ),
        "runner_up_minus_target_margin_p50": _margin_float_or_none(
            margin_stats,
            "margin_p50",
        ),
        "runner_up_minus_target_margin_mean": _margin_float_or_none(
            margin_stats,
            "margin_mean",
        ),
        "live_wrong_to_target_count": live_wrong_to_target,
        "wrong_to_target_count": int(surface_wrong_to_target),
        "surface_wrong_to_target_count": int(surface_wrong_to_target),
        "wrong_to_target_retention_ratio": retention_ratio,
        "scorer_effect_retention_ratio": retention_ratio,
        "scorer_effect_retention_floor": float(retention_floor),
        "scorer_effect_retention_required": bool(retention_required),
        "scorer_effect_survived": bool(scorer_effect_survived),
        "first_failed_surface": first_failed_surface,
        "human_visual_fidelity_objective": False,
        "authority": _NON_AUTHORITY_MARKER,
    }


def _read_mapping_report(report: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(report, Mapping):
        return dict(report)
    path = Path(report).expanduser().resolve(strict=False)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BirthSurvivalError(f"could not read parseback report: {path}") from exc
    if not isinstance(payload, Mapping):
        raise BirthSurvivalError(f"parseback report must be a JSON object: {path}")
    return dict(payload)


def _read_mapping_payload(payload: Mapping[str, Any] | str | Path, *, label: str) -> dict[str, Any]:
    if isinstance(payload, Mapping):
        return dict(payload)
    path = Path(payload).expanduser().resolve(strict=False)
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BirthSurvivalError(f"could not read {label}: {path}") from exc
    if not isinstance(loaded, Mapping):
        raise BirthSurvivalError(f"{label} must be a JSON object: {path}")
    return dict(loaded)


def _sha256_file_or_none(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _direct_receiver_report(report: Mapping[str, Any]) -> Mapping[str, Any]:
    direct = report.get("direct_receiver_cache_report")
    if isinstance(direct, Mapping):
        return direct
    return report


def _parseback_report_archive_path(report: Mapping[str, Any]) -> Path | None:
    direct = _direct_receiver_report(report)
    for source in (direct, report):
        raw = source.get("archive_path")
        if isinstance(raw, str) and raw.strip():
            return Path(raw).expanduser().resolve(strict=False)
    return None


def _parseback_report_archive_sha256(report: Mapping[str, Any]) -> str | None:
    direct = _direct_receiver_report(report)
    for source in (direct, report):
        raw = source.get("archive_sha256")
        if isinstance(raw, str) and len(raw) == 64:
            return raw.lower()
    return None


def _parseback_report_action_id(report: Mapping[str, Any]) -> str | None:
    direct = _direct_receiver_report(report)
    for source in (direct, report):
        for key in ("action_id", "live_birth_action_id", "hard_birth_action_id", "target_region_birth_action_id"):
            raw = source.get(key)
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
    return None


def _requested_pair_indices_np(pair_indices: Any) -> np.ndarray:
    arr = np.asarray(pair_indices, dtype=np.int64).reshape(-1)
    if arr.size == 0 or np.any(arr < 0):
        raise BirthSurvivalError("pair_indices must be non-empty and non-negative")
    return np.ascontiguousarray(arr)


def _target_rgb_birth_row(target: Any | None, *, batch_index: int, name: str) -> np.ndarray | None:
    if target is None:
        return None
    arr = np.asarray(target, dtype=np.float32)
    if arr.ndim != 4 or int(arr.shape[-1]) != 3:
        raise BirthSurvivalError(f"{name} must be NHWC RGB; got shape {arr.shape}")
    if int(arr.shape[0]) == 1:
        return np.ascontiguousarray(arr)
    if not 0 <= int(batch_index) < int(arr.shape[0]):
        raise BirthSurvivalError(
            f"{name} has {int(arr.shape[0])} rows but live birth batch_index={int(batch_index)}"
        )
    return np.ascontiguousarray(arr[int(batch_index) : int(batch_index) + 1])


def _parseback_selected_pairs(report: Mapping[str, Any]) -> set[int] | None:
    direct = _direct_receiver_report(report)
    raw = direct.get("selected_pair_indices")
    if raw is None:
        return None
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise BirthSurvivalError("parseback report selected_pair_indices must be a sequence")
    return {int(value) for value in raw}


def _read_hiv1_payload_from_archive_zip(archive_path: Path) -> tuple[str, bytes]:
    if not archive_path.is_file():
        raise BirthSurvivalError(f"HiNeRV parseback archive.zip missing: {archive_path}")
    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            names = [name for name in zf.namelist() if not name.endswith("/")]
            payload_members = [
                name for name in names if name in {MINIMAL_SINGLE_MEMBER_NAME, "0.bin"}
            ]
            if len(payload_members) != 1:
                raise BirthSurvivalError(
                    "HiNeRV parseback requires exactly one HIV1 payload member "
                    f"named {MINIMAL_SINGLE_MEMBER_NAME!r} or '0.bin'; "
                    f"found {payload_members!r} among {names[:10]!r}"
                )
            member = payload_members[0]
            payload = zf.read(member)
    except zipfile.BadZipFile as exc:
        raise BirthSurvivalError(f"HiNeRV parseback archive.zip is invalid: {archive_path}") from exc
    if not payload.startswith(b"HIV1"):
        digest = hashlib.sha256(payload).hexdigest()
        raise BirthSurvivalError(f"HiNeRV parseback payload is not HIV1: member={member!r} sha256={digest}")
    return member, payload


def _render_parseback_pair01_nhwc01(
    *,
    archive_path: Path,
    pair_indices: np.ndarray,
) -> tuple[tuple[Any, Any], dict[str, Any]]:
    """Render requested pairs from parsed HIV1 archive bytes through receiver."""

    _require_mlx()
    import mlx.core as mx
    import torch

    from tac.substrates.hi_nerv.inflate import build_model_from_archive

    member, payload = _read_hiv1_payload_from_archive_zip(archive_path)
    arc, cfg, receiver_model = build_model_from_archive(payload, device="cpu")
    max_pair = int(pair_indices.max())
    if max_pair >= int(cfg.num_pairs):
        raise BirthSurvivalError(
            "parseback pair index exceeds archive pair count: "
            f"max_pair={max_pair} num_pairs={int(cfg.num_pairs)}"
        )
    idx_t = torch.tensor(pair_indices.tolist(), dtype=torch.long)
    with torch.no_grad():
        rgb0, rgb1 = receiver_model(idx_t)
    frame0 = np.transpose(np.asarray(rgb0.detach().cpu(), dtype=np.float32), (0, 2, 3, 1))
    frame1 = np.transpose(np.asarray(rgb1.detach().cpu(), dtype=np.float32), (0, 2, 3, 1))
    archive_bytes = archive_path.read_bytes()
    return (
        (
            mx.array(frame0, dtype=mx.float32),  # type: ignore[union-attr]
            mx.array(frame1, dtype=mx.float32),  # type: ignore[union-attr]
        ),
        {
            "parseback_archive_path": archive_path.as_posix(),
            "parseback_archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
            "parseback_archive_bytes": len(archive_bytes),
            "parseback_zip_member": str(member),
            "parseback_payload_sha256": hashlib.sha256(payload).hexdigest(),
            "parseback_payload_bytes": len(payload),
            "parseback_archive_schema_version": int(arc.schema_version),
            "parseback_receiver_model": "hiv1_build_model_from_archive_torch_cpu",
            "parseback_pair_indices": [int(value) for value in pair_indices.tolist()],
        },
    )


def _inflated_raw_spans(
    inflated_dir: Path,
    *,
    camera_size_wh: tuple[int, int],
) -> tuple[list[dict[str, Any]], list[str]]:
    width, height = (int(camera_size_wh[0]), int(camera_size_wh[1]))
    frame_bytes = int(width) * int(height) * RGB_CHANNELS
    spans: list[dict[str, Any]] = []
    blockers: list[str] = []
    next_pair = 0
    for raw_path in sorted(inflated_dir.rglob("*.raw")):
        size = int(raw_path.stat().st_size)
        if frame_bytes <= 0 or size % frame_bytes != 0:
            blockers.append(f"birth_survival_inflated_raw_size_mismatch:{raw_path.name}")
            continue
        frame_count = size // frame_bytes
        pair_count = frame_count // SEQ_LEN
        if pair_count <= 0:
            blockers.append(f"birth_survival_inflated_raw_no_complete_pairs:{raw_path.name}")
            continue
        spans.append(
            {
                "path": raw_path,
                "start_pair": next_pair,
                "end_pair_exclusive": next_pair + pair_count,
                "frame_count": frame_count,
                "pair_count": pair_count,
                "bytes": size,
                "sha256": _sha256_file_or_none(raw_path),
            }
        )
        next_pair += pair_count
    if not spans and not blockers:
        blockers.append("birth_survival_inflated_raw_files_missing")
    return spans, blockers


def _read_inflated_pair01_nhwc01(
    *,
    inflated_dir: Path,
    pair_index: int,
    camera_size_wh: tuple[int, int],
) -> tuple[tuple[Any, Any] | None, dict[str, Any], str | None]:
    """Read one upstream-inflated raw pair without materializing the video."""

    _require_mlx()
    import mlx.core as mx

    width, height = (int(camera_size_wh[0]), int(camera_size_wh[1]))
    spans, blockers = _inflated_raw_spans(inflated_dir, camera_size_wh=(width, height))
    if blockers and not spans:
        return None, {"inflated_raw_blockers": blockers}, blockers[0]
    for span in spans:
        if int(span["start_pair"]) <= int(pair_index) < int(span["end_pair_exclusive"]):
            local_pair = int(pair_index) - int(span["start_pair"])
            frame0_index = local_pair * SEQ_LEN
            frame1_index = frame0_index + 1
            raw_path = Path(span["path"])
            try:
                mmap = np.memmap(
                    raw_path,
                    dtype=np.uint8,
                    mode="r",
                    shape=(int(span["frame_count"]), height, width, RGB_CHANNELS),
                )
                pair = np.asarray(
                    mmap[frame0_index : frame1_index + 1],
                    dtype=np.float32,
                ) / 255.0
            except Exception as exc:
                return (
                    None,
                    {
                        "inflated_raw_path": raw_path.as_posix(),
                        "inflated_raw_read_error": f"{type(exc).__name__}:{exc}",
                    },
                    "birth_survival_inflated_raw_pair_read_failed",
                )
            finally:
                try:
                    del mmap  # type: ignore[possibly-used-before-assignment]
                except UnboundLocalError:
                    pass
            if pair.shape != (SEQ_LEN, height, width, RGB_CHANNELS):
                return (
                    None,
                    {
                        "inflated_raw_path": raw_path.as_posix(),
                        "inflated_pair_shape": [int(v) for v in pair.shape],
                    },
                    "birth_survival_inflated_raw_pair_shape_mismatch",
                )
            meta = {
                "inflated_raw_path": raw_path.as_posix(),
                "inflated_raw_sha256": span.get("sha256"),
                "inflated_raw_bytes": int(span["bytes"]),
                "inflated_global_pair_index": int(pair_index),
                "inflated_raw_local_pair_index": int(local_pair),
                "inflated_raw_frame_indices": [int(frame0_index), int(frame1_index)],
                "inflated_raw_frame_shape_nhwc": [int(height), int(width), RGB_CHANNELS],
                "inflated_raw_frame_dtype": "uint8",
                "inflated_raw_pair_read_mode": "numpy.memmap_slice",
                "inflated_raw_span_start_pair": int(span["start_pair"]),
                "inflated_raw_span_end_pair_exclusive": int(span["end_pair_exclusive"]),
            }
            return (
                (
                    mx.array(pair[0:1], dtype=mx.float32),  # type: ignore[union-attr]
                    mx.array(pair[1:2], dtype=mx.float32),  # type: ignore[union-attr]
                ),
                meta,
                None,
            )
    return (
        None,
        {
            "inflated_raw_pair_index": int(pair_index),
            "inflated_raw_spans": [
                {
                    "path": Path(span["path"]).as_posix(),
                    "start_pair": int(span["start_pair"]),
                    "end_pair_exclusive": int(span["end_pair_exclusive"]),
                    "pair_count": int(span["pair_count"]),
                }
                for span in spans
            ],
            "inflated_raw_blockers": blockers,
        },
        "birth_survival_inflated_birth_pair_not_found",
    )


def _live_pose_compensation(live_birth_payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    compensation = live_birth_payload.get("pose_compensation")
    if not isinstance(compensation, Mapping):
        receipt = live_birth_payload.get("receipt")
        if isinstance(receipt, Mapping):
            compensation = receipt.get("pose_compensation")
    if not isinstance(compensation, Mapping) or compensation.get("composite_accepted") is not True:
        return None
    return compensation


def _live_compensation_d_pose(compensation: Mapping[str, Any]) -> float | None:
    candidates: list[float] = []
    raw = compensation.get("composite_d_pose_batch")
    if isinstance(raw, int | float):
        candidates.append(float(raw))
    for attempt in compensation.get("attempts") or []:
        if not isinstance(attempt, Mapping) or attempt.get("accepted") is not True:
            continue
        raw_attempt = attempt.get("composite_d_pose_batch")
        if isinstance(raw_attempt, int | float):
            candidates.append(float(raw_attempt))
    finite = [value for value in candidates if np.isfinite(value) and value >= 0.0]
    if not finite:
        return None
    return min(finite)


def _pose_compensation_survival_on_surface(
    model: Any,
    *,
    pose_teacher: Any | None,
    target_rgb_0: Any | None,
    target_rgb_1: Any | None,
    pair_indices: Any,
    live_pose_compensation: Mapping[str, Any] | None,
    surface_pair01_nhwc01: tuple[Any, Any] | None = None,
    d_pose_tolerance_abs: float,
    d_pose_tolerance_rel: float,
) -> dict[str, Any]:
    """Re-measure frame0 compensation survival for a composite birth action."""

    if live_pose_compensation is None:
        return {
            "required": False,
            "survived": None,
            "blockers": [],
            "live_composite_d_pose_batch": None,
            "surface_d_pose_batch": None,
        }
    live_d_pose = _live_compensation_d_pose(live_pose_compensation)
    if live_d_pose is None:
        return {
            "required": True,
            "survived": False,
            "blockers": ["pose_compensation_survival_live_d_pose_missing"],
            "live_composite_d_pose_batch": None,
            "surface_d_pose_batch": None,
        }
    pose_fn = getattr(pose_teacher, "teacher_pose_for_yuv6_pair_nhwc", None) if pose_teacher is not None else None
    if not callable(pose_fn):
        return {
            "required": True,
            "survived": False,
            "blockers": ["pose_compensation_survival_pose_teacher_missing"],
            "live_composite_d_pose_batch": float(live_d_pose),
            "surface_d_pose_batch": None,
        }

    _require_mlx()
    import mlx.core as mx

    from tac.local_acceleration.pr95_hnerv_mlx_training import rgb_to_yuv6_mlx

    def _receiver_uint8_roundtrip_ste_nhwc01(rgb: Any) -> Any:
        clamped = mx.clip(rgb, 0.0, 1.0)  # type: ignore[union-attr]
        receiver = mx.round(clamped * 255.0) / 255.0  # type: ignore[union-attr]
        return rgb + mx.stop_gradient(receiver - rgb)  # type: ignore[union-attr]

    if target_rgb_0 is None or target_rgb_1 is None:
        target_pose_fn = getattr(pose_teacher, "teacher_pose_for_indices", None)
        if not callable(target_pose_fn):
            return {
                "required": True,
                "survived": False,
                "blockers": ["pose_compensation_survival_target_frames_missing"],
                "live_composite_d_pose_batch": float(live_d_pose),
                "surface_d_pose_batch": None,
            }
        target_pose = target_pose_fn(pair_indices)
    else:
        target0 = mx.array(target_rgb_0).astype(mx.float32)  # type: ignore[union-attr]
        target1 = mx.array(target_rgb_1).astype(mx.float32)  # type: ignore[union-attr]
        target0_receiver = _receiver_uint8_roundtrip_ste_nhwc01(target0)
        target1_receiver = _receiver_uint8_roundtrip_ste_nhwc01(target1)
        target_pose = pose_fn(
            mx.concatenate(  # type: ignore[union-attr]
                [
                    rgb_to_yuv6_mlx(target0_receiver * 255.0),
                    rgb_to_yuv6_mlx(target1_receiver * 255.0),
                ],
                axis=-1,
            )
        )
    if surface_pair01_nhwc01 is None:
        pred0, pred1 = _predict_pair01_nhwc01(model, pair_indices)
    else:
        pred0, pred1 = surface_pair01_nhwc01
    pred0_receiver = _receiver_uint8_roundtrip_ste_nhwc01(pred0)
    pred1_receiver = _receiver_uint8_roundtrip_ste_nhwc01(pred1)
    surface_pose = pose_fn(
        mx.concatenate(  # type: ignore[union-attr]
            [
                rgb_to_yuv6_mlx(pred0_receiver * 255.0),
                rgb_to_yuv6_mlx(pred1_receiver * 255.0),
            ],
            axis=-1,
        )
    )
    mx.eval(target_pose, surface_pose)  # type: ignore[union-attr]
    target_pose_np = np.asarray(target_pose, dtype=np.float64)
    surface_pose_np = np.asarray(surface_pose, dtype=np.float64)
    if target_pose_np.shape != surface_pose_np.shape:
        return {
            "required": True,
            "survived": False,
            "blockers": ["pose_compensation_survival_pose_shape_mismatch"],
            "live_composite_d_pose_batch": float(live_d_pose),
            "surface_d_pose_batch": None,
        }
    surface_d_pose = float(np.mean((surface_pose_np - target_pose_np) ** 2))
    tolerance = float(d_pose_tolerance_abs) + float(d_pose_tolerance_rel) * max(float(live_d_pose), 1.0e-12)
    survived = bool(np.isfinite(surface_d_pose) and surface_d_pose <= float(live_d_pose) + tolerance)
    return {
        "required": True,
        "survived": survived,
        "blockers": ([] if survived else ["pose_compensation_survival_d_pose_regressed"]),
        "live_composite_d_pose_batch": float(live_d_pose),
        "surface_d_pose_batch": surface_d_pose,
        "d_pose_tolerance_abs": float(d_pose_tolerance_abs),
        "d_pose_tolerance_rel": float(d_pose_tolerance_rel),
        "pose_surface_receiver_uint8_roundtrip": True,
        "surface_pose_score_term": float(np.sqrt(10.0 * max(surface_d_pose, 0.0))),
        "live_pose_score_term": float(np.sqrt(10.0 * max(float(live_d_pose), 0.0))),
    }


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
    pose_teacher: Any | None = None,
    target_rgb_0: Any | None = None,
    target_rgb_1: Any | None = None,
    target_labels: Any,
    live_birth_payload: Mapping[str, Any],
    surface: str,
    pair_indices: Any,
    fakequant_bits: int = 8,
    latent_sections: Sequence[str] = DEFAULT_ARCHIVE_ROUNDTRIP_LATENT_SECTIONS,
    latent_codec: str = DEFAULT_ARCHIVE_ROUNDTRIP_LATENT_CODEC,
    pose_d_pose_tolerance_abs: float = 1.0e-8,
    pose_d_pose_tolerance_rel: float = 5.0e-2,
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
    live_pose_compensation = _live_pose_compensation(live_birth_payload)

    idx = mx.array(pair_indices, dtype=mx.int32)  # type: ignore[union-attr]
    if idx.ndim != 1 or int(idx.shape[0]) <= 0:
        raise BirthSurvivalError("pair_indices must be a non-empty rank-1 tensor")

    surface_meta: dict[str, Any] | None = None
    if surface_name == "archive_roundtrip_shadow":
        # Latent-quantizer isolation: route the named latent sections through the
        # exact HIV1 int16 archive decode, keep decoder weights live.  Pose
        # compensation is the fakequant surface's responsibility; the shadow row
        # records pose as not-exercised so its survived boolean is pure
        # target-margin/scorer-effect on the latent quantizer.
        support, surface_meta = _archive_roundtrip_shadow_support(
            model,
            scorer_teacher=scorer_teacher,
            pair_indices=idx,
            region_mask_np=region_mask_np,
            birth_class=birth_class,
            latent_sections=latent_sections,
            latent_codec=latent_codec,
        )
        pose_compensation_survival = {"required": False, "surface": surface_name}
        bits_for_row: int | None = None
    else:
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
            pose_compensation_survival = _pose_compensation_survival_on_surface(
                model,
                pose_teacher=pose_teacher,
                target_rgb_0=target_rgb_0,
                target_rgb_1=target_rgb_1,
                pair_indices=idx,
                live_pose_compensation=live_pose_compensation,
                d_pose_tolerance_abs=float(pose_d_pose_tolerance_abs),
                d_pose_tolerance_rel=float(pose_d_pose_tolerance_rel),
            )
        finally:
            _restore_fake_quant_config(model, fq_snapshot)
        bits_for_row = bits

    region_hard_won = int(support["region_hard_won"])
    net_target_support_delta = int(region_hard_won - initial_in_region_target)
    target_support_survived = bool(region_hard_won >= 1 and net_target_support_delta > 0)
    compensation_required = bool(pose_compensation_survival["required"])
    compensation_survived = (
        True
        if not compensation_required
        else bool(pose_compensation_survival["survived"])
    )
    survived = bool(target_support_survived and compensation_survived)
    region_debt_units = _region_debt_units(int(support["region_unsolved"]), int(support["total_scored_pixels"]))

    return _survival_row(
        action_id=action_id,
        surface=surface_name,
        survived=survived,
        live_wrong_to_target_count=_live_wrong_to_target_count(live_birth_payload),
        birth_class=birth_class,
        region_pixels=region_pixels,
        region_hard_won=region_hard_won,
        target_hard_lost=max(0, initial_in_region_target - region_hard_won),
        net_target_support_delta=net_target_support_delta,
        initial_in_region_target=initial_in_region_target,
        region_unsolved=int(support["region_unsolved"]),
        region_debt_units=region_debt_units,
        margin_stats=support["stats"],
        fakequant_bits=bits_for_row,
        total_scored_pixels=int(support["total_scored_pixels"]),
        worst_region=worst_region,
        pose_compensation_survival=pose_compensation_survival,
        surface_meta=surface_meta,
    )


def measure_birth_parseback_survival_from_report(
    *,
    parseback_report: Mapping[str, Any] | str | Path,
    scorer_teacher: Any,
    pose_teacher: Any | None = None,
    target_rgb_0: Any | None = None,
    target_rgb_1: Any | None = None,
    target_labels: Any,
    live_birth_payload: Mapping[str, Any],
    pair_indices: Any,
    pose_d_pose_tolerance_abs: float = 1.0e-8,
    pose_d_pose_tolerance_rel: float = 5.0e-2,
) -> dict[str, Any]:
    """Re-measure an accepted birth on parsed archive receiver bytes.

    The supplied report is a custody/identity handle produced by the receiver
    cache or archive parse-back path.  This function verifies that the report
    names a real archive and, when selected-pair coverage is present, that it
    includes the live birth pair(s).  The actual metric surface is then rendered
    from the archive bytes through ``build_model_from_archive`` and measured
    through the same scorer teacher as the live/fakequant surfaces.
    """

    action_id = _live_action_id(live_birth_payload)
    report = _read_mapping_report(parseback_report)
    report_action_id = _parseback_report_action_id(report)
    if report_action_id is not None and report_action_id != action_id:
        return _blocked_survival_row(
            action_id=action_id,
            surface="parseback_mlx",
            blocker="birth_survival_parseback_action_id_mismatch",
            reason=(
                "parseback report action_id does not match live birth action_id: "
                f"report={report_action_id} live={action_id}"
            ),
        )
    archive_path = _parseback_report_archive_path(report)
    if archive_path is None:
        return _blocked_survival_row(
            action_id=action_id,
            surface="parseback_mlx",
            blocker="birth_survival_parseback_archive_path_missing",
        )
    expected_archive_sha = _parseback_report_archive_sha256(report)
    if expected_archive_sha is None:
        return _blocked_survival_row(
            action_id=action_id,
            surface="parseback_mlx",
            blocker="birth_survival_parseback_archive_sha256_missing",
        )
    if not archive_path.is_file():
        return _blocked_survival_row(
            action_id=action_id,
            surface="parseback_mlx",
            blocker="birth_survival_parseback_archive_zip_missing",
        )
    actual_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    if actual_sha.lower() != expected_archive_sha.lower():
        return _blocked_survival_row(
            action_id=action_id,
            surface="parseback_mlx",
            blocker="birth_survival_parseback_archive_sha256_mismatch",
        )

    pair_np = _requested_pair_indices_np(pair_indices)
    worst_region = _live_worst_region(live_birth_payload)
    birth_batch = int(worst_region["batch_index"])
    if not 0 <= birth_batch < int(pair_np.shape[0]):
        raise BirthSurvivalError(
            "live birth batch_index is outside the supplied pair_indices: "
            f"batch_index={birth_batch} pair_count={int(pair_np.shape[0])}"
        )
    birth_pair_np = np.ascontiguousarray(np.array([int(pair_np[birth_batch])], dtype=np.int64))
    selected_pairs = _parseback_selected_pairs(report)
    if selected_pairs is not None:
        birth_pair = int(birth_pair_np[0])
        if birth_pair not in selected_pairs:
            return _blocked_survival_row(
                action_id=action_id,
                surface="parseback_mlx",
                blocker="birth_survival_parseback_birth_pair_not_cached",
                reason=(
                    "parseback report did not cache the live hard-birth pair; "
                    f"birth_pair={birth_pair} selected_pairs={sorted(selected_pairs)[:8]}"
                ),
            )

    _require_mlx()
    birth_class = int(worst_region["class_index"])
    region_mask_np, region_pixels = reconstruct_birth_region_mask(target_labels, worst_region)
    region_mask_np = np.ascontiguousarray(region_mask_np[birth_batch : birth_batch + 1])
    initial_in_region_target = _initial_in_region_target_count(worst_region)
    live_pose_compensation = _live_pose_compensation(live_birth_payload)

    pair_frames, surface_meta = _render_parseback_pair01_nhwc01(
        archive_path=archive_path,
        pair_indices=birth_pair_np,
    )
    support = _region_support_from_frame1_nhwc01(
        scorer_teacher=scorer_teacher,
        frame1_nhwc01=pair_frames[1],
        region_mask_np=region_mask_np,
        birth_class=birth_class,
    )
    target_rgb_0_birth = _target_rgb_birth_row(target_rgb_0, batch_index=birth_batch, name="target_rgb_0")
    target_rgb_1_birth = _target_rgb_birth_row(target_rgb_1, batch_index=birth_batch, name="target_rgb_1")
    pose_compensation_survival = _pose_compensation_survival_on_surface(
        object(),
        pose_teacher=pose_teacher,
        target_rgb_0=target_rgb_0_birth,
        target_rgb_1=target_rgb_1_birth,
        pair_indices=birth_pair_np,
        live_pose_compensation=live_pose_compensation,
        surface_pair01_nhwc01=pair_frames,
        d_pose_tolerance_abs=float(pose_d_pose_tolerance_abs),
        d_pose_tolerance_rel=float(pose_d_pose_tolerance_rel),
    )

    region_hard_won = int(support["region_hard_won"])
    net_target_support_delta = int(region_hard_won - initial_in_region_target)
    target_support_survived = bool(region_hard_won >= 1 and net_target_support_delta > 0)
    compensation_required = bool(pose_compensation_survival["required"])
    compensation_survived = (
        True
        if not compensation_required
        else bool(pose_compensation_survival["survived"])
    )
    survived = bool(target_support_survived and compensation_survived)
    region_debt_units = _region_debt_units(int(support["region_unsolved"]), int(support["total_scored_pixels"]))

    return _survival_row(
        action_id=action_id,
        surface="parseback_mlx",
        survived=survived,
        live_wrong_to_target_count=_live_wrong_to_target_count(live_birth_payload),
        birth_class=birth_class,
        region_pixels=region_pixels,
        region_hard_won=region_hard_won,
        target_hard_lost=max(0, initial_in_region_target - region_hard_won),
        net_target_support_delta=net_target_support_delta,
        initial_in_region_target=initial_in_region_target,
        region_unsolved=int(support["region_unsolved"]),
        region_debt_units=region_debt_units,
        margin_stats=support["stats"],
        fakequant_bits=None,
        total_scored_pixels=int(support["total_scored_pixels"]),
        worst_region=worst_region,
        pose_compensation_survival=pose_compensation_survival,
        surface_meta={
            **surface_meta,
            "parseback_report_schema": str(report.get("schema") or ""),
            "parseback_direct_report_schema": str(
                _direct_receiver_report(report).get("schema") or ""
            ),
        },
    )


def measure_birth_inflated_torch_cpu_survival_from_local_replay(
    *,
    local_replay_summary: Mapping[str, Any] | str | Path,
    scorer_teacher: Any,
    pose_teacher: Any | None = None,
    target_rgb_0: Any | None = None,
    target_rgb_1: Any | None = None,
    target_labels: Any,
    live_birth_payload: Mapping[str, Any],
    pair_indices: Any,
    pose_d_pose_tolerance_abs: float = 1.0e-8,
    pose_d_pose_tolerance_rel: float = 5.0e-2,
    camera_size_wh: tuple[int, int] = CAMERA_SIZE_WH,
) -> dict[str, Any]:
    """Re-measure an accepted birth on retained upstream CPU-inflated bytes.

    This is the L5 receiver surface: the candidate archive has already traveled
    through upstream ``evaluate.sh`` / ``inflate.sh`` on local CPU, and this
    producer reads the retained ``*.raw`` RGB tensor exactly as
    ``TensorVideoDataset`` does.  It memmaps only the live birth pair so the
    proof does not persist or load full-video tensors.
    """

    action_id = _live_action_id(live_birth_payload)
    try:
        summary = _read_mapping_payload(local_replay_summary, label="local replay summary")
    except BirthSurvivalError as exc:
        return _blocked_survival_row(
            action_id=action_id,
            surface="inflated_torch_cpu",
            blocker="birth_survival_inflated_local_replay_summary_unreadable",
            reason=str(exc),
        )
    if summary.get("schema") != "local_submission_replay.v1":
        return _blocked_survival_row(
            action_id=action_id,
            surface="inflated_torch_cpu",
            blocker="birth_survival_inflated_local_replay_schema_mismatch",
            reason=f"expected local_submission_replay.v1, got {summary.get('schema')!r}",
        )
    if str(summary.get("device") or "") != "cpu":
        return _blocked_survival_row(
            action_id=action_id,
            surface="inflated_torch_cpu",
            blocker="birth_survival_inflated_local_replay_not_cpu",
            reason=f"local replay device must be cpu, got {summary.get('device')!r}",
        )
    if summary.get("evaluation_passed") is not True:
        return _blocked_survival_row(
            action_id=action_id,
            surface="inflated_torch_cpu",
            blocker="birth_survival_inflated_local_replay_not_passed",
            reason="upstream local replay did not pass, so inflated bytes are not a valid surface",
        )
    cleanup = str(summary.get("inflated_dir_cleanup") or "")
    if cleanup not in {"retained_by_request", "retained_by_request_after_success"}:
        return _blocked_survival_row(
            action_id=action_id,
            surface="inflated_torch_cpu",
            blocker="birth_survival_inflated_raw_not_retained",
            reason=f"local replay inflated_dir_cleanup={cleanup!r}; rerun with keep_inflated=True",
        )
    inflated_dir_raw = summary.get("inflated_dir")
    if not isinstance(inflated_dir_raw, str) or not inflated_dir_raw.strip():
        return _blocked_survival_row(
            action_id=action_id,
            surface="inflated_torch_cpu",
            blocker="birth_survival_inflated_dir_missing_from_summary",
        )
    inflated_dir = Path(inflated_dir_raw).expanduser().resolve(strict=False)
    if not inflated_dir.is_dir():
        return _blocked_survival_row(
            action_id=action_id,
            surface="inflated_torch_cpu",
            blocker="birth_survival_inflated_dir_missing",
            reason=f"retained inflated_dir does not exist: {inflated_dir}",
        )

    pair_np = _requested_pair_indices_np(pair_indices)
    worst_region = _live_worst_region(live_birth_payload)
    birth_batch = int(worst_region["batch_index"])
    if not 0 <= birth_batch < int(pair_np.shape[0]):
        raise BirthSurvivalError(
            "live birth batch_index is outside the supplied pair_indices: "
            f"batch_index={birth_batch} pair_count={int(pair_np.shape[0])}"
        )
    birth_pair = int(pair_np[birth_batch])
    pair_frames, surface_meta, blocker = _read_inflated_pair01_nhwc01(
        inflated_dir=inflated_dir,
        pair_index=birth_pair,
        camera_size_wh=camera_size_wh,
    )
    if blocker is not None or pair_frames is None:
        return _blocked_survival_row(
            action_id=action_id,
            surface="inflated_torch_cpu",
            blocker=blocker or "birth_survival_inflated_pair_read_failed",
            extra={
                **surface_meta,
                "local_replay_report_path": summary.get("report_path"),
                "local_replay_axis_tag": summary.get("axis_tag"),
                "local_replay_summary_schema": summary.get("schema"),
                "local_replay_inflated_dir": inflated_dir.as_posix(),
            },
        )

    birth_class = int(worst_region["class_index"])
    region_mask_np, region_pixels = reconstruct_birth_region_mask(target_labels, worst_region)
    region_mask_np = np.ascontiguousarray(region_mask_np[birth_batch : birth_batch + 1])
    initial_in_region_target = _initial_in_region_target_count(worst_region)
    live_pose_compensation = _live_pose_compensation(live_birth_payload)

    support = _region_support_from_frame1_nhwc01(
        scorer_teacher=scorer_teacher,
        frame1_nhwc01=pair_frames[1],
        region_mask_np=region_mask_np,
        birth_class=birth_class,
    )
    target_rgb_0_birth = _target_rgb_birth_row(target_rgb_0, batch_index=birth_batch, name="target_rgb_0")
    target_rgb_1_birth = _target_rgb_birth_row(target_rgb_1, batch_index=birth_batch, name="target_rgb_1")
    pose_compensation_survival = _pose_compensation_survival_on_surface(
        object(),
        pose_teacher=pose_teacher,
        target_rgb_0=target_rgb_0_birth,
        target_rgb_1=target_rgb_1_birth,
        pair_indices=np.ascontiguousarray(np.array([birth_pair], dtype=np.int64)),
        live_pose_compensation=live_pose_compensation,
        surface_pair01_nhwc01=pair_frames,
        d_pose_tolerance_abs=float(pose_d_pose_tolerance_abs),
        d_pose_tolerance_rel=float(pose_d_pose_tolerance_rel),
    )

    region_hard_won = int(support["region_hard_won"])
    net_target_support_delta = int(region_hard_won - initial_in_region_target)
    target_support_survived = bool(region_hard_won >= 1 and net_target_support_delta > 0)
    compensation_required = bool(pose_compensation_survival["required"])
    compensation_survived = (
        True
        if not compensation_required
        else bool(pose_compensation_survival["survived"])
    )
    survived = bool(target_support_survived and compensation_survived)
    region_debt_units = _region_debt_units(int(support["region_unsolved"]), int(support["total_scored_pixels"]))
    archive_path = Path(str(summary.get("archive_zip_path") or "")).expanduser()
    archive_path = archive_path.resolve(strict=False) if str(archive_path) else archive_path

    return _survival_row(
        action_id=action_id,
        surface="inflated_torch_cpu",
        survived=survived,
        live_wrong_to_target_count=_live_wrong_to_target_count(live_birth_payload),
        birth_class=birth_class,
        region_pixels=region_pixels,
        region_hard_won=region_hard_won,
        target_hard_lost=max(0, initial_in_region_target - region_hard_won),
        net_target_support_delta=net_target_support_delta,
        initial_in_region_target=initial_in_region_target,
        region_unsolved=int(support["region_unsolved"]),
        region_debt_units=region_debt_units,
        margin_stats=support["stats"],
        fakequant_bits=None,
        total_scored_pixels=int(support["total_scored_pixels"]),
        worst_region=worst_region,
        pose_compensation_survival=pose_compensation_survival,
        surface_meta={
            **surface_meta,
            "local_replay_summary_schema": str(summary.get("schema") or ""),
            "local_replay_report_path": summary.get("report_path"),
            "local_replay_stdout_path": summary.get("stdout_path"),
            "local_replay_stderr_path": summary.get("stderr_path"),
            "local_replay_axis_tag": summary.get("axis_tag"),
            "local_replay_device": summary.get("device"),
            "local_replay_archive_zip_path": summary.get("archive_zip_path"),
            "local_replay_archive_zip_sha256": (
                _sha256_file_or_none(archive_path) if archive_path.is_file() else None
            ),
            "local_replay_archive_bytes": summary.get("archive_bytes"),
            "local_replay_inflated_dir": inflated_dir.as_posix(),
            "local_replay_inflated_dir_cleanup": cleanup,
            "inflated_torch_cpu_source": "upstream_evaluate_sh_retained_raw_tensor",
            "inflated_torch_cpu_tensor_dataset_shape_contract": [
                "N",
                int(camera_size_wh[1]),
                int(camera_size_wh[0]),
                RGB_CHANNELS,
            ],
            "inflated_torch_cpu_memmap_full_video_materialized": False,
        },
    )


def measure_birth_decoder_section_shadow(
    model: Any,
    *,
    scorer_teacher: Any,
    archive_decoded_state: Mapping[str, Any],
    section: str,
    target_labels: Any,
    live_birth_payload: Mapping[str, Any],
    pair_indices: Any,
    fakequant_logits_bhwc: np.ndarray,
    parseback_logits_bhwc: np.ndarray,
    live_logits_bhwc: np.ndarray | None = None,
    strong_margin_floor: float = 0.1,
) -> dict[str, Any]:
    """Shadow ONE decoder section with its exact archive-decoded weights.

    Imports only ``section``'s archive-decoded weights into the LIVE MLX model
    (via the round-trip-identity-proven ``import_torch_state_dict``), keeps
    every other section live, renders the same pairs, and prices the named
    birth region with the v2 L-set certificate.  This isolates whether the
    EXACT archive quantization of ``section`` (not 8-bit symmetric fakequant)
    is what moves the won pixels across the SegNet wall at parse-back.

    The original section weights are snapshotted via ``export_state_dict`` and
    restored after, so the model is unchanged on return.  ``causal_hint`` from
    the certificate is ADVISORY; the row reports exactly which keys were
    swapped (``applied_keys``/``applied_key_count``/``applied_keys_sha256``) so
    a prefix typo cannot masquerade as a measurement.
    """

    _require_mlx()
    import mlx.core as mx

    import_fn = getattr(model, "import_torch_state_dict", None)
    export_fn = getattr(model, "export_state_dict", None)
    if not callable(import_fn) or not callable(export_fn):
        raise BirthSurvivalError(
            "model must expose import_torch_state_dict + export_state_dict "
            "(round-trip identity gate) for the decoder-section shadow"
        )
    action_id = _live_action_id(live_birth_payload)
    worst_region = _live_worst_region(live_birth_payload)
    birth_class = int(worst_region["class_index"])
    region_mask_np, _region_pixels = reconstruct_birth_region_mask(target_labels, worst_region)
    idx = mx.array(pair_indices, dtype=mx.int32)  # type: ignore[union-attr]
    if idx.ndim != 1 or int(idx.shape[0]) <= 0:
        raise BirthSurvivalError("pair_indices must be a non-empty rank-1 tensor")

    # Snapshot ONLY this section's current values (full export; restore subset).
    full_snapshot = export_fn()
    section_keys = sorted(
        k for k in archive_decoded_state if k == section or str(k).startswith(f"{section}.")
    )
    if not section_keys:
        raise BirthSurvivalError(
            f"archive_decoded_state has no keys for section {section!r}"
        )
    snapshot_subset = {k: full_snapshot[k] for k in section_keys if k in full_snapshot}
    if not snapshot_subset:
        raise BirthSurvivalError(
            f"section {section!r} not present in live model export; cannot shadow"
        )

    import hashlib

    applied_keys: list[str] = []
    roundtrip_verified = False
    try:
        # Verify the section import is identity-safe on THIS live model first:
        # export(section) -> import(section) -> export(section) must be byte-equal.
        identity_state = {k: full_snapshot[k] for k in snapshot_subset}
        applied_keys = list(import_fn(archive_decoded_state, sections=(section,), strict=False))
        # HARDENING 1 (applied-key exactness): the import must apply EXACTLY the
        # live section keys.  A prefix typo, or a key present in the archive but
        # absent in the live model (or vice versa), would otherwise produce a
        # partial, misleading shadow row.  Fail closed.
        expected_keys = sorted(snapshot_subset)
        if sorted(applied_keys) != expected_keys:
            raise BirthSurvivalError(
                f"decoder-section shadow for {section!r}: applied_keys "
                f"{sorted(applied_keys)!r} != expected section keys "
                f"{expected_keys!r}"
            )
        shadow_export = export_fn()
        # Confirm the swap actually moved the named keys and nothing else.
        for k in full_snapshot:
            if k in snapshot_subset:
                continue
            if not np.array_equal(np.asarray(full_snapshot[k]), np.asarray(shadow_export[k])):
                raise BirthSurvivalError(
                    f"decoder-section shadow for {section!r} perturbed out-of-section key {k!r}"
                )
        # HARDENING 2 (per-row round-trip equality): every applied key's exported
        # (torch-layout) value must round-trip-equal the archive-decoded value it
        # was imported from.  This is the per-row, real-artifact confirmation of
        # the export/import section-transpose identity (the synthetic round-trip
        # identity test proves the code path; this proves it on THESE values).
        for k in applied_keys:
            if not np.array_equal(
                np.asarray(shadow_export[k]), np.asarray(archive_decoded_state[k])
            ):
                raise BirthSurvivalError(
                    f"decoder-section shadow for {section!r}: applied key {k!r} "
                    "export does not round-trip-equal archive_decoded_state "
                    "(export/import section transpose mismatch)"
                )
        # Round-trip identity check on the section itself (restore->reimport equal).
        import_fn(identity_state, sections=(section,), strict=False)
        reverted = export_fn()
        roundtrip_verified = all(
            np.array_equal(np.asarray(identity_state[k]), np.asarray(reverted[k]))
            for k in identity_state
        )
        # Re-apply the archive-decoded section for the actual measurement.
        import_fn(archive_decoded_state, sections=(section,), strict=False)
        support = _region_support_on_surface(
            model,
            scorer_teacher=scorer_teacher,
            pair_indices=idx,
            region_mask_np=region_mask_np,
            birth_class=birth_class,
        )
        shadow_logits = _candidate_logits_np(scorer_teacher, _predict_frame1_nhwc01(model, idx))
    finally:
        import_fn(snapshot_subset, sections=(section,), strict=False)
        mx.eval(model.parameters())  # type: ignore[union-attr]

    cert = lset_subset_conditioned_margin_certificate(
        region_mask_bhw=region_mask_np,
        birth_class=birth_class,
        evaluated_logits_bhwc=shadow_logits,
        fakequant_logits_bhwc=fakequant_logits_bhwc,
        parseback_logits_bhwc=parseback_logits_bhwc,
        live_logits_bhwc=live_logits_bhwc,
        strong_margin_floor=float(strong_margin_floor),
    )
    region_hard_won = int(support["region_hard_won"])
    live_wrong = _live_wrong_to_target_count(live_birth_payload)
    retention = (
        None if live_wrong is None or live_wrong <= 0 else float(region_hard_won) / float(live_wrong)
    )
    # GPT row hardening 3: retention measured against the fakequant surface (the
    # surface that DID win the L-set) is the directly decision-forcing number —
    # it answers "of the pixels fakequant won, how many survive THIS section's
    # exact archive decode?".  vs-live mixes in pixels live never won.
    fakequant_won = cert.get("fakequant_won_count")
    parseback_won = cert.get("parseback_won_count")
    shadow_retention_vs_fakequant = (
        None
        if not isinstance(fakequant_won, (int, float)) or fakequant_won <= 0
        else float(region_hard_won) / float(fakequant_won)
    )
    applied_sha = hashlib.sha256(
        "|".join(
            f"{k}={hashlib.sha256(np.ascontiguousarray(np.asarray(archive_decoded_state[k])).tobytes()).hexdigest()}"
            for k in sorted(applied_keys)
        ).encode("utf-8")
    ).hexdigest()
    collapsed = (
        retention is not None and retention < float(MIN_SCORER_EFFECT_RETENTION_FOR_SURVIVAL)
    )
    row = {
        "schema": DECODER_SECTION_SHADOW_ABLATION_SCHEMA,
        "family": "hinerv",
        "action_id": str(action_id),
        "surface": f"{section}_archive_decode_shadow",
        "section": str(section),
        "requested_section": str(section),
        "applied_keys": list(applied_keys),
        "applied_key_count": len(applied_keys),
        "applied_keys_sha256": applied_sha,
        "section_import_roundtrip_identity_verified": bool(roundtrip_verified),
        "section_archive_decode_applied_exact": True,
        "birth_class_index": birth_class,
        "shadow_wrong_to_target": region_hard_won,
        "shadow_wrong_to_target_retention": retention,
        "shadow_retention_vs_live": retention,
        "shadow_retention_vs_fakequant": shadow_retention_vs_fakequant,
        "fakequant_wrong_to_target": fakequant_won,
        "parseback_wrong_to_target": parseback_won,
        "live_wrong_to_target": live_wrong,
        "region_score_debt_units": _region_debt_units(
            int(support["region_unsolved"]), int(support["total_scored_pixels"])
        ),
        "lset_certificate": cert,
        "causal_hint": cert["causal_hint"],
        "causal_hint_is_advisory": True,
        "section_collapses_l": bool(collapsed),
        "first_failed_surface": (
            f"{section}_archive_decode_shadow" if collapsed else None
        ),
        "human_visual_fidelity_objective": False,
        "authority": _NON_AUTHORITY_MARKER,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
    }
    return row


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
    fakequant_bits: int | None,
    total_scored_pixels: int,
    worst_region: Mapping[str, Any],
    pose_compensation_survival: Mapping[str, Any] | None = None,
    surface_meta: Mapping[str, Any] | None = None,
    live_wrong_to_target_count: int | None = None,
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
    pose_payload = dict(pose_compensation_survival or {})
    pose_required = bool(pose_payload.get("required"))
    pose_survived = None if not pose_required else bool(pose_payload.get("survived"))
    blockers = [str(item) for item in pose_payload.get("blockers") or []]
    surface_name = str(surface)
    live_wrong_to_target = (
        None if live_wrong_to_target_count is None else int(live_wrong_to_target_count)
    )
    surface_wrong_to_target = int(region_hard_won)
    retention_ratio = (
        None
        if live_wrong_to_target is None or live_wrong_to_target <= 0
        else float(surface_wrong_to_target) / float(live_wrong_to_target)
    )
    retention_floor = float(MIN_SCORER_EFFECT_RETENTION_FOR_SURVIVAL)
    retention_required = bool(
        surface_name in {"fakequant_mlx", "archive_roundtrip_shadow", "parseback_mlx", "inflated_torch_cpu"}
    )
    scorer_effect_survived = bool(survived)
    first_failed_surface = None
    target_margin_p10 = _target_margin_value(
        margin_stats,
        "target_margin_p10",
        "margin_p90",
    )
    target_margin_floor_satisfied = (
        None
        if target_margin_p10 is None
        else bool(target_margin_p10 >= TARGET_MARGIN_SAFETY_FLOOR)
    )
    if retention_required:
        if live_wrong_to_target is None or live_wrong_to_target <= 0:
            scorer_effect_survived = False
            blockers.append("birth_survival_live_wrong_to_target_missing_for_retention")
            first_failed_surface = f"{surface_name}_live_effect_reference_missing"
        if target_margin_floor_satisfied is None:
            scorer_effect_survived = False
            blockers.append(
                "birth_survival_target_margin_certificate_missing"
            )
            first_failed_surface = first_failed_surface or f"{surface_name}_target_margin_certificate_missing"
        elif not target_margin_floor_satisfied:
            scorer_effect_survived = False
            blockers.append(
                "hinerv_birth_parseback_margin_floor_failed"
                if surface_name == "parseback_mlx"
                else f"hinerv_birth_{surface_name}_margin_floor_failed"
            )
            first_failed_surface = (
                first_failed_surface
                or (
                    "parseback_margin_floor"
                    if surface_name == "parseback_mlx"
                    else f"{surface_name}_margin_floor"
                )
            )
        if retention_ratio is None or retention_ratio < retention_floor:
            scorer_effect_survived = False
            blockers.append(
                "hinerv_birth_parseback_scorer_effect_collapse"
                if surface_name == "parseback_mlx"
                else f"hinerv_birth_{surface_name}_scorer_effect_collapse"
            )
            first_failed_surface = first_failed_surface or (
                "parseback_scorer_effect_collapse"
                if surface_name == "parseback_mlx"
                else f"{surface_name}_scorer_effect_collapse"
            )
    survived = bool(survived and scorer_effect_survived)
    margin_certificate = _target_margin_certificate(
        action_id=action_id,
        surface=surface_name,
        birth_class=birth_class,
        region_pixels=region_pixels,
        margin_stats=margin_stats,
        worst_region=worst_region,
        live_wrong_to_target=live_wrong_to_target,
        surface_wrong_to_target=surface_wrong_to_target,
        retention_ratio=retention_ratio,
        retention_floor=retention_floor,
        retention_required=retention_required,
        scorer_effect_survived=scorer_effect_survived,
        first_failed_surface=first_failed_surface,
    )
    row: dict[str, Any] = {
        "schema": BIRTH_SURVIVAL_SCHEMA,
        "action_id": str(action_id),
        "surface": surface_name,
        "survived": bool(survived),
        "scorer_effect_survived": bool(scorer_effect_survived),
        "live_wrong_to_target": live_wrong_to_target,
        "live_wrong_to_target_count": live_wrong_to_target,
        "surface_wrong_to_target_count": surface_wrong_to_target,
        f"{surface_name}_wrong_to_target_count": surface_wrong_to_target,
        "scorer_effect_retention_ratio": retention_ratio,
        "wrong_to_target_retention_ratio": retention_ratio,
        "scorer_effect_retention_floor": retention_floor,
        "scorer_effect_retention_required": retention_required,
        "first_failed_surface": first_failed_surface,
        "family": "hinerv",
        "birth_class_index": int(birth_class),
        "region_pixel_count": int(region_pixels),
        "region_hard_won_count": int(region_hard_won),
        "region_unsolved_pixel_count": int(region_unsolved),
        "initial_in_region_target_count": int(initial_in_region_target),
        "region_score_debt_units": float(region_debt_units),
        "region_margin_mean": float(margin_stats["margin_mean"]),
        "region_margin_min": float(margin_stats["margin_min"]),
        "target_margin_min": margin_certificate["target_margin_min"],
        "target_margin_p10": margin_certificate["target_margin_p10"],
        "target_margin_p50": margin_certificate["target_margin_p50"],
        "target_margin_mean": margin_certificate["target_margin_mean"],
        "target_margin_floor": float(TARGET_MARGIN_SAFETY_FLOOR),
        "target_margin_floor_satisfied": target_margin_floor_satisfied,
        "target_margin_certificate": margin_certificate,
        "region_hard_ratio": float(margin_stats["region_hard_ratio"]),
        "total_scored_pixels": int(total_scored_pixels),
        # Canonical receiver-surface aliases the gate's support breakdown reads.
        "receiver_surface_target_hard_won_count": int(region_hard_won),
        "receiver_surface_target_hard_lost_count": int(target_hard_lost),
        "receiver_surface_net_target_support_delta": int(net_target_support_delta),
        # Top-level transition aliases (the gate merges these too).
        **transitions,
        "argmax_transitions": transitions,
        "pose_compensation_required": pose_required,
        "pose_compensation_survived": pose_survived,
        "pose_compensation_survival": pose_payload or None,
        "blockers": blockers,
        "worst_region": dict(worst_region),
        "human_visual_fidelity_objective": False,
        "authority": _NON_AUTHORITY_MARKER,
    }
    if surface_name == "fakequant_mlx":
        row["fakequant_scorer_effect_survived"] = bool(scorer_effect_survived)
        row["fakequant_target_margin_certificate"] = margin_certificate
        row["fakequant_wrong_to_target"] = surface_wrong_to_target
        row["fakequant_wrong_to_target_count"] = surface_wrong_to_target
        row["fakequant_wrong_to_target_retention_ratio"] = retention_ratio
    elif surface_name == "parseback_mlx":
        row["parseback_payload_survived"] = True
        row["parseback_action_program_survived"] = None
        row["parseback_scorer_effect_survived"] = bool(scorer_effect_survived)
        row["parseback_target_margin_certificate"] = margin_certificate
        row["parseback_wrong_to_target"] = surface_wrong_to_target
        row["parseback_wrong_to_target_count"] = surface_wrong_to_target
        row["parseback_wrong_to_target_retention_ratio"] = retention_ratio
    elif surface_name == "inflated_torch_cpu":
        row["inflate_payload_survived"] = True
        row["inflate_scorer_effect_survived"] = bool(scorer_effect_survived)
        row["inflate_target_margin_certificate"] = margin_certificate
        row["inflate_wrong_to_target"] = surface_wrong_to_target
        row["inflate_wrong_to_target_count"] = surface_wrong_to_target
        row["inflate_wrong_to_target_retention_ratio"] = retention_ratio
    elif surface_name == "archive_roundtrip_shadow":
        row["archive_roundtrip_shadow_scorer_effect_survived"] = bool(scorer_effect_survived)
        row["archive_roundtrip_shadow_target_margin_certificate"] = margin_certificate
        row["archive_roundtrip_shadow_wrong_to_target"] = surface_wrong_to_target
        row["archive_roundtrip_shadow_wrong_to_target_count"] = surface_wrong_to_target
        row["archive_roundtrip_shadow_wrong_to_target_retention_ratio"] = retention_ratio
    if fakequant_bits is not None:
        row["fakequant_bits"] = int(fakequant_bits)
    if surface_meta:
        # Nested (not spread) so consumers read a single typed block and the
        # row top-level schema stays stable across surfaces.
        row["surface_meta"] = dict(surface_meta)
    return row


def _blocked_survival_row(
    *,
    action_id: str,
    surface: str,
    blocker: str | None = None,
    reason: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Typed FAIL-CLOSED row for surfaces this producer cannot stand up.

    Deliberately carries NO ``survived`` boolean: the gate finds no satisfying
    survival row for the surface and keeps emitting
    ``birth_survival_receipt_missing:<surface>``.  Honesty over coverage — a
    fabricated ``survived=True`` here would be a fake implementation.
    """

    blocker = blocker or _SURVIVAL_SURFACE_BLOCKER.get(str(surface), "birth_survival_surface_unsupported")
    row = {
        "schema": BIRTH_SURVIVAL_BLOCKED_SCHEMA,
        "action_id": str(action_id),
        "surface": str(surface),
        "blocked": True,
        "blocker": blocker,
        "blockers": [blocker],
        "reason": reason or (
            "scoped MLX survival producer cannot build a faithful "
            f"{surface} surface (no archive parse-back / torch inflate runtime here)"
        ),
        "human_visual_fidelity_objective": False,
        "authority": _NON_AUTHORITY_MARKER,
    }
    if isinstance(extra, Mapping):
        row.update(dict(extra))
    return row


__all__ = [
    "BIRTH_HYSTERESIS_SCHEMA",
    "BIRTH_SURVIVAL_BLOCKED_SCHEMA",
    "BIRTH_SURVIVAL_SCHEMA",
    "BLOCKED_SURVIVAL_SURFACES",
    "FAITHFUL_SURVIVAL_SURFACES",
    "BirthSurvivalError",
    "measure_birth_hysteresis",
    "measure_birth_inflated_torch_cpu_survival_from_local_replay",
    "measure_birth_parseback_survival_from_report",
    "measure_birth_survival",
    "reconstruct_birth_region_mask",
]
