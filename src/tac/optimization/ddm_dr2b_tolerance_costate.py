# SPDX-License-Identifier: MIT
"""Pure laws for DDM DR2b tolerance and receiver-costate measurements.

The measurement tool evaluates one counted E2 coordinate in its canonical
batch-16 receiver window.  Because the runtime packet is pair-separable and
the edit is required to affect exactly one pair, the local distortion delta can
be rebased exactly onto the already measured n600 E2 baseline.  This avoids
pretending that an n16 absolute score is an n600 result while also avoiding a
redundant replay of 584 receiver-invariant pairs.

Nothing in this module transfers E2 coordinate tolerances to the SDWL1 fact
tensor.  Such a transfer requires an explicit, SHA-bound coordinate crosswalk.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import brotli

from tac.canonical_equations.ddm_dr2_description_layer_laws_20260723 import (
    MODE_RACE_CHALLENGER_BYTES,
    SDWL1_EXACT_DESCRIPTION_BYTES,
    STRICT_SUB015_CAP_BYTES_POSE_HELD,
)
from tac.optimization.ddm_min_description_contract import (
    LayerHome,
    StreamType,
    TypedStreamTag,
)
from tac.optimization.ddm_runtime_sensitivity import SCORE_BYTE_DUAL

SCHEMA = "ddm_dr2b_tolerance_costate.v1"
N_PAIRS = 600
SCORER_BATCH_SIZE = 16


class DDMDR2BMeasurementError(ValueError):
    """A tolerance, custody, or exact-rebase contract failed closed."""


def exact_n600_rebase(
    *,
    baseline_d_seg: float,
    baseline_d_pose: float,
    window_d_seg_before: float,
    window_d_seg_after: float,
    window_d_pose_before: float,
    window_d_pose_after: float,
    window_pair_count: int,
    delta_bytes: int,
) -> dict[str, float | int | bool]:
    """Rebase one pair-local edit onto a measured n600 baseline.

    The caller must separately prove that only one pair changed and that the
    window was evaluated at the canonical batch size.  Distortions are means,
    so the exact global delta is the window delta times ``window/600``.
    """

    values = (
        baseline_d_seg,
        baseline_d_pose,
        window_d_seg_before,
        window_d_seg_after,
        window_d_pose_before,
        window_d_pose_after,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in values
    ):
        raise DDMDR2BMeasurementError("distortion inputs must be finite nonnegative numbers")
    if isinstance(window_pair_count, bool) or window_pair_count != SCORER_BATCH_SIZE:
        raise DDMDR2BMeasurementError("exact n600 rebase requires one canonical batch-16 window")
    if isinstance(delta_bytes, bool) or not isinstance(delta_bytes, int):
        raise DDMDR2BMeasurementError("delta_bytes must be an integer")

    scale = window_pair_count / N_PAIRS
    delta_d_seg = (float(window_d_seg_after) - float(window_d_seg_before)) * scale
    delta_d_pose = (float(window_d_pose_after) - float(window_d_pose_before)) * scale
    after_d_seg = float(baseline_d_seg) + delta_d_seg
    after_d_pose = float(baseline_d_pose) + delta_d_pose
    if after_d_seg < 0.0 or after_d_pose < 0.0:
        raise DDMDR2BMeasurementError("rebased distortion became negative")

    seg_term = 100.0 * delta_d_seg
    pose_term = math.sqrt(10.0 * after_d_pose) - math.sqrt(10.0 * float(baseline_d_pose))
    distortion_delta = seg_term + pose_term
    rate_term = SCORE_BYTE_DUAL * delta_bytes
    joint_delta = distortion_delta + rate_term
    bytes_saved = max(0, -delta_bytes)
    bytes_added = max(0, delta_bytes)
    return {
        "n_pairs": N_PAIRS,
        "exact_pair_local_rebase": True,
        "delta_d_seg": delta_d_seg,
        "delta_d_pose": delta_d_pose,
        "d_seg": after_d_seg,
        "d_pose": after_d_pose,
        "delta_bytes": delta_bytes,
        "seg_term": seg_term,
        "pose_term": pose_term,
        "distortion_delta": distortion_delta,
        "rate_term": rate_term,
        "joint_delta": joint_delta,
        "bytes_saved": bytes_saved,
        "bytes_added": bytes_added,
        "distortion_cost_per_byte_saved": (None if bytes_saved == 0 else distortion_delta / bytes_saved),
        "distortion_gain_per_byte_added": (None if bytes_added == 0 else -distortion_delta / bytes_added),
        "rate_break_even_score_per_byte": SCORE_BYTE_DUAL,
        "reverse_waterfill_admissible": bool(joint_delta < 0.0),
    }


def head_flip_distance(*, margin: float, head_normal_norm: float) -> float:
    """Return ``|margin| / ||w_c-w_c'||`` on the frozen head metric."""

    for name, value in (
        ("margin", margin),
        ("head_normal_norm", head_normal_norm),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise DDMDR2BMeasurementError(f"{name} must be finite")
    if float(head_normal_norm) <= 0.0:
        raise DDMDR2BMeasurementError("head_normal_norm must be positive")
    return abs(float(margin)) / float(head_normal_norm)


def frequency_band_admission(
    *,
    exact_r_transfer_zero: bool,
    emitted_description_bytes: int,
    flip_distance: float | None = None,
    delta_d_pose: float | None = None,
) -> dict[str, Any]:
    """Guard free truncation and measured pricing of one spectral band."""

    if not isinstance(exact_r_transfer_zero, bool):
        raise DDMDR2BMeasurementError("exact_r_transfer_zero must be boolean")
    if (
        isinstance(emitted_description_bytes, bool)
        or not isinstance(emitted_description_bytes, int)
        or emitted_description_bytes < 0
    ):
        raise DDMDR2BMeasurementError("emitted_description_bytes must be a nonnegative integer")
    if exact_r_transfer_zero:
        if emitted_description_bytes != 0:
            raise DDMDR2BMeasurementError("an exact-R-null band must emit zero description bytes")
        if flip_distance is not None or delta_d_pose is not None:
            raise DDMDR2BMeasurementError("an exact-R-null band cannot carry scorer-debt measurements")
        return {
            "status": "ADMITTED_EXACT_R_NULL_TRUNCATION",
            "description_bytes": 0,
            "distortion_delta": 0.0,
            "typed_stream_tag": TypedStreamTag(
                type=StreamType.GAUGE,
                layer_home=LayerHome.L3_RASTER,
                evaluate_py_recursion_level_cited=(
                    "L3_raster exact ker(R) -> zero L4_scorer_feature effect"
                ),
                counted_bytes=0,
                free_receiver_code=True,
            ).to_dict(),
            "rate_price_admitted": False,
            "first_rung": True,
        }
    if emitted_description_bytes == 0:
        raise DDMDR2BMeasurementError("a scorer-visible band cannot claim zero description bytes")
    for name, value in (("flip_distance", flip_distance), ("delta_d_pose", delta_d_pose)):
        if (
            value is None
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise DDMDR2BMeasurementError(f"scorer-visible band requires finite nonnegative {name}")
    return {
        "status": "MEASURED_SCORER_VISIBLE_BAND",
        "description_bytes": emitted_description_bytes,
        "flip_distance": float(flip_distance),
        "delta_d_pose": float(delta_d_pose),
        "typed_stream_tag": TypedStreamTag(
            type=StreamType.FIBER,
            layer_home=LayerHome.L3_RASTER,
            evaluate_py_recursion_level_cited=(
                "L3_raster -> measured L4_scorer_feature -> L5_verdict"
            ),
            counted_bytes=emitted_description_bytes,
            free_receiver_code=True,
        ).to_dict(),
        "rate_price_admitted": True,
        "first_rung": True,
    }


def ordered_redundancy_matrix(
    streams: Mapping[str, bytes],
    *,
    decode_order: Sequence[str],
) -> list[dict[str, Any]]:
    """Measure ordered Brotli-Q11 conditional bytes for decoded streams.

    ``conditioned_bytes`` is ``B(A||B)-B(A)``.  It is a coder diagnostic, not
    permission to subtract overlap from an archive unless the production coder
    implements the same condition.
    """

    order = tuple(decode_order)
    if (
        not order
        or len(set(order)) != len(order)
        or set(order) != set(streams)
        or any(not isinstance(streams[name], bytes) for name in order)
    ):
        raise DDMDR2BMeasurementError("streams and decode_order must be one complete unique byte mapping")
    standalone = {name: len(brotli.compress(streams[name], quality=11)) for name in order}
    rows: list[dict[str, Any]] = []
    for conditioner in order:
        for stream in order:
            if conditioner == stream:
                continue
            conditioned = (
                len(
                    brotli.compress(
                        streams[conditioner] + streams[stream],
                        quality=11,
                    )
                )
                - standalone[conditioner]
            )
            rows.append(
                {
                    "conditioner": conditioner,
                    "stream": stream,
                    "standalone_bytes": standalone[stream],
                    "conditioned_bytes": conditioned,
                    "redundancy_bytes": standalone[stream] - conditioned,
                    "first_rung": True,
                    "interpretation": ("Brotli-Q11 diagnostic only; no archive-byte credit"),
                }
            )
    return rows


def exact_layer_controls() -> dict[str, Any]:
    """Return the settled DR2 exact-layer controls without remeasurement."""

    return {
        "sdwl1_exact_description_bytes": SDWL1_EXACT_DESCRIPTION_BYTES,
        "mode_race_challenger_bytes": MODE_RACE_CHALLENGER_BYTES,
        "mode_race_delta_bytes": (MODE_RACE_CHALLENGER_BYTES - SDWL1_EXACT_DESCRIPTION_BYTES),
        "strict_sub015_cap_bytes_pose_held": (STRICT_SUB015_CAP_BYTES_POSE_HELD),
        "description_only_headroom_bytes": (STRICT_SUB015_CAP_BYTES_POSE_HELD - SDWL1_EXACT_DESCRIPTION_BYTES),
        "first_rung": True,
        "epistemic_status": "MEASURED_SETTLED_INPUT",
        "verdict_scope": (
            "FORMULATION-at-exact-SDWL1-layer only; lossy mode race remains "
            "open until an SDWL1-to-E2 coordinate crosswalk is measured."
        ),
    }


def require_description_crosswalk(
    crosswalk: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    """Fail closed rather than transfer E2 tolerances into SDWL1 bytes."""

    if crosswalk is None:
        raise DDMDR2BMeasurementError("missing SHA-bound SDWL1-fact to E2-runtime coordinate crosswalk")
    required = {
        "schema",
        "sdwl1_receipt_sha256",
        "e2_manifest_sha256",
        "coordinate_rows",
    }
    if set(crosswalk) != required or not crosswalk["coordinate_rows"]:
        raise DDMDR2BMeasurementError("description crosswalk is incomplete")
    return crosswalk


def rank_costate_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Rank measured perturbations by reduced cost, then realized efficiency."""

    normalized: list[dict[str, Any]] = []
    for row in rows:
        n600 = row.get("n600_rebase")
        if not isinstance(n600, Mapping):
            raise DDMDR2BMeasurementError("costate row lacks n600_rebase")
        joint = n600.get("joint_delta")
        if isinstance(joint, bool) or not isinstance(joint, (int, float)) or not math.isfinite(float(joint)):
            raise DDMDR2BMeasurementError("costate row joint_delta is not finite")
        normalized.append(dict(row))

    def key(row: Mapping[str, Any]) -> tuple[float, float, str]:
        n600 = row["n600_rebase"]
        efficiency = n600.get("distortion_gain_per_byte_added")
        efficiency_key = (
            -float(efficiency)
            if isinstance(efficiency, (int, float)) and not isinstance(efficiency, bool)
            else math.inf
        )
        return (
            float(n600["joint_delta"]),
            efficiency_key,
            str(row.get("probe_id", "")),
        )

    result = sorted(normalized, key=key)
    for index, row in enumerate(result, start=1):
        row["costate_rank"] = index
    return result


__all__ = [
    "N_PAIRS",
    "SCHEMA",
    "SCORER_BATCH_SIZE",
    "DDMDR2BMeasurementError",
    "exact_layer_controls",
    "exact_n600_rebase",
    "frequency_band_admission",
    "head_flip_distance",
    "ordered_redundancy_matrix",
    "rank_costate_rows",
    "require_description_crosswalk",
]
