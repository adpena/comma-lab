# SPDX-License-Identifier: MIT
"""Path-coded support/action candidates for inverse-evaluate planning.

This module is deliberately small: it turns already-measured hard-region
surfaces into archive-executable support payloads and non-promotable
``ActionEffect`` rows.  It does not score an archive, invent SegNet wall
crossing, or bypass parse-back/inflate gates.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np

from tac.analysis.action_effect import ActionEffect
from tac.analysis.inverse_scorer_actions import build_candidate_queue
from tac.optimization.evaluator_action_waterfill import CandidateActionEvaluation

PATH_ACTION_PRODUCER_SCHEMA = "tac.path_action_producer.v1"
PATH_SUPPORT_SCHEMA = "tac.path_tube_support.v1"
PATH_ACTION_CANDIDATE_SCHEMA = "tac.path_action_candidate.v1"
TEMPORAL_PATH_SCHEMA = "tac.temporal_path_payload.v1"
PATH_SUPPORT_MAGIC = b"PTUB1"
_HEADER_FMT = "<5sHHBHHHII"
_POINT_FMT = "<HH"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_TEMPORAL_QUANT_SCALE = 1_000_000

BLOCKER_PATH_SUPPORT_EMPTY = "path_action_support_empty"
BLOCKER_PATH_SUPPORT_NOT_BIRTH = "path_action_support_without_wrong_to_target_is_not_birth"
BLOCKER_PATH_ACTION_PARSEBACK_MISSING = "path_action_parseback_survival_missing"
BLOCKER_PATH_ACTION_INFLATE_MISSING = "path_action_inflate_survival_missing"
BLOCKER_PATH_POSE_TRAJECTORY_EMPTY = "path_action_pose_trajectory_empty"
BLOCKER_PATH_SELECTOR_TRAJECTORY_EMPTY = "path_action_selector_trajectory_empty"
BLOCKER_PATH_TRAJECTORY_NO_RECEIVER_PROOF = "path_action_trajectory_receiver_proof_missing"

RENT_EVALUATION_SCHEMA = "tac.path_action_rent_evaluation.v1"
RENT_BLOCKER_BASE_ARCHIVE_SHA_MISSING = "rent_evaluation_base_archive_sha256_missing"
RENT_BLOCKER_SCORE_ENDPOINTS_MISSING = "rent_evaluation_exact_score_endpoints_missing"


def _finite(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def candidate_action_evaluation_row(
    effect: ActionEffect,
    *,
    base_archive_sha256: str | None,
    base_scorer_state_hash: str | None = None,
) -> dict[str, Any]:
    """Emit the base-bound rent-law row for one produced ``ActionEffect``.

    Every produced atom must PAY RENT: it is admissible only when ``S(base +
    action) < S(base)`` (lower-is-better) AND its scorer effect survives
    parse-back, measured against the base it was scored against.  When the
    score endpoints or the base sha are missing the row is a typed blocker (the
    atom is NOT yet evaluable), never a fabricated ``pays_rent=True``.
    """

    base_sha = str(base_archive_sha256 or "").strip()
    blockers: list[str] = []
    if not base_sha:
        blockers.append(RENT_BLOCKER_BASE_ARCHIVE_SHA_MISSING)
    score_endpoints = all(
        _finite(getattr(effect, name))
        for name in ("old_d_seg", "new_d_seg", "old_d_pose", "new_d_pose")
    )
    if not score_endpoints or not (_finite(effect.old_bytes) and _finite(effect.new_bytes)):
        blockers.append(RENT_BLOCKER_SCORE_ENDPOINTS_MISSING)
    if blockers:
        return {
            "schema": RENT_EVALUATION_SCHEMA,
            "action_id": effect.action_id,
            "action_kind": effect.action_kind,
            "base_archive_sha256": base_sha or None,
            "evaluable": False,
            "pays_rent": False,
            "blockers": list(dict.fromkeys(blockers)),
            "promotion_eligible": False,
            "score_claim": False,
            "promotable": False,
        }
    evaluation = CandidateActionEvaluation(
        action_id=effect.action_id,
        action_kind=str(effect.action_kind),
        base_archive_sha256=base_sha,
        with_action_archive_sha256=str(
            effect.archive_sha256 or effect.payload_sha256 or base_sha
        ),
        d_seg_base=float(effect.old_d_seg),
        d_pose_base=float(effect.old_d_pose),
        bytes_base=int(effect.old_bytes),
        d_seg_with_action=float(effect.new_d_seg),
        d_pose_with_action=float(effect.new_d_pose),
        bytes_with_action=int(effect.new_bytes),
        scorer_effect_survived=(
            effect.parseback_survived is True
            and effect.inflate_survived is True
            and effect.fakequant_survived is not False
        ),
        base_scorer_state_hash=base_scorer_state_hash or effect.base_state_sha256,
        backend_wrong_to_target=(
            int(effect.wrong_to_target) if _finite(effect.wrong_to_target) else None
        ),
    )
    return evaluation.to_row()


@dataclass(frozen=True)
class PathTubeSupport:
    """Exact support represented as path tube plus residual add/remove pixels."""

    height: int
    width: int
    frame_index: int
    target_class: int
    pair_index: int
    tube_width: int
    path_yx: tuple[tuple[int, int], ...]
    residual_add_yx: tuple[tuple[int, int], ...]
    residual_remove_yx: tuple[tuple[int, int], ...]

    @property
    def path_point_count(self) -> int:
        return len(self.path_yx)

    @property
    def residual_add_count(self) -> int:
        return len(self.residual_add_yx)

    @property
    def residual_remove_count(self) -> int:
        return len(self.residual_remove_yx)

    def base_tube_mask(self) -> np.ndarray:
        mask = np.zeros((int(self.height), int(self.width)), dtype=bool)
        if not self.path_yx:
            return mask
        radius = max(0, math.ceil((int(self.tube_width) - 1) / 2.0))
        points = list(self.path_yx)
        if len(points) == 1:
            _paint_point(mask, points[0], radius)
            return mask
        for start, end in pairwise(points):
            _paint_segment(mask, start, end, radius)
        return mask

    def decode_mask(self) -> np.ndarray:
        mask = self.base_tube_mask()
        for y, x in self.residual_add_yx:
            mask[int(y), int(x)] = True
        for y, x in self.residual_remove_yx:
            mask[int(y), int(x)] = False
        return mask

    def encode_payload(self) -> bytes:
        if not self.path_yx:
            raise ValueError("path_tube support requires at least one path point")
        chunks = [
            struct.pack(
                _HEADER_FMT,
                PATH_SUPPORT_MAGIC,
                int(self.height),
                int(self.width),
                int(self.frame_index),
                int(self.target_class),
                int(self.pair_index),
                int(self.tube_width),
                int(self.path_point_count),
                int(self.residual_add_count),
            ),
            struct.pack("<I", int(self.residual_remove_count)),
        ]
        for seq in (self.path_yx, self.residual_add_yx, self.residual_remove_yx):
            for y, x in seq:
                chunks.append(struct.pack(_POINT_FMT, int(y), int(x)))
        return zlib.compress(b"".join(chunks), level=9)

    def as_dict(self) -> dict[str, Any]:
        decoded = self.decode_mask()
        payload = self.encode_payload()
        return {
            "schema": PATH_SUPPORT_SCHEMA,
            "support_encoding": "path_tube",
            "height": int(self.height),
            "width": int(self.width),
            "frame_index": int(self.frame_index),
            "target_class": int(self.target_class),
            "pair_index": int(self.pair_index),
            "tube_width": int(self.tube_width),
            "path_point_count": int(self.path_point_count),
            "residual_add_count": int(self.residual_add_count),
            "residual_remove_count": int(self.residual_remove_count),
            "support_cardinality": int(np.count_nonzero(decoded)),
            "support_sha256": support_mask_sha256(decoded),
            "support_encoded_bytes": len(payload),
            "support_decode_sha256": hashlib.sha256(decoded.astype(np.uint8).tobytes(order="C")).hexdigest(),
            "archive_executable": True,
            "profile": {
                "schema": "tac.path_tube_profile.v1",
                "kind": "uniform_tube",
                "tube_width": int(self.tube_width),
            },
            "path_yx": [list(point) for point in self.path_yx],
            "residual_add_yx": [list(point) for point in self.residual_add_yx],
            "residual_remove_yx": [list(point) for point in self.residual_remove_yx],
        }


def decode_path_tube_payload(payload: bytes) -> PathTubeSupport:
    raw = zlib.decompress(payload)
    if len(raw) < _HEADER_SIZE + 4:
        raise ValueError("path_tube payload too short")
    magic, height, width, frame_index, target_class, pair_index, tube_width, point_count, add_count = struct.unpack(
        _HEADER_FMT,
        raw[:_HEADER_SIZE],
    )
    if magic != PATH_SUPPORT_MAGIC:
        raise ValueError(f"bad path_tube magic: {magic!r}")
    offset = _HEADER_SIZE
    (remove_count,) = struct.unpack("<I", raw[offset : offset + 4])
    offset += 4

    def read_points(count: int) -> tuple[tuple[int, int], ...]:
        nonlocal offset
        out: list[tuple[int, int]] = []
        for _ in range(int(count)):
            if offset + struct.calcsize(_POINT_FMT) > len(raw):
                raise ValueError("truncated path_tube point stream")
            y, x = struct.unpack(_POINT_FMT, raw[offset : offset + struct.calcsize(_POINT_FMT)])
            offset += struct.calcsize(_POINT_FMT)
            out.append((int(y), int(x)))
        return tuple(out)

    path = read_points(point_count)
    adds = read_points(add_count)
    removes = read_points(remove_count)
    if offset != len(raw):
        raise ValueError("path_tube payload has trailing bytes")
    return PathTubeSupport(
        height=int(height),
        width=int(width),
        frame_index=int(frame_index),
        target_class=int(target_class),
        pair_index=int(pair_index),
        tube_width=int(tube_width),
        path_yx=path,
        residual_add_yx=adds,
        residual_remove_yx=removes,
    )


def build_path_action_candidates_from_arrays(
    *,
    target_labels_bhw: np.ndarray,
    candidate_argmax_bhw: np.ndarray,
    target_margin_bhw: np.ndarray | None = None,
    pair_indices: np.ndarray | None = None,
    target_class: int | None = None,
    batch_index: int = 0,
    frame_index: int = 1,
    rdp_epsilon: float = 2.0,
    base_archive_bytes: int = 0,
    base_archive_sha256: str | None = None,
    base_scorer_state_hash: str | None = None,
    old_d_seg: float = 0.0,
    old_d_pose: float = 0.0,
    authority: str = "batch_local_path_support",
) -> dict[str, Any]:
    labels = np.asarray(target_labels_bhw)
    argmax = np.asarray(candidate_argmax_bhw)
    if labels.shape != argmax.shape or labels.ndim != 3:
        raise ValueError(
            "target_labels_bhw and candidate_argmax_bhw must share BHW shape; "
            f"got {labels.shape} and {argmax.shape}"
        )
    margins = None if target_margin_bhw is None else np.asarray(target_margin_bhw)
    if margins is not None and margins.shape != labels.shape:
        raise ValueError(f"target_margin_bhw must match labels; got {margins.shape} vs {labels.shape}")
    batch = int(batch_index)
    if batch < 0 or batch >= labels.shape[0]:
        raise ValueError(f"batch_index out of range: {batch}")
    pair_index = _pair_index(pair_indices, batch)
    klass = int(target_class) if target_class is not None else _worst_unsolved_class(labels[batch], argmax[batch])
    unsolved = (labels[batch] == klass) & (argmax[batch] != klass)
    component = largest_connected_component(unsolved)
    blockers: list[str] = []
    if not np.any(component):
        blockers.append(BLOCKER_PATH_SUPPORT_EMPTY)
        return {
            "schema": PATH_ACTION_PRODUCER_SCHEMA,
            "candidate_count": 0,
            "blockers": blockers,
            "action_effects": [],
            "candidate_queue": [],
            "comparison": {},
        }
    support = path_tube_support_from_mask(
        component,
        pair_index=pair_index,
        frame_index=frame_index,
        target_class=klass,
        epsilon=rdp_epsilon,
    )
    decoded = support.decode_mask()
    support_hash = support_mask_sha256(decoded)
    support_dict = support.as_dict()
    if support_hash != support_dict["support_sha256"]:
        raise AssertionError("path support hash mismatch after decode")
    cardinality = int(np.count_nonzero(decoded))
    margin_delta = _support_margin_stat(margins[batch] if margins is not None else None, decoded)
    payload_bytes = int(support_dict["support_encoded_bytes"])
    comparison = support_encoding_comparison(component, path_payload_bytes=payload_bytes)
    effect = ActionEffect.build(
        action_id=_path_action_id(pair_index=pair_index, target_class=klass, support_sha256=support_hash),
        family="hinerv",
        action_kind="frame1_seg_margin_frontier_path",
        inverse_source="path_tube_segnet_margin_frontier",
        frame_index=int(frame_index),
        frame_incidence="seg_pose_joint",
        candidate_status="rejected",
        authority=authority,
        normalization_scope="batch_local",
        producer="path_action_producer",
        consumer="inverse_evaluate_candidate_queue",
        pair_ids=[pair_index],
        class_ids=[klass],
        region_ids=[f"b{batch}/c{klass}/path0"],
        payload_sections=["path_tube_support"],
        old_d_seg=float(old_d_seg),
        new_d_seg=float(old_d_seg),
        old_d_pose=float(old_d_pose),
        new_d_pose=float(old_d_pose),
        old_bytes=int(base_archive_bytes),
        new_bytes=int(base_archive_bytes) + payload_bytes,
        receiver_surface={
            "uint8_changed_pixels": 0,
            "seg_argmax_changed_pixels": 0,
            "seg_wrong_to_target_count": 0,
            "seg_target_hard_lost_count": 0,
            "seg_wrong_to_wrong_count": 0,
        },
        exact_score_decision="reject",
        parseback_survived=False,
        inflate_survived=False,
        fakequant_survived=None,
        hard_won_count=0,
        wrong_to_target=0,
        target_to_wrong=0,
        wrong_to_wrong=0,
        net_target_support_delta=0,
        argmax_changed_count_region=0,
        uint8_changed_count_region=0,
        segnet_margin_delta=margin_delta,
        support_source="target_debt_component_path_tube",
        support_cardinality=cardinality,
        support_sha256=support_hash,
        support_encoding="path_tube",
        support_encoded_bytes=payload_bytes,
        support_research_only=False,
        blockers=[
            BLOCKER_PATH_SUPPORT_NOT_BIRTH,
            BLOCKER_PATH_ACTION_PARSEBACK_MISSING,
            BLOCKER_PATH_ACTION_INFLATE_MISSING,
        ],
    )
    queue = build_candidate_queue([effect])
    rent_evaluation = candidate_action_evaluation_row(
        effect,
        base_archive_sha256=base_archive_sha256,
        base_scorer_state_hash=base_scorer_state_hash,
    )
    candidate = {
        "schema": PATH_ACTION_CANDIDATE_SCHEMA,
        "action_id": effect.action_id,
        "family": effect.family,
        "action_kind": effect.action_kind,
        "support": support_dict,
        "comparison": comparison,
        "candidate_action_evaluation": rent_evaluation,
        "blockers": list(effect.blockers),
        "promotion_eligible": False,
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }
    return {
        "schema": PATH_ACTION_PRODUCER_SCHEMA,
        "candidate_count": 1,
        "blockers": [],
        "action_effects": [effect],
        "candidate_queue": queue,
        "path_action_candidates": [candidate],
        "candidate_action_evaluations": [rent_evaluation],
        "comparison": comparison,
        "policy": {
            "path_actions_do_not_clear_launch_gate": True,
            "support_decode_hash_required": True,
            "wrong_to_target_required_for_birth": True,
            "rate_term_included_in_action_effect": True,
            "every_action_must_pay_rent": True,
        },
    }


def build_pose_temporal_path_candidates_from_arrays(
    *,
    pair_indices: Sequence[int] | np.ndarray,
    pose_residuals: Sequence[float] | np.ndarray | None = None,
    pose_action_profile: Sequence[float] | np.ndarray | None = None,
    rdp_epsilon: float = 0.25,
    base_archive_bytes: int = 0,
    base_archive_sha256: str | None = None,
    base_scorer_state_hash: str | None = None,
    old_d_seg: float | None = None,
    new_d_seg: float | None = None,
    old_d_pose: float | None = None,
    new_d_pose: float | None = None,
    authority: str = "batch_local_path_action",
) -> dict[str, Any]:
    """Emit one frame-0 PoseNet temporal-path ActionEffect when data exists."""

    pairs = _int_array(pair_indices, name="pair_indices")
    values_source = pose_action_profile if pose_action_profile is not None else pose_residuals
    values = _float_array(values_source, name="pose_action_profile_or_residuals")
    if pairs.size != values.size:
        raise ValueError(f"pair_indices and pose values must have same length; got {pairs.size} and {values.size}")
    active = np.isfinite(values) & (np.abs(values) > 0.0)
    if not np.any(active):
        return _empty_temporal_result(BLOCKER_PATH_POSE_TRAJECTORY_EMPTY)
    pairs = pairs[active]
    values = values[active]
    payload = _temporal_path_payload(
        family="hinerv",
        action_kind="frame0_pose_temporal_path",
        pair_indices=pairs,
        values=values,
        rdp_epsilon=rdp_epsilon,
    )
    encoded = _encode_temporal_payload(payload)
    payload_bytes = len(encoded)
    payload_sha256 = hashlib.sha256(encoded).hexdigest()
    old_bytes = int(base_archive_bytes)
    effect = ActionEffect.build(
        action_id=f"pose_path:p{int(pairs.min())}-{int(pairs.max())}:{payload_sha256[:16]}",
        family="hinerv",
        action_kind="frame0_pose_temporal_path",
        inverse_source="frame0_pose_temporal_path",
        frame_index=0,
        frame_incidence="pose_only",
        candidate_status="generated",
        authority=authority,
        normalization_scope="batch_local",
        producer="path_action_producer",
        consumer="inverse_evaluate_candidate_queue",
        pair_ids=tuple(int(v) for v in pairs.tolist()),
        payload_sections=["frame0_pose_temporal_path", f"temporal_payload_bytes={payload_bytes}"],
        old_d_seg=old_d_seg,
        new_d_seg=new_d_seg,
        old_d_pose=old_d_pose,
        new_d_pose=new_d_pose,
        old_bytes=old_bytes,
        new_bytes=old_bytes + payload_bytes,
        receiver_surface={
            "pose_output_l2_delta": float(np.linalg.norm(values)),
            "pose_temporal_path_active_pairs": int(pairs.size),
        },
        exact_score_decision="reject",
        parseback_survived=False,
        inflate_survived=False,
        fakequant_survived=None,
        posenet_input_delta_linf_pair=float(np.max(np.abs(values))),
        pose_output_l2_delta=float(np.linalg.norm(values)),
        payload_sha256=payload_sha256,
        blockers=[
            BLOCKER_PATH_TRAJECTORY_NO_RECEIVER_PROOF,
            BLOCKER_PATH_ACTION_PARSEBACK_MISSING,
            BLOCKER_PATH_ACTION_INFLATE_MISSING,
        ],
    )
    return _temporal_result(
        effect=effect,
        payload=payload,
        encoded=encoded,
        base_archive_sha256=base_archive_sha256,
        base_scorer_state_hash=base_scorer_state_hash,
    )


def build_selector_temporal_path_candidates_from_rows(
    rows: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    *,
    rdp_epsilon: float = 0.0,
    base_archive_bytes: int = 0,
    base_archive_sha256: str | None = None,
    base_scorer_state_hash: str | None = None,
    old_d_seg: float | None = None,
    new_d_seg: float | None = None,
    old_d_pose: float | None = None,
    new_d_pose: float | None = None,
    authority: str = "batch_local_selector_path",
) -> dict[str, Any]:
    """Emit one selector temporal-path ActionEffect from PR110-style rows."""

    pair_indices, sequence, source = _selector_sequence_from_rows(rows)
    if sequence.size == 0:
        return _empty_temporal_result(BLOCKER_PATH_SELECTOR_TRAJECTORY_EMPTY)
    payload = _temporal_path_payload(
        family="pr110",
        action_kind="selector_temporal_path",
        pair_indices=pair_indices,
        values=sequence.astype(np.float64),
        rdp_epsilon=rdp_epsilon,
        value_kind="selector_id",
    )
    payload["source"] = source
    encoded = b""
    for _ in range(8):
        prior_len = len(encoded)
        payload["selector_comparison"] = selector_sequence_encoding_comparison(sequence, path_payload_bytes=prior_len)
        encoded = _encode_temporal_payload(payload)
        if len(encoded) == prior_len:
            break
    payload["selector_comparison"] = selector_sequence_encoding_comparison(sequence, path_payload_bytes=len(encoded))
    encoded = _encode_temporal_payload(payload)
    payload_bytes = len(encoded)
    payload_sha256 = hashlib.sha256(encoded).hexdigest()
    old_bytes = int(base_archive_bytes)
    effect = ActionEffect.build(
        action_id=f"selector_path:n{sequence.size}:{payload_sha256[:16]}",
        family="pr110",
        action_kind="selector_temporal_path",
        inverse_source="selector_temporal_path",
        frame_index="both",
        frame_incidence="seg_pose_joint",
        candidate_status="generated",
        authority=authority,
        normalization_scope="batch_local",
        producer="path_action_producer",
        consumer="inverse_evaluate_candidate_queue",
        pair_ids=tuple(int(v) for v in pair_indices.tolist()),
        payload_sections=["selector_temporal_path", f"temporal_payload_bytes={payload_bytes}"],
        old_d_seg=old_d_seg,
        new_d_seg=new_d_seg,
        old_d_pose=old_d_pose,
        new_d_pose=new_d_pose,
        old_bytes=old_bytes,
        new_bytes=old_bytes + payload_bytes,
        receiver_surface={
            "selector_temporal_path_pairs": int(sequence.size),
            "selector_unique_modes": int(np.unique(sequence).size),
        },
        exact_score_decision="reject",
        parseback_survived=False,
        inflate_survived=False,
        fakequant_survived=None,
        payload_sha256=payload_sha256,
        blockers=[
            BLOCKER_PATH_TRAJECTORY_NO_RECEIVER_PROOF,
            BLOCKER_PATH_ACTION_PARSEBACK_MISSING,
            BLOCKER_PATH_ACTION_INFLATE_MISSING,
        ],
    )
    return _temporal_result(
        effect=effect,
        payload=payload,
        encoded=encoded,
        base_archive_sha256=base_archive_sha256,
        base_scorer_state_hash=base_scorer_state_hash,
    )


def path_tube_support_from_mask(
    mask: np.ndarray,
    *,
    pair_index: int,
    frame_index: int,
    target_class: int,
    epsilon: float = 2.0,
) -> PathTubeSupport:
    support = np.asarray(mask, dtype=bool)
    if support.ndim != 2:
        raise ValueError(f"support mask must be HW; got {support.shape}")
    if not np.any(support):
        raise ValueError("support mask is empty")
    height, width = support.shape
    path, tube_width = _centerline_path(support, epsilon=epsilon)
    preliminary = PathTubeSupport(
        height=int(height),
        width=int(width),
        frame_index=int(frame_index),
        target_class=int(target_class),
        pair_index=int(pair_index),
        tube_width=int(tube_width),
        path_yx=tuple(path),
        residual_add_yx=(),
        residual_remove_yx=(),
    )
    base = preliminary.base_tube_mask()
    add = _coords(support & ~base)
    remove = _coords(base & ~support)
    return PathTubeSupport(
        height=int(height),
        width=int(width),
        frame_index=int(frame_index),
        target_class=int(target_class),
        pair_index=int(pair_index),
        tube_width=int(tube_width),
        path_yx=tuple(path),
        residual_add_yx=tuple(add),
        residual_remove_yx=tuple(remove),
    )


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    src = np.asarray(mask, dtype=bool)
    if src.ndim != 2:
        raise ValueError(f"mask must be HW; got {src.shape}")
    visited = np.zeros(src.shape, dtype=bool)
    best: list[tuple[int, int]] = []
    ys, xs = np.nonzero(src)
    for start_y, start_x in zip(ys.tolist(), xs.tolist(), strict=False):
        if visited[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        comp: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            comp.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if (
                    0 <= ny < src.shape[0]
                    and 0 <= nx < src.shape[1]
                    and src[ny, nx]
                    and not visited[ny, nx]
                ):
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if len(comp) > len(best):
            best = comp
    out = np.zeros(src.shape, dtype=bool)
    for y, x in best:
        out[y, x] = True
    return out


def support_mask_sha256(mask: np.ndarray) -> str:
    src = np.asarray(mask, dtype=bool)
    coords = np.argwhere(src).astype("<u2", copy=False)
    h = hashlib.sha256()
    h.update(struct.pack("<HHI", int(src.shape[0]), int(src.shape[1]), int(coords.shape[0])))
    h.update(coords.tobytes(order="C"))
    return h.hexdigest()


def support_encoding_comparison(mask: np.ndarray, *, path_payload_bytes: int) -> dict[str, Any]:
    support = np.asarray(mask, dtype=bool)
    flat = support.reshape(-1)
    cardinality = int(np.count_nonzero(flat))
    bitmap_bytes = math.ceil(flat.size / 8.0)
    return {
        "schema": "tac.path_support_encoding_comparison.v1",
        "support_cardinality": cardinality,
        "bitmap_bytes": bitmap_bytes,
        "coordinate_list_bytes": int(cardinality * 4),
        "rle_bytes": _rle_bytes(flat),
        "path_tube_bytes": int(path_payload_bytes),
        "best_encoding": min(
            {
                "bitmap": bitmap_bytes,
                "coordinate_list": int(cardinality * 4),
                "rle": _rle_bytes(flat),
                "path_tube": int(path_payload_bytes),
            }.items(),
            key=lambda item: item[1],
        )[0],
    }


def selector_sequence_encoding_comparison(sequence: Sequence[int] | np.ndarray, *, path_payload_bytes: int) -> dict[str, Any]:
    values = _int_array(sequence, name="selector_sequence")
    if values.size == 0:
        raw_bits = 0
    else:
        max_symbol = max(1, int(values.max()))
        raw_bits = int(values.size * max(1, math.ceil(math.log2(max_symbol + 1))))
    raw_bytes = math.ceil(raw_bits / 8.0)
    return {
        "schema": "tac.selector_temporal_encoding_comparison.v1",
        "selector_count": int(values.size),
        "unique_modes": int(np.unique(values).size) if values.size else 0,
        "raw_fixed_width_bytes": raw_bytes,
        "rle_bytes": _selector_rle_bytes(values),
        "path_temporal_bytes": int(path_payload_bytes),
        "best_encoding": min(
            {
                "raw_fixed_width": raw_bytes,
                "rle": _selector_rle_bytes(values),
                "path_temporal": int(path_payload_bytes),
            }.items(),
            key=lambda item: item[1],
        )[0],
    }


def _temporal_result(
    *,
    effect: ActionEffect,
    payload: dict[str, Any],
    encoded: bytes,
    base_archive_sha256: str | None = None,
    base_scorer_state_hash: str | None = None,
) -> dict[str, Any]:
    rent_evaluation = candidate_action_evaluation_row(
        effect,
        base_archive_sha256=base_archive_sha256,
        base_scorer_state_hash=base_scorer_state_hash,
    )
    candidate = {
        "schema": PATH_ACTION_CANDIDATE_SCHEMA,
        "action_id": effect.action_id,
        "family": effect.family,
        "action_kind": effect.action_kind,
        "temporal_path": payload,
        "temporal_payload_bytes": len(encoded),
        "temporal_payload_sha256": hashlib.sha256(encoded).hexdigest(),
        "candidate_action_evaluation": rent_evaluation,
        "blockers": list(effect.blockers),
        "promotion_eligible": False,
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }
    return {
        "schema": PATH_ACTION_PRODUCER_SCHEMA,
        "candidate_count": 1,
        "blockers": [],
        "action_effects": [effect],
        "candidate_queue": build_candidate_queue([effect]),
        "path_action_candidates": [candidate],
        "candidate_action_evaluations": [rent_evaluation],
        "comparison": payload.get("selector_comparison", payload.get("comparison", {})),
        "policy": {
            "path_actions_do_not_clear_launch_gate": True,
            "receiver_proof_required": True,
            "rate_term_included_in_action_effect": True,
            "every_action_must_pay_rent": True,
        },
    }


def _empty_temporal_result(blocker: str) -> dict[str, Any]:
    return {
        "schema": PATH_ACTION_PRODUCER_SCHEMA,
        "candidate_count": 0,
        "blockers": [blocker],
        "action_effects": [],
        "candidate_queue": [],
        "path_action_candidates": [],
        "comparison": {},
    }


def _temporal_path_payload(
    *,
    family: str,
    action_kind: str,
    pair_indices: np.ndarray,
    values: np.ndarray,
    rdp_epsilon: float,
    value_kind: str = "float_profile",
) -> dict[str, Any]:
    quantized = np.rint(values.astype(np.float64) * _TEMPORAL_QUANT_SCALE).astype(np.int64)
    points = [(int(pair), int(value)) for pair, value in zip(pair_indices.tolist(), quantized.tolist(), strict=True)]
    simplified = _rdp_simplify_numeric(points, epsilon=rdp_epsilon * _TEMPORAL_QUANT_SCALE)
    payload = {
        "schema": TEMPORAL_PATH_SCHEMA,
        "family": family,
        "action_kind": action_kind,
        "value_kind": value_kind,
        "quant_scale": _TEMPORAL_QUANT_SCALE,
        "pair_count": int(pair_indices.size),
        "pair_min": int(pair_indices.min()),
        "pair_max": int(pair_indices.max()),
        "control_point_count": len(simplified),
        "control_points": [[int(pair), int(value)] for pair, value in simplified],
        "profile_linf": float(np.max(np.abs(values))) if values.size else 0.0,
        "profile_l2": float(np.linalg.norm(values)) if values.size else 0.0,
    }
    payload["payload_decode_sha256"] = hashlib.sha256(
        np.column_stack([pair_indices.astype("<i8"), quantized.astype("<i8")]).tobytes(order="C")
    ).hexdigest()
    payload["comparison"] = {
        "schema": "tac.temporal_path_encoding_comparison.v1",
        "sample_count": int(pair_indices.size),
        "raw_int64_pair_value_bytes": int(pair_indices.size * 16),
        "control_point_int64_bytes": int(len(simplified) * 16),
    }
    return payload


def _encode_temporal_payload(payload: Mapping[str, Any]) -> bytes:
    return zlib.compress(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"), level=9)


def _centerline_path(mask: np.ndarray, *, epsilon: float) -> tuple[list[tuple[int, int]], int]:
    ys, xs = np.nonzero(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    horizontal = (x1 - x0) >= (y1 - y0)
    points: list[tuple[int, int]] = []
    widths: list[int] = []
    if horizontal:
        for x in range(x0, x1 + 1):
            col_ys = ys[xs == x]
            if col_ys.size == 0:
                continue
            points.append((round(float(np.median(col_ys))), int(x)))
            widths.append(int(col_ys.max() - col_ys.min() + 1))
    else:
        for y in range(y0, y1 + 1):
            row_xs = xs[ys == y]
            if row_xs.size == 0:
                continue
            points.append((int(y), round(float(np.median(row_xs)))))
            widths.append(int(row_xs.max() - row_xs.min() + 1))
    if not points:
        points = [(int(ys[0]), int(xs[0]))]
        widths = [1]
    simplified = rdp_simplify(points, epsilon=epsilon)
    tube_width = max(1, round(float(np.median(widths))))
    return simplified, tube_width


def rdp_simplify(points: list[tuple[int, int]], *, epsilon: float) -> list[tuple[int, int]]:
    if len(points) <= 2:
        return list(points)
    start = np.array(points[0], dtype=np.float64)
    end = np.array(points[-1], dtype=np.float64)
    segment = end - start
    denom = float(np.dot(segment, segment))
    max_dist = -1.0
    max_index = 0
    for index, point in enumerate(points[1:-1], start=1):
        p = np.array(point, dtype=np.float64)
        if denom == 0.0:
            dist = float(np.linalg.norm(p - start))
        else:
            t = float(np.dot(p - start, segment) / denom)
            proj = start + max(0.0, min(1.0, t)) * segment
            dist = float(np.linalg.norm(p - proj))
        if dist > max_dist:
            max_dist = dist
            max_index = index
    if max_dist > float(epsilon):
        left = rdp_simplify(points[: max_index + 1], epsilon=epsilon)
        right = rdp_simplify(points[max_index:], epsilon=epsilon)
        return left[:-1] + right
    return [points[0], points[-1]]


def _rdp_simplify_numeric(points: list[tuple[int, int]], *, epsilon: float) -> list[tuple[int, int]]:
    if len(points) <= 2:
        return list(points)
    start = np.array(points[0], dtype=np.float64)
    end = np.array(points[-1], dtype=np.float64)
    segment = end - start
    denom = float(np.dot(segment, segment))
    max_dist = -1.0
    max_index = 0
    for index, point in enumerate(points[1:-1], start=1):
        p = np.array(point, dtype=np.float64)
        if denom == 0.0:
            dist = float(np.linalg.norm(p - start))
        else:
            t = float(np.dot(p - start, segment) / denom)
            proj = start + max(0.0, min(1.0, t)) * segment
            dist = float(np.linalg.norm(p - proj))
        if dist > max_dist:
            max_dist = dist
            max_index = index
    if max_dist > float(epsilon):
        left = _rdp_simplify_numeric(points[: max_index + 1], epsilon=epsilon)
        right = _rdp_simplify_numeric(points[max_index:], epsilon=epsilon)
        return left[:-1] + right
    return [points[0], points[-1]]


def _paint_point(mask: np.ndarray, point: tuple[int, int], radius: int) -> None:
    y, x = point
    y0, y1 = max(0, y - radius), min(mask.shape[0] - 1, y + radius)
    x0, x1 = max(0, x - radius), min(mask.shape[1] - 1, x + radius)
    mask[y0 : y1 + 1, x0 : x1 + 1] = True


def _paint_segment(
    mask: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    radius: int,
) -> None:
    y0, x0 = start
    y1, x1 = end
    steps = max(abs(int(y1) - int(y0)), abs(int(x1) - int(x0)), 1)
    for step in range(steps + 1):
        t = step / steps
        y = round((1.0 - t) * y0 + t * y1)
        x = round((1.0 - t) * x0 + t * x1)
        _paint_point(mask, (y, x), radius)


def _coords(mask: np.ndarray) -> list[tuple[int, int]]:
    return [(int(y), int(x)) for y, x in np.argwhere(np.asarray(mask, dtype=bool))]


def _rle_bytes(flat: np.ndarray) -> int:
    if flat.size == 0:
        return 0
    runs = 1
    prev = bool(flat[0])
    for value in flat[1:]:
        cur = bool(value)
        if cur != prev:
            runs += 1
            prev = cur
    return int(runs * 3)


def _selector_rle_bytes(values: np.ndarray) -> int:
    if values.size == 0:
        return 0
    runs = 1
    prev = int(values[0])
    for value in values[1:]:
        cur = int(value)
        if cur != prev:
            runs += 1
            prev = cur
    return int(runs * 4)


def _worst_unsolved_class(labels: np.ndarray, argmax: np.ndarray) -> int:
    classes, counts = np.unique(labels[labels != argmax], return_counts=True)
    if classes.size == 0:
        return int(np.bincount(labels.reshape(-1).astype(np.int64)).argmax())
    return int(classes[int(np.argmax(counts))])


def _support_margin_stat(margins: np.ndarray | None, support: np.ndarray) -> float | None:
    if margins is None or not np.any(support):
        return None
    values = np.asarray(margins, dtype=np.float64)[support]
    if values.size == 0:
        return None
    return float(-np.median(values))


def _pair_index(pair_indices: np.ndarray | None, batch: int) -> int:
    if pair_indices is None:
        return batch
    arr = np.asarray(pair_indices).reshape(-1)
    if batch >= arr.size:
        return batch
    return int(arr[batch])


def _path_action_id(*, pair_index: int, target_class: int, support_sha256: str) -> str:
    return f"path_tube:p{int(pair_index)}:c{int(target_class)}:{support_sha256[:16]}"


def _int_array(values: Sequence[int] | np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D sequence; got {arr.shape}")
    if arr.size == 0:
        return arr.astype(np.int64)
    numeric = arr.astype(np.float64)
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} must contain finite values")
    if not np.all(numeric == np.rint(numeric)):
        raise ValueError(f"{name} must contain integer-valued entries")
    return arr.astype(np.int64)


def _float_array(values: Sequence[float] | np.ndarray | None, *, name: str) -> np.ndarray:
    if values is None:
        raise ValueError(f"{name} is required")
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D sequence; got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite values")
    return arr


def _selector_sequence_from_rows(
    rows: Sequence[Mapping[str, Any]] | Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if isinstance(rows, Mapping):
        direct = _first_sequence(
            rows,
            "selector_sequence",
            "selector_ids",
            "selectors",
            "mode_sequence",
            "modes",
        )
        if direct is not None:
            sequence = _int_array(direct, name="selector_sequence")
            return (
                np.arange(sequence.size, dtype=np.int64),
                sequence,
                {"source_kind": "direct_selector_sequence"},
            )
        nested = rows.get("selector")
        if isinstance(nested, Mapping):
            direct = _first_sequence(nested, "selector_sequence", "selector_ids", "selectors", "mode_sequence", "modes")
            if direct is not None:
                sequence = _int_array(direct, name="selector_sequence")
                return (
                    np.arange(sequence.size, dtype=np.int64),
                    sequence,
                    {"source_kind": "nested_selector_sequence"},
                )
        atoms = rows.get("atoms")
        if isinstance(atoms, Sequence) and not isinstance(atoms, (str, bytes)):
            rows = atoms
        else:
            return (
                np.asarray([], dtype=np.int64),
                np.asarray([], dtype=np.int64),
                {"source_kind": "missing_selector_sequence"},
            )
    by_pair: dict[int, int] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        pair = _first_int(row, "pair_index", "pair_id")
        mode = _first_int(row, "selector_id", "selector", "mode", "mode_id", "atom_mode")
        scope = row.get("scope")
        if pair is None and isinstance(scope, Mapping):
            pair = _first_int(scope, "pair_index", "pair_id")
        if mode is None and isinstance(scope, Mapping):
            mode = _first_int(scope, "selector_id", "selector", "mode", "mode_id")
        if pair is not None and mode is not None:
            by_pair[int(pair)] = int(mode)
    if not by_pair:
        return (
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.int64),
            {"source_kind": "missing_selector_rows"},
        )
    pair_indices = np.asarray(sorted(by_pair), dtype=np.int64)
    sequence = np.asarray([by_pair[int(pair)] for pair in pair_indices], dtype=np.int64)
    return pair_indices, sequence, {"source_kind": "pair_mode_rows", "row_count": len(by_pair)}


def _first_sequence(row: Mapping[str, Any], *keys: str) -> Sequence[Any] | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return value
    return None


def _first_int(row: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float) and value.is_integer():
            return int(value)
    return None


__all__ = [
    "BLOCKER_PATH_ACTION_INFLATE_MISSING",
    "BLOCKER_PATH_ACTION_PARSEBACK_MISSING",
    "BLOCKER_PATH_POSE_TRAJECTORY_EMPTY",
    "BLOCKER_PATH_SELECTOR_TRAJECTORY_EMPTY",
    "BLOCKER_PATH_SUPPORT_NOT_BIRTH",
    "BLOCKER_PATH_TRAJECTORY_NO_RECEIVER_PROOF",
    "PATH_ACTION_CANDIDATE_SCHEMA",
    "PATH_ACTION_PRODUCER_SCHEMA",
    "PATH_SUPPORT_SCHEMA",
    "RENT_BLOCKER_BASE_ARCHIVE_SHA_MISSING",
    "RENT_BLOCKER_SCORE_ENDPOINTS_MISSING",
    "RENT_EVALUATION_SCHEMA",
    "TEMPORAL_PATH_SCHEMA",
    "PathTubeSupport",
    "build_path_action_candidates_from_arrays",
    "build_pose_temporal_path_candidates_from_arrays",
    "build_selector_temporal_path_candidates_from_rows",
    "candidate_action_evaluation_row",
    "decode_path_tube_payload",
    "largest_connected_component",
    "path_tube_support_from_mask",
    "rdp_simplify",
    "selector_sequence_encoding_comparison",
    "support_encoding_comparison",
    "support_mask_sha256",
]
