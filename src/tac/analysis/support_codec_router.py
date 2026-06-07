# SPDX-License-Identifier: MIT
"""Thin support-codec selection for ActionEffect support payloads."""

from __future__ import annotations

import json
import struct
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from tac.analysis.action_effect import ActionEffect
from tac.analysis.inverse_scorer_actions import build_candidate_queue
from tac.analysis.path_action_producer import (
    BLOCKER_PATH_ACTION_INFLATE_MISSING,
    BLOCKER_PATH_ACTION_PARSEBACK_MISSING,
    PathTubeSupport,
    support_mask_sha256,
)

SUPPORT_CODEC_ROUTER_SCHEMA = "tac.support_codec_router.v1"
SUPPORT_CODEC_CANDIDATE_SCHEMA = "tac.support_codec_candidate.v1"
SUPPORT_CODEC_REPORT_SCHEMA = "tac.support_codec_router_report.v1"

BLOCKER_CODEC_DECODED_HASH_MISMATCH = "support_codec_decoded_support_sha256_mismatch"
BLOCKER_CODEC_RESEARCH_ONLY = "support_codec_research_only_support_not_archive_closed"
BLOCKER_CODEC_UNAVAILABLE = "support_codec_unavailable"
BLOCKER_CODEC_NOT_ARCHIVE_EXECUTABLE = "support_codec_not_archive_executable"
BLOCKER_CODEC_DOMINATED = "support_codec_dominated_by_selected_codec"


@dataclass(frozen=True)
class SupportCodecCandidate:
    """One archive-executable support encoding candidate for one support hash."""

    schema: str
    action_id: str
    support_sha256: str
    support_encoding: str
    support_cardinality: int
    decoded_support_sha256: str | None
    encoded_bytes: int | None
    action_payload_bytes: int = 0
    metadata_bytes: int = 0
    total_cost_bytes: int | None = None
    decoder_runtime_estimate: str | None = None
    archive_executable: bool = True
    selected: bool = False
    parseback_survived: bool = False
    inflate_survived: bool = False
    survival_receipt_id: str | None = None
    blocker_status: str = "ok"
    blockers: tuple[str, ...] = ()
    rejection_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "action_id": self.action_id,
            "support_sha256": self.support_sha256,
            "support_encoding": self.support_encoding,
            "support_cardinality": int(self.support_cardinality),
            "decoded_support_sha256": self.decoded_support_sha256,
            "encoded_bytes": self.encoded_bytes,
            "action_payload_bytes": int(self.action_payload_bytes),
            "metadata_bytes": int(self.metadata_bytes),
            "total_cost_bytes": self.total_cost_bytes,
            "decoder_runtime_estimate": self.decoder_runtime_estimate,
            "archive_executable": bool(self.archive_executable),
            "selected": bool(self.selected),
            "parseback_survived": bool(self.parseback_survived),
            "inflate_survived": bool(self.inflate_survived),
            "survival_receipt_id": self.survival_receipt_id,
            "blocker_status": self.blocker_status,
            "blockers": list(self.blockers),
            "rejection_reason": self.rejection_reason,
        }


def route_support_codecs_for_path_candidate(
    candidate: Mapping[str, Any],
    *,
    source_effect: ActionEffect | Mapping[str, Any] | None = None,
    survival_receipts: Iterable[Mapping[str, Any]] = (),
    action_payload_bytes: int = 0,
    metadata_bytes: int = 0,
    tile_sizes: Sequence[int] = (8, 16),
) -> dict[str, Any]:
    """Route one PathActionProducer support to its cheapest valid codec."""

    support = _mapping(candidate.get("support"))
    if not support:
        raise ValueError("path candidate must contain a support object")
    action_id = str(candidate.get("action_id") or "")
    if not action_id:
        raise ValueError("path candidate must contain action_id")
    claimed_hash = _required_text(support, "support_sha256")
    support_research_only = bool(support.get("support_research_only") is True)
    support_archive_executable = bool(support.get("archive_executable") is True)
    mask = _mask_from_path_tube_support(support)
    decoded_hash = support_mask_sha256(mask)
    decoded_mismatch = decoded_hash != claimed_hash
    base_blockers: list[str] = []
    if decoded_mismatch:
        base_blockers.append(BLOCKER_CODEC_DECODED_HASH_MISMATCH)
    if support_research_only:
        base_blockers.append(BLOCKER_CODEC_RESEARCH_ONLY)
    if not support_archive_executable:
        base_blockers.append(BLOCKER_CODEC_NOT_ARCHIVE_EXECUTABLE)

    rows = _codec_candidates(
        action_id=action_id,
        support_sha256=claimed_hash,
        decoded_support_sha256=decoded_hash,
        mask=mask,
        support=support,
        action_payload_bytes=action_payload_bytes,
        metadata_bytes=metadata_bytes,
        tile_sizes=tile_sizes,
        base_blockers=base_blockers,
    )
    selected = _select_candidate(rows)
    survival_receipt = (
        None
        if selected is None
        else _matched_survival_receipt(selected, survival_receipts)
    )
    selected_rows: list[SupportCodecCandidate] = []
    for row in rows:
        if selected is not None and row.support_encoding == selected.support_encoding:
            selected_rows.append(
                _replace_candidate(
                    row,
                    selected=True,
                    parseback_survived=bool(
                        survival_receipt
                        and survival_receipt.get("parseback_survived") is True
                    ),
                    inflate_survived=bool(
                        survival_receipt
                        and survival_receipt.get("inflate_survived") is True
                    ),
                    survival_receipt_id=(
                        _survival_receipt_id(survival_receipt)
                        if survival_receipt is not None
                        else None
                    ),
                )
            )
        elif selected is not None and not row.blockers:
            selected_rows.append(
                _replace_candidate(
                    row,
                    blockers=(BLOCKER_CODEC_DOMINATED,),
                    blocker_status="blocked",
                    rejection_reason=BLOCKER_CODEC_DOMINATED,
                )
            )
        else:
            selected_rows.append(row)
    effect = (
        None
        if selected is None or source_effect is None
        else _selected_action_effect(
            source_effect,
            selected,
            survival_receipt=survival_receipt,
        )
    )
    queue_rows = [] if effect is None else build_candidate_queue([effect])
    return {
        "schema": SUPPORT_CODEC_REPORT_SCHEMA,
        "action_id": action_id,
        "support_sha256": claimed_hash,
        "decoded_support_sha256": decoded_hash,
        "support_cardinality": int(np.count_nonzero(mask)),
        "selected_support_encoding": None if selected is None else selected.support_encoding,
        "selected_total_cost_bytes": None if selected is None else selected.total_cost_bytes,
        "support_codec_candidates": [row.as_dict() for row in selected_rows],
        "selected_action_effect": None if effect is None else effect.as_dict(),
        "candidate_queue": queue_rows,
        "promotion_eligible": False,
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
        "blockers": base_blockers,
    }


def route_support_codecs_for_path_candidates(
    candidates: Iterable[Mapping[str, Any]],
    *,
    source_effects: Iterable[ActionEffect | Mapping[str, Any]] = (),
    survival_receipts: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Route multiple path candidates and emit a selected-only queue."""

    effects_by_id = {
        effect.action_id if isinstance(effect, ActionEffect) else str(effect.get("action_id") or ""): effect
        for effect in source_effects
    }
    survival_rows = [dict(row) for row in survival_receipts if isinstance(row, Mapping)]
    reports = [
        route_support_codecs_for_path_candidate(
            row,
            source_effect=effects_by_id.get(str(row.get("action_id") or "")),
            survival_receipts=survival_rows,
        )
        for row in candidates
        if _mapping(row.get("support"))
    ]
    return {
        "schema": SUPPORT_CODEC_ROUTER_SCHEMA,
        "report_count": len(reports),
        "reports": reports,
        "candidate_queue": [queue for report in reports for queue in report.get("candidate_queue", [])],
        "promotion_eligible": False,
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _codec_candidates(
    *,
    action_id: str,
    support_sha256: str,
    decoded_support_sha256: str,
    mask: np.ndarray,
    support: Mapping[str, Any],
    action_payload_bytes: int,
    metadata_bytes: int,
    tile_sizes: Sequence[int],
    base_blockers: Sequence[str],
) -> list[SupportCodecCandidate]:
    rows = [
        _candidate(
            action_id=action_id,
            support_sha256=support_sha256,
            support_encoding="rle",
            decoded_support_sha256=decoded_support_sha256,
            mask=mask,
            encoded=_encode_rle(mask),
            action_payload_bytes=action_payload_bytes,
            metadata_bytes=metadata_bytes,
            blockers=base_blockers,
            runtime="linear_rle_decode",
        ),
        _candidate(
            action_id=action_id,
            support_sha256=support_sha256,
            support_encoding="bitmap_bitset",
            decoded_support_sha256=decoded_support_sha256,
            mask=mask,
            encoded=_encode_bitmap(mask),
            action_payload_bytes=action_payload_bytes,
            metadata_bytes=metadata_bytes,
            blockers=base_blockers,
            runtime="linear_bit_unpack",
        ),
        _candidate(
            action_id=action_id,
            support_sha256=support_sha256,
            support_encoding="coordinate_list_u16",
            decoded_support_sha256=decoded_support_sha256,
            mask=mask,
            encoded=_encode_coordinate_list(mask),
            action_payload_bytes=action_payload_bytes,
            metadata_bytes=metadata_bytes,
            blockers=base_blockers,
            runtime="scatter_coordinate_list",
        ),
        _candidate(
            action_id=action_id,
            support_sha256=support_sha256,
            support_encoding="path_tube",
            decoded_support_sha256=decoded_support_sha256,
            mask=mask,
            encoded=_path_tube_payload_from_support(support),
            action_payload_bytes=action_payload_bytes,
            metadata_bytes=metadata_bytes,
            blockers=base_blockers,
            runtime="polyline_tube_plus_residuals",
        ),
    ]
    for tile in tile_sizes:
        rows.append(
            _candidate(
                action_id=action_id,
                support_sha256=support_sha256,
                support_encoding=f"tile_set_{int(tile)}x{int(tile)}_bitmap",
                decoded_support_sha256=decoded_support_sha256,
                mask=mask,
                encoded=_encode_tile_set(mask, tile=int(tile)),
                action_payload_bytes=action_payload_bytes,
                metadata_bytes=metadata_bytes,
                blockers=base_blockers,
                runtime="tile_scatter_bit_unpack",
            )
        )
    rows.extend(_optional_declared_codec_rows(action_id, support_sha256, mask, support, "semantic_grammar", base_blockers))
    rows.extend(_optional_declared_codec_rows(action_id, support_sha256, mask, support, "latent_derived", base_blockers))
    return rows


def _candidate(
    *,
    action_id: str,
    support_sha256: str,
    support_encoding: str,
    decoded_support_sha256: str,
    mask: np.ndarray,
    encoded: bytes | None,
    action_payload_bytes: int,
    metadata_bytes: int,
    blockers: Sequence[str],
    runtime: str,
) -> SupportCodecCandidate:
    encoded_bytes = None if encoded is None else len(encoded)
    total = (
        None
        if encoded_bytes is None
        else int(encoded_bytes) + int(action_payload_bytes) + int(metadata_bytes)
    )
    blocker_tuple = tuple(dict.fromkeys(str(item) for item in blockers))
    return SupportCodecCandidate(
        schema=SUPPORT_CODEC_CANDIDATE_SCHEMA,
        action_id=action_id,
        support_sha256=support_sha256,
        support_encoding=support_encoding,
        support_cardinality=int(np.count_nonzero(mask)),
        decoded_support_sha256=decoded_support_sha256,
        encoded_bytes=encoded_bytes,
        action_payload_bytes=int(action_payload_bytes),
        metadata_bytes=int(metadata_bytes),
        total_cost_bytes=total,
        decoder_runtime_estimate=runtime,
        archive_executable=not blocker_tuple,
        blocker_status="ok" if not blocker_tuple else "blocked",
        blockers=blocker_tuple,
    )


def _replace_candidate(
    row: SupportCodecCandidate,
    *,
    selected: bool | None = None,
    parseback_survived: bool | None = None,
    inflate_survived: bool | None = None,
    survival_receipt_id: str | None = None,
    blockers: Sequence[str] | None = None,
    blocker_status: str | None = None,
    rejection_reason: str | None = None,
) -> SupportCodecCandidate:
    blocker_tuple = row.blockers if blockers is None else tuple(dict.fromkeys(str(item) for item in blockers))
    return SupportCodecCandidate(
        schema=row.schema,
        action_id=row.action_id,
        support_sha256=row.support_sha256,
        support_encoding=row.support_encoding,
        support_cardinality=row.support_cardinality,
        decoded_support_sha256=row.decoded_support_sha256,
        encoded_bytes=row.encoded_bytes,
        action_payload_bytes=row.action_payload_bytes,
        metadata_bytes=row.metadata_bytes,
        total_cost_bytes=row.total_cost_bytes,
        decoder_runtime_estimate=row.decoder_runtime_estimate,
        archive_executable=row.archive_executable and not blocker_tuple,
        selected=row.selected if selected is None else selected,
        parseback_survived=(
            row.parseback_survived
            if parseback_survived is None
            else parseback_survived
        ),
        inflate_survived=(
            row.inflate_survived
            if inflate_survived is None
            else inflate_survived
        ),
        survival_receipt_id=(
            row.survival_receipt_id
            if survival_receipt_id is None
            else survival_receipt_id
        ),
        blocker_status=row.blocker_status if blocker_status is None else blocker_status,
        blockers=blocker_tuple,
        rejection_reason=row.rejection_reason if rejection_reason is None else rejection_reason,
    )


def _select_candidate(rows: Sequence[SupportCodecCandidate]) -> SupportCodecCandidate | None:
    valid = [row for row in rows if not row.blockers and row.total_cost_bytes is not None]
    if not valid:
        return None
    return min(valid, key=lambda row: (int(row.total_cost_bytes or 0), row.support_encoding))


def _selected_action_effect(
    source_effect: ActionEffect | Mapping[str, Any],
    selected: SupportCodecCandidate,
    *,
    survival_receipt: Mapping[str, Any] | None = None,
) -> ActionEffect:
    effect = source_effect if isinstance(source_effect, ActionEffect) else ActionEffect.from_dict(source_effect)
    payload = effect.as_dict()
    old_bytes = int(payload["old_bytes"] or 0)
    total_bytes = int(selected.total_cost_bytes or 0)
    parseback_survived = bool(
        survival_receipt and survival_receipt.get("parseback_survived") is True
    )
    inflate_survived = bool(
        survival_receipt and survival_receipt.get("inflate_survived") is True
    )
    blockers = list(effect.blockers)
    if parseback_survived:
        blockers = [
            blocker
            for blocker in blockers
            if blocker != BLOCKER_PATH_ACTION_PARSEBACK_MISSING
        ]
    if inflate_survived:
        blockers = [
            blocker
            for blocker in blockers
            if blocker != BLOCKER_PATH_ACTION_INFLATE_MISSING
        ]
    return ActionEffect.build(
        action_id=effect.action_id,
        family=effect.family,
        action_kind=effect.action_kind,
        inverse_source=effect.inverse_source,
        frame_index=effect.frame_index,
        frame_incidence=effect.frame_incidence,
        candidate_status="selected",
        authority=effect.authority,
        normalization_scope=effect.normalization_scope,
        producer="support_codec_router",
        consumer=effect.consumer,
        pair_ids=effect.pair_ids,
        class_ids=effect.class_ids,
        region_ids=effect.region_ids,
        payload_sections=(
            *effect.payload_sections,
            f"support_codec={selected.support_encoding}",
            f"action_payload_bytes={selected.action_payload_bytes}",
            f"metadata_bytes={selected.metadata_bytes}",
            f"total_cost_bytes={total_bytes}",
        ),
        old_d_seg=effect.old_d_seg,
        new_d_seg=effect.new_d_seg,
        old_d_pose=effect.old_d_pose,
        new_d_pose=effect.new_d_pose,
        old_bytes=old_bytes,
        new_bytes=old_bytes + total_bytes,
        receiver_surface=effect.receiver_surface,
        exact_score_decision="reject",
        parseback_survived=parseback_survived,
        inflate_survived=inflate_survived,
        fakequant_survived=effect.fakequant_survived,
        hard_won_count=effect.hard_won_count,
        wrong_to_target=effect.wrong_to_target,
        target_to_wrong=effect.target_to_wrong,
        wrong_to_wrong=effect.wrong_to_wrong,
        net_target_support_delta=effect.net_target_support_delta,
        uint8_changed_count_region=effect.uint8_changed_count_region,
        seg_input_delta_linf_region=effect.seg_input_delta_linf_region,
        posenet_input_delta_linf_pair=effect.posenet_input_delta_linf_pair,
        support_source=effect.support_source,
        support_cardinality=selected.support_cardinality,
        support_sha256=selected.support_sha256,
        support_encoding=selected.support_encoding,
        support_encoded_bytes=selected.encoded_bytes,
        support_research_only=False,
        arm=effect.arm,
        old_region_debt=effect.old_region_debt,
        new_region_debt=effect.new_region_debt,
        argmax_changed_count_region=effect.argmax_changed_count_region,
        pose_output_l2_delta=effect.pose_output_l2_delta,
        seg_score_delta=effect.seg_score_delta,
        pose_score_delta=effect.pose_score_delta,
        segnet_margin_delta=effect.segnet_margin_delta,
        fakequant_segnet_margin_delta=effect.fakequant_segnet_margin_delta,
        parseback_segnet_margin_delta=effect.parseback_segnet_margin_delta,
        blockers=blockers,
    )


def _matched_survival_receipt(
    selected: SupportCodecCandidate,
    receipts: Iterable[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    for receipt in receipts:
        if not isinstance(receipt, Mapping):
            continue
        if str(receipt.get("action_id") or "") != selected.action_id:
            continue
        if str(receipt.get("support_sha256") or "") != selected.support_sha256:
            continue
        if str(receipt.get("support_encoding") or "") != selected.support_encoding:
            continue
        if (
            receipt.get("parseback_survived") is True
            and receipt.get("inflate_survived") is True
        ):
            return receipt
    return None


def _survival_receipt_id(receipt: Mapping[str, Any]) -> str | None:
    for key in ("receipt_id", "survival_receipt_id", "artifact_sha256", "sha256"):
        value = receipt.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _mask_from_path_tube_support(support: Mapping[str, Any]) -> np.ndarray:
    tube = PathTubeSupport(
        height=_required_int(support, "height"),
        width=_required_int(support, "width"),
        frame_index=_required_int(support, "frame_index"),
        target_class=_required_int(support, "target_class"),
        pair_index=_required_int(support, "pair_index"),
        tube_width=_required_int(support, "tube_width"),
        path_yx=_points(support.get("path_yx")),
        residual_add_yx=_points(support.get("residual_add_yx")),
        residual_remove_yx=_points(support.get("residual_remove_yx")),
    )
    return tube.decode_mask()


def _path_tube_payload_from_support(support: Mapping[str, Any]) -> bytes:
    tube = PathTubeSupport(
        height=_required_int(support, "height"),
        width=_required_int(support, "width"),
        frame_index=_required_int(support, "frame_index"),
        target_class=_required_int(support, "target_class"),
        pair_index=_required_int(support, "pair_index"),
        tube_width=_required_int(support, "tube_width"),
        path_yx=_points(support.get("path_yx")),
        residual_add_yx=_points(support.get("residual_add_yx")),
        residual_remove_yx=_points(support.get("residual_remove_yx")),
    )
    return tube.encode_payload()


def _encode_bitmap(mask: np.ndarray) -> bytes:
    packed = np.packbits(np.asarray(mask, dtype=np.uint8).reshape(-1))
    return b"SCBM1" + struct.pack("<HHI", int(mask.shape[0]), int(mask.shape[1]), int(packed.size)) + packed.tobytes()


def _encode_coordinate_list(mask: np.ndarray) -> bytes:
    coords = np.argwhere(np.asarray(mask, dtype=bool)).astype("<u2", copy=False)
    return b"SCCL1" + struct.pack("<HHI", int(mask.shape[0]), int(mask.shape[1]), int(coords.shape[0])) + coords.tobytes(order="C")


def _encode_rle(mask: np.ndarray) -> bytes:
    flat = np.asarray(mask, dtype=bool).reshape(-1)
    if flat.size == 0:
        runs = np.asarray([], dtype="<u4")
        start = 0
    else:
        changes = np.flatnonzero(flat[1:] != flat[:-1]) + 1
        boundaries = np.concatenate(([0], changes, [flat.size]))
        runs = np.diff(boundaries).astype("<u4", copy=False)
        start = int(flat[0])
    return (
        b"SCRL1"
        + struct.pack("<HHBI", int(mask.shape[0]), int(mask.shape[1]), start, int(runs.size))
        + runs.tobytes(order="C")
    )


def _encode_tile_set(mask: np.ndarray, *, tile: int) -> bytes:
    src = np.asarray(mask, dtype=bool)
    chunks: list[bytes] = [b"SCTS1", struct.pack("<HHH", int(src.shape[0]), int(src.shape[1]), int(tile))]
    tile_count = 0
    for y0 in range(0, src.shape[0], tile):
        for x0 in range(0, src.shape[1], tile):
            block = src[y0 : y0 + tile, x0 : x0 + tile]
            if not np.any(block):
                continue
            tile_count += 1
            packed = np.packbits(block.astype(np.uint8).reshape(-1)).tobytes()
            chunks.append(struct.pack("<HHH", int(y0), int(x0), len(packed)))
            chunks.append(packed)
    chunks.insert(2, struct.pack("<I", tile_count))
    return b"".join(chunks)


def _optional_declared_codec_rows(
    action_id: str,
    support_sha256: str,
    mask: np.ndarray,
    support: Mapping[str, Any],
    key: str,
    base_blockers: Sequence[str],
) -> list[SupportCodecCandidate]:
    rows = support.get(f"{key}_candidates")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return [
            SupportCodecCandidate(
                schema=SUPPORT_CODEC_CANDIDATE_SCHEMA,
                action_id=action_id,
                support_sha256=support_sha256,
                support_encoding=key,
                support_cardinality=int(np.count_nonzero(mask)),
                decoded_support_sha256=None,
                encoded_bytes=None,
                total_cost_bytes=None,
                decoder_runtime_estimate=None,
                archive_executable=False,
                blocker_status="blocked",
                blockers=tuple(dict.fromkeys([*base_blockers, BLOCKER_CODEC_UNAVAILABLE])),
                rejection_reason=BLOCKER_CODEC_UNAVAILABLE,
            )
        ]
    out: list[SupportCodecCandidate] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        decoded = str(row.get("decoded_support_sha256") or row.get("support_sha256") or "")
        blockers = list(base_blockers)
        if decoded != support_sha256:
            blockers.append(BLOCKER_CODEC_DECODED_HASH_MISMATCH)
        if row.get("archive_executable") is not True:
            blockers.append(BLOCKER_CODEC_NOT_ARCHIVE_EXECUTABLE)
        encoded = _optional_int(row.get("encoded_bytes"))
        out.append(
            SupportCodecCandidate(
                schema=SUPPORT_CODEC_CANDIDATE_SCHEMA,
                action_id=action_id,
                support_sha256=support_sha256,
                support_encoding=str(row.get("support_encoding") or f"{key}_{index}"),
                support_cardinality=int(np.count_nonzero(mask)),
                decoded_support_sha256=decoded or None,
                encoded_bytes=encoded,
                total_cost_bytes=encoded,
                decoder_runtime_estimate=str(row.get("decoder_runtime_estimate") or key),
                archive_executable=not blockers,
                blocker_status="ok" if not blockers else "blocked",
                blockers=tuple(blockers),
            )
        )
    return out


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _required_text(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"required text field missing: {key}")
    return value


def _required_int(row: Mapping[str, Any], key: str) -> int:
    value = row.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"required integer field missing: {key}")
    return int(value)


def _optional_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return int(value)
    return None


def _points(value: Any) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    out: list[tuple[int, int]] = []
    for item in value:
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) != 2:
            raise ValueError("support point streams must contain [y, x] pairs")
        out.append((int(item[0]), int(item[1])))
    return tuple(out)


def write_support_codec_router_report(report: Mapping[str, Any], output_dir: str) -> dict[str, Any]:
    from pathlib import Path

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "support_codec_router_report.json"
    queue_path = out_dir / "candidate_queue.jsonl"
    rows_path = out_dir / "selected_action_effect_rows.jsonl"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    queue_rows = list(report.get("candidate_queue", []))
    with open(queue_path, "w", encoding="utf-8") as out:
        for row in queue_rows:
            out.write(json.dumps(row, sort_keys=True) + "\n")
    selected_rows = [
        sub.get("selected_action_effect")
        for sub in report.get("reports", [])
        if isinstance(sub, Mapping) and isinstance(sub.get("selected_action_effect"), Mapping)
    ]
    with open(rows_path, "w", encoding="utf-8") as out:
        for row in selected_rows:
            out.write(json.dumps(row, sort_keys=True) + "\n")
    return {
        "report_path": report_path.as_posix(),
        "candidate_queue_path": queue_path.as_posix(),
        "selected_action_effect_rows_path": rows_path.as_posix(),
        "candidate_queue_row_count": len(queue_rows),
        "selected_action_effect_row_count": len(selected_rows),
    }


__all__ = [
    "BLOCKER_CODEC_DECODED_HASH_MISMATCH",
    "BLOCKER_CODEC_DOMINATED",
    "BLOCKER_CODEC_NOT_ARCHIVE_EXECUTABLE",
    "BLOCKER_CODEC_RESEARCH_ONLY",
    "BLOCKER_CODEC_UNAVAILABLE",
    "SUPPORT_CODEC_CANDIDATE_SCHEMA",
    "SUPPORT_CODEC_REPORT_SCHEMA",
    "SUPPORT_CODEC_ROUTER_SCHEMA",
    "SupportCodecCandidate",
    "route_support_codecs_for_path_candidate",
    "route_support_codecs_for_path_candidates",
    "write_support_codec_router_report",
]
