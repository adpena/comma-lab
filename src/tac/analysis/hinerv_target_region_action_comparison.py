# SPDX-License-Identifier: MIT
"""Compare receiver-survived HiNeRV target-region actions against backend fit.

This module turns a proven target-region action sidecar into a fixed-object
compiler comparison: same action id, same support hash, measured receiver
survival, byte grammar alternatives, and whatever backend realization receipt
is present.  It intentionally does not promote a candidate; non-current
grammars are blocked until a receiver decoder actually consumes them.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import brotli  # type: ignore[import-not-found]
import numpy as np

from tac.analysis.action_effect import ActionEffect
from tac.analysis.evaluator_action_lowering_race import build_lowering_race_report
from tac.analysis.path_action_producer import path_tube_support_from_mask
from tac.repo_io import sha256_file
from tac.submission_archive import (
    MINIMAL_SINGLE_MEMBER_NAME,
    build_minimal_single_member_archive_bytes,
)
from tac.substrates.hi_nerv.archive import (
    HIV1_HEADER_FMT,
    HIV1_HEADER_SIZE,
    HIV1_MAGIC,
    split_archive_sections,
)
from tac.substrates.hi_nerv.archive_candidate import _read_hiv1_payload_from_archive_zip
from tac.substrates.hi_nerv.target_region_actions import (
    TARGET_REGION_ACTION_META_KEY,
    TargetRegionPixelAction,
    decode_target_region_actions,
    encode_target_region_actions,
    encode_target_region_actions_payload,
    encode_target_region_actions_payload_variants,
    target_region_action_payload_codec,
    target_region_action_section_telemetry_for_payload,
    target_region_action_support_sha256,
)

HI_NERV_TARGET_REGION_ACTION_COMPARISON_SCHEMA = (
    "hi_nerv_target_region_action_sidecar_backend_comparison.v1"
)
HI_NERV_TARGET_REGION_ACTION_COMPARISON_ROW_SCHEMA = (
    "hi_nerv_target_region_action_sidecar_backend_comparison_row.v1"
)
EVALUATOR_ACTION_LOWERING_RACE_SCHEMA = "hi_nerv_target_region_action_lowering_race.v1"
_SUPPORT_IDENTITY_MISMATCH = "support_identity_mismatch"
_SURVIVAL_IDENTITY_MISMATCH = "target_region_action_survival_identity_mismatch"

_BACKEND_LADDER_TIERS = (
    ("latents_fine", ("latents_fine",)),
    ("latents_fine+head_rgb_1", ("latents_fine", "head_rgb_1")),
    (
        "latents_fine+head_rgb_1+fine_injector",
        ("latents_fine", "head_rgb_1", "fine_injector"),
    ),
    (
        "latents_fine+head_rgb_1+fine_injector+feature_grids",
        ("latents_fine", "head_rgb_1", "fine_injector", "feature_grids"),
    ),
    (
        "latents_fine+head_rgb_1+fine_injector+feature_grids+pair_adapter_class_basis",
        (
            "latents_fine",
            "head_rgb_1",
            "fine_injector",
            "feature_grids",
            "pair_adapter_class_basis",
        ),
    ),
)


def build_hinerv_target_region_action_comparison_from_archive(
    archive_zip_path: str | Path,
    *,
    survival_receipt: Mapping[str, Any] | str | Path,
    runner_report: Mapping[str, Any] | str | Path | None = None,
    action_id: str | None = None,
    action_effect_sources: Sequence[ActionEffect | Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build a comparison report from a byte-closed HIV1 archive and receipts."""

    archive_path = Path(archive_zip_path).expanduser().resolve(strict=True)
    survival = _load_mapping(survival_receipt)
    report = None if runner_report is None else _load_mapping(runner_report)
    wall_normal = _find_first_wall_normal_lift(report) if report is not None else {}
    sidecar_candidate = _find_first_sidecar_candidate(report) if report is not None else {}

    payload = _read_hiv1_payload_from_archive_zip(archive_path)
    sections = split_archive_sections(payload)
    meta = dict(sections.meta)
    raw_b64 = meta.get(TARGET_REGION_ACTION_META_KEY)
    if not isinstance(raw_b64, str) or not raw_b64:
        raise ValueError("archive does not contain a target-region action meta payload")
    stored_payload = base64.b64decode(raw_b64.encode("ascii"), validate=True)
    actions = decode_target_region_actions(stored_payload)
    if not actions:
        raise ValueError("target-region action payload decodes to no actions")
    program = _ActionProgram(
        actions=tuple(actions),
        stored_payload=bytes(stored_payload),
        payload=payload,
        meta=meta,
        archive_path=archive_path,
        archive_bytes=int(archive_path.stat().st_size),
        archive_sha256=sha256_file(archive_path),
    )
    return build_hinerv_target_region_action_comparison(
        program=program,
        survival_receipt=survival,
        wall_normal_lift=wall_normal,
        sidecar_candidate=sidecar_candidate,
        action_id=action_id,
        action_effect_sources=action_effect_sources,
    )


def build_hinerv_target_region_action_comparison(
    *,
    program: _ActionProgram,
    survival_receipt: Mapping[str, Any],
    wall_normal_lift: Mapping[str, Any] | None = None,
    sidecar_candidate: Mapping[str, Any] | None = None,
    action_id: str | None = None,
    action_effect_sources: Sequence[ActionEffect | Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build the fixed-action sidecar-vs-backend report."""

    wall = dict(wall_normal_lift or {})
    chosen_action_id = (
        str(action_id)
        if action_id
        else str(wall.get("action_id") or _first_text(survival_receipt, "action_id") or "")
    )
    if not chosen_action_id:
        chosen_action_id = _program_action_id(program)

    sidecar_admission = _sidecar_admission_from_wall(wall, sidecar_candidate or {})
    direct = _direct_teacher_from_wall(wall)
    backend = _backend_fit_from_wall(wall)
    support_sha256 = target_region_action_support_sha256(list(program.actions))
    action_count = len(program.actions)
    support_masks = [_action_mask(action) for action in program.actions]
    support_rows = _support_encoding_rows(program.actions, support_masks)
    action_rows = _action_encoding_rows(program.actions)
    byte_decomposition = _byte_decomposition(program)
    direct_support_identity = _direct_teacher_support_identity(direct)
    direct_support_sha = direct_support_identity["comparison_support_sha256"]
    sidecar_support_mismatch = bool(direct_support_sha and direct_support_sha != support_sha256)
    support_identity_blockers = (
        ["direct_teacher_and_survived_sidecar_support_hashes_diverge"]
        if sidecar_support_mismatch
        else []
    )
    current_section_telemetry = dict(
        byte_decomposition.get("target_region_action_section_telemetry") or {}
    )
    current_support_encoding = str(
        current_section_telemetry.get("support_encoding")
        or "explicit_yx_u16_coordinates"
    )
    current_support_encoded_bytes = int(
        current_section_telemetry.get("support_encoded_bytes")
        or byte_decomposition["support_coord_u16_bytes"]
    )
    current_encoded_program_sha256 = str(
        current_section_telemetry.get("encoded_program_sha256")
        or hashlib.sha256(program.stored_payload).hexdigest()
    )
    current_decoded_support_sha256 = _optional_text(
        current_section_telemetry.get("decoded_support_sha256")
    )
    current_decoded_action_sha256 = _optional_text(
        current_section_telemetry.get("decoded_action_sha256")
    )
    survival_identity = _survival_identity_status(
        survival_receipt,
        action_id=chosen_action_id,
        support_sha256=support_sha256,
        archive_sha256=program.archive_sha256,
        program_sha256=_program_base64_sha256(program),
        decoded_support_sha256=current_decoded_support_sha256,
        decoded_action_sha256=current_decoded_action_sha256,
    )
    survival_identity_blockers = list(survival_identity["blockers"])
    current_receiver_bound = not survival_identity_blockers
    current_first_failed = (
        _SURVIVAL_IDENTITY_MISMATCH
        if survival_identity_blockers
        else (_SUPPORT_IDENTITY_MISMATCH if sidecar_support_mismatch else None)
    )

    base_zip = _action_free_archive_bytes(program)
    old_bytes = int(base_zip["archive_bytes_without_target_region_actions"])
    current_row = _comparison_row(
        candidate_id="current_hiv1_target_region_action_brotli",
        action_id=chosen_action_id,
        candidate_kind="sidecar_grammar",
        support_sha256=support_sha256,
        archive_sha256=program.archive_sha256,
        payload_sha256=hashlib.sha256(program.stored_payload).hexdigest(),
        encoded_program_sha256=current_encoded_program_sha256,
        program_sha256=_program_base64_sha256(program),
        decoded_support_sha256=current_decoded_support_sha256,
        decoded_action_sha256=current_decoded_action_sha256,
        support_encoding=current_support_encoding,
        action_encoding="exact_rgb_u8",
        encoded_payload_bytes=int(byte_decomposition["stored_payload_bytes"]),
        support_encoded_bytes=current_support_encoded_bytes,
        action_payload_bytes=int(byte_decomposition["rgb_u8_bytes"]),
        metadata_bytes=int(byte_decomposition["meta_json_action_delta_bytes"]),
        byte_authority="exact_archive_zip_delta",
        old_bytes=old_bytes,
        new_bytes=int(program.archive_bytes),
        sidecar_admission=sidecar_admission,
        survival_receipt=survival_receipt if current_receiver_bound else {},
        first_failed_surface=current_first_failed,
        blockers=[*survival_identity_blockers, *support_identity_blockers],
        exact_payload_equivalent=True,
        receiver_bound=current_receiver_bound,
    )
    sidecar_rows = [current_row]
    variant_rows = _receiver_payload_variant_rows(
        program=program,
        action_id=chosen_action_id,
        support_sha256=support_sha256,
        decoded_support_sha256=current_decoded_support_sha256,
        decoded_action_sha256=current_decoded_action_sha256,
        action_free_archive=base_zip,
        sidecar_admission=sidecar_admission,
        survival_identity_blockers=survival_identity_blockers,
        support_identity_blockers=support_identity_blockers,
    )
    sidecar_rows.extend(variant_rows)
    for support in support_rows:
        for action in action_rows:
            if support["encoding"] == "explicit_yx_u16_coordinates" and action["encoding"] == "exact_rgb_u8":
                continue
            support_encoded_bytes = (
                None if support["encoded_bytes"] is None else int(support["encoded_bytes"])
            )
            action_payload_bytes = (
                None if action["encoded_bytes"] is None else int(action["encoded_bytes"])
            )
            exact_payload_equivalent = bool(
                support["lossless"] is True and action["lossless"] is True
            )
            if support_encoded_bytes is None or action_payload_bytes is None:
                exact_payload_equivalent = False
            encoded_bytes = (
                int(byte_decomposition["raw_fixed_header_bytes"])
                + (0 if support_encoded_bytes is None else support_encoded_bytes)
                + (0 if action_payload_bytes is None else action_payload_bytes)
            )
            blockers = [*survival_identity_blockers, *support_identity_blockers]
            first_failed = (
                _SURVIVAL_IDENTITY_MISMATCH
                if survival_identity_blockers
                else (
                    _SUPPORT_IDENTITY_MISMATCH
                    if sidecar_support_mismatch
                    else "runtime_decoder_not_bound"
                )
            )
            if support_encoded_bytes is None or action_payload_bytes is None:
                blockers.append("target_region_action_candidate_byte_accounting_missing")
                if not sidecar_support_mismatch and not survival_identity_blockers:
                    first_failed = "byte_accounting_missing"
            if not exact_payload_equivalent:
                blockers.append("target_region_action_candidate_not_lossless")
                if (
                    not sidecar_support_mismatch
                    and not survival_identity_blockers
                    and first_failed == "runtime_decoder_not_bound"
                ):
                    first_failed = "action_or_support_not_lossless"
            blockers.append("target_region_action_runtime_decoder_not_bound")
            sidecar_rows.append(
                _comparison_row(
                    candidate_id=f"{support['encoding']}__{action['encoding']}",
                    action_id=chosen_action_id,
                    candidate_kind="sidecar_grammar_candidate",
                    support_sha256=support_sha256,
                    archive_sha256=program.archive_sha256,
                    payload_sha256=None,
                    encoded_program_sha256=None,
                    program_sha256=None,
                    decoded_support_sha256=current_decoded_support_sha256,
                    decoded_action_sha256=(
                        current_decoded_action_sha256 if exact_payload_equivalent else None
                    ),
                    support_encoding=str(support["encoding"]),
                    action_encoding=str(action["encoding"]),
                    encoded_payload_bytes=encoded_bytes,
                    support_encoded_bytes=support_encoded_bytes,
                    action_payload_bytes=action_payload_bytes,
                    metadata_bytes=0,
                    byte_authority="exact_candidate_payload_bytes_not_receiver_bound",
                    old_bytes=None,
                    new_bytes=None,
                    sidecar_admission=sidecar_admission if exact_payload_equivalent else {},
                    survival_receipt={},
                    first_failed_surface=first_failed,
                    blockers=blockers,
                    exact_payload_equivalent=exact_payload_equivalent,
                    receiver_bound=False,
                )
            )

    backend_ladder = _backend_ladder_rows(
        action_id=chosen_action_id,
        backend_fit=backend,
        support_sha256=support_sha256,
    )
    best_sidecar = min(
        sidecar_rows,
        key=lambda row: (
            row["first_failed_surface"] is not None,
            -float(row["action_effect"].get("value_per_byte") or -1.0e99),
        ),
    )
    measured_backend = [row for row in backend_ladder if row["status"] == "measured"]
    best_backend = max(
        measured_backend,
        key=lambda row: int(row.get("wrong_to_target") or 0),
        default=None,
    )
    same_support = _same_action_support_summary(
        action_id=chosen_action_id,
        support_sha256=support_sha256,
        sidecar_rows=sidecar_rows,
        backend_ladder=backend_ladder,
    )
    sidecar_economics = _sidecar_economics_summary(
        byte_decomposition=byte_decomposition,
        action_free_archive=base_zip,
        support_cardinality=int(sum(action.pixel_count for action in program.actions)),
        current_row=current_row,
        best_sidecar=best_sidecar,
        best_backend=best_backend,
    )
    next_blocker = (
        "optimize_sidecar_grammar_current_receiver_survives_backend_does_not"
        if bool(current_row["survival"]["inflate_survived"]) and not _backend_realized(backend)
        else "rerun_receiver_surface_proof_or_backend_ladder"
    )
    if sidecar_support_mismatch:
        next_blocker = "direct_teacher_and_survived_sidecar_support_hashes_diverge"
    if survival_identity_blockers:
        next_blocker = survival_identity_blockers[0]
    lowering_race = _lowering_race_verdict(
        action_id=chosen_action_id,
        support_sha256=support_sha256,
        direct_teacher=direct,
        backend_fit=backend,
        current_sidecar=current_row,
        sidecar_rows=sidecar_rows,
        action_effect_sources=action_effect_sources,
        sidecar_support_mismatch=sidecar_support_mismatch,
        sidecar_survival_identity_mismatch=bool(survival_identity_blockers),
    )

    return {
        "schema": HI_NERV_TARGET_REGION_ACTION_COMPARISON_SCHEMA,
        "action_id": chosen_action_id,
        "family": "hinerv",
        "archive_path": program.archive_path.as_posix(),
        "archive_sha256": program.archive_sha256,
        "archive_bytes": int(program.archive_bytes),
        "payload_sha256": hashlib.sha256(program.stored_payload).hexdigest(),
        "encoded_program_sha256": current_encoded_program_sha256,
        "target_region_action_program_sha256": _program_base64_sha256(program),
        "action_count": action_count,
        "support_sha256": support_sha256,
        "decoded_support_sha256": current_decoded_support_sha256,
        "decoded_action_sha256": current_decoded_action_sha256,
        "support_cardinality": int(sum(action.pixel_count for action in program.actions)),
        "byte_decomposition": byte_decomposition,
        "sidecar_economics": sidecar_economics,
        "action_free_archive": base_zip,
        "receiver_payload_variants": variant_rows,
        "support_identity": {
            "sidecar_support_sha256": support_sha256,
            "direct_teacher_support_sha256": direct_support_sha,
            "direct_teacher_mask_support_sha256": direct_support_identity[
                "mask_support_sha256"
            ],
            "direct_teacher_archive_executable_support_sha256": (
                direct_support_identity["archive_executable_support_sha256"]
            ),
            "direct_teacher_comparison_hash_domain": direct_support_identity[
                "comparison_hash_domain"
            ],
            "same_as_direct_teacher": not sidecar_support_mismatch,
            "blockers": (
                ["direct_teacher_and_survived_sidecar_support_hashes_diverge"]
                if sidecar_support_mismatch
                else []
            ),
        },
        "survival_identity": survival_identity,
        "same_action_support": same_support,
        "sidecar_encoding_candidates": sidecar_rows,
        "backend_ladder": backend_ladder,
        "lowering_race": lowering_race,
        "comparison": {
            "best_receiver_bound_sidecar_candidate_id": current_row["candidate_id"],
            "best_sidecar_candidate_id": best_sidecar["candidate_id"],
            "best_sidecar_value_per_byte": _row_value_per_byte(best_sidecar),
            "best_backend_tier": None if best_backend is None else best_backend["tier"],
            "best_backend_wrong_to_target": (
                None if best_backend is None else best_backend.get("wrong_to_target")
            ),
            "best_lowering": lowering_race["best_lowering"],
            "first_failing_surface": lowering_race["first_failing_surface"],
            "backend_realized": _backend_realized(backend),
            "sidecar_current_inflate_survived": bool(current_row["survival"]["inflate_survived"]),
            "next_blocker": next_blocker,
            "promotable": False,
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "direct_teacher": direct,
        "backend_fit": backend,
        "promotion_eligible": False,
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }


@dataclass(frozen=True)
class TargetRegionActionProgram:
    """Decoded target-region action payload plus archive custody."""

    actions: tuple[TargetRegionPixelAction, ...]
    stored_payload: bytes
    payload: bytes
    meta: dict[str, object]
    archive_path: Path
    archive_bytes: int
    archive_sha256: str


class _ActionProgram(TargetRegionActionProgram):
    """Backward-compatible alias for tests and private callers."""

    def __init__(
        self,
        *,
        actions: tuple[TargetRegionPixelAction, ...],
        stored_payload: bytes,
        payload: bytes,
        meta: Mapping[str, object],
        archive_path: Path,
        archive_bytes: int,
        archive_sha256: str,
    ) -> None:
        super().__init__(
            actions=tuple(actions),
            stored_payload=bytes(stored_payload),
            payload=bytes(payload),
            meta=dict(meta),
            archive_path=Path(archive_path),
            archive_bytes=int(archive_bytes),
            archive_sha256=str(archive_sha256),
        )


def _comparison_row(
    *,
    candidate_id: str,
    action_id: str,
    candidate_kind: str,
    support_sha256: str,
    archive_sha256: str | None,
    payload_sha256: str | None,
    encoded_program_sha256: str | None,
    program_sha256: str | None,
    decoded_support_sha256: str | None,
    decoded_action_sha256: str | None,
    support_encoding: str,
    action_encoding: str,
    encoded_payload_bytes: int,
    support_encoded_bytes: int | None,
    action_payload_bytes: int | None,
    metadata_bytes: int | None,
    byte_authority: str,
    old_bytes: int | None,
    new_bytes: int | None,
    sidecar_admission: Mapping[str, Any],
    survival_receipt: Mapping[str, Any],
    first_failed_surface: str | None,
    blockers: Sequence[str],
    exact_payload_equivalent: bool,
    receiver_bound: bool,
) -> dict[str, Any]:
    old_d_seg = _first_float(sidecar_admission, "old_d_seg")
    new_d_seg = _first_float(sidecar_admission, "new_d_seg")
    old_d_pose = _first_float(sidecar_admission, "old_d_pose")
    new_d_pose = _first_float(sidecar_admission, "new_d_pose")
    transitions = _transition_counts(sidecar_admission)
    parseback_survived = bool(survival_receipt.get("parseback_survived") is True)
    inflate_survived = bool(survival_receipt.get("inflate_survived") is True)
    fakequant_survived = bool(survival_receipt.get("fakequant_survived") is True)
    row_blockers = list(dict.fromkeys(str(value) for value in blockers if str(value)))
    if receiver_bound:
        if not fakequant_survived:
            row_blockers.append("target_region_action_fakequant_survival_missing")
        if not parseback_survived:
            row_blockers.append("target_region_action_parseback_survival_missing")
        if not inflate_survived:
            row_blockers.append("target_region_action_inflate_survival_missing")

    effect = ActionEffect.build(
        action_id=action_id,
        family="hinerv",
        action_kind=candidate_kind,
        inverse_source="receiver_surface_masked_rgb_residual_on_support",
        frame_index=1,
        frame_incidence="seg_pose_joint",
        candidate_status="measured" if receiver_bound and not row_blockers else "rejected",
        authority="inflate_raw" if receiver_bound else "analysis_payload_model",
        normalization_scope="batch_local",
        producer="hinerv_target_region_action_comparison",
        consumer="long_run_readiness_dag",
        pair_ids=_int_tuple(survival_receipt.get("pair_indices")),
        class_ids=_int_tuple(sidecar_admission.get("target_class")),
        region_ids=_str_tuple(sidecar_admission.get("region_id")),
        payload_sections=(
            f"lowering_target={_candidate_lowering_target(support_encoding=support_encoding, action_encoding=action_encoding)}",
            f"byte_authority={byte_authority}",
            f"support_codec={support_encoding}",
            f"action_codec={action_encoding}",
            f"encoded_payload_bytes={int(encoded_payload_bytes)}",
            f"support_encoded_bytes={support_encoded_bytes}",
            f"action_payload_bytes={action_payload_bytes}",
            f"metadata_bytes={metadata_bytes}",
            f"old_archive_zip_bytes={old_bytes}",
            f"new_archive_zip_bytes={new_bytes}",
            f"archive_sha256={archive_sha256}",
            f"payload_sha256={payload_sha256}",
            f"encoded_program_sha256={encoded_program_sha256}",
            f"target_region_action_program_sha256={program_sha256}",
            f"decoded_support_sha256={decoded_support_sha256}",
            f"decoded_action_sha256={decoded_action_sha256}",
        ),
        old_d_seg=old_d_seg,
        new_d_seg=new_d_seg,
        old_d_pose=old_d_pose,
        new_d_pose=new_d_pose,
        old_bytes=old_bytes,
        new_bytes=new_bytes,
        receiver_surface={
            "uint8_changed_pixels": _first_int(survival_receipt, "inflated_raw_action_changed_pixels")
            or _first_int(survival_receipt, "receiver_changed_action_pixels"),
            "seg_argmax_changed_pixels": transitions.get("argmax_changed_count_region"),
            "seg_wrong_to_target_count": transitions.get("wrong_to_target_count"),
            "seg_target_hard_lost_count": transitions.get("target_to_wrong_count"),
            "seg_wrong_to_wrong_count": transitions.get("wrong_to_wrong_count"),
        },
        exact_score_decision="accept" if receiver_bound and not row_blockers else "reject",
        parseback_survived=parseback_survived if receiver_bound else False,
        inflate_survived=inflate_survived if receiver_bound else False,
        fakequant_survived=fakequant_survived if receiver_bound else None,
        archive_sha256=archive_sha256 if receiver_bound else None,
        payload_sha256=payload_sha256 if receiver_bound else None,
        wrong_to_target=transitions.get("wrong_to_target_count"),
        target_to_wrong=transitions.get("target_to_wrong_count"),
        wrong_to_wrong=transitions.get("wrong_to_wrong_count"),
        net_target_support_delta=transitions.get("net_target_support_delta"),
        argmax_changed_count_region=transitions.get("argmax_changed_count_region"),
        uint8_changed_count_region=_first_int(survival_receipt, "inflated_raw_action_changed_pixels")
        or _first_int(survival_receipt, "receiver_changed_action_pixels"),
        support_source="survived_target_region_action_sidecar",
        support_cardinality=_first_int(survival_receipt, "total_action_pixels")
        or _first_int(sidecar_admission, "target_region_action_pixel_count"),
        support_sha256=support_sha256,
        support_encoding=support_encoding,
        support_encoded_bytes=support_encoded_bytes,
        support_research_only=not receiver_bound,
        seg_score_delta=_first_float(sidecar_admission, "seg_score_delta"),
        pose_score_delta=_first_float(sidecar_admission, "pose_score_delta"),
        blockers=row_blockers,
    )
    return {
        "schema": HI_NERV_TARGET_REGION_ACTION_COMPARISON_ROW_SCHEMA,
        "candidate_id": candidate_id,
        "candidate_kind": candidate_kind,
        "action_id": action_id,
        "support_sha256": support_sha256,
        "decoded_support_sha256": decoded_support_sha256,
        "decoded_action_sha256": decoded_action_sha256,
        "archive_sha256": archive_sha256,
        "payload_sha256": payload_sha256,
        "encoded_program_sha256": encoded_program_sha256,
        "target_region_action_program_sha256": program_sha256,
        "support_encoding": support_encoding,
        "action_encoding": action_encoding,
        "encoded_payload_bytes": int(encoded_payload_bytes),
        "support_encoded_bytes": None if support_encoded_bytes is None else int(support_encoded_bytes),
        "action_payload_bytes": None if action_payload_bytes is None else int(action_payload_bytes),
        "metadata_bytes": None if metadata_bytes is None else int(metadata_bytes),
        "byte_authority": byte_authority,
        "old_archive_zip_bytes": None if old_bytes is None else int(old_bytes),
        "new_archive_zip_bytes": None if new_bytes is None else int(new_bytes),
        "archive_zip_delta_bytes": (
            None if old_bytes is None or new_bytes is None else int(new_bytes) - int(old_bytes)
        ),
        "exact_payload_equivalent": bool(exact_payload_equivalent),
        "receiver_bound": bool(receiver_bound),
        "first_failed_surface": first_failed_surface,
        "blockers": row_blockers,
        "survival": {
            "fakequant_survived": bool(fakequant_survived) if receiver_bound else None,
            "parseback_survived": bool(parseback_survived) if receiver_bound else False,
            "inflate_survived": bool(inflate_survived) if receiver_bound else False,
        },
        "action_effect": effect.as_dict(),
        "promotion_eligible": False,
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _receiver_payload_variant_rows(
    *,
    program: _ActionProgram,
    action_id: str,
    support_sha256: str,
    decoded_support_sha256: str | None,
    decoded_action_sha256: str | None,
    action_free_archive: Mapping[str, Any],
    sidecar_admission: Mapping[str, Any],
    survival_identity_blockers: Sequence[str],
    support_identity_blockers: Sequence[str],
) -> list[dict[str, Any]]:
    """Rows for concrete receiver-decoded payload variants with archive bytes.

    These rows are still blocked until a generated archive is actually
    parseback/inflate-tested.  The useful thing they prove is narrower and
    exact: for the same decoded evaluator action, this payload grammar would
    charge this many final minimal ``archive.zip`` bytes.
    """

    old_bytes = _first_int(
        action_free_archive,
        "archive_bytes_without_target_region_actions",
    )
    rows: list[dict[str, Any]] = []
    seen_payload_sha256: set[str] = {hashlib.sha256(program.stored_payload).hexdigest()}
    variants = encode_target_region_actions_payload_variants(list(program.actions))
    logical_rgb_bytes = int(sum(action.rgb_u8.nbytes for action in program.actions))
    for codec, payload in variants.items():
        payload_sha = hashlib.sha256(payload).hexdigest()
        if payload_sha in seen_payload_sha256:
            continue
        seen_payload_sha256.add(payload_sha)
        try:
            decoded = decode_target_region_actions(payload)
            telemetry = target_region_action_section_telemetry_for_payload(decoded, payload)
            variant_decoded_support = _optional_text(telemetry.get("decoded_support_sha256"))
            variant_decoded_action = _optional_text(telemetry.get("decoded_action_sha256"))
            same_action = (
                bool(variant_decoded_support)
                and bool(variant_decoded_action)
                and variant_decoded_support == decoded_support_sha256
                and variant_decoded_action == decoded_action_sha256
            )
            archive_record = _archive_bytes_with_target_region_payload(program, payload)
            blockers = [
                *survival_identity_blockers,
                *support_identity_blockers,
            ]
            first_failed = "target_region_action_payload_variant_not_inflated"
            if not same_action:
                blockers.append("target_region_action_payload_variant_decoded_identity_mismatch")
                first_failed = "target_region_action_variant_decoded_identity_mismatch"
            blockers.append("target_region_action_payload_variant_not_inflated")
            rows.append(
                _comparison_row(
                    candidate_id=f"receiver_payload_variant_{codec}",
                    action_id=action_id,
                    candidate_kind="sidecar_receiver_payload_variant",
                    support_sha256=support_sha256,
                    archive_sha256=str(archive_record["archive_zip_sha256"]),
                    payload_sha256=payload_sha,
                    encoded_program_sha256=payload_sha,
                    program_sha256=str(archive_record["program_base64_sha256"]),
                    decoded_support_sha256=variant_decoded_support,
                    decoded_action_sha256=variant_decoded_action,
                    support_encoding=str(
                        telemetry.get("support_encoding")
                        or f"{codec}_support"
                    ),
                    action_encoding="exact_rgb_u8_receiver_payload_variant",
                    encoded_payload_bytes=len(payload),
                    support_encoded_bytes=_first_int(telemetry, "support_encoded_bytes"),
                    action_payload_bytes=logical_rgb_bytes,
                    metadata_bytes=_first_int(archive_record, "meta_json_action_delta_bytes"),
                    byte_authority="exact_rebuilt_minimal_archive_zip_variant_not_inflated",
                    old_bytes=old_bytes if same_action else None,
                    new_bytes=(
                        _first_int(archive_record, "archive_zip_bytes")
                        if same_action
                        else None
                    ),
                    sidecar_admission=sidecar_admission if same_action else {},
                    survival_receipt={},
                    first_failed_surface=first_failed,
                    blockers=blockers,
                    exact_payload_equivalent=same_action,
                    receiver_bound=False,
                )
            )
        except Exception as exc:
            rows.append(
                {
                    "schema": HI_NERV_TARGET_REGION_ACTION_COMPARISON_ROW_SCHEMA,
                    "candidate_id": f"receiver_payload_variant_{codec}",
                    "candidate_kind": "sidecar_receiver_payload_variant",
                    "action_id": action_id,
                    "support_sha256": support_sha256,
                    "payload_sha256": payload_sha,
                    "byte_authority": "payload_variant_decode_failed",
                    "first_failed_surface": "target_region_action_payload_variant_decode_failed",
                    "blockers": [
                        f"target_region_action_payload_variant_decode_failed:{type(exc).__name__}"
                    ],
                    "promotion_eligible": False,
                    "score_claim": False,
                    "ready_for_exact_eval_dispatch": False,
                }
            )
    return rows


def _candidate_lowering_target(*, support_encoding: str, action_encoding: str) -> str:
    text = f"{support_encoding} {action_encoding}".lower()
    if (
        "path_tube" in text
        or "class_attractor" in text
        or "semantic" in text
        or "median_scalar" in text
        or "low_rank" in text
    ):
        return "semantic_pose_primitive"
    return "byte_priced_sidecar"


def _byte_decomposition(program: _ActionProgram) -> dict[str, Any]:
    raw_payload = encode_target_region_actions(list(program.actions))
    selected_payload = encode_target_region_actions_payload(list(program.actions))
    support_payload = b"".join(action.yx.tobytes(order="C") for action in program.actions)
    rgb_payload = b"".join(action.rgb_u8.tobytes(order="C") for action in program.actions)
    support_bytes = len(support_payload)
    rgb_bytes = len(rgb_payload)
    raw_fixed = int(len(raw_payload) - support_bytes - rgb_bytes)
    meta_text = base64.b64encode(program.stored_payload)
    meta_text_bytes = len(meta_text)
    meta_without = dict(program.meta)
    meta_without.pop(TARGET_REGION_ACTION_META_KEY, None)
    meta_with_payload = json.dumps(program.meta, separators=(",", ":"), sort_keys=True).encode("utf-8")
    meta_without_payload = json.dumps(
        meta_without,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    meta_with_bytes = len(meta_with_payload)
    meta_without_bytes = len(meta_without_payload)
    meta_delta_bytes = meta_with_bytes - meta_without_bytes
    return {
        "schema": "hi_nerv_target_region_action_byte_decomposition.v1",
        "raw_payload_bytes": len(raw_payload),
        "stored_payload_bytes": len(program.stored_payload),
        "selected_payload_reencoded_bytes": len(selected_payload),
        "payload_codec": target_region_action_payload_codec(program.stored_payload),
        "support_coord_u16_bytes": support_bytes,
        "rgb_u8_bytes": rgb_bytes,
        "raw_fixed_header_bytes": raw_fixed,
        "payload_compression_savings_bytes": len(raw_payload) - len(program.stored_payload),
        "base64_text_bytes": meta_text_bytes,
        "meta_json_bytes_with_action": meta_with_bytes,
        "meta_json_bytes_without_action": meta_without_bytes,
        "meta_json_action_delta_bytes": meta_delta_bytes,
        "sections": [
            {
                "name": "support",
                "codec": "explicit_yx_u16_coordinates",
                "bytes": support_bytes,
                "bytes_per_support_pixel": _safe_div(support_bytes, sum(action.pixel_count for action in program.actions)),
            },
            {
                "name": "action",
                "codec": "exact_rgb_u8",
                "bytes": rgb_bytes,
                "bytes_per_support_pixel": _safe_div(rgb_bytes, sum(action.pixel_count for action in program.actions)),
            },
            {
                "name": "metadata",
                "codec": "hiv1_json_base64_target_region_actions",
                "bytes": meta_delta_bytes,
                "bytes_per_support_pixel": _safe_div(meta_delta_bytes, sum(action.pixel_count for action in program.actions)),
            },
            {
                "name": "entropy",
                "codec": target_region_action_payload_codec(program.stored_payload),
                "raw_payload_bytes": len(raw_payload),
                "stored_payload_bytes": len(program.stored_payload),
                "payload_compression_savings_bytes": len(raw_payload) - len(program.stored_payload),
                "stored_over_raw_ratio": _safe_div(len(program.stored_payload), len(raw_payload)),
            },
        ],
        "entropy_sections": {
            "support_coord_u16": _byte_entropy_section(support_payload),
            "action_rgb_u8": _byte_entropy_section(rgb_payload),
            "raw_payload": _byte_entropy_section(raw_payload),
            "stored_payload": _byte_entropy_section(program.stored_payload),
            "metadata_base64_text": _byte_entropy_section(meta_text),
            "metadata_json_with_action": _byte_entropy_section(meta_with_payload),
        },
        "action_count": len(program.actions),
        "pixel_count": int(sum(action.pixel_count for action in program.actions)),
        "target_region_action_section_telemetry": target_region_action_section_telemetry_for_payload(
            list(program.actions),
            program.stored_payload,
        ),
    }


def _byte_entropy_section(payload: bytes) -> dict[str, Any]:
    data = bytes(payload)
    if not data:
        return {
            "bytes": 0,
            "empirical_entropy_bits_per_byte": 0.0,
            "empirical_entropy_floor_bytes": 0,
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    probs = counts[counts > 0].astype(np.float64) / float(len(data))
    entropy = float(-np.sum(probs * np.log2(probs)))
    return {
        "bytes": len(data),
        "empirical_entropy_bits_per_byte": entropy,
        "empirical_entropy_floor_bytes": math.ceil(entropy * len(data) / 8.0),
        "unique_symbols": int(np.count_nonzero(counts)),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _sidecar_economics_summary(
    *,
    byte_decomposition: Mapping[str, Any],
    action_free_archive: Mapping[str, Any],
    support_cardinality: int,
    current_row: Mapping[str, Any],
    best_sidecar: Mapping[str, Any],
    best_backend: Mapping[str, Any] | None,
) -> dict[str, Any]:
    current_effect = current_row.get("action_effect") if isinstance(current_row, Mapping) else {}
    best_effect = best_sidecar.get("action_effect") if isinstance(best_sidecar, Mapping) else {}
    archive_delta = _first_int(
        action_free_archive,
        "archive_delta_bytes_rebuilt_minimal_vs_without",
    )
    return {
        "schema": "hi_nerv_target_region_action_sidecar_economics.v1",
        "support_cardinality": int(support_cardinality),
        "sections": list(byte_decomposition.get("sections") or []),
        "entropy_sections": dict(byte_decomposition.get("entropy_sections") or {}),
        "archive_delta_bytes": archive_delta,
        "archive_delta_score_cost": (
            None if archive_delta is None else float(25.0 * archive_delta / 37_545_489.0)
        ),
        "current_receiver_bound": _effect_value_brief(current_row),
        "best_sidecar_by_value": _effect_value_brief(best_sidecar),
        "best_backend_fit": (
            None
            if best_backend is None
            else {
                "tier": best_backend.get("tier"),
                "status": best_backend.get("status"),
                "wrong_to_target": best_backend.get("wrong_to_target"),
                "target_to_wrong": best_backend.get("target_to_wrong"),
                "accepted_step_count": best_backend.get("accepted_step_count"),
                "first_failed_surface": best_backend.get("first_failed_surface"),
                "blockers": list(best_backend.get("blockers") or []),
            }
        ),
        "current_delta_score_nonrate": _first_float(
            current_effect if isinstance(current_effect, Mapping) else {},
            "delta_score_nonrate",
        ),
        "current_delta_score_total": _first_float(
            current_effect if isinstance(current_effect, Mapping) else {},
            "delta_score_total",
        ),
        "current_value_per_byte": _first_float(
            current_effect if isinstance(current_effect, Mapping) else {},
            "value_per_byte",
        ),
        "best_sidecar_value_per_byte": _first_float(
            best_effect if isinstance(best_effect, Mapping) else {},
            "value_per_byte",
        ),
        "decision_axis": "exact_score_saved_per_charged_byte",
        "promotion_eligible": False,
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _same_action_support_summary(
    *,
    action_id: str,
    support_sha256: str,
    sidecar_rows: Sequence[Mapping[str, Any]],
    backend_ladder: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    sidecar_ok = all(
        row.get("action_id") == action_id and row.get("support_sha256") == support_sha256
        for row in sidecar_rows
    )
    backend_ok = all(
        row.get("action_id") == action_id and row.get("support_sha256") == support_sha256
        for row in backend_ladder
    )
    return {
        "schema": "hi_nerv_target_region_action_same_support_check.v1",
        "action_id": action_id,
        "comparison_support_sha256": support_sha256,
        "sidecar_encoding_rows_same_action_support": bool(sidecar_ok),
        "backend_ladder_rows_same_action_support": bool(backend_ok),
        "all_rows_same_action_support": bool(sidecar_ok and backend_ok),
        "sidecar_encoding_row_count": len(sidecar_rows),
        "backend_ladder_row_count": len(backend_ladder),
    }


def _effect_value_brief(row: Mapping[str, Any]) -> dict[str, Any]:
    effect = row.get("action_effect") if isinstance(row, Mapping) else {}
    if not isinstance(effect, Mapping):
        effect = {}
    return {
        "candidate_id": row.get("candidate_id"),
        "candidate_kind": row.get("candidate_kind"),
        "first_failed_surface": row.get("first_failed_surface"),
        "blockers": list(row.get("blockers") or []),
        "delta_bytes": effect.get("delta_bytes"),
        "delta_score_nonrate": effect.get("delta_score_nonrate"),
        "delta_score_total": effect.get("delta_score_total"),
        "value_per_byte": effect.get("value_per_byte"),
        "wrong_to_target": effect.get("wrong_to_target"),
        "target_to_wrong": effect.get("target_to_wrong"),
        "wrong_to_wrong": effect.get("wrong_to_wrong"),
        "parseback_survived": effect.get("parseback_survived"),
        "inflate_survived": effect.get("inflate_survived"),
    }


def _row_value_per_byte(row: Mapping[str, Any]) -> float | None:
    effect = row.get("action_effect") if isinstance(row, Mapping) else {}
    if not isinstance(effect, Mapping):
        return None
    return _first_float(effect, "value_per_byte")


def _safe_div(numerator: int | float | None, denominator: int | float | None) -> float | None:
    if denominator in (None, 0):
        return None
    if numerator is None:
        return None
    return float(numerator) / float(denominator)


def _action_free_archive_bytes(program: _ActionProgram) -> dict[str, Any]:
    sections = split_archive_sections(program.payload)
    meta_without = dict(sections.meta)
    meta_without.pop(TARGET_REGION_ACTION_META_KEY, None)
    meta_bytes = json.dumps(meta_without, separators=(",", ":"), sort_keys=True).encode("utf-8")
    rebuilt = (
        struct.pack(
            HIV1_HEADER_FMT,
            HIV1_MAGIC,
            int(sections.schema_version),
            int(sections.latent_dim_coarse),
            int(sections.latent_dim_mid),
            int(sections.latent_dim_fine),
            int(sections.num_pairs),
            len(sections.decoder_blob),
            len(sections.latents_coarse_blob),
            len(sections.latents_mid_blob),
            len(sections.latents_fine_blob),
            len(meta_bytes),
        )
        + sections.decoder_blob
        + sections.latents_coarse_blob
        + sections.latents_mid_blob
        + sections.latents_fine_blob
        + meta_bytes
    )
    archive_without, method_without = build_minimal_single_member_archive_bytes(
        rebuilt,
        member_name=MINIMAL_SINGLE_MEMBER_NAME,
    )
    archive_with, method_with = build_minimal_single_member_archive_bytes(
        program.payload,
        member_name=MINIMAL_SINGLE_MEMBER_NAME,
    )
    return {
        "schema": "hi_nerv_target_region_action_free_archive_bytes.v1",
        "archive_bytes_with_target_region_actions_actual": int(program.archive_bytes),
        "archive_bytes_with_target_region_actions_rebuilt_minimal": len(archive_with),
        "archive_bytes_without_target_region_actions": len(archive_without),
        "archive_delta_bytes_actual_vs_without": int(program.archive_bytes) - len(archive_without),
        "archive_delta_bytes_rebuilt_minimal_vs_without": len(archive_with) - len(archive_without),
        "hiv1_payload_bytes_with_target_region_actions": len(program.payload),
        "hiv1_payload_bytes_without_target_region_actions": len(rebuilt),
        "hiv1_payload_delta_bytes": len(program.payload) - len(rebuilt),
        "zip_method_with_target_region_actions": method_with,
        "zip_method_without_target_region_actions": method_without,
        "payload_without_target_region_actions_sha256": hashlib.sha256(rebuilt).hexdigest(),
        "header_size": HIV1_HEADER_SIZE,
    }


def _archive_bytes_with_target_region_payload(
    program: _ActionProgram,
    target_region_payload: bytes,
) -> dict[str, Any]:
    sections = split_archive_sections(program.payload)
    meta_without = dict(sections.meta)
    meta_without.pop(TARGET_REGION_ACTION_META_KEY, None)
    meta_with = dict(meta_without)
    program_base64 = base64.b64encode(bytes(target_region_payload)).decode("ascii")
    meta_with[TARGET_REGION_ACTION_META_KEY] = program_base64
    meta_with_bytes = json.dumps(meta_with, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    meta_without_bytes = json.dumps(
        meta_without,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    rebuilt = (
        struct.pack(
            HIV1_HEADER_FMT,
            HIV1_MAGIC,
            int(sections.schema_version),
            int(sections.latent_dim_coarse),
            int(sections.latent_dim_mid),
            int(sections.latent_dim_fine),
            int(sections.num_pairs),
            len(sections.decoder_blob),
            len(sections.latents_coarse_blob),
            len(sections.latents_mid_blob),
            len(sections.latents_fine_blob),
            len(meta_with_bytes),
        )
        + sections.decoder_blob
        + sections.latents_coarse_blob
        + sections.latents_mid_blob
        + sections.latents_fine_blob
        + meta_with_bytes
    )
    archive_bytes, method = build_minimal_single_member_archive_bytes(
        rebuilt,
        member_name=MINIMAL_SINGLE_MEMBER_NAME,
    )
    return {
        "schema": "hi_nerv_target_region_action_variant_archive_bytes.v1",
        "payload_codec": target_region_action_payload_codec(bytes(target_region_payload)),
        "target_region_payload_bytes": len(target_region_payload),
        "target_region_payload_sha256": hashlib.sha256(bytes(target_region_payload)).hexdigest(),
        "program_base64_sha256": hashlib.sha256(program_base64.encode("ascii")).hexdigest(),
        "hiv1_payload_bytes": len(rebuilt),
        "hiv1_payload_sha256": hashlib.sha256(rebuilt).hexdigest(),
        "archive_zip_bytes": len(archive_bytes),
        "archive_zip_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "zip_method": method,
        "meta_json_bytes_with_action": len(meta_with_bytes),
        "meta_json_bytes_without_action": len(meta_without_bytes),
        "meta_json_action_delta_bytes": len(meta_with_bytes) - len(meta_without_bytes),
        "archive_zip_materialized_on_disk": False,
        "archive_zip_bytes_are_rebuilt_minimal": True,
    }


def _support_encoding_rows(
    actions: Sequence[TargetRegionPixelAction],
    masks: Sequence[np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    coord_bytes = int(sum(action.yx.nbytes for action in actions))
    rows.append({"encoding": "explicit_yx_u16_coordinates", "encoded_bytes": coord_bytes, "lossless": True})
    rows.append({"encoding": "coordinate_list_brotli_q11", "encoded_bytes": _brotli_len(b"".join(action.yx.tobytes(order="C") for action in actions)), "lossless": True})
    rows.append({"encoding": "bitmap_packbits", "encoded_bytes": sum(_bitmap_bytes(mask) for mask in masks), "lossless": True})
    rows.append({"encoding": "rle_u32_start_len", "encoded_bytes": sum(_rle_start_len_bytes(mask) for mask in masks), "lossless": True})
    for tile in (8, 16, 32):
        rows.append(
            {
                "encoding": f"tile_set_{tile}x{tile}_bitmap",
                "encoded_bytes": sum(_tile_set_bytes(mask, tile=tile) for mask in masks),
                "lossless": True,
            }
        )
    try:
        path_total = 0
        for action, mask in zip(actions, masks, strict=True):
            support = path_tube_support_from_mask(
                mask,
                pair_index=int(action.pair_index),
                frame_index=int(action.frame_index),
                target_class=0,
                epsilon=2.0,
            )
            path_total += int(support.as_dict()["support_encoded_bytes"])
        rows.append({"encoding": "path_tube_zlib_rdp2", "encoded_bytes": path_total, "lossless": True})
    except Exception as exc:
        rows.append(
            {
                "encoding": "path_tube_zlib_rdp2",
                "encoded_bytes": None,
                "lossless": False,
                "blockers": [f"path_tube_support_failed:{type(exc).__name__}"],
            }
        )
    return rows


def _action_encoding_rows(actions: Sequence[TargetRegionPixelAction]) -> list[dict[str, Any]]:
    rgb = np.concatenate([np.asarray(action.rgb_u8, dtype=np.uint8) for action in actions], axis=0)
    raw = rgb.tobytes(order="C")
    rows: list[dict[str, Any]] = [
        {"encoding": "exact_rgb_u8", "encoded_bytes": len(raw), "lossless": True},
        {"encoding": "exact_rgb_u8_brotli", "encoded_bytes": _brotli_len(raw), "lossless": True},
    ]
    unique = np.unique(rgb, axis=0)
    index_width = 1 if unique.shape[0] <= 256 else 2 if unique.shape[0] <= 65536 else 4
    rows.append(
        {
            "encoding": "palette_rgb_u8_indices",
            "encoded_bytes": int(unique.shape[0] * 3 + rgb.shape[0] * index_width),
            "lossless": True,
            "palette_size": int(unique.shape[0]),
            "index_width_bytes": int(index_width),
        }
    )
    constant = bool(unique.shape[0] == 1)
    rows.append(
        {
            "encoding": "constant_class_attractor_rgb_u8",
            "encoded_bytes": 3,
            "lossless": constant,
            "blockers": [] if constant else ["constant_rgb_not_lossless_for_this_action"],
        }
    )
    rows.append(
        {
            "encoding": "per_channel_median_scalar_rgb_u8",
            "encoded_bytes": 3,
            "lossless": constant,
            "blockers": [] if constant else ["median_rgb_not_lossless_without_scorer_replay"],
        }
    )
    rows.append(
        {
            "encoding": "low_rank_rank1_rgb_float16",
            "encoded_bytes": int((rgb.shape[0] + 3) * 2),
            "lossless": False,
            "blockers": ["low_rank_action_needs_scorer_replay"],
        }
    )
    return rows


def _backend_ladder_rows(
    *,
    action_id: str,
    backend_fit: Mapping[str, Any],
    support_sha256: str,
) -> list[dict[str, Any]]:
    measured_groups = {str(value) for value in backend_fit.get("trained_groups") or []}
    measured = bool(backend_fit)
    rows: list[dict[str, Any]] = []
    for tier, groups in _BACKEND_LADDER_TIERS:
        is_measured = measured and set(groups) == measured_groups
        status = "not_run_current_artifact" if not is_measured else "measured"
        blockers = []
        if status != "measured":
            blockers.append("hinerv_backend_fit_ladder_tier_not_run_current_artifact")
        elif not _backend_realized(backend_fit):
            blockers.extend(_backend_blockers(backend_fit))
            blockers.append("hinerv_backend_fit_ladder_tier_not_realized")
        rows.append(
            {
                "schema": "hi_nerv_target_region_backend_ladder_row.v1",
                "action_id": action_id,
                "support_sha256": support_sha256,
                "tier": tier,
                "trained_groups": list(groups),
                "status": status,
                "wrong_to_target": _first_int(backend_fit, "wrong_to_target_count") if status == "measured" else None,
                "target_to_wrong": _first_int(backend_fit, "target_to_wrong_count") if status == "measured" else None,
                "accepted_step_count": _first_int(backend_fit, "accepted_step_count") if status == "measured" else None,
                "realized_target_wall": bool(backend_fit.get("realized_target_wall") is True) if status == "measured" else False,
                "first_failed_surface": None if not blockers else "backend_realization",
                "blockers": list(dict.fromkeys(blockers)),
            }
        )
    if measured and (not measured_groups or all(set(groups) != measured_groups for _, groups in _BACKEND_LADDER_TIERS)):
        rows.append(
            {
                "schema": "hi_nerv_target_region_backend_ladder_row.v1",
                "action_id": action_id,
                "support_sha256": support_sha256,
                "tier": "observed_current_backend_attempt",
                "trained_groups": sorted(measured_groups),
                "status": "measured",
                "wrong_to_target": _first_int(backend_fit, "wrong_to_target_count"),
                "target_to_wrong": _first_int(backend_fit, "target_to_wrong_count"),
                "accepted_step_count": _first_int(backend_fit, "accepted_step_count"),
                "realized_target_wall": bool(backend_fit.get("realized_target_wall") is True),
                "first_failed_surface": "backend_realization" if not _backend_realized(backend_fit) else None,
                "blockers": _backend_blockers(backend_fit),
            }
        )
    return rows


def _lowering_race_verdict(
    *,
    action_id: str,
    support_sha256: str,
    direct_teacher: Mapping[str, Any],
    backend_fit: Mapping[str, Any],
    current_sidecar: Mapping[str, Any],
    sidecar_rows: Sequence[Mapping[str, Any]],
    action_effect_sources: Sequence[ActionEffect | Mapping[str, Any]],
    sidecar_support_mismatch: bool,
    sidecar_survival_identity_mismatch: bool,
) -> dict[str, Any]:
    """Summarize the fixed-action backend/sidecar lowering race.

    The race consumes the same ActionEffect rows written to disk.  If the direct
    teacher and survived sidecar supports differ, the report is still emitted
    for diagnostics, but the verdict is forced fail-closed so a different
    support cannot accidentally clear the backend decision.
    """

    effects: list[ActionEffect] = []
    for row in sidecar_rows:
        effect = row.get("action_effect") if isinstance(row, Mapping) else None
        if isinstance(effect, Mapping):
            effects.append(ActionEffect.from_dict(effect))
    backend_effect = _nested_mapping(backend_fit, ("action_effect",))
    if backend_effect:
        effects.append(ActionEffect.from_dict(backend_effect))
    effects.extend(_coerce_action_effect_source_rows(action_effect_sources))

    report = build_lowering_race_report(
        action_id=action_id,
        action_effects=effects,
        expected_support_sha256=support_sha256,
    )
    verdict = dict(report["verdict"])
    if sidecar_support_mismatch or sidecar_survival_identity_mismatch:
        first_failing_surface = (
            _SURVIVAL_IDENTITY_MISMATCH
            if sidecar_survival_identity_mismatch
            else _SUPPORT_IDENTITY_MISMATCH
        )
        verdict.update(
            {
                "support_sha256": support_sha256,
                "direct_teacher_status": first_failing_surface,
                "backend_status": first_failing_surface,
                "sidecar_status": first_failing_surface,
                "best_lowering": "none",
                "first_failing_surface": first_failing_surface,
                "backend_realization_complete": False,
                "sidecar_lowering_complete": False,
                "authority": "none",
                "promotion_eligible": False,
                "delta_score_nonrate": None,
                "delta_score_total": None,
                "delta_bytes": None,
                "value_per_byte": None,
            }
        )

    return {
        "schema": EVALUATOR_ACTION_LOWERING_RACE_SCHEMA,
        "action_id": action_id,
        "support_sha256": support_sha256,
        "direct_teacher_support_sha256": _direct_teacher_support_identity(direct_teacher)[
            "comparison_support_sha256"
        ],
        "same_support_as_direct_teacher": not sidecar_support_mismatch,
        "same_survival_identity_as_archive": not sidecar_survival_identity_mismatch,
        "best_lowering": verdict["best_lowering"],
        "first_failing_surface": verdict["first_failing_surface"],
        "backend_realization_complete": bool(
            verdict.get("backend_realization_complete") is True
        ),
        "sidecar_lowering_complete": bool(
            verdict.get("sidecar_lowering_complete") is True
        ),
        "verdict": verdict,
        "current_sidecar_candidate_id": current_sidecar.get("candidate_id"),
        "candidate_count": len(report.get("lowering_candidates") or []),
        "lowering_candidates": report.get("lowering_candidates") or [],
        "target_accounting": report.get("target_accounting") or {},
        "support_identity": report.get("support_identity") or {},
        "promotion_eligible": False,
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _coerce_action_effect_source_rows(
    rows: Sequence[ActionEffect | Mapping[str, Any]],
) -> list[ActionEffect]:
    effects: list[ActionEffect] = []
    for row in rows:
        try:
            if isinstance(row, ActionEffect):
                effects.append(row)
            elif isinstance(row, Mapping):
                effects.append(ActionEffect.from_dict(row))
        except Exception:
            continue
    return effects


def write_hinerv_target_region_action_comparison(
    report: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    out = Path(output_dir).expanduser().resolve(strict=False)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "hinerv_target_region_action_sidecar_backend_comparison.json"
    rows_path = out / "hinerv_target_region_action_sidecar_backend_action_effects.jsonl"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = list(report.get("sidecar_encoding_candidates") or [])
    with open(rows_path, "w", encoding="utf-8") as handle:
        for row in rows:
            effect = row.get("action_effect") if isinstance(row, Mapping) else None
            if isinstance(effect, Mapping):
                handle.write(json.dumps(effect, sort_keys=True) + "\n")
    return {
        "report_path": report_path.as_posix(),
        "action_effect_rows_path": rows_path.as_posix(),
        "row_count": len(rows),
        "report_sha256": sha256_file(report_path),
    }


def _action_mask(action: TargetRegionPixelAction) -> np.ndarray:
    mask = np.zeros((int(action.height), int(action.width)), dtype=bool)
    y = action.yx[:, 0].astype(np.int64, copy=False)
    x = action.yx[:, 1].astype(np.int64, copy=False)
    mask[y, x] = True
    return mask


def _bitmap_bytes(mask: np.ndarray) -> int:
    packed = np.packbits(np.asarray(mask, dtype=np.uint8).reshape(-1), bitorder="little")
    return int(len(packed) + 8)


def _rle_start_len_bytes(mask: np.ndarray) -> int:
    flat = np.asarray(mask, dtype=bool).reshape(-1)
    starts: list[int] = []
    lengths: list[int] = []
    cursor = 0
    while cursor < flat.size:
        if not bool(flat[cursor]):
            cursor += 1
            continue
        start = cursor
        while cursor < flat.size and bool(flat[cursor]):
            cursor += 1
        starts.append(start)
        lengths.append(cursor - start)
    return int(8 + 8 * len(starts))


def _tile_set_bytes(mask: np.ndarray, *, tile: int) -> int:
    src = np.asarray(mask, dtype=bool)
    total = 8
    for y0 in range(0, src.shape[0], tile):
        for x0 in range(0, src.shape[1], tile):
            block = src[y0 : y0 + tile, x0 : x0 + tile]
            if not np.any(block):
                continue
            total += 4
            total += len(np.packbits(block.astype(np.uint8).reshape(-1), bitorder="little"))
    return int(total)


def _brotli_len(payload: bytes) -> int:
    return len(brotli.compress(bytes(payload), quality=11))


def _find_first_wall_normal_lift(payload: Any) -> dict[str, Any]:
    if isinstance(payload, Mapping):
        if payload.get("schema") == "tac.target_region_wall_normal_lift.v1":
            return dict(payload)
        direct = payload.get("target_region_wall_normal_lift")
        if isinstance(direct, Mapping):
            return dict(direct)
        for value in payload.values():
            found = _find_first_wall_normal_lift(value)
            if found:
                return found
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for value in payload:
            found = _find_first_wall_normal_lift(value)
            if found:
                return found
    return {}


def _find_first_sidecar_candidate(payload: Any) -> dict[str, Any]:
    if isinstance(payload, Mapping):
        masked = _nested_mapping(payload, ("candidate_frontier_telemetry", "masked_residual_oracle"))
        if isinstance(masked.get("best_candidate"), Mapping):
            return dict(masked["best_candidate"])
        if payload.get("schema") == "hi_nerv_target_region_masked_residual_oracle_candidate.v1":
            return dict(payload)
        for value in payload.values():
            found = _find_first_sidecar_candidate(value)
            if found:
                return found
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for value in payload:
            found = _find_first_sidecar_candidate(value)
            if found:
                return found
    return {}


def _sidecar_admission_from_wall(
    wall: Mapping[str, Any],
    sidecar_candidate: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = dict(sidecar_candidate)
    if not candidate:
        candidate = _nested_mapping(
            wall,
            ("sidecar_fallback",),
            ("candidate_frontier_telemetry", "masked_residual_oracle", "best_candidate"),
        )
    # Most reports keep the full best-candidate deeper in the runner report, so
    # fall back to direct fields when only the compact wall record is available.
    admission = _nested_mapping(candidate, ("admission_decision",))
    out = dict(admission)
    for key in (
        "target_class",
        "region_id",
        "exact_delta_score_nonrate",
        "target_region_action_payload_bytes",
        "region_argmax_transitions",
        "global_target_transitions",
        "mask_name",
    ):
        if key in candidate and key not in out:
            out[key] = candidate[key]
        if key in wall and key not in out:
            out[key] = wall[key]
    fallback = _nested_mapping(wall, ("sidecar_fallback",))
    if fallback:
        out.setdefault("exact_delta_score_nonrate", fallback.get("exact_delta_score_nonrate"))
        out.setdefault("target_region_action_payload_bytes", fallback.get("payload_bytes"))
    return out


def _direct_teacher_from_wall(wall: Mapping[str, Any]) -> dict[str, Any]:
    return dict(_nested_mapping(wall, ("direct_teacher",)))


def _backend_fit_from_wall(wall: Mapping[str, Any]) -> dict[str, Any]:
    return dict(_nested_mapping(wall, ("backend_fit",)))


def _direct_teacher_support_identity(direct: Mapping[str, Any]) -> dict[str, str | None]:
    archive_sha = _nested_first_text(
        direct,
        ("archive_executable_support_sha256",),
        ("action_effect", "archive_executable_support_sha256"),
    )
    mask_sha = _nested_first_text(
        direct,
        ("support_sha256",),
        ("action_effect", "support_sha256"),
    )
    comparison_sha = archive_sha or mask_sha
    return {
        "comparison_support_sha256": comparison_sha,
        "comparison_hash_domain": (
            "target_region_action_coordinates_v1"
            if archive_sha
            else ("bool_mask_bhw" if mask_sha else None)
        ),
        "archive_executable_support_sha256": archive_sha,
        "mask_support_sha256": mask_sha,
    }


def _survival_identity_status(
    survival: Mapping[str, Any],
    *,
    action_id: str,
    support_sha256: str,
    archive_sha256: str,
    program_sha256: str | None,
    decoded_support_sha256: str | None,
    decoded_action_sha256: str | None,
) -> dict[str, Any]:
    observed_action_id = _nested_first_text(
        survival,
        ("action_id",),
        ("target_region_actions", "action_id"),
    )
    observed_support_sha256 = _nested_first_text(
        survival,
        ("target_region_actions", "support_sha256"),
        ("expected_support_sha256",),
        ("support_sha256",),
    )
    observed_archive_sha256 = _nested_first_text(
        survival,
        ("archive_sha256",),
        ("expected_archive_sha256",),
        ("archive", "sha256"),
    )
    observed_program_sha256 = _nested_first_text(
        survival,
        ("target_region_action_program_sha256",),
        ("expected_program_sha256",),
        ("target_region_actions", "program_sha256"),
        ("target_region_actions", "target_region_action_program_sha256"),
        ("target_region_actions", "program_base64_sha256"),
    )
    observed_decoded_support_sha256 = _nested_first_text(
        survival,
        ("decoded_support_sha256",),
        ("target_region_actions", "decoded_support_sha256"),
    )
    observed_decoded_action_sha256 = _nested_first_text(
        survival,
        ("decoded_action_sha256",),
        ("target_region_actions", "decoded_action_sha256"),
    )
    blockers: list[str] = []
    decoded_support_matches = (
        decoded_support_sha256 is not None
        and observed_decoded_support_sha256 == decoded_support_sha256
    )
    if not observed_action_id:
        blockers.append("target_region_action_survival_action_id_missing")
    elif observed_action_id != action_id:
        blockers.append("target_region_action_survival_action_id_mismatch")
    if not observed_support_sha256 and not decoded_support_matches:
        blockers.append("target_region_action_survival_support_sha256_missing")
    elif observed_support_sha256 != support_sha256 and not decoded_support_matches:
        blockers.append("target_region_action_survival_support_sha256_mismatch")
    if not observed_archive_sha256:
        blockers.append("target_region_action_survival_archive_sha256_missing")
    elif observed_archive_sha256 != archive_sha256:
        blockers.append("target_region_action_survival_archive_sha256_mismatch")
    if program_sha256 is not None:
        if not observed_program_sha256:
            blockers.append("target_region_action_survival_program_sha256_missing")
        elif observed_program_sha256 != program_sha256:
            blockers.append("target_region_action_survival_program_sha256_mismatch")
    if decoded_support_sha256 is not None:
        if not observed_decoded_support_sha256:
            blockers.append("target_region_action_survival_decoded_support_sha256_missing")
        elif observed_decoded_support_sha256 != decoded_support_sha256:
            blockers.append("target_region_action_survival_decoded_support_sha256_mismatch")
    if decoded_action_sha256 is not None:
        if not observed_decoded_action_sha256:
            blockers.append("target_region_action_survival_decoded_action_sha256_missing")
        elif observed_decoded_action_sha256 != decoded_action_sha256:
            blockers.append("target_region_action_survival_decoded_action_sha256_mismatch")
    return {
        "schema": "hi_nerv_target_region_action_survival_identity.v1",
        "action_id": action_id,
        "support_sha256": support_sha256,
        "decoded_support_sha256": decoded_support_sha256,
        "decoded_action_sha256": decoded_action_sha256,
        "archive_sha256": archive_sha256,
        "target_region_action_program_sha256": program_sha256,
        "survival_action_id": observed_action_id,
        "survival_support_sha256": observed_support_sha256,
        "survival_decoded_support_sha256": observed_decoded_support_sha256,
        "survival_decoded_action_sha256": observed_decoded_action_sha256,
        "survival_archive_sha256": observed_archive_sha256,
        "survival_program_sha256": observed_program_sha256,
        "same_action_id": observed_action_id == action_id,
        "same_support_sha256": observed_support_sha256 == support_sha256,
        "same_decoded_support_sha256": (
            observed_decoded_support_sha256 == decoded_support_sha256
            if decoded_support_sha256 is not None
            else None
        ),
        "same_decoded_action_sha256": (
            observed_decoded_action_sha256 == decoded_action_sha256
            if decoded_action_sha256 is not None
            else None
        ),
        "same_archive_sha256": observed_archive_sha256 == archive_sha256,
        "same_program_sha256": (
            observed_program_sha256 == program_sha256 if program_sha256 is not None else None
        ),
        "passed": not blockers,
        "blockers": blockers,
    }


def _program_base64_sha256(program: _ActionProgram) -> str | None:
    raw_b64 = program.meta.get(TARGET_REGION_ACTION_META_KEY)
    if not isinstance(raw_b64, str) or not raw_b64:
        return None
    return hashlib.sha256(raw_b64.encode("ascii")).hexdigest()


def _backend_realized(backend: Mapping[str, Any]) -> bool:
    return bool(
        backend.get("realized_target_wall") is True
        and int(backend.get("wrong_to_target_count") or 0) > 0
        and int(backend.get("accepted_step_count") or 0) > 0
    )


def _backend_blockers(backend: Mapping[str, Any]) -> list[str]:
    blockers = [str(value) for value in backend.get("blockers") or []]
    action_effect = _nested_mapping(backend, ("action_effect",))
    blockers.extend(str(value) for value in action_effect.get("blockers") or [])
    return list(dict.fromkeys(blockers))


def _transition_counts(admission_or_candidate: Mapping[str, Any]) -> dict[str, int | None]:
    transitions = _nested_mapping(admission_or_candidate, ("region_argmax_transitions",))
    if not transitions:
        transitions = _nested_mapping(admission_or_candidate, ("argmax_transitions",))
    return {
        "wrong_to_target_count": _first_int(transitions, "wrong_to_target_count", "target_hard_won_count"),
        "target_to_wrong_count": _first_int(transitions, "target_to_wrong_count", "target_hard_lost_count"),
        "wrong_to_wrong_count": _first_int(transitions, "wrong_to_wrong_count"),
        "net_target_support_delta": _first_int(transitions, "net_target_support_delta"),
        "argmax_changed_count_region": _first_int(transitions, "argmax_changed_count_region"),
    }


def _load_mapping(payload: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(payload, Mapping):
        return dict(payload)
    path = Path(payload).expanduser().resolve(strict=True)
    return json.loads(path.read_text(encoding="utf-8"))


def _program_action_id(program: _ActionProgram) -> str:
    digest = hashlib.sha256()
    digest.update(program.stored_payload)
    return f"hinerv_target_region_action:{digest.hexdigest()[:24]}"


def _nested_mapping(payload: Mapping[str, Any], *paths: Sequence[str]) -> dict[str, Any]:
    for path in paths:
        cur: Any = payload
        for part in path:
            if not isinstance(cur, Mapping):
                cur = None
                break
            cur = cur.get(part)
        if isinstance(cur, Mapping):
            return dict(cur)
    return {}


def _nested_first_text(payload: Mapping[str, Any], *paths: Sequence[str]) -> str | None:
    for path in paths:
        cur: Any = payload
        for part in path:
            if not isinstance(cur, Mapping):
                cur = None
                break
            cur = cur.get(part)
        if isinstance(cur, str) and cur:
            return cur
    return None


def _first_text(payload: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _first_float(payload: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, bool) or value is None:
            continue
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(val):
            return val
    return None


def _first_int(payload: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, bool) or value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _int_tuple(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, int):
        return (int(value),)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        out: list[int] = []
        for item in value:
            if item is None or isinstance(item, bool):
                continue
            try:
                out.append(int(item))
            except (TypeError, ValueError):
                continue
        return tuple(out)
    try:
        return (int(value),)
    except (TypeError, ValueError):
        return ()


def _str_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(str(item) for item in value if str(item))
    return (str(value),)


__all__ = [
    "HI_NERV_TARGET_REGION_ACTION_COMPARISON_SCHEMA",
    "TargetRegionActionProgram",
    "build_hinerv_target_region_action_comparison",
    "build_hinerv_target_region_action_comparison_from_archive",
    "write_hinerv_target_region_action_comparison",
]
