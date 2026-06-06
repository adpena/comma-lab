# SPDX-License-Identifier: MIT
"""ActionEffect conversion for HiNeRV scorer-domain bootstrap telemetry."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.analysis.action_effect import (
    ActionEffect,
    ScoreAuthority,
    compute_delta_scores,
)
from tac.repo_io import sha256_file

BOOTSTRAP_SCHEMA = "hi_nerv_scorer_domain_bootstrap.v1"


def build_action_effect_from_hinerv_scorer_bootstrap_artifact(
    training_artifact: Mapping[str, Any],
    *,
    training_artifact_path: Path | None = None,
    consumer: str = "nerv_long_run_launch_gate",
) -> ActionEffect:
    """Build one ActionEffect row from a training artifact containing bootstrap telemetry."""

    bootstrap_path, bootstrap = select_hinerv_scorer_bootstrap(training_artifact)
    return build_action_effect_from_hinerv_scorer_bootstrap_row(
        bootstrap,
        artifact_ref=(training_artifact_path.as_posix() if training_artifact_path is not None else None),
        artifact_sha256=(
            sha256_file(training_artifact_path)
            if training_artifact_path is not None and training_artifact_path.is_file()
            else None
        ),
        archive_sha256=_first_text(training_artifact, "archive_sha256"),
        source_bootstrap_path=bootstrap_path,
        consumer=consumer,
    )


def build_action_effect_from_hinerv_scorer_bootstrap_row(
    bootstrap: Mapping[str, Any],
    *,
    artifact_ref: str | None = None,
    artifact_sha256: str | None = None,
    archive_sha256: str | None = None,
    source_bootstrap_path: str | None = None,
    consumer: str = "nerv_long_run_launch_gate",
) -> ActionEffect:
    """Build one ActionEffect row directly from a scorer-domain bootstrap row.

    Only observed fields are copied. Missing official Pose/byte/survival terms
    become blockers on the row; no zero endpoint is fabricated.
    """

    if not isinstance(bootstrap, Mapping):
        raise TypeError("bootstrap row must be a mapping")
    if bootstrap.get("schema") != BOOTSTRAP_SCHEMA:
        raise ValueError(f"bootstrap schema must be {BOOTSTRAP_SCHEMA!r}")

    metrics_before = _mapping(bootstrap.get("metrics_before"))
    metrics_after = _mapping(bootstrap.get("metrics_after"))
    old_d_seg = _first_float(
        metrics_before,
        "segnet_margin_bootstrap_argmax_disagreement",
        "segnet_hard_birth_bootstrap_argmax_disagreement",
    )
    new_d_seg = _first_float(
        metrics_after,
        "segnet_margin_bootstrap_argmax_disagreement",
        "segnet_hard_birth_bootstrap_argmax_disagreement",
    )
    old_region_debt = _first_float(
        metrics_before,
        "segnet_margin_bootstrap_score_weighted_total_unsolved_argmax_mass",
        "segnet_hard_birth_bootstrap_score_weighted_total_unsolved_argmax_mass",
    )
    new_region_debt = _first_float(
        metrics_after,
        "segnet_margin_bootstrap_score_weighted_total_unsolved_argmax_mass",
        "segnet_hard_birth_bootstrap_score_weighted_total_unsolved_argmax_mass",
    )
    worst_class = _first_int(
        metrics_after,
        "segnet_margin_bootstrap_worst_class_index",
        "segnet_hard_birth_bootstrap_worst_class_index",
    )
    old_d_pose = _first_float(bootstrap, "old_d_pose", "old_posenet_distortion")
    new_d_pose = _first_float(bootstrap, "new_d_pose", "new_posenet_distortion")
    old_bytes = _first_int(bootstrap, "old_archive_bytes", "archive_bytes_old")
    new_bytes = _first_int(bootstrap, "new_archive_bytes", "archive_bytes_new")
    deltas = compute_delta_scores(old_d_seg, new_d_seg, old_d_pose, new_d_pose, old_bytes, new_bytes)

    digest = artifact_sha256 or _stable_row_digest(bootstrap)
    action_id = _first_text(bootstrap, "action_id", "actuator_id") or f"hinerv-scorer-domain-bootstrap-{digest[:12]}"
    payload_sections = _strings(bootstrap.get("archive_charged_decoder_tensors"))
    if not payload_sections:
        payload_sections = _strings(bootstrap.get("bootstrap_update_applied_tensor_names"))

    blockers = [
        *_bootstrap_blockers(
            bootstrap=bootstrap,
            old_d_seg=old_d_seg,
            new_d_seg=new_d_seg,
            old_d_pose=old_d_pose,
            new_d_pose=new_d_pose,
            old_bytes=old_bytes,
            new_bytes=new_bytes,
            delta_score_nonrate=deltas.delta_score_nonrate,
            delta_score_total=deltas.delta_score_total,
        ),
        *([f"source_bootstrap_path:{source_bootstrap_path}"] if source_bootstrap_path else []),
        f"accepted_step_count:{_int_or_none(bootstrap.get('accepted_step_count')) or 0}",
        f"raw_bootstrap_authority:{bootstrap.get('authority') or ''}",
    ]

    return ActionEffect.build(
        action_id=action_id,
        family="hinerv",
        action_kind="scorer_domain_hard_birth_bootstrap",
        authority=ScoreAuthority.BATCH_LOCAL_LIVE_MLX.value,
        producer="hinerv_scorer_domain_bootstrap",
        consumer=consumer,
        pair_ids=_pair_ids(bootstrap),
        region_ids=(f"class:{worst_class}",) if worst_class is not None else (),
        payload_sections=payload_sections,
        old_d_seg=old_d_seg,
        new_d_seg=new_d_seg,
        old_d_pose=old_d_pose,
        new_d_pose=new_d_pose,
        old_bytes=old_bytes,
        new_bytes=new_bytes,
        exact_score_decision="not_applicable",
        artifact_ref=artifact_ref,
        archive_sha256=archive_sha256,
        taint_status="unknown",
        old_region_debt=old_region_debt,
        new_region_debt=new_region_debt,
        seg_score_delta=None if old_d_seg is None or new_d_seg is None else 100.0 * (new_d_seg - old_d_seg),
        blockers=tuple(dict.fromkeys(blockers)),
    )


def select_hinerv_scorer_bootstrap(payload: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    candidates = list(_iter_bootstrap_rows(payload))
    if not candidates:
        raise ValueError(f"{BOOTSTRAP_SCHEMA} row missing from training artifact")

    def key(item: tuple[tuple[str, ...], Mapping[str, Any]]) -> tuple[int, int, str]:
        row_path, row = item
        accepted = _int_or_none(row.get("accepted_step_count")) or 0
        preferred = int("substrate_supplied_score_aware_training" in ".".join(row_path))
        return accepted, preferred, ".".join(row_path)

    row_path, row = max(candidates, key=key)
    return ".".join(row_path), row


def _iter_bootstrap_rows(payload: Any, *, path: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], Mapping[str, Any]]]:
    if isinstance(payload, Mapping):
        if payload.get("schema") == BOOTSTRAP_SCHEMA:
            yield path, payload
        for key, value in payload.items():
            yield from _iter_bootstrap_rows(value, path=(*path, str(key)))
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for index, value in enumerate(payload):
            yield from _iter_bootstrap_rows(value, path=(*path, f"[{index}]"))


def _bootstrap_blockers(
    *,
    bootstrap: Mapping[str, Any],
    old_d_seg: float | None,
    new_d_seg: float | None,
    old_d_pose: float | None,
    new_d_pose: float | None,
    old_bytes: int | None,
    new_bytes: int | None,
    delta_score_nonrate: float | None,
    delta_score_total: float | None,
) -> tuple[str, ...]:
    blockers: list[str] = []
    if (_int_or_none(bootstrap.get("accepted_step_count")) or 0) <= 0:
        blockers.append("hinerv_scorer_bootstrap_no_accepted_update")
    if old_d_seg is None or new_d_seg is None:
        blockers.append("hinerv_scorer_bootstrap_seg_endpoint_missing")
    if old_d_pose is None or new_d_pose is None:
        blockers.append("hinerv_scorer_bootstrap_pose_endpoint_missing")
    if old_bytes is None or new_bytes is None:
        blockers.append("hinerv_scorer_bootstrap_archive_byte_endpoint_missing")
    if delta_score_nonrate is None:
        blockers.append("hinerv_scorer_bootstrap_exact_nonrate_delta_missing")
    if delta_score_total is None:
        blockers.append("hinerv_scorer_bootstrap_exact_total_delta_missing")
    blockers.extend(
        [
            "hinerv_scorer_bootstrap_uint8_changed_pixels_missing",
            "hinerv_scorer_bootstrap_seg_input_delta_missing",
            "hinerv_scorer_bootstrap_posenet_input_delta_missing",
            "hinerv_scorer_bootstrap_fakequant_survival_missing",
            "hinerv_scorer_bootstrap_parseback_survival_missing",
            "hinerv_scorer_bootstrap_inflate_survival_missing",
        ]
    )
    return tuple(dict.fromkeys(blockers))


def _stable_row_digest(row: Mapping[str, Any]) -> str:
    rendered = json.dumps(row, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first_float(row: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, bool) or value is None:
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if parsed == parsed and abs(parsed) != float("inf") and parsed >= 0.0:
            return parsed
    return None


def _first_int(row: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        parsed = _int_or_none(row.get(key))
        if parsed is not None:
            return parsed
    return None


def _first_text(row: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return ()
    out: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            out.append(text)
    return tuple(dict.fromkeys(out))


def _pair_ids(bootstrap: Mapping[str, Any]) -> tuple[int, ...]:
    direct = bootstrap.get("pair_ids")
    if isinstance(direct, Sequence) and not isinstance(direct, (str, bytes, bytearray)):
        out = tuple(item for item in (_int_or_none(value) for value in direct) if item is not None)
        if out:
            return out
    pair_index = _first_int(bootstrap, "pair_index")
    return (pair_index,) if pair_index is not None else ()


__all__ = [
    "BOOTSTRAP_SCHEMA",
    "build_action_effect_from_hinerv_scorer_bootstrap_artifact",
    "build_action_effect_from_hinerv_scorer_bootstrap_row",
    "select_hinerv_scorer_bootstrap",
]
