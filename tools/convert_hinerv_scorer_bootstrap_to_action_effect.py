#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Convert HiNeRV scorer-domain bootstrap telemetry into ActionEffect JSONL.

This is the scorer-domain-bootstrap sibling of the target-region birth receipt
path already emitted by the live runner.  It does not invent a birth receipt and
does not mint score authority: only fields present in the training artifact are
copied into ``tac.action_effect.v1``.  Missing exact Pose/byte/survival endpoints
become explicit blockers on the row so launch gates and planners can fail
closed while still seeing the accepted update as a typed action.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.action_effect import (  # noqa: E402
    ActionEffect,
    ScoreAuthority,
    append_action_effect,
    compute_delta_scores,
    validate_action_effect_payload,
)
from tac.repo_io import sha256_file  # noqa: E402

BOOTSTRAP_SCHEMA = "hi_nerv_scorer_domain_bootstrap.v1"
CONVERSION_SCHEMA = "tac.hinerv_scorer_bootstrap_action_effect_conversion.v1"


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} JSON malformed at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _iter_bootstrap_rows(payload: Any, *, path: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], Mapping[str, Any]]]:
    if isinstance(payload, Mapping):
        if payload.get("schema") == BOOTSTRAP_SCHEMA:
            yield path, payload
        for key, value in payload.items():
            yield from _iter_bootstrap_rows(value, path=(*path, str(key)))
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for index, value in enumerate(payload):
            yield from _iter_bootstrap_rows(value, path=(*path, f"[{index}]"))


def _select_bootstrap(payload: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
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


def build_action_effect_from_hinerv_scorer_bootstrap(
    training_artifact: Mapping[str, Any],
    *,
    training_artifact_path: Path,
    consumer: str = "nerv_long_run_launch_gate",
) -> ActionEffect:
    """Build one ActionEffect row from a real scorer-domain bootstrap object."""

    bootstrap_path, bootstrap = _select_bootstrap(training_artifact)
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
    pair_ids = _pair_ids(bootstrap)
    accepted_steps = _int_or_none(bootstrap.get("accepted_step_count")) or 0
    action_id = (
        _first_text(bootstrap, "action_id", "actuator_id")
        or f"hinerv-scorer-domain-bootstrap-{sha256_file(training_artifact_path)[:12]}"
    )

    old_d_pose = _first_float(bootstrap, "old_d_pose", "old_posenet_distortion")
    new_d_pose = _first_float(bootstrap, "new_d_pose", "new_posenet_distortion")
    old_bytes = _first_int(bootstrap, "old_archive_bytes", "archive_bytes_old")
    new_bytes = _first_int(bootstrap, "new_archive_bytes", "archive_bytes_new")
    deltas = compute_delta_scores(old_d_seg, new_d_seg, old_d_pose, new_d_pose, old_bytes, new_bytes)

    blockers = _bootstrap_blockers(
        bootstrap=bootstrap,
        old_d_seg=old_d_seg,
        new_d_seg=new_d_seg,
        old_d_pose=old_d_pose,
        new_d_pose=new_d_pose,
        old_bytes=old_bytes,
        new_bytes=new_bytes,
        delta_score_nonrate=deltas.delta_score_nonrate,
        delta_score_total=deltas.delta_score_total,
    )
    payload_sections = _strings(bootstrap.get("archive_charged_decoder_tensors"))
    if not payload_sections:
        payload_sections = _strings(bootstrap.get("bootstrap_update_applied_tensor_names"))

    return ActionEffect.build(
        action_id=action_id,
        family="hinerv",
        action_kind="scorer_domain_hard_birth_bootstrap",
        authority=ScoreAuthority.BATCH_LOCAL_LIVE_MLX.value,
        producer="hinerv_scorer_domain_bootstrap",
        consumer=consumer,
        pair_ids=pair_ids,
        region_ids=(f"class:{worst_class}",) if worst_class is not None else (),
        payload_sections=payload_sections,
        old_d_seg=old_d_seg,
        new_d_seg=new_d_seg,
        old_d_pose=old_d_pose,
        new_d_pose=new_d_pose,
        old_bytes=old_bytes,
        new_bytes=new_bytes,
        exact_score_decision="not_applicable",
        artifact_ref=training_artifact_path.as_posix(),
        archive_sha256=_first_text(training_artifact, "archive_sha256"),
        taint_status="unknown",
        old_region_debt=old_region_debt,
        new_region_debt=new_region_debt,
        seg_score_delta=None if old_d_seg is None or new_d_seg is None else 100.0 * (new_d_seg - old_d_seg),
        blockers=(
            *blockers,
            f"source_bootstrap_path:{bootstrap_path}",
            f"accepted_step_count:{accepted_steps}",
            f"raw_bootstrap_authority:{bootstrap.get('authority') or ''}",
        ),
    )


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-artifact", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--consumer", default="nerv_long_run_launch_gate")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        artifact_path = args.training_artifact.expanduser().resolve(strict=False)
        artifact = _load_json_object(artifact_path, label="training artifact")
        effect = build_action_effect_from_hinerv_scorer_bootstrap(
            artifact,
            training_artifact_path=artifact_path,
            consumer=args.consumer,
        )
        record = append_action_effect(effect, args.output_jsonl)
        validation = validate_action_effect_payload(record)
    except (OSError, ValueError, TypeError) as exc:
        print(f"FATAL: could not convert HiNeRV scorer bootstrap to ActionEffect: {exc}", file=sys.stderr)
        return 2

    summary = {
        "schema": CONVERSION_SCHEMA,
        "training_artifact": artifact_path.as_posix(),
        "training_artifact_sha256": sha256_file(artifact_path),
        "output_jsonl": args.output_jsonl.as_posix(),
        "action_id": record["action_id"],
        "authority": record["authority"],
        "delta_score_nonrate": record["delta_score_nonrate"],
        "delta_score_total": record["delta_score_total"],
        "delta_bytes": record["delta_bytes"],
        "value_per_byte": record["value_per_byte"],
        "blockers": record["blockers"],
        "validation": validation,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True) + "\n", end="")
    return 0 if validation.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
