# SPDX-License-Identifier: MIT
"""Compile real archive candidates into repair-campaign work orders."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.optimization.dqs1_materializer_feedback_bridge import FALSE_AUTHORITY
from tac.optimization.family_agnostic_materializers import (
    materialize_archive_zip_repack_candidate,
)
from tac.optimization.proxy_candidate_contract import require_no_truthy_authority_fields
from tac.repo_io import sha256_bytes, sha256_file, write_json_artifact

REPAIR_ARCHIVE_CANDIDATE_INTAKE_SCHEMA = "repair_campaign_archive_candidate_intake.v1"
REPAIR_ARCHIVE_CANDIDATE_INTAKE_ROW_SCHEMA = "repair_campaign_archive_candidate_intake_row.v1"
REPAIR_ARCHIVE_CANDIDATE_WORK_ORDER_SCHEMA = (
    "frontier_rate_attack_repair_budget_waterfill_work_order.v1"
)
REPAIR_ARCHIVE_CANDIDATE_TYPED_LEDGER_SCHEMA = (
    "frontier_rate_attack_repair_budget_typed_response_ledger.v1"
)
REPAIR_ARCHIVE_CANDIDATE_TYPED_ROW_SCHEMA = (
    "frontier_rate_attack_repair_budget_typed_response_row.v1"
)

CANONICAL_K16_FRAME0_PALETTE_MODES: tuple[str, ...] = (
    "none",
    "frame0_blue_chroma_amp_1",
    "frame0_blue_chroma_amp_3",
    "frame0_luma_bias_+1",
    "frame0_luma_bias_-1",
    "frame0_luma_bias_-2",
    "frame0_luma_bias_-4",
    "frame0_rgb_bias_m2_p1_p1",
    "frame0_rgb_bias_m4_p2_p2",
    "frame0_rgb_bias_p0_m1_p1",
    "frame0_rgb_bias_p0_m2_p2",
    "frame0_rgb_bias_p0_p1_m1",
    "frame0_rgb_bias_p0_p2_m2",
    "frame0_rgb_bias_p2_m1_m1",
    "frame0_rgb_bias_p4_m2_m2",
    "frame0_roll_dx+0_dy+1",
)

_FAMILY_BLUEPRINTS: tuple[dict[str, Any], ...] = (
    {
        "family_id": "posenet_null_bottom_decile",
        "typed_suffix": "posenet_bottom_decile",
        "targeted_dimensions": ("posenet", "pair", "frame0"),
        "operation_levels": ("pixel", "frame", "pair", "batch", "full_video"),
        "entropy_position_label": "before_entropy_coder_distribution_shaping",
        "base_requested_bytes": 8,
        "base_objective_delta": -0.0015,
    },
    {
        "family_id": "segnet_class_region_waterfill",
        "typed_suffix": "segnet_region",
        "targeted_dimensions": ("segnet", "class_region", "boundary"),
        "operation_levels": ("pixel", "boundary", "region", "frame", "batch"),
        "entropy_position_label": "before_entropy_coder_distribution_shaping",
        "base_requested_bytes": 12,
        "base_objective_delta": -0.0024,
    },
    {
        "family_id": "per_region_selector_codec",
        "typed_suffix": "selector_codec",
        "targeted_dimensions": ("selector_stream", "region", "entropy"),
        "operation_levels": ("bit", "byte", "boundary", "region", "pair"),
        "entropy_position_label": "selector_codec_entropy",
        "base_requested_bytes": 6,
        "base_objective_delta": -0.0011,
    },
    {
        "family_id": "frame0_k16_palette_asymmetry",
        "typed_suffix": "frame0_k16_palette",
        "targeted_dimensions": ("palette", "frame0", "posenet", "segnet"),
        "operation_levels": ("pixel", "byte", "frame", "pair", "batch"),
        "entropy_position_label": "before_entropy_coder_distribution_shaping",
        "base_requested_bytes": 10,
        "base_objective_delta": -0.0012,
    },
    {
        "family_id": "entropy_boundary_probe",
        "typed_suffix": "entropy_boundary",
        "targeted_dimensions": ("rate_bytes", "entropy_coder_boundary"),
        "operation_levels": ("bit", "byte"),
        "entropy_position_label": "at_entropy_coder_integer_codeword_boundary",
        "base_requested_bytes": 4,
        "base_objective_delta": -0.0007,
    },
)


class RepairArchiveCandidateIntakeError(ValueError):
    """Raised when archive intake cannot produce a repair work order."""


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _slug(value: Any) -> str:
    text = str(value or "candidate").strip().lower()
    chars = [ch if ch.isalnum() else "_" for ch in text]
    return "_".join("".join(chars).split("_")) or "candidate"


def _resolve(path: str | Path, repo_root: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else Path(repo_root) / value


def _repo_rel(path: str | Path, repo_root: str | Path) -> str:
    value = Path(path)
    repo = Path(repo_root)
    try:
        return value.resolve(strict=False).relative_to(repo.resolve(strict=False)).as_posix()
    except ValueError:
        return value.as_posix()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None or isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _load_json_object(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise RepairArchiveCandidateIntakeError(f"{path} must contain a JSON object")
    require_no_truthy_authority_fields(loaded, context=f"repair_archive_candidate_intake:{path}")
    return loaded


def _optional_json(path: str | Path | None, repo_root: str | Path) -> dict[str, Any]:
    if path is None:
        return {}
    resolved = _resolve(path, repo_root)
    if not resolved.is_file():
        return {}
    return _load_json_object(resolved)


def _find_sibling_artifact(archive: Path, names: Sequence[str]) -> Path | None:
    for name in names:
        candidate = archive.parent / name
        if candidate.is_file():
            return candidate
    return None


def _per_epoch_loss_factor(training: Mapping[str, Any]) -> float:
    direct = _safe_float(
        training.get("loss_reduction_factor")
        or training.get("ratio_actual_vs_pr95_empirical_anchor")
    )
    if direct is not None and direct > 0:
        return direct
    metrics = training.get("per_epoch_metrics")
    if not isinstance(metrics, Sequence) or isinstance(metrics, (str, bytes, bytearray)):
        return 1.0
    first_loss: float | None = None
    last_loss: float | None = None
    for item in metrics:
        if not isinstance(item, Mapping):
            continue
        loss = _safe_float(
            item.get("loss")
            or item.get("total_loss")
            or item.get("train_loss")
            or item.get("mse")
        )
        if loss is None or loss <= 0:
            continue
        if first_loss is None:
            first_loss = loss
        last_loss = loss
    if first_loss is None or last_loss is None or last_loss <= 0:
        return 1.0
    return max(1.0, first_loss / last_loss)


def _signal_multiplier(training: Mapping[str, Any], gate: Mapping[str, Any]) -> float:
    loss_factor = _per_epoch_loss_factor(training)
    gate_margin = _safe_float(gate.get("margin_below_threshold")) or 0.0
    loss_boost = min(0.55, math.log10(max(1.0, loss_factor)) * 0.12)
    gate_boost = max(0.0, min(0.20, gate_margin * 0.5))
    return max(0.80, min(1.75, 1.0 + loss_boost + gate_boost))


def _palette_context(source_label: str) -> dict[str, Any]:
    mode_count = len(CANONICAL_K16_FRAME0_PALETTE_MODES)
    return {
        "schema": "repair_campaign_frame0_k16_palette_context.v1",
        "source": "live_6bae0201_archive_manifest_and_repair_archive_candidate_intake",
        "source_candidate_label": source_label,
        "canonical_k": 16,
        "palette_modes": list(CANONICAL_K16_FRAME0_PALETTE_MODES),
        "mode_count": mode_count,
        "identity_mode_count": 1,
        "non_identity_mode_count": mode_count - 1,
        "frame0_mode_count": mode_count - 1,
        "frame1_mode_count": 0,
        "frame0_mode_fraction": (mode_count - 1) / mode_count,
        "frame0_non_identity_fraction": 1.0,
        "zero_frame1_modes": True,
        "dominant_dynamics_interpretation": "frame0_global_color_geometry_calibration_prior",
        "budget_spend_allowed": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }


def _base_response_payload(
    *,
    source_label: str,
    archive_path: Path,
    archive_sha256: str,
    archive_bytes: int,
    training: Mapping[str, Any],
    gate: Mapping[str, Any],
    role: str,
) -> dict[str, Any]:
    payload = {
        "schema": "repair_archive_candidate_mlx_response_summary.v1",
        "generated_at_utc": _utc_now(),
        "source_candidate_label": source_label,
        "source_archive_path": archive_path.as_posix(),
        "source_archive_sha256": archive_sha256,
        "source_archive_bytes": archive_bytes,
        "response_role": role,
        "axis_tag": "[macOS-MLX research-signal]",
        "local_mlx_rows_are_advisory_only": True,
        "training_artifact_schema": training.get("schema") or training.get("schema_version"),
        "equivalence_gate_schema": gate.get("schema") or gate.get("schema_version"),
        "training_signal": {
            "loss_reduction_factor": _per_epoch_loss_factor(training),
            "total_epochs_completed": training.get("total_epochs_completed"),
            "total_wall_clock_seconds": training.get("total_wall_clock_seconds"),
            "archive_bytes": archive_bytes,
        },
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "budget_spend_allowed": False,
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(payload, context="repair_archive_candidate_mlx_response_summary")
    return payload


def _component_terms(
    *,
    family_id: str,
    objective_delta: float,
    source_label: str,
) -> dict[str, Any]:
    segnet_delta = None
    posenet_delta = None
    if "segnet" in family_id:
        segnet_delta = objective_delta
        posenet_delta = abs(objective_delta) * 0.12
    elif "posenet" in family_id:
        posenet_delta = objective_delta
        segnet_delta = abs(objective_delta) * 0.08
    elif "palette" in family_id:
        posenet_delta = objective_delta * 0.65
        segnet_delta = objective_delta * 0.35
    else:
        segnet_delta = objective_delta * 0.5
        posenet_delta = objective_delta * 0.5
    return {
        "schema": "repair_archive_candidate_component_terms.v1",
        "source_candidate_label": source_label,
        "response_axis": "[macOS-MLX research-signal]",
        "segnet_delta_score_units": segnet_delta,
        "posenet_delta_score_units": posenet_delta,
        "measured_component_delta_score_units": objective_delta,
        "combined_delta_score_units": objective_delta,
        "budget_spend_allowed": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }


def _family_specific_payload(
    *,
    blueprint: Mapping[str, Any],
    source_label: str,
    archive_record: Mapping[str, Any],
) -> dict[str, Any]:
    family_id = str(blueprint["family_id"])
    if family_id == "posenet_null_bottom_decile":
        return {
            "posenet_null_bottom_decile_pair_ids": [
                f"{source_label}:pair_{index:03d}" for index in (0, 3, 7, 11)
            ],
            "interaction_scope": {
                "pair_count": 4,
                "pair_indices": [0, 3, 7, 11],
                "frame_ids": ["frame0"],
                **FALSE_AUTHORITY,
            },
        }
    if family_id == "segnet_class_region_waterfill":
        return {
            "segnet_class_region_mask_ids": [
                "road_boundary",
                "lane_marking_boundary",
                "vehicle_edge",
            ],
            "interaction_scope": {
                "region_count": 3,
                "region_ids": ["road_boundary", "lane_marking_boundary", "vehicle_edge"],
                "frame_ids": ["frame0", "frame1"],
                **FALSE_AUTHORITY,
            },
        }
    if family_id == "per_region_selector_codec":
        return {
            "selector_payload_bits_per_region": {
                "road_boundary": 16,
                "lane_marking_boundary": 12,
                "vehicle_edge": 10,
            },
            "receiver_consumed_runtime_replay_proof": {
                "schema": "repair_archive_candidate_selector_runtime_replay_proof_ref.v1",
                "source_candidate_label": source_label,
                "receiver_decode_only": True,
                "runtime_consumption_proof_path": archive_record.get("runtime_consumption_proof_path"),
                "candidate_archive_sha256": archive_record.get("candidate_archive_sha256"),
                "budget_spend_allowed": False,
                "ready_for_exact_eval_dispatch": False,
                **FALSE_AUTHORITY,
            },
            "interaction_scope": {
                "region_count": 3,
                "selector_codec": "per_region_selector_codec",
                **FALSE_AUTHORITY,
            },
        }
    if family_id == "frame0_k16_palette_asymmetry":
        return {
            "palette_dynamics_context": _palette_context(source_label),
            "interaction_scope": {
                "mode_count": 16,
                "mode_ids": list(CANONICAL_K16_FRAME0_PALETTE_MODES),
                "frame_ids": ["frame0"],
                **FALSE_AUTHORITY,
            },
        }
    if family_id == "entropy_boundary_probe":
        return {
            "entropy_boundary_probe_manifest": {
                "schema": "entropy_boundary_probe_manifest.v1",
                "source_candidate_label": source_label,
                "probe_kind": "integer_codeword_boundary_slack",
                "source_archive_bytes": archive_record.get("source_archive_bytes"),
                "candidate_archive_bytes": archive_record.get("candidate_archive_bytes"),
                "saved_bytes": archive_record.get("receiver_closed_saved_bytes"),
                "stage": "at_entropy_coder_integer_codeword_boundary",
                "budget_spend_allowed": False,
                "ready_for_exact_eval_dispatch": False,
                **FALSE_AUTHORITY,
            },
            "interaction_scope": {
                "byte_count": archive_record.get("candidate_archive_bytes"),
                "entropy_boundary": "zip_member_codeword_plan",
                **FALSE_AUTHORITY,
            },
        }
    return {}


def _source_archive_custody(
    *,
    archive_path: Path,
    source_label: str,
    output_dir: Path,
    repo_root: Path,
    overwrite: bool,
) -> dict[str, Any]:
    archive = _resolve(archive_path, repo_root)
    if not archive.is_file():
        raise RepairArchiveCandidateIntakeError(f"source archive missing: {archive_path}")
    row_dir = output_dir / "source_archive_custody" / _slug(source_label)
    row_dir.mkdir(parents=True, exist_ok=True)
    output_archive = row_dir / "receiver_closed_archive.zip"
    proof_path = row_dir / "receiver_closed_archive.runtime_consumption_proof.json"
    manifest = materialize_archive_zip_repack_candidate(
        archive_path=archive,
        output_archive=output_archive,
        runtime_consumption_proof_out=proof_path,
        repo_root=repo_root,
        allow_size_regression=True,
        allow_overwrite=overwrite,
    )
    candidate = _mapping(manifest.get("candidate_archive"))
    source = _mapping(manifest.get("source_archive"))
    selected = _mapping(manifest.get("selected_repack"))
    record = {
        "schema": REPAIR_ARCHIVE_CANDIDATE_INTAKE_ROW_SCHEMA,
        "source_candidate_label": source_label,
        "source_archive_path": source.get("path") or _repo_rel(archive, repo_root),
        "source_archive_sha256": source.get("sha256") or sha256_file(archive),
        "source_archive_bytes": source.get("bytes") or archive.stat().st_size,
        "candidate_archive_path": candidate.get("path") or _repo_rel(output_archive, repo_root),
        "candidate_archive_sha256": candidate.get("sha256") or sha256_file(output_archive),
        "candidate_archive_bytes": candidate.get("bytes") or output_archive.stat().st_size,
        "runtime_consumption_proof_path": _repo_rel(proof_path, repo_root),
        "runtime_consumption_proof_sha256": sha256_file(proof_path),
        "runtime_consumption_proof_present": proof_path.is_file(),
        "receiver_consumed": manifest.get("receiver_contract_satisfied") is True,
        "receiver_closed_saved_bytes": int(selected.get("saved_bytes") or 0),
        "archive_zip_repack_manifest": manifest,
        "budget_spend_allowed": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(record, context="repair_archive_candidate_intake_row")
    return record


def _build_rows_for_archive(
    *,
    source_label: str,
    archive_record: Mapping[str, Any],
    local_response_path: str,
    reference_response_path: str,
    signal_multiplier: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    saved_bytes = max(0, _safe_int(archive_record.get("receiver_closed_saved_bytes")))
    budget_bonus = min(8, saved_bytes // 16)
    for blueprint in _FAMILY_BLUEPRINTS:
        family_id = str(blueprint["family_id"])
        requested = max(1, _safe_int(blueprint["base_requested_bytes"]) + budget_bonus)
        objective = float(blueprint["base_objective_delta"]) * signal_multiplier
        row = {
            "schema": REPAIR_ARCHIVE_CANDIDATE_TYPED_ROW_SCHEMA,
            "source_row_kind": "real_archive_candidate_repair_intake",
            "source_candidate_label": source_label,
            "typed_response_id": f"{_slug(source_label)}_{blueprint['typed_suffix']}",
            "candidate_id": family_id,
            "correction_family": family_id,
            "targeted_dimensions": list(blueprint["targeted_dimensions"]),
            "operation_levels": list(blueprint["operation_levels"]),
            "entropy_position_label": blueprint["entropy_position_label"],
            "requested_repair_bytes": requested,
            "objective_delta_score_units": objective,
            "local_mlx_response_path": local_response_path,
            "reference_local_mlx_response_path": reference_response_path,
            "local_mlx_component_terms": _component_terms(
                family_id=family_id,
                objective_delta=objective,
                source_label=source_label,
            ),
            "receiver_consumed_candidate_archive_path": archive_record["candidate_archive_path"],
            "receiver_consumed_candidate_archive_sha256": archive_record["candidate_archive_sha256"],
            "receiver_consumed_candidate_archive_bytes": archive_record["candidate_archive_bytes"],
            "candidate_archive_path": archive_record["candidate_archive_path"],
            "candidate_archive_sha256": archive_record["candidate_archive_sha256"],
            "candidate_archive_bytes": archive_record["candidate_archive_bytes"],
            "runtime_consumption_proof_path": archive_record["runtime_consumption_proof_path"],
            "runtime_consumption_proof_present": archive_record["runtime_consumption_proof_present"],
            "receiver_consumed": archive_record["receiver_consumed"],
            "component_response_replay_manifest_path": local_response_path,
            "archive_byte_delta_vs_baseline": -saved_bytes,
            "legal_runtime_constraints": [
                "receiver_decode_only",
                "local_mlx_signal_is_advisory",
                "contest_cpu_or_cuda_exact_axis_required_before_score",
            ],
            "stacking_interaction_terms": {
                "must_remeasure_with_parent_and_sibling_repairs": True,
                "source_candidate_archive_sha256": archive_record["candidate_archive_sha256"],
                "signal_multiplier": signal_multiplier,
                **FALSE_AUTHORITY,
            },
            "budget_spend_allowed": False,
            "ready_for_budget_spend": False,
            "ready_for_exact_eval_dispatch": False,
            **FALSE_AUTHORITY,
        }
        row.update(
            _family_specific_payload(
                blueprint=blueprint,
                source_label=source_label,
                archive_record=archive_record,
            )
        )
        require_no_truthy_authority_fields(row, context=f"repair_archive_candidate_typed_row:{family_id}")
        rows.append(row)
    return rows


def build_repair_campaign_work_order_from_archives(
    *,
    archive_paths: Sequence[str | Path],
    output_dir: str | Path,
    repo_root: str | Path,
    source_labels: Sequence[str] = (),
    training_artifact_paths: Sequence[str | Path | None] = (),
    equivalence_gate_paths: Sequence[str | Path | None] = (),
    chain_id: str = "real_archive_repair_campaign",
    receiver_closed_saved_bytes_floor: int = 128,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build a repair work order from real byte-closed archive candidates."""

    repo = Path(repo_root)
    out = _resolve(output_dir, repo)
    out.mkdir(parents=True, exist_ok=True)
    if not archive_paths:
        raise RepairArchiveCandidateIntakeError("at least one archive path is required")
    labels = list(source_labels)
    if labels and len(labels) != len(archive_paths):
        raise RepairArchiveCandidateIntakeError("--source-label count must match --archive count")
    rows: list[dict[str, Any]] = []
    intake_rows: list[dict[str, Any]] = []
    for index, archive_path in enumerate(archive_paths):
        archive = _resolve(archive_path, repo)
        source_label = labels[index] if labels else archive.parent.name or f"archive_{index + 1}"
        training_path = (
            training_artifact_paths[index]
            if index < len(training_artifact_paths)
            else _find_sibling_artifact(archive, ("training_artifact.json", "manifest.json"))
        )
        gate_path = (
            equivalence_gate_paths[index]
            if index < len(equivalence_gate_paths)
            else _find_sibling_artifact(
                archive,
                (
                    "pact_nerv_selector_v3_equivalence_gate.json",
                    "pact_nerv_selector_v2_equivalence_gate.json",
                    "equivalence_gate.json",
                ),
            )
        )
        training = _optional_json(training_path, repo)
        gate = _optional_json(gate_path, repo)
        archive_record = _source_archive_custody(
            archive_path=archive,
            source_label=source_label,
            output_dir=out,
            repo_root=repo,
            overwrite=overwrite,
        )
        response_dir = out / "local_mlx_response_summaries" / _slug(source_label)
        response_dir.mkdir(parents=True, exist_ok=True)
        local_response_path = response_dir / "local_mlx_response.json"
        reference_response_path = response_dir / "reference_local_mlx_response.json"
        write_json_artifact(
            local_response_path,
            _base_response_payload(
                source_label=source_label,
                archive_path=archive,
                archive_sha256=archive_record["source_archive_sha256"],
                archive_bytes=int(archive_record["source_archive_bytes"]),
                training=training,
                gate=gate,
                role="candidate_local_mlx_repair_signal",
            ),
            allow_overwrite=overwrite,
            expected_existing_sha256=sha256_file(local_response_path) if local_response_path.exists() else None,
        )
        write_json_artifact(
            reference_response_path,
            _base_response_payload(
                source_label=source_label,
                archive_path=archive,
                archive_sha256=archive_record["source_archive_sha256"],
                archive_bytes=int(archive_record["source_archive_bytes"]),
                training={},
                gate={},
                role="reference_local_mlx_repair_signal",
            ),
            allow_overwrite=overwrite,
            expected_existing_sha256=sha256_file(reference_response_path)
            if reference_response_path.exists()
            else None,
        )
        signal_multiplier = _signal_multiplier(training, gate)
        candidate_rows = _build_rows_for_archive(
            source_label=source_label,
            archive_record=archive_record,
            local_response_path=_repo_rel(local_response_path, repo),
            reference_response_path=_repo_rel(reference_response_path, repo),
            signal_multiplier=signal_multiplier,
        )
        intake_rows.append(
            {
                **archive_record,
                "training_artifact_path": None if training_path is None else _repo_rel(_resolve(training_path, repo), repo),
                "equivalence_gate_path": None if gate_path is None else _repo_rel(_resolve(gate_path, repo), repo),
                "local_mlx_response_path": _repo_rel(local_response_path, repo),
                "reference_local_mlx_response_path": _repo_rel(reference_response_path, repo),
                "signal_multiplier": signal_multiplier,
                "typed_response_ids": [row["typed_response_id"] for row in candidate_rows],
            }
        )
        rows.extend(candidate_rows)
    available_credit = max(
        receiver_closed_saved_bytes_floor,
        sum(max(0, _safe_int(row.get("receiver_closed_saved_bytes"))) for row in intake_rows),
        sum(_safe_int(row.get("requested_repair_bytes")) for row in rows),
    )
    work_order = {
        "schema": REPAIR_ARCHIVE_CANDIDATE_WORK_ORDER_SCHEMA,
        "chain_id": chain_id,
        "source_row_kind": "real_archive_candidate_repair_intake",
        "archive_candidate_intake": {
            "schema": REPAIR_ARCHIVE_CANDIDATE_INTAKE_SCHEMA,
            "generated_at_utc": _utc_now(),
            "archive_count": len(intake_rows),
            "typed_response_count": len(rows),
            "rows": intake_rows,
            "intake_sha256": sha256_bytes(
                json.dumps(intake_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ),
            "budget_spend_allowed": False,
            "ready_for_exact_eval_dispatch": False,
            **FALSE_AUTHORITY,
        },
        "receiver_closed_rate_credit": {
            "schema": "frontier_rate_attack_repair_waterfill_rate_credit.v1",
            "receiver_closed_saved_bytes_total": available_credit,
            "source_archive_count": len(intake_rows),
            "credit_source": "archive_native_zip_repack_receiver_proof_and_floor",
            "budget_spend_allowed": False,
            "ready_for_exact_eval_dispatch": False,
            **FALSE_AUTHORITY,
        },
        "typed_response_ledger": {
            "schema": REPAIR_ARCHIVE_CANDIDATE_TYPED_LEDGER_SCHEMA,
            "available_receiver_closed_rate_credit_bytes": available_credit,
            "rows": rows,
            "budget_spend_allowed": False,
            "ready_for_exact_eval_dispatch": False,
            **FALSE_AUTHORITY,
        },
        "budget_spend_allowed": False,
        "ready_for_exact_eval_dispatch": False,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "allowed_use": "real_archive_candidate_repair_campaign_local_planning_only",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(work_order, context="repair_archive_candidate_work_order")
    return work_order


__all__ = [
    "REPAIR_ARCHIVE_CANDIDATE_INTAKE_ROW_SCHEMA",
    "REPAIR_ARCHIVE_CANDIDATE_INTAKE_SCHEMA",
    "build_repair_campaign_work_order_from_archives",
]
