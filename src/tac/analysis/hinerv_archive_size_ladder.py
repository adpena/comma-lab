# SPDX-License-Identifier: MIT
"""Measured HiNeRV archive-size ladder.

This is rate evidence only. It exports actual receiver-shaped HiNeRV archives
for the local model-size configs, records archive ZIP bytes and hashes, and
keeps non-rate scorer authority closed until a scorer replay is attached.
"""

from __future__ import annotations

import json
import math
import zlib
from collections.abc import Iterable, Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

from comma_lab.storage_tiers import (
    DEFAULT_RESERVE_FREE_GB,
    StorageTierError,
    StorageTierSpec,
    bytes_from_gib,
    plan_experiment_storage,
    require_selected_storage,
)
from tac.analysis.nerv_decoder_weight_waterfill import (
    DEFAULT_ACTION_BITS,
    NERV_DECODER_WEIGHT_WATERFILL_SCHEMA,
    build_nerv_decoder_weight_waterfill_plan,
    load_saliency_json,
    load_state_npz_from_manifest,
)
from tac.analysis.nerv_modelsize_budget import (
    MODELSIZE_CONTROL_CONTRACT_REQUIRED_TRUE_FIELDS,
    MODELSIZE_RATE_AUTHORITY_SURFACE,
    analyze_hinerv_modelsize_candidate,
    build_hinerv_config_from_size_knobs,
)
from tac.analysis.nerv_modelsize_ladder import (
    SCORER_ONLY_OBJECTIVE_AUTHORITY,
    hi_nerv_modelsize_config_rows,
)
from tac.repo_io import sha256_file, write_json
from tac.substrates._shared.mlx_score_aware.modelsize_budget_plan import (
    CONTEST_BYTE_PRICE_SCORE,
)
from tac.substrates._shared.mlx_score_aware.nerv_byte_price_controller import (
    build_nerv_byte_price_plan,
)
from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY

HINERV_ARCHIVE_SIZE_LADDER_SCHEMA = "hinerv_archive_size_ladder.v1"
HINERV_MODELSIZE_RECEIVER_CONTRACT_SCHEMA = (
    "hinerv_archive_size_ladder_modelsize_receiver_contract.v1"
)
_NERV_MODELSIZE_CONTROL_CONTRACT_SCHEMA = "nerv_modelsize_control_contract.v1"
_HINERV_TARGET_MODELSIZE_CONTROL_SEMANTICS = (
    "local_receiver_visible_grid_search_nearest_target"
)
_HINERV_MANUAL_MODELSIZE_CONTROL_SEMANTICS = "manual_receiver_visible_architecture_knobs"
_HINERV_TARGET_MODELSIZE_CONSUMPTION = "nearest_local_param_count_target"
REQUIRED_ALLOCATOR_BINDINGS: tuple[str, ...] = (
    "adaptive_quantization_by_decoder_weight_group",
    "ablate_or_zero_groups_with_nonpositive_measured_value",
    "waterfill_group_bits_against_fixed_contest_byte_price",
    "inverse_steg_saliency_decoder_weight_binding",
    "packed_zero_and_entropy_coded_low_value_groups",
)
_NESTED_AUTHORITY_FIELDS: tuple[str, ...] = (
    "score_claim",
    "score_claim_valid",
    "frontier_score_claim",
    "promotion_eligible",
    "rank_or_kill_eligible",
    "production_hardened_claim",
    "ready_for_exact_eval_dispatch",
)
_TRUSTED_SCORE_ROW_SCHEMAS = frozenset(
    {
        "hinerv_full_video_mlx_score_rows.v1",
        "compact_renderer_full_video_mlx_replay.v1",
        "nerv_full_video_mlx_section_value_profile.v1",
        "nerv_section_value_profile.v1",
    }
)
_SCORE_PROVENANCE_HASH_KEYS: tuple[str, ...] = (
    "archive_sha256",
    "archive_zip_sha256",
    "candidate_archive_sha256",
    "source_archive_sha256",
    "runtime_tree_sha256",
    "scorer_profile_sha256",
    "source_report_sha256",
    "receiver_proof_sha256",
)


def build_hinerv_archive_size_ladder(
    *,
    output_dir: str | Path,
    repo_root: str | Path,
    num_pairs: int = 600,
    row_ids: Iterable[str] | None = None,
    hinerv_modelsize_budget: Mapping[str, Any] | None = None,
    decoder_codec: str = "int8_mixed",
    emit_receiver_proof: bool = False,
    retain_receiver_proof_output: bool = False,
    allow_local_output_dir: bool = False,
    storage_expected_bytes: int = 512 * 1024 * 1024,
    storage_reserve_free_gb: float = DEFAULT_RESERVE_FREE_GB,
    emit_decoder_weight_waterfill_plan: bool = False,
    decoder_weight_saliency_json: str | Path | None = None,
    decoder_weight_waterfill_action_bits: Sequence[int] = DEFAULT_ACTION_BITS,
) -> dict[str, Any]:
    """Export measured archive ZIP rows for the local HiNeRV size ladder."""

    root = Path(repo_root).expanduser().resolve(strict=False)
    out, storage_plan = _resolve_output_dir(
        output_dir=output_dir,
        repo_root=root,
        allow_local_output_dir=bool(allow_local_output_dir),
        storage_expected_bytes=int(storage_expected_bytes),
        storage_reserve_free_gb=float(storage_reserve_free_gb),
    )
    from tac.substrates.hi_nerv.archive_candidate import export_hi_nerv_mlx_archive

    decoder_weight_saliency_payload = (
        {}
        if decoder_weight_saliency_json is None
        else _read_json_if_exists(Path(decoder_weight_saliency_json).expanduser())
    )
    decoder_weight_saliency = (
        None
        if decoder_weight_saliency_json is None
        else load_saliency_json(decoder_weight_saliency_json)
    )
    decoder_weight_saliency_metadata = _decoder_weight_saliency_metadata(
        decoder_weight_saliency_payload,
        num_pairs=int(num_pairs),
    )
    selected = {str(row_id) for row_id in row_ids} if row_ids is not None else None
    specs = _archive_ladder_specs(
        num_pairs=int(num_pairs),
        selected=selected,
        hinerv_modelsize_budget=hinerv_modelsize_budget,
    )
    missing = sorted(selected - {str(spec["row_id"]) for spec in specs}) if selected else []
    rows = []
    blockers: list[str] = []
    if missing:
        blockers.append("hinerv_archive_size_ladder_requested_rows_missing")
    for spec in specs:
        row_id = str(spec["row_id"])
        cfg = spec["config"]
        row_decoder_codec = str(spec.get("decoder_codec") or decoder_codec)
        row_dir = out / row_id
        row_dir.mkdir(parents=True, exist_ok=True)
        model, archive_export_backend, backend_claim_blockers = _make_export_model(
            cfg,
            row_id=row_id,
        )
        archive_path, archive_sha256, archive_bytes = export_hi_nerv_mlx_archive(
            model,
            row_dir,
            repo_root=root,
            emit_archive_bound_candidate_package=bool(emit_receiver_proof),
            retain_receiver_proof_output=bool(retain_receiver_proof_output),
            decoder_codec=row_decoder_codec,
            source_backend=archive_export_backend,
        )
        proof_path = row_dir / "receiver_proof" / "hi_nerv_mlx_receiver_proof.json"
        proof = _read_json_if_exists(proof_path)
        state_npz_manifest_path = row_dir / "hi_nerv_mlx_exported_state_npz_manifest.json"
        modelsize_candidate = (
            dict(spec["modelsize_candidate"])
            if isinstance(spec.get("modelsize_candidate"), Mapping)
            else None
        )
        modelsize_receiver_contract = (
            dict(spec["modelsize_receiver_contract"])
            if isinstance(spec.get("modelsize_receiver_contract"), Mapping)
            else _legacy_hinerv_modelsize_receiver_contract(spec)
        )
        nominal_total_payload_bytes = (
            None
            if modelsize_candidate is None
            else _optional_int(modelsize_candidate.get("nominal_total_payload_bytes"))
        )
        waterfill_path = None
        waterfill_summary = None
        proof_ready = proof.get("runtime_consumption_proof_ready") is True
        proof_passed = proof.get("runtime_consumption_proof_passed") is True
        receiver_contract_satisfied = proof.get("receiver_contract_satisfied") is True
        receiver_closed = bool(
            proof_ready
            and proof_passed
            and receiver_contract_satisfied
            and not proof.get("blockers")
        )
        if emit_decoder_weight_waterfill_plan:
            proof_status = (
                "receiver_closed" if receiver_closed else "missing"
            )
            waterfill_path = row_dir / "decoder_weight_waterfill_plan.json"
            waterfill = build_nerv_decoder_weight_waterfill_plan(
                load_state_npz_from_manifest(state_npz_manifest_path),
                saliency_by_name=decoder_weight_saliency,
                family="hi_nerv",
                candidate_id=row_id,
                action_bits=decoder_weight_waterfill_action_bits,
                full_video_coverage=bool(
                    decoder_weight_saliency_metadata["full_video_coverage"]
                ),
                receiver_proof_status=proof_status,
                archive_sha256=archive_sha256,
            )
            write_json(waterfill_path, waterfill)
            waterfill_summary = _decoder_weight_waterfill_summary(waterfill)
        row_blockers = [
            "hinerv_archive_size_row_has_no_nonrate_score",
            "contest_cpu_cuda_exact_eval_not_executed",
            *backend_claim_blockers,
        ]
        if not emit_receiver_proof:
            row_blockers.append("receiver_proof_not_executed_for_archive_size_ladder")
        else:
            if not proof_ready:
                row_blockers.append("receiver_proof_not_ready_for_archive_size_ladder_row")
            if not proof_passed:
                row_blockers.append("runtime_consumption_proof_not_passed_for_archive_size_ladder_row")
            if not receiver_contract_satisfied:
                row_blockers.append("receiver_contract_not_satisfied_for_archive_size_ladder_row")
            if proof.get("blockers"):
                row_blockers.append("receiver_proof_blockers_present_for_archive_size_ladder_row")
        rows.append(
            {
                "family": "hi_nerv",
                "row_id": row_id,
                "modelsize_scale": float(spec["modelsize_scale"]),
                "modelsize_scale_source": modelsize_receiver_contract[
                    "modelsize_scale_source"
                ],
                "modelsize_scale_unit": modelsize_receiver_contract[
                    "modelsize_scale_unit"
                ],
                "modelsize_receiver_contract": modelsize_receiver_contract,
                "modelsize_candidate": modelsize_candidate,
                "config": _config_snapshot(cfg),
                "decoder_codec": row_decoder_codec,
                "archive_export_backend": archive_export_backend,
                "backend_claim_blockers": backend_claim_blockers,
                "num_parameters": int(model.num_parameters()),
                "archive_path": archive_path.as_posix(),
                "archive_sha256": archive_sha256,
                "archive_bytes": int(archive_bytes),
                "nominal_total_payload_bytes": nominal_total_payload_bytes,
                "measured_minus_nominal_bytes": (
                    None
                    if nominal_total_payload_bytes is None
                    else int(archive_bytes) - int(nominal_total_payload_bytes)
                ),
                "archive_rate_score_at_contest_price": float(
                    int(archive_bytes) * CONTEST_BYTE_PRICE_SCORE
                ),
                "spine_manifest_path": _path_if_exists(
                    row_dir / "hprc_representation_spine_hi_nerv_manifest.json"
                ),
                "state_npz_manifest_path": _path_if_exists(state_npz_manifest_path),
                "decoder_weight_waterfill_plan_path": _path_if_exists(waterfill_path)
                if waterfill_path is not None
                else None,
                "decoder_weight_waterfill_summary": waterfill_summary,
                "decoder_weight_saliency_full_video_coverage": bool(
                    decoder_weight_saliency_metadata["full_video_coverage"]
                ),
                "submission_dir": _path_if_exists(row_dir / "submission"),
                "receiver_proof_executed": bool(emit_receiver_proof),
                "receiver_proof_path": _path_if_exists(proof_path),
                "runtime_consumption_proof_ready": (
                    bool(proof_ready) if emit_receiver_proof else None
                ),
                "runtime_consumption_proof_passed": (
                    bool(proof_passed) if emit_receiver_proof else None
                ),
                "receiver_contract_satisfied": (
                    bool(receiver_contract_satisfied) if emit_receiver_proof else None
                ),
                "receiver_closed": receiver_closed if emit_receiver_proof else None,
                "required_allocator_bindings": list(REQUIRED_ALLOCATOR_BINDINGS),
                "blockers": row_blockers,
                **FALSE_AUTHORITY,
            }
        )
    rows.sort(key=lambda row: int(row["archive_bytes"]))
    blockers.extend(
        [
            "hinerv_archive_size_ladder_false_authority_no_nonrate_score",
            "contest_cpu_cuda_exact_eval_not_executed",
        ]
    )
    if not emit_receiver_proof:
        blockers.append("receiver_proof_not_executed_for_archive_size_ladder")
    blockers.extend(
        blocker
        for row in rows
        for blocker in row.get("backend_claim_blockers", ())
    )
    marginal_gates = _marginal_archive_gates(rows)
    section_value_rows = hinerv_modelsize_increment_section_value_rows(marginal_gates)
    report = {
        "schema": HINERV_ARCHIVE_SIZE_LADDER_SCHEMA,
        "authority": "false_authority_archive_size_ladder_no_score_claim",
        "family": "hi_nerv",
        "axis_tag": "[planning/control]",
        "repo_root": root.as_posix(),
        "output_dir": out.as_posix(),
        "storage_preflight": storage_plan.to_dict(),
        "local_output_explicitly_allowed": bool(allow_local_output_dir),
        "storage_expected_bytes": int(storage_expected_bytes),
        "storage_reserve_free_gb": float(storage_reserve_free_gb),
        "artifact_retention_policy": (
            "durable_evidence_on_selected_storage; archive rows preserve bytes, "
            "sha256, paths, false-authority flags, and blockers; local output "
            "requires explicit opt-in"
        ),
        "num_pairs": int(num_pairs),
        "decoder_codec": str(decoder_codec),
        "decoder_codec_policy": (
            "modelsize_budget_candidate_decoder_codec_overrides_top_level_default"
            if hinerv_modelsize_budget is not None
            else "top_level_decoder_codec_for_all_legacy_ladder_rows"
        ),
        "hinerv_modelsize_budget_schema": (
            None
            if hinerv_modelsize_budget is None
            else hinerv_modelsize_budget.get("schema")
        ),
        "modelsize_receiver_contract": _hinerv_archive_modelsize_receiver_contract(),
        "archive_export_backend_counts": _archive_export_backend_counts(rows),
        "emit_receiver_proof": bool(emit_receiver_proof),
        "emit_decoder_weight_waterfill_plan": bool(emit_decoder_weight_waterfill_plan),
        "decoder_weight_waterfill_schema": NERV_DECODER_WEIGHT_WATERFILL_SCHEMA,
        "decoder_weight_saliency_json": (
            None
            if decoder_weight_saliency_json is None
            else Path(decoder_weight_saliency_json).expanduser().as_posix()
        ),
        "decoder_weight_saliency_metadata": decoder_weight_saliency_metadata,
        "decoder_weight_waterfill_action_bits": [
            int(value) for value in decoder_weight_waterfill_action_bits
        ],
        "objective_authority": SCORER_ONLY_OBJECTIVE_AUTHORITY,
        "contest_byte_price_score_per_byte": CONTEST_BYTE_PRICE_SCORE,
        "selection_rule": (
            "measured archive bytes are only admissible after adaptive quantization, "
            "ablation, waterfilling, inverse-steg saliency, packed-zero, and entropy "
            "coding bind per decoder/latent group"
        ),
        "required_allocator_bindings": list(REQUIRED_ALLOCATOR_BINDINGS),
        "row_count": len(rows),
        "archive_rows": rows,
        "marginal_archive_gates": marginal_gates,
        "section_value_rows": section_value_rows,
        "missing_requested_row_ids": missing,
        "blockers": _ordered_unique(blockers),
        **FALSE_AUTHORITY,
    }
    report["byte_price_plan"] = build_nerv_byte_price_plan(report)
    return report


def build_hinerv_archive_size_ladder_from_checkpoint_exports(
    checkpoint_exports: Sequence[Mapping[str, Any]],
    *,
    report_path: str | Path | None = None,
    num_pairs: int | None = None,
) -> dict[str, Any]:
    """Wrap trained HiNeRV checkpoint exports as archive-size ladder rows.

    ``build_hinerv_archive_size_ladder`` measures fresh untrained size-ladder
    rows from model-size configs.  This bridge is for the opposite direction:
    preserve an already-trained/exported checkpoint package as the standard
    ``hinerv_archive_size_ladder.v1`` surface so waterfill, saliency replay,
    planner reingest, and byte-price consumers can use the trained state
    without rebuilding or pretending the row has scorer authority.
    """

    rows: list[dict[str, Any]] = []
    blockers: list[str] = [
        "hinerv_archive_size_ladder_false_authority_no_nonrate_score",
        "contest_cpu_cuda_exact_eval_not_executed",
    ]
    detected_num_pairs: int | None = int(num_pairs) if num_pairs is not None else None
    for index, export in enumerate(checkpoint_exports):
        if not isinstance(export, Mapping):
            raise TypeError(f"checkpoint_exports[{index}] must be a mapping")
        row = _checkpoint_export_ladder_row(export, row_index=index)
        rows.append(row)
        blockers.extend(row.get("backend_claim_blockers") or ())
        blockers.extend(
            blocker
            for blocker in row.get("blockers") or ()
            if blocker
            not in {
                "hinerv_archive_size_row_has_no_nonrate_score",
                "contest_cpu_cuda_exact_eval_not_executed",
            }
        )
        modelsize_candidate = row.get("modelsize_candidate")
        if detected_num_pairs is None and isinstance(modelsize_candidate, Mapping):
            maybe_pairs = _positive_int(modelsize_candidate.get("num_pairs"))
            if maybe_pairs is not None:
                detected_num_pairs = int(maybe_pairs)
    rows.sort(key=lambda row: int(row.get("archive_bytes") or 0))
    report = {
        "schema": HINERV_ARCHIVE_SIZE_LADDER_SCHEMA,
        "authority": "false_authority_trained_checkpoint_archive_ladder_no_score_claim",
        "family": "hi_nerv",
        "axis_tag": "[planning/control]",
        "repo_root": None,
        "output_dir": None,
        "report_path": (
            None
            if report_path is None
            else Path(report_path).expanduser().resolve(strict=False).as_posix()
        ),
        "storage_preflight": {
            "schema": "trained_checkpoint_archive_ladder_externalized_storage.v1",
            "selected_workload_root": "source_checkpoint_export_paths",
            "large_artifact_policy": (
                "no large artifact copied; existing checkpoint export archive, "
                "state npz, and receiver proof paths remain source custody"
            ),
            **FALSE_AUTHORITY,
        },
        "local_output_explicitly_allowed": False,
        "artifact_retention_policy": (
            "trained checkpoint export bridge is metadata-only; archive/state/"
            "receiver-proof bytes stay at their original durable paths"
        ),
        "num_pairs": int(detected_num_pairs or 0),
        "decoder_codec": None,
        "decoder_codec_policy": "trained_checkpoint_export_row_decoder_codec",
        "hinerv_modelsize_budget_schema": None,
        "modelsize_receiver_contract": _hinerv_archive_modelsize_receiver_contract(),
        "archive_export_backend_counts": {
            backend: sum(1 for row in rows if row.get("archive_export_backend") == backend)
            for backend in sorted(
                {
                    str(row.get("archive_export_backend") or "")
                    for row in rows
                    if row.get("archive_export_backend")
                }
            )
        },
        "emit_receiver_proof": True,
        "emit_decoder_weight_waterfill_plan": False,
        "decoder_weight_waterfill_schema": NERV_DECODER_WEIGHT_WATERFILL_SCHEMA,
        "decoder_weight_saliency_json": None,
        "decoder_weight_saliency_metadata": _decoder_weight_saliency_metadata(
            {},
            num_pairs=int(detected_num_pairs or 0),
        ),
        "decoder_weight_waterfill_action_bits": [
            int(value) for value in DEFAULT_ACTION_BITS
        ],
        "objective_authority": SCORER_ONLY_OBJECTIVE_AUTHORITY,
        "contest_byte_price_score_per_byte": CONTEST_BYTE_PRICE_SCORE,
        "selection_rule": (
            "trained checkpoint archive rows are receiver-closed byte evidence "
            "only; attach score replay and decoder-weight saliency before launch"
        ),
        "required_allocator_bindings": list(REQUIRED_ALLOCATOR_BINDINGS),
        "row_count": len(rows),
        "archive_rows": rows,
        "marginal_archive_gates": _marginal_archive_gates(rows),
        "section_value_rows": hinerv_modelsize_increment_section_value_rows(
            _marginal_archive_gates(rows)
        ),
        "missing_requested_row_ids": [],
        "blockers": _ordered_unique(blockers),
        **FALSE_AUTHORITY,
    }
    report["byte_price_plan"] = build_nerv_byte_price_plan(report)
    return report


def _archive_ladder_specs(
    *,
    num_pairs: int,
    selected: set[str] | None,
    hinerv_modelsize_budget: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    if hinerv_modelsize_budget is None:
        return [
            spec
            for spec in hi_nerv_modelsize_config_rows(num_pairs=int(num_pairs))
            if selected is None or str(spec["row_id"]) in selected
        ]
    if hinerv_modelsize_budget.get("schema") != "nerv_modelsize_budget.v1":
        raise ValueError(
            "hinerv_modelsize_budget schema must be nerv_modelsize_budget.v1; got "
            f"{hinerv_modelsize_budget.get('schema')!r}"
        )
    out: list[dict[str, Any]] = []
    for candidate in hinerv_modelsize_budget.get("selected_candidates") or ():
        if not isinstance(candidate, Mapping) or candidate.get("family") != "hi_nerv":
            continue
        row_id = str(candidate.get("candidate_id") or "").strip()
        if not row_id or (selected is not None and row_id not in selected):
            continue
        out.append(_hinerv_modelsize_candidate_spec(candidate, num_pairs=int(num_pairs)))
    return out


def _checkpoint_export_ladder_row(
    export: Mapping[str, Any],
    *,
    row_index: int,
) -> dict[str, Any]:
    blockers: list[str] = [
        "hinerv_archive_size_row_has_no_nonrate_score",
        "contest_cpu_cuda_exact_eval_not_executed",
    ]
    schema = str(export.get("schema") or "")
    if schema != "hinerv_checkpoint_archive_export.v1":
        blockers.append("hinerv_checkpoint_export_schema_unexpected")
    if export.get("family") not in {None, "hi_nerv"}:
        blockers.append("hinerv_checkpoint_export_family_unexpected")

    candidate_id = str(export.get("candidate_id") or f"checkpoint_export_{row_index:04d}")
    modelsize_candidate = (
        dict(export["modelsize_candidate"])
        if isinstance(export.get("modelsize_candidate"), Mapping)
        else None
    )
    num_pairs = 600
    modelsize_receiver_contract: dict[str, Any]
    config_snapshot: dict[str, Any] | None = None
    modelsize_scale = 0.0
    modelsize_scale_source = "missing_modelsize_candidate"
    modelsize_scale_unit = "unknown"
    if modelsize_candidate is None:
        blockers.append("hinerv_checkpoint_export_modelsize_candidate_missing")
        modelsize_receiver_contract = {
            "schema": HINERV_MODELSIZE_RECEIVER_CONTRACT_SCHEMA,
            "candidate_id": candidate_id,
            "source_contract_schema": None,
            "capacity_source": "missing",
            "modelsize_scale_source": modelsize_scale_source,
            "modelsize_scale_value": modelsize_scale,
            "modelsize_scale_unit": modelsize_scale_unit,
            "archive_bytes_authority": "archive_rows[].archive_bytes",
            "rate_authority_surface": MODELSIZE_RATE_AUTHORITY_SURFACE,
            "modelsize_mparams_is_official_upstream_flag": False,
            "modelsize_mparams_caps_archive_zip_bytes": False,
            **FALSE_AUTHORITY,
        }
    else:
        num_pairs = int(modelsize_candidate.get("num_pairs") or num_pairs)
        try:
            spec = _hinerv_modelsize_candidate_spec(
                modelsize_candidate,
                num_pairs=num_pairs,
            )
        except ValueError as exc:
            blockers.append(
                "hinerv_checkpoint_export_modelsize_candidate_invalid:"
                f"{type(exc).__name__}"
            )
            modelsize_receiver_contract = _legacy_hinerv_modelsize_receiver_contract(
                {
                    "row_id": candidate_id,
                    "modelsize_scale": float(
                        modelsize_candidate.get("modelsize_mparams") or 0.0
                    ),
                }
            )
        else:
            config_snapshot = _config_snapshot(spec["config"])
            modelsize_receiver_contract = dict(spec["modelsize_receiver_contract"])
            modelsize_scale = float(spec["modelsize_scale"])
            modelsize_scale_source = str(
                modelsize_receiver_contract["modelsize_scale_source"]
            )
            modelsize_scale_unit = str(
                modelsize_receiver_contract["modelsize_scale_unit"]
            )

    output_dir_raw = str(export.get("output_dir") or "").strip()
    output_dir = Path(output_dir_raw).expanduser().resolve(strict=False) if output_dir_raw else None
    archive_path = _resolve_export_path(
        export.get("archive_path"),
        output_dir=output_dir,
        fallback_name="archive.zip",
    )
    archive_bytes = _optional_int(export.get("archive_bytes")) or 0
    archive_sha256 = str(export.get("archive_sha256") or "").strip().lower()
    if not archive_path.is_file():
        blockers.append("hinerv_checkpoint_export_archive_missing")
    else:
        actual_bytes = archive_path.stat().st_size
        actual_sha = sha256_file(archive_path)
        if archive_bytes and archive_bytes != actual_bytes:
            blockers.append("hinerv_checkpoint_export_archive_bytes_mismatch")
        archive_bytes = actual_bytes
        if archive_sha256 and archive_sha256 != actual_sha:
            blockers.append("hinerv_checkpoint_export_archive_sha256_mismatch")
        archive_sha256 = actual_sha
    if len(archive_sha256) != 64:
        blockers.append("hinerv_checkpoint_export_archive_sha256_missing")

    state_npz_manifest_path = _resolve_export_path(
        export.get("state_npz_manifest_path"),
        output_dir=output_dir,
        fallback_name="hi_nerv_mlx_exported_state_npz_manifest.json",
    )
    _append_state_manifest_blockers(
        blockers,
        state_npz_manifest_path=state_npz_manifest_path,
    )

    proof_path = _resolve_export_path(
        export.get("receiver_proof_path"),
        output_dir=output_dir,
        fallback_name="receiver_proof/hi_nerv_mlx_receiver_proof.json",
    )
    proof_ready = export.get("receiver_proof_ready") is True
    proof_sha256 = str(export.get("receiver_proof_sha256") or "").strip().lower()
    runtime_consumption_proof_ready = False
    runtime_consumption_proof_passed = False
    receiver_contract_satisfied = False
    if not proof_ready:
        blockers.append("receiver_proof_not_ready_for_archive_size_ladder_row")
    if not proof_path.is_file():
        blockers.append("hinerv_checkpoint_export_receiver_proof_missing")
    else:
        actual_proof_sha = sha256_file(proof_path)
        if proof_sha256 and proof_sha256 != actual_proof_sha:
            blockers.append("hinerv_checkpoint_export_receiver_proof_sha256_mismatch")
        proof_sha256 = actual_proof_sha
        proof_payload = _read_json_if_exists(proof_path)
        proof_archive_sha = str(proof_payload.get("archive_sha256") or "").strip().lower()
        if proof_archive_sha != archive_sha256:
            blockers.append("hinerv_checkpoint_export_receiver_proof_archive_sha256_mismatch")
        runtime_consumption_proof_ready = proof_payload.get("runtime_consumption_proof_ready") is True
        runtime_consumption_proof_passed = (
            proof_payload.get("runtime_consumption_proof_passed") is True
        )
        receiver_contract_satisfied = (
            proof_payload.get("receiver_contract_satisfied") is True
        )
        if not runtime_consumption_proof_ready:
            blockers.append("hinerv_checkpoint_export_receiver_proof_runtime_not_ready")
        if not runtime_consumption_proof_passed:
            blockers.append("hinerv_checkpoint_export_receiver_proof_runtime_not_passed")
        if not receiver_contract_satisfied:
            blockers.append("hinerv_checkpoint_export_receiver_contract_not_satisfied")
        if proof_payload.get("blockers"):
            blockers.append("hinerv_checkpoint_export_receiver_proof_blockers_present")
    receiver_closed = bool(
        runtime_consumption_proof_ready
        and runtime_consumption_proof_passed
        and receiver_contract_satisfied
    )
    receiver_cache_quality = _checkpoint_export_receiver_cache_quality_summary(
        export,
        output_dir=output_dir,
        archive_sha256=archive_sha256,
    )
    receiver_cache_quality_blockers = list(
        receiver_cache_quality.get("row_blockers") or ()
    )
    blockers.extend(receiver_cache_quality_blockers)

    nominal_total_payload_bytes = (
        None
        if modelsize_candidate is None
        else _optional_int(modelsize_candidate.get("nominal_total_payload_bytes"))
    )
    row_output_dir = output_dir or archive_path.parent
    return {
        "family": "hi_nerv",
        "row_id": candidate_id,
        "source_schema": schema,
        "source_checkpoint_export_report_path": export.get("report_path"),
        "checkpoint_epoch": export.get("checkpoint_epoch"),
        "modelsize_scale": float(modelsize_scale),
        "modelsize_scale_source": modelsize_scale_source,
        "modelsize_scale_unit": modelsize_scale_unit,
        "modelsize_receiver_contract": modelsize_receiver_contract,
        "modelsize_candidate": modelsize_candidate,
        "config": config_snapshot,
        "decoder_codec": str(
            export.get("decoder_codec")
            or (modelsize_candidate or {}).get("decoder_codec")
            or "int8_mixed"
        ),
        "archive_export_backend": "trained_checkpoint_export_bridge",
        "backend_claim_blockers": [],
        "num_parameters": int(
            (modelsize_candidate or {}).get("total_trainable_params") or 0
        ),
        "archive_path": archive_path.as_posix(),
        "archive_sha256": archive_sha256 or None,
        "archive_bytes": int(archive_bytes),
        "nominal_total_payload_bytes": nominal_total_payload_bytes,
        "measured_minus_nominal_bytes": (
            None
            if nominal_total_payload_bytes is None
            else int(archive_bytes) - int(nominal_total_payload_bytes)
        ),
        "archive_rate_score_at_contest_price": float(
            int(archive_bytes) * CONTEST_BYTE_PRICE_SCORE
        ),
        "spine_manifest_path": _path_if_exists(
            row_output_dir / "hprc_representation_spine_hi_nerv_manifest.json"
        ),
        "state_npz_manifest_path": state_npz_manifest_path.as_posix(),
        "decoder_weight_waterfill_plan_path": None,
        "decoder_weight_waterfill_summary": None,
        "decoder_weight_saliency_full_video_coverage": False,
        "submission_dir": _path_if_exists(row_output_dir / "submission"),
        "receiver_proof_executed": proof_path.is_file(),
        "receiver_proof_path": proof_path.as_posix() if proof_path.is_file() else None,
        "receiver_proof_sha256": proof_sha256 or None,
        "runtime_consumption_proof_ready": bool(runtime_consumption_proof_ready),
        "runtime_consumption_proof_passed": bool(runtime_consumption_proof_passed),
        "receiver_contract_satisfied": bool(receiver_contract_satisfied),
        "receiver_closed": receiver_closed,
        "post_export_receiver_cache_quality": receiver_cache_quality.get("summary"),
        "receiver_cache_quality_report_path": receiver_cache_quality.get("report_path"),
        "receiver_cache_quality_report_sha256": receiver_cache_quality.get(
            "report_sha256"
        ),
        "receiver_cache_quality_gate_passed": bool(
            receiver_cache_quality.get("quality_gate_passed")
        ),
        "receiver_cache_quality_gate_verdict": receiver_cache_quality.get(
            "quality_gate_verdict"
        ),
        "receiver_cache_quality_blockers": receiver_cache_quality_blockers,
        "receiver_cache_quality_required_for_replay": True,
        "required_allocator_bindings": list(REQUIRED_ALLOCATOR_BINDINGS),
        "blockers": _ordered_unique(blockers),
        **FALSE_AUTHORITY,
    }


def _resolve_export_path(
    value: Any,
    *,
    output_dir: Path | None,
    fallback_name: str,
) -> Path:
    raw = str(value or "").strip()
    if raw:
        path = Path(raw).expanduser()
        if not path.is_absolute() and output_dir is not None:
            path = output_dir / path
        return path.resolve(strict=False)
    if output_dir is not None:
        return (output_dir / fallback_name).resolve(strict=False)
    return Path(fallback_name).expanduser().resolve(strict=False)


def _checkpoint_export_receiver_cache_quality_summary(
    export: Mapping[str, Any],
    *,
    output_dir: Path | None,
    archive_sha256: str,
) -> dict[str, Any]:
    """Return cache-quality summary plus row blockers for a checkpoint export."""

    row_blockers: list[str] = []
    summary_payload = _receiver_cache_quality_summary_from_export(export)
    report_path = _resolve_receiver_cache_quality_report_path(
        export,
        output_dir=output_dir,
        summary=summary_payload,
    )
    report_payload: Mapping[str, Any] | None = None
    report_sha256: str | None = None
    if report_path is not None and report_path.is_file():
        try:
            report_payload = _read_json_if_exists(report_path)
        except (OSError, json.JSONDecodeError):
            row_blockers.append(
                "hinerv_checkpoint_export_receiver_cache_quality_unreadable"
            )
        else:
            report_sha256 = sha256_file(report_path)
    elif summary_payload is None:
        row_blockers.append("hinerv_checkpoint_export_receiver_cache_quality_missing")

    source_payload = report_payload or summary_payload
    summary = (
        _normalize_receiver_cache_quality_summary(source_payload)
        if isinstance(source_payload, Mapping)
        else None
    )
    if summary is None:
        if summary_payload is not None or report_payload is not None:
            row_blockers.append(
                "hinerv_checkpoint_export_receiver_cache_quality_schema_unexpected"
            )
        return {
            "summary": None,
            "report_path": report_path.as_posix() if report_path is not None else None,
            "report_sha256": report_sha256,
            "quality_gate_passed": False,
            "quality_gate_verdict": None,
            "row_blockers": _ordered_unique(row_blockers),
        }

    report_path_raw = str(summary.get("report_path") or "").strip()
    if report_path_raw and report_path is None:
        report_path = Path(report_path_raw).expanduser().resolve(strict=False)
    summary_archive_sha = str(summary.get("archive_sha256") or "").strip().lower()
    expected_archive_sha = str(archive_sha256 or "").strip().lower()
    if (
        len(summary_archive_sha) == 64
        and len(expected_archive_sha) == 64
        and summary_archive_sha != expected_archive_sha
    ):
        row_blockers.append(
            "hinerv_checkpoint_export_receiver_cache_quality_archive_sha256_mismatch"
        )
    if summary.get("quality_gate_passed") is not True:
        row_blockers.append(
            "hinerv_checkpoint_export_receiver_cache_quality_gate_failed"
        )
    for blocker in summary.get("blockers") or ():
        normalized = str(blocker)
        if normalized and normalized != "hi_nerv_receiver_cache_quality_is_false_authority":
            row_blockers.append(normalized)
    return {
        "summary": summary,
        "report_path": report_path.as_posix() if report_path is not None else None,
        "report_sha256": report_sha256,
        "quality_gate_passed": bool(summary.get("quality_gate_passed")),
        "quality_gate_verdict": summary.get("quality_gate_verdict"),
        "row_blockers": _ordered_unique(row_blockers),
    }


def _receiver_cache_quality_summary_from_export(
    export: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    summary = export.get("post_export_receiver_cache_quality")
    if isinstance(summary, Mapping):
        return summary
    metadata = export.get("substrate_artifact_metadata")
    if isinstance(metadata, Mapping):
        summary = metadata.get("post_export_receiver_cache_quality")
        if isinstance(summary, Mapping):
            return summary
        score_training = metadata.get("score_aware_training")
        if isinstance(score_training, Mapping):
            summary = score_training.get("post_export_receiver_cache_quality")
            if isinstance(summary, Mapping):
                return summary
    return None


def _resolve_receiver_cache_quality_report_path(
    export: Mapping[str, Any],
    *,
    output_dir: Path | None,
    summary: Mapping[str, Any] | None,
) -> Path | None:
    candidates = (
        export.get("receiver_cache_quality_report_path"),
        export.get("post_export_receiver_cache_quality_report_path"),
        (summary or {}).get("report_path"),
    )
    for value in candidates:
        raw = str(value or "").strip()
        if not raw:
            continue
        path = Path(raw).expanduser()
        if not path.is_absolute() and output_dir is not None:
            path = output_dir / path
        return path.resolve(strict=False)
    if output_dir is None:
        return None
    fallback = (
        output_dir
        / "post_export_receiver_cache_quality"
        / "hi_nerv_receiver_cache_quality_report.json"
    )
    return fallback.resolve(strict=False)


def _normalize_receiver_cache_quality_summary(
    payload: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None
    schema = payload.get("schema")
    if schema == "hi_nerv_receiver_cache_quality_summary.v1":
        return {
            "schema": schema,
            "report_path": payload.get("report_path"),
            "archive_path": payload.get("archive_path"),
            "archive_sha256": payload.get("archive_sha256"),
            "candidate_cache_dir": payload.get("candidate_cache_dir"),
            "reference_cache_dir": payload.get("reference_cache_dir"),
            "quality_gate_path": payload.get("quality_gate_path"),
            "quality_gate_verdict": payload.get("quality_gate_verdict"),
            "quality_gate_passed": bool(payload.get("quality_gate_passed")),
            "candidate_segnet_last_rgb_stats": payload.get(
                "candidate_segnet_last_rgb_stats"
            ),
            "candidate_posenet_yuv6_pair_stats": payload.get(
                "candidate_posenet_yuv6_pair_stats"
            ),
            "distance_to_reference": payload.get("distance_to_reference"),
            "blockers": [str(v) for v in payload.get("blockers") or ()],
            **FALSE_AUTHORITY,
        }
    if schema != "hi_nerv_receiver_cache_quality_report.v1":
        return None
    gate = payload.get("quality_gate")
    gate_stats = gate.get("stats") if isinstance(gate, Mapping) else None
    return {
        "schema": "hi_nerv_receiver_cache_quality_summary.v1",
        "report_path": payload.get("report_path"),
        "archive_path": payload.get("archive_path"),
        "archive_sha256": payload.get("archive_sha256"),
        "candidate_cache_dir": payload.get("candidate_cache_dir"),
        "reference_cache_dir": payload.get("reference_cache_dir"),
        "quality_gate_path": payload.get("quality_gate_path"),
        "quality_gate_verdict": (
            gate.get("verdict") if isinstance(gate, Mapping) else None
        ),
        "quality_gate_passed": bool(payload.get("quality_gate_passed")),
        "candidate_segnet_last_rgb_stats": (
            gate_stats.get("candidate_segnet_last_rgb")
            if isinstance(gate_stats, Mapping)
            else None
        ),
        "candidate_posenet_yuv6_pair_stats": (
            gate_stats.get("candidate_posenet_yuv6_pair")
            if isinstance(gate_stats, Mapping)
            else None
        ),
        "distance_to_reference": (
            gate.get("distance_to_reference") if isinstance(gate, Mapping) else None
        ),
        "blockers": [str(v) for v in payload.get("blockers") or ()],
        **FALSE_AUTHORITY,
    }


def _append_state_manifest_blockers(
    blockers: list[str],
    *,
    state_npz_manifest_path: Path,
) -> None:
    if not state_npz_manifest_path.is_file():
        blockers.append("hinerv_checkpoint_export_state_npz_manifest_missing")
        return
    try:
        manifest = json.loads(state_npz_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        blockers.append("hinerv_checkpoint_export_state_npz_manifest_unreadable")
        return
    if manifest.get("schema") != "framework_agnostic_npz_bridge_manifest.v1":
        blockers.append("hinerv_checkpoint_export_state_npz_manifest_schema_unexpected")
    if manifest.get("consumption_recommended") is not True:
        blockers.append("hinerv_checkpoint_export_state_npz_manifest_not_recommended")
    artifact_raw = str(manifest.get("artifact_path") or "").strip()
    if not artifact_raw:
        blockers.append("hinerv_checkpoint_export_state_npz_manifest_artifact_missing")
        return
    artifact_path = Path(artifact_raw).expanduser()
    if not artifact_path.is_absolute():
        artifact_path = state_npz_manifest_path.parent / artifact_path
    artifact_path = artifact_path.resolve(strict=False)
    if not artifact_path.is_file():
        blockers.append("hinerv_checkpoint_export_state_npz_artifact_missing")
        return
    expected_sha = str(manifest.get("artifact_sha256") or "").strip().lower()
    if len(expected_sha) != 64:
        blockers.append("hinerv_checkpoint_export_state_npz_artifact_sha256_missing")
    elif sha256_file(artifact_path) != expected_sha:
        blockers.append("hinerv_checkpoint_export_state_npz_artifact_sha256_mismatch")


def _hinerv_modelsize_candidate_spec(
    candidate: Mapping[str, Any],
    *,
    num_pairs: int,
) -> dict[str, Any]:
    row_id = str(candidate.get("candidate_id") or "").strip()
    latent_dim = _positive_int(candidate.get("latent_dim")) or _positive_int(
        candidate.get("latent_dim_mid")
    )
    embed_dim = _positive_int(candidate.get("embed_dim"))
    decoder_channel = _positive_int(candidate.get("decoder_channel"))
    hard_byte_ceiling = _positive_int(candidate.get("hard_byte_ceiling"))
    decoder_codec = str(candidate.get("decoder_codec") or "").strip()
    if (
        not row_id
        or latent_dim is None
        or embed_dim is None
        or decoder_channel is None
        or hard_byte_ceiling is None
        or not decoder_codec
    ):
        raise ValueError(
            "hinerv modelsize candidate must include candidate_id, latent_dim, "
            "embed_dim, decoder_channel, decoder_codec, and hard_byte_ceiling"
        )
    _reject_true_nested_authority(candidate, row_id=row_id)
    cfg = build_hinerv_config_from_size_knobs(
        num_pairs=int(candidate.get("num_pairs") or num_pairs),
        latent_dim=int(latent_dim),
        embed_dim=int(embed_dim),
        decoder_channel=int(decoder_channel),
        use_hierarchical_feature_grid=bool(
            candidate.get("use_hierarchical_feature_grid", False)
        ),
        use_convnext_blocks=bool(candidate.get("use_convnext_blocks", False)),
        local_grid_levels=int(candidate.get("local_grid_levels") or 2),
        local_grid_channels=int(candidate.get("local_grid_channels") or 4),
        convnext_mlp_ratio=int(candidate.get("convnext_mlp_ratio") or 2),
        convnext_kernel_size=int(candidate.get("convnext_kernel_size") or 7),
        mid_injection_block_index=_int_or_default(
            candidate.get("mid_injection_block_index"), 1
        ),
        fine_injection_block_index=_int_or_default(
            candidate.get("fine_injection_block_index"), 4
        ),
    )
    _validate_candidate_config_snapshot(candidate, cfg, row_id=row_id)
    _validate_hinerv_candidate_id_source_controls(
        candidate,
        row_id=row_id,
        num_pairs=int(candidate.get("num_pairs") or num_pairs),
        latent_dim=int(latent_dim),
        embed_dim=int(embed_dim),
        decoder_channel=int(decoder_channel),
        decoder_codec=decoder_codec,
        hard_byte_ceiling=int(hard_byte_ceiling),
    )
    modelsize_receiver_contract = _validate_hinerv_modelsize_receiver_contract(
        candidate,
        row_id=row_id,
    )
    return {
        "row_id": row_id,
        "modelsize_scale": float(modelsize_receiver_contract["modelsize_scale_value"]),
        "modelsize_receiver_contract": modelsize_receiver_contract,
        "config": cfg,
        "decoder_codec": decoder_codec,
        "modelsize_candidate": dict(candidate),
    }


def _validate_hinerv_modelsize_receiver_contract(
    candidate: Mapping[str, Any],
    *,
    row_id: str,
) -> dict[str, Any]:
    contract = candidate.get("modelsize_control_contract")
    if not isinstance(contract, Mapping):
        raise ValueError(
            "hinerv modelsize candidate must include modelsize_control_contract "
            f"for {row_id}"
        )

    errors: list[str] = []
    target_present = (
        "target_modelsize_mparams" in candidate
        and candidate.get("target_modelsize_mparams") is not None
    )
    modelsize_mparams = _positive_float(candidate.get("modelsize_mparams"))
    target_modelsize_mparams = _positive_float(
        candidate.get("target_modelsize_mparams")
    )
    if modelsize_mparams is None:
        errors.append("modelsize_mparams_positive_required")
    if target_present and target_modelsize_mparams is None:
        errors.append("target_modelsize_mparams_positive_required")

    expected_capacity_source = (
        "local_hinerv_target_modelsize"
        if target_present
        else "manual_local_knobs"
    )
    if candidate.get("capacity_source") != expected_capacity_source:
        errors.append(
            "capacity_source_must_be_"
            f"{expected_capacity_source}_for_modelsize_scale_source"
        )

    expected_control_semantics = (
        _HINERV_TARGET_MODELSIZE_CONTROL_SEMANTICS
        if target_present
        else _HINERV_MANUAL_MODELSIZE_CONTROL_SEMANTICS
    )
    expected_target_consumption = (
        _HINERV_TARGET_MODELSIZE_CONSUMPTION if target_present else None
    )
    if contract.get("schema") != _NERV_MODELSIZE_CONTROL_CONTRACT_SCHEMA:
        errors.append("modelsize_control_contract_schema_invalid")
    if contract.get("family") != "hi_nerv":
        errors.append("modelsize_control_contract_family_must_be_hi_nerv")
    if contract.get("control_semantics") != expected_control_semantics:
        errors.append(
            "modelsize_control_contract_control_semantics_must_be_"
            f"{expected_control_semantics}"
        )
    if contract.get("shared_target_modelsize_mparams_consumed_as") != (
        expected_target_consumption
    ):
        errors.append(
            "target_modelsize_mparams_consumption_semantics_invalid"
        )
    if contract.get("modelsize_mparams_is_official_upstream_flag") is not False:
        errors.append("modelsize_mparams_must_not_be_official_upstream_flag")
    if contract.get("modelsize_mparams_caps_archive_zip_bytes") is not False:
        errors.append("modelsize_mparams_must_not_cap_archive_zip_bytes")
    if contract.get("rate_authority_surface") != MODELSIZE_RATE_AUTHORITY_SURFACE:
        errors.append("rate_authority_surface_must_be_measured_archive_bytes")
    if contract.get("hard_byte_ceiling_is_archive_budget_filter") is not True:
        errors.append("hard_byte_ceiling_must_be_archive_budget_filter_only")
    if candidate.get("requires_archive_byte_oracle") is not True:
        errors.append("requires_archive_byte_oracle_true_required")

    missing_true_fields = [
        field
        for field in MODELSIZE_CONTROL_CONTRACT_REQUIRED_TRUE_FIELDS
        if contract.get(field) is not True
    ]
    if missing_true_fields:
        errors.append(
            "missing_required_true_contract_fields:"
            + ",".join(sorted(missing_true_fields))
        )

    true_authority_fields = [
        field
        for field in _NESTED_AUTHORITY_FIELDS
        if _truthy_authority_value(contract.get(field))
    ]
    if true_authority_fields:
        errors.append(
            "modelsize_control_contract_true_authority_fields:"
            + ",".join(sorted(true_authority_fields))
        )

    if target_present and _nonnegative_float(
        candidate.get("modelsize_error_mparams")
    ) is None:
        errors.append("target_modelsize_mparams_requires_modelsize_error_mparams")

    if errors:
        raise ValueError(
            "hinerv modelsize candidate modelsize_control_contract invalid for "
            f"{row_id}: {errors}"
        )

    modelsize_scale_value = (
        float(target_modelsize_mparams)
        if target_present
        else float(modelsize_mparams)
    )
    modelsize_scale_source = (
        "target_modelsize_mparams" if target_present else "modelsize_mparams"
    )
    return {
        "schema": HINERV_MODELSIZE_RECEIVER_CONTRACT_SCHEMA,
        "candidate_id": row_id,
        "source_contract_schema": contract.get("schema"),
        "capacity_source": str(candidate.get("capacity_source")),
        "modelsize_scale_source": modelsize_scale_source,
        "modelsize_scale_value": modelsize_scale_value,
        "modelsize_scale_unit": "mparams",
        "modelsize_mparams": float(modelsize_mparams),
        "target_modelsize_mparams": (
            float(target_modelsize_mparams) if target_present else None
        ),
        "modelsize_error_mparams": (
            _nonnegative_float(candidate.get("modelsize_error_mparams"))
            if target_present
            else None
        ),
        "modelsize_scale_semantics": (
            "nearest local parameter-count target in millions of parameters"
            if target_present
            else "measured local trainable parameter count in millions"
        ),
        "target_modelsize_mparams_semantics": (
            "nearest local parameter-count target, not an official upstream "
            "--modelsize flag and not an archive-byte cap"
        ),
        "modelsize_mparams_semantics": (
            "local trainable parameter count in millions; never official "
            "upstream --modelsize for HiNeRV and never archive-byte authority"
        ),
        "hard_byte_ceiling_semantics": (
            "planner filter only; receiver-closed archive bytes remain measured "
            "authority"
        ),
        "archive_bytes_authority": "archive_rows[].archive_bytes",
        "rate_authority_surface": MODELSIZE_RATE_AUTHORITY_SURFACE,
        "modelsize_mparams_is_official_upstream_flag": False,
        "modelsize_mparams_caps_archive_zip_bytes": False,
        **FALSE_AUTHORITY,
    }


def _legacy_hinerv_modelsize_receiver_contract(
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": HINERV_MODELSIZE_RECEIVER_CONTRACT_SCHEMA,
        "candidate_id": str(spec.get("row_id") or ""),
        "source_contract_schema": None,
        "capacity_source": "legacy_hi_nerv_modelsize_config_rows",
        "modelsize_scale_source": "legacy_modelsize_scale",
        "modelsize_scale_value": float(spec.get("modelsize_scale") or 0.0),
        "modelsize_scale_unit": "relative_local_ladder_multiplier",
        "modelsize_mparams": None,
        "target_modelsize_mparams": None,
        "modelsize_error_mparams": None,
        "modelsize_scale_semantics": (
            "legacy local config ladder label, not modelsize_mparams, not "
            "official upstream --modelsize, and not an archive-byte cap"
        ),
        "target_modelsize_mparams_semantics": (
            "not provided for legacy local ladder rows"
        ),
        "modelsize_mparams_semantics": (
            "not provided for legacy local ladder rows; archive bytes are "
            "measured separately"
        ),
        "hard_byte_ceiling_semantics": "not provided for legacy local ladder rows",
        "archive_bytes_authority": "archive_rows[].archive_bytes",
        "rate_authority_surface": MODELSIZE_RATE_AUTHORITY_SURFACE,
        "modelsize_mparams_is_official_upstream_flag": False,
        "modelsize_mparams_caps_archive_zip_bytes": False,
        **FALSE_AUTHORITY,
    }


def _hinerv_archive_modelsize_receiver_contract() -> dict[str, Any]:
    return {
        "schema": HINERV_MODELSIZE_RECEIVER_CONTRACT_SCHEMA,
        "family": "hi_nerv",
        "budget_candidate_required_contract_schema": (
            _NERV_MODELSIZE_CONTROL_CONTRACT_SCHEMA
        ),
        "budget_candidate_required_true_fields": list(
            MODELSIZE_CONTROL_CONTRACT_REQUIRED_TRUE_FIELDS
        ),
        "modelsize_scale_field_rule": (
            "legacy rows use relative local ladder labels; modelsize-budget "
            "candidates use target_modelsize_mparams when present, otherwise "
            "local modelsize_mparams"
        ),
        "target_modelsize_mparams_semantics": (
            "nearest executable local HiNeRV parameter-count target in millions "
            "of parameters; not official upstream --modelsize and not an "
            "archive-byte cap"
        ),
        "modelsize_mparams_semantics": (
            "local trainable parameter count in millions; not official upstream "
            "--modelsize for HiNeRV and not archive-byte authority"
        ),
        "archive_bytes_authority": "archive_rows[].archive_bytes",
        "rate_authority_surface": MODELSIZE_RATE_AUTHORITY_SURFACE,
        "score_or_exact_promotion_rule": (
            "score, rank, promotion, and exact-dispatch authority remain false "
            "until receiver-closed archive bytes and scorer replay are attached"
        ),
        "modelsize_mparams_is_official_upstream_flag": False,
        "modelsize_mparams_caps_archive_zip_bytes": False,
        **FALSE_AUTHORITY,
    }


def _reject_true_nested_authority(candidate: Mapping[str, Any], *, row_id: str) -> None:
    flagged = [
        field
        for field in _NESTED_AUTHORITY_FIELDS
        if _truthy_authority_value(candidate.get(field))
    ]
    if flagged:
        raise ValueError(
            "hinerv modelsize candidate carries forbidden true authority flags "
            f"for {row_id}: {flagged}"
        )


def _truthy_authority_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized not in {"", "0", "false", "no", "none", "null"}
    return bool(value)


def _validate_hinerv_candidate_id_source_controls(
    candidate: Mapping[str, Any],
    *,
    row_id: str,
    num_pairs: int,
    latent_dim: int,
    embed_dim: int,
    decoder_channel: int,
    decoder_codec: str,
    hard_byte_ceiling: int,
) -> None:
    expected = analyze_hinerv_modelsize_candidate(
        hard_byte_ceiling=int(hard_byte_ceiling),
        num_pairs=int(num_pairs),
        latent_dim=int(latent_dim),
        embed_dim=int(embed_dim),
        decoder_channel=int(decoder_channel),
        decoder_codec=str(decoder_codec),
        use_hierarchical_feature_grid=bool(
            candidate.get("use_hierarchical_feature_grid", False)
        ),
        use_convnext_blocks=bool(candidate.get("use_convnext_blocks", False)),
        local_grid_levels=int(candidate.get("local_grid_levels") or 2),
        local_grid_channels=int(candidate.get("local_grid_channels") or 4),
        convnext_mlp_ratio=int(candidate.get("convnext_mlp_ratio") or 2),
        convnext_kernel_size=int(candidate.get("convnext_kernel_size") or 7),
        mid_injection_block_index=_int_or_default(
            candidate.get("mid_injection_block_index"), 1
        ),
        fine_injection_block_index=_int_or_default(
            candidate.get("fine_injection_block_index"), 4
        ),
    ).candidate_id
    target = _positive_float(candidate.get("target_modelsize_mparams"))
    if target is not None:
        expected = f"{expected}_tgtmp{_float_id_token(target)}"
    if row_id != expected:
        raise ValueError(
            "hinerv modelsize candidate_id source controls mismatch for "
            f"{row_id}: expected {expected}"
        )


def _validate_candidate_config_snapshot(
    candidate: Mapping[str, Any],
    cfg: Any,
    *,
    row_id: str,
) -> None:
    snapshot = _config_snapshot(cfg)
    expected_fields = (
        "latent_dim_coarse",
        "latent_dim_mid",
        "latent_dim_fine",
        "embed_dim",
        "decoder_channels",
        "mid_injection_block_index",
        "fine_injection_block_index",
        "use_hierarchical_feature_grid",
        "use_convnext_blocks",
        "local_grid_levels",
        "local_grid_channels",
        "convnext_mlp_ratio",
        "convnext_kernel_size",
        "num_pairs",
    )
    mismatches = []
    for field in expected_fields:
        if field not in candidate:
            continue
        actual = snapshot[field]
        expected = _normalizable_config_value(candidate[field])
        if actual != expected:
            mismatches.append(
                {
                    "field": field,
                    "candidate_value": expected,
                    "reconstructed_value": actual,
                }
            )
    if mismatches:
        raise ValueError(
            "hinerv modelsize candidate config mismatch for "
            f"{row_id}: {json.dumps(mismatches, sort_keys=True)}"
        )


def _normalizable_config_value(value: Any) -> Any:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [int(item) for item in value]
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def _positive_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0.0 else None


def _nonnegative_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0.0 else None


def _float_id_token(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def _decoder_weight_waterfill_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": report.get("schema"),
        "group_count": report.get("group_count"),
        "total_selected_byte_delta": report.get("total_selected_byte_delta"),
        "total_selected_delta_rate_score": report.get(
            "total_selected_delta_rate_score"
        ),
        "total_selected_delta_nonrate_score_proxy": report.get(
            "total_selected_delta_nonrate_score_proxy"
        ),
        "blockers": list(report.get("blockers") or ()),
        **FALSE_AUTHORITY,
    }


class _TorchExportableHinervModel:
    """PyTorch HiNeRV model exposing the MLX exporter state-dict protocol."""

    def __init__(self, cfg: Any, *, seed: int) -> None:
        import torch

        from tac.substrates.hi_nerv.architecture import HinervSubstrate

        self.cfg = cfg
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self._model = HinervSubstrate(cfg).eval()

    def export_state_dict(self) -> dict[str, Any]:
        return {
            name: tensor.detach().cpu().numpy().copy()
            for name, tensor in self._model.state_dict().items()
        }

    def num_parameters(self) -> int:
        return sum(int(param.numel()) for param in self._model.parameters())


def _make_export_model(cfg: Any, *, row_id: str) -> tuple[Any, str, list[str]]:
    try:
        from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX

        return HinervSubstrateMLX(cfg), "mlx", []
    except RuntimeError as exc:
        if "MLX is not available" not in str(exc):
            raise
    except ImportError:
        pass

    return (
        _TorchExportableHinervModel(
            cfg,
            seed=_stable_hinerv_seed(row_id),
        ),
        "pytorch_portable_fallback",
        ["archive_export_backend_not_mlx"],
    )


def _stable_hinerv_seed(row_id: str) -> int:
    return 2_026_060_2 + int(zlib.crc32(str(row_id).encode("utf-8")) % 1_000_000)


def _archive_export_backend_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        backend = str(row.get("archive_export_backend") or "unknown")
        counts[backend] = counts.get(backend, 0) + 1
    return counts


def _resolve_output_dir(
    *,
    output_dir: str | Path,
    repo_root: Path,
    allow_local_output_dir: bool,
    storage_expected_bytes: int,
    storage_reserve_free_gb: float,
) -> tuple[Path, Any]:
    output = Path(output_dir).expanduser()
    if not output.is_absolute():
        output = repo_root / output
    output = output.resolve(strict=False)
    if _looks_like_local_output(output) and not allow_local_output_dir:
        raise StorageTierError(
            "hinerv_archive_size_ladder_output_storage_preflight_failed: "
            "local_disk_tier_disabled; choose /Volumes/VertigoDataTier/pact or "
            "/Volumes/APDataStore/pact, or pass allow_local_output_dir=True"
        )
    plan = plan_experiment_storage(
        (
            StorageTierSpec(
                name="explicit_hinerv_archive_size_ladder_output",
                root=output,
                priority=0,
                reserve_free_bytes=bytes_from_gib(float(storage_reserve_free_gb)),
                allow_create=True,
                allow_local_disk=bool(allow_local_output_dir),
            ),
        ),
        workload_subdir=".",
        requested_bytes=int(storage_expected_bytes),
        min_free_bytes=0,
        create=True,
        probe_writable=True,
    )
    try:
        selected = require_selected_storage(plan)
    except StorageTierError as exc:
        raise StorageTierError(
            "hinerv_archive_size_ladder_output_storage_preflight_failed: "
            f"{exc}"
        ) from exc
    return selected, plan


def _looks_like_local_output(path: Path) -> bool:
    return not str(path.expanduser().resolve(strict=False)).startswith("/Volumes/")


def render_hinerv_archive_size_ladder_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact Markdown report."""

    lines = [
        "# HiNeRV archive-size ladder",
        "",
        f"Schema: `{report.get('schema')}`",
        f"Authority: `{report.get('authority')}`",
        f"Axis: `{report.get('axis_tag')}`",
        f"Decoder codec: `{report.get('decoder_codec')}`",
        f"Decoder codec policy: `{report.get('decoder_codec_policy')}`",
        f"Modelsize budget schema: `{report.get('hinerv_modelsize_budget_schema')}`",
        "Modelsize receiver contract: "
        f"`{(report.get('modelsize_receiver_contract') or {}).get('schema')}`",
        "",
        "| row | params | nominal bytes | archive bytes | measured-minus-nominal | rate score [planning/control] | proof ready |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report.get("archive_rows", ()):
        lines.append(
            "| {row_id} | {params} | {nominal} | {bytes} | {delta} | {rate:.6f} | {proof} |".format(
                row_id=row["row_id"],
                params=row["num_parameters"],
                nominal=row.get("nominal_total_payload_bytes"),
                bytes=row["archive_bytes"],
                delta=row.get("measured_minus_nominal_bytes"),
                rate=row["archive_rate_score_at_contest_price"],
                proof=row.get("receiver_closed"),
            )
        )
    lines.extend(["", "## Marginal Gates", ""])
    for gate in report.get("marginal_archive_gates", ()):
        lines.append(
            "- `{from_row_id}` -> `{to_row_id}` adds `{bytes_added}` B; "
            "requires non-rate drop >= `{required_nonrate_score_improvement}`".format(
                **gate
            )
        )
    lines.extend(["", "## Blockers", ""])
    for blocker in report.get("blockers", ()):
        lines.append(f"- `{blocker}`")
    lines.append("")
    return "\n".join(lines)


def _marginal_archive_gates(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    gates = []
    for low, high in pairwise(rows):
        bytes_added = int(high["archive_bytes"]) - int(low["archive_bytes"])
        if bytes_added <= 0:
            continue
        gates.append(
            {
                "from_row_id": str(low["row_id"]),
                "to_row_id": str(high["row_id"]),
                "from_archive_bytes": int(low["archive_bytes"]),
                "to_archive_bytes": int(high["archive_bytes"]),
                "bytes_added": int(bytes_added),
                "required_nonrate_score_improvement": float(
                    bytes_added * CONTEST_BYTE_PRICE_SCORE
                ),
                "contest_byte_price_score_per_byte": CONTEST_BYTE_PRICE_SCORE,
                "spend_rule": (
                    "spend_only_if_measured_nonrate_drop_exceeds_required_improvement"
                ),
            }
        )
    return gates


def hinerv_modelsize_increment_section_value_rows(
    gates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    out = []
    for gate in gates:
        from_row = str(gate["from_row_id"])
        to_row = str(gate["to_row_id"])
        out.append(
            {
                "row_id": f"hinerv_modelsize_increment_{from_row}_to_{to_row}",
                "section_id": f"hinerv_modelsize_increment:{from_row}->{to_row}",
                "family": "hi_nerv",
                "scope": "modelsize_increment",
                "row_kind": "new_residual_or_sidecar",
                "from_row_id": from_row,
                "to_row_id": to_row,
                "bytes": int(gate["bytes_added"]),
                "byte_delta": int(gate["bytes_added"]),
                "delta_nonrate_score": None,
                "required_nonrate_score_improvement": float(
                    gate["required_nonrate_score_improvement"]
                ),
                "axis_tag": "[planning/control]",
                "receiver_proof_status": "missing",
                "full_video_coverage": False,
                "blockers": [
                    "hinerv_modelsize_increment_has_no_measured_nonrate_score",
                    "hinerv_modelsize_increment_needs_decoder_weight_saliency_replay",
                    "hinerv_modelsize_increment_needs_byte_closed_receiver_proof",
                ],
                **FALSE_AUTHORITY,
            }
        )
    return out


def attach_hinerv_archive_ladder_score_rows(
    ladder_report: Mapping[str, Any],
    score_artifacts_or_rows: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    *,
    score_source_path: str | Path | None = None,
    require_full_video: bool = True,
) -> dict[str, Any]:
    """Attach measured non-rate scorer rows to a HiNeRV archive-size ladder.

    The archive ladder is byte authority only. This helper makes the next step
    explicit: a measured scorer artifact provides per-row non-rate scores, then
    modelsize increments become byte-priced section-value rows. It never creates
    score authority; MLX/advisory rows remain blocked from promotion by the byte
    price controller until exact replay lands.
    """

    report = dict(ladder_report)
    if report.get("schema") != HINERV_ARCHIVE_SIZE_LADDER_SCHEMA:
        raise ValueError(
            "expected hinerv archive-size ladder schema; got "
            f"{report.get('schema')!r}"
        )
    score_rows = _score_rows_from_artifacts(score_artifacts_or_rows)
    score_by_row = _score_rows_by_id(score_rows)
    updated_rows = []
    rows_with_score = 0
    rows_with_full_video = 0
    rows_with_trusted_score = 0
    for row in report.get("archive_rows", ()):
        updated = dict(row)
        row_id = str(updated.get("row_id") or "")
        score = score_by_row.get(row_id)
        blockers = [
            str(blocker) for blocker in updated.get("blockers") or () if blocker
        ]
        if score is None:
            blockers.append("hinerv_archive_size_row_measured_score_missing")
        else:
            rows_with_score += 1
            updated.update(
                {
                    "nonrate_score": score["nonrate_score"],
                    "avg_segnet_dist": score.get("avg_segnet_dist"),
                    "avg_posenet_dist": score.get("avg_posenet_dist"),
                    "measured_score_axis_tag": score["axis_tag"],
                    "measured_score_full_video_coverage": score["full_video_coverage"],
                    "measured_score_custody_trusted": score[
                        "trusted_score_custody"
                    ],
                    "measured_score_trust_blockers": list(
                        score.get("trust_blockers") or []
                    ),
                    "measured_score_source_row_id": score["source_row_id"],
                    "measured_score_source_schema": score.get("source_schema"),
                    "measured_score_source_path": (
                        None
                        if score_source_path is None
                        else Path(score_source_path).expanduser().as_posix()
                    ),
                }
            )
            if score["trusted_score_custody"]:
                rows_with_trusted_score += 1
            else:
                blockers.append("hinerv_archive_size_row_measured_score_untrusted")
            if score["full_video_coverage"] and score["trusted_score_custody"]:
                rows_with_full_video += 1
                blockers = [
                    blocker
                    for blocker in blockers
                    if blocker != "hinerv_archive_size_row_has_no_nonrate_score"
                ]
            else:
                blockers.append("hinerv_archive_size_row_measured_score_not_full_video")
        updated["blockers"] = _ordered_unique(blockers)
        updated_rows.append(updated)

    marginal_gates = _marginal_archive_gates(updated_rows)
    section_rows = _measured_hinerv_increment_section_value_rows(
        marginal_gates,
        archive_rows=updated_rows,
        require_full_video=bool(require_full_video),
    )
    blockers = [
        str(blocker) for blocker in report.get("blockers") or () if blocker
    ]
    if rows_with_trusted_score:
        blockers = [
            blocker
            for blocker in blockers
            if blocker != "hinerv_archive_size_ladder_false_authority_no_nonrate_score"
        ]
    else:
        blockers.append("hinerv_archive_size_ladder_measured_scores_missing")
    if rows_with_score and rows_with_trusted_score != rows_with_score:
        blockers.append("hinerv_archive_size_ladder_measured_scores_untrusted")
    if require_full_video and rows_with_full_video != len(updated_rows):
        blockers.append("hinerv_archive_size_ladder_full_video_scores_incomplete")
    report.update(
        {
            "archive_rows": updated_rows,
            "marginal_archive_gates": marginal_gates,
            "section_value_rows": section_rows,
            "score_attachment": {
                "schema": "hinerv_archive_size_ladder_score_attachment.v1",
                "source_path": (
                    None
                    if score_source_path is None
                    else Path(score_source_path).expanduser().as_posix()
                ),
                "input_score_row_count": len(score_rows),
                "matched_archive_row_count": rows_with_score,
                "matched_full_video_row_count": rows_with_full_video,
                "trusted_score_row_count": rows_with_trusted_score,
                "require_full_video": bool(require_full_video),
                **FALSE_AUTHORITY,
            },
            "blockers": _ordered_unique(blockers),
            **FALSE_AUTHORITY,
        }
    )
    report["byte_price_plan"] = build_nerv_byte_price_plan(report)
    return report


def _measured_hinerv_increment_section_value_rows(
    gates: Sequence[Mapping[str, Any]],
    *,
    archive_rows: Sequence[Mapping[str, Any]],
    require_full_video: bool,
) -> list[dict[str, Any]]:
    by_row = {str(row.get("row_id")): row for row in archive_rows}
    out = []
    for gate in gates:
        from_row_id = str(gate["from_row_id"])
        to_row_id = str(gate["to_row_id"])
        from_row = by_row.get(from_row_id, {})
        to_row = by_row.get(to_row_id, {})
        from_nonrate = _finite_float(from_row.get("nonrate_score"))
        to_nonrate = _finite_float(to_row.get("nonrate_score"))
        from_full = from_row.get("measured_score_full_video_coverage") is True
        to_full = to_row.get("measured_score_full_video_coverage") is True
        from_trusted = from_row.get("measured_score_custody_trusted") is True
        to_trusted = to_row.get("measured_score_custody_trusted") is True
        proof_ready = (
            from_row.get("runtime_consumption_proof_ready") is True
            and to_row.get("runtime_consumption_proof_ready") is True
        )
        proof_passed = (
            from_row.get("runtime_consumption_proof_passed") is True
            and to_row.get("runtime_consumption_proof_passed") is True
        )
        receiver_contract_satisfied = (
            from_row.get("receiver_contract_satisfied") is True
            and to_row.get("receiver_contract_satisfied") is True
        )
        receiver_closed = bool(
            proof_ready and proof_passed and receiver_contract_satisfied
        )
        from_cache_quality = from_row.get("receiver_cache_quality_gate_passed") is True
        to_cache_quality = to_row.get("receiver_cache_quality_gate_passed") is True
        blockers: list[str] = []
        if from_nonrate is None or to_nonrate is None:
            blockers.append("hinerv_modelsize_increment_measured_nonrate_missing")
        if require_full_video and not (from_full and to_full):
            blockers.append("hinerv_modelsize_increment_full_video_score_missing")
        if not (from_trusted and to_trusted):
            blockers.append("hinerv_modelsize_increment_measured_score_untrusted")
        if not receiver_closed:
            blockers.append("hinerv_modelsize_increment_receiver_proof_missing")
        if not (from_cache_quality and to_cache_quality):
            blockers.append(
                "hinerv_modelsize_increment_receiver_cache_quality_missing_or_failed"
            )
        delta_nonrate = (
            None
            if from_nonrate is None or to_nonrate is None
            else float(to_nonrate) - float(from_nonrate)
        )
        out.append(
            {
                "row_id": f"hinerv_modelsize_increment_{from_row_id}_to_{to_row_id}",
                "section_id": f"hinerv_modelsize_increment:{from_row_id}->{to_row_id}",
                "family": "hi_nerv",
                "scope": "modelsize_increment",
                "row_kind": "new_residual_or_sidecar",
                "from_row_id": from_row_id,
                "to_row_id": to_row_id,
                "from_nonrate_score": from_nonrate,
                "to_nonrate_score": to_nonrate,
                "delta_nonrate_score": delta_nonrate,
                "required_nonrate_score_improvement": float(
                    gate["required_nonrate_score_improvement"]
                ),
                "bytes": int(gate["bytes_added"]),
                "byte_delta": int(gate["bytes_added"]),
                "archive_sha256": to_row.get("archive_sha256"),
                "axis_tag": _score_axis_for_increment(from_row, to_row),
                "receiver_proof_status": (
                    "receiver_closed" if receiver_closed else "missing"
                ),
                "runtime_consumption_proof_ready": bool(proof_ready),
                "runtime_consumption_proof_passed": bool(proof_passed),
                "receiver_contract_satisfied": bool(receiver_contract_satisfied),
                "receiver_closed": receiver_closed,
                "receiver_cache_quality_gate_passed": bool(
                    from_cache_quality and to_cache_quality
                ),
                "from_receiver_cache_quality_gate_passed": bool(from_cache_quality),
                "to_receiver_cache_quality_gate_passed": bool(to_cache_quality),
                "full_video_coverage": bool(from_full and to_full),
                "measured_score_custody_trusted": bool(from_trusted and to_trusted),
                "blockers": blockers,
                **FALSE_AUTHORITY,
            }
        )
    return out


def _score_axis_for_increment(
    from_row: Mapping[str, Any],
    to_row: Mapping[str, Any],
) -> str:
    for row in (to_row, from_row):
        axis = row.get("measured_score_axis_tag")
        if axis:
            return str(axis)
    return "[macOS-MLX research-signal]"


def _score_rows_from_artifacts(
    artifacts_or_rows: Sequence[Mapping[str, Any]] | Mapping[str, Any],
) -> list[dict[str, Any]]:
    if isinstance(artifacts_or_rows, Mapping):
        return _score_rows_from_one_artifact(artifacts_or_rows)
    rows: list[dict[str, Any]] = []
    for item in artifacts_or_rows:
        if isinstance(item, Mapping):
            rows.extend(_score_rows_from_one_artifact(item))
    return rows


def _score_rows_from_one_artifact(artifact: Mapping[str, Any]) -> list[dict[str, Any]]:
    for key in (
        "archive_rows",
        "modelsize_score_rows",
        "score_rows",
        "rows",
        "section_value_rows",
        "modelsize_budget_rows",
        "normalized_rows",
    ):
        rows = artifact.get(key)
        if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
            out = []
            for row in rows:
                if isinstance(row, Mapping):
                    normalized = _normalize_score_row(
                        row,
                        source_schema=artifact.get("schema"),
                        artifact_axis=artifact.get("axis_tag"),
                    )
                    if normalized is not None:
                        out.append(normalized)
            return out
    normalized = _normalize_score_row(
        artifact,
        source_schema=artifact.get("schema"),
        artifact_axis=artifact.get("axis_tag"),
    )
    return [] if normalized is None else [normalized]


def _normalize_score_row(
    row: Mapping[str, Any],
    *,
    source_schema: Any,
    artifact_axis: Any,
) -> dict[str, Any] | None:
    row_id = _first_string(
        row,
        (
            "row_id",
            "modelsize_row_id",
            "candidate_id",
            "id",
            "to_row_id",
            "source_row_id",
        ),
    )
    if not row_id:
        return None
    d_seg = _finite_float_from_keys(
        row,
        ("avg_segnet_dist", "d_seg", "segnet_dist", "segnet_distance"),
    )
    d_pose = _finite_float_from_keys(
        row,
        ("avg_posenet_dist", "d_pose", "posenet_dist", "posenet_distance"),
    )
    nonrate = _finite_float_from_keys(
        row,
        (
            "nonrate_score",
            "nonrate_score_value",
            "nonrate_score_advisory",
            "score_linf_without_rate",
        ),
    )
    if nonrate is None and d_seg is not None and d_pose is not None:
        nonrate = float(100.0 * d_seg + math.sqrt(10.0 * d_pose))
    if nonrate is None:
        return None
    trust_blockers = _score_row_trust_blockers(row, source_schema=source_schema)
    return {
        "source_row_id": str(row_id),
        "nonrate_score": float(nonrate),
        "avg_segnet_dist": d_seg,
        "avg_posenet_dist": d_pose,
        "full_video_coverage": _score_row_full_video(row),
        "trusted_score_custody": not trust_blockers,
        "trust_blockers": trust_blockers,
        "axis_tag": str(
            row.get("axis_tag")
            or row.get("score_axis")
            or row.get("evidence_axis")
            or artifact_axis
            or "[macOS-MLX research-signal]"
        ),
        "source_schema": source_schema,
    }


def _score_row_trust_blockers(
    row: Mapping[str, Any],
    *,
    source_schema: Any,
) -> list[str]:
    blockers: list[str] = []
    if str(source_schema or "") not in _TRUSTED_SCORE_ROW_SCHEMAS:
        blockers.append("score_row_source_schema_not_allowlisted")
    if not any(_nonempty_string(row.get(key)) for key in _SCORE_PROVENANCE_HASH_KEYS):
        blockers.append("score_row_provenance_hash_missing")
    return blockers


def _nonempty_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _score_rows_by_id(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    out = {}
    for row in rows:
        row_id = str(row.get("source_row_id") or "")
        if row_id:
            out[row_id] = row
    return out


def _score_row_full_video(row: Mapping[str, Any]) -> bool:
    explicit = row.get("full_video_coverage")
    if isinstance(explicit, bool):
        return explicit
    full_video = row.get("full_video")
    if isinstance(full_video, bool):
        return full_video
    if isinstance(full_video, str):
        return full_video.lower() in {"executed", "true", "full", "full_video"}
    for key in (
        "num_pairs",
        "n_pairs",
        "n_samples",
        "num_samples",
        "scored_pairs",
        "evaluated_pairs",
    ):
        value = _positive_int(row.get(key))
        if value is not None:
            return value >= 600
    return False


def _first_string(row: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value):
            return str(value)
    return None


def _finite_float_from_keys(row: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        value = _finite_float(row.get(key))
        if value is not None:
            return value
    return None


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _positive_int(value: Any) -> int | None:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out if out > 0 else None


def _int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _config_snapshot(cfg: Any) -> dict[str, Any]:
    return {
        "latent_dim_coarse": int(cfg.latent_dim_coarse),
        "latent_dim_mid": int(cfg.latent_dim_mid),
        "latent_dim_fine": int(cfg.latent_dim_fine),
        "embed_dim": int(cfg.embed_dim),
        "decoder_channels": [int(value) for value in cfg.decoder_channels],
        "num_upsample_blocks": int(cfg.num_upsample_blocks),
        "mid_injection_block_index": int(cfg.mid_injection_block_index),
        "fine_injection_block_index": int(cfg.fine_injection_block_index),
        "use_hierarchical_feature_grid": bool(cfg.use_hierarchical_feature_grid),
        "use_convnext_blocks": bool(cfg.use_convnext_blocks),
        "local_grid_levels": int(cfg.local_grid_levels),
        "local_grid_channels": int(cfg.local_grid_channels),
        "convnext_mlp_ratio": int(cfg.convnext_mlp_ratio),
        "convnext_kernel_size": int(cfg.convnext_kernel_size),
        "num_pairs": int(cfg.num_pairs),
        "output_height": int(cfg.output_height),
        "output_width": int(cfg.output_width),
    }


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _decoder_weight_saliency_metadata(
    payload: Mapping[str, Any],
    *,
    num_pairs: int,
) -> dict[str, Any]:
    pair_schedule = payload.get("pair_schedule")
    schedule_full = False
    if isinstance(pair_schedule, Mapping):
        schedule_full = bool(
            int(pair_schedule.get("max_pairs") or 0) >= int(num_pairs)
            and int(pair_schedule.get("start_pair") or 0) == 0
            and int(pair_schedule.get("pair_stride") or 0) == 1
        )
    declared_full = payload.get("full_video_coverage") is True
    # Declared booleans are provenance, not authority.  A decoder-weight
    # saliency map is full-video only when its schedule proves the exact
    # deterministic reduction over the requested video.
    full_video_coverage = bool(schedule_full)
    blockers = [str(blocker) for blocker in payload.get("blockers") or () if str(blocker)]
    coverage_blockers = []
    if payload and not full_video_coverage:
        coverage_blockers.append("decoder_weight_saliency_full_video_coverage_missing")
    if payload and declared_full and not schedule_full:
        coverage_blockers.append("decoder_weight_saliency_declared_full_without_schedule_proof")
    return {
        "schema": "hinerv_archive_size_ladder_decoder_weight_saliency_metadata.v1",
        "source_schema": payload.get("schema"),
        "provided": bool(payload),
        "declared_full_video_coverage": bool(declared_full),
        "schedule_full_video_coverage": bool(schedule_full),
        "full_video_coverage": bool(full_video_coverage),
        "num_pairs": int(num_pairs),
        "source_blockers": blockers,
        "coverage_blockers": coverage_blockers,
        **FALSE_AUTHORITY,
    }


def _path_if_exists(path: Path) -> str | None:
    return path.as_posix() if path.exists() else None


def _ordered_unique(items: Iterable[str]) -> list[str]:
    out = []
    seen = set()
    for item in items:
        text = str(item)
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


__all__ = [
    "HINERV_ARCHIVE_SIZE_LADDER_SCHEMA",
    "HINERV_MODELSIZE_RECEIVER_CONTRACT_SCHEMA",
    "REQUIRED_ALLOCATOR_BINDINGS",
    "attach_hinerv_archive_ladder_score_rows",
    "build_hinerv_archive_size_ladder",
    "build_hinerv_archive_size_ladder_from_checkpoint_exports",
    "hinerv_modelsize_increment_section_value_rows",
    "render_hinerv_archive_size_ladder_markdown",
]
