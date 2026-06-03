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
    analyze_hinerv_modelsize_candidate,
    build_hinerv_config_from_size_knobs,
)
from tac.analysis.nerv_modelsize_ladder import (
    SCORER_ONLY_OBJECTIVE_AUTHORITY,
    hi_nerv_modelsize_config_rows,
)
from tac.repo_io import write_json
from tac.substrates._shared.mlx_score_aware.modelsize_budget_plan import (
    CONTEST_BYTE_PRICE_SCORE,
)
from tac.substrates._shared.mlx_score_aware.nerv_byte_price_controller import (
    build_nerv_byte_price_plan,
)
from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY

HINERV_ARCHIVE_SIZE_LADDER_SCHEMA = "hinerv_archive_size_ladder.v1"
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

    decoder_weight_saliency = (
        None
        if decoder_weight_saliency_json is None
        else load_saliency_json(decoder_weight_saliency_json)
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
        nominal_total_payload_bytes = (
            None
            if modelsize_candidate is None
            else _optional_int(modelsize_candidate.get("nominal_total_payload_bytes"))
        )
        waterfill_path = None
        waterfill_summary = None
        if emit_decoder_weight_waterfill_plan:
            proof_status = (
                "runtime_consumption_proof_ready"
                if proof.get("runtime_consumption_proof_ready") is True
                else "missing"
            )
            waterfill_path = row_dir / "decoder_weight_waterfill_plan.json"
            waterfill = build_nerv_decoder_weight_waterfill_plan(
                load_state_npz_from_manifest(state_npz_manifest_path),
                saliency_by_name=decoder_weight_saliency,
                family="hi_nerv",
                candidate_id=row_id,
                action_bits=decoder_weight_waterfill_action_bits,
                full_video_coverage=False,
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
        elif proof.get("runtime_consumption_proof_ready") is not True:
            row_blockers.append("receiver_proof_not_ready_for_archive_size_ladder_row")
        rows.append(
            {
                "family": "hi_nerv",
                "row_id": row_id,
                "modelsize_scale": float(spec["modelsize_scale"]),
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
                "submission_dir": _path_if_exists(row_dir / "submission"),
                "receiver_proof_executed": bool(emit_receiver_proof),
                "receiver_proof_path": _path_if_exists(proof_path),
                "runtime_consumption_proof_ready": proof.get(
                    "runtime_consumption_proof_ready"
                ),
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
        "archive_export_backend_counts": _archive_export_backend_counts(rows),
        "emit_receiver_proof": bool(emit_receiver_proof),
        "emit_decoder_weight_waterfill_plan": bool(emit_decoder_weight_waterfill_plan),
        "decoder_weight_waterfill_schema": NERV_DECODER_WEIGHT_WATERFILL_SCHEMA,
        "decoder_weight_saliency_json": (
            None
            if decoder_weight_saliency_json is None
            else Path(decoder_weight_saliency_json).expanduser().as_posix()
        ),
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
    return {
        "row_id": row_id,
        "modelsize_scale": float(
            candidate.get("target_modelsize_mparams")
            or candidate.get("modelsize_mparams")
            or 0.0
        ),
        "config": cfg,
        "decoder_codec": decoder_codec,
        "modelsize_candidate": dict(candidate),
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
                proof=row.get("runtime_consumption_proof_ready"),
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
                    "measured_score_source_row_id": score["source_row_id"],
                    "measured_score_source_schema": score.get("source_schema"),
                    "measured_score_source_path": (
                        None
                        if score_source_path is None
                        else Path(score_source_path).expanduser().as_posix()
                    ),
                }
            )
            if score["full_video_coverage"]:
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
    if rows_with_score:
        blockers = [
            blocker
            for blocker in blockers
            if blocker != "hinerv_archive_size_ladder_false_authority_no_nonrate_score"
        ]
    else:
        blockers.append("hinerv_archive_size_ladder_measured_scores_missing")
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
        proof_ready = (
            from_row.get("runtime_consumption_proof_ready") is True
            and to_row.get("runtime_consumption_proof_ready") is True
        )
        blockers: list[str] = []
        if from_nonrate is None or to_nonrate is None:
            blockers.append("hinerv_modelsize_increment_measured_nonrate_missing")
        if require_full_video and not (from_full and to_full):
            blockers.append("hinerv_modelsize_increment_full_video_score_missing")
        if not proof_ready:
            blockers.append("hinerv_modelsize_increment_receiver_proof_missing")
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
                    "runtime_consumption_proof_ready" if proof_ready else "missing"
                ),
                "full_video_coverage": bool(from_full and to_full),
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
    return {
        "source_row_id": str(row_id),
        "nonrate_score": float(nonrate),
        "avg_segnet_dist": d_seg,
        "avg_posenet_dist": d_pose,
        "full_video_coverage": _score_row_full_video(row),
        "axis_tag": str(
            row.get("axis_tag")
            or row.get("score_axis")
            or row.get("evidence_axis")
            or artifact_axis
            or "[macOS-MLX research-signal]"
        ),
        "source_schema": source_schema,
    }


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
    "REQUIRED_ALLOCATOR_BINDINGS",
    "attach_hinerv_archive_ladder_score_rows",
    "build_hinerv_archive_size_ladder",
    "hinerv_modelsize_increment_section_value_rows",
    "render_hinerv_archive_size_ladder_markdown",
]
