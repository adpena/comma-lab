# SPDX-License-Identifier: MIT
"""Measured HiNeRV archive-size ladder.

This is rate evidence only. It exports actual receiver-shaped HiNeRV archives
for the local model-size configs, records archive ZIP bytes and hashes, and
keeps non-rate scorer authority closed until a scorer replay is attached.
"""

from __future__ import annotations

import json
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


def build_hinerv_archive_size_ladder(
    *,
    output_dir: str | Path,
    repo_root: str | Path,
    num_pairs: int = 600,
    row_ids: Iterable[str] | None = None,
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
    specs = [
        spec
        for spec in hi_nerv_modelsize_config_rows(num_pairs=int(num_pairs))
        if selected is None or str(spec["row_id"]) in selected
    ]
    missing = sorted(selected - {str(spec["row_id"]) for spec in specs}) if selected else []
    rows = []
    blockers: list[str] = []
    if missing:
        blockers.append("hinerv_archive_size_ladder_requested_rows_missing")
    for spec in specs:
        row_id = str(spec["row_id"])
        cfg = spec["config"]
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
            decoder_codec=str(decoder_codec),
            source_backend=archive_export_backend,
        )
        proof_path = row_dir / "receiver_proof" / "hi_nerv_mlx_receiver_proof.json"
        proof = _read_json_if_exists(proof_path)
        state_npz_manifest_path = row_dir / "hi_nerv_mlx_exported_state_npz_manifest.json"
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
                "config": _config_snapshot(cfg),
                "decoder_codec": str(decoder_codec),
                "archive_export_backend": archive_export_backend,
                "backend_claim_blockers": backend_claim_blockers,
                "num_parameters": int(model.num_parameters()),
                "archive_path": archive_path.as_posix(),
                "archive_sha256": archive_sha256,
                "archive_bytes": int(archive_bytes),
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
        f"Decoder codec: `{report.get('decoder_codec')}`",
        "",
        "| row | params | archive bytes | rate score | proof ready |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report.get("archive_rows", ()):
        lines.append(
            "| {row_id} | {params} | {bytes} | {rate:.6f} | {proof} |".format(
                row_id=row["row_id"],
                params=row["num_parameters"],
                bytes=row["archive_bytes"],
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


def _config_snapshot(cfg: Any) -> dict[str, Any]:
    return {
        "latent_dim_coarse": int(cfg.latent_dim_coarse),
        "latent_dim_mid": int(cfg.latent_dim_mid),
        "latent_dim_fine": int(cfg.latent_dim_fine),
        "embed_dim": int(cfg.embed_dim),
        "decoder_channels": [int(value) for value in cfg.decoder_channels],
        "num_upsample_blocks": int(cfg.num_upsample_blocks),
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
    "build_hinerv_archive_size_ladder",
    "hinerv_modelsize_increment_section_value_rows",
    "render_hinerv_archive_size_ladder_markdown",
]
