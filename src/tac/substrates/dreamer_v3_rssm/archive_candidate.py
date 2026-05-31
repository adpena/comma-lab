# SPDX-License-Identifier: MIT
"""Archive-bound runtime bridge for DreamerV3 RSSM MLX candidates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from tac.optimization.archive_bound_candidate_runtime_bridge import (
    emit_archive_bound_candidate_runtime_package,
)
from tac.repo_io import sha256_file
from tac.substrates._shared.pact_nerv_full_main import (
    build_archive_zip,
    write_contest_runtime,
)
from tac.substrates.dreamer_v3_rssm.archive import pack_archive

DREAMER_V3_RSSM_MLX_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA = (
    "dreamer_v3_rssm_mlx_archive_bound_adapter_package.v1"
)
DREAMER_V3_RSSM_MLX_RECEIVER_PROOF_SCHEMA = (
    "dreamer_v3_rssm_mlx_generated_receiver_proof.v1"
)
DREAMER_V3_RSSM_MLX_ARCHIVE_BOUND_ADAPTER_ID = (
    "dreamer_v3_rssm_mlx_archive_export"
)
DREAMER_V3_RSSM_MLX_ARCHIVE_CANDIDATE_FAMILY = "dreamer_v3_rssm_mlx"
DREAMER_V3_RSSM_MLX_ARCHIVE_TRANSFORM_KIND = (
    "dreamer_v3_rssm_mlx_categorical_predictive_coding_archive"
)
DREAMER_V3_RSSM_CONTEST_NUM_PAIRS = 600
DREAMER_V3_RSSM_MLX_CONTEST_RAW_BYTES = 1164 * 874 * 1200 * 3
DREAMER_V3_RSSM_CAMERA_H = 874
DREAMER_V3_RSSM_CAMERA_W = 1164


def _repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[4]


def _resolve_output_dir(
    output_dir: str | Path,
    *,
    repo_root: str | Path | None,
) -> tuple[Path, Path]:
    root = Path(repo_root) if repo_root is not None else _repo_root_from_here()
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    return root, out_dir


def _flatten_parameters(value: Any) -> dict[str, Any]:
    try:
        from mlx.utils import tree_flatten  # type: ignore[import-not-found]

        return {str(key): item for key, item in tree_flatten(value)}
    except Exception:
        pass

    out: dict[str, Any] = {}

    def visit(prefix: str, item: Any) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                next_prefix = f"{prefix}.{key}" if prefix else str(key)
                visit(next_prefix, child)
        else:
            out[prefix] = item

    visit("", value)
    return out


def dreamer_v3_rssm_meta_from_config(cfg: Any) -> dict[str, object]:
    """Receiver-visible metadata for the RSSMC1 archive."""

    return {
        "schema_version": "dreamer_v3_rssm_rssmc1_v1",
        "gumbel_temperature": float(cfg.gumbel_temperature),
        "use_straight_through": bool(cfg.use_straight_through),
        "unimix_alpha": float(cfg.unimix_alpha),
        "categorical_bits_per_sample": float(cfg.categorical_bits_per_sample),
        "latent_packing_bytes_per_pair": int(cfg.latent_packing_bytes_per_pair),
        "axis_tag": "[macOS-MLX research-signal]",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def expected_receiver_output_bytes_for_pairs(num_pairs: int) -> int:
    """Return contest raw bytes required for exact-ready Dreamer handoff.

    Partial MLX-local archives are valid advisory artifacts, but their receiver
    proof must not satisfy the archive-bound exact-handoff gate. Until the
    runtime implements deterministic 600-pair cycling, the proof expectation is
    the full contest output size for every archive.
    """

    if not isinstance(num_pairs, int) or num_pairs < 1:
        raise ValueError(f"num_pairs must be a positive int; got {num_pairs!r}")
    return DREAMER_V3_RSSM_MLX_CONTEST_RAW_BYTES


def _partial_archive_blockers_from_model(model: Any) -> list[str]:
    if int(model.cfg.num_pairs) == DREAMER_V3_RSSM_CONTEST_NUM_PAIRS:
        return []
    return [
        "dreamer_v3_rssm_partial_archive_not_contest_runtime_closure",
        "dreamer_v3_rssm_600_pair_archive_or_runtime_cycling_required",
    ]


def _runtime_manifest_extra_from_model(
    model: Any,
    *,
    expected_receiver_output_bytes: int,
    meta: Mapping[str, Any] | None,
) -> dict[str, Any]:
    runtime_extra: dict[str, Any] = {
        "schema": "dreamer_v3_rssm_mlx_runtime_adapter_manifest.v1",
        "predictive_coding_family": "dreamer_v3_categorical_rssm",
        "rssm_num_pairs": int(model.cfg.num_pairs),
        "contest_num_pairs_required": DREAMER_V3_RSSM_CONTEST_NUM_PAIRS,
        "partial_archive_receiver_proof_advisory_only": (
            int(model.cfg.num_pairs) != DREAMER_V3_RSSM_CONTEST_NUM_PAIRS
        ),
        "rssm_num_groups": int(model.cfg.num_groups),
        "rssm_num_categories": int(model.cfg.num_categories),
        "receiver_expected_raw_bytes": int(expected_receiver_output_bytes),
    }
    archive_meta = dict(meta or {})
    score_aware = archive_meta.get("score_aware_training")
    if isinstance(score_aware, Mapping):
        runtime_extra["score_aware_training"] = dict(score_aware)
    return runtime_extra


def exported_decoder_state_dict_from_model(model: Any) -> dict[str, np.ndarray]:
    """Return archive-ready decoder weights, excluding training-only logits."""

    if hasattr(model, "export_state_dict"):
        raw = model.export_state_dict()
    elif hasattr(model, "parameters"):
        raw = _flatten_parameters(model.parameters())
    else:
        raise ValueError("Dreamer model must expose export_state_dict() or parameters()")

    exported: dict[str, np.ndarray] = {}
    for key, value in raw.items():
        text_key = str(key)
        if text_key.startswith("logits"):
            continue
        exported[text_key] = np.asarray(value, dtype=np.float32)
    if not exported:
        raise ValueError("Dreamer model exported no decoder weights")
    return exported


def category_indices_from_model(model: Any) -> np.ndarray:
    """Return deterministic argmax category indices from trained logits."""

    logits = getattr(model, "logits", None)
    if logits is None:
        raise ValueError("Dreamer model lacks logits; pass category_indices explicitly")
    return np.asarray(logits).argmax(axis=-1).astype(np.int32, copy=False)


def pack_archive_from_model(
    model: Any,
    *,
    exported_state_dict: Mapping[str, Any] | None = None,
    category_indices: np.ndarray | None = None,
    meta: Mapping[str, Any] | None = None,
) -> bytes:
    """Pack a Dreamer MLX model into RSSMC1 bytes."""

    cfg = model.cfg
    decoder_state_dict = (
        {str(key): np.asarray(value, dtype=np.float32) for key, value in exported_state_dict.items()}
        if exported_state_dict is not None
        else exported_decoder_state_dict_from_model(model)
    )
    indices = (
        np.asarray(category_indices, dtype=np.int32)
        if category_indices is not None
        else category_indices_from_model(model)
    )
    archive_meta = dreamer_v3_rssm_meta_from_config(cfg)
    archive_meta.update(dict(meta or {}))
    archive_meta.update(
        {
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
    )
    return pack_archive(
        decoder_state_dict,
        indices,
        archive_meta,
        num_groups=int(cfg.num_groups),
        num_categories=int(cfg.num_categories),
        num_pairs=int(cfg.num_pairs),
        decoder_latent_dim=int(cfg.decoder_latent_dim),
        base_channels=int(cfg.base_channels),
    )


def export_dreamer_v3_rssm_mlx_archive(
    model: Any,
    output_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
    emit_archive_bound_candidate_package: bool = True,
    retain_receiver_proof_output: bool = False,
    mlx_triage_argv: Sequence[str] | None = None,
    category_indices: np.ndarray | None = None,
    meta: Mapping[str, Any] | None = None,
) -> tuple[Path, str, int]:
    """Export a DreamerV3 RSSM MLX model as a contest-shaped archive."""

    root, out_dir = _resolve_output_dir(output_dir, repo_root=repo_root)
    bin_bytes = pack_archive_from_model(
        model,
        category_indices=category_indices,
        meta=meta,
    )
    bin_path = out_dir / "0.bin"
    bin_path.write_bytes(bin_bytes)

    submission_dir = out_dir / "submission"
    write_contest_runtime(
        submission_dir,
        substrate_pkg_name="dreamer_v3_rssm",
        repo_root=root,
        runtime_module_files=("archive.py", "inflate.py"),
        vendor_shared_inflate_runtime=True,
    )
    (submission_dir / "0.bin").write_bytes(bin_bytes)
    archive_zip_path = out_dir / "archive.zip"
    build_archive_zip(
        archive_zip_path,
        bin_bytes=bin_bytes,
        submission_dir=submission_dir,
    )
    archive_sha256 = sha256_file(archive_zip_path)
    archive_bytes = archive_zip_path.stat().st_size
    expected_raw_bytes = expected_receiver_output_bytes_for_pairs(int(model.cfg.num_pairs))
    extra_blockers = _partial_archive_blockers_from_model(model)
    if emit_archive_bound_candidate_package:
        emit_archive_bound_candidate_runtime_package(
            adapter_id=DREAMER_V3_RSSM_MLX_ARCHIVE_BOUND_ADAPTER_ID,
            candidate_family=DREAMER_V3_RSSM_MLX_ARCHIVE_CANDIDATE_FAMILY,
            candidate_id_prefix="dreamer_v3_rssm_mlx",
            transform_kind=DREAMER_V3_RSSM_MLX_ARCHIVE_TRANSFORM_KIND,
            archive_zip_path=archive_zip_path,
            archive_sha256=archive_sha256,
            archive_bytes=archive_bytes,
            submission_dir=submission_dir,
            output_dir=out_dir,
            repo_root=root,
            receiver_contract_kind=(
                "dreamer_v3_rssm_mlx_generated_inflate_sh_decode_only_receiver"
            ),
            proof_schema=DREAMER_V3_RSSM_MLX_RECEIVER_PROOF_SCHEMA,
            proof_filename="dreamer_v3_rssm_mlx_receiver_proof.json",
            candidate_label="dreamer_v3_rssm",
            expected_receiver_output_bytes=expected_raw_bytes,
            retain_receiver_output=retain_receiver_proof_output,
            runtime_adapter_manifest_extra=_runtime_manifest_extra_from_model(
                model,
                expected_receiver_output_bytes=expected_raw_bytes,
                meta=meta,
            ),
            candidate_row_schema="dreamer_v3_rssm_mlx_archive_bound_candidate_row.v1",
            wrapper_schema=DREAMER_V3_RSSM_MLX_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA,
            extra_blockers=extra_blockers,
            mlx_triage_argv=mlx_triage_argv,
        )
    return (archive_zip_path, archive_sha256, archive_bytes)


def export_dreamer_v3_rssm_mlx_archive_bound_candidate_package(
    model: Any,
    output_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
    retain_receiver_proof_output: bool = False,
    mlx_triage_argv: Sequence[str] | None = None,
    category_indices: np.ndarray | None = None,
    meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Export Dreamer RSSMC1 bytes and emit the shared bridge package."""

    archive_zip_path, archive_sha256, archive_bytes = export_dreamer_v3_rssm_mlx_archive(
        model,
        output_dir,
        repo_root=repo_root,
        emit_archive_bound_candidate_package=False,
        category_indices=category_indices,
        meta=meta,
    )
    root, out_dir = _resolve_output_dir(output_dir, repo_root=repo_root)
    expected_raw_bytes = expected_receiver_output_bytes_for_pairs(int(model.cfg.num_pairs))
    extra_blockers = _partial_archive_blockers_from_model(model)
    return emit_archive_bound_candidate_runtime_package(
        adapter_id=DREAMER_V3_RSSM_MLX_ARCHIVE_BOUND_ADAPTER_ID,
        candidate_family=DREAMER_V3_RSSM_MLX_ARCHIVE_CANDIDATE_FAMILY,
        candidate_id_prefix="dreamer_v3_rssm_mlx",
        transform_kind=DREAMER_V3_RSSM_MLX_ARCHIVE_TRANSFORM_KIND,
        archive_zip_path=archive_zip_path,
        archive_sha256=archive_sha256,
        archive_bytes=archive_bytes,
        submission_dir=out_dir / "submission",
        output_dir=out_dir,
        repo_root=root,
        receiver_contract_kind=(
            "dreamer_v3_rssm_mlx_generated_inflate_sh_decode_only_receiver"
        ),
        proof_schema=DREAMER_V3_RSSM_MLX_RECEIVER_PROOF_SCHEMA,
        proof_filename="dreamer_v3_rssm_mlx_receiver_proof.json",
        candidate_label="dreamer_v3_rssm",
        expected_receiver_output_bytes=expected_raw_bytes,
        retain_receiver_output=retain_receiver_proof_output,
        runtime_adapter_manifest_extra=_runtime_manifest_extra_from_model(
            model,
            expected_receiver_output_bytes=expected_raw_bytes,
            meta=meta,
        ),
        candidate_row_schema="dreamer_v3_rssm_mlx_archive_bound_candidate_row.v1",
        wrapper_schema=DREAMER_V3_RSSM_MLX_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA,
        extra_blockers=extra_blockers,
        mlx_triage_argv=mlx_triage_argv,
    )


__all__ = [
    "DREAMER_V3_RSSM_MLX_ARCHIVE_BOUND_ADAPTER_ID",
    "DREAMER_V3_RSSM_MLX_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA",
    "DREAMER_V3_RSSM_MLX_ARCHIVE_CANDIDATE_FAMILY",
    "DREAMER_V3_RSSM_MLX_ARCHIVE_TRANSFORM_KIND",
    "DREAMER_V3_RSSM_MLX_CONTEST_RAW_BYTES",
    "DREAMER_V3_RSSM_MLX_RECEIVER_PROOF_SCHEMA",
    "category_indices_from_model",
    "dreamer_v3_rssm_meta_from_config",
    "expected_receiver_output_bytes_for_pairs",
    "export_dreamer_v3_rssm_mlx_archive",
    "export_dreamer_v3_rssm_mlx_archive_bound_candidate_package",
    "exported_decoder_state_dict_from_model",
    "pack_archive_from_model",
]
