# SPDX-License-Identifier: MIT
"""Z4 Atick-Redlich archive candidate builder — canonical bridge to Z4ATR bytes.

Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"
+ Catalog #146 contest-compliant runtime contract + Catalog #220
operational mechanism: this helper bridges a trained
``Z4AtickRedlichSubstrate`` into Z4ATR archive bytes that the inflate
runtime consumes faithfully.

Canonical pattern (mirrors sister
``z6_v2_cargo_cult_unwind/archive_candidate.py``):

1. Extract decoder state_dict (everything EXCEPT ``latents``,
   ``decorrelator.proj.weight``, ``decorrelator.proj.bias``).
2. Extract latents tensor + decorrelator weight + decorrelator bias as
   separate canonical blob sections.
3. Pack meta JSON with architectural hyperparameters required by
   ``inflate.py::inflate_one_video`` to re-instantiate the substrate
   without ambient state.
4. Delegate to ``archive.pack_archive`` for byte-deterministic encoding.

The canonical extraction discipline keeps the decorrelator blob as a
SEPARATE archive section (per Catalog #272 distinguishing-feature
integration contract) — it does NOT get folded into the decoder
state_dict. This is the operational mechanism that Catalog #220 +
Catalog #272 + Catalog #105/#139 no-op detector contracts verify.

[verified-against: src/tac/substrates/z6_v2_cargo_cult_unwind/archive_candidate.py
 sister bridge pattern (extracts latents + ego_vecs as separate sections)]
[verified-against: Catalog #146 + Catalog #220 + Catalog #272 contracts]
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from tac.optimization.archive_bound_candidate_runtime_bridge import (
    emit_archive_bound_candidate_runtime_package,
)
from tac.repo_io import sha256_file
from tac.substrates._shared.pact_nerv_full_main import (
    build_archive_zip,
    write_contest_runtime,
)

from .archive import Z4ATR_SCHEMA_VERSION, pack_archive
from .inflate import CONTEST_RAW_BYTES

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch

    from .architecture import Z4AtickRedlichConfig, Z4AtickRedlichSubstrate

# Canonical state_dict keys that are NOT part of the decoder blob (they
# get their own dedicated archive sections per Catalog #272 distinguishing-
# feature integration contract).
DECODER_EXCLUDED_KEYS: frozenset[str] = frozenset(
    {
        "latents",
        "decorrelator.proj.weight",
        "decorrelator.proj.bias",
    }
)
Z4_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA = "z4_archive_bound_adapter_package.v1"
Z4_RECEIVER_PROOF_SCHEMA = "z4_generated_receiver_proof.v1"
Z4_ARCHIVE_BOUND_ADAPTER_ID = "z4_atick_redlich_archive_export"
Z4_ARCHIVE_CANDIDATE_FAMILY = "time_traveler_l5_z4_atick_redlich"
Z4_ARCHIVE_TRANSFORM_KIND = "z4_atick_redlich_cooperative_receiver_neural_archive"


def extract_decoder_state_dict(
    model: Z4AtickRedlichSubstrate,
) -> dict[str, torch.Tensor]:
    """Extract the decoder state_dict (everything EXCEPT latents + decorrelator).

    Per Catalog #272 distinguishing-feature contract: the decorrelator
    weight + bias are stored in their OWN archive section (the
    distinguishing-feature payload). The latents are stored in their OWN
    int16-quantized section. Only the renderer weights remain in the
    "decoder" state_dict.
    """
    full = model.state_dict()
    decoder = {
        k: v for k, v in full.items() if k not in DECODER_EXCLUDED_KEYS
    }
    return decoder


def build_meta(cfg: Z4AtickRedlichConfig) -> dict[str, object]:
    """Build the canonical meta dict for Z4ATR archive packing.

    Per Catalog #146 inflate runtime contract: the meta dict MUST contain
    every architectural hyperparameter needed to re-instantiate
    ``Z4AtickRedlichConfig`` at inflate time.
    """
    return {
        "embed_dim": int(cfg.embed_dim),
        "initial_grid_h": int(cfg.initial_grid_h),
        "initial_grid_w": int(cfg.initial_grid_w),
        "decoder_channels": list(cfg.decoder_channels),
        "num_upsample_blocks": int(cfg.num_upsample_blocks),
        "sin_frequency": float(cfg.sin_frequency),
        "output_height": int(cfg.output_height),
        "output_width": int(cfg.output_width),
        "apply_decorrelator": bool(cfg.apply_decorrelator),
        "cooperative_receiver_beta": float(cfg.cooperative_receiver_beta),
        "_substrate_id": "time_traveler_l5_z4",
        "_substrate_variant": "atick_redlich_cooperative_receiver",
        "_archive_grammar_version": Z4ATR_SCHEMA_VERSION,
    }


def build_archive_bytes(
    model: Z4AtickRedlichSubstrate,
    *,
    extra_meta: dict[str, object] | None = None,
) -> bytes:
    """Build canonical Z4ATR archive bytes from a trained substrate.

    Per CLAUDE.md "Bit-level deconstruction and entropy discipline" +
    "Canonical leaderboard binding-depth discipline" L20 + L21 + L29 +
    L32: byte-deterministic under deterministic input.
    """
    decoder = extract_decoder_state_dict(model)
    latents = model.latents.detach().cpu()
    decorrelator_w = model.decorrelator.proj.weight.detach().cpu()
    decorrelator_b = model.decorrelator.proj.bias.detach().cpu()
    meta = build_meta(model.cfg)
    if extra_meta:
        # extra_meta keys MUST NOT collide with canonical meta keys
        # (defensive guard so callers cannot silently overwrite the
        # canonical architectural hyperparameters).
        for k in extra_meta:
            if k in meta:
                raise ValueError(
                    f"extra_meta key {k!r} collides with canonical meta key"
                )
        meta.update(extra_meta)
    return pack_archive(
        decoder_state_dict=decoder,
        latents=latents,
        decorrelator_weight=decorrelator_w,
        decorrelator_bias=decorrelator_b,
        meta=meta,
    )


def export_z4_archive(
    model: Z4AtickRedlichSubstrate,
    output_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
    emit_archive_bound_candidate_package: bool = True,
    retain_receiver_proof_output: bool = False,
    mlx_triage_argv: Sequence[str] | None = None,
) -> tuple[Path, str, int]:
    """Export a Z4 model as a contest-shaped ``archive.zip``.

    This is the runtime bridge counterpart to ``build_archive_bytes``: the
    emitted archive, packaged inflate runtime, receiver proof, exact blocker,
    replay bundle, and posterior hook all flow through the shared
    archive-bound candidate contract. Local/MLX training outputs stay advisory
    until contest CPU/CUDA authority consumes the resulting packet.
    """

    root = (
        Path(repo_root)
        if repo_root is not None
        else Path(__file__).resolve().parents[4]
    )
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    bin_bytes = build_archive_bytes(model)
    bin_path = out_dir / "0.bin"
    bin_path.write_bytes(bin_bytes)

    submission_dir = out_dir / "submission"
    write_contest_runtime(
        submission_dir,
        substrate_pkg_name="time_traveler_l5_z4",
        repo_root=root,
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
    if emit_archive_bound_candidate_package:
        emit_archive_bound_candidate_runtime_package(
            adapter_id=Z4_ARCHIVE_BOUND_ADAPTER_ID,
            candidate_family=Z4_ARCHIVE_CANDIDATE_FAMILY,
            candidate_id_prefix="z4_atick_redlich",
            transform_kind=Z4_ARCHIVE_TRANSFORM_KIND,
            archive_zip_path=archive_zip_path,
            archive_sha256=archive_sha256,
            archive_bytes=archive_bytes,
            submission_dir=submission_dir,
            output_dir=out_dir,
            repo_root=root,
            receiver_contract_kind="z4_generated_inflate_sh_decode_only_receiver",
            proof_schema=Z4_RECEIVER_PROOF_SCHEMA,
            proof_filename="z4_receiver_proof.json",
            candidate_label="z4",
            expected_receiver_output_bytes=CONTEST_RAW_BYTES,
            retain_receiver_output=retain_receiver_proof_output,
            runtime_adapter_manifest_extra={
                "schema": "z4_runtime_adapter_manifest.v1",
                "cooperative_receiver_family": "atick_redlich_spatial_decorrelation",
                "cooperative_receiver_beta": float(
                    model.cfg.cooperative_receiver_beta
                ),
            },
            candidate_row_schema="z4_archive_bound_candidate_row.v1",
            wrapper_schema=Z4_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA,
            mlx_triage_argv=mlx_triage_argv,
        )
    return (archive_zip_path, archive_sha256, archive_bytes)


def export_z4_archive_bound_candidate_package(
    model: Z4AtickRedlichSubstrate,
    output_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
    retain_receiver_proof_output: bool = False,
    mlx_triage_argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Export Z4 bytes and emit the shared archive-bound package."""

    archive_zip_path, archive_sha256, archive_bytes = export_z4_archive(
        model,
        output_dir,
        repo_root=repo_root,
        emit_archive_bound_candidate_package=False,
    )
    root = (
        Path(repo_root)
        if repo_root is not None
        else Path(__file__).resolve().parents[4]
    )
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    return emit_archive_bound_candidate_runtime_package(
        adapter_id=Z4_ARCHIVE_BOUND_ADAPTER_ID,
        candidate_family=Z4_ARCHIVE_CANDIDATE_FAMILY,
        candidate_id_prefix="z4_atick_redlich",
        transform_kind=Z4_ARCHIVE_TRANSFORM_KIND,
        archive_zip_path=archive_zip_path,
        archive_sha256=archive_sha256,
        archive_bytes=archive_bytes,
        submission_dir=out_dir / "submission",
        output_dir=out_dir,
        repo_root=root,
        receiver_contract_kind="z4_generated_inflate_sh_decode_only_receiver",
        proof_schema=Z4_RECEIVER_PROOF_SCHEMA,
        proof_filename="z4_receiver_proof.json",
        candidate_label="z4",
        expected_receiver_output_bytes=CONTEST_RAW_BYTES,
        retain_receiver_output=retain_receiver_proof_output,
        runtime_adapter_manifest_extra={
            "schema": "z4_runtime_adapter_manifest.v1",
            "cooperative_receiver_family": "atick_redlich_spatial_decorrelation",
            "cooperative_receiver_beta": float(model.cfg.cooperative_receiver_beta),
        },
        candidate_row_schema="z4_archive_bound_candidate_row.v1",
        wrapper_schema=Z4_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA,
        mlx_triage_argv=mlx_triage_argv,
    )


__all__ = [
    "DECODER_EXCLUDED_KEYS",
    "Z4_ARCHIVE_BOUND_ADAPTER_ID",
    "Z4_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA",
    "Z4_ARCHIVE_CANDIDATE_FAMILY",
    "Z4_ARCHIVE_TRANSFORM_KIND",
    "Z4_RECEIVER_PROOF_SCHEMA",
    "build_archive_bytes",
    "build_meta",
    "export_z4_archive",
    "export_z4_archive_bound_candidate_package",
    "extract_decoder_state_dict",
]
