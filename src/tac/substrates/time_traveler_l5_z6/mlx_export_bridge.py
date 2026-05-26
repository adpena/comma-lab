# SPDX-License-Identifier: MIT
"""Z6 MLX → PyTorch → Z6PCWM1 archive export bridge — L0 SCAFFOLD.

Wraps the canonical :mod:`tac.local_acceleration.mlx_to_pytorch_export` (#1251
bridge) for the Z6 substrate so MLX-trained weights flow to a contest Z6PCWM1
archive that the canonical PyTorch :mod:`tac.substrates.time_traveler_l5_z6.inflate`
runtime can decode. The exported archive feeds the #1265 contest-equivalence
gate (``tools/gate_mlx_candidate_contest_equivalence.py``) for non-promotable
validation before any paid CUDA dispatch.

Canonical flow
--------------

    Z6PredictiveCodingMLXRenderer (MLX-trained)
        |
        v   export_state_dict() + export_auxiliary_buffers()
        |
    (numpy state_dict in PyTorch layout) + (numpy aux: latent_init, residuals, ego_motion)
        |
        v   build_mlx_z6pcwm1_archive_from_mlx_renderer(...)
        |
        v   route state_dict through tac.substrates.time_traveler_l5_z6.architecture
        v   route aux buffers through tac.substrates.time_traveler_l5_z6.archive.pack_archive
        |
    Z6PCWM1 0.bin bytes (contest-grade archive)
        |
        v   tools/gate_mlx_candidate_contest_equivalence.py
        |
    PASS / FAIL → operator routes paid CUDA dispatch via operator_authorize.py

Non-promotable canonical contract
---------------------------------

Per Catalog #287/#323 the export manifest carries:
- ``axis_tag = "[macOS-MLX research-signal]"``
- ``evidence_grade = "macOS-MLX research-signal"``
- ``score_claim = False``
- ``promotion_eligible = False``
- ``ready_for_exact_eval_dispatch = False``

The Z6PCWM1 archive bytes themselves are byte-stable contest-grade format
(same magic / grammar / brotli quality as the PyTorch sister); only the
training-provenance metadata is tagged non-promotable.

Cross-references
----------------

- Canonical MLX-to-PyTorch export bridge (#1251):
  :mod:`tac.local_acceleration.mlx_to_pytorch_export`
- Canonical Z6 archive grammar:
  :func:`tac.substrates.time_traveler_l5_z6.archive.pack_archive`
- Canonical PyTorch architecture (target for state_dict load):
  :class:`tac.substrates.time_traveler_l5_z6.architecture.Z6PredictiveCodingSubstrate`
- Canonical PyTorch inflate (consumer of the archive):
  :mod:`tac.substrates.time_traveler_l5_z6.inflate`
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tac.local_acceleration.mlx_to_pytorch_export import (
    export_mlx_state_dict_to_torch_pt,
)
from tac.substrates.time_traveler_l5_z6.archive import pack_archive
from tac.substrates.time_traveler_l5_z6.mlx_renderer import (
    EVIDENCE_GRADE,
    EVIDENCE_TAG,
    LANE_ID,
    Z6PredictiveCodingMLXRenderer,
)

__all__ = [
    "Z6_MLX_ARCHIVE_BUILD_SCHEMA",
    "Z6_MLX_TO_PYTORCH_EXPORT_SCHEMA",
    "build_z6_pytorch_pt_from_mlx_renderer",
    "build_z6pcwm1_archive_from_mlx_renderer",
]


Z6_MLX_TO_PYTORCH_EXPORT_SCHEMA = "z6_mlx_to_pytorch_state_dict_export.v1"
Z6_MLX_ARCHIVE_BUILD_SCHEMA = "z6_mlx_z6pcwm1_archive_build.v1"


def build_z6_pytorch_pt_from_mlx_renderer(
    renderer: Z6PredictiveCodingMLXRenderer,
    output_pt_path: Path | str,
    *,
    run_id: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Export the MLX renderer's state_dict to a PyTorch .pt file via canonical bridge.

    Aux buffers (latent_init, residuals, ego_motion) are NOT in the .pt — they
    are stored separately in the Z6PCWM1 archive via
    :func:`build_z6pcwm1_archive_from_mlx_renderer`.

    Returns the canonical export manifest with non-promotable Provenance markers.
    """
    if run_id is None:
        run_id = datetime.now(UTC).strftime("z6_mlx_export_%Y%m%dT%H%M%SZ")
    state_dict_np = renderer.export_state_dict()
    manifest = export_mlx_state_dict_to_torch_pt(
        state_dict_np,
        output_pt_path,
        substrate_id="time_traveler_l5_z6",
        run_id=run_id,
        overwrite=overwrite,
    )
    # Stamp Z6-specific schema for downstream consumer dispatch
    manifest["z6_mlx_export_schema_version"] = Z6_MLX_TO_PYTORCH_EXPORT_SCHEMA
    manifest["lane_id"] = LANE_ID
    return manifest


def build_z6pcwm1_archive_from_mlx_renderer(
    renderer: Z6PredictiveCodingMLXRenderer,
    output_archive_path: Path | str,
    *,
    run_id: str | None = None,
    overwrite: bool = False,
    lambda_residual_entropy: float = 1.0,
) -> dict[str, Any]:
    """Build a contest Z6PCWM1 0.bin archive from MLX-trained weights.

    Routes the MLX renderer's state_dict + aux buffers through the canonical
    PyTorch substrate's :func:`pack_archive` so the resulting bytes are
    byte-stable contest format. The canonical PyTorch inflate runtime
    (:mod:`tac.substrates.time_traveler_l5_z6.inflate`) consumes the archive
    without modification.

    Args:
        renderer: MLX-trained Z6 renderer (provides state_dict + aux buffers).
        output_archive_path: destination path for the Z6PCWM1 0.bin bytes.
        run_id: canonical run identifier (default = UTC timestamp marker).
        overwrite: if False, raise FileExistsError when destination exists.
        lambda_residual_entropy: Lagrangian weight stamped in archive meta
            (sister to the trainer's loss weight; default 1.0 per design memo).

    Returns:
        Canonical build manifest with archive sha256, byte size, parameter
        counts, and non-promotable Provenance markers.
    """
    import torch

    if run_id is None:
        run_id = datetime.now(UTC).strftime("z6_mlx_archive_%Y%m%dT%H%M%SZ")

    out_path = Path(output_archive_path)
    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"output archive exists; pass overwrite=True to replace: {out_path}"
        )

    state_dict_np = renderer.export_state_dict()
    aux = renderer.export_auxiliary_buffers()

    # Split state_dict by submodule prefix for pack_archive (which takes
    # encoder/decoder/predictor state_dicts as separate args).
    def _split(prefix: str) -> dict[str, torch.Tensor]:
        return {
            k[len(prefix) + 1 :]: torch.from_numpy(v.copy()).float()
            for k, v in state_dict_np.items()
            if k.startswith(prefix + ".")
        }

    encoder_sd = _split("encoder")
    decoder_sd = _split("decoder")
    predictor_sd = _split("predictor")

    latent_init = torch.from_numpy(aux["latent_init"].copy()).float()
    residuals = torch.from_numpy(aux["residuals"].copy()).float()
    ego_motion = torch.from_numpy(aux["ego_motion"].copy()).float()

    # Build the meta dict expected by pack_archive (fields consumed by inflate.py)
    cfg = renderer.cfg
    meta: dict[str, Any] = {
        "encoder_input_channels": int(cfg.encoder_input_channels),
        "encoder_hidden_dim": int(cfg.encoder_hidden_dim),
        "decoder_embed_dim": int(cfg.decoder_embed_dim),
        "decoder_initial_grid_h": int(cfg.decoder_initial_grid_h),
        "decoder_initial_grid_w": int(cfg.decoder_initial_grid_w),
        "decoder_channels": [int(c) for c in cfg.decoder_channels],
        "decoder_num_upsample_blocks": int(cfg.decoder_num_upsample_blocks),
        "output_height": int(cfg.output_height),
        "output_width": int(cfg.output_width),
        "predictor_hidden_dim": int(cfg.predictor_hidden_dim),
        "predictor_film_mlp_hidden_dim": int(cfg.predictor_film_mlp_hidden_dim),
        "latent_init_std": float(cfg.latent_init_std),
        # Non-promotable training-provenance markers per Catalog #287/#323
        "mlx_training_evidence_grade": EVIDENCE_GRADE,
        "mlx_training_evidence_tag": EVIDENCE_TAG,
        "mlx_training_score_claim": False,
        "mlx_training_promotion_eligible": False,
        "mlx_training_ready_for_exact_eval_dispatch": False,
        "mlx_training_lane_id": LANE_ID,
        "mlx_training_run_id": run_id,
    }

    archive_bytes = pack_archive(
        encoder_sd,
        decoder_sd,
        predictor_sd,
        latent_init,
        residuals,
        ego_motion,
        meta,
        lambda_residual_entropy=lambda_residual_entropy,
        predictor_kernel_size=int(cfg.predictor_kernel_size),
        identity_predictor=bool(cfg.identity_predictor),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(archive_bytes)
    archive_sha256 = hashlib.sha256(archive_bytes).hexdigest()
    archive_size = len(archive_bytes)

    breakdown = renderer.num_parameters_breakdown()
    return {
        "schema_version": Z6_MLX_ARCHIVE_BUILD_SCHEMA,
        "substrate_id": "time_traveler_l5_z6",
        "lane_id": LANE_ID,
        "run_id": run_id,
        "output_archive_path": str(out_path),
        "archive_sha256": archive_sha256,
        "archive_size_bytes": archive_size,
        "parameter_breakdown": breakdown,
        # Canonical Provenance non-promotable markers per Catalog #287/#323/#192/#1.
        "training_evidence_grade": EVIDENCE_GRADE,
        "training_evidence_tag": EVIDENCE_TAG,
        "score_claim": False,
        "promotion_eligible": False,
        "promotable": False,
        "ready_for_exact_eval_dispatch": False,
        "blockers": [
            "macos_mlx_research_signal_training_axis_only",
            "requires_paired_cuda_t4_or_linux_x86_64_eval_for_promotion",
            "requires_pass_verdict_from_tools_gate_mlx_candidate_contest_equivalence",
        ],
        "operator_routable_next_step": (
            "Run tools/gate_mlx_candidate_contest_equivalence.py with this archive "
            "to validate contest-equivalence (#1265 gate); on PASS, route paid "
            "CUDA dispatch via tools/operator_authorize.py per Catalog #313."
        ),
    }
