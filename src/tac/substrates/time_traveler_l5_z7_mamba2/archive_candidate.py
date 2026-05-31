# SPDX-License-Identifier: MIT
"""Byte-closed Z7-Mamba-2 archive export helpers for MLX-trained artifacts.

Mirrors the canonical sister
:mod:`tac.substrates.z6_v2_cargo_cult_unwind.archive_candidate` per the 11th
INDIVIDUALLY-FRACTAL standing directive 2026-05-27 — Z7-Mamba-2's OWN archive
export pass, NOT shared-helper shortcut from Z6-v2 sister.

Z7-Mamba-2's substrate-distinguishing primitive (Mamba-2 selective state-space
recurrence + Z6-compatible PixelShuffle decoder) is implemented at the
``architecture.py`` + ``mlx_native.py`` + ``mlx_module.py`` surfaces; this
module bridges the MLX-trained state_dict into the Z7MCM2 byte-closed archive
used by the inflate runtime.

The MLX → Z7MCM2 path:

1. ``Z7Mamba2MLXModule.export_state_dict()`` returns a numpy dict with
   PyTorch-layout keys + shapes (the existing
   ``Z7Mamba2MLXNativeRenderer.export_state_dict`` does the heavy lifting:
   transposes MLX channels-last conv weights ``(out, kH, kW, in)`` to
   PyTorch ``(out, in, kH, kW)``; keys match the canonical PyTorch sister
   ``Z7Mamba2PredictiveCodingSubstrate.state_dict()`` 1:1 per Catalog #1251).
2. THIS module splits that flat dict into ``encoder_state_dict`` (empty for
   ``context_conditioning_mode="none"``; populated for ``"latent_affine"``
   per Z7-LSTM/GRU sister convention) / ``decoder_state_dict`` /
   ``predictor_state_dict`` per the
   :func:`tac.substrates.time_traveler_l5_z7_mamba2.archive.pack_archive`
   contract.
3. Converts each section's numpy arrays to ``torch.Tensor`` (required by
   ``pack_archive`` signature) + extracts ``latent_init`` / ``residuals`` /
   ``ego_motion`` separately.
4. Delegates to ``pack_archive`` for the canonical Z7MCM2 byte stream.
5. Writes ``0.bin`` + builds the contest-shaped ``archive.zip`` via the
   canonical PR95-family helpers ``write_contest_runtime`` +
   ``build_archive_zip`` per Catalog #146 / #205 / #295 / #367.

[verified-against: Catalog #146 contest-compliant inflate runtime template]
[verified-against: Catalog #205 canonical select_inflate_device]
[verified-against: Catalog #295 PYTHONPATH self-containment]
[verified-against: tac.substrates.z6_v2_cargo_cult_unwind.archive_candidate canonical sister pattern]
[verified-against: tac.substrates.time_traveler_l5_z7_mamba2.archive.pack_archive signature]
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from tac.optimization.archive_bound_candidate_adapter_spine import (
    build_archive_bound_candidate_adapter_package,
)
from tac.repo_io import sha256_file, tree_sha256, write_json
from tac.substrates._shared.pact_nerv_full_main import (
    build_archive_zip,
    write_contest_runtime,
)
from tac.substrates.time_traveler_l5_z7_mamba2.architecture import (
    Z7Mamba2PredictiveCodingConfig,
)
from tac.substrates.time_traveler_l5_z7_mamba2.archive import pack_archive

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from tac.substrates.time_traveler_l5_z7_mamba2.mlx_native import (
        Z7Mamba2MLXRenderConfig,
    )

Z7_MAMBA2_MLX_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA = (
    "z7_mamba2_mlx_archive_bound_adapter_package.v1"
)
Z7_MAMBA2_MLX_RECEIVER_PROOF_SCHEMA = "z7_mamba2_mlx_generated_receiver_proof.v1"
Z7_MAMBA2_MLX_ARCHIVE_BOUND_ADAPTER_ID = "z7_mamba2_mlx_archive_export"
Z7_MAMBA2_MLX_ARCHIVE_CANDIDATE_FAMILY = "time_traveler_l5_z7_mamba2_mlx"
Z7_MAMBA2_MLX_ARCHIVE_TRANSFORM_KIND = (
    "z7_mamba2_mlx_ssd_predictive_coding_archive"
)
Z7_MAMBA2_MLX_REFERENCE_ARCHIVE_TRANSFORM_KIND = (
    "z7_mamba2_mlx_reference_s6_predictive_coding_archive"
)


def z7_mamba2_pytorch_config_from_mlx(
    mlx_cfg: Z7Mamba2MLXRenderConfig,
) -> Z7Mamba2PredictiveCodingConfig:
    """Build a canonical PyTorch config from the MLX render config.

    The MLX render config is a strict subset of the PyTorch config (context
    conditioning is DEFERRED to a sister L1 EXTENSION per the existing
    ``Z7Mamba2MLXRenderConfig`` docstring); we fill in the PyTorch-specific
    fields with their canonical defaults so ``pack_archive`` validates clean.
    """
    use_ssd = bool(getattr(mlx_cfg, "use_canonical_ssd_mlx_backend", False))
    return Z7Mamba2PredictiveCodingConfig(
        latent_dim=mlx_cfg.latent_dim,
        ego_motion_dim=mlx_cfg.ego_motion_dim,
        d_model=mlx_cfg.d_model,
        d_state=mlx_cfg.d_state,
        expand=mlx_cfg.expand,
        d_conv=mlx_cfg.d_conv,
        stateful=mlx_cfg.stateful,
        identity_predictor=mlx_cfg.identity_predictor,
        num_pairs=mlx_cfg.num_pairs,
        decoder_embed_dim=mlx_cfg.decoder_embed_dim,
        decoder_initial_grid_h=mlx_cfg.decoder_initial_grid_h,
        decoder_initial_grid_w=mlx_cfg.decoder_initial_grid_w,
        decoder_channels=tuple(int(c) for c in mlx_cfg.decoder_channels),
        decoder_num_upsample_blocks=mlx_cfg.decoder_num_upsample_blocks,
        output_height=mlx_cfg.output_height,
        output_width=mlx_cfg.output_width,
        latent_init_std=mlx_cfg.latent_init_std,
        # MLX scaffold is "none" context mode only (per existing
        # ``Z7Mamba2MLXRenderConfig`` docstring; ``latent_affine`` is a sister
        # L1 EXTENSION). The PyTorch sister default also matches "none".
        backend="ssd_reference" if use_ssd else "reference_torch",
        ssd_nheads=(
            int(mlx_cfg.effective_ssd_nheads)
            if use_ssd and mlx_cfg.effective_ssd_nheads is not None
            else None
        ),
        ssd_headdim=(
            int(mlx_cfg.effective_ssd_headdim)
            if use_ssd and mlx_cfg.effective_ssd_headdim is not None
            else 64
        ),
        context_conditioning_mode="none",
    )


def _split_state_dict_for_z7mcm2(
    exported_state_dict: dict[str, np.ndarray],
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Split MLX-exported state_dict into Z7MCM2 sections.

    The exported dict has PyTorch-layout keys:
    - ``predictor.*`` (Mamba-2 cell + input/output projections)
    - ``decoder.*`` (initial_proj + PixelShuffle conv stack + final conv)
    - ``latent_init`` / ``residuals`` / ``ego_motion_buffer``
    - (optional) ``context_conditioner.*`` for latent_affine mode

    Returns:
        ``(encoder_sd, decoder_sd, predictor_sd, latent_init, residuals, ego)``
        where the encoder_sd is empty when context mode is "none" and populated
        with ``context_conditioner.*`` keys when "latent_affine" is active
        (sister of Z7-LSTM/GRU convention).
    """
    encoder_sd: dict[str, torch.Tensor] = {}
    decoder_sd: dict[str, torch.Tensor] = {}
    predictor_sd: dict[str, torch.Tensor] = {}
    latent_init: torch.Tensor | None = None
    residuals: torch.Tensor | None = None
    ego_motion: torch.Tensor | None = None

    for key, arr in exported_state_dict.items():
        tensor = torch.from_numpy(np.asarray(arr).copy()).to(dtype=torch.float32)
        if key == "latent_init":
            latent_init = tensor
        elif key == "residuals":
            residuals = tensor
        elif key == "ego_motion_buffer":
            ego_motion = tensor
        elif key.startswith("predictor."):
            predictor_sd[key] = tensor
        elif key.startswith("decoder."):
            decoder_sd[key] = tensor
        elif key.startswith("context_conditioner."):
            encoder_sd[key] = tensor
        else:
            # Unknown key — surface explicitly so a future state_dict shape
            # drift fails closed instead of silently routing into the wrong
            # section (sister of Catalog #229 premise-verification discipline).
            raise ValueError(
                f"unexpected Z7-Mamba-2 exported state_dict key {key!r}; "
                "expected one of {predictor.*, decoder.*, "
                "context_conditioner.*, latent_init, residuals, "
                "ego_motion_buffer}"
            )

    if latent_init is None or residuals is None or ego_motion is None:
        raise ValueError(
            "exported_state_dict missing required {latent_init, residuals, "
            "ego_motion_buffer}; got keys: "
            f"{sorted(exported_state_dict.keys())}"
        )
    return encoder_sd, decoder_sd, predictor_sd, latent_init, residuals, ego_motion


def z7_mamba2_meta_from_config(
    cfg: Z7Mamba2MLXRenderConfig,
) -> dict[str, object]:
    """Minimal receiver metadata needed to rebuild Z7Mamba2 at inflate time."""
    return {
        "latent_dim": int(cfg.latent_dim),
        "ego_motion_dim": int(cfg.ego_motion_dim),
        "d_model": int(cfg.d_model),
        "d_state": int(cfg.d_state),
        "expand": int(cfg.expand),
        "d_conv": int(cfg.d_conv),
        "decoder_embed_dim": int(cfg.decoder_embed_dim),
        "decoder_initial_grid_h": int(cfg.decoder_initial_grid_h),
        "decoder_initial_grid_w": int(cfg.decoder_initial_grid_w),
        "decoder_channels": [int(c) for c in cfg.decoder_channels],
        "decoder_num_upsample_blocks": int(cfg.decoder_num_upsample_blocks),
        "output_height": int(cfg.output_height),
        "output_width": int(cfg.output_width),
        "stateful": bool(cfg.stateful),
        "identity_predictor": bool(cfg.identity_predictor),
        "context_conditioning_mode": "none",
        "mamba2_archive_backend": (
            "ssd_reference"
            if bool(getattr(cfg, "use_canonical_ssd_mlx_backend", False))
            else "reference_torch"
        ),
        "use_canonical_ssd_mlx_backend": bool(
            getattr(cfg, "use_canonical_ssd_mlx_backend", False)
        ),
        "ssd_nheads": (
            int(cfg.effective_ssd_nheads)
            if getattr(cfg, "effective_ssd_nheads", None) is not None
            else None
        ),
        "ssd_headdim": (
            int(cfg.effective_ssd_headdim)
            if getattr(cfg, "effective_ssd_headdim", None) is not None
            else None
        ),
        "mamba2_mlx_backend_lineage": str(cfg.mamba2_mlx_backend_lineage),
        "canonical_ssd_mlx_backend_wired": bool(cfg.canonical_ssd_mlx_backend_wired),
        "canonical_ssd_mlx_runtime_bridge_wired": bool(
            getattr(cfg, "use_canonical_ssd_mlx_backend", False)
        ),
        "canonical_ssd_mlx_blocker": str(cfg.canonical_ssd_mlx_blocker),
    }


def pack_archive_from_exported_state_dict(
    *,
    exported_state_dict: dict[str, np.ndarray],
    mlx_cfg: Z7Mamba2MLXRenderConfig,
) -> bytes:
    """Pack a PyTorch-layout exported MLX state dict into Z7MCM2 ``0.bin`` bytes."""
    (
        encoder_sd,
        decoder_sd,
        predictor_sd,
        latent_init,
        residuals,
        ego_motion,
    ) = _split_state_dict_for_z7mcm2(exported_state_dict)
    pytorch_cfg = z7_mamba2_pytorch_config_from_mlx(mlx_cfg)
    return pack_archive(
        encoder_state_dict=encoder_sd,
        decoder_state_dict=decoder_sd,
        predictor_state_dict=predictor_sd,
        latent_init=latent_init,
        residuals=residuals,
        ego_motion=ego_motion,
        meta=z7_mamba2_meta_from_config(mlx_cfg),
        config=pytorch_cfg,
    )


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return (
            path.resolve(strict=False)
            .relative_to(repo_root.resolve(strict=False))
            .as_posix()
        )
    except ValueError:
        return path.as_posix()


def _safe_text(value: object, *, max_chars: int = 4096) -> str:
    text = "" if value is None else str(value)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


def _run_generated_receiver_proof(
    *,
    archive_zip_path: Path,
    archive_sha256: str,
    archive_bytes: int,
    submission_dir: Path,
    output_dir: Path,
    repo_root: Path,
    retain_receiver_output: bool = False,
    timeout_seconds: int = 1800,
) -> dict[str, object]:
    """Exercise the generated contest runtime and emit a proof artifact.

    This deliberately proves the same contract the receiver gets: the generated
    ``inflate.sh`` consumes the byte-closed ``0.bin`` packet through the shipped
    runtime tree. It does not inspect scorers and never grants score authority.
    """
    proof_dir = output_dir / "receiver_proof"
    proof_dir.mkdir(parents=True, exist_ok=True)
    file_list = proof_dir / "file_list.txt"
    file_list.write_text("0.mkv\n", encoding="utf-8")
    receiver_out_dir = proof_dir / "runtime_out"
    receiver_out_dir.mkdir(parents=True, exist_ok=True)
    inflate_argv = [
        str(submission_dir / "inflate.sh"),
        str(submission_dir),
        str(receiver_out_dir),
        str(file_list),
    ]

    started = time.monotonic()
    returncode: int | None = None
    stdout = ""
    stderr = ""
    timed_out = False
    try:
        result = subprocess.run(
            inflate_argv,
            check=False,
            capture_output=True,
            env={**os.environ, "PYTHON": sys.executable},
            text=True,
            timeout=int(timeout_seconds),
        )
        returncode = int(result.returncode)
        stdout = _safe_text(result.stdout)
        stderr = _safe_text(result.stderr)
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = _safe_text(exc.stdout)
        stderr = _safe_text(exc.stderr)
    wall_seconds = round(time.monotonic() - started, 6)

    receiver_raw = receiver_out_dir / "0"
    output_present = receiver_raw.is_file()
    output_sha256 = sha256_file(receiver_raw) if output_present else None
    output_bytes = receiver_raw.stat().st_size if output_present else None
    if output_present and not retain_receiver_output:
        receiver_raw.unlink()
    if not retain_receiver_output:
        try:
            receiver_out_dir.rmdir()
        except OSError:
            pass

    blockers: list[str] = []
    if timed_out:
        blockers.append("z7_mamba2_generated_inflate_sh_timed_out")
    if returncode not in (0, None):
        blockers.append("z7_mamba2_generated_inflate_sh_returned_nonzero")
    if returncode is None and not timed_out:
        blockers.append("z7_mamba2_generated_inflate_sh_not_executed")
    if not output_present:
        blockers.append("z7_mamba2_generated_inflate_sh_output_missing")
    if output_present and not output_bytes:
        blockers.append("z7_mamba2_generated_inflate_sh_output_empty")
    passed = returncode == 0 and output_present and bool(output_bytes) and not timed_out
    proof = {
        "schema": Z7_MAMBA2_MLX_RECEIVER_PROOF_SCHEMA,
        "archive_path": _repo_relative(archive_zip_path, repo_root),
        "archive_sha256": archive_sha256,
        "archive_bytes": int(archive_bytes),
        "submission_dir": _repo_relative(submission_dir, repo_root),
        "runtime_tree_sha256": tree_sha256(submission_dir),
        "inflate_argv": [_repo_relative(Path(inflate_argv[0]), repo_root), *inflate_argv[1:]],
        "file_list_path": _repo_relative(file_list, repo_root),
        "receiver_output_dir": _repo_relative(receiver_out_dir, repo_root),
        "receiver_output_path": _repo_relative(receiver_raw, repo_root),
        "receiver_output_present_during_proof": output_present,
        "receiver_output_retained": bool(retain_receiver_output and output_present),
        "receiver_output_sha256": output_sha256,
        "receiver_output_bytes": output_bytes,
        "returncode": returncode,
        "timed_out": timed_out,
        "wall_seconds": wall_seconds,
        "stdout": stdout,
        "stderr": stderr,
        "runtime_consumption_proof_ready": passed,
        "receiver_contract_satisfied": passed,
        "blockers": blockers,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
    }
    write_json(proof_dir / "z7_mamba2_mlx_receiver_proof.json", proof)
    proof["proof_path"] = _repo_relative(
        proof_dir / "z7_mamba2_mlx_receiver_proof.json",
        repo_root,
    )
    write_json(proof_dir / "z7_mamba2_mlx_receiver_proof.json", proof)
    return proof


class _SingleZ7Mamba2MLXArchiveCandidateAdapter:
    adapter_id = Z7_MAMBA2_MLX_ARCHIVE_BOUND_ADAPTER_ID
    candidate_family = Z7_MAMBA2_MLX_ARCHIVE_CANDIDATE_FAMILY

    def __init__(self, row: Mapping[str, Any]) -> None:
        self._row = dict(row)

    def emit_archive_bound_candidate_rows(
        self,
        context: Mapping[str, Any],
    ) -> Sequence[Mapping[str, Any]]:
        return [dict(self._row)]


def _archive_transform_kind_from_config(cfg: Any) -> str:
    if bool(getattr(cfg, "use_canonical_ssd_mlx_backend", False)):
        return Z7_MAMBA2_MLX_ARCHIVE_TRANSFORM_KIND
    return Z7_MAMBA2_MLX_REFERENCE_ARCHIVE_TRANSFORM_KIND


def _build_archive_bound_adapter_package(
    *,
    model: Any,
    output_dir: Path,
    repo_root: Path,
    archive_zip_path: Path,
    archive_sha256: str,
    archive_bytes: int,
    submission_dir: Path,
    receiver_proof: Mapping[str, Any],
    mlx_triage_argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    proof_passed = receiver_proof.get("runtime_consumption_proof_ready") is True
    proof_path = str(receiver_proof.get("proof_path") or "")
    cfg = model.cfg
    transform_kind = _archive_transform_kind_from_config(cfg)
    blockers = list(receiver_proof.get("blockers") or [])
    if bool(getattr(cfg, "use_canonical_ssd_mlx_backend", False)):
        ssd_blocker = str(getattr(cfg, "canonical_ssd_mlx_blocker", "") or "")
        if ssd_blocker:
            blockers.append(ssd_blocker)
    row = {
        "schema": "z7_mamba2_mlx_archive_bound_candidate_row.v1",
        "candidate_id": f"z7_mamba2_mlx_{archive_sha256[:16]}",
        "candidate_family": Z7_MAMBA2_MLX_ARCHIVE_CANDIDATE_FAMILY,
        "target_kind": transform_kind,
        "archive_native_transform_kind": transform_kind,
        "candidate_archive_path": _repo_relative(archive_zip_path, repo_root),
        "candidate_archive_sha256": archive_sha256,
        "candidate_archive_bytes": int(archive_bytes),
        "byte_closed_candidate_emitted": True,
        "byte_closed_candidate_materialized": True,
        "candidate_archive_materialized": True,
        "runtime_consumption_proof_status": "present" if proof_passed else "blocked",
        "runtime_consumption_proof_ready": proof_passed,
        "runtime_consumption_proof_path": proof_path,
        "receiver_contract_kind": "z7_mamba2_mlx_generated_inflate_sh_decode_only_receiver",
        "receiver_contract_satisfied": proof_passed,
        "runtime_adapter_ready": True,
        "contest_runtime_decoder_adapter_ready": True,
        "runtime_adapter_manifest": {
            "schema": "z7_mamba2_mlx_runtime_adapter_manifest.v1",
            "runtime_adapter_ready": True,
            "contest_runtime_decoder_adapter_ready": True,
            "decode_only_receiver_contract": True,
            "submission_dir": _repo_relative(submission_dir, repo_root),
            "runtime_tree_sha256": tree_sha256(submission_dir),
            "runtime_receiver_proof_path": proof_path,
            "mamba2_archive_backend": (
                "ssd_reference"
                if bool(getattr(cfg, "use_canonical_ssd_mlx_backend", False))
                else "reference_torch"
            ),
            "canonical_ssd_mlx_runtime_bridge_wired": bool(
                getattr(cfg, "use_canonical_ssd_mlx_backend", False)
            ),
        },
        "semantic_payload_changed": True,
        "score_affecting_payload_changed": True,
        "exact_axis_score_affecting_adjudication_required": True,
        "charged_bits_changed": True,
        "replay_argv": list(receiver_proof.get("inflate_argv") or []),
        "mlx_triage_argv": list(mlx_triage_argv or []),
        "input_artifacts": [
            _repo_relative(archive_zip_path, repo_root),
            _repo_relative(submission_dir / "0.bin", repo_root),
            proof_path,
        ],
        "blockers": blockers,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "gpu_launched": False,
    }
    package = build_archive_bound_candidate_adapter_package(
        _SingleZ7Mamba2MLXArchiveCandidateAdapter(row),
        repo_root=repo_root,
    )
    wrapped = {
        "schema": Z7_MAMBA2_MLX_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA,
        "archive_bound_candidate_adapter_package": package,
        "receiver_proof": dict(receiver_proof),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
    }
    write_json(output_dir / "archive_bound_candidate_adapter_package.json", wrapped)
    return wrapped


def export_z7_mamba2_mlx_archive(
    model: Any,
    output_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
    emit_archive_bound_candidate_package: bool = True,
    retain_receiver_proof_output: bool = False,
    mlx_triage_argv: Sequence[str] | None = None,
) -> tuple[Path, str, int]:
    """Export an MLX Z7-Mamba-2 model as a contest-shaped ``archive.zip``.

    Per the canonical mlx_score_aware harness contract
    (``tac.substrates._shared.mlx_score_aware.bundle.RendererBundle.export_archive_fn``):
    returns ``(archive_zip_path, sha256_hex, size_bytes)``.

    Sister of :func:`tac.substrates.z6_v2_cargo_cult_unwind.archive_candidate.export_z6_v2_mlx_archive`.
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

    cfg = model.cfg
    bin_bytes = pack_archive_from_exported_state_dict(
        exported_state_dict=model.export_state_dict(),
        mlx_cfg=cfg,
    )
    bin_path = out_dir / "0.bin"
    bin_path.write_bytes(bin_bytes)

    submission_dir = out_dir / "submission"
    # Wave N+9 Slot 1 self-containment fix (Catalog #295 + #361): Z7-Mamba-2
    # `inflate.py` transitively imports (a) `tac.substrates._shared.inflate_runtime`
    # for `select_inflate_device` / `raw_output_path` / `write_rgb_pair_to_raw`,
    # (b) `tac.substrates.time_traveler_l5_z6.architecture` for `_Z6Decoder` +
    # `EVAL_HW`, and (c) `tac.substrates.time_traveler_l5_z7_lstm_predictive_coding.architecture`
    # for `LatentAffineContextConditioner` + `Z7GruPredictiveCodingConfig` +
    # `normalize_context_conditioning_mode`. Without explicit vendoring the
    # shipped submission fails at first import (empirically verified at
    # Wave N+9 Slot 1 landing). The canonical helper was extended in the
    # same commit batch to accept `vendor_extra_substrate_packages` +
    # `vendor_shared_inflate_runtime` per the sister-extinction architecture.
    write_contest_runtime(
        submission_dir,
        substrate_pkg_name="time_traveler_l5_z7_mamba2",
        repo_root=root,
        vendor_shared_inflate_runtime=True,
        vendor_extra_substrate_packages=(
            ("time_traveler_l5_z6", ("architecture.py",)),
            ("time_traveler_l5_z7_lstm_predictive_coding", ("architecture.py",)),
        ),
        vendor_extra_tac_subpackages=(
            ("optimization", ("mamba2_predictor.py",)),
            ("framework_agnostic", ("backend.py",)),
            (
                "substrates._shared.mamba2_ssd",
                (
                    "__init__.py",
                    "numpy_backend.py",
                    "pytorch_backend.py",
                    "mlx_backend.py",
                ),
            ),
        ),
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
        receiver_proof = _run_generated_receiver_proof(
            archive_zip_path=archive_zip_path,
            archive_sha256=archive_sha256,
            archive_bytes=archive_bytes,
            submission_dir=submission_dir,
            output_dir=out_dir,
            repo_root=root,
            retain_receiver_output=retain_receiver_proof_output,
        )
        _build_archive_bound_adapter_package(
            model=model,
            output_dir=out_dir,
            repo_root=root,
            archive_zip_path=archive_zip_path,
            archive_sha256=archive_sha256,
            archive_bytes=archive_bytes,
            submission_dir=submission_dir,
            receiver_proof=receiver_proof,
            mlx_triage_argv=mlx_triage_argv,
        )
    return (archive_zip_path, archive_sha256, archive_bytes)


def export_z7_mamba2_mlx_archive_bound_candidate_package(
    model: Any,
    output_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
    retain_receiver_proof_output: bool = False,
    mlx_triage_argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Export Z7 MLX bytes and emit the shared archive-bound package."""
    archive_zip_path, archive_sha256, archive_bytes = export_z7_mamba2_mlx_archive(
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
    submission_dir = out_dir / "submission"
    receiver_proof = _run_generated_receiver_proof(
        archive_zip_path=archive_zip_path,
        archive_sha256=archive_sha256,
        archive_bytes=archive_bytes,
        submission_dir=submission_dir,
        output_dir=out_dir,
        repo_root=root,
        retain_receiver_output=retain_receiver_proof_output,
    )
    return _build_archive_bound_adapter_package(
        model=model,
        output_dir=out_dir,
        repo_root=root,
        archive_zip_path=archive_zip_path,
        archive_sha256=archive_sha256,
        archive_bytes=archive_bytes,
        submission_dir=submission_dir,
        receiver_proof=receiver_proof,
        mlx_triage_argv=mlx_triage_argv,
    )


__all__ = [
    "Z7_MAMBA2_MLX_ARCHIVE_BOUND_ADAPTER_ID",
    "Z7_MAMBA2_MLX_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA",
    "Z7_MAMBA2_MLX_ARCHIVE_CANDIDATE_FAMILY",
    "Z7_MAMBA2_MLX_ARCHIVE_TRANSFORM_KIND",
    "Z7_MAMBA2_MLX_REFERENCE_ARCHIVE_TRANSFORM_KIND",
    "export_z7_mamba2_mlx_archive",
    "export_z7_mamba2_mlx_archive_bound_candidate_package",
    "pack_archive_from_exported_state_dict",
    "z7_mamba2_meta_from_config",
    "z7_mamba2_pytorch_config_from_mlx",
]
